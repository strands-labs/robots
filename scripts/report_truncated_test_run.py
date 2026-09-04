#!/usr/bin/env python3
"""Report whether the required test suite ran to completion, and name what it skipped.

Why this exists
---------------
``.github/workflows/test-lint.yml`` runs the one required check as::

    hatch run test -x --strict-markers

``-x`` stops the session at the first failure. The check is not wrong -- it is
red when the tree is red -- but **the report is not a count**. A red
``call-test-lint`` names one failing cell, and the number of failing cells is
unknown until the run is repeated on a tree where that one passes.

Measured on run 33690980247, the ``Test and Lint`` job of #3161 at ``48881ea``::

    line  2207:  collecting ... collected 46583 items
    line 37134:  !!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!
    line 37135:  ==== 1 failed, 34268 passed, 278 skipped, 57 warnings in 1259.55s ====

34268 + 278 + 1 = 34547 of 46583 items executed, so **12036 items -- 25.8% of
the suite -- never ran**. The failure that was hidden was neither hypothetical
nor distant: reproduced on the same commit, the next failure was four lines
below the first, in the same class, with the same cause. Both would have been in
one report from an un-truncated run.

Why the numbers above are not already legible
---------------------------------------------
pytest *does* emit the ``stopping after N failures`` banner, so the truncation is
not literally unrecorded. Three things stop that from being an answer:

- The banner does not say how much was skipped. The count needs the
  ``collected`` line as well, and the two sit **34,927 lines apart** in a 6.4 MB
  log -- nobody subtracts them by hand.
- The banner sits one line above the summary, which is the line every reader is
  already looking at, so it reads as part of the failure rather than as a
  statement about the run's extent.
- Neither number reaches a surface short of downloading that log. A check's
  annotations and the job summary are what a reviewer sees, and the truncation
  appears in neither.

``--cov-fail-under=80`` is not a backstop either: on that truncated run coverage
reported ``Total coverage: 81.54%`` and the gate read as a pass, so it adds no
signal that the run was short.

Why the flag stays
------------------
Removing ``-x`` would make a red run cost what a green one costs, and that is
already close to the bound. Measured over the last 40 ``main`` pushes of
``pr-and-push.yml`` (2026-09-01T23:51Z -> 2026-09-03T02:11Z), the ``Test and
Lint`` job on the 37 runs that completed took **32.6 to 57.5 minutes, median
46.9**, against the job's ``timeout-minutes: 60`` -- and run 33631903156 was
already reaped at **60.12 minutes** with every step reporting success. The
bound cannot absorb the difference, and raising it is explicitly spent: the
comment on that ``timeout-minutes`` line records that "the next raise cannot be
spent here" (#2457, #2239).

So the early exit is load-bearing and this module does not touch it. What it
removes is the *ambiguity*: an aborted run and a complete run with one failure
are indistinguishable in the summary line, and that -- not the early exit -- is
what costs a round, because each hidden failure costs ~21 minutes of runner time
plus one review approval, approval being this repository's scarcest resource
(#1905).

The session states its own extent, and this reading is the fallback
------------------------------------------------------------------

``tests/session_truncation.py`` states the same subtraction in the terminal
summary, counted in process -- ``collected`` is ``len(session.items)`` after
deselection and ``started`` is counted per test entered::

    ===== session truncated: 34547 of 46583 collected tests ran =====
    12036 collected tests never started, so the counts below are a floor, not a total.

That is the owner of the extent, so this module reads that line when it is
present and reports its numbers unchanged. Re-deriving them from the text is the
fallback for a log that does not carry the section -- a run killed before the
terminal summary, or a log kept from before it existed.

Deriving it has to be pytest's own derivation
---------------------------------------------

The fallback reads the collection line, whose trailing tokens pytest writes in a
fixed order and prints only when nonzero, so their positions relative to each
other are not fixed::

    collected 10 items / 1 error / 4 deselected / 1 skipped / 6 selected

Two rules follow, and both are pytest's rather than this module's. The number
selected is ``collected - deselected`` (``report_collect`` computes exactly
that, and prints the token only when it differs from ``collected``), so it is
read from ``deselected`` rather than from the presence of a ``selected`` token
that another token can sit in front of. And a module skipped or failing at
*import* is counted on both lines while being none of the collected items, so
the counts line's totals exceed the item count by whatever the collection line
already accounted for.

A run killed mid-suite reports differently again
-----------------------------------------------
A job killed while pytest is still running writes no summary line at all, which
is the other way a run can be incomplete. It is reported as
``incomplete-no-summary`` rather than folded into either of the two above, so a
run that never finished reporting is not read as a run with one failure.

That branch is defensive rather than observed, and the distinction is worth
keeping straight. The one reaped job in the measured window, run 33631903156 at
60.12 minutes, turns out **not** to be an instance: its suite completed
(46449 of 46449 items, ``46152 passed, 297 skipped``) at 54.5 minutes and the
job was killed in a later step. So the reap in #3143 did not truncate that
suite, and this module reports it as ``complete``, which is the honest answer.

Why this never fails the job
----------------------------
It is wired into the required check's own job, so a nonzero exit here would be
indistinguishable from the suite failing -- and on a *green* run a parsing bug
would turn the one required check red for every open pull request. The verdict
this module produces is a description of a run whose outcome is already decided
by pytest, so it is reported and never gated: ``main`` returns 0 for every
input, including a log it cannot read. That is deliberate and is pinned by
tests/test_truncated_test_run_is_reported.py.

Usage
-----
::

    python3 scripts/report_truncated_test_run.py "$RUNNER_TEMP/pytest.log"

See issue #3164, #3143 for the bound, and the "PR Workflow" section of
AGENTS.md.
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass, field

# pytest's collection line. Matched with ``search`` rather than anchored,
# because the same text is read both from the raw file this script is pointed at
# and, when a reader pastes one, from a job log whose lines carry a timestamp
# prefix added by the Actions log service. The trailing tokens are read as a run
# of ``/ N label`` pairs rather than as an ordered alternation: pytest emits
# ``errors``, ``deselected``, ``skipped`` and ``selected`` in that order and only
# when each is nonzero, so an alternation naming two of them requires them to be
# adjacent and a third token between them hides both.
_COLLECTED = re.compile(r"collected (?P<collected>\d+) items?(?P<tokens>(?: / \d+ [a-z]+)*)")
_COLLECT_TOKEN = re.compile(r"/ (?P<count>\d+) (?P<label>[a-z]+)")

# The section tests/session_truncation.py writes above the counts line. The
# session counted its own extent, so that is where the subtraction belongs and
# this module delegates to it rather than keeping a second reading that can
# report differently.
_STATED_EXTENT = re.compile(r"session truncated: (?P<executed>\d+) of (?P<selected>\d+) collected tests ran")

# The banner ``-x`` / ``--maxfail`` prints when it ends the session early.
_STOPPING = re.compile(r"!+\s*stopping after (?P<failures>\d+) failures?\s*!+")

# The final ``= 1 failed, 34268 passed, ... in 1259.55s =`` line. Only the
# surrounding rule and the trailing duration are fixed; the counts in between
# vary, so they are read by scanning rather than by one exhaustive alternation.
_SUMMARY_LINE = re.compile(r"^=+ .*\bin \d+(?:\.\d+)?s.*=+$")
_COUNT = re.compile(r"(?P<count>\d+) (?P<label>[a-z]+)")

# The Actions log service prefixes every line it stores with an RFC3339 stamp.
# The file this script is pointed at in CI is the raw `tee` copy and carries
# none, but a maintainer diagnosing a past run downloads the stored log, and the
# summary line is the one pattern anchored to the start of a line -- so without
# this the script would silently report ``incomplete-no-summary`` on exactly the
# input a human reaches for. Stripped rather than tolerated in each pattern so
# there is one place that knows about the prefix.
_LOG_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z ")

# Outcome labels that mean a test item was actually executed. ``warnings`` and
# ``deselected`` are deliberately absent: a warning is not an item, and a
# deselected item was never going to run, so counting either would understate
# what the abort skipped. ``rerun`` is absent because a rerun is a second
# attempt at an item already counted under its final outcome.
_EXECUTED_LABELS = frozenset({"failed", "passed", "skipped", "xfailed", "xpassed", "error", "errors"})

# Outcomes the collection line has already accounted for, which are therefore not
# among the items. A module skipped or failing at import is counted once there and
# once on the counts line, so leaving it in the executed total makes the number of
# items executed exceed the number collected.
_NON_ITEM_COLLECT_LABELS = frozenset({"skipped", "error", "errors"})

#: Who the reported extent came from, named in the report so a reader knows
#: whether it was counted or reconstructed.
_STATED_BY_SESSION = "the session's own count"
_DERIVED_FROM_LOG = "derived from the log's counts"

_COMPLETE = "complete"
_TRUNCATED = "truncated"
_NO_SUMMARY = "incomplete-no-summary"
_UNREADABLE = "unreadable"


@dataclass
class RunReport:
    """What one pytest log says about its own extent."""

    outcome: str
    collected: int | None = None
    selected: int | None = None
    executed: int | None = None
    never_ran: int | None = None
    stopped_after: int | None = None
    counts: dict[str, int] = field(default_factory=dict)
    extent_source: str = ""
    detail: str = ""

    @property
    def share_skipped(self) -> float | None:
        """Fraction of the selected items that never ran, or None if unknown."""
        if self.selected in (None, 0) or self.never_ran is None:
            return None
        assert self.selected is not None
        return self.never_ran / self.selected

    @property
    def summary(self) -> str:
        """One line naming the run's extent, suitable for an annotation."""
        if self.outcome == _COMPLETE:
            return f"the suite ran to completion: {self.executed} of {self.selected} items executed"
        if self.outcome == _TRUNCATED:
            share = self.share_skipped
            share_text = f" ({share:.1%} of the suite)" if share is not None else ""
            return (
                f"the run was truncated: {self.executed} of {self.selected} items executed, "
                f"{self.never_ran} never ran{share_text}. The failure count is a lower "
                f"bound, not a total."
            )
        if self.outcome == _NO_SUMMARY:
            return (
                "the run produced no summary line, so it did not finish reporting and "
                "neither its failure count nor its extent is known."
            )
        return f"the run's extent could not be read: {self.detail}"


def parse_run(text: str) -> RunReport:
    """Read a pytest log and report whether the session covered every selected item.

    Args:
        text: The captured stdout of one pytest session.

    Returns:
        A RunReport. ``outcome`` is ``complete`` when every selected item was
        accounted for, ``truncated`` when the session ended early, and
        ``incomplete-no-summary`` when no summary line was written at all --
        which is what a job killed at its timeout bound leaves behind.
    """
    collected: int | None = None
    selected: int | None = None
    non_items = 0
    stopped_after: int | None = None
    summary: str | None = None
    stated: tuple[int, int] | None = None

    for raw in text.splitlines():
        line = _LOG_TIMESTAMP.sub("", raw)
        if collected is None:
            found = _COLLECTED.search(line)
            if found:
                collected = int(found.group("collected"))
                tokens = {
                    token.group("label"): int(token.group("count"))
                    for token in _COLLECT_TOKEN.finditer(found.group("tokens"))
                }
                # pytest's own derivation, so a token between ``deselected`` and
                # ``selected`` cannot change the answer.
                selected = collected - tokens.get("deselected", 0)
                non_items = sum(count for label, count in tokens.items() if label in _NON_ITEM_COLLECT_LABELS)
        told = _STATED_EXTENT.search(line)
        if told:
            stated = (int(told.group("executed")), int(told.group("selected")))
        stopping = _STOPPING.search(line)
        if stopping:
            stopped_after = int(stopping.group("failures"))
        stripped = line.strip()
        if _SUMMARY_LINE.match(stripped):
            summary = stripped

    if collected is None:
        return RunReport(
            outcome=_UNREADABLE,
            stopped_after=stopped_after,
            detail="no 'collected N items' line was found",
        )

    if summary is None:
        return RunReport(
            outcome=_NO_SUMMARY,
            collected=collected,
            selected=selected,
            stopped_after=stopped_after,
            detail="no pytest summary line was found",
        )

    counts = {
        match.group("label"): int(match.group("count"))
        for match in _COUNT.finditer(summary)
        if match.group("label") in _EXECUTED_LABELS
    }
    assert selected is not None
    if stated is None:
        extent_source = _DERIVED_FROM_LOG
        executed = max(sum(counts.values()) - non_items, 0)
    else:
        extent_source = _STATED_BY_SESSION
        executed, selected = stated
    never_ran = max(selected - executed, 0)
    truncated = stopped_after is not None or never_ran > 0

    return RunReport(
        outcome=_TRUNCATED if truncated else _COMPLETE,
        collected=collected,
        selected=selected,
        executed=executed,
        never_ran=never_ran,
        stopped_after=stopped_after,
        counts=counts,
        extent_source=extent_source,
    )


def render(report: RunReport) -> str:
    """Render a report as the markdown written to the job summary."""
    lines = ["## Did the required suite run to completion?", ""]
    lines.append(report.summary)
    lines.append("")
    lines.append("| field | value |")
    lines.append("|---|---|")
    lines.append(f"| outcome | `{report.outcome}` |")
    if report.collected is not None:
        lines.append(f"| items collected | {report.collected} |")
    if report.selected is not None and report.selected != report.collected:
        lines.append(f"| items selected | {report.selected} |")
    if report.executed is not None:
        lines.append(f"| items executed | {report.executed} |")
    if report.never_ran is not None:
        lines.append(f"| items that never ran | {report.never_ran} |")
    if report.stopped_after is not None:
        lines.append(f"| stopped after | {report.stopped_after} failure(s) |")
    if report.extent_source:
        lines.append(f"| extent | {report.extent_source} |")
    for label in sorted(report.counts):
        lines.append(f"| {label} | {report.counts[label]} |")

    if report.outcome == _TRUNCATED:
        lines.extend(
            [
                "",
                "The suite runs under `-x`, so it stops at the first failure and the",
                "remaining items are not evidence of anything. Fix the failure named",
                "above and expect the next run to reach further -- possibly onto another",
                "failure that this run could not reach. The flag is deliberate: a full",
                "run of this suite measures 32.6 to 57.5 min against a 60 min bound, so",
                "letting a red run continue would put it in reach of the reap (#3143).",
            ]
        )
    elif report.outcome == _NO_SUMMARY:
        lines.extend(
            [
                "",
                "No summary line means pytest did not finish reporting, so neither the",
                "failure count nor the extent of this run is known. A job killed while",
                "the suite was still running looks like this; a reap at the",
                "`timeout-minutes` bound renders as CANCELLED and so reads as a",
                "concurrency cancel rather than a timeout (#3143). Check the job's",
                "duration before re-running it.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Report a run's extent to the job summary and as an annotation.

    Always returns 0. See this module's docstring: the verdict describes a run
    whose pass/fail outcome pytest has already decided, and this runs inside the
    one required check's job, so failing here would either duplicate the suite's
    own red or -- on a parsing bug -- invent one on a green tree.
    """
    parser = argparse.ArgumentParser(description="Report whether a pytest run was truncated.")
    parser.add_argument("log", help="Path to the captured pytest output.")
    args = parser.parse_args(argv)

    try:
        with open(args.log, encoding="utf-8", errors="replace") as handle:
            text = handle.read()
    except OSError as exc:
        report = RunReport(outcome=_UNREADABLE, detail=f"{args.log} could not be read ({exc.strerror})")
    else:
        report = parse_run(text)

    document = render(report)
    print(document)

    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        try:
            with open(summary_path, "a", encoding="utf-8") as handle:
                handle.write(document)
        except OSError as exc:
            print(f"note: could not write the job summary ({exc.strerror})")

    if report.outcome == _TRUNCATED:
        print(f"::warning title=The test run was truncated::{report.summary}")
    elif report.outcome == _NO_SUMMARY:
        print(f"::warning title=The test run did not finish reporting::{report.summary}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
