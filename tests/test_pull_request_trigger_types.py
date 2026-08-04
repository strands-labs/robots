"""Contract pins for which pull_request events may start a workflow.

``pr-and-push.yml`` carries the repository's one required check
(``call-test-lint / Test and Lint``) and cancels its own in-flight run for the
same pull request (``cancel-in-progress: true``). Those two facts together make
the trigger list load-bearing: an event that cannot change the head sha, but is
subscribed to anyway, discards a run of the required check and starts an
identical one over the same commit.

It was subscribed to three such events -- ``ready_for_review``,
``review_requested`` and ``review_request_removed``. None of them can change
``github.event.pull_request.head.sha``, which is the ref the reusable workflow is
handed, so the second run can only recompute the first run's verdict.

Measured on #1899, #1901 and #1902, where two reviewers were requested seconds
after the pull request opened (#1914 records the full measurement). Each head sha ended up carrying **two** completed
``call-test-lint / Test and Lint`` check runs -- one CANCELLED, one SUCCESS::

    #1902  0734d8a7
      07:03:05  pull_request (opened)           CANCELLED  (ran 26s)
      07:03:19  pull_request (review_requested) CANCELLED  (no job started)
      07:03:19  pull_request (review_requested) SUCCESS    (ran 27m16s)

That a CANCELLED check aggregates into a ``FAILURE`` roll-up is already known
and already written down: #1800 measured it on three consecutive ``main``
commits and ``AGENTS.md`` > PR Workflow tells a reader to consult each context's
own ``conclusion`` before believing the roll-up. What that entry describes is a
*different producer* of the symptom -- pushes to ``main`` share one concurrency
group because a push carries no pull request number, so each merge kills the
previous merge's run. That producer is arguably wanted (a superseded ``main`` run
is of no interest) and is not touched here.

This one is on pull request head shas, and unlike the ``main`` case it discards
work for no possible gain, so it is removable rather than merely readable-around.
Measured twice, ten minutes apart, no new run in between::

    19:35Z   #1899 FAILURE   #1901 FAILURE   #1902 FAILURE
    19:45Z   #1899 FAILURE   #1901 SUCCESS   #1902 SUCCESS

The FAILURE column is #1800's documented aggregation. The second read is the part
that reading-discipline cannot cover: the aggregate is not even stable, so two
sweeps of one unchanged sha disagree and neither is reproducible. Removing the
producer is what makes the question moot for a pull request.

The cost is also unbounded rather than the ~20s these three happened to lose: a
reviewer requested at minute 25 of a 27-minute run discards all 25 minutes,
because when the trigger arrives has nothing to do with how far the run has got.

The remedy is to subscribe only to the events that can change the head sha,
which is exactly GitHub's default ``types`` for ``pull_request`` -- so the fix
is the *absence* of an override, and this module pins that no workflow
reintroduces one. Every other ``pull_request`` workflow in the repository
(``changelog-fragment``, ``merge-base-overlap``, ``dependency-review``,
``codeql``, ``breaking-change-check``, ``agent-api-check``,
``llm-input-safety``) already takes that default; ``pr-and-push.yml`` was the
only one that did not, and it is the only one holding a required check.

``test_no_workflow_gates_on_draft_status`` is the premise behind dropping
``ready_for_review`` specifically: nothing in the fleet skips a draft pull
request, so a draft's CI has already run and marking it ready re-runs the same
sha. If a draft skip is ever added, that pin fails and says so, because
``ready_for_review`` would then be the event that starts the first real run.

These are text assertions rather than parsed YAML for the reason the sibling
CI-config pins state (``tests/test_codeql_query_filters.py``,
``tests/test_merge_base_overlap.py``): ``pyyaml`` is an optional dependency
here, and a pin that skips when a dep is missing is not a pin.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOW_DIR = _REPO_ROOT / ".github" / "workflows"
_REQUIRED_CHECK_WORKFLOW = _WORKFLOW_DIR / "pr-and-push.yml"

#: The only ``pull_request`` activity types that can change
#: ``github.event.pull_request.head.sha``, and therefore the only ones that can
#: give a run something new to check. This is also GitHub's default ``types``
#: for ``pull_request``, which is why omitting the key entirely is the preferred
#: spelling: there is then no second copy of this list to drift from it.
_SHA_CHANGING_TYPES = frozenset({"opened", "synchronize", "reopened"})

#: Matches a top-level trigger key inside an ``on:`` mapping, e.g.
#: ``  pull_request:`` or ``  push:``.
_TRIGGER_RE = re.compile(r"^(?P<indent> +)(?P<name>[a-z_]+):\s*$")

#: Matches an inline activity-type list, e.g. ``types: [opened, synchronize]``.
_INLINE_TYPES_RE = re.compile(r"^ +types:\s*\[(?P<items>[^\]]*)\]\s*$")

#: Matches the opening of a block activity-type list, e.g. ``types:`` followed
#: by ``  - published`` lines.
_BLOCK_TYPES_RE = re.compile(r"^(?P<indent> +)types:\s*$")

#: Matches one entry of a block list, e.g. ``  - published``.
_BLOCK_ITEM_RE = re.compile(r"^ +- (?P<item>[a-z_]+)\s*$")


def _workflow_paths() -> list[Path]:
    return sorted(_WORKFLOW_DIR.glob("*.yml"))


def _pull_request_types(text: str) -> list[str] | None:
    """Return the activity types a workflow's ``pull_request`` trigger lists.

    ``None`` means the workflow has no ``pull_request`` trigger, or has one that
    lists no ``types`` and so takes GitHub's default. An empty list means the
    key is present and lists nothing, which is a different statement and is
    reported as such rather than folded into the default.

    Only the ``types`` belonging to ``pull_request`` are returned: a workflow may
    also carry ``push`` or ``release`` triggers with their own list (as
    ``pypi-publish-on-release.yml`` does), and those say nothing about pull
    requests.
    """
    current_trigger: str | None = None
    lines = text.splitlines()
    for index, line in enumerate(lines):
        trigger = _TRIGGER_RE.match(line)
        if trigger is not None and trigger.group("name") not in {"types"}:
            current_trigger = trigger.group("name")
            continue
        if current_trigger != "pull_request":
            continue
        inline = _INLINE_TYPES_RE.match(line)
        if inline is not None:
            return [item.strip() for item in inline.group("items").split(",") if item.strip()]
        if _BLOCK_TYPES_RE.match(line) is not None:
            items = []
            for follower in lines[index + 1 :]:
                item = _BLOCK_ITEM_RE.match(follower)
                if item is None:
                    break
                items.append(item.group("item"))
            return items
    return None


def test_the_scanner_finds_the_pull_request_workflows() -> None:
    """Guard the pins below against a regex that quietly matches nothing.

    Every assertion in this module is a statement about the workflows the
    scanner found. If it found none, they would all hold for the wrong reason,
    so the count is asserted before the properties are.
    """
    paths = _workflow_paths()
    assert paths, f"no workflows under {_WORKFLOW_DIR}"
    with_pull_request = [p.name for p in paths if "pull_request:" in p.read_text(encoding="utf-8")]
    assert _REQUIRED_CHECK_WORKFLOW.name in with_pull_request
    # Seven siblings plus the required-check workflow itself at the time of
    # writing. Asserted as a floor, not an equality: adding a pull_request
    # workflow is routine and must not fail this pin, while a scanner that stops
    # seeing them is exactly what this test is for.
    assert len(with_pull_request) >= 8, with_pull_request


def test_no_pull_request_trigger_subscribes_to_a_sha_invariant_type() -> None:
    """No workflow may be started by a pull_request event that changes no code.

    The required check is cancelled and restarted by such an event
    (``test_the_required_check_discards_its_in_flight_run``), and the restarted
    run reads the same ``head.sha``, so the only thing subscribing to one can
    produce is a second verdict on one commit -- and, measured on #1899, #1901
    and #1902, a CANCELLED check run that makes the roll-up state unstable.
    """
    offenders: dict[str, list[str]] = {}
    for path in _workflow_paths():
        types = _pull_request_types(path.read_text(encoding="utf-8"))
        if types is None:
            continue
        extra = sorted(set(types) - _SHA_CHANGING_TYPES)
        if extra:
            offenders[path.name] = extra
    assert not offenders, (
        "these workflows are started by pull_request events that cannot change "
        f"the head sha: {offenders}. Such an event can only recompute a verdict "
        "already being computed, and it cancels the in-flight required check to "
        "do it. Omit 'types' to take GitHub's default "
        f"({sorted(_SHA_CHANGING_TYPES)})."
    )


def test_the_required_check_workflow_takes_the_default_types() -> None:
    """The required check states its trigger list by not restating it.

    The default is precisely :data:`_SHA_CHANGING_TYPES`, so an explicit list
    spelling out those three would pass the pin above while adding a second copy
    of a list that already exists -- and it is a copy in the one file where
    getting it wrong cancels a 27-minute run.
    """
    types = _pull_request_types(_REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8"))
    assert types is None, (
        f"{_REQUIRED_CHECK_WORKFLOW.name} lists pull_request types explicitly ({types}); "
        "omit the key so there is one definition of which events can change a head sha"
    )


def test_the_required_check_discards_its_in_flight_run() -> None:
    """Premise: this is why a redundant trigger is destructive rather than idle.

    Pins today's behaviour, not a wish. Were ``cancel-in-progress`` ever turned
    off, a redundant trigger would merely waste a runner instead of throwing
    away the required check's progress, and the pin above would be a
    cost-control rule rather than a correctness one. Either way the reasoning
    above needs to be re-read, which is what this failing would say.
    """
    text = _REQUIRED_CHECK_WORKFLOW.read_text(encoding="utf-8")
    assert "cancel-in-progress: true" in text
    assert "group: ${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}" in text


def test_no_workflow_gates_on_draft_status() -> None:
    """Premise for dropping ``ready_for_review``: no draft's CI is skipped.

    Nothing in the fleet reads ``pull_request.draft``, so a draft pull request
    runs the same checks as any other and marking it ready re-runs a sha that
    has already been checked. If a draft skip is added, ``ready_for_review``
    becomes the event that starts the first real run for that sha and belongs in
    the trigger list of whatever workflow does the skipping -- so this pin
    failing is a prompt to re-derive the list, not to delete the pin.
    """
    gating = [p.name for p in _workflow_paths() if "draft" in p.read_text(encoding="utf-8")]
    assert not gating, (
        f"these workflows mention 'draft': {gating}. If one now skips draft pull requests, "
        "re-derive whether ready_for_review must start it"
    )
