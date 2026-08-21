"""Contract pins for the CodeQL query filters.

``.github/codeql/codeql-config.yml`` exists because a CodeQL alert on a pull
request is a hard merge gate in this repository, and two note-severity *quality*
rules in the ``security-and-quality`` suite fire only on idioms the codebase is
obliged to use. The gate is not the CodeQL job, which never fails on an alert:
``github-advanced-security`` opens a review thread per new alert and the
``default`` branch ruleset sets ``required_review_thread_resolution: true``, so
severity never enters into it. That interaction is invisible to the workflow,
which is how its own comment came to describe a policy the repository does not
implement. See #1810.

A suppression is the kind of change that decays by widening: the cheapest way to
clear any future alert is to append its rule id here, one line at a time, until
the file quietly opts out of the whole quality suite. So the properties below are
about *scope*, not about CodeQL working:

- the filter set is **exactly two** rule ids, named individually, so adding a
  third is a deliberate edit that fails this test until someone changes it;
- ``py/empty-except`` is **absent**, which #1810 names as an explicit non-goal --
  it is the largest class (88 open), a swallowed exception genuinely hides bugs,
  and the instances need reading one at a time;
- the config is **reachable**, i.e. the workflow actually passes it, since an
  unreferenced config file silently filters nothing;
- ``AGENTS.md`` states the gate the same way the workflow does. It carried the
  same false sentence, and survived the correction for a structural reason: #1810
  fixed and pinned only the workflow's copy, so nothing failed while the file every
  contributor reads first still said the opposite. A claim with two homes needs the
  pin to cover both, which is why this assertion lives here beside the workflow's
  rather than in a module of its own;
- ruff still selects **B015 and B018**, which is the load-bearing one. Excluding
  ``py/ineffectual-statement`` is only a no-loss trade because the real no-op
  statement class moved to a check that is merge-blocking here where CodeQL is
  advisory. Drop those two codes and the exclusion silently becomes a capability
  loss, with nothing else in the tree recording the connection.

These are text assertions rather than parsed YAML because that is the shape the
existing CI-config pin uses (``tests/test_merge_base_overlap.py`` reads
``.github/workflows/merge-base-overlap.yml`` the same way) and because ``pyyaml``
is an optional dependency here -- a pin that skips when a dep is missing is not a
pin.
"""

from __future__ import annotations

import ast
import re
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CONFIG_PATH = _REPO_ROOT / ".github" / "codeql" / "codeql-config.yml"
_WORKFLOW_PATH = _REPO_ROOT / ".github" / "workflows" / "codeql.yml"
_PYPROJECT_PATH = _REPO_ROOT / "pyproject.toml"
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: The only two rule ids this repository filters, and the reason each is here.
#:
#: ``py/ineffectual-statement`` -- 27 of 27 open alerts were ``...`` used as a
#: typing-construct body (``Protocol`` methods, ``@abstractmethod`` bodies,
#: ``@overload`` signatures, ``TYPE_CHECKING`` stubs). No rewrite exists.
#:
#: ``py/import-and-import-from`` -- 63 of 64 open alerts were the pytest
#: monkeypatch idiom, where the module alias is the patch target and the ``from``
#: import names the subject, so both are load-bearing.
_EXPECTED_EXCLUDED_RULES = frozenset(
    {
        "py/ineffectual-statement",
        "py/import-and-import-from",
    }
)

#: Ruff codes carrying the no-op-statement capability that the
#: ``py/ineffectual-statement`` exclusion would otherwise give up.
_RELOCATED_RUFF_CODES = ("B015", "B018")

#: Matches the two-line ``- exclude:`` / ``id:`` form the config is written in.
_EXCLUDED_ID_RE = re.compile(
    r"^[ \t]*-[ \t]*exclude:[ \t]*\r?\n[ \t]*id:[ \t]*(?P<rule>[A-Za-z0-9/_-]+)[ \t]*$",
    re.MULTILINE,
)


def _excluded_rule_ids() -> list[str]:
    return _EXCLUDED_ID_RE.findall(_CONFIG_PATH.read_text(encoding="utf-8"))


class TestTheFilterSetStaysNarrow:
    def test_the_config_file_exists(self):
        assert _CONFIG_PATH.is_file(), (
            f"{_CONFIG_PATH.relative_to(_REPO_ROOT)} is missing. If the CodeQL filters were "
            "removed on purpose, delete this module in the same change so the tree does not "
            "carry a pin for a file nobody has."
        )

    def test_exactly_the_two_documented_rules_are_excluded(self):
        found = _excluded_rule_ids()
        assert len(found) == len(set(found)), f"a rule id is excluded twice: {found}"
        assert set(found) == set(_EXPECTED_EXCLUDED_RULES), (
            "the CodeQL filter set changed. Every id here suppresses a real query for the whole "
            "repository, so adding one is a decision that needs its own reasoning recorded next to "
            "it in the config -- and then this expectation updated deliberately.\n"
            f"  expected: {sorted(_EXPECTED_EXCLUDED_RULES)}\n"
            f"  found:    {sorted(found)}"
        )

    def test_empty_except_is_not_excluded(self):
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        assert "py/empty-except" not in _excluded_rule_ids(), (
            "py/empty-except must keep gating merges. It is the largest alert class, a swallowed "
            "exception genuinely hides bugs, and its instances are not one mechanical idiom - "
            "#1810 names quieting it as an explicit non-goal."
        )
        assert "py/empty-except" in text, (
            "the config should keep naming py/empty-except as the deliberate non-exclusion, so the "
            "next reader looking for it finds the reason rather than an omission."
        )

    def test_each_exclusion_carries_its_reasoning(self):
        """A bare rule id is how the next reader loses the argument for it."""
        text = _CONFIG_PATH.read_text(encoding="utf-8")
        assert "#1810" in text, "the config must link the issue that measured the cost"
        for rule in _EXPECTED_EXCLUDED_RULES:
            # The id appears once in a comment block explaining it and once in the
            # filter itself; a filter with no prose above it is the decay case.
            assert text.count(rule) >= 2, (
                f"{rule} is excluded without a comment naming why. A suppression with no stated "
                "reason cannot be re-litigated, only inherited."
            )


class TestTheConfigIsReachable:
    def test_the_workflow_passes_the_config_file(self):
        workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "config-file: ./.github/codeql/codeql-config.yml" in workflow, (
            "codeql.yml must pass config-file, or the filters above are dead text: an "
            "unreferenced CodeQL config silently filters nothing and every alert keeps gating."
        )

    def test_the_workflow_no_longer_claims_alerts_do_not_block(self):
        """The comment that was false is the reason #1810 was filed."""
        workflow = _WORKFLOW_PATH.read_text(encoding="utf-8")
        assert "PRs are not blocked on" not in workflow, (
            "codeql.yml used to state that PRs are not blocked on CodeQL alerts. Thread-resolution "
            "on bot-authored review threads makes every new alert a merge gate, so that sentence "
            "described a policy the repository does not implement. Do not restore it."
        )
        assert "hard merge gate" in workflow, (
            "codeql.yml must say what actually happens, not merely stop saying the wrong thing. A "
            "contributor reading it needs to know an alert blocks the merge before they spend a "
            "round wondering why an approved, green PR will not go in."
        )


class TestTheRulesFileStatesTheGate:
    """``AGENTS.md`` is where a contributor looks before they read any workflow.

    #1810 corrected ``codeql.yml`` and pinned it above, but the identical false
    sentence in ``AGENTS.md`` went unpinned and survived - so the file that frames
    every contribution told the reader an alert is advisory while the ruleset
    blocked the merge on it. #1890 is the shape #1892 recorded: approved, required
    check green, one unresolved note-severity thread - free to merge by that file's
    account, and in fact not merging for 53 minutes, until the thread was resolved.

    A negative assertion alone would not hold this - the claim can be restated in
    new words and still be false - so it is paired with the two positives that make
    the section actionable: the ruleset property that does the gating, and the
    dismissal path that clears a deliberate finding without editing the code.
    """

    def test_it_no_longer_claims_a_finding_does_not_block(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "not PR-blocking" not in text, (
            "AGENTS.md used to state that CodeQL findings are not PR-blocking. A "
            "`github-advanced-security` review thread on an alert is a merge gate under "
            "`required_review_thread_resolution`, whatever the severity, so that sentence "
            "described a policy the repository does not implement. Do not restore it - and see "
            "the workflow assertion above, which is the same claim in its other home."
        )

    def test_it_names_the_ruleset_property_that_gates(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "required_review_thread_resolution" in text, (
            "AGENTS.md must name what actually blocks the merge, not merely stop denying it. A "
            "contributor who knows only that 'CodeQL is advisory' - true of the check context, "
            "which is not in the required set - has no account of an approved, green PR that "
            "will not go in, and spends the round #1892 was filed to prevent."
        )

    def test_it_records_the_dismissal_path(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "dismissed_reason" in text, (
            "AGENTS.md must keep naming dismissal as the way to clear a deliberate, test-only "
            "finding. Without it the only visible options are editing the flagged code - which "
            "cost #1879 a round, and on #1890 would have left a fixture asserting nothing, since "
            "the IndexError the query asks for is what CPython clears to end iteration - or "
            "widening the filter set the test above pins at two."
        )


class TestTheRelocatedCapabilityStaysSelected:
    def test_ruff_still_selects_the_no_op_statement_codes(self):
        pyproject = _PYPROJECT_PATH.read_text(encoding="utf-8")
        for code in _RELOCATED_RUFF_CODES:
            assert f'"{code}"' in pyproject, (
                f"ruff must keep selecting {code}. Excluding py/ineffectual-statement from CodeQL "
                "is only a no-loss trade because the real no-op-statement class is enforced by "
                "ruff, which gates merges here where CodeQL is advisory. Removing this code while "
                "the exclusion stands drops the capability with nothing recording it."
            )


class TestTheRulesFileSettlesTheCrossThreadMarshalClass:
    """``py/catch-base-exception`` clears under none of the three dispositions.

    The three tools the section offers are fix / dismiss-if-test-only /
    filter-if-every-instance-is-obliged. This rule's whole alert surface in the
    tree is one construct - a cross-thread exception-marshal box, the only
    ``except BaseException`` handler in the tree that does not re-raise lexically,
    measured by ``TestTheMarshalBoxCensusIsDerivedFromTheTree`` below - and
    each tool refuses it in turn: narrowing deletes a ``SystemExit`` outright
    (pinned below), the flagged site is not test-only, and not every instance is
    obliged because ``concurrent.futures`` is a genuine route whenever the caller
    owns the thread.

    Left unwritten, that gap does not read as a gap. It reads as a judgment call,
    and it cost #1899 two threads that each argued the idiom at length and then
    deferred to a human rather than applying a rule nobody had written down. So
    what is pinned here is the *distinction* that resolves it - which thread the
    box marshals onto - since a passage restating the three tools without it would
    pass any assertion about the rule id alone.
    """

    def test_the_rule_id_is_not_filtered(self):
        assert "py/catch-base-exception" not in _excluded_rule_ids(), (
            "py/catch-base-exception must not join the filter set. Filtering requires every "
            "instance to be an obliged idiom, and the marshal-onto-a-new-thread case is a "
            "standing counter-example: concurrent.futures does it, so the exclusion would opt "
            "the repository out of a rule that is right about half its own alerts."
        )

    def test_agents_md_names_the_direction_that_decides(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "py/catch-base-exception" in text, (
            "AGENTS.md must name the rule id, or a contributor meeting the alert has only the "
            "three dispositions, none of which fits, and no way to tell that is expected."
        )
        assert "concurrent.futures" in text, (
            "AGENTS.md must name concurrent.futures as the route for a box that marshals off a "
            "thread the caller creates. That is the half of the rule which removes the alert "
            "instead of excusing it; without it the only recorded outcome is a dismissal, and "
            "the avoidable case gets dismissed too."
        )
        assert "run_on_main" in text, (
            "AGENTS.md must name the obliged case concretely. concurrent.futures cannot target "
            "an already-running foreign thread, so run_on_main's box has no stdlib replacement - "
            "and a rule that reads 'use concurrent.futures' with no exception would send the "
            "next contributor to rewrite the one site that cannot be rewritten."
        )

    def test_agents_md_records_that_narrowing_loses_systemexit(self):
        text = _AGENTS_PATH.read_text(encoding="utf-8")
        assert "SystemExit" in text, (
            "AGENTS.md must record what narrowing to Exception actually costs. 'Except block "
            "handles BaseException' reads as a style note, so the reason not to take the "
            "query's advice has to be stated where the advice is met - and it is a silent "
            "deletion, pinned by the test below."
        )


class TestTheNarrowingMeasurementStillHolds:
    """The prescription above rests on a CPython behaviour, so it is executed here.

    ``AGENTS.md`` tells a contributor not to narrow a cross-thread marshal box,
    because a ``SystemExit`` raised on a worker is *deleted* rather than relocated,
    and to prefer ``concurrent.futures`` because it preserves the exception. Both
    are claims about the interpreter rather than about this repository, which is
    the kind of claim that rots without noise when the floor moves: a future Python
    could route thread exceptions differently and the passage would still read
    plausibly while advising the wrong thing.

    ``SystemExit`` is the case pinned, rather than ``KeyboardInterrupt``, because it
    is the silent one. Both escape a narrowed handler, but ``threading.excepthook``
    *is* called for each and its default implementation ignores ``SystemExit``
    specifically, so a worker that dies of one writes nothing to stderr while a
    ``RuntimeError`` prints a full traceback. Silence is what makes the narrowing
    cost invisible in review, so the escape is captured through a stubbed hook here
    rather than left to reach the runner's unhandled-thread reporting.
    """

    @staticmethod
    def _run_box(handler: type[BaseException]) -> tuple[BaseException | None, list[type]]:
        """Run a hand-rolled marshal box whose handler catches ``handler``.

        Returns what the caller could re-raise from the box, and the exception
        types that escaped the thread instead.
        """
        import threading

        sentinel = SystemExit("pinned")
        box: dict[str, BaseException] = {}
        escaped: list[type] = []

        def job() -> None:
            try:
                raise sentinel
            except handler as exc:  # noqa: BLE001 - the clause under measurement
                box["exc"] = exc

        original_hook = threading.excepthook
        threading.excepthook = lambda args: escaped.append(args.exc_type)
        try:
            thread = threading.Thread(target=job)
            thread.start()
            thread.join()
        finally:
            threading.excepthook = original_hook

        return box.get("exc"), escaped

    def test_a_narrowed_handler_deletes_a_systemexit(self):
        observed, escaped = self._run_box(Exception)
        assert observed is None, (
            "narrowing a cross-thread marshal box to Exception must still be observed to lose a "
            "SystemExit. If this now passes the exception through, the AGENTS.md advice against "
            "narrowing has lost its reason and the passage should be re-measured, not kept."
        )
        assert escaped == [SystemExit], (
            "the SystemExit must be seen escaping the thread, not merely missing from the box. "
            "Asserting only the absence would also pass if the exception were never raised, "
            "which is the way this measurement could rot into a tautology."
        )

    def test_the_base_exception_handler_carries_it(self):
        observed, escaped = self._run_box(BaseException)
        assert isinstance(observed, SystemExit), (
            "except BaseException is what makes the box work at all: it is the clause that puts "
            "the SystemExit in the box for the caller to re-raise. This is precisely the "
            "behaviour the CodeQL alert asks a contributor to remove."
        )
        assert escaped == [], "nothing should escape the thread when the handler catches it"

    def test_concurrent_futures_preserves_the_exception_identity(self):
        """The stdlib route, and why it is better rather than merely quieter."""
        import concurrent.futures

        sentinel = SystemExit("pinned")

        def job() -> None:
            raise sentinel

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(job)

        # pytest.raises rather than a hand-rolled catch: the executor's __exit__ has
        # joined, so result() re-raises here on the caller's thread, and there is no
        # thread boundary left for an exception to be marshalled across. Writing
        # `except BaseException` for it would be the alert this module is about,
        # earned for nothing.
        with pytest.raises(SystemExit) as caught:
            future.result()

        assert caught.value is sentinel, (
            "Future.result() must re-raise the very object the worker raised. Identity - not "
            "merely type - is what lets AGENTS.md call delegating strictly better than a "
            "hand-rolled box, so a regression here weakens the prescription rather than any "
            "code, and the passage would need rewording."
        )


#: Trees searched for ``except BaseException`` handlers. The passage's claim is
#: about the whole repository, so the scan is too - a handler added under
#: ``examples/`` is as much a new alert as one under ``strands_robots/``.
_HANDLER_TREES = ("strands_robots", "tests", "tests_integ", "examples", "scripts")

#: The one handler the section is about, as ``path::function``.
_MARSHAL_BOX = "strands_robots/simulation/isaac/simulation.py::_job"

#: Floor for the census, so a scan that stops finding handlers fails loudly
#: instead of reporting a clean tree it never read.
_MIN_HANDLERS = 10


class _Handler(NamedTuple):
    """One ``except BaseException`` handler, keyed the way the table names it."""

    path: str
    lineno: int
    owner: str
    reraises: bool

    @property
    def key(self) -> str:
        return f"{self.path}::{self.owner}"


def _owning_definition(tree: ast.AST, lineno: int) -> str:
    """Name of the innermost def/class containing ``lineno``."""
    best: ast.AST | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            end = node.end_lineno or node.lineno
            if node.lineno <= lineno <= end and (best is None or node.lineno > best.lineno):  # type: ignore[attr-defined]
                best = node
    return getattr(best, "name", "<module>")


@lru_cache(maxsize=1)
def _base_exception_handlers() -> tuple[_Handler, ...]:
    """Every ``except BaseException`` handler in the tree, with its disposition.

    ``reraises`` is whether the handler's last statement is a lexical ``raise``,
    which is exactly what ``py/catch-base-exception`` accepts - so this is the
    census the AGENTS.md passage argues from, measured rather than transcribed.
    """
    found: list[_Handler] = []
    for tree_name in _HANDLER_TREES:
        root = _REPO_ROOT / tree_name
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*.py")):
            try:
                parsed = ast.parse(path.read_text(encoding="utf-8"))
            except (OSError, SyntaxError):  # pragma: no cover - unreadable source
                continue
            for node in ast.walk(parsed):
                if not isinstance(node, ast.ExceptHandler) or node.type is None:
                    continue
                named = {n.id for n in ast.walk(node.type) if isinstance(n, ast.Name)}
                if "BaseException" not in named:
                    continue
                found.append(
                    _Handler(
                        path=path.relative_to(_REPO_ROOT).as_posix(),
                        lineno=node.lineno,
                        owner=_owning_definition(parsed, node.lineno),
                        reraises=isinstance(node.body[-1], ast.Raise),
                    )
                )
    return tuple(found)


def _census_table_rows() -> list[tuple[str, bool]]:
    """The ``| handler | ends in | flagged |`` rows, as ``(handler, flagged)``."""
    text = _AGENTS_PATH.read_text(encoding="utf-8")
    start = text.index("| handler | ends in | flagged |")
    end = text.index("\n\n", start)
    rows: list[tuple[str, bool]] = []
    for line in text[start:end].splitlines()[2:]:
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        handler = cells[0].strip("`")
        rows.append((handler, "yes" in cells[-1]))
    return rows


class TestTheMarshalBoxCensusIsDerivedFromTheTree:
    """The passage above argues from a census of the tree, so the census is measured here.

    Its load-bearing claim is that the rule's *entire* alert surface is one
    construct: every ``except BaseException`` handler re-raises lexically, which
    ``py/catch-base-exception`` accepts, except the cross-thread marshal box. That
    claim is what makes the section's disposition (dismiss the box, delegate
    everything else to ``concurrent.futures``) exhaustive rather than a guess.

    Nothing checked it. The class above grades which *concepts* the passage names -
    the rule id, ``concurrent.futures``, ``run_on_main``, ``SystemExit`` - all of
    which survive a census going stale underneath them, and it had: the passage
    counted seven handlers against a tree holding sixteen, omitted one in
    ``strands_robots/`` outright, and cited five of its seven sites at line numbers
    that had moved, the flagged one by 745 lines. Every assertion here passed
    throughout.

    Two failures matter differently. A row that names nothing real is a stale
    citation and costs a reader a search. A *second* handler that does not re-raise
    is the section becoming wrong: it is a new alert of a rule whose thread gates
    the merge, arriving with no recorded disposition, and the passage would still
    read as if it had one.
    """

    def test_the_scan_reaches_the_tree(self):
        """A census that finds nothing would satisfy every assertion below."""
        handlers = _base_exception_handlers()
        assert len(handlers) >= _MIN_HANDLERS, (
            f"only {len(handlers)} `except BaseException` handlers found across "
            f"{_HANDLER_TREES}; the scan stopped reading the tree, so the census "
            "assertions below hold vacuously"
        )

    def test_exactly_one_handler_does_not_reraise_lexically(self):
        """The claim the section's disposition rests on."""
        loose = [h for h in _base_exception_handlers() if not h.reraises]
        assert len(loose) == 1, (
            "AGENTS.md argues that py/catch-base-exception's whole alert surface here is one "
            f"construct, and the tree now holds {len(loose)}: "
            f"{[f'{h.path}:{h.lineno} in {h.owner}()' for h in loose]}. Each one that does not "
            "end in a lexical raise is a separate alert whose review thread gates the merge, so "
            "either give it a lexical raise (cleanup-and-reraise, the majority form above) or "
            "record its disposition in that section - a passage claiming a single construct "
            "while the tree holds several sends the next contributor to the wrong bullet."
        )

    def test_the_one_that_does_not_is_the_cross_thread_marshal_box(self):
        """Which handler it is decides which bullet applies, so it is pinned by identity."""
        loose = [h for h in _base_exception_handlers() if not h.reraises]
        assert [h.key for h in loose] == [_MARSHAL_BOX], (
            f"the handler that does not re-raise lexically is {[h.key for h in loose]}, not "
            f"{_MARSHAL_BOX}. The section's disposition is specific to a box marshalling onto an "
            "already-running foreign thread - obliged, because concurrent.futures cannot target "
            "one - and reads as advice to dismiss whatever is flagged if the flagged site is "
            "something else."
        )

    def test_every_row_names_a_handler_that_exists(self):
        """A row is a citation, and a citation that resolves to nothing is worse than none."""
        by_key = {h.key for h in _base_exception_handlers()}
        missing = [handler for handler, _ in _census_table_rows() if handler not in by_key]
        assert not missing, (
            f"AGENTS.md's handler table names sites that do not exist: {missing}. Name each one "
            "as `path::function`; the function that owns the handler survives edits above it, "
            "which a line number does not - that is how the previous table came to cite five of "
            "seven sites at lines they had moved away from."
        )

    def test_the_table_is_the_whole_package_census(self):
        """An omitted handler reads as a handler that is not there, which is the same mistake."""
        in_package = {h.key for h in _base_exception_handlers() if h.path.startswith("strands_robots/")}
        listed = {handler for handler, _ in _census_table_rows()}
        assert listed == in_package, (
            f"AGENTS.md's handler table must list every `except BaseException` handler in "
            f"strands_robots/. Missing: {sorted(in_package - listed)}. Listed but not in the "
            f"package: {sorted(listed - in_package)}. The table is the evidence for the claim "
            "above it, so a handler left out of it is a site a reader concludes does not exist."
        )

    def test_the_flagged_column_marks_exactly_the_one_that_is_flagged(self):
        """The column is the whole point of the table: it separates the two dispositions."""
        loose = {h.key for h in _base_exception_handlers() if not h.reraises}
        marked = {handler for handler, flagged in _census_table_rows() if flagged}
        assert marked == loose, (
            f"the table marks {sorted(marked)} as flagged and the tree says {sorted(loose)}. The "
            "column is what tells a contributor whether the handler they are looking at is the "
            "obliged case or the avoidable one, so it cannot be maintained by hand against a "
            "property this file already measures."
        )
