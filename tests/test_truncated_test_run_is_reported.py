"""Contract pins for the report of whether the required suite ran to completion.

The one required check runs the suite under ``-x``, so it stops at the first
failure and **the report is not a count**. Measured on run 33690980247, the
``Test and Lint`` job of #3161 at ``48881ea``::

    line  2207:  collecting ... collected 46583 items
    line 37134:  !!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!
    line 37135:  ==== 1 failed, 34268 passed, 278 skipped, 57 warnings in 1259.55s ====

12036 items -- 25.8% of the suite -- never ran, and the next failure was four
lines below the first in the same class, so the truncation turned a one-round fix
into two. pytest does print the ``stopping after`` banner, so this is not an
unrecorded event; what is missing is the subtraction, because the two numbers it
needs sit 34,927 lines apart in a 6.4 MB log and neither reaches a surface a
reviewer reads.

``-x`` itself stays, and that is a measurement rather than a preference. Over the
last 40 ``main`` pushes the ``Test and Lint`` job took 32.6 to 57.5 minutes on
the 37 runs that completed, median 46.9, against ``timeout-minutes: 60`` -- so
letting a red run continue would put it in reach of the reap, and the comment on
that bound records that a further raise cannot be spent there (#2457, #2239).

See scripts/report_truncated_test_run.py, issue #3164, and issue #3143 for the
bound.
"""

from __future__ import annotations

import importlib.util
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

yaml = pytest.importorskip("yaml", reason="pyyaml is an optional dev dependency")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPT = _REPO_ROOT / "scripts" / "report_truncated_test_run.py"
_WORKFLOW = _REPO_ROOT / ".github" / "workflows" / "test-lint.yml"


def _load() -> Any:
    spec = importlib.util.spec_from_file_location("report_truncated_test_run", _SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


mod = _load()

# Reached through the importlib load above, so these are module attributes at
# runtime rather than names mypy can resolve. ``hatch run lint`` covers only the
# package and its tests, so the script is checked separately.
parse_run: Any = mod.parse_run
render: Any = mod.render
main: Any = mod.main


# The tail of run 33690980247's log, verbatim apart from the elision, so the
# arithmetic below is the arithmetic that run actually produced.
TRUNCATED_LOG = """\
collecting ... collected 46583 items

tests/test_a.py::test_one PASSED                                         [  0%]
tests/test_hardware_robot_lifecycle.py::TestExecuteTaskSync::test_sync_runner_no_running_loop_completes FAILED [ 74%]

=========================== short test summary info ============================
FAILED tests/test_hardware_robot_lifecycle.py::TestExecuteTaskSync::test_sync_runner_no_running_loop_completes
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
==== 1 failed, 34268 passed, 278 skipped, 57 warnings in 1259.55s (0:20:59) ====
"""

# Run 33702826136 at ``0a04d2b``, a complete green run on ``main``.
COMPLETE_LOG = """\
collecting ... collected 46575 items

tests/test_a.py::test_one PASSED                                         [  0%]

Required test coverage of 80% reached. Total coverage: 81.62%
============ 46278 passed, 297 skipped, 61 warnings in 2101.31s (0:35:01) ======
"""


class TestATruncatedRunNamesWhatItSkipped:
    """The extent of an aborted run is reported as a number, not left implicit."""

    def test_the_measured_truncation_is_reported_with_its_own_numbers(self) -> None:
        report = parse_run(TRUNCATED_LOG)

        assert report.outcome == "truncated"
        assert report.collected == 46583
        # 1 failed + 34268 passed + 278 skipped, which is what that run executed.
        assert report.executed == 34547
        assert report.never_ran == 12036
        assert report.stopped_after == 1

    def test_the_skipped_share_is_the_figure_the_issue_computed_by_hand(self) -> None:
        report = parse_run(TRUNCATED_LOG)

        share = report.share_skipped
        assert share is not None
        assert round(share * 100, 1) == 25.8

    def test_the_summary_says_the_failure_count_is_a_lower_bound(self) -> None:
        # The whole cost this addresses is that a reader takes "1 failed" for a
        # total, so the report has to contradict that reading in words.
        summary = parse_run(TRUNCATED_LOG).summary

        assert "12036 never ran" in summary
        assert "lower bound" in summary

    def test_warnings_are_not_counted_as_executed_items(self) -> None:
        # 57 warnings appear in that summary line and are not test items. Counting
        # them would understate what the abort skipped, which is the one direction
        # this report must not err in.
        report = parse_run(TRUNCATED_LOG)

        assert "warnings" not in report.counts
        assert report.never_ran == 12036


class TestACompleteRunIsNotReportedAsTruncated:
    """The negative control: a green run must add no warning to any pull request."""

    def test_a_complete_run_reports_complete(self) -> None:
        report = parse_run(COMPLETE_LOG)

        assert report.outcome == "complete"
        assert report.collected == 46575
        assert report.executed == 46575
        assert report.never_ran == 0

    def test_a_complete_run_raises_no_annotation(self, capsys: pytest.CaptureFixture[str]) -> None:
        assert render(parse_run(COMPLETE_LOG)).count("::warning") == 0
        capsys.readouterr()

    def test_deselected_items_are_not_reported_as_never_run(self) -> None:
        # ``-m 'not slow'`` selects a subset. Those items were never going to run,
        # so counting them as skipped-by-the-abort would warn on every such run.
        log = "collected 900 items / 400 deselected / 500 selected\n===== 500 passed, 400 deselected in 12.00s =====\n"
        report = parse_run(log)

        assert report.outcome == "complete"
        assert report.selected == 500
        assert report.never_ran == 0


class TestARunThatNeverReportedIsNamedApart:
    """A run killed mid-suite is not readable as a run with one failure."""

    def test_a_log_with_no_summary_line_is_not_called_complete(self) -> None:
        report = parse_run("collecting ... collected 46583 items\ntests/a.py::t PASSED [ 10%]\n")

        assert report.outcome == "incomplete-no-summary"
        assert report.executed is None

    def test_a_log_with_no_collection_line_is_unreadable_rather_than_complete(self) -> None:
        report = parse_run("Killed\n")

        assert report.outcome == "unreadable"


class TestTheReportNeverDecidesTheJob:
    """It runs inside the required check's own job, so it must not be able to fail it."""

    @pytest.mark.parametrize(
        "log_text",
        [TRUNCATED_LOG, COMPLETE_LOG, "collected 5 items\n", "", "Killed\n"],
        ids=["truncated", "complete", "no-summary", "empty", "unreadable"],
    )
    def test_every_input_exits_zero(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], log_text: str) -> None:
        log = tmp_path / "pytest.log"
        log.write_text(log_text, encoding="utf-8")

        assert main([str(log)]) == 0
        capsys.readouterr()

    def test_an_absent_log_exits_zero_rather_than_raising(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # ``if: always()`` means this step also runs when an earlier step failed
        # before the suite produced a log at all.
        assert main([str(tmp_path / "absent.log")]) == 0
        capsys.readouterr()

    def test_a_truncated_run_is_reported_as_an_annotation(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        log = tmp_path / "pytest.log"
        log.write_text(TRUNCATED_LOG, encoding="utf-8")

        assert main([str(log)]) == 0
        out = capsys.readouterr().out
        assert "::warning title=The test run was truncated::" in out


class TestTheStoredLogFormIsReadableToo:
    """A maintainer diagnosing a past run downloads the timestamped log."""

    def test_the_actions_timestamp_prefix_does_not_hide_the_summary_line(self) -> None:
        stamped = "".join(f"2026-09-03T00:07:28.1897683Z {line}\n" for line in TRUNCATED_LOG.splitlines())

        report = parse_run(stamped)

        # Without the prefix being stripped, the anchored summary pattern misses
        # and this reports ``incomplete-no-summary`` on exactly the input a human
        # reaches for -- a wrong answer shaped like a different finding.
        assert report.outcome == "truncated"
        assert report.never_ran == 12036


def _run_tests_step() -> dict[str, Any]:
    document = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
    steps = [step for job in document["jobs"].values() for step in job["steps"]]
    piped = [step for step in steps if "hatch run test" in str(step.get("run", ""))]
    assert len(piped) == 1, [step.get("name") for step in piped]
    return piped[0]


class TestCapturingTheSuiteCannotHideItsVerdict:
    """The pipe that captures the log must not become the pipeline's exit status."""

    def test_a_piped_suite_sets_pipefail(self) -> None:
        step = _run_tests_step()
        body = str(step["run"])

        if "|" not in body.split("hatch run test", 1)[1].splitlines()[0]:
            pytest.skip("the suite's output is not piped, so pipefail is not required")

        # The default shell for ``run:`` is ``bash -e``, which does NOT set
        # pipefail, so the pipeline would exit with ``tee``'s status. A failing
        # suite would then report SUCCESS on the one required check, and nothing
        # else in this repository reads the suite's verdict independently.
        assert "set -o pipefail" in body, body

    def test_pipefail_is_set_before_the_suite_runs(self) -> None:
        body = str(_run_tests_step()["run"])
        if "set -o pipefail" not in body:
            pytest.skip("the suite's output is not piped")

        assert body.index("set -o pipefail") < body.index("hatch run test -x")

    def test_the_early_exit_flag_is_still_in_place(self) -> None:
        # Removing ``-x`` would make a red run cost what a green one costs, which
        # is 32.6-57.5 min against a 60 min bound. This module's whole premise is
        # that the flag stays and the truncation is described instead.
        assert "-x" in str(_run_tests_step()["run"])


class TestTheReportIsWiredIntoTheRequiredCheck:
    """A reporter nothing calls is the failure mode #1905 records for its own cause."""

    def test_a_step_runs_the_reporter(self) -> None:
        document = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        steps = [step for job in document["jobs"].values() for step in job["steps"]]
        callers = [step for step in steps if "report_truncated_test_run.py" in str(step.get("run", ""))]

        assert len(callers) == 1, [step.get("name") for step in callers]

    def test_the_reporter_runs_even_when_the_suite_failed(self) -> None:
        document = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        steps = [step for job in document["jobs"].values() for step in job["steps"]]
        caller = next(step for step in steps if "report_truncated_test_run.py" in str(step.get("run", "")))

        # A truncated run is by definition a failed run, so a step that only runs
        # on success would describe every case except the one it exists for.
        assert str(caller.get("if", "")).strip() == "always()"

    def test_the_reporter_reads_the_file_the_suite_wrote(self) -> None:
        document = yaml.safe_load(_WORKFLOW.read_text(encoding="utf-8"))
        steps = [step for job in document["jobs"].values() for step in job["steps"]]
        bodies = [str(step.get("run", "")) for step in steps]
        producer = next(body for body in bodies if "hatch run test" in body)
        consumer = next(body for body in bodies if "report_truncated_test_run.py" in body)

        # One path, spelled the same on both sides: a reporter pointed at a file
        # nothing writes reports ``unreadable`` forever and exits 0, so the
        # mismatch would be silent.
        assert "${RUNNER_TEMP}/pytest.log" in producer
        assert "${RUNNER_TEMP}/pytest.log" in consumer


# One deselecting run of each shape pytest can print, with the collection line
# exactly as pytest 9.0.3 writes it. The token order is pytest's -- ``errors``,
# ``deselected``, ``skipped``, ``selected``, each printed only when nonzero -- so
# which tokens sit between ``deselected`` and ``selected`` is a property of the
# run, not of the format. TestTheseAreTheLinesPytestWrites grounds them.
DESELECTING_RUNS = {
    "deselected alone": (
        "collected 900 items / 400 deselected / 500 selected\n===== 500 passed, 400 deselected in 12.00s =====\n",
        500,
        500,
    ),
    "a collection skip between deselected and selected": (
        "collected 10 items / 4 deselected / 1 skipped / 6 selected\n"
        "===== 6 passed, 1 skipped, 4 deselected in 0.01s =====\n",
        6,
        6,
    ),
    "a collection error before deselected": (
        "collected 10 items / 1 error / 4 deselected / 1 skipped / 6 selected\n"
        "===== 6 passed, 1 skipped, 4 deselected, 1 error in 0.02s =====\n",
        6,
        6,
    ),
    "every item deselected": (
        "collected 10 items / 10 deselected / 1 skipped / 0 selected\n===== 1 skipped, 10 deselected in 0.01s =====\n",
        0,
        0,
    ),
    "a collection skip and no deselection": (
        "collected 10 items / 1 skipped\n===== 10 passed, 1 skipped in 0.01s =====\n",
        10,
        10,
    ),
}


class TestADeselectingRunIsNotReportedAsTruncated:
    """Every one of these ran each item it selected, so none is a truncated run.

    A false truncation is the one direction that costs something: it puts a
    warning annotation on a green pull request telling the author most of the
    suite never ran, and there is nothing in the log to contradict it.
    """

    @pytest.mark.parametrize(("log", "selected", "executed"), DESELECTING_RUNS.values(), ids=DESELECTING_RUNS)
    def test_a_run_that_reached_every_selected_item_reports_complete(
        self, log: str, selected: int, executed: int
    ) -> None:
        report = parse_run(log)

        assert report.outcome == "complete"
        assert report.selected == selected
        assert report.executed == executed
        assert report.never_ran == 0

    @pytest.mark.parametrize("log", [run[0] for run in DESELECTING_RUNS.values()], ids=DESELECTING_RUNS)
    def test_more_items_cannot_execute_than_were_collected(self, log: str) -> None:
        # A module skipped or failing at import is counted on the counts line and
        # is not one of the items, so leaving it in the total states an extent the
        # run cannot have had -- "11 of 10 items executed" reads as a typo and is
        # in fact the number the share of the suite is computed from.
        report = parse_run(log)

        assert report.executed is not None
        assert report.collected is not None
        assert report.executed <= report.collected


class TestTheSessionsOwnCountOwnsTheExtent:
    """The session counted its extent, so this module reports it rather than a second reading.

    tests/session_truncation.py states ``started`` and ``collected`` from
    ``pytest_runtest_logfinish`` and ``len(session.items)``. Re-deriving the same
    two numbers from the text gives a second answer with nothing to say which is
    right, so the derivation is the fallback for a log that carries no section.
    """

    STATED = (
        "collected 13 items / 5 deselected / 1 skipped / 8 selected\n"
        "==== session truncated: 4 of 8 collected tests ran ====\n"
        "4 collected tests never started, so the counts below are a floor, not a total.\n"
        "!!!! stopping after 1 failures !!!!\n"
        "==== 1 failed, 3 passed, 1 skipped, 5 deselected in 22.14s ====\n"
    )

    def test_the_stated_extent_is_the_reported_extent(self) -> None:
        report = parse_run(self.STATED)

        assert report.outcome == "truncated"
        assert report.executed == 4
        assert report.selected == 8
        assert report.never_ran == 4
        assert report.extent_source == "the session's own count"

    def test_the_report_names_which_reading_it_used(self) -> None:
        # The two readings are not interchangeable -- one was counted and one
        # reconstructed -- so a reader comparing this table against the terminal
        # can see which spoke without diffing the numbers.
        assert "| extent | the session's own count |" in render(parse_run(self.STATED))
        assert "| extent | derived from the log's counts |" in render(parse_run(TRUNCATED_LOG))

    def test_a_log_carrying_no_section_is_still_read(self) -> None:
        # The section is written at terminal summary time, so a run killed before
        # then carries none, as does any log kept from before it existed.
        report = parse_run(TRUNCATED_LOG)

        assert report.outcome == "truncated"
        assert report.never_ran == 12036
        assert report.extent_source == "derived from the log's counts"

    def test_the_section_is_not_mistaken_for_the_collection_or_counts_line(self) -> None:
        # It carries the words "collected" and "tests ran" and sits between the two
        # lines this module reads, so a pattern matching it as either would report
        # the section's own numbers as the whole run's.
        report = parse_run(self.STATED)

        assert report.collected == 13
        assert report.counts == {"failed": 1, "passed": 3, "skipped": 1}


class TestTheseAreTheLinesPytestWrites:
    """Grade the fixtures above against pytest itself rather than against this module.

    Every fixture here is a hand-written line, so the pins are only as good as the
    claim that pytest writes lines of that shape. One nested run settles it, and
    the same run is the oracle for the extent: the reporter counts it in process,
    so the derivation has to reproduce the number the session states.
    """

    @staticmethod
    def _run(tmp_path: Path, *args: str) -> str:
        (tmp_path / "test_many.py").write_text(
            "import pytest\n\n"
            '@pytest.mark.parametrize("i", range(8))\n'
            "def test_alpha(i):\n    assert i < 3\n\n"
            '@pytest.mark.parametrize("i", range(5))\n'
            "def test_beta(i):\n    assert True\n"
        )
        (tmp_path / "test_skipped_module.py").write_text(
            'import pytest\n\npytest.importorskip("a_module_that_is_not_installed")\n\ndef test_never(): ...\n'
        )
        env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
        env.pop("PYTEST_ADDOPTS", None)
        # cwd is the tmp tree so the nested run inherits none of this
        # repository's configuration, and no ``-q``: quiet suppresses the
        # collection line, which is the line under test.
        finished = subprocess.run(
            [sys.executable, "-m", "pytest", ".", "-p", "no:cacheprovider", "-p", "no:randomly", *args],
            capture_output=True,
            text=True,
            cwd=tmp_path,
            env=env,
            timeout=300,
        )
        return finished.stdout

    def test_a_token_really_does_sit_between_deselected_and_selected(self, tmp_path: Path) -> None:
        out = self._run(tmp_path, "-k", "alpha")

        assert "collected 13 items / 5 deselected / 1 skipped / 8 selected" in out
        # And that run reached all 8 of the items it selected.
        assert parse_run(out).outcome == "complete"

    def test_the_derived_extent_is_the_extent_the_session_counted(self, tmp_path: Path) -> None:
        out = self._run(tmp_path, "-k", "alpha", "-x", "-p", "tests.session_truncation")

        stated = re.search(r"session truncated: (\d+) of (\d+) collected tests ran", out)
        assert stated, "the reporter did not state an extent for a truncated run"
        counted = (int(stated.group(1)), int(stated.group(2)))

        # With the section present the numbers are the session's; with it removed
        # they are this module's arithmetic. Both must be what the session counted.
        without = "\n".join(
            line for line in out.splitlines() if "truncated" not in line and "never started" not in line
        )
        for report in (parse_run(out), parse_run(without)):
            assert report.outcome == "truncated"
            assert (report.executed, report.selected) == counted
