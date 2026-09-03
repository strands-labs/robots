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
