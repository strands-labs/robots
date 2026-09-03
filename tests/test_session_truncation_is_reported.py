"""A session that stops before every collected test ran says how much never ran.

``.github/workflows/test-lint.yml`` runs the required check under ``-x``, so the
suite can abort at the first failure with a quarter of its items unexecuted. The
counts line pytest then prints (``1 failed, 34268 passed, 278 skipped``) is
shaped exactly like a complete run's, and the abort banner names the stop
without sizing it - so a reader cannot tell a total from a floor. These cells
pin the statement that distinguishes them, and pin that it stays absent when the
session did run everything it collected.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]

_FIVE_TESTS = """\
def test_one(): pass
def test_two(): assert False
def test_three(): assert False
def test_four(): pass
def test_five(): pass
"""


def _run_pytest(target: Path, *args: str) -> subprocess.CompletedProcess[str]:
    """Run a nested pytest over ``target`` with the reporter loaded as a plugin.

    The nested run deliberately does not sit under this repository's rootdir, so
    it inherits none of its ``addopts`` and the only plugin under test is the one
    named on the command line.
    """
    env = {**os.environ, "PYTHONPATH": str(_REPO_ROOT)}
    env.pop("PYTEST_ADDOPTS", None)
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(target),
            "-p",
            "tests.session_truncation",
            "-p",
            "no:cacheprovider",
            "-q",
            *args,
        ],
        capture_output=True,
        text=True,
        cwd=target.parent,
        env=env,
        timeout=120,
    )


class TestTheSummaryStatesTheSizeOfWhatDidNotRun:
    """The message is derived from the two counts and says nothing else."""

    @staticmethod
    def _summary(collected: int | None, started: int) -> tuple[str, str] | None:
        from tests.session_truncation import truncation_summary

        return truncation_summary(collected=collected, started=started)

    @pytest.mark.parametrize(
        ("collected", "started", "expected_numbers"),
        [
            pytest.param(46583, 34547, ("34547", "46583", "12036"), id="the-measured-ci-run"),
            pytest.param(5, 2, ("2", "5", "3"), id="stopped-at-the-second-of-five"),
            pytest.param(2, 1, ("1", "2", "1"), id="one-test-short"),
        ],
    )
    def test_a_short_session_is_sized(self, collected: int, started: int, expected_numbers: tuple[str, ...]) -> None:
        summary = self._summary(collected, started)
        assert summary is not None, "a session that ran fewer tests than it collected has something to say"
        heading, detail = summary
        assert (heading, detail) == (
            f"session truncated: {started} of {collected} collected tests ran",
            f"{collected - started} collected tests never started, so the counts below are a floor, not a total.",
        )
        for number in expected_numbers:
            assert number in heading + detail

    @pytest.mark.parametrize(
        ("collected", "started"),
        [
            pytest.param(46583, 46583, id="every-collected-test-ran"),
            pytest.param(0, 0, id="nothing-was-collected"),
            pytest.param(None, 3, id="collection-never-finished"),
            pytest.param(5, 7, id="more-runs-than-items-stepwise-or-rerun"),
        ],
    )
    def test_a_session_with_nothing_to_report_is_silent(self, collected: int | None, started: int) -> None:
        assert self._summary(collected, started) is None


class TestTheSuiteReportsItsOwnTruncation:
    """The reporter is wired into this suite, not only importable by it."""

    def test_the_reporter_is_registered_for_this_session(self, pytestconfig: pytest.Config) -> None:
        from tests.session_truncation import PLUGIN_NAME, SessionTruncationReporter

        plugin = pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME)
        assert isinstance(plugin, SessionTruncationReporter), (
            "tests/conftest.py must register the reporter, or a truncated run of this suite says nothing"
        )

    def test_registering_twice_leaves_one_reporter(self, pytestconfig: pytest.Config) -> None:
        from tests.session_truncation import PLUGIN_NAME, register_truncation_reporter

        before = pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME)
        register_truncation_reporter(pytestconfig)
        assert pytestconfig.pluginmanager.get_plugin(PLUGIN_NAME) is before


class TestARunThatStopsEarlyReportsHowMuchNeverRan:
    """Driven end to end through a nested pytest, on its terminal output."""

    def test_maxfail_one_names_the_three_tests_that_never_started(self, tmp_path: Path) -> None:
        (tmp_path / "test_five.py").write_text(_FIVE_TESTS)
        result = _run_pytest(tmp_path, "-x")
        assert "1 failed, 1 passed" in result.stdout, result.stdout
        assert "session truncated: 2 of 5 collected tests ran" in result.stdout, result.stdout
        assert "3 collected tests never started, so the counts below are a floor, not a total." in result.stdout

    def test_a_complete_run_says_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "test_five.py").write_text(_FIVE_TESTS)
        result = _run_pytest(tmp_path)
        assert "2 failed, 3 passed" in result.stdout, result.stdout
        assert "session truncated" not in result.stdout, result.stdout

    def test_a_deselecting_run_that_finishes_says_nothing(self, tmp_path: Path) -> None:
        (tmp_path / "test_five.py").write_text(_FIVE_TESTS)
        result = _run_pytest(tmp_path, "-k", "test_one or test_four")
        assert "2 passed, 3 deselected" in result.stdout, result.stdout
        assert "session truncated" not in result.stdout, result.stdout

    def test_collecting_without_running_is_not_a_truncated_session(self, tmp_path: Path) -> None:
        (tmp_path / "test_five.py").write_text(_FIVE_TESTS)
        result = _run_pytest(tmp_path, "--collect-only")
        assert "5 tests collected" in result.stdout, result.stdout
        assert "session truncated" not in result.stdout, result.stdout
