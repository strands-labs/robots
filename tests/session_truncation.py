"""State the size of a pytest session that stopped before every test ran.

A session that aborts early reports the same counts line as one that ran to
completion: ``1 failed, 34268 passed, 278 skipped`` reads as a total whether
34547 tests ran or 46583 did. pytest names the abort - ``!!! stopping after 1
failures !!!`` - but not its size, so how much of the suite never started is
recoverable only from the trailing progress percentage of the last verbose line,
one token at the end of a multi-megabyte log.

The distinction matters because the two reports mean different things. A
complete run's counts are a total, so a red check is a list of everything to
fix. A truncated run's counts are a floor, and the number of failures is
unknown until the suite is run again on a tree where the named one passes -
which costs another full run of the suite, and (because a push dismisses
approval) another review.

This plugin states the size and nothing else: how many collected tests ran, how
many never started, and that the counts are therefore a floor. It is silent on a
complete session, so the statement appears only where it changes what the counts
below it mean, and it is independent of *why* the session stopped - a
``--maxfail`` budget, an interrupt, or a test that took the process down all
truncate the report the same way.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from _pytest.terminal import TerminalReporter

#: Name the reporter is registered under, so a second registration is a no-op
#: rather than a duplicate section.
PLUGIN_NAME = "session_truncation_reporter"


def truncation_summary(*, collected: int | None, started: int) -> tuple[str, str] | None:
    """Describe a session that ran fewer tests than it collected.

    Args:
        collected: Number of tests collection settled on, after deselection, or
            ``None`` when collection never finished (nothing can be said then).
        started: Number of those tests that were entered.

    Returns:
        A ``(heading, detail)`` pair for the terminal summary, or ``None`` when
        the session ran every test it collected and there is nothing to say.
    """
    if collected is None or started >= collected:
        return None
    missed = collected - started
    heading = f"session truncated: {started} of {collected} collected tests ran"
    detail = f"{missed} collected tests never started, so the counts below are a floor, not a total."
    return heading, detail


class SessionTruncationReporter:
    """Report the size of a truncated session in the terminal summary."""

    def __init__(self) -> None:
        self._collected: int | None = None
        self._started = 0

    def pytest_collection_finish(self, session: pytest.Session) -> None:
        """Record the item count collection settled on, after deselection."""
        self._collected = len(session.items)

    def pytest_runtest_logfinish(self, nodeid: str) -> None:
        """Count a test that was entered, whatever its outcome."""
        self._started += 1

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self, terminalreporter: TerminalReporter, config: pytest.Config) -> None:
        """Write the truncation section, last, so it sits beside the counts."""
        if config.getoption("collectonly", default=False):
            return  # Collecting without running is not a truncated run.
        summary = truncation_summary(collected=self._collected, started=self._started)
        if summary is None:
            return
        heading, detail = summary
        terminalreporter.write_sep("=", heading, red=True)
        terminalreporter.write_line(detail)


def register_truncation_reporter(config: pytest.Config) -> None:
    """Register :class:`SessionTruncationReporter` once for this session."""
    if config.pluginmanager.get_plugin(PLUGIN_NAME) is not None:
        return
    config.pluginmanager.register(SessionTruncationReporter(), PLUGIN_NAME)


def pytest_configure(config: pytest.Config) -> None:
    """Register the reporter when this module is loaded as a plugin (``-p``)."""
    register_truncation_reporter(config)
