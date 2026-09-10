"""A kwarg ``Robot.__init__`` refuses must not be followed by a cleanup error.

``control_frequency`` and ``action_horizon`` are validated at the top of
``Robot.__init__``, before the executor, the shutdown latch, the mesh handle
or the device exist -- that ordering is the point, a refused rate never touches
the arm. The half-built instance is still finalised, and ``__del__`` called
``cleanup()`` on it unconditionally, so the first line of ``cleanup()`` raised
``AttributeError: 'Robot' object has no attribute '_shutdown_event'`` and the
handler logged it at ERROR beside the ``ValueError`` the caller was owed.

An instance without the shutdown latch never acquired anything, so there is
nothing to release and nothing to report. A bring-up that fails *after* the
latch exists -- the ``_initialize_robot`` path pinned in
``tests/test_hardware_cleanup_survives_a_failed_robot_init.py`` -- still runs
the full teardown.
"""

from __future__ import annotations

import gc
import logging
from typing import Any

import pytest

from strands_robots.hardware_robot import Robot as HwRobot

_LOGGER = "strands_robots.hardware_robot"

REFUSED_KWARGS: list[dict[str, Any]] = [
    {"control_frequency": 0},
    {"control_frequency": float("nan")},
    {"action_horizon": 0},
]


def _construct_refused(**kwargs: Any) -> None:
    """Construct with a kwarg ``__init__`` refuses, then force the finalizer.

    The traceback keeps the half-built instance alive while the exception
    propagates, so the ``ValueError`` is caught here and ``gc.collect()`` runs
    once the frame chain is dropped.
    """
    try:
        HwRobot(tool_name="probe", robot="so101_follower", **kwargs)
    except ValueError:
        gc.collect()
        return
    raise AssertionError(f"__init__ accepted {kwargs}")


def _cleanup_errors(caplog: pytest.LogCaptureFixture) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR and "Cleanup error" in r.getMessage()]


@pytest.mark.parametrize("kwargs", REFUSED_KWARGS, ids=lambda k: next(iter(k.items())).__repr__())
def test_a_refused_kwarg_logs_no_cleanup_error(caplog: pytest.LogCaptureFixture, kwargs: dict[str, Any]) -> None:
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        _construct_refused(**kwargs)
    assert _cleanup_errors(caplog) == []


def test_the_finalizer_skips_an_instance_without_a_shutdown_latch(caplog: pytest.LogCaptureFixture) -> None:
    """``__del__`` returns before ``cleanup()`` when ``__init__`` never got that far."""
    instance = HwRobot.__new__(HwRobot)
    instance.tool_name_str = "probe"
    with caplog.at_level(logging.ERROR, logger=_LOGGER):
        instance.__del__()
    assert _cleanup_errors(caplog) == []
