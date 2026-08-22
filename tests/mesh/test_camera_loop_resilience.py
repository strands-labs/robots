"""Resilience + clean-shutdown contract for the mesh camera-publish loop.

``Mesh._camera_loop`` is the background thread that publishes camera frames on
the mesh at a fixed rate. Two guarantees matter and are pinned here:

  1. A transient error from a single ``_publish_cameras_once`` tick (a camera
     that momentarily fails to render or JPEG-encode) MUST NOT kill the loop -
     it is logged and the loop keeps publishing on the next tick.
  2. The loop shuts down promptly when ``_stop_event`` is signalled, and paces
     itself at ``period = 1 / hz`` (so stop is observed within a fraction of an
     interval rather than after a full sleep).

Both are asserted through a real :class:`threading.Event` that the tick body sets,
so neither depends on which primitive the loop paces on. An earlier version drove
the loop by scripting ``_stop_event.wait`` return values, which made the test a
statement about the pacing call rather than about either guarantee; the period is
now read off the ticker the loop builds.

The loop only touches ``_running``, ``_publish_cameras_once``, ``_stop_event``
and ``peer_id``, so it is exercised on a bare instance built with
``Mesh.__new__`` (the same construction pattern used by the other mesh unit
tests) - no zenoh transport or live robot required.
"""

from __future__ import annotations

import threading
from unittest.mock import MagicMock

import pytest

import strands_robots.mesh.core as core_mod
from strands_robots.mesh.core import Mesh


class _InstantTicker:
    """Records the period the loop paces on, and never actually sleeps.

    Standing in for the real ticker keeps these tests instant while leaving the
    loop's own structure - build a ticker, ask it once per tick, close it -
    exactly as it runs in production.
    """

    periods: list[float] = []

    def __init__(self, period, stop_event=None):
        self.period = period
        self._stop = stop_event
        type(self).periods.append(period)

    def wait(self):
        return bool(self._stop is not None and self._stop.is_set())

    def close(self):
        pass

    # The loop enters the ticker with `with`, so the double carries the same
    # protocol as the real one.
    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()


def _bare_mesh(publish, stop):
    """A Mesh with just the attributes ``_camera_loop`` reads.

    Args:
        publish: the ``_publish_cameras_once`` callable (a mock).
        stop: the loop's stop event, a real :class:`threading.Event`.
    """
    mesh = Mesh.__new__(Mesh)
    mesh.peer_id = "test__arm"
    mesh._running = True
    mesh._publish_cameras_once = publish
    mesh._stop_event = stop
    return mesh


def _stop_after(stop, n, side_effect=None):
    """A tick body that signals ``stop`` on its ``n``-th call.

    Bounding the loop from the tick body rather than from the pacing call is
    what makes these tests independent of how the loop waits.
    """
    calls = {"n": 0}

    def tick():
        calls["n"] += 1
        if calls["n"] >= n:
            stop.set()
        if side_effect is not None:
            raise side_effect

    return tick


def test_camera_loop_publishes_each_tick_and_stops_on_event(monkeypatch: pytest.MonkeyPatch):
    _InstantTicker.periods = []
    monkeypatch.setattr(core_mod, "Ticker", _InstantTicker)
    stop = threading.Event()
    publish = MagicMock(side_effect=_stop_after(stop, 3))
    mesh = _bare_mesh(publish, stop)

    mesh._camera_loop(10.0)

    # Three ticks ran, and the third one's stop ended the loop.
    assert publish.call_count == 3
    # Paces at period = 1 / hz so a stop is observed within a fraction of one.
    assert _InstantTicker.periods == [0.1]


def test_camera_loop_swallows_tick_error_and_keeps_going(monkeypatch: pytest.MonkeyPatch):
    # Every tick raises; the loop must log and continue rather than die on the
    # first failure. Stop is signalled by the second tick, which also raises -
    # so the loop has to survive the error AND honour the stop set alongside it.
    _InstantTicker.periods = []
    monkeypatch.setattr(core_mod, "Ticker", _InstantTicker)
    stop = threading.Event()
    publish = MagicMock(side_effect=_stop_after(stop, 2, RuntimeError("camera render blipped")))
    mesh = _bare_mesh(publish, stop)

    # No exception escapes the loop.
    mesh._camera_loop(20.0)

    # It kept publishing after the first error (resilience), then stopped.
    assert publish.call_count == 2
    assert _InstantTicker.periods == [0.05]
