"""A timed ``move`` keeps the twist fresh for the SDK's dead-man watchdog.

The earth-rovers-sdk arms ``CONTROL_WATCHDOG_S`` (default 3 s) on every
accepted motion command and delivers a stop once no fresh command has been
confirmed for that long (``main.py``: ``arm_control_watchdog``); its README asks
moving clients to stream at about 10 Hz. The driver sent one twist and slept
``duration_s``, so ``move(linear=0.5, duration_s=10)`` drove for 3 s, stopped,
and still answered ``held_s=10, stopped=True``.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.drivers import earthrover
from strands_robots.drivers.earthrover import EarthRoverDriver

#: ``earth-rovers-sdk/main.py``: ``CONTROL_WATCHDOG_S = float(os.getenv("CONTROL_WATCHDOG_S", "3"))``.
SDK_WATCHDOG_S = 3.0
TWIST = {"linear": 0.5, "angular": 0.0}
STOP = {"linear": 0.0, "angular": 0.0}


class _Recorder:
    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def get(self, url: str, timeout: float = 0.0, **_: Any) -> Any:
        return types.SimpleNamespace(status_code=200, text="", json=lambda: {"battery": 90})

    def post(self, url: str, json: Any = None, timeout: float = 0.0, **_: Any) -> Any:
        self.posts.append(json["command"])
        return types.SimpleNamespace(status_code=200, text="", json=lambda: {})

    def close(self) -> None:
        pass


class _FakeClock:
    """``time`` stand-in for the driver: ``sleep`` advances ``monotonic``."""

    def __init__(self) -> None:
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def recorder(monkeypatch: pytest.MonkeyPatch) -> _Recorder:
    rec = _Recorder()
    fake = types.ModuleType("requests")
    fake.Session = lambda: rec  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "requests", fake)
    return rec


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> _FakeClock:
    fake = _FakeClock()
    monkeypatch.setattr(earthrover, "time", fake)
    return fake


def test_refresh_period_outpaces_the_sdk_watchdog() -> None:
    assert earthrover.MOVE_REFRESH_PERIOD_S < SDK_WATCHDOG_S


def test_a_long_hold_never_leaves_the_twist_stale_for_the_watchdog(recorder: _Recorder, clock: _FakeClock) -> None:
    driver = EarthRoverDriver()
    assert driver.connect_eagerly() is None
    answer = driver.move(linear=0.5, duration_s=10.0)
    assert answer["status"] == "success"
    assert answer["content"][0]["json"] == {"commanded": TWIST, "held_s": 10.0, "stopped": True}
    assert recorder.posts[-1] == STOP
    twists = recorder.posts[:-1]
    assert twists and set(map(tuple, (t.items() for t in twists))) == {tuple(TWIST.items())}
    # A single twist left alone for 10 s is stopped by the SDK after 3 s; the
    # hold must re-send it well inside that window, every time.
    assert len(twists) >= 10.0 / SDK_WATCHDOG_S
    assert clock.now == pytest.approx(10.0)


def test_a_short_hold_is_still_one_twist_and_one_stop(recorder: _Recorder, clock: _FakeClock) -> None:
    driver = EarthRoverDriver()
    assert driver.connect_eagerly() is None
    assert driver.move(linear=0.5, duration_s=0.01)["status"] == "success"
    assert recorder.posts == [TWIST, STOP]
