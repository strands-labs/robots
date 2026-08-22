"""A mesh duration must survive a wall-clock step.

Three surfaces in ``strands_robots.mesh`` decide or report something from a
*duration* - a difference between two readings of a clock:

- the peer registry (``session.py``): ``PeerInfo.age`` and :func:`prune_peers`
  decide whether a peer is still alive, and the cap eviction picks the peer
  whose heartbeat is oldest;
- the LiDAR state throttle (``sensors.py``): a publish interval;
- the input publisher/receiver ``stats`` (``input.py``): ``hz_actual``.

Each was measured against :func:`time.time`, which is not a clock but the
current opinion about the date. An NTP correction, a ``date -s`` or a resume
from suspend moves it by an arbitrary amount, and every one of those durations
moved with it.

The tests drive the real functions through a clock double that takes a known
step, and assert on what the mesh *does* - which peers survive a prune, what age
is reported, which peer the cap evicts, how often state is published, what rate
is reported. The contract is pinned on behaviour, not on which clock is called,
so a future refactor cannot satisfy it by renaming a call.

Absolute stamps are deliberately not covered here: the ``"t"`` field of a
published envelope, and the ``_age`` an input receiver computes against a
*remote* peer's ``t``, name a point in time that something off this process
correlates, and those must stay on the wall clock.
"""

from __future__ import annotations

import threading
from typing import Any
from unittest.mock import patch

import pytest

from strands_robots.mesh import input as mesh_input
from strands_robots.mesh import sensors as mesh_sensors
from strands_robots.mesh import session as mesh_session

# Clock double


class SteppableClock:
    """A wall clock that can be corrected, beside a monotonic one that cannot.

    ``advance`` is real time passing: both clocks move together. ``step_wall``
    is the correction - an NTP slew, a ``date -s``, a resume from suspend - and
    moves only :meth:`time`. Correct code reads :meth:`monotonic` and is
    therefore immune to any number of steps.
    """

    def __init__(self, *, wall_base: float = 1_700_000_000.0, mono_base: float = 1000.0) -> None:
        self._elapsed = 0.0
        self._offset = 0.0
        self._wall_base = wall_base
        self._mono_base = mono_base

    def advance(self, seconds: float) -> None:
        self._elapsed += seconds

    def step_wall(self, seconds: float) -> None:
        self._offset += seconds

    def monotonic(self) -> float:
        return self._mono_base + self._elapsed

    def time(self) -> float:
        return self._wall_base + self._elapsed + self._offset


@pytest.fixture
def clock() -> SteppableClock:
    return SteppableClock()


# The double itself, so that a clean run below means something


def test_the_double_advance_moves_both_clocks(clock: SteppableClock) -> None:
    wall, mono = clock.time(), clock.monotonic()
    clock.advance(7.0)
    assert clock.time() == pytest.approx(wall + 7.0)
    assert clock.monotonic() == pytest.approx(mono + 7.0)


@pytest.mark.parametrize("step", [30.0, 3600.0, -30.0, -3600.0])
def test_the_double_step_moves_only_the_wall_clock(clock: SteppableClock, step: float) -> None:
    wall, mono = clock.time(), clock.monotonic()
    clock.step_wall(step)
    assert clock.time() == pytest.approx(wall + step), "the double must really hand out the step"
    assert clock.monotonic() == pytest.approx(mono), "monotonic must not move on a wall-clock step"


# Peer registry: liveness


@pytest.fixture
def peers(clock: SteppableClock, monkeypatch: pytest.MonkeyPatch):
    """The real peer registry, reading the clock double, empty before and after."""
    monkeypatch.setattr(mesh_session, "time", clock)
    mesh_session.clear_peers()
    try:
        yield mesh_session
    finally:
        mesh_session.clear_peers()


def _register(peers, *peer_ids: str) -> None:
    for peer_id in peer_ids:
        peers.update_peer(peer_id, "robot", "host", {})


def test_a_live_peer_survives_a_prune_and_reports_its_real_age(peers, clock: SteppableClock) -> None:
    """No step: the control, and the bound on this change.

    This is what fails if the fix is ever mistaken for a change in how long a
    peer is kept.
    """
    _register(peers, "arm-1", "arm-2")
    clock.advance(5.0)

    assert peers.prune_peers() == []
    assert peers.peer_count() == 2
    assert {p["peer_id"]: p["age"] for p in peers.get_peers()} == {"arm-1": 5.0, "arm-2": 5.0}


@pytest.mark.parametrize("step", [30.0, 3600.0])
def test_a_forward_wall_clock_step_does_not_prune_a_heartbeating_peer(
    peers, clock: SteppableClock, step: float
) -> None:
    """Forward, every peer's apparent age jumps by the step at once.

    ``prune_peers`` runs on every heartbeat tick, so a step larger than
    PEER_TIMEOUT made the next tick delete the entire fleet - each of them
    heartbeating normally.
    """
    _register(peers, "arm-1", "arm-2", "base-1")
    clock.advance(5.0)
    clock.step_wall(step)

    pruned = peers.prune_peers()

    assert pruned == [], f"a +{step}s wall-clock step pruned live peers: {sorted(pruned)}"
    assert peers.peer_count() == 3


@pytest.mark.parametrize("step", [-30.0, -3600.0])
def test_a_backward_wall_clock_step_still_prunes_a_dead_peer(peers, clock: SteppableClock, step: float) -> None:
    """Backward is the direction that matters: a dead peer reported live.

    The peer below stopped heartbeating 15s ago, past the 10s PEER_TIMEOUT. A
    backward step makes its apparent age negative, so the comparison never
    fires and the registry keeps advertising a peer that is gone.
    """
    _register(peers, "dead-1")
    clock.advance(mesh_session.PEER_TIMEOUT + 5.0)
    clock.step_wall(step)

    pruned = peers.prune_peers()

    assert pruned == ["dead-1"], (
        f"a {step}s wall-clock step kept a peer silent for {mesh_session.PEER_TIMEOUT + 5.0}s in the registry as live"
    )
    assert peers.peer_count() == 0


@pytest.mark.parametrize("step", [30.0, -30.0])
def test_a_reported_peer_age_tracks_real_time_across_a_wall_clock_step(
    peers, clock: SteppableClock, step: float
) -> None:
    """``age`` is what an operator and an agent read out of ``to_dict``.

    A negative age is the visible form of the backward case; an inflated one is
    the forward case.
    """
    _register(peers, "arm-1")
    clock.advance(4.0)
    clock.step_wall(step)

    (peer,) = peers.get_peers()

    assert peer["age"] == pytest.approx(4.0), f"a {step}s wall-clock step misreported the age"
    assert peer["age"] >= 0.0, "an age is a duration and cannot be negative"


def test_the_peer_cap_evicts_the_oldest_peer_across_a_wall_clock_step(
    peers, clock: SteppableClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The flood guard evicts the oldest heartbeat; a step inverted the order.

    ``update_peer`` bounds a phantom-peer flood by evicting the peer whose
    heartbeat is oldest. Peers stamped either side of a backward step are
    ordered by the step rather than by age, so the guard evicted the *freshest*
    peer and kept the stalest - the opposite of what bounds the flood.
    """
    monkeypatch.setenv("STRANDS_MESH_MAX_PEERS", "3")

    _register(peers, "oldest")
    clock.advance(1.0)
    _register(peers, "middle")
    clock.advance(1.0)
    clock.step_wall(-30.0)
    _register(peers, "newest")  # stamped on the far side of the step
    clock.advance(1.0)

    before = {p["peer_id"] for p in peers.get_peers()}
    _register(peers, "arriving")  # at the cap: forces one eviction
    after = {p["peer_id"] for p in peers.get_peers()}

    assert before - after == {"oldest"}, (
        f"a -30.0s wall-clock step evicted {sorted(before - after)} rather than the oldest peer"
    )


# LiDAR state throttle


class _ScriptedTicker:
    """Stands in for the loop's ticker, advancing the clock per tick.

    The ticker's ``wait()`` is where a real run spends its time, so driving it is
    what makes the cadence deterministic: each call advances the double by exactly
    one period, applies any scripted wall-clock step, and stops the loop after
    ``ticks`` iterations. It sits exactly where the loop's old
    ``_stop_event.wait(period)`` sat, so the cadence this drives is unchanged -
    only the primitive being driven moved.
    """

    def __init__(self, clock: SteppableClock, ticks: int, step_at: int | None, step: float) -> None:
        self._clock = clock
        self._ticks = ticks
        self._step_at = step_at
        self._step = step
        self.count = 0
        self.period = 0.0
        self.closed = False

    def build(self, period: float, stop_event: Any = None) -> _ScriptedTicker:
        """Ticker-shaped constructor, so the loop builds THIS from its period."""
        self.period = period
        return self

    def wait(self) -> bool:
        self.count += 1
        self._clock.advance(self.period)
        if self._step_at is not None and self.count == self._step_at:
            self._clock.step_wall(self._step)
        return self.count >= self._ticks

    def close(self) -> None:
        self.closed = True

    # The loops enter the ticker with `with`, so the double carries the same
    # protocol: a stand-in that is not a context manager would fail on the
    # construct rather than on the cadence this test is about.
    def __enter__(self) -> _ScriptedTicker:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


class _LidarHost:
    """The minimum surface ``SensorLoopsMixin._lidar_loop`` reads.

    Each state publish is recorded against real elapsed time - the monotonic
    reading, which no scripted step moves - so the assertions can pin *when*
    state was published rather than only how often.
    """

    def __init__(self, clock: SteppableClock, stop_event: threading.Event) -> None:
        self.peer_id = "test-peer"
        self.robot = None
        self._running = True
        self._stop_event = stop_event
        self._clock = clock
        self._origin = clock.monotonic()
        self.state_at: list[float] = []
        self.summary_at: list[float] = []

    # The real pacing generator, so this drives the shipped ownership rules
    # (one ticker per loop, released even when the body raises) rather than a
    # reimplementation of them. The ticker it builds is the scripted double.
    _paced = mesh_sensors.SensorLoopsMixin._paced

    def publish(self, topic: str, payload: dict[str, Any]) -> None:
        elapsed = round(self._clock.monotonic() - self._origin, 6)
        if topic.endswith("/lidar/state"):
            self.state_at.append(elapsed)
        elif topic.endswith("/lidar/summary"):
            self.summary_at.append(elapsed)

    def _read_lidar_summary(self) -> dict[str, Any]:
        return {"n": 1}

    def _read_lidar_state(self) -> dict[str, Any]:
        return {"state": "ok"}


def _run_lidar(clock: SteppableClock, ticks: int, step_at: int | None = None, step: float = 0.0) -> _LidarHost:
    """Run the real loop for *ticks* summary ticks against the clock double."""
    ticker = _ScriptedTicker(clock, ticks, step_at, step)
    host = _LidarHost(clock, threading.Event())
    with patch.object(mesh_sensors, "Ticker", ticker.build):
        mesh_sensors.SensorLoopsMixin._lidar_loop(host)  # type: ignore[arg-type]
    assert ticker.period == pytest.approx(1.0 / mesh_sensors.LIDAR_SUMMARY_HZ), (
        "the loop must pace on its summary period"
    )
    assert ticker.closed, "the loop must release its ticker"
    return host


# Summary runs at 5 Hz and state at 1 Hz, so eleven summary ticks span 2.0s of
# real time and state is due at 0.0s, 1.0s and 2.0s.
LIDAR_TICKS = 11
STATE_DUE_AT = [0.0, 1.0, 2.0]


@pytest.fixture
def lidar_clock(clock: SteppableClock, monkeypatch: pytest.MonkeyPatch) -> SteppableClock:
    monkeypatch.setattr(mesh_sensors, "time", clock)
    return clock


def test_lidar_state_publishes_on_its_period(lidar_clock: SteppableClock) -> None:
    """No step: the control, and the bound on this change."""
    host = _run_lidar(lidar_clock, ticks=LIDAR_TICKS)

    assert host.state_at == STATE_DUE_AT
    assert len(host.summary_at) == LIDAR_TICKS, "summary is unthrottled and ticks every period"


@pytest.mark.parametrize("step", [-30.0, -3600.0, 30.0, 3600.0])
def test_a_wall_clock_step_does_not_move_the_lidar_state_period(lidar_clock: SteppableClock, step: float) -> None:
    """A publish interval is a duration, so a step must not move it.

    Backward, ``now - last_publish`` goes negative and stays negative, so state
    publishing stalls for the size of the step while summary keeps flowing.
    Forward, the interval reads as satisfied immediately and state publishes a
    period early. Both are pinned by *when* state was published, because a count
    alone can come out right for the wrong reason.
    """
    host = _run_lidar(lidar_clock, ticks=LIDAR_TICKS, step_at=1, step=step)

    assert host.state_at == STATE_DUE_AT, (
        f"a {step}s wall-clock step published lidar state at {host.state_at} rather than {STATE_DUE_AT}"
    )
    assert len(host.summary_at) == LIDAR_TICKS, "summary must be unaffected either way"


# Input stats: hz_actual


class _StubMesh:
    peer_id = "test-peer"

    def subscribe(self, *_args: Any, **_kwargs: Any) -> str:
        return "sub-1"


def _publisher(monkeypatch: pytest.MonkeyPatch) -> Any:
    pub = mesh_input.InputPublisher(
        mesh=_StubMesh(),  # type: ignore[arg-type]
        teleoperator=object(),
        device_name="leader",
        hz=50.0,
    )
    # The loop body is not under test here; the stamp ``start()`` takes and the
    # duration ``stats`` derives from it are.
    monkeypatch.setattr(pub, "_publish_loop", lambda: None)
    return pub


def _receiver(monkeypatch: pytest.MonkeyPatch) -> Any:
    return mesh_input.InputReceiver(
        mesh=_StubMesh(),  # type: ignore[arg-type]
        robot=object(),
        source_peer_id="leader-peer",
        device_name="leader",
    )


@pytest.fixture(params=["publisher", "receiver"])
def input_side(request: pytest.FixtureRequest, clock: SteppableClock, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(mesh_input, "time", clock)
    side = _publisher(monkeypatch) if request.param == "publisher" else _receiver(monkeypatch)
    side.start()
    try:
        yield side
    finally:
        side._running = False


@pytest.mark.parametrize("step", [0.0, 30.0, -30.0, 3600.0])
def test_input_hz_actual_is_measured_on_a_clock_that_cannot_step(
    input_side: Any, clock: SteppableClock, step: float
) -> None:
    """``hz_actual`` is ``frames / elapsed``, so elapsed is a duration.

    ``step == 0.0`` is the control: the reported rate is unchanged by this fix
    when nothing corrects the clock.
    """
    clock.advance(2.0)
    input_side._frame_count = 100
    clock.step_wall(step)

    assert input_side.stats["hz_actual"] == pytest.approx(50.0), (
        f"a {step}s wall-clock step misreported the achieved rate"
    )


def test_input_thread_is_not_left_running(monkeypatch: pytest.MonkeyPatch) -> None:
    """Guard for the fixture above, which stubs the publisher's loop body."""
    pub = _publisher(monkeypatch)
    pub.start()
    pub.stop()
    assert pub._running is False
    assert threading.active_count() >= 1
