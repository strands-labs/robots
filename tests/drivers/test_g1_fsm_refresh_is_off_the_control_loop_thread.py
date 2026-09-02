"""The FSM re-read is off the control-loop thread, and the cache it fills is bounded.

Review feedback on the FSM-producer PR named a physical-safety regression that
no test in the tree could see: wiring ``_refresh_fsm_id()`` into the top of
``_check_motion_gates`` put a **synchronous DDS RPC inside the 500 Hz control
loop**.  ``_ControlLoop._run`` re-gates every step, so with a real
``MotionSwitcherClient`` every 2 ms step performed a ``CheckMode()``
request/response round trip - and on a transient transport failure the loop
blocked for the SDK's whole RPC timeout, during which no frame published, no
``_stop_event`` was observed, and the joints drooped.  The injected recording
client in the sibling acceptance tests returns instantly, which is exactly why
those tests were blind to it.

This file grades the property that makes the wiring safe, in the shape that
made it unsafe:

1. **No ``CheckMode()`` ever runs on the control-loop thread.** Graded by
   thread identity, not by call count - a re-gate that switched to a cheaper
   RPC would still be a per-step RPC.
2. **A ``CheckMode()`` that blocks does not stall frame publication.** The
   direct regression pin: a client whose every read parks for many control
   periods, and a rollout that still publishes its whole step budget inside a
   wall-clock bound the pre-fix code cannot meet.
3. **The cache the per-step gate now reads is bounded.** Taking the RPC off the
   hot path means the gate consults a cached FSM; a cache nothing renews is not
   evidence about the robot, so the gate refuses once it passes the staleness
   bound and the rollout exits with a named reason and a zero-torque frame.
4. **The shared client is opened once and spoken to by one thread at a time.**
   ``_refresh_fsm_id`` is now reachable concurrently from the refresher thread
   and from an agent thread in ``send_action``; the check-then-act on
   ``_motion_switcher_client`` would otherwise open (and leak) two clients, and
   two ``CheckMode()`` calls would interleave on one request/response client the
   SDK does not document as thread-safe.  AGENTS.md > Review Learnings (#85):
   a shared-device lock is only a guarantee where every caller takes it.
5. **Admission is still authoritative.** Moving the RPC off the *loop* must not
   quietly move it off ``send_action`` / ``run_policy`` too: those pay one round
   trip so the decision that opens the wire is made on a fresh reading.

The sibling files stay where they were:
``test_g1_send_action_succeeds_on_a_healthy_wired_driver`` grades that a healthy
driver reaches the wire, and
``test_g1_battery_floor_reaches_with_wired_fsm`` grades the flipped
reachability.  Neither can see a cadence property, because neither runs the
loop against a slow wire.  This file is the one that does.
"""

from __future__ import annotations

import sys
import threading
import time
import types
from typing import Any

import pytest

from strands_robots.drivers import g1 as g1_module
from strands_robots.drivers.g1 import (
    _CONTROL_LOOP_HZ,
    _FSM_REFRESH_DT,
    _FSM_REFRESH_HZ,
    _FSM_STALE_AFTER_READS,
    _FSM_STALE_AFTER_S,
    G1Driver,
    _ControlLoop,
)

# ``501`` is in both HANDSHAKE_FSMS and WALK_FSMS, so the gate admits it for
# every scope and the rollout reaches the wire.
_HEALTHY_FSM_ID = 501
_HEALTHY_MODE_MACHINE = 9
_HEALTHY_PACK_PCT = 92.0


class _RecordingPublisher:
    """A ``_pubs`` stand-in whose ``.publish`` reports success (``None``)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls: list[tuple[str, Any, Any]] = []

    def publish(self, topic: str, msg_type: Any, cmd: Any) -> str | None:
        with self._lock:
            self.calls.append((topic, msg_type, cmd))
        return None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self.calls)

    def close(self) -> None:
        pass


class _ThreadRecordingMotionSwitcherClient:
    """A ``MotionSwitcherClient`` stand-in that records who called it, and when.

    ``delay_s`` is what makes this client able to see the regression: a real
    ``CheckMode()`` is a DDS round trip, and on a transient failure it parks
    until the SDK's RPC timeout.  Sleeping here reproduces that shape without a
    bus, so a test can assert the control loop kept its cadence *through* a
    slow wire rather than merely that it called the wire less often.
    """

    def __init__(
        self,
        fsm_id: int = _HEALTHY_FSM_ID,
        mode_name: str = "ai",
        delay_s: float = 0.0,
        refuse_after: int | None = None,
    ) -> None:
        self._fsm_id = fsm_id
        self._mode_name = mode_name
        self._delay_s = delay_s
        self._refuse_after = refuse_after
        self._lock = threading.Lock()
        self._in_flight = 0
        self.init_calls = 0
        self.check_mode_calls = 0
        self.overlaps = 0
        self.caller_idents: list[int] = []

    def Init(self) -> None:  # noqa: N802 - SDK spelling
        self.init_calls += 1

    def CheckMode(self) -> Any:  # noqa: N802 - SDK spelling
        with self._lock:
            self.check_mode_calls += 1
            n = self.check_mode_calls
            self.caller_idents.append(threading.get_ident())
            self._in_flight += 1
            if self._in_flight > 1:
                # Two callers inside one request/response client at once.
                self.overlaps += 1
        try:
            if self._delay_s:
                time.sleep(self._delay_s)
        finally:
            with self._lock:
                self._in_flight -= 1
        if self._refuse_after is not None and n > self._refuse_after:
            # ``status != 0`` is what the decoder turns into a named refusal,
            # which is the branch that deliberately keeps the previous
            # ``_fsm_id`` - i.e. the branch the staleness bound backs.
            return (3103, {})
        return (0, {"name": self._mode_name, "form": self._fsm_id})


def _install_sdk_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the ``unitree_sdk2py`` modules the publish path lazy-imports.

    ``_build_lowcmd_from_action`` imports ``idl.default`` and ``utils.crc``;
    ``_ControlLoop._run`` imports ``idl.unitree_hg.msg.dds_.LowCmd_``.  Installed
    with ``monkeypatch.setitem`` so ``sys.modules`` is restored at end of test
    and the module-load hygiene pins stay valid.
    """

    class _MotorCmdStub:
        def __init__(self) -> None:
            self.mode = 0
            self.q = 0.0
            self.dq = 0.0
            self.tau = 0.0
            self.kp = 0.0
            self.kd = 0.0
            self.reserve = 0

    class _StubLowCmd:
        def __init__(self) -> None:
            self.mode_machine = 0
            self.mode_pr = 0
            self.crc = 0
            self.motor_cmd = [_MotorCmdStub() for _ in range(35)]

    class _StubCRC:
        def Crc(self, _cmd: Any) -> int:  # noqa: N802 - SDK spelling
            return 0

    dds_ = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    dds_.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    crc_mod = types.ModuleType("unitree_sdk2py.utils.crc")
    crc_mod.CRC = _StubCRC  # type: ignore[attr-defined]

    for name, module in [
        ("unitree_sdk2py", types.ModuleType("unitree_sdk2py")),
        ("unitree_sdk2py.idl", types.ModuleType("unitree_sdk2py.idl")),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", types.ModuleType("unitree_sdk2py.idl.unitree_hg")),
        ("unitree_sdk2py.idl.unitree_hg.msg", types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds_),
        ("unitree_sdk2py.utils", types.ModuleType("unitree_sdk2py.utils")),
        ("unitree_sdk2py.utils.crc", crc_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, module)


def _healthy_driver(client: Any, publisher: _RecordingPublisher | None = None) -> G1Driver:
    """A driver whose every field is what a real, healthy G1 produces.

    ``_mode_machine`` and ``_battery`` are driven through their real decoders
    rather than assigned, so a decoder change is visible here instead of hidden
    behind a fixture that fabricates the decoder's output.
    """
    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        motion_switcher_client_factory=lambda _iface: client,
    )
    driver._connected = True
    driver._pubs = publisher if publisher is not None else _RecordingPublisher()  # type: ignore[assignment]
    driver._on_lowstate(
        types.SimpleNamespace(
            mode_machine=_HEALTHY_MODE_MACHINE,
            imu_state=types.SimpleNamespace(
                rpy=[0.0, 0.0, 0.0],
                gyroscope=[0.0, 0.0, 0.0],
                accelerometer=[0.0, 0.0, 9.81],
                quaternion=[1.0, 0.0, 0.0, 0.0],
            ),
        )
    )
    driver._on_bms(types.SimpleNamespace(soc=_HEALTHY_PACK_PCT, charge=0, current=0.0, cycle=0))
    return driver


def _wait_finished(driver: G1Driver, timeout: float = 5.0) -> dict[str, Any]:
    """Poll until the driver's loop has joined; return its terminal snapshot."""
    deadline = time.monotonic() + timeout
    loop = driver._loop
    assert loop is not None, "run_policy did not install a loop"
    while loop.is_running and time.monotonic() < deadline:
        time.sleep(0.005)
    assert not loop.is_running, "loop did not finish within timeout"
    return loop.snapshot()


def _constant_policy(_obs: Any) -> dict[str, float]:
    return {"left_knee": 0.0}


# ---------------------------------------------------------------------------
# The cadence relationship between the two threads.
# ---------------------------------------------------------------------------


class TestRefreshCadence:
    """The refresher runs far below the control loop, and the bound follows it."""

    def test_refresh_is_orders_of_magnitude_below_the_control_loop(self) -> None:
        # The whole point of the second thread is that the RPC cadence is not
        # the frame cadence.  A refresh rate that crept up towards 500 Hz
        # would be a per-step RPC wearing a thread.
        assert _FSM_REFRESH_HZ < _CONTROL_LOOP_HZ / 10

    def test_dt_is_the_reciprocal(self) -> None:
        assert _FSM_REFRESH_DT == pytest.approx(1.0 / _FSM_REFRESH_HZ, rel=1e-9)

    def test_staleness_bound_is_expressed_in_missed_reads(self) -> None:
        # The tolerated failure count is the invariant; the seconds are derived.
        # Retuning the cadence must not silently retune how many transient
        # ``CheckMode`` failures the gate absorbs.
        assert _FSM_STALE_AFTER_S == pytest.approx(_FSM_STALE_AFTER_READS * _FSM_REFRESH_DT, rel=1e-9)
        assert _FSM_STALE_AFTER_READS > 1


# ---------------------------------------------------------------------------
# 1 + 2: the RPC is not on the loop thread, and a slow RPC does not stall it.
# ---------------------------------------------------------------------------


class TestNoDdsRpcOnTheControlLoopThread:
    """``CheckMode()`` never runs on the thread that publishes frames."""

    def test_check_mode_never_runs_on_the_loop_thread(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_sdk_stubs(monkeypatch)
        client = _ThreadRecordingMotionSwitcherClient()
        publisher = _RecordingPublisher()
        driver = _healthy_driver(client, publisher)

        loop_idents: set[int] = set()

        def policy(_obs: Any) -> dict[str, float]:
            # The policy is invoked from inside ``_run``, so this *is* the
            # control-loop thread's identity - captured from production code
            # rather than guessed from a thread name.
            loop_idents.add(threading.get_ident())
            return {"left_knee": 0.0}

        result = driver.run_policy(policy, duration=60.0, n_steps=8)
        assert result["status"] == "success"
        _wait_finished(driver)

        assert len(loop_idents) == 1, "the loop should run on exactly one thread"
        assert client.check_mode_calls >= 1, "admission must still read the wire"
        assert loop_idents.isdisjoint(client.caller_idents), (
            "CheckMode() ran on the control-loop thread: "
            f"loop={loop_idents}, callers={sorted(set(client.caller_idents))}"
        )

    def test_a_blocking_check_mode_does_not_stall_frame_publication(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The direct regression pin for the reported hazard.

        Every ``CheckMode()`` parks for ``delay_s``, which is 25 control
        periods - the shape of one transient transport failure, scaled down so
        the test is fast.  With the read on the loop thread the 8-step budget
        could not complete in under 8 * 0.05 s = 0.4 s; the bound below is
        deliberately tighter than that, so this test fails on the pre-fix
        arrangement and passes only when the frames are decoupled from the RPC.
        """
        _install_sdk_stubs(monkeypatch)
        delay_s = 25 * (1.0 / _CONTROL_LOOP_HZ)  # 0.05 s
        client = _ThreadRecordingMotionSwitcherClient(delay_s=delay_s)
        publisher = _RecordingPublisher()
        driver = _healthy_driver(client, publisher)

        n_steps = 8
        # Admission pays one round trip by design; time only the rollout.
        result = driver.run_policy(_constant_policy, duration=60.0, n_steps=n_steps)
        assert result["status"] == "success"
        started = time.monotonic()
        snap = _wait_finished(driver)
        rollout_s = time.monotonic() - started

        assert snap["exit_reason"] == "n_steps", snap
        assert snap["steps"] == n_steps
        # n_steps frames plus the zero-torque stop frame.
        assert publisher.count == n_steps + 1
        assert rollout_s < n_steps * delay_s, (
            f"the rollout took {rollout_s:.3f}s for {n_steps} steps against a "
            f"{delay_s:.3f}s wire: frame publication is still serialised behind the FSM read"
        )


# ---------------------------------------------------------------------------
# 3: the cache the per-step gate reads is bounded.
# ---------------------------------------------------------------------------


class TestStalenessBoundEndsTheRollout:
    """A cached FSM nothing renews stops admitting, with a named reason."""

    def test_a_cache_nothing_renews_refuses_the_step(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_sdk_stubs(monkeypatch)
        # One good reading (admission), refusals forever after.  This is the
        # branch ``_refresh_fsm_id`` deliberately absorbs: ``_fsm_id`` keeps its
        # value, so without a bound the gate would admit for the whole rollout
        # on a reading that stopped being evidence.
        client = _ThreadRecordingMotionSwitcherClient(refuse_after=1)
        publisher = _RecordingPublisher()
        # Shrink the bound rather than sleep through the shipped one: the
        # constant is what production reads, so patching it grades the same
        # branch in a fraction of the wall clock.
        monkeypatch.setattr(g1_module, "_FSM_STALE_AFTER_S", 0.05)
        driver = _healthy_driver(client, publisher)

        result = driver.run_policy(_constant_policy, duration=60.0, n_steps=None)
        assert result["status"] == "success"
        snap = _wait_finished(driver, timeout=5.0)

        assert snap["exit_reason"] == "gate", snap
        assert snap["exit_detail"] is not None
        assert "staleness bound" in snap["exit_detail"], snap["exit_detail"]
        assert str(_HEALTHY_FSM_ID) in snap["exit_detail"], snap["exit_detail"]
        # A refused step is a controlled stop, not a freeze on the last posture.
        assert publisher.count >= 1
        # The rollout published before it stopped: the bound ends a rollout that
        # went blind, it does not prevent one from starting.
        assert snap["steps"] >= 1, snap

    def test_a_renewed_cache_keeps_admitting(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The bound must not fire on a healthy wire - otherwise it is a shutdown."""
        _install_sdk_stubs(monkeypatch)
        client = _ThreadRecordingMotionSwitcherClient()
        publisher = _RecordingPublisher()
        monkeypatch.setattr(g1_module, "_FSM_REFRESH_DT", 0.005)
        monkeypatch.setattr(g1_module, "_FSM_STALE_AFTER_S", 0.2)
        driver = _healthy_driver(client, publisher)

        driver.run_policy(_constant_policy, duration=60.0, n_steps=200)
        snap = _wait_finished(driver, timeout=5.0)

        assert snap["exit_reason"] == "n_steps", snap
        assert snap["steps"] == 200


# ---------------------------------------------------------------------------
# 4: the shared client is opened once and spoken to by one thread at a time.
# ---------------------------------------------------------------------------


class TestSharedClientIsSerialised:
    """Two refresh callers do not open two clients, nor interleave one."""

    def test_concurrent_refresh_opens_exactly_one_client(self) -> None:
        created: list[_ThreadRecordingMotionSwitcherClient] = []
        factory_calls = 0
        factory_lock = threading.Lock()

        def factory(_iface: str) -> Any:
            nonlocal factory_calls
            with factory_lock:
                factory_calls += 1
            # Widen the check-then-act window the way a real ``Init()`` +
            # DDS bring-up does; an unguarded early return loses this race.
            time.sleep(0.05)
            client = _ThreadRecordingMotionSwitcherClient()
            created.append(client)
            return client

        driver = G1Driver(tool_name="g1", motion_switcher_client_factory=factory)

        threads = [threading.Thread(target=driver._refresh_fsm_id) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert all(not t.is_alive() for t in threads)

        assert factory_calls == 1, f"the factory ran {factory_calls} times; a client was leaked"
        assert len(created) == 1
        assert created[0].init_calls == 0, "an injected client is opened by the factory, not by the driver"
        assert driver._motion_switcher_client is created[0]
        assert driver._fsm_id == _HEALTHY_FSM_ID

    def test_concurrent_check_mode_calls_do_not_interleave(self) -> None:
        # ``MotionSwitcherClient`` is a request/response client and is not
        # documented thread-safe: two in-flight CheckMode() calls on one client
        # is the failure this serialises away.
        client = _ThreadRecordingMotionSwitcherClient(delay_s=0.02)
        driver = G1Driver(tool_name="g1", motion_switcher_client_factory=lambda _i: client)

        threads = [threading.Thread(target=driver._refresh_fsm_id) for _ in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=5.0)
        assert all(not t.is_alive() for t in threads)

        assert client.check_mode_calls == 6
        assert client.overlaps == 0, f"{client.overlaps} CheckMode() calls overlapped on one client"

    def test_the_loop_thread_never_takes_the_client_lock(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A refresher parked in an RPC must not be able to block a frame.

        Holding the driver's motion-switcher lock for the whole of a slow read
        is only safe because the publishing thread never asks for it.  Held
        here across an entire rollout, from outside the driver.
        """
        _install_sdk_stubs(monkeypatch)
        client = _ThreadRecordingMotionSwitcherClient()
        publisher = _RecordingPublisher()
        driver = _healthy_driver(client, publisher)

        # Admission first (it does take the lock), then hold the lock for the
        # rest of the rollout as a wedged refresher would.
        monkeypatch.setattr(g1_module, "_FSM_STALE_AFTER_S", 60.0)
        result = driver.run_policy(_constant_policy, duration=60.0, n_steps=6)
        assert result["status"] == "success"
        with driver._motion_switcher_lock:
            snap = _wait_finished(driver, timeout=5.0)

        assert snap["exit_reason"] == "n_steps", snap
        assert snap["steps"] == 6


# ---------------------------------------------------------------------------
# 5: admission still pays for an authoritative reading.
# ---------------------------------------------------------------------------


class TestAdmissionStillReadsThroughTheWire:
    """Taking the RPC off the loop must not take it off the entry points."""

    def test_send_action_takes_a_fresh_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_sdk_stubs(monkeypatch)
        client = _ThreadRecordingMotionSwitcherClient()
        driver = _healthy_driver(client)

        first = driver.send_action({"left_knee": 0.1})
        assert first["status"] == "success", first
        assert client.check_mode_calls == 1

        # A second call reads again rather than trusting the first reading:
        # ``send_action`` is a one-shot entry point, and the FSM may have moved.
        second = driver.send_action({"left_knee": 0.2})
        assert second["status"] == "success", second
        assert client.check_mode_calls == 2

    def test_run_policy_admission_reads_before_it_starts_a_loop(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_sdk_stubs(monkeypatch)
        # A wire that never yields an FSM must refuse admission rather than
        # start a loop that will discover it from a cache.
        client = _ThreadRecordingMotionSwitcherClient(refuse_after=0)
        driver = _healthy_driver(client)

        result = driver.run_policy(_constant_policy, duration=1.0, n_steps=1)

        assert result["status"] == "error", result
        assert driver._loop is None, "a refused admission must not leave a loop installed"
        assert client.check_mode_calls == 1


# ---------------------------------------------------------------------------
# The snapshot names the thread the gate now depends on.
# ---------------------------------------------------------------------------


class TestSnapshotNamesTheRefresher:
    """ "The gate reads a cache" is only safe while something fills it."""

    def test_snapshot_reports_the_refresh_cadence_and_read_count(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_sdk_stubs(monkeypatch)
        client = _ThreadRecordingMotionSwitcherClient()
        monkeypatch.setattr(g1_module, "_FSM_REFRESH_DT", 0.002)
        driver = _healthy_driver(client)

        driver.run_policy(_constant_policy, duration=60.0, n_steps=100)
        snap = _wait_finished(driver, timeout=5.0)

        assert snap["fsm_refresh_hz"] == _FSM_REFRESH_HZ
        assert snap["fsm_reads"] >= 1, (
            "the refresher landed no readings during the rollout; the per-step gate was admitting on an unrenewed cache"
        )

    def test_snapshot_shape_is_stable_before_start(self) -> None:
        client = _ThreadRecordingMotionSwitcherClient()
        driver = G1Driver(tool_name="g1", motion_switcher_client_factory=lambda _i: client)
        loop = _ControlLoop(driver=driver, policy=_constant_policy, duration=1.0, n_steps=1)

        snap = loop.snapshot()
        assert snap["fsm_reads"] == 0
        assert snap["fsm_refresh_hz"] == _FSM_REFRESH_HZ
        # No refresher thread exists until ``start()``, so nothing has touched
        # the wire: constructing a loop is not an SDK side effect.
        assert client.check_mode_calls == 0
