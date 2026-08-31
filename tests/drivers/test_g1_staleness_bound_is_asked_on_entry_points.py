"""Pin the staleness bound on entry-point paths (not just the control loop).

Peer bot on harness#361 measured: at PR #2916 head e8fbfb63, the staleness
bound sits inside ``if not refresh:`` in ``_check_motion_gates``, so it fires
for the loop's per-step re-gate but not for ``send_action`` / ``start_task`` /
``run_policy`` admission.  When the refresher's ``CheckMode()`` starts failing
transiently, ``_refresh_fsm_id``'s refused-reading branch deliberately keeps
``_fsm_id`` and deliberately does not stamp ``_fsm_read_at`` - so an entry
point on a driver whose FSM has not been confirmed for arbitrarily long still
publishes a frame.

The pin: same driver, same cached reading, same age past the same bound - the
loop stops and ``send_action`` refuses.  The gate exists to prevent silent
writes to a robot whose handshake we can no longer see; asking it on one path
and not the other is the same shape of silent-open the previous fix closed for
the RPC-on-hot-path hazard.

These cells fail against PR #2916 head e8fbfb63 and pass against the narrow
fix that asks the staleness question on both paths (keeping the
``age is None`` branch tolerated on entry points, because
``_fsm_id is not None`` implies the OK branch ran and the OK branch stamps).
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

from strands_robots.drivers.g1 import (
    _FSM_STALE_AFTER_S,
    G1Driver,
)

# The healthy fixture values below match the exact ones in
# test_g1_fsm_refresh_is_off_the_control_loop_thread.py so a change to that
# fixture is visible here rather than hidden behind a duplicated shape.
_HEALTHY_MODE_MACHINE = 9
_HEALTHY_FSM_ID = 501  # a HANDSHAKE_FSMS member


class _RecordingPublisher:
    """A ``_pubs`` stand-in whose ``.publish`` reports success (``None``)."""

    def __init__(self) -> None:
        self.calls: list = []

    def publish(self, topic: str, msg_type: Any, cmd: Any) -> str | None:
        self.calls.append((topic, msg_type, cmd))
        return None

    @property
    def count(self) -> int:
        return len(self.calls)


class _RecordingMotionSwitcherClient:
    """Answers ``CheckMode()`` once with a healthy reading then raises forever.

    The pattern the pin needs: one good admission read stamps ``_fsm_read_at``,
    then a run of transient failures keeps ``_fsm_id`` intact but ages it out.
    Every entry point should refuse on the second-and-later calls the moment
    the age crosses the bound.
    """

    def __init__(self) -> None:
        self._calls = 0

    def CheckMode(self) -> tuple[int, dict[str, Any]]:  # noqa: N802
        self._calls += 1
        if self._calls == 1:
            return (0, {"name": "ai", "form": _HEALTHY_FSM_ID})
        # ``status != 0`` is the branch the decoder turns into a refusal, and
        # the branch ``_refresh_fsm_id`` deliberately keeps ``_fsm_id`` on.
        return (3103, {})


def _install_sdk_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ``unitree_sdk2py`` for the publish path.

    Kept byte-identical in shape to the stubs in the sibling test file so a
    module-load hygiene change is visible in both.
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
        def Crc(self, _cmd: Any) -> int:  # noqa: N802
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


def _healthy_driver(
    client: Any,
    publisher: _RecordingPublisher | None = None,
) -> G1Driver:
    """A driver whose battery + lowstate are healthy and whose FSM was
    admitted exactly once."""
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
    driver._on_bms(types.SimpleNamespace(soc=92.0, charge=0, current=0.0, cycle=0))
    return driver


def _age_the_cache(driver: G1Driver, seconds: float) -> None:
    """Move ``_fsm_read_at`` ``seconds`` into the past without wall time
    tricks; the gate reads ``time.monotonic() - _fsm_read_at``."""
    read_at = driver._fsm_read_at
    assert read_at is not None, "test setup: first refresh should have stamped"
    driver._fsm_read_at = read_at - seconds


class TestStalenessBoundIsAskedOnBothPaths:
    """Pin the peer bot's finding: the bound belongs on entry-point paths too."""

    def test_send_action_refuses_when_the_cached_fsm_is_over_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Same driver as the loop path sees.  When the age exceeds
        ``_FSM_STALE_AFTER_S``, ``send_action`` must refuse and not publish."""
        _install_sdk_stubs(monkeypatch)
        client = _RecordingMotionSwitcherClient()
        pub = _RecordingPublisher()
        driver = _healthy_driver(client, pub)

        # Prime: one good CheckMode(), stamps _fsm_read_at.
        envelope = driver._check_motion_gates("arm", refresh=True)
        assert envelope is None, f"setup: healthy admission should not refuse; got {envelope}"
        assert driver._fsm_id == _HEALTHY_FSM_ID
        assert driver._fsm_read_at is not None
        assert pub.count == 0

        # Age past the bound; every subsequent CheckMode() raises the refused
        # reading so _fsm_read_at is never renewed.
        _age_the_cache(driver, _FSM_STALE_AFTER_S + 0.5)

        action = {"left_shoulder_pitch": 0.1}
        result = driver.send_action(action)

        # THE PIN: must refuse, no frames on the wire.
        assert result.get("status") == "error", (
            "send_action must refuse when the cached FSM is over the staleness bound; "
            f"got status={result.get('status')} with {pub.count} frames published"
        )
        assert pub.count == 0, (
            f"send_action published {pub.count} frame(s) on a stale FSM; "
            "the staleness bound must be asked on entry points too"
        )
        # Reason names the bound (mirroring the loop path's phrasing).
        reason_text = " ".join(c.get("text", "") for c in result.get("content", []))
        assert "staleness bound" in reason_text or "last confirmed" in reason_text, (
            f"refusal must name the staleness bound; got: {reason_text}"
        )

    def test_send_action_still_succeeds_inside_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The narrow fix must not disturb the healthy call site."""
        _install_sdk_stubs(monkeypatch)
        client = _RecordingMotionSwitcherClient()
        pub = _RecordingPublisher()
        driver = _healthy_driver(client, pub)

        envelope = driver._check_motion_gates("arm", refresh=True)
        assert envelope is None
        # Well inside the 1.0s bound.
        _age_the_cache(driver, _FSM_STALE_AFTER_S * 0.3)

        action = {"left_shoulder_pitch": 0.1}
        result = driver.send_action(action)

        assert result.get("status") == "success", f"send_action inside the bound must succeed; got {result}"
        assert pub.count == 1, f"one frame expected, got {pub.count}"

    def test_send_action_publishes_forever_on_a_stale_cache_is_the_regression(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Explicit unboundedness pin: measure at multiples of the bound.

        Before the fix, all four ages publish.  After the fix, only the age
        inside the bound publishes.  This is the direct pin for
        `unbounded, not merely late` from the peer-bot review.
        """
        _install_sdk_stubs(monkeypatch)
        client = _RecordingMotionSwitcherClient()
        pub = _RecordingPublisher()
        driver = _healthy_driver(client, pub)

        envelope = driver._check_motion_gates("arm", refresh=True)
        assert envelope is None
        base_read_at = driver._fsm_read_at
        # The gate admitted, so the OK branch of ``_refresh_fsm_id`` ran, and
        # that is the one branch that both writes ``_fsm_id`` and stamps
        # ``_fsm_read_at``.  Narrowing on that invariant is the same claim the
        # fix under test rests on, so asserting it here is a pin rather than a
        # typing convenience: a refresh that stopped stamping fails this line.
        assert base_read_at is not None

        action = {"left_shoulder_pitch": 0.1}

        # Age inside the bound: 0.3 * bound.
        driver._fsm_read_at = base_read_at - (_FSM_STALE_AFTER_S * 0.3)
        baseline = pub.count
        result = driver.send_action(action)
        assert result.get("status") == "success"
        assert pub.count == baseline + 1, "inside-bound: exactly one frame"

        # Progressively larger ages, all past the bound.  Every one must
        # refuse, publishing zero frames.
        for multiple in (1.5, 3.0, 6.0):
            driver._fsm_read_at = base_read_at - (_FSM_STALE_AFTER_S * multiple)
            frames_before = pub.count
            result = driver.send_action(action)
            frames_after = pub.count
            published = frames_after - frames_before

            assert result.get("status") == "error", (
                f"age={multiple}x the bound: send_action must refuse, "
                f"got status={result.get('status')} and {published} new frames"
            )
            assert published == 0, (
                f"age={multiple}x the bound: 0 frames expected, got {published}. "
                "send_action publishes on an arbitrarily stale FSM without this fix."
            )

    def test_start_task_refuses_over_the_bound(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``start_task`` runs the same admission gate ``send_action`` does;
        the fix must apply to it too, or start_task admission on a stale FSM
        opens a rollout that keeps writing until the per-step re-gate stops it."""
        _install_sdk_stubs(monkeypatch)
        client = _RecordingMotionSwitcherClient()
        pub = _RecordingPublisher()
        driver = _healthy_driver(client, pub)

        # Prime the cache, then age.
        envelope = driver._check_motion_gates("arm", refresh=True)
        assert envelope is None
        _age_the_cache(driver, _FSM_STALE_AFTER_S + 0.5)

        # Direct gate call at scope="motion" (what start_task uses).  This
        # bypasses the actual start_task orchestration but pins the gate
        # semantics start_task depends on.
        envelope = driver._check_motion_gates("motion", refresh=True)
        assert envelope is not None, (
            "the admission gate must refuse a stale FSM on scope='motion' just "
            "as it does on scope='loco' for the loop's per-step re-gate"
        )
        reason_text = " ".join(c.get("text", "") for c in envelope.get("content", []))
        assert "staleness bound" in reason_text or "last confirmed" in reason_text, (
            f"gate refusal must name the staleness bound; got: {reason_text}"
        )

    def test_the_bound_tolerates_age_is_none_on_entry_points_which_production_cannot_reach(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The narrow-fix invariant on entry-point paths: ``_fsm_id is not
        None`` implies the OK branch of ``_refresh_fsm_id`` stamped
        ``_fsm_read_at``.  Fixtures that assign ``_fsm_id`` directly break
        that invariant, and the entry-point path must tolerate the
        production-unreachable case rather than fail closed on it - the
        alternative would ripple failures into ~40 unrelated cells that
        assign ``_fsm_id`` without going through admission.

        The loop path (``refresh=False``) keeps refusing on ``age is None``
        - it is the backstop for a refresher that never runs.
        """
        _install_sdk_stubs(monkeypatch)
        client = _RecordingMotionSwitcherClient()
        pub = _RecordingPublisher()
        driver = _healthy_driver(client, pub)

        # Force the production-unreachable state: _fsm_id set, _fsm_read_at None.
        # Do this after the healthy setup so the mode_machine gate does not
        # trip first.
        driver._fsm_id = _HEALTHY_FSM_ID
        driver._fsm_read_at = None

        # The entry-point gate must NOT refuse on this fixture state.
        # We drive the gate directly with ``refresh=False`` disabled by
        # bypassing ``_refresh_fsm_id`` via a factory that keeps _fsm_id but
        # would re-stamp on the next call.  The simplest reproduction is to
        # patch out the refresh so the fixture state survives one call.
        original_refresh = driver._refresh_fsm_id
        driver._refresh_fsm_id = lambda: None  # type: ignore[method-assign]
        try:
            envelope = driver._check_motion_gates("arm", refresh=True)
        finally:
            driver._refresh_fsm_id = original_refresh  # type: ignore[method-assign]

        assert envelope is None, f"the bound must tolerate age is None on the entry-point path; got refusal {envelope}"

        # And the loop path (refresh=False) must still refuse on age is None
        # - it is the backstop for a refresher that never runs.
        loop_envelope = driver._check_motion_gates("arm", refresh=False)
        assert loop_envelope is not None, (
            "the loop path must refuse on age is None (its own backstop); the entry-point tolerance is orthogonal"
        )
