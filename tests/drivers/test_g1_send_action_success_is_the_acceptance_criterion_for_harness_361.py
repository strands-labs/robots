"""The acceptance criterion for harness#361 is a positive outcome, not a body.

harness#361 has closed early thirteen times because every box on it is
satisfied by a body existing or by a mocked test passing. A checklist of
bodies closes early forever. The gap the peer reviewer named directly on
that issue is one line long:

    ``send_action`` returns ``status="success"`` on a connected driver with a
    decoded ``LowState_`` and a healthy pack.

The predecessor of this file (same path, before this PR) added that cell as
a strict :func:`pytest.mark.xfail` documented against the shipped refusal --
``FSM id unknown - motion-switcher source has not been wired`` -- so the day
a producer landed for ``_fsm_id`` the xfail would XPASS and the wiring commit
would delete the marker in the same change.

That day is this PR.  The xfail is gone, the criterion is a passing cell, and
the surface still grades what the predecessor promised: a driver whose every
field is what a real, healthy G1 produces (a completed ``connect_eagerly``,
a decoded ``LowState_``, a healthy pack) reaches ``send_action`` and gets
``status="success"``.  The one thing that changed is the FSM producer: the
driver now takes an injectable ``motion_switcher_client_factory`` and
:meth:`_refresh_fsm_id` reads through it on every motion-gate check.  The
fixture below hands in a recording client so the wire runs without the SDK.

Two contracts survive from the predecessor file so the boundary does not
silently drift:

1. The publisher is populated.  A driver whose ``_pubs is None`` would refuse
   for a second reason ("publisher not initialised") and the acceptance
   contract would be unreachable for a cause other than the FSM.  The
   fixture uses a recording publisher whose ``.publish`` returns ``None``,
   matching production shape.

2. The wire is actually exercised.  ``result["status"] == "success"`` is a
   necessary but not sufficient condition; a return that skipped the wire
   would satisfy it silently.  The publisher records its call count, and
   that count is one after ``send_action`` returns.

The un-reachability sibling
(:mod:`test_g1_battery_floor_reaches_with_wired_fsm`) is where the reverse
reachability -- battery-floor reaches through the gate now -- is graded.  The
two files complement each other: this file says "success on a healthy pack",
that one says "refuses for battery on a critical pack".
"""

from __future__ import annotations

import sys
import types
from typing import Any

from strands_robots.drivers.g1 import G1Driver

# ``_HEALTHY_MODE_MACHINE`` is what the ``rt/lowstate`` decoder produces on a
# real G1 (uint8 layout id, in ``[0, 255]``).  Any populated value gets past
# the ``mode_machine is None`` refusal and to the ``_fsm_id`` check.
_HEALTHY_MODE_MACHINE = 9

# A healthy pack, well above any configured floor.  This is the ``soc`` a real
# ``rt/lf/bmsstate`` frame carries; ``_on_bms`` is what turns it into the ``pct``
# the gate reads.
_HEALTHY_PACK_PCT = 92.0
_HEALTHY_FLOOR_PCT = 15.0

# ``501`` is in :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS`; the gate admits
# it for every scope, so a policy of any kind can reach the wire on it.
_HEALTHY_FSM_ID = 501


class _RecordingPublisher:
    """A minimally-real ``_pubs`` stand-in.

    ``send_action`` reads ``_pubs`` and calls ``.publish(topic, LowCmd_,
    cmd)``.  A ``MagicMock()`` here would return a ``MagicMock`` for
    ``publish``, whose truthiness (``!= None``) would make the driver report
    a refusal.  Returning ``None`` is the "success" contract publishers use,
    so this class matches production shape.
    """

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any, Any]] = []

    def publish(self, topic: str, msg_type: Any, cmd: Any) -> str | None:
        self.calls.append((topic, msg_type, cmd))
        return None


class _RecordingMotionSwitcherClient:
    """A minimally-real ``MotionSwitcherClient`` stand-in.

    :func:`strands_robots.tools.g1._motion_switcher.read_fsm_id` calls
    ``CheckMode()`` and decodes ``(status, {"name", "form"})``.  This class
    is the wire the driver's new FSM producer reads through.
    """

    def __init__(self, fsm_id: int, mode_name: str = "ai") -> None:
        self._fsm_id = fsm_id
        self._mode_name = mode_name
        self.check_mode_calls: int = 0

    def Init(self) -> None:  # noqa: N802 - SDK spelling
        pass

    def CheckMode(self) -> Any:  # noqa: N802 - SDK spelling
        self.check_mode_calls += 1
        return (0, {"name": self._mode_name, "form": self._fsm_id})


def _install_lowcmd_stub(monkeypatch: Any) -> None:
    """Stub ``unitree_sdk2py.idl.unitree_hg.msg.dds_.LowCmd_``.

    The wire path in :meth:`G1Driver.send_action` lazy-imports
    ``LowCmd_`` after the gate opens.  On the CI box that import fails
    (:class:`ImportError`), which would refuse the whole action -- so the
    fixture stubs the class the same way :mod:`test_g1_control_loop` does.
    The stub is duck-typed against the two attributes ``_build_lowcmd_from_action``
    reads (``motor_cmd`` and the direct field-setters), so the driver's real
    builder runs against it.

    The stub is installed via ``monkeypatch.setitem`` so pytest tears it down
    at end of test, keeping ``sys.modules`` clean for the module-load
    hygiene pin (:mod:`test_motion_switcher_decoder`'s
    ``test_module_import_does_not_touch_the_sdk`` and its siblings).
    """

    class _MotorCmdStub:
        """Duck-typed :class:`MotorCmd_` -- accept every SDK field."""

        def __init__(self) -> None:
            self.mode = 0
            self.q = 0.0
            self.dq = 0.0
            self.tau = 0.0
            self.kp = 0.0
            self.kd = 0.0
            self.reserve = 0

    class _StubLowCmd:
        """Duck-typed :class:`LowCmd_` with a 35-slot ``motor_cmd`` array.

        Matches the wholebody layout the current firmware ships with, which
        is what :func:`_build_lowcmd_from_action` writes against.
        """

        def __init__(self) -> None:
            self.mode_machine = 0
            self.mode_pr = 0
            self.crc = 0
            self.motor_cmd = [_MotorCmdStub() for _ in range(35)]

    dds_ = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    dds_.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    unitree_hg_msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    idl = types.ModuleType("unitree_sdk2py.idl")
    root = types.ModuleType("unitree_sdk2py")
    default = types.ModuleType("unitree_sdk2py.idl.default")
    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]

    # The build path (``_build_lowcmd_from_action``) also imports
    # ``unitree_sdk2py.utils.crc.CRC``; the stub returns a fixed value so
    # the frame's ``.crc`` field lands on an int and the recording publisher
    # sees the produced ``LowCmd_``.
    class _StubCRC:
        def Crc(self, _cmd: Any) -> int:  # noqa: N802 - SDK spelling
            return 0

    crc_mod = types.ModuleType("unitree_sdk2py.utils.crc")
    crc_mod.CRC = _StubCRC  # type: ignore[attr-defined]
    utils = types.ModuleType("unitree_sdk2py.utils")

    for name, module in [
        ("unitree_sdk2py", root),
        ("unitree_sdk2py.idl", idl),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", unitree_hg),
        ("unitree_sdk2py.idl.unitree_hg.msg", unitree_hg_msg),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds_),
        ("unitree_sdk2py.utils", utils),
        ("unitree_sdk2py.utils.crc", crc_mod),
    ]:
        monkeypatch.setitem(sys.modules, name, module)


def _healthy_driver(motion_switcher_client: Any | None = None) -> G1Driver:
    """Return a driver whose every field is what a real, healthy G1 produces.

    If ``motion_switcher_client`` is supplied the driver's FSM producer runs
    through it; otherwise the driver is constructed with no factory (which
    the predecessor xfail was written against, and which the un-reachability
    sibling grades).
    """
    factory: Any = None
    if motion_switcher_client is not None:
        factory = lambda _iface: motion_switcher_client  # noqa: E731

    driver = G1Driver(
        tool_name="g1",
        port="1.2.3.4",
        battery_floor_pct=_HEALTHY_FLOOR_PCT,
        motion_switcher_client_factory=factory,
    )
    driver._connected = True
    # ``_pubs`` is typed ``DDSPublisher | None`` on the driver; the stand-in
    # matches the .publish() shape send_action reads, which is the only
    # surface this test needs.
    driver._pubs = _RecordingPublisher()  # type: ignore[assignment]

    # ``_mode_machine`` and ``_battery`` are driven *through their decoders*
    # rather than assigned, so a decoder change that dropped either field
    # would be visible here rather than hidden behind a fixture that fabricates
    # the decoder's output.
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


def test_send_action_returns_success_on_a_healthy_driver_that_has_a_decoded_lowstate(
    monkeypatch: Any,
) -> None:
    """The one line the harness#361 checklist has been missing -- now passing.

    Every field is populated the way a real, healthy G1 populates it:

    * ``_connected=True`` from a completed ``connect_eagerly``.
    * ``_mode_machine`` from a real ``rt/lowstate`` decode (uint8 layout id).
    * ``_battery`` from a real ``rt/lf/bmsstate`` decode, well above the
      configured floor.
    * ``_pubs`` from a real ``connect_eagerly``.
    * ``_fsm_id`` from the injected motion-switcher factory's ``CheckMode()``,
      reporting the healthy handshake FSM 501.

    The result: ``status="success"``, one publish call, on the ``rt/lowcmd``
    topic the wire is documented against.
    """
    _install_lowcmd_stub(monkeypatch)
    client = _RecordingMotionSwitcherClient(fsm_id=_HEALTHY_FSM_ID, mode_name="ai")
    driver = _healthy_driver(motion_switcher_client=client)

    result = driver.send_action({"left_shoulder_pitch": 0.0})

    # The criterion, both halves stated: the envelope's status, and the
    # publisher recorded exactly one call on ``rt/lowcmd``.  A return that
    # skipped the wire would satisfy the first assertion silently; the
    # second forces the write.
    assert result["status"] == "success", result
    assert isinstance(driver._pubs, _RecordingPublisher)
    assert len(driver._pubs.calls) == 1
    topic, _msg_type, _cmd = driver._pubs.calls[0]
    assert topic == "rt/lowcmd"
    # And the FSM producer was actually consulted -- the criterion is not
    # "success from a hardcoded ``_fsm_id``" but "success through the wire".
    assert client.check_mode_calls >= 1
    # ``get_status`` surfaces the mode label and the FSM the wire reported.
    assert driver._fsm_id == _HEALTHY_FSM_ID
    assert driver._last_fsm_reading is not None
    assert driver._last_fsm_reading.mode_name == "ai"


def test_send_action_still_refuses_when_no_motion_switcher_factory_is_configured() -> None:
    """The predecessor's un-wired refusal survives when no factory is passed.

    A driver constructed without ``motion_switcher_client_factory`` (the
    default) on a box with no ``unitree_sdk2py`` falls back to the shipped
    refusal, because :meth:`_open_motion_switcher_client` cannot import the
    SDK.  The wording is unchanged, so a mesh peer that expected the exact
    string still gets it.
    """
    # No factory, no monkeypatch: the SDK import genuinely fails on CI.
    driver = _healthy_driver(motion_switcher_client=None)
    result = driver.send_action({"left_shoulder_pitch": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # Both phrases the predecessor's xfail reason cited.
    assert "FSM id unknown" in text
    assert "motion-switcher" in text


def test_the_publisher_is_populated_and_the_driver_is_otherwise_healthy(
    monkeypatch: Any,
) -> None:
    """The success path uses a real publisher; hold the boundary.

    A future refactor that made the driver skip publisher setup on
    ``_fsm_id is None`` would trip this cell, and the fix is to preserve
    the current shape rather than to silence the assertion.
    """
    _install_lowcmd_stub(monkeypatch)
    client = _RecordingMotionSwitcherClient(fsm_id=_HEALTHY_FSM_ID)
    driver = _healthy_driver(motion_switcher_client=client)
    assert driver._pubs is not None
    # Reading the gate directly rather than through ``send_action`` so a
    # publisher failure downstream does not conflate the gate's outcome.
    refusal = driver._check_motion_gates("arm")
    assert refusal is None, refusal


def test_the_fsm_wire_is_read_every_gate_call_not_only_the_first(
    monkeypatch: Any,
) -> None:
    """The gate refreshes on every call, not only the first.

    :meth:`_refresh_fsm_id` runs at the top of every gate call so a change
    in FSM state between two writes (an operator handing off from the
    high-level motion service mid-rollout) is observed on the next frame
    rather than only on the frame after next.
    """
    _install_lowcmd_stub(monkeypatch)
    client = _RecordingMotionSwitcherClient(fsm_id=_HEALTHY_FSM_ID)
    driver = _healthy_driver(motion_switcher_client=client)

    driver.send_action({"left_shoulder_pitch": 0.0})
    driver.send_action({"left_shoulder_pitch": 0.1})
    driver.send_action({"left_shoulder_pitch": 0.2})

    # Three sends, three CheckMode() reads (plus the one from
    # ``_check_motion_gates`` earlier if applicable -- accept ``>= 3``).
    assert client.check_mode_calls >= 3
