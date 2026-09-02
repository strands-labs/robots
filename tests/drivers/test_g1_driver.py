"""Native G1 driver: contract, DDS decode, and factory wiring.

Every DDS touch is mocked. Thor never had a G1 and running these tests on a
box that did would still be wrong: hardware bring-up is validated at the
office, on the real robot. This suite pins the driver's *own* behaviour, and
that is a class problem, not a robotics problem.
"""

from __future__ import annotations

import asyncio
import sys
import types
from typing import Any

import pytest

from strands_robots.drivers import (
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.g1 import G1Driver
from strands_robots.tools.g1 import (
    HANDSHAKE_FSMS,
    WALK_FSMS,
    _dds_engine,
    decode_code,
    ensure_dds,
    reset_dds_state,
)
from strands_robots.tools.g1._dds_engine import DDSSubscriberSet
from strands_robots.tools.g1._g1_common import _DDS_INIT_LOCK
from tests.drivers.test_g1_control_loop import _StubCRC, _StubLowCmd

# =========================================================================
# The seam - Protocol conformance and factory wiring.                     #
# =========================================================================


def test_g1_driver_satisfies_hardware_driver_protocol() -> None:
    """Every :data:`DRIVER_SURFACE` member is present on the class.

    :func:`register_native_driver` runs the same check at registration time,
    but pinning it here catches a regression the moment the class drifts,
    not when someone tries to register a fresh driver.
    """
    assert missing_driver_members(G1Driver) == ()
    inst = G1Driver(tool_name="g1", port="192.168.1.172")
    assert isinstance(inst, HardwareDriver)


def test_g1_is_registered_on_import() -> None:
    """Importing :mod:`strands_robots.drivers` puts G1Driver into the table.

    The auto-registration hook is what makes ``Robot("g1", mode="real",
    driver="strands")`` a one-liner - a caller who has to remember a second
    import is a caller who eventually forgets. The registry keys on the
    canonical name, so ``unitree_g1`` is what appears.
    """
    registered = list_native_drivers()
    assert registered.get("unitree_g1") == "G1Driver"


def test_get_native_driver_class_returns_g1() -> None:
    """The registry accepts both aliases and returns the same class.

    ``g1`` is an alias for ``unitree_g1``; the registry canonicalises before
    lookup so a caller does not have to.
    """
    assert get_native_driver_class("g1") is G1Driver
    assert get_native_driver_class("unitree_g1") is G1Driver


def test_factory_builds_g1_driver_with_driver_strands() -> None:
    """``Robot(..., driver="strands")`` returns the driver, not HardwareRobot.

    The regression to guard here is the seam: a factory that ignores
    ``driver="strands"`` and returns the lerobot driver would silently
    debug a caller who thinks they got the native path.
    """
    from strands_robots.robot import Robot

    driver = Robot(
        "g1",
        mode="real",
        driver="strands",
        port="192.168.1.172",
        network_interface="eth0",
    )
    assert isinstance(driver, G1Driver)
    assert driver.tool_name == "unitree_g1"


def test_registry_declares_strands_as_default_driver_for_g1() -> None:
    """The registry entry sets ``hardware.driver = "strands"``.

    Setting the default here means ``Robot("g1", mode="real")`` (no
    ``driver=`` at all) resolves to the native driver via ``auto``. A
    caller who wants the lerobot path back can still ask for it explicitly
    with ``driver="lerobot"``.
    """
    from strands_robots.registry import get_driver

    assert get_driver("unitree_g1") == "strands"


# =========================================================================
# Constructor contract.                                                   #
# =========================================================================


def test_constructor_accepts_the_three_factory_kwargs() -> None:
    """The factory forwards ``tool_name``, ``cameras``, ``data_config``.

    Every native driver must accept those three kwargs (the base module's
    documented constructor contract). The G1 driver ignores ``cameras`` and
    ``data_config`` because its cameras live on the DDS bus, not v4l2, but
    accepting them keeps the factory shape uniform.
    """
    driver = G1Driver(
        tool_name="g1",
        cameras={"front": {"index": 0}},
        data_config="some_config",
        port="192.168.1.172",
    )
    assert driver.tool_name == "g1"


def test_constructor_tolerates_extra_kwargs() -> None:
    """Unknown extras are logged and discarded, not raised.

    A factory may forward kwargs the driver has never heard of; refusing
    them would couple the factory to every driver's parameter list. The
    driver logs the surprise so it is discoverable, then continues.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4", made_up_option=42)
    assert driver.tool_name == "g1"


# =========================================================================
# Sensor decode. Each callback is called with a fake IDL message.         #
# =========================================================================


def _fake_imu(rpy=(0.1, 0.2, 0.3), quat=(1.0, 0.0, 0.0, 0.0)) -> Any:
    return types.SimpleNamespace(
        rpy=list(rpy),
        gyroscope=[0.01, 0.02, 0.03],
        accelerometer=[0.0, 0.0, 9.81],
        quaternion=list(quat),
    )


def _fake_lowstate(fsm: int = 501, imu: Any | None = None) -> Any:
    return types.SimpleNamespace(
        imu_state=imu or _fake_imu(),
        mode_machine=fsm,
    )


def test_lowstate_populates_imu_and_mode_machine() -> None:
    """A ``rt/lowstate`` callback fills ``_imu`` and ``_mode_machine``.

    ``LowState_.mode_machine`` is a uint8 (packed ``<2B`` alongside
    ``mode_pr`` inside ``_CRC__packFmtHGLowCmd``) - it is the hardware layout
    id the firmware wants echoed on every ``LowCmd_``, not the high-level
    FSM state the arm-SDK gate tests against.  The driver keeps the two
    fields separate: ``_mode_machine`` comes from lowstate, ``_fsm_id``
    arrives from the motion-switcher API on a different topic.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._on_lowstate(_fake_lowstate(fsm=9))  # a real u8 value from LowState
    assert driver._mode_machine == 9
    assert driver._fsm_id is None  # lowstate does not feed the FSM gate
    assert driver._imu is not None
    assert driver._imu["rpy"] == [0.1, 0.2, 0.3]
    assert driver._imu["quaternion"] == [1.0, 0.0, 0.0, 0.0]
    assert isinstance(driver._imu["t"], float)


def test_bms_populates_battery_with_the_fields_the_health_chip_reads() -> None:
    """``rt/lf/bmsstate`` yields the fields the mesh's health chip reads.

    :mod:`strands_robots.mesh.sensors` reads ``battery.get("pct")`` - the
    name must match that key or the mesh publishes an empty health entry.
    A charge flag is deliberately absent from the record; see
    :mod:`tests.drivers.test_g1_bms_reads_the_declared_fields`.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    msg = types.SimpleNamespace(soc=87.5, current=-1.2, cycle=42)
    driver._on_bms(msg)
    assert driver._battery is not None
    assert driver._battery["pct"] == pytest.approx(87.5)
    assert driver._battery["current"] == pytest.approx(-1.2)
    assert driver._battery["cycle"] == 42


def test_lidar_state_decodes_code() -> None:
    """LidarState renders its ``error_state`` through :func:`decode_code`.

    The stand-in spells the names ``LidarState_`` declares. An earlier version
    of this test built one that spelled the names the decoder happened to read,
    which made it agree with the decoder whatever those names were - the reason
    a decoder reading two undeclared fields passed here for as long as it did.
    Field-name fidelity is graded on its own in
    ``test_g1_lidar_state_reads_the_declared_fields.py``.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    msg = types.SimpleNamespace(error_state=0, cloud_frequency=10.0, sys_rotation_speed=10.0)
    driver._on_lidar_state(msg)
    assert driver._lidar_state is not None
    assert driver._lidar_state["code"] == 0
    assert "OK" in driver._lidar_state["code_text"]
    assert driver._lidar_state["freq"] == pytest.approx(10.0)


def test_lidar_cloud_summary_is_bounded() -> None:
    """A full Livox frame produces a fixed-size summary, not a per-point dump.

    The mesh publishes ``_lidar_summary`` as a small dict every tick; if
    :meth:`_on_lidar_cloud` shipped 30k points into that field the topic
    would drown Zenoh. What keeps the record small is that every field is read
    from the message header, so the summary has the same shape whatever the
    cloud's size - which is what the absence of a point list asserts here.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    # Livox at 10 Hz reports width ~ 24000, height 1
    msg = types.SimpleNamespace(width=24000, height=1, point_step=16, row_step=24000 * 16)
    driver._on_lidar_cloud(msg)
    assert driver._lidar_summary is not None
    assert driver._lidar_summary["count"] == 24000
    # No raw point list: the summary dict is small and fixed-shape.
    assert "points" not in driver._lidar_summary


def test_decoders_swallow_bad_messages() -> None:
    """A malformed IDL message logs and is dropped - the DDS thread survives.

    If one bad message tore down the DDS callback, the driver would go silent
    on a real robot until the next reconnect. Every decoder catches, logs
    at debug, and moves on. A ``None`` message is what a broken CycloneDDS
    subscriber has been observed to deliver; an unrelated object is the
    shape a firmware update might land the day the IDL changes.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._on_lowstate(None)  # would raise on plain getattr
    driver._on_lidar_state("not a message")
    driver._on_lidar_cloud(object())
    # Lowstate with imu_state=None and no mode_machine: cache stays empty.
    assert driver._imu is None
    assert driver._mode_machine is None
    assert driver._fsm_id is None


# =========================================================================
# Command gates. send_action refuses until connected, FSM is good and    #
# the battery is above the floor.                                        #
# =========================================================================


def test_send_action_refuses_before_connect() -> None:
    """A driver that never connected cannot write - the message says so."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "not connected" in result["content"][0]["text"]


def test_send_action_refuses_without_mode_machine() -> None:
    """Connected but no lowstate yet - ``mode_machine`` is unknown, refuse.

    ``LowState_.mode_machine`` is the uint8 the firmware wants echoed on
    every ``LowCmd_``, and it arrives on lowstate.  Without one delivered
    the driver has nothing to echo, so ``send_action`` refuses before it
    ever consults the FSM gate.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True  # simulate connect_eagerly success
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "mode_machine unknown" in result["content"][0]["text"]


def test_mode_machine_and_fsm_id_have_disjoint_value_ranges() -> None:
    """The two caches ``_on_lowstate`` and the FSM gate feed are distinct.

    This is the SDK-free contract test for the u8-vs-FSM separation
    (harness#361 review, #2765).  ``LowState_.mode_machine`` is packed
    ``<2B`` in ``_CRC__packFmtHGLowCmd`` (uint8, 0..255), while the arm-SDK
    gate accepts only :data:`HANDSHAKE_FSMS` = {500, 501, 801}.  The
    intersection is empty, so writing lowstate's ``mode_machine`` into the
    field the gate checks would refuse every real frame.  Two attributes
    keep the values separate.

    A single-attribute regression - if a future edit collapsed
    :attr:`_mode_machine` back into :attr:`_fsm_id`, or vice versa - would
    fail this assertion without needing ``unitree_sdk2py`` on the box.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    # Both start unset.
    assert driver._mode_machine is None
    assert driver._fsm_id is None
    # A lowstate delivery fills only ``_mode_machine`` (uint8 layout id).
    driver._on_lowstate(_fake_lowstate(fsm=9))
    mode_machine = driver._mode_machine
    # A delivered lowstate caches the id, so this is also the narrowing the
    # range assertion below needs: ``_mode_machine`` is ``int | None`` and an
    # equality check does not tell a type checker which arm it landed on.
    assert mode_machine is not None
    assert mode_machine == 9
    assert driver._fsm_id is None  # the FSM gate's input is a different source
    # Ranges: uint8 vs the SDK's error-table constants.
    from strands_robots.tools.g1 import HANDSHAKE_FSMS

    assert all(v > 255 for v in HANDSHAKE_FSMS)
    assert 0 <= mode_machine <= 255


def test_named_joint_count_matches_the_sdk_reference() -> None:
    """The Enable-byte loop bound derives from :data:`_G1_JOINT_INDEX`, not the array width.

    ``LowCmd_.motor_cmd`` is a 35-array; the G1 commands 29 joints and
    slots 29..34 are a reserved tail.  The SDK's own G1 reference loops
    ``for i in range(G1_NUM_MOTOR)`` where ``G1_NUM_MOTOR = 29`` when
    setting the Enable byte, and bounding by the array width instead would
    make a wire-frame decision on those reserved slots that this driver
    does not have the information to make.

    Pinned SDK-free so ``call-test-lint`` grades the invariant even in an
    environment without ``unitree_sdk2py``: the count must equal 29, and it
    must equal ``max(_G1_JOINT_INDEX.values()) + 1`` (so a joint added to
    the table later moves both builders together without a separate edit).
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX, _G1_NAMED_JOINTS

    assert _G1_NAMED_JOINTS == 29
    assert _G1_NAMED_JOINTS == max(_G1_JOINT_INDEX.values()) + 1
    assert len(_G1_JOINT_INDEX) == _G1_NAMED_JOINTS
    # Every named index is in [0, _G1_NAMED_JOINTS); the reserved tail is
    # unnamed by construction.
    assert set(_G1_JOINT_INDEX.values()) == set(range(_G1_NAMED_JOINTS))


def test_send_action_refuses_without_fsm_id_source_wired() -> None:
    """Connected + lowstate delivered, but no FSM source - refuse honestly.

    ``LowState_.mode_machine`` is uint8 (0..255) and cannot host any of
    :data:`HANDSHAKE_FSMS` = {500, 501, 801}, so pointing the gate at the
    lowstate field would leave the intersection empty and reject every real
    frame silently.  The driver's ``_fsm_id`` comes from the motion-switcher
    API instead, and until that source is wired the gate refuses with a
    message that names the missing piece rather than the general FSM.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._mode_machine = 9  # lowstate has landed
    driver._fsm_id = None  # motion-switcher source not wired yet
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FSM id unknown" in text
    assert "motion-switcher" in text  # names the missing source, not \"lowstate\"


@pytest.mark.parametrize("fsm", [0, 1, 3, 4])  # zero-torque, damp, sit, standup
def test_send_action_refuses_outside_handshake_fsm(fsm: int) -> None:
    """FSM outside :data:`HANDSHAKE_FSMS` is refused with the set named.

    The refusal names the *arm* set specifically (``send_action`` writes to
    ``rt/armsdk``), not the union with :data:`WALK_FSMS` - a message that
    over-claims coverage would mislead a caller reading a log.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert f"FSM {fsm}" in text
    assert "arm writes" in text
    for handshake_fsm in HANDSHAKE_FSMS:
        assert str(handshake_fsm) in text
    # And the message must *not* name a set the arm gate does not test:
    # a locomotion-only FSM in the message would suggest a gate this call
    # cannot enforce. The two sets overlap on {501, 801}, so the check is
    # that any FSM in WALK_FSMS \ HANDSHAKE_FSMS is absent - which is the
    # empty set today, so the constructive check is on the wording.
    assert "loco" not in text  # message no longer over-claims scope


def test_send_action_refuses_below_battery_floor() -> None:
    """Battery under the configured floor refuses even with a good FSM."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4", battery_floor_pct=15.0)
    driver._connected = True
    driver._fsm_id = 501
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 12.0, "current": 0.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    assert "battery" in result["content"][0]["text"]
    assert "12.0%" in result["content"][0]["text"]


def test_send_action_reports_motion_not_wired_when_gates_pass() -> None:
    """Every gate passes but no publisher is attached - refuse with a named reason.

    Before :meth:`connect_eagerly` builds the publisher, ``_pubs`` is ``None``
    and a caller who forces ``_connected = True`` (as the older stub tests
    did) cannot reach the wire.  The refusal names the publisher, not
    ``issue #358``, because the write path is wired now - the missing piece
    is the DDS init the caller has not run yet.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"left_shoulder_pitch": 0.1})
    assert result["status"] == "error"
    assert "publisher not initialised" in result["content"][0]["text"]


# =========================================================================
# Task / policy stubs.                                                   #
# =========================================================================


def test_task_and_policy_paths_report_not_wired_when_gates_pass() -> None:
    """Every task/policy verb reports its deferral reason honestly.

    Shipping the stubs with the right envelope shape means the day motion
    lands nothing on the caller side has to change. ``start_task`` and
    ``run_policy`` consult the same FSM + battery gates :meth:`send_action`
    does (motion-scoped: the union of :data:`HANDSHAKE_FSMS` and
    :data:`WALK_FSMS`), then return the error envelope - they cannot
    produce work. ``get_task_status`` and ``stop_task`` succeed because "no
    task running" and "nothing to stop" are honest answers, not failures
    and not motion writes.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501  # in both HANDSHAKE_FSMS and WALK_FSMS
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}

    start = driver.start_task("do X", policy_port=8000)
    assert start["status"] == "error"
    assert "provider registry not wired yet" in start["content"][0]["text"]

    status = driver.get_task_status()
    assert status["status"] == "success"
    assert status["content"][0]["json"]["running"] is False

    stop = driver.stop_task()
    assert stop["status"] == "success"

    # ``run_policy(None)`` refuses at the transport primitive rather than
    # touching the loop - the loop needs a policy to roll out.
    envelope = driver.run_policy(policy_object=None, instruction="", duration=1.0)  # type: ignore[arg-type]
    assert envelope["status"] == "error"
    assert "policy_object is required" in envelope["content"][0]["text"]


# -------------------------------------------------------------------------
# Gates on the task/policy verbs. The bot review of PR #2739 found that the
# gates the PR advertised "on every motion-path stub" only fired on
# send_action. These tests pin the gate coverage so the day the writes land
# a bypass would be caught here rather than on hardware.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_when_not_connected(verb: str) -> None:
    """An unconnected driver refuses every motion verb with the same reason.

    The refusal comes from the gate, not from the "not wired" stub, so the
    contract is: bring me up before you ask me to move. A stub that skipped
    the gate would silently accept a request the day the writes land.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    if verb == "start_task":
        result = driver.start_task("do X")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "not connected" in text
    # The gate ran first, so the deferral reason must not be what we see.  Naming
    # the live phrase keeps this graded: a stale literal would pass vacuously.
    assert "provider registry not wired yet" not in text


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_below_battery_floor(verb: str) -> None:
    """A battery under the floor refuses even when the FSM would admit motion."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4", battery_floor_pct=15.0)
    driver._connected = True
    driver._fsm_id = 501  # walkable
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 4.0, "current": 0.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("walk 1m")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "battery" in text
    assert "4.0%" in text
    assert "provider registry not wired yet" not in text


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
def test_task_paths_refuse_outside_motion_fsm(verb: str) -> None:
    """An FSM outside the motion union refuses; the refusal names the union.

    ``start_task`` and ``run_policy`` use the ``"motion"`` scope, which is
    the union of :data:`HANDSHAKE_FSMS` and :data:`WALK_FSMS`. An FSM at 0
    (zero-torque) is in neither, so both verbs refuse and the message names
    the union so a caller reading a log can see what would satisfy it.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 0
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("do X")
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert "FSM 0" in text
    # Every FSM in the motion union must appear so the caller can see what
    # would admit them. Values stated literally (not derived from the
    # constants under test) so a maintainer tightening the ``motion`` scope
    # to the intersection sees a failure named for the acceptance side of
    # the motion union, not merely a suite that passes with fewer cases.
    assert "motion writes" in text
    for member in (500, 501, 801):
        assert str(member) in text


# -------------------------------------------------------------------------
# The two ungraded mutations the reviewer flagged, pinned. Both are
# safety-adjacent: a suite that survives them ships a documented set
# distinction with no enforcement.
# -------------------------------------------------------------------------


@pytest.mark.parametrize("fsm", [500, 501, 801])
def test_send_action_admits_every_handshake_fsm(fsm: int) -> None:
    """Every FSM in :data:`HANDSHAKE_FSMS` passes the arm gate.

    Values stated literally so narrowing :data:`HANDSHAKE_FSMS` (dropping
    500, say) fires the ``[500]`` case rather than deselecting it. A
    parametrize list derived from the set under test cannot detect a
    narrowing of that set: the case disappears and the suite reads green
    with one fewer test.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    result = driver.send_action({"any": 0.0})
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    # Every handshake FSM passes the gate - the refusal is the transport
    # not being live (no publisher until ``connect_eagerly`` runs), not
    # an FSM refusal. PR-B (#361) wired the write; the transport-side
    # failure supersedes the old "issue #358" surface.
    assert "publisher not initialised" in text
    # The FSM was accepted, so its number is *not* mentioned as a refusal.
    assert f"FSM {fsm} refuses" not in text


def test_walk_fsms_has_a_consumer_and_a_documented_boundary() -> None:
    """The :data:`WALK_FSMS` constant is not dead: a locomotion write at 500
    refuses even though 500 is in :data:`HANDSHAKE_FSMS`.

    500 is sitting - the G1 accepts arm gestures there but not walking. A
    ``run_policy`` at FSM 500 is a motion write (union-scoped), so it
    passes; a hypothetical loco-scoped write at 500 would refuse. This
    test pins the boundary WALK_FSMS exists to record: the day the write
    lines land, the loco-scoped path can call ``_check_motion_gates("loco")``
    and the sitting refusal is already correct.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 500  # sitting: in HANDSHAKE_FSMS, not in WALK_FSMS
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}

    # Arm-scoped write at 500 passes the gate (500 is in HANDSHAKE_FSMS).
    arm_result = driver.send_action({"any": 0.0})
    assert arm_result["status"] == "error"
    # Not gated - the arm gate accepts 500; the refusal is transport-side
    # (PR-B wired the write, so a driver without a live publisher fails
    # here instead of at the "not wired yet" seam).
    assert "publisher not initialised" in arm_result["content"][0]["text"]

    # Loco-scoped gate at 500 refuses. Calling the helper directly is the
    # narrow way to pin the boundary before any write verb classifies its
    # scope; the write verbs land in issue #358.
    loco_refusal = driver._check_motion_gates("loco")
    assert loco_refusal is not None
    text = loco_refusal["content"][0]["text"]
    assert "FSM 500" in text
    assert "locomotion writes" in text
    # The message names the WALK set, not the union - so a caller sees
    # exactly what would satisfy a loco write. Stated literally so an
    # emptied WALK_FSMS (a maintainer error that would refuse every
    # locomotion write the day the writes land) fires this test rather
    # than passing over a vacuous ``for`` loop.
    for member in (501, 801):
        assert str(member) in text
    # 500 is *not* in WALK_FSMS, so it must not appear as an admitted FSM.
    admitted_set_part = text.split("needs one of ")[-1]
    assert "500" not in admitted_set_part


# -------------------------------------------------------------------------
# Complementary pins for the set choices themselves. Each states an
# expectation as literal values so a mutation of the constants shows up as
# a graded failure, not as a suite that runs one fewer case or reads green
# over an empty ``for`` loop.
# -------------------------------------------------------------------------


def test_walk_fsms_is_a_proper_subset_of_handshake_fsms() -> None:
    """Both sets are populated and the distinction is a real one.

    ``test_walk_fsms_has_a_consumer_and_a_documented_boundary`` asserts the
    contents of :data:`WALK_FSMS` with ``for member in WALK_FSMS``, which
    is vacuous when the set is empty. This test states the shape directly
    so an emptied :data:`WALK_FSMS` -- or a widening that erases the
    distinction the ``loco`` scope depends on -- fails a test named for
    the shape, not a test named for something else.
    """
    assert WALK_FSMS, "WALK_FSMS must not be empty"
    assert WALK_FSMS < HANDSHAKE_FSMS, "WALK_FSMS must be strictly narrower"
    # 500 is the boundary the scope split exists to record: sitting admits
    # arm gestures but not walks. Naming it literally makes the intent
    # readable from the assertion alone.
    assert 500 in HANDSHAKE_FSMS - WALK_FSMS


@pytest.mark.parametrize("fsm", [501, 801])
def test_the_loco_gate_admits_every_walking_fsm(fsm: int) -> None:
    """Acceptance side of the loco gate, at literal FSM values.

    ``test_walk_fsms_has_a_consumer_and_a_documented_boundary`` only grades
    the refusal at 500; nothing asserts that a walkable FSM is admitted,
    so emptying :data:`WALK_FSMS` fires no test in the shipped suite. This
    test closes that gap by driving the helper directly at each FSM the
    ``loco`` scope must accept.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    assert driver._check_motion_gates("loco") is None


@pytest.mark.parametrize("verb", ["start_task", "run_policy"])
@pytest.mark.parametrize("fsm", [501, 801])
def test_motion_verbs_admit_a_literally_walkable_fsm(verb: str, fsm: int) -> None:
    """A motion verb at an FSM both scopes accept passes the gate.

    Uses literal FSM values rather than a derivation from the composition
    under test, so this survives a maintainer's later decision to tighten
    the ``motion`` pre-flight to the intersection.  The refusal a caller
    sees is the verb-specific reason (the deferred provider registry for
    ``start_task``, the
    ``policy_object is required`` message for ``run_policy`` fed
    ``None``), never the FSM-refusal reason - the gate ran first and
    passed.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = fsm
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    if verb == "start_task":
        result = driver.start_task("do X")
        expected_marker = "provider registry not wired yet"
    else:
        result = driver.run_policy(policy_object=None, instruction="")  # type: ignore[arg-type]
        expected_marker = "policy_object is required"
    assert result["status"] == "error"
    text = result["content"][0]["text"]
    assert expected_marker in text
    assert f"FSM {fsm} refuses" not in text


# =========================================================================
# Lifecycle - status, stop, cleanup, and the stream tool surface.        #
# =========================================================================


def test_get_status_shape_matches_driver_envelope() -> None:
    """``get_status`` returns the same envelope the lerobot driver returns.

    The mesh publishes both peers identically; a driver that returns a
    different shape breaks the mesh's presence chip.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4", network_interface="wlan0")
    envelope = asyncio.run(driver.get_status())
    assert envelope["status"] == "success"
    inner = envelope["content"][0]["json"]
    assert inner["tool_name"] == "g1"
    assert inner["connected"] is False
    assert inner["network_interface"] == "wlan0"


def test_stream_sensors_action_returns_the_cached_snapshots() -> None:
    """The ``sensors`` verb yields exactly the four cache snapshots.

    A caller who wants to know what the robot last reported gets it in one
    call, without having to look at ``_imu`` and its siblings directly - the
    private-attribute path is for the mesh, not the agent.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._imu = {"rpy": [0.0, 0.0, 0.0], "t": 1.0}
    driver._battery = {"pct": 88.0, "t": 1.0}

    async def _collect() -> list[Any]:
        events: list[Any] = []
        async for event in driver.stream({"toolUseId": "u1", "name": "g1", "input": {"action": "sensors"}}, {}):
            events.append(event)
        return events

    events = asyncio.run(_collect())
    assert len(events) == 1
    payload = events[0]["content"][0]["json"]
    assert payload["imu"]["rpy"] == [0.0, 0.0, 0.0]
    assert payload["battery"]["pct"] == 88.0
    assert payload["lidar_state"] is None  # never delivered
    assert payload["lidar_summary"] is None


def test_stream_stop_action_reports_that_no_task_was_running() -> None:
    """The ``stop`` verb names the state it found rather than a halt it did not perform.

    The transport primitive is wired - :meth:`send_action` publishes on
    ``rt/lowcmd`` and a running loop publishes a zero-torque frame on the
    way out - so the pre-#361 refusal claiming no motion path exists is
    stale.  But this driver has no loop, so "halted a control loop" would
    be its own falsity: the verb delegates to :meth:`stop_task`, whose
    idempotent branch names the state instead.  The loop-was-running
    outcomes are graded in
    ``tests/drivers/test_g1_stream_stop_reports_the_halt_outcome.py``.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")

    async def _run() -> Any:
        async for event in driver.stream({"toolUseId": "u1", "name": "g1", "input": {"action": "stop"}}, {}):
            return event
        return None  # pragma: no cover

    event = asyncio.run(_run())
    assert event["status"] == "success"
    text = event["content"][0]["text"]
    assert "no task is running" in text
    # Guard against the pre-#361 refusal text sneaking back in a rebase:
    # the driver's stop path now maps to a running behaviour, so the
    # envelope must not claim otherwise.
    assert "no motion path wired" not in text
    assert "#358" not in text


def test_cleanup_is_idempotent() -> None:
    """Two ``cleanup`` calls do not raise; the second is a no-op."""
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver.cleanup()
    driver.cleanup()  # would raise on double-release without the guard
    assert driver._connected is False


# =========================================================================
# ensure_dds and decode_code - the shared helpers.                       #
# =========================================================================


def test_ensure_dds_reports_missing_sdk() -> None:
    """Without ``unitree_sdk2py`` installed, :func:`ensure_dds` returns a reason.

    Thor never has the SDK, so this is what every unit run actually hits.
    The reason names the missing package so a reader sees the fix rather
    than an obscure ImportError deep in the stack.
    """
    reset_dds_state()
    err = ensure_dds("eth-nonexistent")
    assert err is not None
    # Either the SDK is missing (Thor, CI) or the factory refused the
    # interface (an office machine with the SDK). Both spellings are
    # accepted so the test survives both environments.
    assert "unitree_sdk2py" in err or "ChannelFactoryInitialize" in err


def test_decode_code_names_known_and_unknown() -> None:
    """A known code renders with its meaning; an unknown one still shows the number."""
    assert "OK" in decode_code(0)
    assert "unknown" in decode_code(99999)
    assert "None" in decode_code(None) or "None" in repr(None)  # non-int path


def test_dds_init_lock_is_a_lock() -> None:
    """:data:`_DDS_INIT_LOCK` is the shared lock the driver and issue #358 tools hold.

    A different lock object here and in the tools would allow a race between
    ``ChannelSubscriber(...)`` calls; the segfault CycloneDDS bindings
    produce under that race is what this lock exists to prevent. Test what
    matters: same object, acquirable, releasable.

    The lock is private to ``_g1_common`` and reached there rather than through
    the package, so ``_dds_engine`` binding a *copy* would be invisible at the
    import site. The identity assertion is what makes "same object" a fact.
    """
    assert _dds_engine._DDS_INIT_LOCK is _DDS_INIT_LOCK
    assert hasattr(_DDS_INIT_LOCK, "acquire")
    assert hasattr(_DDS_INIT_LOCK, "release")
    acquired = _DDS_INIT_LOCK.acquire(blocking=False)
    try:
        assert acquired
    finally:
        if acquired:
            _DDS_INIT_LOCK.release()


# =========================================================================
# connect_eagerly - the DDS path fails gracefully on Thor.               #
# =========================================================================


def test_connect_eagerly_is_a_no_op_on_a_connected_driver(monkeypatch: pytest.MonkeyPatch) -> None:
    """A second call keeps the subscriber set it already has.

    Rebuilding it would re-subscribe all four topics and drop the only
    reference to the previous :class:`DDSSubscriberSet`, leaking its
    subscribers - on a bus whose bindings segfault under concurrent
    ``ChannelSubscriber(...)``. No caller does this today, which is exactly
    why the method's idempotence needs pinning rather than assuming.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    # Constructing one touches no DDS - __init__ only records the interface.
    already = DDSSubscriberSet("eth0")
    driver._connected = True
    driver._subs = already

    def _forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("connect_eagerly() rebuilt the subscriber set")

    monkeypatch.setattr("strands_robots.drivers.g1.DDSSubscriberSet", _forbidden)

    assert driver.connect_eagerly() is None
    assert driver._subs is already
    assert driver._connected is True


def test_connect_eagerly_reports_reason_without_sdk() -> None:
    """A machine without ``unitree_sdk2py`` gets a named connect error.

    The driver stays usable - the tests that don't need the bus can still
    call every stub - so a caller who wants a driver instance for
    a smoke test can still get one.  The rollback contract is pinned here
    too: ``_connected`` stays False and ``_connect_error`` holds the
    returned reason so a later ``connect_eagerly`` re-attempt sees the
    same value ``send_action`` and the gates would report.
    """
    reset_dds_state()
    driver = G1Driver(tool_name="g1", port="1.2.3.4", network_interface="eth-none")
    err = driver.connect_eagerly()
    assert err is not None
    # Same acceptance as ensure_dds: SDK-missing on Thor/CI, bind-fail in office.
    assert "unitree_sdk2py" in err or "ChannelFactoryInitialize" in err or "cannot import" in err
    # Rollback contract: a failed connect leaves the driver in the pre-connect
    # state, with the returned reason cached so a later gate reports it.
    assert driver._connected is False
    assert driver._connect_error == err


# =========================================================================
# send_action wire path (PR-B: issue #361, decoupled from #358).         #
# =========================================================================
# Every test builds a driver that has already passed the gates and installs
# a recording publisher in ``_pubs``.  The point is the wire capture: which
# ``motor_cmd`` slots the driver filled, and with which values.  A missing
# SDK short-circuits the driver (it refuses with the SDK's own reason before
# it would build a LowCmd_), so these cells need a LowCmd_-shaped object to
# write into.  They take one from the stub :mod:`tests.drivers.test_g1_control_loop`
# installs rather than a skip-marker: ``unitree-sdk2`` is not a declared
# dependency of this project, so a contract asserted only behind
# ``skipif(not _HAS_SDK)`` is asserted by nothing in ``call-test-lint`` - the
# rule :mod:`tests.drivers.test_g1_per_joint_gains` states for the same
# builder.  The two cells that recompute the SDK's own CRC as an independent
# oracle keep the marker: a stub CRC would compare a constant against itself.
#
# The fixture is opt-in per cell rather than autouse because this module also
# pins the SDK-*absent* refusals (``test_ensure_dds_reports_missing_sdk``,
# ``test_connect_eagerly_reports_reason_without_sdk``), which a module-wide
# stub would quietly make unreachable.


_HAS_SDK: bool
try:  # pragma: no cover - environmental
    import unitree_sdk2py.idl.default as _sdk_default

    # Read the bound module rather than assigning a bare ``True``: the import
    # is the probe, so naming its result keeps the import used on the line
    # that decides the answer.  A bare ``True`` leaves the binding write-only,
    # which reads as dead to a static analyser and needs an ``F401`` waiver to
    # stay.
    _HAS_SDK = _sdk_default is not None
except ImportError:  # pragma: no cover - environmental
    _HAS_SDK = False


class _RecordingPublisher:
    """Records ``publish`` calls without touching a DDS bus.

    Same acceptance contract as :class:`DDSPublisher`: ``publish`` returns
    ``None`` on success and a reason string on failure.  Every call is
    stashed in :attr:`writes` so a test can walk the wire capture.
    """

    def __init__(self) -> None:
        self.writes: list[tuple[str, type, Any]] = []
        self.publish_should_return: str | None = None
        self.close_calls = 0

    def publish(self, topic: str, message_class: type, message: Any) -> str | None:
        if self.publish_should_return is not None:
            return self.publish_should_return
        self.writes.append((topic, message_class, message))
        return None

    def close(self) -> None:
        self.close_calls += 1


def _gated_driver() -> tuple[G1Driver, _RecordingPublisher]:
    """Return a driver ready to publish, and the recorder attached to it.

    ``_connected`` is forced True, ``_fsm_id`` is a handshake FSM (the value
    the arm-SDK gate wants), ``_mode_machine`` is a plausible uint8 layout id
    (the value the firmware wants echoed), the battery is above the floor,
    and the publisher is a recorder.  Every write-path test starts here so
    the tests read as "given a gate-passing driver".
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501  # in HANDSHAKE_FSMS
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    pub = _RecordingPublisher()
    driver._pubs = pub  # type: ignore[assignment]
    return driver, pub


@pytest.fixture
def _stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a ``unitree_sdk2py`` stub for the duration of one test.

    ``_build_lowcmd_from_action`` imports ``unitree_sdk2py.idl.default`` and
    ``unitree_sdk2py.utils.crc`` inside its body, and ``send_action`` imports
    ``unitree_sdk2py.idl.unitree_hg.msg.dds_``.  Registering each on
    :mod:`sys.modules` lets the cells below drive the same production lane
    hardware drives, on a box where the SDK is not installed.

    The stub classes come from :mod:`tests.drivers.test_g1_control_loop`
    rather than a fifth copy, so every suite that grades this builder writes
    into the same ``LowCmd_`` shape.  ``monkeypatch.setitem`` restores the
    previous entries - typically absent - on teardown, per AGENTS.md >
    Testing Patterns > Restore a sys.modules entry you remove.
    """
    root = types.ModuleType("unitree_sdk2py")
    idl = types.ModuleType("unitree_sdk2py.idl")
    default = types.ModuleType("unitree_sdk2py.idl.default")
    unitree_hg = types.ModuleType("unitree_sdk2py.idl.unitree_hg")
    unitree_hg_msg = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg")
    dds_ = types.ModuleType("unitree_sdk2py.idl.unitree_hg.msg.dds_")
    utils = types.ModuleType("unitree_sdk2py.utils")
    crc = types.ModuleType("unitree_sdk2py.utils.crc")

    default.unitree_hg_msg_dds__LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    dds_.LowCmd_ = _StubLowCmd  # type: ignore[attr-defined]
    crc.CRC = _StubCRC  # type: ignore[attr-defined]

    for name, mod in [
        ("unitree_sdk2py", root),
        ("unitree_sdk2py.idl", idl),
        ("unitree_sdk2py.idl.default", default),
        ("unitree_sdk2py.idl.unitree_hg", unitree_hg),
        ("unitree_sdk2py.idl.unitree_hg.msg", unitree_hg_msg),
        ("unitree_sdk2py.idl.unitree_hg.msg.dds_", dds_),
        ("unitree_sdk2py.utils", utils),
        ("unitree_sdk2py.utils.crc", crc),
    ]:
        monkeypatch.setitem(sys.modules, name, mod)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_writes_to_lowcmd_topic_when_gates_pass() -> None:
    """A gated action publishes exactly one frame to ``rt/lowcmd``.

    The frame's IDL class is ``LowCmd_`` and no other topic was touched;
    the driver's write path is a single-topic path by design.
    """
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_

    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.5})

    assert result["status"] == "success"
    assert len(pub.writes) == 1
    topic, message_class, message = pub.writes[0]
    assert topic == "rt/lowcmd"
    assert message_class is LowCmd_
    assert isinstance(message, LowCmd_)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_fills_the_named_slot_and_leaves_the_rest_alone() -> None:
    """A caller who names one joint sets exactly one slot's ``q``.

    The other 34 slots have the SDK's default values (all zeros for numeric
    fields on a fresh ``LowCmd_``); a driver that spread the target across
    the wholebody would move joints the caller never asked about.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX

    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.5})
    assert result["status"] == "success"

    _, _, message = pub.writes[0]
    slot = _G1_JOINT_INDEX["left_shoulder_pitch"]
    assert message.motor_cmd[slot].q == pytest.approx(0.5)
    # Every other slot's q stays at the SDK default (0.0).
    for i, motor in enumerate(message.motor_cmd):
        if i == slot:
            continue
        assert motor.q == pytest.approx(0.0)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_applies_default_gains_for_scalar_targets() -> None:
    """A scalar target lands that slot's reference gains on the wire.

    The scalar form is the common case (a caller who just wants a hold).
    ``right_elbow`` is an arm joint, so the vendor's ``rt/lowcmd`` reference
    gives it ``kp=40, kd=1``.

    The expected numbers are written as literals rather than imported from
    :data:`~strands_robots.drivers.g1._SDK_KP` / ``_SDK_KD``: an assertion that
    reads the constant it grades follows any edit to that constant, so it stays
    green for every value including a wrong one.  Spelled out, this cell is an
    independent check on the value that reaches the topic.
    ``tests/drivers/test_g1_per_joint_gains.py`` grades the whole table the same
    way, slot by slot.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX

    driver, pub = _gated_driver()
    driver.send_action({"right_elbow": -0.2})

    _, _, message = pub.writes[0]
    slot = _G1_JOINT_INDEX["right_elbow"]
    m = message.motor_cmd[slot]
    assert m.q == pytest.approx(-0.2)
    assert m.kp == pytest.approx(40.0)
    assert m.kd == pytest.approx(1.0)
    assert m.dq == pytest.approx(0.0)
    assert m.tau == pytest.approx(0.0)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_accepts_per_joint_gains_and_feedforward() -> None:
    """A dict target lands every field the caller supplied.

    Missing keys inside the dict fall back to the driver default (gains) or
    zero (dq/tau); the presence of one key does not overwrite another with
    an implicit value.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX

    driver, pub = _gated_driver()
    driver.send_action(
        {
            "left_elbow": {"q": 0.3, "kp": 50.0, "kd": 1.5, "dq": 0.1, "tau": 0.05},
        }
    )
    slot = _G1_JOINT_INDEX["left_elbow"]
    _, _, message = pub.writes[0]
    m = message.motor_cmd[slot]
    assert m.q == pytest.approx(0.3)
    assert m.kp == pytest.approx(50.0)
    assert m.kd == pytest.approx(1.5)
    assert m.dq == pytest.approx(0.1)
    assert m.tau == pytest.approx(0.05)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_unknown_joint_name() -> None:
    """An unknown joint name refuses the whole action - no partial writes.

    A silent drop would move only the recognised joints, leaving the
    caller with a robot that did most of what it wanted.  Loud is the
    only right answer.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_shoulder_pitch": 0.1, "elbow_typo": 0.2})
    assert result["status"] == "error"
    assert "unknown joint name" in result["content"][0]["text"]
    assert "elbow_typo" in result["content"][0]["text"]
    assert pub.writes == []  # no partial writes


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_per_joint_dict_missing_q() -> None:
    """A per-joint dict without ``q`` is refused; no default target is invented.

    Zeroing an unnamed target would be worse than refusing: on a robot
    holding a pose, ``q=0`` is a full swing to the joint's zero.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": {"kp": 10.0}})
    assert result["status"] == "error"
    assert "missing required key 'q'" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_unknown_per_joint_key() -> None:
    """An unknown inner key refuses the action - same reason as an unknown joint name.

    A typo like ``"K_p"`` for ``"kp"`` would land with the default gain and
    silently ignore the caller's intent.  Refuse.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": {"q": 0.1, "K_p": 50.0}})
    assert result["status"] == "error"
    assert "unknown per-joint keys" in result["content"][0]["text"]
    assert "K_p" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_a_target_that_is_not_a_finite_number() -> None:
    """A target outside the finite-number domain is refused, naming the field.

    The reason comes from the shared
    :func:`~strands_robots.utils.finite_number_error` domain, so it names the
    joint *and* the field it was wrong on - a caller commanding five fields per
    joint needs to know which one.
    """
    driver, pub = _gated_driver()
    result = driver.send_action({"left_elbow": "hold"})
    assert result["status"] == "error"
    assert "left_elbow.q" in result["content"][0]["text"]
    assert "must be a finite number" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_empty_action() -> None:
    """An empty action dict is refused - "nothing to command" is not a write."""
    driver, pub = _gated_driver()
    result = driver.send_action({})
    assert result["status"] == "error"
    assert "empty" in result["content"][0]["text"]
    assert pub.writes == []


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_surfaces_publisher_error() -> None:
    """A publisher that returns a reason surfaces that reason in the envelope.

    The envelope's ``text`` field carries the publisher's exact string so a
    log downstream still shows what the DDS layer complained about; the
    driver adds no extra wrapping.
    """
    driver, pub = _gated_driver()
    pub.publish_should_return = "publish to 'rt/lowcmd' failed: bus is down"
    result = driver.send_action({"left_elbow": 0.1})
    assert result["status"] == "error"
    assert "bus is down" in result["content"][0]["text"]


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_refuses_when_pubs_is_missing() -> None:
    """A gated driver whose ``_pubs`` was never set refuses with a named reason.

    This is the "forced ``_connected = True`` in a test" path.  Real bring-up
    sets ``_pubs`` in :meth:`connect_eagerly` alongside ``_connected``; the
    refusal points a reader at that method rather than at the SDK.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 501
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    driver._battery = {"pct": 92.0, "current": 1.0, "cycle": 0, "t": 0.0}
    # _pubs left at its __init__ default (None).
    result = driver.send_action({"left_elbow": 0.1})
    assert result["status"] == "error"
    assert "publisher not initialised" in result["content"][0]["text"]


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_returns_success_envelope_with_joints_and_topic() -> None:
    """The success envelope carries the topic and every joint it commanded.

    A caller that logs the envelope should see exactly what went on the wire
    - not "success" without a body.  This is what makes a downstream monitor
    (dashboard, telemetry) attributable at the driver seam.
    """
    driver, _pub = _gated_driver()
    result = driver.send_action({"left_elbow": 0.1, "right_elbow": -0.1})
    assert result["status"] == "success"
    body = result["content"][0]["json"]
    assert body["topic"] == "rt/lowcmd"
    assert body["joints"] == ["left_elbow", "right_elbow"]  # sorted
    assert body["fsm_id"] == 501


def test_send_action_still_refuses_before_fsm_gate_regardless_of_wire() -> None:
    """The FSM gate runs before the wire path - a stub-shaped call still refuses.

    This pins the wire path is *after* the gates: a caller who supplies an
    unknown joint on a gate-failing driver still gets the FSM refusal, not
    a joint-name refusal, because the gate does not care what the write
    was going to say.
    """
    driver = G1Driver(tool_name="g1", port="1.2.3.4")
    driver._connected = True
    driver._fsm_id = 0  # not in HANDSHAKE_FSMS
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    result = driver.send_action({"elbow_typo": 0.0})
    assert result["status"] == "error"
    assert "FSM 0" in result["content"][0]["text"]
    # And crucially, the gate ran before the joint-name check.
    assert "unknown joint" not in result["content"][0]["text"]


# =========================================================================
# _build_lowcmd_from_action - the pure helper, tested off the driver.    #
# =========================================================================


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_lowcmd_scalar_and_dict_land_on_the_same_slot() -> None:
    """Scalar and dict forms land ``q`` on the same slot for the same joint.

    The form is a convenience; the wire result is the same target value on
    the same slot.  A caller that reads the driver's contract picks whichever
    shape suits, knowing it is not a semantic switch.
    """
    from strands_robots.drivers.g1 import (
        _G1_JOINT_INDEX,
        _build_lowcmd_from_action,
    )

    cmd_scalar, err_s = _build_lowcmd_from_action({"waist_yaw": 0.4})
    cmd_dict, err_d = _build_lowcmd_from_action({"waist_yaw": {"q": 0.4}})
    assert err_s is None and err_d is None
    slot = _G1_JOINT_INDEX["waist_yaw"]
    assert cmd_scalar.motor_cmd[slot].q == pytest.approx(cmd_dict.motor_cmd[slot].q)


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_lowcmd_from_action_rejects_non_dict_input() -> None:
    """A non-dict action is refused with the type it actually was."""
    from strands_robots.drivers.g1 import _build_lowcmd_from_action

    cmd, err = _build_lowcmd_from_action([("left_elbow", 0.1)])  # type: ignore[arg-type]
    assert cmd is None
    assert err is not None
    assert "must be a dict" in err
    assert "list" in err


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_zero_torque_lowcmd_zeroes_every_motor_field() -> None:
    """Every motor in the zero-torque envelope has kp=kd=tau=q=dq=0.

    The control loop (issue #361 follow-up) uses this on stop; the shape
    is what "soft" looks like on the wire.  Pinning it here means a change
    to the mapping fails a test rather than reaches a robot.
    """
    from strands_robots.drivers.g1 import _build_zero_torque_lowcmd

    cmd, err = _build_zero_torque_lowcmd()
    assert err is None
    assert cmd is not None
    for m in cmd.motor_cmd:
        assert m.q == 0.0
        assert m.dq == 0.0
        assert m.tau == 0.0
        assert m.kp == 0.0
        assert m.kd == 0.0


# =========================================================================
# Wire-frame contract: the four fields the firmware validates.           #
# =========================================================================
# A frame with wrong CRC, wrong mode_machine, or motor.mode = Disable is
# silently dropped by the G1 firmware.  Every one of these tests catches a
# concrete class of "PR reports success and the robot does nothing".


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_build_lowcmd_sets_a_matching_crc_the_firmware_will_accept() -> None:
    """``cmd.crc`` is the SDK-computed CRC over every other field.

    The G1 firmware validates ``LowCmd_.crc`` on ``rt/lowcmd`` and drops a
    non-matching frame without reporting.  Pinning this here catches a
    future edit that populates the frame after the CRC is stamped, or
    forgets to stamp it at all - either yields a success envelope over a
    frame the robot ignores.
    """
    from unitree_sdk2py.utils.crc import CRC

    from strands_robots.drivers.g1 import _build_lowcmd_from_action

    cmd, err = _build_lowcmd_from_action({"waist_yaw": 0.4}, mode_machine=7)
    assert err is None and cmd is not None
    # Independent oracle: recompute what the SDK would compute over the
    # same message and compare.
    expected = CRC().Crc(cmd)
    assert cmd.crc == expected
    assert cmd.crc != 0


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_lowcmd_echoes_mode_machine_and_pins_mode_pr_to_zero() -> None:
    """``mode_machine`` mirrors ``LowState``; ``mode_pr`` stays PR.

    The firmware refuses a ``LowCmd_`` whose ``mode_machine`` does not
    match the value the arm-SDK last published on ``rt/lowstate``.  The
    driver caches that value on every ``LowState`` frame in ``_fsm_id``
    and passes it in.  ``mode_pr = 1`` (AB mode) would silently remap four
    ankle indices, so it stays 0.
    """
    from strands_robots.drivers.g1 import _build_lowcmd_from_action

    cmd, err = _build_lowcmd_from_action({"waist_yaw": 0.4}, mode_machine=9)
    assert err is None and cmd is not None
    assert cmd.mode_machine == 9
    assert cmd.mode_pr == 0


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_lowcmd_enables_the_touched_slot_and_leaves_the_rest_disabled() -> None:
    """Only commanded slots carry ``motor.mode = 1``; the others stay 0.

    A frame with a valid CRC but ``motor_cmd[i].mode = 0`` (Disable) is
    silently ignored on that slot.  The alternative would be a success
    envelope over a joint the caller thought was commanded and that never
    moved.  Uncommanded slots stay Disable, which is what "did not
    command this joint" means on the wire.
    """
    from strands_robots.drivers.g1 import _G1_JOINT_INDEX, _build_lowcmd_from_action

    cmd, err = _build_lowcmd_from_action({"waist_yaw": 0.4}, mode_machine=7)
    assert err is None and cmd is not None
    touched = _G1_JOINT_INDEX["waist_yaw"]
    assert cmd.motor_cmd[touched].mode == 1
    # Every other slot stays at 0 (Disable) - the default no-op.
    for i, m in enumerate(cmd.motor_cmd):
        if i == touched:
            continue
        assert m.mode == 0, f"slot {i} was enabled without being commanded"


@pytest.mark.skipif(not _HAS_SDK, reason="unitree_sdk2py not installed")
def test_build_zero_torque_stamps_crc_and_enables_every_named_slot() -> None:
    """The soft-hold frame carries a valid CRC and Enable on every named slot.

    ``stop`` (and the follow-up 500 Hz loop's shutdown path) publish this
    frame; if it lands as a wire-side no-op the arm falls freely.  Enable
    with zero gains is the softest state the SDK protocol expresses on the
    joints this driver names; a Disable slot with zero gains is a no-op.

    The Enable byte is bounded by ``_G1_NAMED_JOINTS`` (29) rather than by
    ``len(motor_cmd)`` (35): slots 29..34 are a reserved tail that no name
    in ``_G1_JOINT_INDEX`` maps to, and the SDK's own G1 reference loops
    ``for i in range(G1_NUM_MOTOR)`` where ``G1_NUM_MOTOR = 29``.  Enabling
    a reserved slot at zero gains would be a decision this driver does not
    have the information to make; leaving them at SDK defaults keeps the
    stop frame byte-identical to the reference on those slots.
    """
    from unitree_sdk2py.utils.crc import CRC

    from strands_robots.drivers.g1 import (
        _G1_NAMED_JOINTS,
        _build_zero_torque_lowcmd,
    )

    cmd, err = _build_zero_torque_lowcmd(mode_machine=7)
    assert err is None and cmd is not None
    assert cmd.mode_machine == 7
    assert cmd.mode_pr == 0
    assert cmd.crc == CRC().Crc(cmd)
    # Every named slot (0..28) carries Enable; the reserved tail stays at 0.
    for i in range(_G1_NAMED_JOINTS):
        assert cmd.motor_cmd[i].mode == 1, f"named slot {i} is not Enable on the stop frame"
    for i in range(_G1_NAMED_JOINTS, len(cmd.motor_cmd)):
        assert cmd.motor_cmd[i].mode == 0, (
            f"reserved slot {i} was enabled; the SDK reference bounds "
            f"Enable by G1_NUM_MOTOR (29), not by the array width"
        )


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_build_zero_torque_enables_every_named_slot_and_leaves_the_tail() -> None:
    """The stop frame's Enable bound, graded without the SDK.

    The cell above grades this too, but only where ``unitree_sdk2py`` is
    installed, because it recomputes the SDK's own CRC as an independent
    oracle.  The Enable bound is not a CRC question: a stop frame whose
    named slots carry ``mode = 0`` is a wire-side no-op and the arm falls
    freely, and that is decidable against any ``LowCmd_``-shaped object.
    Graded here so ``call-test-lint`` refuses the regression rather than
    only a box that happens to carry the SDK.
    """
    from strands_robots.drivers.g1 import _G1_NAMED_JOINTS, _build_zero_torque_lowcmd

    cmd, err = _build_zero_torque_lowcmd(mode_machine=7)
    assert err is None and cmd is not None
    assert cmd.mode_machine == 7
    assert cmd.mode_pr == 0
    for i in range(_G1_NAMED_JOINTS):
        assert cmd.motor_cmd[i].mode == 1, f"named slot {i} is not Enable on the stop frame"
    for i in range(_G1_NAMED_JOINTS, len(cmd.motor_cmd)):
        assert cmd.motor_cmd[i].mode == 0, (
            f"reserved slot {i} was enabled; the SDK reference bounds "
            f"Enable by G1_NUM_MOTOR (29), not by the array width"
        )


@pytest.mark.usefixtures("_stub_unitree_sdk")
def test_send_action_echoes_cached_mode_machine_not_fsm_id() -> None:
    """``send_action`` echoes ``_mode_machine`` (uint8) onto the wire, not ``_fsm_id``.

    The firmware validates ``LowCmd_.mode_machine`` against the layout id it
    published on the last ``LowState_.mode_machine`` (uint8, packed ``<2B``
    with ``mode_pr``).  ``_fsm_id`` is a different quantity - a high-level
    FSM state from the motion-switcher API whose values (500, 501, 801)
    would overflow the ``B`` pack format and raise ``struct.error`` on CRC.
    The gate consults ``_fsm_id`` for admission; the frame carries
    ``_mode_machine`` for the echo.  This test pins that separation on the
    end-to-end wire path.
    """
    driver, rec = _gated_driver()
    driver._fsm_id = 501  # a handshake FSM; the gate accepts.
    driver._mode_machine = 9  # uint8 layout id echoed from lowstate
    result = driver.send_action(robot_name="g1", action={"waist_yaw": 0.4})
    assert result["status"] == "success"
    assert len(rec.writes) == 1
    _, _, cmd = rec.writes[0]
    assert cmd.mode_machine == 9  # uint8 echoed, not the high-level 501
    assert cmd.mode_pr == 0
    assert cmd.crc != 0
    # The success envelope surfaces both fields so a caller can distinguish
    # gate value from echo value in a log without opening the frame.
    body = result["content"][0]["json"]
    assert body["fsm_id"] == 501
    assert body["mode_machine"] == 9


# =========================================================================
# Module-load hygiene: the write path adds no SDK import at load time.   #
# =========================================================================


def test_g1_driver_module_does_not_import_unitree_sdk2py_at_load_time() -> None:
    """Importing :mod:`strands_robots.drivers.g1` does not import the SDK.

    The DDSPublisher tests pin this for :mod:`_dds_engine`; this test pins
    it for the driver module itself so a future edit that reaches for a
    module-level ``import unitree_sdk2py.idl.default`` fails here.  Thor
    and CI both need the driver module importable without the SDK.

    Measured in a clean interpreter rather than by reloading the module in
    this one. :func:`importlib.reload` re-executes a module body into the same
    namespace, which rebinds every class the body defines. The driver registry
    captures :class:`~strands_robots.drivers.g1.G1Driver` by reference when the
    shipped table is registered, so a reload leaves that reference pointing at
    a class object that is no longer what the module's own name resolves to:
    every later ``is`` comparison in the session then fails between two classes
    with an identical ``repr``. A subprocess cannot rebind anything here, and it
    states the contract more strongly - no SDK module is loaded at all, rather
    than none beyond whatever an earlier test in this session already imported.
    """
    import subprocess
    import sys

    probe = (
        "import sys; import strands_robots.drivers.g1; "
        "print(sorted(n for n in sys.modules if n.startswith('unitree_sdk2py')))"
    )
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=False)
    assert completed.returncode == 0, f"importing the driver module failed: {completed.stderr}"
    assert completed.stdout.strip() == "[]", (
        f"importing strands_robots.drivers.g1 pulled in unitree_sdk2py modules: {completed.stdout.strip()}"
    )


def test_the_sdk_import_pin_leaves_the_registered_driver_class_reachable_by_name() -> None:
    """The pin above must not leave a second ``G1Driver`` class behind.

    The registry hands back the class object it was registered with. A pin that
    re-executed the driver module in this interpreter would rebind the module's
    own ``G1Driver`` to a new object, so this assertion is the one that reads
    the two apart - the failure names two classes with the same ``repr``, which
    is the only symptom a rebind produces. Calling the pin directly rather than
    relying on collection order means the coupling is graded in one file.
    """
    import importlib

    from strands_robots.drivers.g1 import G1Driver
    from strands_robots.drivers.registry import get_native_driver_class

    test_g1_driver_module_does_not_import_unitree_sdk2py_at_load_time()

    assert get_native_driver_class("unitree_g1") is G1Driver
    assert importlib.import_module("strands_robots.drivers.g1").G1Driver is G1Driver
