"""Every field a Unitree low-command carries to a motor must be a finite number.

The two Unitree drivers build one ``LowCmd_`` per control step from a
joint-name-keyed action dict, and each commanded slot carries five physical
quantities: the position target ``q``, the gains ``kp`` / ``kd``, the velocity
feed-forward ``dq`` and the effort ``tau``. Both builders coerced every one of
them with a bare ``float()``, which is a *type* check and not a *domain* check:
``float()`` accepts ``nan``, ``inf``, the string ``"nan"`` and ``True``.

Measured on the real ``unitree_sdk2py`` IDL before this gate existed, on
``G1Driver`` and ``Go2Driver`` alike, every row below produced a fully populated
frame with the slot's enable byte set and a valid CRC:

===========================================  =========================
action                                       what reached ``motor_cmd``
===========================================  =========================
``{"left_shoulder_pitch": float("nan")}``    ``q=nan``
``{"left_shoulder_pitch": float("inf")}``    ``q=inf``
``{"left_shoulder_pitch": "nan"}``           ``q=nan``
``{... {"q": 0.1, "kp": float("inf")}}``     ``kp=inf``
``{... {"q": 0.1, "tau": float("nan")}}``    ``tau=nan``
``{"left_shoulder_pitch": True}``            ``q=1.0``
===========================================  =========================

A ``nan`` is the dangerous spelling rather than merely an odd one, for the same
reason it was on this driver's battery floor (see
``tests.drivers.test_g1_battery_floor_is_a_finite_percentage``): it survives the
coercion and then serializes onto the wire as a perfectly valid IEEE-754 float,
so the firmware accepts the frame, the CRC matches, and the motor controller
integrates an unrepresentable target into its own state. Nothing in any log says
why. A refusal is not available at that point - the frame is already gone.

The realistic route is not a hand-written action but a policy: an inference
server that diverges, a checkpoint with a ``NaN`` weight, or a normalisation
that divides by a zero-variance statistic all return a ``nan`` action while
reporting success. The control loop already owns a lane for that -
``exit_reason="policy"``, detail ``"policy returned an unusable action"`` - and
``nan`` simply was not in the definition of unusable, so the frame published
instead of taking it.

The domain itself is not new and is not defined here: ten of the twelve native
drivers already put their action values through
:func:`~strands_robots.utils.finite_number_error` (UR per joint in
``targets_from_action``, Booster per field in ``send_action``, Franka, Reachy,
Microduck, Earthrover, Crazyflie, and Feetech / Robotiq through their own wire
layers). The two Unitree builders were the exception, and both *imported* that
helper already - for the constructor's ``battery_floor_pct`` and for
``run_policy``'s ``duration``, but not for the values that move the robot.

Scope, deliberately: this is not a magnitude limit. How far a joint may be
commanded is the arm-SDK client's question and both drivers say so; whether a
value names a pose at all is this one.
"""

from __future__ import annotations

import math
import sys
import time
import types
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from strands_robots.drivers.g1 import (
    _G1_JOINT_INDEX,
    _WIRE_FIELDS,
    _build_lowcmd_from_action,
    _ControlLoop,
)
from strands_robots.drivers.go2 import _WIRE_FIELDS as _GO2_WIRE_FIELDS
from strands_robots.drivers.go2 import GO2_JOINT_INDEX, build_lowcmd_from_action
from strands_robots.utils import finite_number_error

# One joint per driver is enough: the builders loop over the action dict, so the
# gate is per-value and not per-slot. These two are named rather than derived so
# a reader can see which joint the messages below are about.
G1_JOINT = "left_shoulder_pitch"
GO2_JOINT = "FL_hip_joint"

#: ``(label, builder, joint_name, slot_index)`` for each driver, so every domain
#: cell below grades both without being written twice.
BUILDERS = (
    pytest.param("g1", _build_lowcmd_from_action, G1_JOINT, _G1_JOINT_INDEX[G1_JOINT], id="g1"),
    pytest.param("go2", build_lowcmd_from_action, GO2_JOINT, GO2_JOINT_INDEX[GO2_JOINT], id="go2"),
)

#: Values outside the domain, each with the reason it is not merely unusual.
OUTSIDE_THE_DOMAIN = (
    pytest.param(float("nan"), id="nan"),
    pytest.param(float("inf"), id="inf"),
    pytest.param(float("-inf"), id="-inf"),
    pytest.param("nan", id="nan-as-a-string"),
    pytest.param("0.25", id="a-numeric-string"),
    pytest.param(True, id="bool-true-would-be-one-radian"),
    pytest.param(None, id="none"),
    pytest.param([0.25], id="a-list"),
    pytest.param(10**400, id="past-the-float64-range"),
)

#: Real scalars a policy legitimately produces. NumPy is the case that matters:
#: an action read off a policy tensor arrives as ``np.float32``, and refusing it
#: would break every inference path. Same acceptance the battery floor has.
INSIDE_THE_DOMAIN = (
    pytest.param(0.25, id="float"),
    pytest.param(0, id="int-zero"),
    pytest.param(-1.5, id="negative"),
    pytest.param(np.float32(0.25), id="numpy-float32"),
    pytest.param(np.float64(-0.25), id="numpy-float64"),
)


# ---------------------------------------------------------------------------
# One stub for both IDL namespaces. The builders import the SDK inside their
# bodies, so registering the modules on sys.modules drives the same production
# lane hardware drives - and puts these cells in front of CI, which has no SDK.
# ---------------------------------------------------------------------------


class _StubMotorCmd:
    """One ``motor_cmd`` slot: the five wire fields plus the enable byte."""

    def __init__(self) -> None:
        self.mode: int = 0
        self.q: float = 0.0
        self.dq: float = 0.0
        self.tau: float = 0.0
        self.kp: float = 0.0
        self.kd: float = 0.0


class _StubLowCmd:
    """Stand-in for a ``LowCmd_`` default instance, carrying both headers.

    The slot width is per-subclass because it is not decoration: the Go2 builder
    refuses an IDL whose ``motor_cmd`` is not exactly 20 wide, and the G1's is
    35. One stub class serving both would be caught by that check - which is the
    driver doing its job.
    """

    _SLOTS = 0

    def __init__(self) -> None:
        self.mode_pr: int = 0
        self.mode_machine: int = 0
        self.head: list[int] = [0, 0]
        self.level_flag: int = 0
        self.gpio: int = 0
        self.motor_cmd: list[_StubMotorCmd] = [_StubMotorCmd() for _ in range(self._SLOTS)]
        self.crc: int = 0


class _StubHgLowCmd(_StubLowCmd):
    """The G1's ``unitree_hg`` width, which the builder asserts is wide enough."""

    _SLOTS = 35


class _StubGoLowCmd(_StubLowCmd):
    """The Go2's ``unitree_go`` width, which its builder requires exactly."""

    _SLOTS = 20


class _StubCRC:
    """Stand-in for ``unitree_sdk2py.utils.crc.CRC``; ``42`` is a live marker."""

    def Crc(self, _cmd: Any) -> int:
        return 42


@pytest.fixture(autouse=True)
def _stub_unitree_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Register both IDL namespaces the two builders import.

    ``monkeypatch.setitem`` restores the previous entries on teardown, so a box
    that does have the real SDK installed is left as it was.
    """
    modules = {
        name: types.ModuleType(name)
        for name in (
            "unitree_sdk2py",
            "unitree_sdk2py.idl",
            "unitree_sdk2py.idl.default",
            "unitree_sdk2py.idl.unitree_hg",
            "unitree_sdk2py.idl.unitree_hg.msg",
            "unitree_sdk2py.idl.unitree_hg.msg.dds_",
            "unitree_sdk2py.idl.unitree_go",
            "unitree_sdk2py.idl.unitree_go.msg",
            "unitree_sdk2py.idl.unitree_go.msg.dds_",
            "unitree_sdk2py.utils",
            "unitree_sdk2py.utils.crc",
        )
    }
    modules["unitree_sdk2py.idl.default"].unitree_hg_msg_dds__LowCmd_ = _StubHgLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.idl.default"].unitree_go_msg_dds__LowCmd_ = _StubGoLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.idl.unitree_hg.msg.dds_"].LowCmd_ = _StubHgLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.idl.unitree_go.msg.dds_"].LowCmd_ = _StubGoLowCmd  # type: ignore[attr-defined]
    modules["unitree_sdk2py.utils.crc"].CRC = _StubCRC  # type: ignore[attr-defined]
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


# ---------------------------------------------------------------------------
# The domain, on every field of both builders.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(("label", "build", "joint", "slot"), BUILDERS)
@pytest.mark.parametrize("field", _WIRE_FIELDS)
@pytest.mark.parametrize("value", OUTSIDE_THE_DOMAIN)
def test_a_field_outside_the_domain_refuses_the_whole_frame(
    label: str, build: Any, joint: str, slot: int, field: str, value: Any
) -> None:
    """No frame is built at all - not a frame with the bad field dropped.

    Refusing whole is the same posture an unknown joint name gets: silently
    dropping a field the caller believed was commanded is worse than an error,
    because the slot would then take a default the caller never chose.
    """
    del label, slot
    cmd, reason = build({joint: {"q": 0.1, field: value}} if field != "q" else {joint: {"q": value}})
    assert cmd is None, f"{field}={value!r} built a frame that would reach the wire"
    assert reason is not None
    assert f"{joint}.{field}" in reason, "the reason must name the joint and the field"


@pytest.mark.parametrize(("label", "build", "joint", "slot"), BUILDERS)
@pytest.mark.parametrize("value", OUTSIDE_THE_DOMAIN)
def test_a_scalar_action_outside_the_domain_refuses_the_whole_frame(
    label: str, build: Any, joint: str, slot: int, value: Any
) -> None:
    """The scalar spelling is the position target, and takes the same domain."""
    del label, slot
    cmd, reason = build({joint: value})
    assert cmd is None
    assert reason is not None and f"{joint}.q" in reason


@pytest.mark.parametrize(("label", "build", "joint", "slot"), BUILDERS)
@pytest.mark.parametrize("value", INSIDE_THE_DOMAIN)
def test_a_real_scalar_still_reaches_the_slot(label: str, build: Any, joint: str, slot: int, value: Any) -> None:
    """The controls. A gate that refused these would break every policy.

    ``np.float32`` is the row that matters: an action read off a policy tensor
    arrives as a NumPy scalar, not a Python float.
    """
    del label
    cmd, reason = build({joint: value})
    assert reason is None, f"{value!r} is a usable target and must not be refused"
    assert cmd is not None
    assert cmd.motor_cmd[slot].q == pytest.approx(float(value))
    assert cmd.motor_cmd[slot].mode != 0, "a commanded slot must be enabled"


@pytest.mark.parametrize(("label", "build", "joint", "slot"), BUILDERS)
def test_every_field_the_builder_accepts_is_a_field_it_checks(label: str, build: Any, joint: str, slot: int) -> None:
    """The vocabulary accepted and the vocabulary checked are one list.

    A sixth wire field added to ``_WIRE_FIELDS`` later is graded on arrival: it
    becomes an accepted inner key *and* a checked one in the same edit, so the
    two cannot drift into a field that is written without being judged.
    """
    del slot
    assert _WIRE_FIELDS == _GO2_WIRE_FIELDS, "both drivers frame the same five fields"
    for field in _WIRE_FIELDS:
        action = {joint: {"q": 0.1, field: float("nan")}} if field != "q" else {joint: {"q": float("nan")}}
        cmd, reason = build(action)
        assert cmd is None and reason is not None, f"{label}: {field} is accepted but not checked"
    # An unknown inner key is still refused, so the accepted set is exactly this.
    cmd, reason = build({joint: {"q": 0.1, "torque": 1.0}})
    assert cmd is None and reason is not None and "unknown per-joint keys" in reason


@pytest.mark.parametrize(("label", "build", "joint", "slot"), BUILDERS)
def test_the_reason_is_the_shared_domain_verbatim(label: str, build: Any, joint: str, slot: int) -> None:
    """Neither driver carries its own copy of the domain or of its wording.

    Compared against the helper's own output rather than a fragment, so a
    private re-implementation that happened to refuse ``nan`` would still fail
    here - the point is one domain, not two that agree today.
    """
    del label, slot
    _, reason = build({joint: float("nan")})
    assert reason == finite_number_error(float("nan"), f"{joint}.q", "send_action")


# ---------------------------------------------------------------------------
# The realistic route: a policy that returns a non-finite action.
# ---------------------------------------------------------------------------


class _RecordingPublisher:
    """Records every ``publish``; returns ``None`` like the real one on success."""

    def __init__(self) -> None:
        self.calls: list[Any] = []
        self.closed = False

    def publish(self, topic: str, klass: Any, cmd: Any) -> str | None:
        self.calls.append((topic, klass, cmd))
        return None

    def close(self) -> None:
        self.closed = True


def _fake_driver(publisher: _RecordingPublisher) -> Any:
    """A driver good enough for :class:`_ControlLoop`, with gates that admit."""
    import threading

    driver = MagicMock(
        spec=[
            "_mode_machine",
            "_fsm_id",
            "_battery",
            "_imu",
            "_pubs",
            "_check_motion_gates",
            "_loop",
            "_task_admission",
            "_tool_name",
            "_refresh_fsm_id",
            "_fsm_read_at",
            "_motion_switcher_lock",
        ]
    )
    driver._mode_machine = 9
    driver._fsm_id = 500
    driver._tool_name = "g1"
    driver._refresh_fsm_id = MagicMock(side_effect=lambda: setattr(driver, "_fsm_read_at", time.monotonic()))
    driver._fsm_read_at = time.monotonic()
    driver._motion_switcher_lock = threading.Lock()
    driver._battery = {"pct": 80.0}
    driver._imu = {"rpy": [0.0, 0.0, 0.0]}
    driver._pubs = publisher
    driver._check_motion_gates = MagicMock(return_value=None)
    driver._loop = None
    driver._task_admission = threading.Lock()
    return driver


def test_a_policy_that_diverges_stops_the_loop_instead_of_publishing_nan() -> None:
    """The money case: a diverged policy takes the loop's own refusal lane.

    Two good steps, then ``nan``. The loop already reports an unusable action as
    ``exit_reason="policy"``; what this pins is that a ``nan`` action *is*
    unusable, so the frame carrying it is never published. Every position frame
    on the wire holds finite targets, and the loop still soft-stops on the way
    out - a refusal must not leave the motors hot.
    """
    publisher = _RecordingPublisher()
    driver = _fake_driver(publisher)
    steps = {"n": 0}

    def policy(_obs: Any) -> dict[str, float]:
        steps["n"] += 1
        if steps["n"] > 2:
            return {G1_JOINT: float("nan")}
        return {G1_JOINT: 0.1}

    # Bounded by ``n_steps`` so the loop terminates whether or not the gate
    # exists: without it a published ``nan`` is simply the next setpoint and the
    # loop runs its full duration, which would make this cell fail on a timeout
    # rather than on the claim it is here to make.
    loop = _ControlLoop(driver=driver, policy=policy, duration=60.0, n_steps=6)
    loop.start()
    deadline = time.monotonic() + 10.0
    while loop.is_running and time.monotonic() < deadline:
        time.sleep(0.01)
    assert not loop.is_running, "loop did not finish"

    snap = loop.snapshot()
    assert snap["exit_reason"] == "policy"
    assert "unusable action" in (snap["exit_detail"] or "")
    assert "must be a finite number" in (snap["exit_detail"] or "")

    published = [cmd for _topic, _klass, cmd in publisher.calls]
    assert published, "the two good steps must have reached the wire"
    for cmd in published:
        for slot in range(len(cmd.motor_cmd)):
            motor = cmd.motor_cmd[slot]
            for field in _WIRE_FIELDS:
                # ``math.isfinite`` and deliberately not the builder's own
                # :func:`finite_number_error`, which this module reaches for
                # everywhere else: a scan has to be able to fail when the
                # shared domain is the thing that regressed, and asking that
                # helper would make the oracle circular - gate and scan would
                # then regress together and the wire would read clean with a
                # ``nan`` on it. Finiteness is the whole domain here anyway,
                # because only ``float`` can reach a field: the ``float()``
                # coercion sits downstream of the gate, and a value with no
                # float64 form makes the builder raise before it writes.
                value = getattr(motor, field)
                assert math.isfinite(value), f"a non-finite {field}={value!r} reached the wire"
