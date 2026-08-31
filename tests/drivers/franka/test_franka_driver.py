"""Tests for :class:`strands_robots.drivers.franka.driver.FrankaDriver`.

Grades the driver's surface, its per-arm joint vocabulary, its state decode, its
command gates and its lifecycle. Nothing here opens an FCI link: the vendor
binding is a fake module installed in ``sys.modules``, shaped like the part of
``panda_py`` the driver actually calls, so a change to *which* calls the driver
makes shows up as a fake that no longer answers rather than as a silent pass.

Table-driven where the behaviour is a rule over many inputs (the stride, the
decode refusals, the action refusals), one cell per behaviour otherwise. The
whole family is one file because the driver is one contract - a reader who wants
to know what a Franka command must satisfy should not have to open nine.
"""

from __future__ import annotations

import asyncio
import sys
import threading
import time
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
from strands.types.tools import ToolUse

from strands_robots.drivers import (
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.franka import FrankaDriver
from strands_robots.drivers.franka.driver import (
    _NO_POLICY_PROVIDER,
    DEFAULT_SPEED_FACTOR,
    DOF,
    FCI_RATE_HZ,
    GRIPPER_KEY,
    JOINT_PREFIXES,
    SUPPORTED_ROBOTS,
    action_keys_for,
    action_to_targets,
    decode_robot_state,
    downsample_stride,
    joint_names_for,
)
from strands_robots.policies import MockPolicy
from strands_robots.registry import get_robot

_HOST = "172.16.0.2"

# A pose and its telemetry, in FCI order. Distinct values per joint so a decode
# that transposed or reused a slot produces a wrong answer rather than a
# coincidentally right one.
_Q = (0.1, -0.2, 0.3, -2.4, 0.5, 1.6, 0.7)
_DQ = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07)
_TAU = (1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7)


# ============================================================================
# The fake FCI. Shaped like the calls the driver makes, and no others.
# ============================================================================


class FakeRobot:
    """Stands in for ``panda_py.libfranka.Robot``, the object ``Panda.get_robot()``
    hands back.

    It carries the halt. ``franka::Robot::stop()`` is bound here and nowhere else
    in the binding, and it is the call designed to abort a running control loop
    from another thread - so a driver that halts through anything else is either
    calling a method that does not exist or ending the wrong thing.
    """

    def __init__(self, *, fail: Exception | None = None) -> None:
        self.stops = 0
        self._fail = fail

    def stop(self) -> None:
        if self._fail is not None:
            raise self._fail
        self.stops += 1


class FakePanda:
    """Stands in for ``panda_py.Panda``: exactly the calls the binding exposes.

    Shaped from the binding's own surface rather than from what the driver
    happens to call, because the two differing is the drift this fake exists to
    catch. Three properties of the real object are load-bearing and are all
    modelled here:

    * **There is no ``stop``.** The wrapper binds ``stop_controller`` (which ends
      a torque controller) and ``get_robot``; the halt lives on the object the
      latter returns.
    * **``move_to_joint_position`` blocks and returns a ``bool``.** It runs the
      whole trajectory before returning, then reports whether the arm ended
      within the success threshold of the goal. ``block`` models the first,
      ``reached`` the second.
    * **A control error does not propagate.** The realtime thread catches it and
      parks it for ``raise_error()``, so the motion call returns normally and the
      error surfaces only when it is collected. ``parked`` models that.
    """

    def __init__(
        self,
        hostname: str,
        *,
        fail: Exception | None = None,
        reached: bool = True,
        parked: Exception | None = None,
        block: threading.Event | None = None,
        entered: threading.Event | None = None,
    ) -> None:
        self.hostname = hostname
        self.moves: list[tuple[list[float], float]] = []
        self.closed = False
        self.robot = FakeRobot(fail=fail)
        self.raise_error_calls = 0
        self._fail = fail
        self._reached = reached
        self._parked = parked
        self._block = block
        self._entered = entered
        self.state: Any = SimpleNamespace(q=list(_Q), dq=list(_DQ), tau_J=list(_TAU))

    def get_state(self) -> Any:
        if self._fail is not None:
            raise self._fail
        return self.state

    def get_robot(self) -> FakeRobot:
        return self.robot

    def stop_controller(self) -> None:
        """Bound by the real wrapper, and not a halt: it ends a torque controller."""

    def move_to_joint_position(self, q: list[float], speed_factor: float = 1.0) -> bool:
        if self._fail is not None:
            raise self._fail
        self.moves.append((list(q), speed_factor))
        if self._entered is not None:
            self._entered.set()
        if self._block is not None:
            assert self._block.wait(timeout=10), "the blocking motion was never released"
        return self._reached

    def raise_error(self) -> None:
        """Re-raise the error the realtime thread parked, and clear it.

        Clearing matters: the real call moves the parked exception out before
        throwing, so an error can only ever be reported once.
        """
        self.raise_error_calls += 1
        error, self._parked = self._parked, None
        if error is not None:
            raise error

    def close(self) -> None:
        self.closed = True


class FakeGripper:
    """Stands in for ``panda_py.libfranka.Gripper``: one read, one move, one stop.

    ``move`` returns a ``bool`` and ``stop`` exists, both as libfranka declares
    them - the Hand runs its own motion, so it has its own verdict and its own
    halt.
    """

    def __init__(self, hostname: str, *, reached: bool = True) -> None:
        self.hostname = hostname
        self.moves: list[tuple[float, float]] = []
        self.stops = 0
        self.closed = False
        self._reached = reached
        self.state: Any = SimpleNamespace(width=0.037, max_width=0.08)

    def read_once(self) -> Any:
        return self.state

    def move(self, width: float, speed: float) -> bool:
        self.moves.append((width, speed))
        return self._reached

    def stop(self) -> None:
        self.stops += 1

    def close(self) -> None:
        self.closed = True


def _install_fake_panda_py(
    monkeypatch: pytest.MonkeyPatch,
    *,
    panda: FakePanda | None = None,
    with_hand: bool = True,
    connect_error: Exception | None = None,
) -> ModuleType:
    """Install a fake ``panda_py`` in ``sys.modules`` and return it.

    Args:
        monkeypatch: The test's monkeypatch, so the module is removed again.
        panda: The arm object ``Panda(hostname)`` hands back; a fresh one by
            default.
        with_hand: ``False`` makes the Hand refuse its connection, which is an
            arm running with no Franka Hand bolted on.
        connect_error: Raised by ``Panda(hostname)`` when given, standing in for
            an unreachable or locked control box.

    Returns:
        The installed module, carrying ``Panda`` and ``libfranka.Gripper``.
    """
    arm = panda if panda is not None else FakePanda(_HOST)
    hand = FakeGripper(_HOST) if with_hand else None

    def _panda(hostname: str) -> FakePanda:
        if connect_error is not None:
            raise connect_error
        arm.hostname = hostname
        return arm

    def _gripper(hostname: str) -> FakeGripper:
        if hand is None:
            raise OSError("no Franka Hand at this address")
        hand.hostname = hostname
        return hand

    module = ModuleType("panda_py")
    module.Panda = _panda  # type: ignore[attr-defined]
    module.libfranka = SimpleNamespace(Gripper=_gripper)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "panda_py", module)
    return module


def _connected(
    monkeypatch: pytest.MonkeyPatch,
    tool_name: str = "panda",
    *,
    panda: FakePanda | None = None,
    with_hand: bool = True,
    **kwargs: Any,
) -> tuple[FrankaDriver, FakePanda]:
    """Build a driver on a fake FCI link, asserting it connected.

    Returns:
        The connected driver and the arm object it is driving, so a cell can
        assert on what the driver actually asked the arm to do.
    """
    arm = panda if panda is not None else FakePanda(_HOST)
    _install_fake_panda_py(monkeypatch, panda=arm, with_hand=with_hand)
    driver = FrankaDriver(tool_name=tool_name, port=_HOST, **kwargs)
    assert driver.connect_eagerly() is None
    assert driver.is_connected
    return driver, arm


# ============================================================================
# Surface and registration.
# ============================================================================


class TestSurface:
    """The driver satisfies the seam, and the seam can reach it."""

    def test_class_and_instance_satisfy_the_driver_surface(self) -> None:
        """Missing a member registers fine and fails on the first agent call."""
        assert missing_driver_members(FrankaDriver) == ()
        driver = FrankaDriver(tool_name="panda")
        assert isinstance(driver, HardwareDriver)
        assert missing_driver_members(driver) == ()

    @pytest.mark.parametrize("canonical", SUPPORTED_ROBOTS)
    def test_every_supported_robot_registers_and_is_in_the_registry(self, canonical: str) -> None:
        """Both halves of the chain a caller walks.

        Registration alone is not enough: the factory reads the registry before
        it builds anything, so a name this driver claims and ``robots.json``
        omits resolves to the driver and then fails the factory's own lookup.
        """
        assert get_native_driver_class(canonical) is FrankaDriver, (
            f"{canonical!r} resolved to {get_native_driver_class(canonical)}; registered: {list_native_drivers()}"
        )
        assert get_robot(canonical) is not None, f"{canonical!r} is claimed by the driver but absent from robots.json"

    def test_importing_the_driver_does_not_import_the_vendor_binding(self) -> None:
        """``panda_py`` is resolved in a method body, never at module load.

        The binding ships a compiled libfranka and is Linux-x86-only, so a
        module-level import would make this driver - and every
        ``from strands_robots.drivers import ...`` behind it - unimportable on a
        developer laptop and in CI.
        """
        assert "panda_py" not in sys.modules, "a previous test leaked its fake; this cell cannot mean anything"
        import importlib

        importlib.reload(sys.modules["strands_robots.drivers.franka.driver"])
        assert "panda_py" not in sys.modules


# ============================================================================
# The per-arm joint vocabulary.
# ============================================================================


class TestJointVocabulary:
    """Each arm speaks its own model's joint names, not a shared invention."""

    def test_every_supported_arm_has_an_explicit_prefix(self) -> None:
        """A missing entry silently hands an arm the Panda's names.

        :func:`joint_names_for` falls back to the unprefixed spelling for a name
        it does not know, which is the right answer for a caller holding the
        class directly and the *wrong* one for a supported arm - it would accept
        ``joint1`` from an FR3 owner and refuse ``fr3_joint1``, their own
        model's name for the same joint.
        """
        assert set(JOINT_PREFIXES) == set(SUPPORTED_ROBOTS), (
            f"JOINT_PREFIXES covers {sorted(JOINT_PREFIXES)} but the driver serves "
            f"{sorted(SUPPORTED_ROBOTS)} - the difference gets the Panda's joint names by accident"
        )

    @pytest.mark.parametrize(
        ("robot", "first", "last"),
        [
            ("panda", "joint1", "joint7"),
            ("franka", "joint1", "joint7"),  # an alias resolves to the same vocabulary
            ("fr3", "fr3_joint1", "fr3_joint7"),
            ("fr3_v2", "fr3v2_joint1", "fr3v2_joint7"),
        ],
    )
    def test_joint_names_match_the_arms_own_model(self, robot: str, first: str, last: str) -> None:
        """The names are each arm's MuJoCo asset names, so a sim action carries over."""
        names = joint_names_for(robot)
        assert len(names) == DOF
        assert (names[0], names[-1]) == (first, last)

    def test_the_driver_reports_the_vocabulary_it_accepts(self) -> None:
        """A caller must be able to read the names rather than guess a prefix."""
        assert FrankaDriver(tool_name="fr3").joint_names == joint_names_for("fr3")

    def test_the_accepted_action_keys_are_the_joints_plus_the_gripper(self) -> None:
        assert action_keys_for(joint_names_for("fr3")) == (*joint_names_for("fr3"), GRIPPER_KEY)


# ============================================================================
# The FCI cadence.
# ============================================================================


class TestDownsampleStride:
    """The 1 kHz link is read at a cadence a consumer can meet."""

    @pytest.mark.parametrize(
        ("rate", "stride"),
        [(1.0, 1000), (30.0, 33), (100.0, 10), (500.0, 2), (1000.0, 1), (999.0, 1)],
    )
    def test_a_servable_rate_gives_its_stride(self, rate: float, stride: int) -> None:
        assert downsample_stride(rate) == stride

    @pytest.mark.parametrize(
        ("rate", "expected"),
        [
            (1001.0, "exceeds the FCI state rate"),
            (0.0, "stream_rate_hz"),
            (-30.0, "stream_rate_hz"),
            (float("nan"), "stream_rate_hz"),
            (float("inf"), "stream_rate_hz"),
            ("30", "stream_rate_hz"),
        ],
    )
    def test_an_unservable_rate_is_refused_by_name(self, rate: Any, expected: str) -> None:
        """Refused, not clamped: a caller told "fine" who then measures 1 kHz
        has been given a wrong answer about their own deadline."""
        reason = downsample_stride(rate)
        assert isinstance(reason, str) and expected in reason

    def test_a_driver_cannot_be_built_with_an_unservable_cadence(self) -> None:
        """Every read of such a driver would be at a rate it cannot deliver."""
        with pytest.raises(ValueError, match="exceeds the FCI state rate"):
            FrankaDriver(tool_name="panda", stream_rate_hz=5000)

    def test_the_stride_is_reported_beside_the_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A frame stands for a run of ticks, and a consumer must be told which."""
        driver, _ = _connected(monkeypatch, stream_rate_hz=50.0)
        assert driver.downsample_stride == FCI_RATE_HZ // 50
        payload = _invoke(driver, "sensors")
        assert payload["downsample_stride"] == FCI_RATE_HZ // 50
        assert payload["fci_rate_hz"] == FCI_RATE_HZ
        assert payload["stream_rate_hz"] == 50.0


# ============================================================================
# The state decode.
# ============================================================================


class TestDecodeRobotState:
    """A libfranka state becomes joints, velocities, torques and a width."""

    def test_all_three_vectors_are_keyed_by_the_arms_own_names(self) -> None:
        names = joint_names_for("fr3")
        state = SimpleNamespace(q=list(_Q), dq=list(_DQ), tau_J=list(_TAU))
        snapshot = decode_robot_state(state, names, SimpleNamespace(width=0.02, max_width=0.08))
        assert isinstance(snapshot, dict)
        assert snapshot["joints"] == dict(zip(names, _Q, strict=True))
        assert snapshot["velocities"] == dict(zip(names, _DQ, strict=True))
        assert snapshot["torques"] == dict(zip(names, _TAU, strict=True))
        assert (snapshot["gripper_width"], snapshot["gripper_max_width"]) == (0.02, 0.08)

    def test_an_arm_with_no_hand_decodes_to_a_smaller_snapshot(self) -> None:
        """The Hand is a separate FCI device: absent is not a failure."""
        snapshot = decode_robot_state(SimpleNamespace(q=_Q, dq=_DQ, tau_J=_TAU), joint_names_for("panda"))
        assert isinstance(snapshot, dict)
        assert snapshot["gripper_width"] is None and snapshot["gripper_max_width"] is None
        assert len(snapshot["joints"]) == DOF

    @pytest.mark.parametrize(
        ("state", "expected"),
        [
            (SimpleNamespace(dq=_DQ, tau_J=_TAU), "carries no 'q'"),
            (SimpleNamespace(q=_Q, tau_J=_TAU), "carries no 'dq'"),
            (SimpleNamespace(q=_Q, dq=_DQ), "carries no 'tau_J'"),
            (SimpleNamespace(q=_Q[:6], dq=_DQ, tau_J=_TAU), "has 6 values, expected 7"),
            (SimpleNamespace(q=(*_Q, 0.0), dq=_DQ, tau_J=_TAU), "has 8 values, expected 7"),
            (SimpleNamespace(q=0.1, dq=_DQ, tau_J=_TAU), "is float, expected a sequence"),
            (SimpleNamespace(q=(float("nan"), *_Q[1:]), dq=_DQ, tau_J=_TAU), "q[0]"),
        ],
    )
    def test_a_state_that_is_not_one_is_refused_by_name(self, state: Any, expected: str) -> None:
        """A short vector is the failure that matters: mapped positionally it
        attributes one joint's number to another and every consumer believes it."""
        reason = decode_robot_state(state, joint_names_for("panda"))
        assert isinstance(reason, str) and expected in reason

    def test_a_gripper_reporting_a_non_number_is_refused(self) -> None:
        state = SimpleNamespace(q=_Q, dq=_DQ, tau_J=_TAU)
        reason = decode_robot_state(state, joint_names_for("panda"), SimpleNamespace(width=float("inf")))
        assert isinstance(reason, str) and "gripper" in reason


# ============================================================================
# The command gates.
# ============================================================================


class TestActionToTargets:
    """What a Franka command must satisfy before libfranka sees it."""

    def test_a_full_joint_command_becomes_a_vector_in_wire_order(self) -> None:
        names = joint_names_for("panda")
        targets = action_to_targets(dict(zip(names, _Q, strict=True)), names)
        assert targets == (list(_Q), None)

    def test_a_gripper_only_command_is_expressible(self) -> None:
        assert action_to_targets({GRIPPER_KEY: 0.04}, joint_names_for("panda")) == (None, 0.04)

    def test_both_halves_travel_together(self) -> None:
        names = joint_names_for("panda")
        action = {**dict(zip(names, _Q, strict=True)), GRIPPER_KEY: 0.0}
        assert action_to_targets(action, names) == (list(_Q), 0.0)

    def test_a_partial_joint_command_is_refused_rather_than_completed(self) -> None:
        """The gate this driver exists to have.

        FCI commands a whole configuration. Filling the unnamed joints from the
        arm's present pose turns "move joint4" into a seven-joint motion the
        caller never wrote - and it would run at speed, on an arm that can reach
        a person.
        """
        names = joint_names_for("panda")
        reason = action_to_targets({names[3]: -2.0}, names)
        assert isinstance(reason, str)
        assert "names all 7 joints" in reason
        for missing in (names[0], names[6]):
            assert missing in reason

    @pytest.mark.parametrize(
        ("action", "expected"),
        [
            ({}, "non-empty mapping"),
            ({"fr3_joint1": 0.1}, "which this arm does not have"),  # another Franka's vocabulary
            ({"gripper": 0.04}, "which this arm does not have"),
            ({GRIPPER_KEY: float("nan")}, GRIPPER_KEY),
            ({GRIPPER_KEY: "wide"}, GRIPPER_KEY),
        ],
    )
    def test_an_action_this_arm_cannot_take_is_refused_by_name(self, action: Any, expected: str) -> None:
        """Named, never dropped: a silently ignored key is a motion the caller
        believes they commanded."""
        reason = action_to_targets(action, joint_names_for("panda"))
        assert isinstance(reason, str) and expected in reason


# ============================================================================
# Over the fake FCI link.
# ============================================================================


def _invoke(driver: FrankaDriver, action: str) -> Any:
    """Run one agent tool call and return its single content payload."""

    async def _run() -> Any:
        tool_use: ToolUse = {"toolUseId": "t1", "name": driver.tool_name, "input": {"action": action}}
        results = [event async for event in driver.stream(tool_use, {})]
        assert len(results) == 1
        return results[-1]

    result = asyncio.run(_run())
    assert result["toolUseId"] == "t1"
    content = result["content"][0]
    return content.get("json", content.get("text")) if result["status"] == "success" else content["text"]


class TestOverTheLink:
    """Connect, read, command, stop, release - against the fake binding."""

    def test_connect_reads_the_arm_and_the_hand(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, arm = _connected(monkeypatch)
        assert arm.hostname == _HOST
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert (status["connected"], status["gripper"], status["hostname"]) == (True, True, _HOST)
        assert status["joint_names"] == list(joint_names_for("panda"))
        assert status["speed_factor"] == DEFAULT_SPEED_FACTOR

    def test_a_second_connect_does_not_touch_the_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = _connected(monkeypatch)
        monkeypatch.delitem(sys.modules, "panda_py")
        assert driver.connect_eagerly() is None, "an idempotent connect must not re-resolve the binding"

    def test_an_arm_with_no_hand_still_connects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A Hand is optional hardware; refusing the arm for its absence would
        refuse a perfectly good installation."""
        driver, _ = _connected(monkeypatch, with_hand=False)
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["connected"] is True and status["gripper"] is False
        snapshot = driver.read_state()
        assert isinstance(snapshot, dict) and snapshot["gripper_width"] is None

    def test_the_sensors_verb_reports_the_decoded_state(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = _connected(monkeypatch)
        payload = _invoke(driver, "sensors")
        assert payload["joints"] == dict(zip(joint_names_for("panda"), _Q, strict=True))
        assert payload["torques"]["joint7"] == _TAU[-1]
        assert payload["gripper_width"] == 0.037

    def test_the_observation_is_spelled_the_way_a_joint_consumer_reads_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``bus_access`` admits a native driver by this call, and publishes
        ``joints`` from it; a driver without one publishes every other section
        of telemetry and no joints, which reads as a healthy silent arm."""
        from strands_robots.bus_access import joint_read_source

        driver, _ = _connected(monkeypatch)
        assert joint_read_source(driver) is driver
        observation = driver.get_observation()
        assert observation["joint1.pos"] == _Q[0]
        assert observation["joint7.pos"] == _Q[-1]
        assert observation["gripper.pos"] == 0.037

    def test_a_command_reaches_the_guarded_motion_generator(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, arm = _connected(monkeypatch, speed_factor=0.5)
        names = joint_names_for("panda")
        result = driver.send_action({**dict(zip(names, _Q, strict=True)), GRIPPER_KEY: 0.02})
        assert result["status"] == "success"
        assert arm.moves == [(list(_Q), 0.5)], "the arm must be commanded at the driver's speed factor"
        assert driver._gripper.moves == [(0.02, 0.05)], "the hand's speed scales with the same knob"
        assert result["content"][0]["json"]["commanded"]["joints"][names[0]] == _Q[0]

    @pytest.mark.parametrize(
        ("kwargs", "expected"),
        [
            ({"robot_name": "fr3"}, "fronts 'panda' only"),
            ({}, "which this arm does not have"),
        ],
    )
    def test_a_command_that_fails_a_gate_moves_nothing(
        self, monkeypatch: pytest.MonkeyPatch, kwargs: dict[str, Any], expected: str
    ) -> None:
        driver, arm = _connected(monkeypatch)
        result = driver.send_action({"elbow": 0.1}, **kwargs)
        assert result["status"] == "error"
        assert expected in result["content"][0]["text"]
        assert arm.moves == [], "a refused command must not have moved the arm"

    def test_a_gripper_command_with_no_hand_attached_is_refused_by_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, _ = _connected(monkeypatch, with_hand=False)
        result = driver.send_action({GRIPPER_KEY: 0.04})
        assert result["status"] == "error"
        assert "no Franka Hand answered" in result["content"][0]["text"]

    def test_libfrankas_own_refusal_is_reported_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The arm's limits are libfranka's to enforce; its message names the
        limit that was hit, which is more than this driver could establish."""
        arm = FakePanda(_HOST, fail=RuntimeError("joint_position_limits_violation"))
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        result = driver.send_action(dict(zip(names, _Q, strict=True)))
        assert result["status"] == "error"
        assert "joint_position_limits_violation" in result["content"][0]["text"]
        assert "FCI refused the command" in result["content"][0]["text"]

    def test_a_failing_state_read_is_a_reason_not_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mesh reads on its own schedule; an exception here takes the
        publisher down with the arm."""
        arm = FakePanda(_HOST, fail=OSError("link lost"))
        driver, _ = _connected(monkeypatch, panda=arm)
        reason = driver.read_state()
        assert isinstance(reason, str) and "link lost" in reason
        assert driver.get_observation() == {}

    def test_the_stop_verb_and_stop_task_are_the_same_halt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, arm = _connected(monkeypatch)
        assert driver.stop_task()["status"] == "success"
        assert _invoke(driver, "stop") == f"stop_task: {driver.tool_name} motion stopped"
        asyncio.run(driver.stop())
        assert arm.robot.stops == 3, "all three halts go through libfranka's Robot.stop()"

    def test_cleanup_closes_both_devices_and_disconnects(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The control box admits one session: a dropped reference that was
        never closed leaves the arm unreachable to the next process."""
        driver, arm = _connected(monkeypatch)
        hand = driver._gripper
        driver.cleanup()
        assert arm.closed and hand.closed
        assert not driver.is_connected
        assert "not connected" in str(driver.read_state())


class TestWithoutALink:
    """Every path answers, with a reason, when there is no arm to answer it."""

    def test_no_host_names_the_keyword_that_supplies_one(self) -> None:
        reason = FrankaDriver(tool_name="panda").connect_eagerly()
        assert isinstance(reason, str) and "port=" in reason

    def test_a_missing_binding_names_the_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A reason, not a ``ModuleNotFoundError`` through the tool surface."""
        monkeypatch.setitem(sys.modules, "panda_py", None)
        driver = FrankaDriver(tool_name="panda", port=_HOST)
        reason = driver.connect_eagerly()
        assert isinstance(reason, str)
        assert "panda_py" in reason and "pip install panda-py" in reason
        assert not driver.is_connected
        status = asyncio.run(driver.get_status())["content"][0]["json"]
        assert status["connect_error"] == reason

    def test_an_unreachable_control_box_names_what_to_check(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _install_fake_panda_py(monkeypatch, connect_error=OSError("connection refused"))
        driver = FrankaDriver(tool_name="panda", port=_HOST)
        reason = driver.connect_eagerly()
        assert isinstance(reason, str)
        assert "connection refused" in reason and "unlocked in Desk" in reason

    @pytest.mark.parametrize("call", ["send_action", "stop_task", "read_state"])
    def test_every_write_and_read_refuses_while_disconnected(self, call: str) -> None:
        driver = FrankaDriver(tool_name="panda", port=_HOST)
        result = getattr(driver, call)({}) if call == "send_action" else getattr(driver, call)()
        text = result if isinstance(result, str) else result["content"][0]["text"]
        assert "not connected" in text

    def test_stop_on_a_disconnected_driver_is_a_no_op(self) -> None:
        """A caller shutting down an arm that never connected has nothing to do."""
        asyncio.run(FrankaDriver(tool_name="panda").stop())

    def test_cleanup_on_a_disconnected_driver_is_a_no_op(self) -> None:
        FrankaDriver(tool_name="panda").cleanup()


# ============================================================================
# Construction and the policy paths.
# ============================================================================


class TestConstruction:
    """The factory's signature, and the knobs that must be refused early."""

    def test_the_factory_signature_builds_it(self) -> None:
        """``driver_cls(tool_name=, cameras=, data_config=, **kwargs)`` - a driver
        refusing one of the three named keywords is one the factory cannot build."""
        driver = FrankaDriver(tool_name="panda", cameras=None, data_config=None, port=_HOST, unknown_extra=1)
        assert driver.tool_name == "panda" and driver.tool_type == "robot"

    @pytest.mark.parametrize("speed_factor", [0.0, -0.5, 1.5, float("nan"), "fast"])
    def test_an_unusable_speed_factor_is_refused_at_construction(self, speed_factor: Any) -> None:
        """It scales the arm's rated speed, so above 1 asks for a speed the arm
        does not have and at 0 the arm would never arrive."""
        with pytest.raises(ValueError, match="speed_factor"):
            FrankaDriver(tool_name="panda", speed_factor=speed_factor)

    def test_the_tool_spec_declares_only_verbs_that_work(self) -> None:
        """Motion is not an agent verb: seven radians whose safety depends on a
        cell the agent cannot see is not a schema an agent should plan against."""
        spec = FrankaDriver(tool_name="panda").tool_spec
        assert spec["name"] == "panda"
        assert spec["inputSchema"]["json"]["properties"]["action"]["enum"] == ["sensors", "status", "stop"]


class TestPolicyPathsRefuse:
    """No provider emits Franka-shaped actions yet, and the refusal says so."""

    @pytest.mark.parametrize("verb", ["start_task", "run_policy"])
    def test_both_policy_verbs_refuse_naming_the_missing_provider(self, verb: str) -> None:
        """Refused even when handed a policy that works.

        ``run_policy`` gets a real :class:`~strands_robots.policies.MockPolicy`
        rather than a stand-in, so the refusal is about this driver having no
        Franka action space wired - not about the policy being unusable.
        """
        driver = FrankaDriver(tool_name="panda", port=_HOST)
        result = driver.start_task("pick the cube") if verb == "start_task" else driver.run_policy(MockPolicy())
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert text == f"{verb}: {_NO_POLICY_PROVIDER}"
        assert "send_action" in text, "a refusal must name the path that does work"

    def test_task_status_reports_nothing_in_flight_and_why(self) -> None:
        payload = FrankaDriver(tool_name="panda").get_task_status()["content"][0]["json"]
        assert payload["in_flight"] is False
        assert payload["reason"] == _NO_POLICY_PROVIDER


# ============================================================================
# The halt, and the verdict a motion returns. Both are properties of the real
# binding that a fake shaped from the driver's own calls cannot express, so the
# fake above is shaped from the binding and these cells hold it there.
# ============================================================================


class TestTheHaltReachesTheArm:
    """``libfranka``'s ``Robot::stop()`` is the halt, and it must preempt.

    Two separate claims, and the second is the one a fake with an instant motion
    cannot see: the halt has to reach the *right call*, and it has to reach it
    *while the motion is still running*. A halt that lands after the motion it
    was asked to interrupt reports success for an arm that stopped on its own.
    """

    def test_the_wrapper_has_no_halt_of_its_own(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The premise the other cells rest on.

        ``panda_py.Panda`` binds ``stop_controller`` and ``get_robot``; the halt
        is on the object the latter returns. A driver calling ``Panda.stop()``
        raises ``AttributeError`` on a real arm, which no ``except`` on the FCI
        paths catches - so the fake must not offer one either.
        """
        _, arm = _connected(monkeypatch)
        assert not hasattr(arm, "stop"), "the real wrapper has no stop(); the fake must not invent one"
        assert hasattr(arm, "stop_controller"), "it has stop_controller, which ends a controller, not a motion"
        assert hasattr(arm.get_robot(), "stop"), "libfranka's Robot carries the halt"

    def test_the_halt_goes_through_libfrankas_robot(self, monkeypatch: pytest.MonkeyPatch) -> None:
        driver, arm = _connected(monkeypatch)
        assert driver.stop_task()["status"] == "success"
        assert arm.robot.stops == 1

    def test_the_hand_is_halted_with_the_arm(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Hand runs its own motion: an arm that stopped while the fingers
        keep closing is a partial halt."""
        driver, arm = _connected(monkeypatch)
        hand = driver._gripper
        assert driver.stop_task()["status"] == "success"
        assert arm.robot.stops == 1
        assert hand.stops == 1

    def test_a_binding_with_no_get_robot_is_a_reason_not_an_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The agent ``stop`` verb is dispatched through ``stream``, where the
        contract is an error envelope. A binding whose surface differs from the
        one measured here has to be reported rather than raised: a stop request
        that produced a traceback left the arm moving."""
        driver, _ = _connected(monkeypatch)
        # A binding without it: the surface question, not a fault in this object.
        monkeypatch.delattr(FakePanda, "get_robot")
        result = driver.stop_task()
        assert result["status"] == "error"
        assert "get_robot()" in result["content"][0]["text"]
        assert _invoke(driver, "stop").startswith("stop_task: this panda_py binding exposes no get_robot()")

    def test_the_halt_preempts_a_motion_in_flight(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``move_to_joint_position`` runs the whole trajectory before returning.
        The halt must not queue behind it."""
        entered, release = threading.Event(), threading.Event()
        arm = FakePanda(_HOST, block=release, entered=entered)
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        mover = threading.Thread(target=driver.send_action, args=(dict(zip(names, _Q, strict=True)),), daemon=True)
        mover.start()
        try:
            assert entered.wait(timeout=5), "the motion never started"
            started = time.monotonic()
            result = driver.stop_task()
            elapsed = time.monotonic() - started
            assert result["status"] == "success"
            assert arm.robot.stops == 1, "the halt reached the arm while it was moving"
            assert elapsed < 2.0, f"the halt waited {elapsed:.2f}s on the motion instead of preempting it"
            assert mover.is_alive(), "the motion is still in flight, which is the point"
        finally:
            release.set()
            mover.join(timeout=5)

    def test_telemetry_answers_while_the_arm_is_moving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The mesh reads state on its own thread. A read serialized behind the
        motion blanks this arm's joints for the motion's full duration, which
        reads downstream as a healthy arm reporting nothing."""
        entered, release = threading.Event(), threading.Event()
        arm = FakePanda(_HOST, block=release, entered=entered)
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        mover = threading.Thread(target=driver.send_action, args=(dict(zip(names, _Q, strict=True)),), daemon=True)
        mover.start()
        try:
            assert entered.wait(timeout=5), "the motion never started"
            started = time.monotonic()
            snapshot = driver.read_state()
            elapsed = time.monotonic() - started
            assert isinstance(snapshot, dict), f"telemetry stalled behind the motion: {snapshot}"
            assert snapshot["joints"]["joint1"] == pytest.approx(_Q[0])
            assert elapsed < 2.0, f"the read waited {elapsed:.2f}s on the motion"
        finally:
            release.set()
            mover.join(timeout=5)

    def test_shutdown_waits_for_a_motion_rather_than_closing_under_it(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The one path that does wait on the motion lock, and the lock-ordering
        rule that makes it safe: the command path releases the state lock before
        taking the motion lock, so ``cleanup`` may take them in the other order.
        A link closed under a running control loop leaves the control box with a
        session it never ended."""
        entered, release = threading.Event(), threading.Event()
        arm = FakePanda(_HOST, block=release, entered=entered)
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        mover = threading.Thread(target=driver.send_action, args=(dict(zip(names, _Q, strict=True)),), daemon=True)
        mover.start()
        assert entered.wait(timeout=5), "the motion never started"
        closer = threading.Thread(target=driver.cleanup, daemon=True)
        closer.start()
        closer.join(timeout=0.5)
        assert closer.is_alive(), "shutdown must not close the link under a running motion"
        assert not arm.closed
        release.set()
        mover.join(timeout=5)
        closer.join(timeout=5)
        assert not closer.is_alive() and arm.closed
        assert not driver.is_connected


class TestTheMotionsVerdictIsRead:
    """A motion reports its outcome two ways, and neither one raises.

    ``panda_py`` catches ``franka::Exception`` on its realtime thread and parks
    it for ``raise_error()``, then returns a ``bool`` saying whether the arm
    ended within the success threshold of the goal. A driver reading neither
    reports the commanded configuration as though the arm were holding it.
    """

    def test_a_parked_control_error_is_reported_verbatim(self, monkeypatch: pytest.MonkeyPatch) -> None:
        parked = RuntimeError("joint_position_limits_violation: joint4 -2.4000 exceeds -2.3562")
        arm = FakePanda(_HOST, reached=False, parked=parked)
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        result = driver.send_action(dict(zip(names, _Q, strict=True)))
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "joint_position_limits_violation" in text, "libfranka names the limit; the driver cannot"
        assert "FCI refused the command" in text
        assert arm.raise_error_calls == 1, "the parked error has to be collected - it is never thrown at the caller"

    def test_a_parked_error_is_not_charged_to_the_next_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Drained after every motion, so an error is reported against the motion
        that caused it. A driver that only collected on failure would surface a
        parked error against whichever command ran next."""
        arm = FakePanda(_HOST, reached=False, parked=RuntimeError("cartesian_reflex"))
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        action = dict(zip(names, _Q, strict=True))
        assert "cartesian_reflex" in driver.send_action(action)["content"][0]["text"]
        arm._reached = True
        second = driver.send_action(action)
        assert second["status"] == "success", f"the parked error outlived its own motion: {second}"
        assert arm.raise_error_calls == 2

    def test_a_motion_that_ended_short_of_the_goal_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """No exception, no parked error: the motion generator simply reports the
        goal was not met. Reporting success here names joints the arm is not
        holding, which every consumer of the envelope then records as a motion
        that happened."""
        arm = FakePanda(_HOST, reached=False)
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        result = driver.send_action(dict(zip(names, _Q, strict=True)))
        assert result["status"] == "error"
        text = result["content"][0]["text"]
        assert "did not reach the commanded configuration" in text
        assert "joint4" in text, "the refusal names what was asked for"
        assert arm.moves, "the arm did move - the refusal is about where it ended up"

    def test_a_binding_that_reports_nothing_is_not_read_as_a_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The verdict is compared against ``False``, not falsiness. A binding
        whose motion returns nothing has not reported a failure, and treating its
        ``None`` as one would refuse every motion it ever completes."""
        arm = FakePanda(_HOST)
        arm._reached = None  # type: ignore[assignment]
        driver, _ = _connected(monkeypatch, panda=arm)
        names = joint_names_for("panda")
        result = driver.send_action(dict(zip(names, _Q, strict=True)))
        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"]["joints"]["joint1"] == pytest.approx(_Q[0])

    def test_a_hand_that_did_not_reach_its_width_is_refused(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The Hand has its own verdict for the same reason the arm does: it
        reports ``False`` for a width it did not reach, which is what a grasp
        that closed on an object looks like."""
        arm = FakePanda(_HOST)
        _install_fake_panda_py(monkeypatch, panda=arm)
        sys.modules["panda_py"].libfranka.Gripper = lambda host: FakeGripper(host, reached=False)
        driver = FrankaDriver(tool_name="panda", port=_HOST)
        assert driver.connect_eagerly() is None
        result = driver.send_action({GRIPPER_KEY: 0.04})
        assert result["status"] == "error"
        assert "did not reach 0.04 m" in result["content"][0]["text"]
