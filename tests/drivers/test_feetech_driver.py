"""Tests for :class:`strands_robots.drivers.feetech.driver.FeetechDriver`.

Grades the driver's surface, its wired write/read paths, and the refusals that
remain. Nothing here opens a serial port: :class:`FakeServoPort` stands in for
``serial.Serial`` and answers with real status frames, so the codec is
exercised rather than mocked out.

The bus itself is graded in :mod:`test_feetech_bus`; this file grades what the
*driver* adds on top - envelope shapes, the ``.pos`` suffix, the ``bus`` /
``is_connected`` pair the mesh resolves, and the policy verbs still deferred.

The codec is graded separately in :mod:`test_feetech_protocol`. The module-
load pin against ``scservo_sdk`` lives in :mod:`test_feetech_module_load`,
and this file must stay compatible with that pin - importing anything from
:mod:`strands_robots.drivers.feetech` must not pull the vendor SDK.

Structure mirrors :mod:`test_dynamixel_driver` on purpose: a reader who has
read one should recognise the other's test layout on sight. Dynamixel is still
a stub, so its refusals cover verbs this driver now honours.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Any

import numpy as np
import pytest
from strands.types.tools import ToolSpec, ToolUse

from strands_robots.bus_access import bus_lock, joint_read_source, read_joints
from strands_robots.drivers import (
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
)
from strands_robots.drivers.feetech import FeetechDriver
from strands_robots.drivers.feetech.bus import SO_ARM_MOTORS
from strands_robots.drivers.feetech.driver import _NO_POLICY_LOOP, SUPPORTED_ROBOTS
from strands_robots.registry import get_robot
from tests.drivers.conftest import FakeServoPort


def _wired(**kwargs: Any) -> FeetechDriver:
    """A driver whose bus already holds a fake port, as if connected."""
    driver = FeetechDriver(tool_name="so101", port="/dev/fake", **kwargs)
    driver.bus._conn = FakeServoPort(dict.fromkeys((1, 2, 3, 4, 5, 6), 2048))
    return driver


def _port(driver: FeetechDriver) -> FakeServoPort:
    """The fake port behind ``driver``'s bus, narrowed to its recording surface.

    The bus holds its connection as ``Any | None`` because a real one is a
    ``serial.Serial``; asserting the type here is what lets a test read
    ``writes`` without the checker guessing.
    """
    port = driver.bus._conn
    assert isinstance(port, FakeServoPort)
    return port


# ============================================================================
# Surface.
# ============================================================================


class TestSurface:
    """Grade :class:`FeetechDriver` against :data:`DRIVER_SURFACE`.

    A driver that misses a member registered fine and then failed on the
    first agent call, one process and several minutes away from the line
    that was wrong. These tests fail at import time instead.
    """

    def test_driver_class_satisfies_the_driver_surface(self) -> None:
        """Every member :class:`HardwareDriver` requires is present."""
        assert missing_driver_members(FeetechDriver) == ()

    def test_driver_instance_satisfies_the_driver_surface(self) -> None:
        """A constructed instance also satisfies the protocol.

        ``@runtime_checkable`` :class:`HardwareDriver` grades an instance the
        same way ``missing_driver_members`` grades the class; both must agree.
        """
        driver = FeetechDriver(tool_name="so101")
        assert isinstance(driver, HardwareDriver)
        assert missing_driver_members(driver) == ()

    def test_every_supported_robot_registers_the_driver_on_package_import(self) -> None:
        """Importing :mod:`strands_robots.drivers` registers this driver.

        The registration happens through :data:`_SHIPPED_DRIVERS`, and every
        robot :data:`SUPPORTED_ROBOTS` names must resolve to
        :class:`FeetechDriver`. A missing entry surfaces as
        ``Robot("so101", mode="real", driver="strands")`` raising
        ``ValueError`` - the exact failure this driver is here to remove.
        """
        registered = list_native_drivers()
        for canonical in SUPPORTED_ROBOTS:
            cls = get_native_driver_class(canonical)
            assert cls is FeetechDriver, (
                f"canonical={canonical!r} resolved to {cls} rather than FeetechDriver; "
                f"the driver's not-wired refusal cannot reach a caller who cannot build it. "
                f"Currently registered: {registered}"
            )

    def test_every_supported_robot_is_one_the_registry_carries(self) -> None:
        """The other half of the chain the cell above names.

        Resolving to :class:`FeetechDriver` is registration; the factory reads
        the *registry* before it builds anything, so a name this tuple declares
        and ``robots.json`` omits resolves to the driver and then raises
        ``ValueError: Unknown robot`` - the refusal the cell above says this
        driver exists to remove. ``"moss"`` was such a name: it appeared in
        exactly one place in the package, this tuple, with no registry entry
        and no lerobot type.

        Graded tree-wide over every shipped driver by
        ``tests/test_driver_seam.py``; kept here too because this tuple is
        where a name is added.
        """
        unregistered = [name for name in SUPPORTED_ROBOTS if get_robot(name) is None]
        assert not unregistered, (
            f"SUPPORTED_ROBOTS names robots the registry does not carry: {unregistered}. "
            "Each resolves to FeetechDriver and then fails the factory's own lookup, so "
            "the driver advertises a robot no caller can build."
        )


# ============================================================================
# Constructor.
# ============================================================================


class TestConstructor:
    """The constructor accepts what the factory hands it and refuses only
    what the driver knows cannot work."""

    def test_construct_with_the_factory_signature(self) -> None:
        """``driver_cls(tool_name=, cameras=, data_config=, **kwargs)`` works.

        The factory builds every native driver as this signature; a driver
        that refuses one of the three named keywords is a driver the factory
        cannot build. See :class:`~strands_robots.drivers.base.HardwareDriver`
        module docstring for why the constructor contract is not in the
        Protocol itself.
        """
        driver = FeetechDriver(
            tool_name="so101",
            cameras=None,
            data_config=None,
            port="/dev/tty.usbserial-1",
            baud_rate=1_000_000,
            motor_ids=(1, 2, 3, 4, 5, 6),
        )
        assert driver.tool_name == "so101"
        assert driver.tool_type == "robot"

    def test_extras_pass_through_kwargs_are_kept(self) -> None:
        """Unknown keywords are kept for a downstream driver package.

        Refusing every unknown keyword here would refuse a valid future
        extension, and the driver factory has no way to filter them at
        construction time.
        """
        driver = FeetechDriver(tool_name="so101", weird_extension="yes")
        assert driver._extras == {"weird_extension": "yes"}

    def test_ports_multi_bus_is_refused_by_name(self) -> None:
        """A caller passing ``ports=[...]`` gets a named refusal.

        Feetech arms in :data:`SUPPORTED_ROBOTS` are single-bus; a
        multi-bus rig on this family would be a new-family PR, not a
        silent-tolerate here. The message names both keywords and the
        family so a caller reading the traceback knows which decision was
        made.
        """
        with pytest.raises(ValueError) as excinfo:
            FeetechDriver(tool_name="so101", ports=["/dev/a", "/dev/b"])
        assert "port=" in str(excinfo.value)
        assert "multi-bus" in str(excinfo.value)

    def test_baud_rate_default_is_the_feetech_default(self) -> None:
        """The STS3215's factory default baud rate is 1_000_000."""
        driver = FeetechDriver(tool_name="so101")
        assert driver._baud_rate == 1_000_000


# ============================================================================
# Motion, task and policy refusals.
# ============================================================================


class TestWrites:
    """``send_action`` reaches the wire, and refuses what the arm cannot do."""

    def test_send_action_writes_the_commanded_joints(self) -> None:
        """A success envelope names what was commanded, in its unit."""
        driver = _wired()
        result = driver.send_action({"shoulder_pan": 90.0, "gripper": 100.0})
        assert result["status"] == "success"
        body = result["content"][0]["json"]
        assert body["commanded"] == {"shoulder_pan": 90.0, "gripper": 100.0}
        assert "degrees" in body["unit"]

    def test_a_lerobot_pos_suffix_is_accepted(self) -> None:
        """``shoulder_pan.pos`` and ``shoulder_pan`` name the same joint.

        lerobot's action dicts carry the suffix, so a policy's output must
        work unchanged rather than needing a rename at the call site.
        """
        driver = _wired()
        result = driver.send_action({"shoulder_pan.pos": 0.0})
        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"] == {"shoulder_pan": 0.0}

    def test_one_sync_write_frame_carries_the_whole_command(self) -> None:
        """Six joints, one frame - decoded back out of the wire bytes."""
        driver = _wired()
        port = _port(driver)
        port.writes.clear()
        driver.send_action({name: 0.0 for name in SO_ARM_MOTORS})
        (frame,) = port.writes
        assert frame[4] == 0x83  # SYNC_WRITE
        assert frame[5] == 0x2A  # Goal_Position

    @pytest.mark.parametrize(
        ("action", "fragment"),
        [
            ({}, "non-empty mapping"),
            ({"shoulder_lift": 400.0}, "outside range"),
            ({"nope": 0.0}, "unknown motor"),
            ({"gripper": float("nan")}, "must be finite"),
        ],
    )
    def test_a_command_the_arm_cannot_honour_returns_an_error_envelope(
        self, action: dict[str, float], fragment: str
    ) -> None:
        """Refused in an envelope, never raised.

        A driver is invoked as an agent tool; an exception past dispatch is
        not something the caller can handle.
        """
        driver = _wired()
        result = driver.send_action(action)
        assert result["status"] == "error"
        assert fragment in result["content"][0]["text"]

    def test_send_action_without_a_port_names_the_missing_port(self) -> None:
        """No port configured is a caller error, reported not swallowed."""
        result = FeetechDriver(tool_name="so101").send_action({"gripper": 0.0})
        assert result["status"] == "error"
        assert "no port configured" in result["content"][0]["text"]


class TestPolicyRefusals:
    """The policy verbs still refuse - and blame the control loop, not the bus.

    Naming the bus here would send a caller to read serial code that works.
    """

    def test_start_task_refuses_naming_the_policy_loop(self) -> None:
        result = FeetechDriver(tool_name="so101").start_task("pick up the cube")
        assert result == {
            "status": "error",
            "content": [{"text": f"start_task: {_NO_POLICY_LOOP}"}],
        }

    def test_run_policy_refuses_naming_the_policy_loop(self) -> None:
        # ``policy=None`` because the refusal fires before the argument is
        # inspected.
        result = FeetechDriver(tool_name="so101").run_policy(policy=None)  # type: ignore[arg-type]
        assert result == {
            "status": "error",
            "content": [{"text": f"run_policy: {_NO_POLICY_LOOP}"}],
        }

    def test_the_refusal_does_not_blame_the_serial_bus(self) -> None:
        """The bus is wired; a refusal saying otherwise is a stale message."""
        assert "bus" not in _NO_POLICY_LOOP

    def test_get_task_status_reports_nothing_in_flight(self) -> None:
        """Polling task status must not raise; nothing is running."""
        result = FeetechDriver(tool_name="so101").get_task_status()
        assert result["status"] == "success"
        assert result["content"][0]["json"] == {"in_flight": False, "reason": _NO_POLICY_LOOP}

    def test_stop_task_is_a_success_noop(self) -> None:
        """There is nothing to stop; refusing would break idempotent stops."""
        result = FeetechDriver(tool_name="so101").stop_task()
        assert result == {
            "status": "success",
            "content": [{"text": f"stop_task: {_NO_POLICY_LOOP}"}],
        }

    def test_cleanup_releases_the_port_and_is_idempotent(self) -> None:
        """``cleanup()`` closes the bus and completes without raising.

        ``cleanup`` is annotated ``-> None`` (the mesh discards its return
        value), so this test pins the contract by exercising the call and
        letting the type checker confirm no return value leaks. A caller
        that ``assert``s on the returned value would be the bug this
        annotation exists to prevent - see
        https://github.com/strands-labs/robots/pull/2880#discussion for
        the pattern.
        """
        driver = _wired()
        assert driver.is_connected
        driver.cleanup()
        assert not driver.is_connected
        driver.cleanup()  # idempotent; a second call must not raise either

        # A driver that never opened a port is also safe to clean up.
        FeetechDriver(tool_name="so101").cleanup()


# ============================================================================
# Lifecycle and status.
# ============================================================================


class TestLifecycle:
    """Connect / status / stop paths behave as the mesh expects.

    Every field the mesh reads with ``getattr(robot, name, None)`` is either
    absent (fine) or names its unwired state (also fine); nothing here
    pretends to a value that has not been measured.
    """

    def test_connect_eagerly_names_why_it_could_not_open(self) -> None:
        """A failed connection reports the reason as a string.

        Returning ``None`` (which callers read as success) or raising
        (indistinguishable from a real hardware failure mid-session) are both
        worse than the named reason.
        """
        driver = FeetechDriver(tool_name="so101")  # no port
        reason = driver.connect_eagerly()
        assert reason is not None
        assert "no port configured" in reason
        assert driver._connect_error == reason

    def test_connect_eagerly_returns_none_once_the_bus_is_open(self) -> None:
        """``None`` is the success signal, and the error record is cleared."""
        driver = _wired()
        assert driver.connect_eagerly() is None
        assert driver._connect_error is None

    def test_get_status_reports_the_construction_state(self) -> None:
        """Status carries what the driver knows about itself.

        The port, baud rate and motor IDs the caller passed at construction
        show up here; the mesh publishes this envelope as the peer's
        presence.
        """
        driver = FeetechDriver(
            tool_name="so101",
            port="/dev/tty.usbserial-1",
            baud_rate=500_000,
            motor_ids=(1, 2, 3),
        )
        payload = asyncio.run(driver.get_status())
        assert payload["status"] == "success"
        body = payload["content"][0]["json"]
        assert body["tool_name"] == "so101"
        assert body["tool_type"] == "robot"
        assert body["connected"] is False
        assert body["port"] == "/dev/tty.usbserial-1"
        assert body["baud_rate"] == 500_000
        assert body["motor_ids"] == [1, 2, 3]
        assert body["supported_robots"] == list(SUPPORTED_ROBOTS)
        # ``motor_ids`` narrowed the arm, so status reports only those joints.
        assert body["motors"] == {"shoulder_pan": 1, "shoulder_lift": 2, "elbow_flex": 3}

    def test_stop_releases_torque_on_every_motor(self) -> None:
        """``stop`` de-energizes the arm - the whole arm, in one pass."""
        driver = _wired()
        port = _port(driver)
        port.writes.clear()
        asyncio.run(driver.stop())
        assert len(port.writes) == len(SO_ARM_MOTORS)
        for frame in port.writes:
            assert frame[5] == 0x28  # Torque_Enable
            assert frame[6] == 0  # released

    def test_stop_on_a_driver_with_no_port_does_not_raise(self) -> None:
        """``stop`` runs on teardown paths that cannot handle an exception."""
        asyncio.run(FeetechDriver(tool_name="so101").stop())


class TestBusLockParity:
    """Driver-side bus traffic takes the SAME lock as :mod:`bus_access`.

    The bus is half duplex: one frame at a time in either direction. The mesh
    reads joints through :func:`read_joints`, which holds
    :func:`bus_lock` on the driver; an agent commands the arm through the
    driver's own methods. If those do not share that lock, a 30Hz joints
    publisher and a move land on the wire together and both frames are lost -
    the read answers with the tail of the write.
    """

    @pytest.mark.parametrize(
        ("name", "call"),
        [
            ("send_action", lambda d: d.send_action({"elbow_flex": 10.0})),
            ("sensors", lambda d: _run_stream(d, {"toolUseId": "t", "name": "so101", "input": {"action": "sensors"}})),
            ("set_torque", lambda d: _run_stream(d, {"toolUseId": "t", "name": "so101", "input": {"action": "stop"}})),
        ],
    )
    def test_a_bus_path_waits_for_the_lock_a_reader_holds(self, name: str, call: Any) -> None:
        """While a reader holds the lock, the driver writes nothing.

        Deterministic rather than timed: the lock is taken before the worker
        starts and released only once the worker has been observed to make no
        progress, so a driver that ignores the lock fails every run rather
        than on an unlucky interleaving.
        """
        driver = _wired()
        finished = threading.Event()

        def _worker() -> None:
            call(driver)
            finished.set()

        with bus_lock(driver):  # stand in for a mesh-side read_joints
            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()
            assert not finished.wait(timeout=0.5), f"{name} touched the bus while a reader held the lock"
            assert _port(driver).writes == [], f"{name} wrote a frame while a reader held the lock"
        worker.join(timeout=5.0)
        assert finished.is_set(), f"{name} did not proceed once the lock was released"
        assert _port(driver).writes, f"{name} never reached the wire"

    def test_connect_eagerly_waits_for_the_lock_a_reader_holds(self) -> None:
        """The fourth bus path, graded apart because it writes no goal frame.

        ``connect_eagerly`` is the one ``_connect_if_needed`` caller that is not
        already inside a locked block, and opening the port is bus traffic of
        its own. It cannot join the parametrised cases above because those end
        by asserting a frame reached the wire, which this verb never produces -
        so an unlocked ``connect_eagerly`` would otherwise be graded by nothing.
        """
        driver = _wired()
        finished = threading.Event()

        def _worker() -> None:
            driver.connect_eagerly()
            finished.set()

        with bus_lock(driver):  # stand in for a mesh-side read_joints
            worker = threading.Thread(target=_worker, daemon=True)
            worker.start()
            assert not finished.wait(timeout=0.5), "connect_eagerly touched the bus while a reader held the lock"
        worker.join(timeout=5.0)
        assert finished.is_set(), "connect_eagerly did not proceed once the lock was released"

    def test_a_read_through_bus_access_and_a_driver_write_share_one_lock(self) -> None:
        """The lock is keyed on the driver, so both halves serialise.

        Pins the key, not just the presence of a lock: a driver locking some
        other object would pass the wait test above while still colliding
        with :func:`read_joints`.
        """
        driver = _wired()
        assert joint_read_source(driver) is driver
        # Re-entrant on one thread: a caller already holding the lock can read
        # and command without deadlocking itself.
        with bus_lock(driver):
            assert read_joints(driver)
            assert driver.send_action({"elbow_flex": 5.0})["status"] == "success"


class TestJointTelemetry:
    """The ``bus`` / ``is_connected`` pair is how an SO-arm reaches the mesh.

    :func:`strands_robots.bus_access.joint_read_source` resolves a native
    driver by looking for exactly these two members. A driver that satisfies
    every other member of the surface but not these published no ``joints``
    section at all - a real incident (an arm reported zero joints for eleven
    hours), which is why this is pinned rather than assumed.
    """

    def test_the_driver_is_its_own_joint_read_source(self) -> None:
        """No inner device: a native driver owns its bus, so it *is* the source."""
        driver = _wired()
        assert joint_read_source(driver) is driver

    def test_read_joints_returns_pos_suffixed_positions(self) -> None:
        """The shape every joints consumer already parses.

        The suffix is added by ``read_joints``, not by the bus, so this pins
        the seam between the two rather than either one alone.
        """
        driver = _wired()
        joints = read_joints(driver)
        assert set(joints) == {f"{name}.pos" for name in SO_ARM_MOTORS}
        # Midpoint counts (2048 of 4095) sit at the middle of each range.
        assert joints["shoulder_pan.pos"] == pytest.approx(0.0, abs=0.1)
        assert joints["gripper.pos"] == pytest.approx(50.0, abs=0.1)

    def test_is_connected_tracks_the_port(self) -> None:
        """A consumer reads this to tell a live arm from a stale one."""
        driver = FeetechDriver(tool_name="so101", port="/dev/fake")
        assert driver.is_connected is False
        driver.bus._conn = FakeServoPort({1: 2048})
        assert driver.is_connected is True


# ============================================================================
# Agent tool surface (stream).
# ============================================================================


def _run_stream(driver: FeetechDriver, tool_use: ToolUse) -> dict[str, Any]:
    """Drive :meth:`FeetechDriver.stream` and return its one yielded result."""

    async def _collect() -> dict[str, Any]:
        results: list[dict[str, Any]] = []
        async for event in driver.stream(tool_use, {}):
            results.append(event)
        assert len(results) == 1, f"expected one yielded result, got {len(results)}"
        return results[0]

    return asyncio.run(_collect())


class TestStream:
    """The agent-facing ``stream`` yields exactly one result per invocation.

    A schema that declares a verb must accept it in ``stream`` or the agent
    plans against a schema that lies.
    """

    def test_stream_status_reports_the_get_status_envelope(self) -> None:
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-1", "name": "so101", "input": {"action": "status"}},
        )
        assert result["toolUseId"] == "tid-1"
        assert result["status"] == "success"
        # The status envelope is nested inside content[0].json; the outer
        # shape and the inner shape both carry the "success" flag.
        outer_body = result["content"][0]["json"]
        assert outer_body["status"] == "success"
        assert outer_body["content"][0]["json"]["tool_name"] == "so101"

    def test_stream_sensors_reports_the_joint_positions(self) -> None:
        driver = _wired()
        result = _run_stream(
            driver,
            {"toolUseId": "tid-2", "name": "so101", "input": {"action": "sensors"}},
        )
        assert result["toolUseId"] == "tid-2"
        assert result["status"] == "success"
        body = result["content"][0]["json"]
        assert set(body["joint_state"]) == set(SO_ARM_MOTORS)
        assert "degrees" in body["unit"]

    def test_stream_move_to_commands_the_targets(self) -> None:
        driver = _wired()
        result = _run_stream(
            driver,
            {
                "toolUseId": "tid-3",
                "name": "so101",
                "input": {"action": "move_to", "targets": {"elbow_flex": 10.0}},
            },
        )
        assert result["status"] == "success"
        assert result["content"][0]["json"]["commanded"] == {"elbow_flex": 10.0}

    def test_stream_move_to_without_targets_is_refused(self) -> None:
        """An agent firing the verb with no targets gets a reason, not a crash."""
        result = _run_stream(
            _wired(),
            {"toolUseId": "tid-4", "name": "so101", "input": {"action": "move_to"}},
        )
        assert result["status"] == "error"

    @pytest.mark.parametrize(
        ("action_input", "expected"),
        [
            ({"action": "set_torque", "enabled": True}, True),
            ({"action": "set_torque", "enabled": False}, False),
            ({"action": "stop"}, False),  # stop is a release, whatever it is called
        ],
    )
    def test_stream_torque_verbs_report_what_they_set(self, action_input: dict[str, Any], expected: bool) -> None:
        result = _run_stream(_wired(), {"toolUseId": "tid-5", "name": "so101", "input": action_input})
        assert result["status"] == "success"
        assert result["content"][0]["json"] == {"torque_enabled": expected}

    @pytest.mark.parametrize("flag", [True, False])
    def test_a_numpy_boolean_is_honoured_rather_than_refused(self, flag: bool) -> None:
        """A numpy boolean is a boolean, and refusing one strands the operator.

        The check goes through ``boolean_flag_error`` rather than
        ``isinstance(x, bool)`` precisely so this passes: ``np.bool_`` is not a
        subclass of ``bool``, and every action that reaches this driver from a
        policy or an array path carries numpy scalars. Refusing
        ``np.bool_(False)`` would leave a caller with no way to de-energize the
        arm, which is the failure the refusal exists to prevent, inverted.
        """
        numpy_flag = np.bool_(flag)
        assert not isinstance(numpy_flag, bool), "np.bool_ must not be a bool subclass, or this pins nothing"

        result = _run_stream(
            _wired(),
            {"toolUseId": "tid-np", "name": "so101", "input": {"action": "set_torque", "enabled": numpy_flag}},
        )

        assert result["status"] == "success", result
        assert result["content"][0]["json"] == {"torque_enabled": flag}

    @pytest.mark.parametrize("enabled", ["false", "true", "no", 0, 1, None, [], {}])
    def test_a_non_boolean_enabled_is_refused_rather_than_coerced(self, enabled: Any) -> None:
        """``enabled`` is read as a boolean, never coerced into one.

        ``bool("false")`` is ``True``, so coercion turns a request to RELEASE
        an arm into a request to energize it - and answers
        ``torque_enabled: True`` as a success, so nothing downstream can tell.
        Every value here is one an agent emits for a boolean field.
        """
        driver = _wired()
        result = _run_stream(
            driver,
            {"toolUseId": "tid-9", "name": "so101", "input": {"action": "set_torque", "enabled": enabled}},
        )
        assert result["status"] == "error", f"enabled={enabled!r} must be refused, got {result}"
        assert "enabled" in result["content"][0]["text"]
        assert _port(driver).writes == [], f"enabled={enabled!r} reached the wire"

    def test_stream_default_action_is_status(self) -> None:
        """A ``stream`` call without an ``action`` field defaults to status.

        Missing-input tolerance is deliberate: agents sometimes fire empty
        tool calls to discover the schema, and refusing that would give
        them no way in.
        """
        driver = FeetechDriver(tool_name="so101")
        result = _run_stream(
            driver,
            {"toolUseId": "tid-4", "name": "so101", "input": {}},
        )
        assert result["status"] == "success"

    @pytest.mark.parametrize("verb", ["home", "dance", "calibrate", "enable", ""])
    def test_a_verb_the_schema_never_declared_is_refused(self, verb: str) -> None:
        """An undeclared verb is refused, and never lands as a torque release.

        The fallthrough branch used to be ``stop``, so any verb outside the
        schema released torque on every motor and answered ``success``.
        ``home`` is the one an agent reaches for first - this driver
        deliberately does not declare it - and a held payload dropped while
        the transcript read as a clean move.
        """
        driver = _wired()
        result = _run_stream(driver, {"toolUseId": "tid-7", "name": "so101", "input": {"action": verb}})
        assert result["status"] == "error", f"undeclared verb {verb!r} must be refused, not run"
        assert _port(driver).writes == [], f"undeclared verb {verb!r} reached the wire"
        text = result["content"][0]["text"]
        for declared in driver.declared_verbs:
            assert declared in text, f"the refusal must name declared verb {declared!r}: {text}"

    def test_the_refusal_reads_its_verb_list_off_the_schema(self) -> None:
        """The verbs named in a refusal come from ``tool_spec``, not a copy.

        A restated list drifts the first time the schema gains or loses a
        verb, and the agent then corrects itself towards a verb that is not
        there. A driver narrowing or extending its own schema is the case
        that catches the drift.
        """

        class _ExtraVerbDriver(FeetechDriver):
            """A driver whose schema declares one verb more than the base."""

            @property
            def tool_spec(self) -> ToolSpec:
                spec = super().tool_spec
                spec["inputSchema"]["json"]["properties"]["action"]["enum"].append("wiggle")
                return spec

        driver = _ExtraVerbDriver(tool_name="so101", port="/dev/fake")
        assert "wiggle" in driver.declared_verbs
        result = _run_stream(driver, {"toolUseId": "tid-8", "name": "so101", "input": {"action": "home"}})
        assert result["status"] == "error"
        assert "wiggle" in result["content"][0]["text"], (
            f"refusal restated a verb list instead of reading the schema: {result['content'][0]['text']}"
        )

    def test_tool_spec_declares_only_the_verbs_stream_handles(self) -> None:
        """A verb in the schema must have a code path in ``stream``.

        An agent that plans against the schema picks a verb it sees; a
        declared verb the driver refuses is worse than one it does not
        declare at all.
        """
        driver = _wired()
        spec = driver.tool_spec
        declared = set(spec["inputSchema"]["json"]["properties"]["action"]["enum"])
        assert declared == {"status", "sensors", "move_to", "set_torque", "stop"}
        # Every declared verb reaches a code path that answers.
        for verb in declared:
            result = _run_stream(
                _wired(),
                {"toolUseId": f"tid-{verb}", "name": "so101", "input": {"action": verb, "targets": {"gripper": 0.0}}},
            )
            assert result["status"] == "success", f"declared verb {verb} did not answer"
