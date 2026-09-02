"""``Robot(mode="real")`` can be told which driver to build, and refuses clearly.

The factory built exactly one thing for a real robot:
``strands_robots.hardware_robot.Robot``, which constructs a lerobot
``RobotConfig`` and wraps a lerobot driver. A robot lerobot does not model - a
humanoid with its own state machine, a rover reporting GPS, a base publishing a
point cloud - had no way in, and nothing recorded which members of that
3000-line class the mesh, the teleop rail and the agent tool surface actually
rely on.

These tests grade the seam that answers both: a ``driver=`` choice with a
precedence order, a contract
(:class:`~strands_robots.drivers.base.HardwareDriver`) the reference
implementation is measured against, and a registration point that refuses a
half-built driver at the line that registers it.

The load-bearing test here is :class:`TestTheLerobotPathIsUnchanged`: the seam is
only worth having if adding it moved nothing. It builds the same robot three ways
- not mentioning ``driver`` at all, ``driver="auto"``, ``driver="lerobot"`` - and
compares the class and the lerobot config all three produce.

Everything runs against a patched lerobot factory, so no serial port is opened
and no arm moves.
"""

from __future__ import annotations

import ast
import importlib
import inspect
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots.drivers as drivers_mod
import strands_robots.drivers.registry as drivers_registry_mod
import strands_robots.registry.robots as registry_robots_mod
from strands_robots import Robot
from strands_robots.drivers import (
    DEFAULT_DRIVER,
    DRIVER_CHOICES,
    DRIVER_SURFACE,
    HardwareDriver,
    get_native_driver_class,
    list_native_drivers,
    missing_driver_members,
    register_native_driver,
    resolve_driver,
    shipped_robot_names,
)
from strands_robots.registry import get_driver, get_robot
from strands_robots.registry.loader import _validate

# A robot every real-mode test builds. Registered, has a lerobot type, and its
# driver comes from the default rather than a declaration.
_ROBOT = "so101"

# A name no registry entry carries, used to drive the two halves of the seam
# apart on purpose. Must stay absent from robots.json for that cell to mean
# anything, which the cell asserts before it relies on it.
_UNREGISTERED_ROBOT = "not_a_registered_robot"


@pytest.fixture(autouse=True)
def _isolate_native_driver_table(monkeypatch: pytest.MonkeyPatch) -> None:
    """Give each test its own empty native-driver table.

    The table is module state that :func:`register_native_driver` writes, so a
    test that registers a driver would otherwise decide what a later test sees -
    and the refusal tests assert on a table being *empty*. The shipped-driver
    registrations that :mod:`strands_robots.drivers.__init__` performs at
    import time (Dynamixel for koch/viperx/widowx/trossen, G1 for
    g1/unitree_g1) are exactly that source of test-order dependence, so this
    fixture resets to an empty dict rather than a copy of whatever is there.
    """
    monkeypatch.setattr(drivers_registry_mod, "_NATIVE_DRIVERS", {})


class _CompleteDriver:
    """A native driver satisfying the whole driver surface.

    Records the keywords the factory forwarded, which is what the constructor
    contract in :mod:`strands_robots.drivers.base` promises a driver receives.
    """

    def __init__(self, tool_name: str, cameras: Any = None, data_config: Any = None, **kwargs: Any) -> None:
        self.forwarded = {"tool_name": tool_name, "cameras": cameras, "data_config": data_config, **kwargs}

    @property
    def tool_name(self) -> str:
        return str(self.forwarded["tool_name"])

    @property
    def tool_type(self) -> str:
        return "robot"

    @property
    def tool_spec(self) -> Any:
        return {"name": self.tool_name, "description": "test driver", "inputSchema": {}}

    def stream(self, tool_use: Any, invocation_state: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def send_action(self, action: Any, robot_name: str | None = None) -> dict[str, Any]:
        return {"status": "success"}

    def start_task(self, instruction: str, **policy_kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def run_policy(self, policy_object: Any, instruction: str = "", **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    def get_task_status(self) -> dict[str, Any]:
        return {"status": "success"}

    def stop_task(self) -> dict[str, Any]:
        return {"status": "success"}

    async def get_status(self) -> dict[str, Any]:
        return {"status": "success"}

    async def stop(self) -> None:
        return None

    def cleanup(self) -> None:
        return None


def _fake_lerobot_device() -> MagicMock:
    """A stand-in for a built lerobot robot, so no serial port is touched."""
    device = MagicMock(name="lerobot_device")
    device.name = "so_follower"
    device.config = MagicMock()
    device.config.cameras = {}
    return device


def _build_real(**factory_kwargs: Any) -> tuple[Any, Any]:
    """Build ``_ROBOT`` in real mode, returning ``(robot, lerobot_config)``.

    The patch target is lerobot's own ``make_robot_from_config``, which leaves
    the whole strands-side config build on the call chain - so the returned
    config is the one the factory's forwarding actually produced.
    """
    with patch("lerobot.robots.utils.make_robot_from_config", return_value=_fake_lerobot_device()) as made:
        robot = Robot(_ROBOT, mode="real", port="/dev/null", **factory_kwargs)
    config = made.call_args.args[0] if made.called else None
    return robot, config


class TestTheLerobotPathIsUnchanged:
    """Adding the seam must not change what a real robot is today."""

    def test_the_three_spellings_of_the_default_build_the_same_robot(self) -> None:
        """Omitting ``driver``, ``"auto"`` and ``"lerobot"`` agree in every detail."""
        pytest.importorskip("lerobot.robots.so_follower")

        readings = []
        for spelling in ({}, {"driver": "auto"}, {"driver": DEFAULT_DRIVER}):
            robot, config = _build_real(**spelling)
            readings.append(
                (
                    type(robot),
                    robot.tool_name,
                    config.id,
                    config.port,
                    type(config).__name__,
                )
            )

        assert readings[0] == readings[1] == readings[2], (
            f"driver= changed what the default path builds: {readings}. The seam is only "
            "safe if a caller who never mentions driver gets today's robot."
        )

    def test_the_default_still_builds_the_lerobot_hardware_class(self) -> None:
        """The class, not merely something duck-shaped like it."""
        pytest.importorskip("lerobot.robots.so_follower")

        from strands_robots.hardware_robot import Robot as HardwareRobotCls

        robot, config = _build_real()
        assert isinstance(robot, HardwareRobotCls)
        assert config is not None, "the lerobot config build must stay on the call chain"

    def test_caller_kwargs_still_reach_the_lerobot_config(self) -> None:
        """``id=`` and friends are forwarded, not swallowed by the new parameter."""
        pytest.importorskip("lerobot.robots.so_follower")

        _, config = _build_real(id="left_arm", use_degrees=True)
        assert config.id == "left_arm"
        assert config.use_degrees is True


class TestDriverResolutionPrecedence:
    """Explicit choice, then the registry, then the default."""

    def _with_declared_driver(self, monkeypatch: pytest.MonkeyPatch, declared: str | None) -> None:
        """Serve a registry in which ``_ROBOT`` declares ``declared``."""
        hardware: dict[str, Any] = {"lerobot_type": "so101_follower"}
        if declared is not None:
            hardware["driver"] = declared
        monkeypatch.setattr(
            registry_robots_mod,
            "_load",
            lambda name: {"robots": {_ROBOT: {"description": "test", "hardware": hardware}}},
        )

    def test_no_declaration_and_no_choice_is_the_default(self) -> None:
        assert get_driver(_ROBOT) is None, f"{_ROBOT} declares no driver, so the default decides"
        assert resolve_driver(_ROBOT) == DEFAULT_DRIVER

    def test_a_registry_declaration_beats_the_default(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._with_declared_driver(monkeypatch, "strands")
        assert get_driver(_ROBOT) == "strands"
        assert resolve_driver(_ROBOT) == "strands"

    def test_an_explicit_choice_beats_a_registry_declaration(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self._with_declared_driver(monkeypatch, "strands")
        assert resolve_driver(_ROBOT, DEFAULT_DRIVER) == DEFAULT_DRIVER

    def test_auto_defers_rather_than_deciding(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``"auto"`` must behave as "unset", not as a choice of its own."""
        self._with_declared_driver(monkeypatch, "strands")
        assert resolve_driver(_ROBOT, "auto") == resolve_driver(_ROBOT) == "strands"

    def test_a_declared_auto_states_no_preference(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A registry saying ``"auto"`` is the same as a registry saying nothing."""
        self._with_declared_driver(monkeypatch, "auto")
        assert resolve_driver(_ROBOT) == DEFAULT_DRIVER

    def test_resolution_never_reports_auto(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Whatever is declared, the answer is a driver a caller can build."""
        for declared in (None, "auto", "lerobot", "strands"):
            self._with_declared_driver(monkeypatch, declared)
            assert resolve_driver(_ROBOT) != "auto"

    @pytest.mark.parametrize("bad", ["lerbot", "strand", "LEROBOT", "", "none"])
    def test_an_unknown_driver_name_is_refused_by_name(self, bad: str) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            resolve_driver(_ROBOT, bad)


class TestTheDriverValueIsCheckedInEveryMode:
    """A typo must not be accepted by whichever branch happens not to read it."""

    def test_sim_mode_refuses_a_driver_that_is_not_a_driver(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            Robot(_ROBOT, mode="sim", driver="lerbot")

    def test_real_mode_refuses_a_driver_that_is_not_a_driver(self) -> None:
        with pytest.raises(ValueError, match="must be one of"):
            Robot(_ROBOT, mode="real", driver="lerbot", port="/dev/null")

    def test_the_refusal_precedes_any_construction(self) -> None:
        """Nothing is built for a call that is going to be refused."""
        with (
            patch("lerobot.robots.utils.make_robot_from_config") as made,
            pytest.raises(ValueError, match="must be one of"),
        ):
            Robot(_ROBOT, mode="real", driver="lerbot", port="/dev/null")
        assert not made.called


class TestAskingForANativeDriverThatIsNotThere:
    """``driver="strands"`` with no driver registered is refused, not substituted."""

    def test_the_refusal_names_the_robot_and_both_remedies(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            Robot(_ROBOT, mode="real", driver="strands", port="/dev/null")

        message = str(excinfo.value)
        assert _ROBOT in message
        assert f"driver='{DEFAULT_DRIVER}'" in message, "a caller needs the working alternative"
        assert "register_native_driver" in message, "and the way to supply what they asked for"

    def test_no_lerobot_robot_is_built_instead(self) -> None:
        """Silently serving the lerobot driver would hide the missing driver."""
        with (
            patch("lerobot.robots.utils.make_robot_from_config") as made,
            pytest.raises(ValueError, match="No native driver"),
        ):
            Robot(_ROBOT, mode="real", driver="strands", port="/dev/null")
        assert not made.called

    def test_the_refusal_reports_which_robots_do_have_one(self) -> None:
        """The discovery surface: what is available, not only what is not."""
        register_native_driver("unitree_g1", _CompleteDriver)
        with pytest.raises(ValueError, match="unitree_g1"):
            Robot(_ROBOT, mode="real", driver="strands", port="/dev/null")


class TestEveryAdvertisedRobotIsOneTheFactoryCanBuild:
    """A driver may only advertise a robot the registry carries.

    Registration and resolution are two halves of one chain. The seam maps a
    canonical name to a driver class; the factory maps that same name to a
    registry entry before it builds anything. A name present in the first and
    absent from the second registers cleanly, reports itself through
    ``list_native_drivers()``, and then raises ``ValueError: Unknown robot`` at
    the call the driver exists to serve -- the exact refusal
    :class:`TestAskingForANativeDriverThatIsNotThere` grades from the other
    side.

    ``FeetechDriver`` shipped naming ``"moss"``, which appeared in exactly one
    place in the package: its own ``SUPPORTED_ROBOTS``. No registry entry, no
    lerobot type, no asset. The driver's own comment above that tuple states
    the invariant this class now measures -- "Every entry corresponds to a
    canonical name in ``strands_robots/registry/robots.json``" -- and its next
    sentence gives the disposition: "registering for a robot we cannot verify
    is a promise this driver does not yet keep."

    Derived from :data:`~strands_robots.drivers._SHIPPED_DRIVERS` through the
    same :func:`~strands_robots.drivers.shipped_robot_names` helper the
    registration itself uses, so the sixth driver is held to this the hour it
    lands rather than inheriting an exemption by being absent from a list.
    """

    @staticmethod
    def _advertised() -> list[tuple[str, str]]:
        """Every ``(driver class, canonical name)`` the shipped table declares."""
        pairs: list[tuple[str, str]] = []
        for module_path, class_name, names in drivers_mod._SHIPPED_DRIVERS:
            module = importlib.import_module(module_path)
            for canonical in shipped_robot_names(module, names):
                pairs.append((class_name, canonical))
        return pairs

    def test_the_derivation_reaches_every_shipped_driver(self) -> None:
        """Non-vacuity: an empty or truncated table would pass silently."""
        pairs = self._advertised()
        classes = {cls for cls, _ in pairs}
        assert len(drivers_mod._SHIPPED_DRIVERS) >= 5, drivers_mod._SHIPPED_DRIVERS
        assert len(classes) == len(drivers_mod._SHIPPED_DRIVERS), classes
        assert len(pairs) >= 10, pairs

    def test_every_advertised_name_resolves_in_the_registry(self) -> None:
        """The relation itself: advertised implies registered."""
        unregistered = [(cls, name) for cls, name in self._advertised() if get_robot(name) is None]
        assert not unregistered, (
            f"these drivers advertise robots the registry does not carry: {unregistered}. "
            "Each one registers cleanly and then raises ValueError('Unknown robot') from "
            "Robot(name, mode='real', driver='strands'). Either register the robot or stop "
            "advertising it."
        )

    def test_an_advertised_name_that_is_not_registered_is_refused_by_the_factory(self) -> None:
        """Why the relation matters: the two halves disagreeing is a hard refusal.

        Registers a complete driver for a name no registry entry carries and
        drives the call the seam exists to serve. The driver resolves; the
        factory refuses. That pair is the failure the relation above prevents.
        """
        register_native_driver(_UNREGISTERED_ROBOT, _CompleteDriver)
        assert get_native_driver_class(_UNREGISTERED_ROBOT) is _CompleteDriver
        assert get_robot(_UNREGISTERED_ROBOT) is None

        with pytest.raises(ValueError, match="Unknown robot"):
            Robot(_UNREGISTERED_ROBOT, mode="real", driver="strands", port="/dev/null")


class TestBuildingARegisteredNativeDriver:
    """The seam's point: a registered driver is what the factory returns."""

    def test_the_factory_returns_the_registered_class(self) -> None:
        register_native_driver(_ROBOT, _CompleteDriver)
        robot = Robot(_ROBOT, mode="real", driver="strands", port="192.168.1.10")
        assert isinstance(robot, _CompleteDriver)

    def test_the_documented_constructor_keywords_arrive(self) -> None:
        register_native_driver(_ROBOT, _CompleteDriver)
        robot = Robot(
            _ROBOT,
            mode="real",
            driver="strands",
            cameras={"wrist": {"type": "opencv", "index_or_path": 0}},
            data_config="so100_dualcam",
            port="192.168.1.10",
        )
        # ``driver="strands"`` is typed as the driver contract, so reading a
        # member of this particular driver needs the narrowing - which is the
        # overload doing its job.
        assert isinstance(robot, _CompleteDriver)
        assert robot.forwarded["tool_name"] == _ROBOT
        assert robot.forwarded["data_config"] == "so100_dualcam"
        assert robot.forwarded["cameras"] == {"wrist": {"type": "opencv", "index_or_path": 0}}
        assert robot.forwarded["port"] == "192.168.1.10", "port stays polymorphic - here an IP"

    def test_the_lerobot_type_is_not_forwarded(self) -> None:
        """``robot=`` names a lerobot type, which means nothing to a native driver."""
        register_native_driver(_ROBOT, _CompleteDriver)
        robot = Robot(_ROBOT, mode="real", driver="strands", port="192.168.1.10")
        assert isinstance(robot, _CompleteDriver)
        assert "robot" not in robot.forwarded

    def test_a_native_driver_joins_the_fleet_like_any_other_robot(self) -> None:
        """The post-build tail is shared, not written once for lerobot.

        A driver reached through the seam is a robot on the fleet: the mesh
        attach and the Device Connect ``.run()`` hook are what put it on the
        network, and a native driver that skipped them would be built, returned,
        and invisible to every peer.
        """
        register_native_driver(_ROBOT, _CompleteDriver)
        # Annotated ``Any`` on purpose: these three are attached to the instance
        # by the factory, so they are on no class the checker can consult - the
        # same reason the production helpers take ``Any``.
        robot: Any = Robot(_ROBOT, mode="real", driver="strands", port="192.168.1.10")

        assert robot._peer_type == "robot", "a real driver is a robot peer, not a sim peer"
        assert robot._peer_id, "every peer needs an identity on the mesh"
        assert callable(robot.run), "``.run()`` is what brings a device online"

    def test_a_registry_declaration_alone_routes_to_the_driver(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """A robot may declare its own driver, with no ``driver=`` at the call."""
        monkeypatch.setattr(
            registry_robots_mod,
            "_load",
            lambda name: {"robots": {_ROBOT: {"description": "t", "hardware": {"driver": "strands"}}}},
        )
        register_native_driver(_ROBOT, _CompleteDriver)
        assert isinstance(Robot(_ROBOT, mode="real", port="192.168.1.10"), _CompleteDriver)


class TestRegisteringADriver:
    """Registration is where a half-built driver is caught."""

    def test_a_driver_missing_members_is_refused_and_they_are_named(self) -> None:
        class Halfway:
            def send_action(self, action: Any, robot_name: str | None = None) -> dict[str, Any]:
                return {}

        with pytest.raises(TypeError) as excinfo:
            register_native_driver(_ROBOT, Halfway)

        message = str(excinfo.value)
        assert "Halfway" in message
        for absent in missing_driver_members(Halfway):
            assert absent in message, f"the refusal must name {absent!r}, which the driver lacks"
        assert get_native_driver_class(_ROBOT) is None, "a refused driver must not be registered"

    def test_a_complete_driver_is_accepted(self) -> None:
        register_native_driver(_ROBOT, _CompleteDriver)
        assert get_native_driver_class(_ROBOT) is _CompleteDriver

    def test_an_alias_and_its_canonical_name_are_one_registration(self) -> None:
        """Otherwise two spellings of one robot could hold two different drivers."""
        register_native_driver("so-101", _CompleteDriver)
        assert get_native_driver_class(_ROBOT) is _CompleteDriver
        assert list(list_native_drivers()) == [_ROBOT]

    def test_a_second_registration_is_refused_unless_asked_for(self) -> None:
        class Other(_CompleteDriver):
            pass

        register_native_driver(_ROBOT, _CompleteDriver)
        with pytest.raises(ValueError, match="already driven by"):
            register_native_driver(_ROBOT, Other)
        assert get_native_driver_class(_ROBOT) is _CompleteDriver

        register_native_driver(_ROBOT, Other, overwrite=True)
        assert get_native_driver_class(_ROBOT) is Other


class TestTheContractIsTheMeasuredOne:
    """:class:`HardwareDriver` describes what the reference implementation has."""

    def test_the_reference_implementation_satisfies_the_whole_surface(self) -> None:
        """Derived, so a member no real robot has cannot be added to the contract."""
        pytest.importorskip("lerobot")
        from strands_robots.hardware_robot import Robot as HardwareRobotCls

        assert missing_driver_members(HardwareRobotCls) == ()

    def test_a_built_hardware_robot_is_a_driver(self) -> None:
        """The ``isinstance`` a caller can run, on a real instance."""
        pytest.importorskip("lerobot.robots.so_follower")

        robot, _ = _build_real()
        assert isinstance(robot, HardwareDriver)

    def test_a_half_built_driver_is_not_one(self) -> None:
        """Non-vacuity: the check must be able to say no."""

        class Halfway:
            def send_action(self, action: Any, robot_name: str | None = None) -> dict[str, Any]:
                return {}

        assert not isinstance(Halfway(), HardwareDriver)

    def test_the_agent_tool_surface_is_part_of_the_contract(self) -> None:
        """Derived from the SDK: a driver an agent cannot invoke is not a driver."""
        from strands.tools.tools import AgentTool

        required = set(AgentTool.__abstractmethods__)
        assert required, "the SDK declares abstract members; the derivation is not vacuous"
        assert required <= set(DRIVER_SURFACE), (
            f"the contract must cover what makes an object callable by an agent, missing "
            f"{sorted(required - set(DRIVER_SURFACE))}"
        )

    def test_the_contract_requires_no_private_member(self) -> None:
        """A public contract naming a private attribute institutionalises it.

        The sensor attributes a mesh publishes (``_pose``, ``_imu`` and their
        siblings) are read with a ``getattr(robot, name, None)`` default, so they
        are optional by construction - a driver with no lidar is complete. A
        Protocol cannot say "optional", so requiring them would refuse a working
        arm.
        """
        assert [name for name in DRIVER_SURFACE if name.startswith("_")] == []

    def test_the_surface_is_derived_from_the_protocol(self) -> None:
        """A hand-written second copy is the one that drifts."""
        declared = {
            node.name
            for node in ast.walk(ast.parse(inspect.getsource(HardwareDriver)))
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and not node.name.startswith("_")
        }
        assert declared == set(DRIVER_SURFACE)


class TestARegistryDeclarationIsValidated:
    """``hardware.driver`` is checked where it is loaded, once."""

    @pytest.mark.parametrize("declared", DRIVER_CHOICES)
    def test_every_advertised_driver_is_accepted(self, declared: str) -> None:
        _validate("robots", {"robots": {"probe": {"hardware": {"lerobot_type": "x", "driver": declared}}}})

    def test_no_declaration_is_accepted(self) -> None:
        _validate("robots", {"robots": {"probe": {"hardware": {"lerobot_type": "x"}}}})

    def test_a_driver_that_does_not_exist_is_refused_naming_the_robot(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _validate("robots", {"robots": {"probe": {"hardware": {"driver": "lerbot"}}}})

        message = str(excinfo.value)
        assert "probe" in message
        assert "lerbot" in message
        for choice in DRIVER_CHOICES:
            assert choice in message, "the refusal must name what is accepted"

    def test_the_package_registry_passes_its_own_check(self) -> None:
        """The shipped registry is loadable, which is what a load asserts."""
        assert get_driver(_ROBOT) is None

    def test_a_user_declared_driver_is_validated_too(self) -> None:
        """The overlay is merged before validation, so a user entry is graded.

        Read off the loader rather than by writing a file: the claim is about the
        *order* of two calls, and an ordering is what an ordering test should
        assert.
        """
        from strands_robots.registry import loader as loader_mod

        source = inspect.getsource(loader_mod._load)
        merge_at = source.index("_merge_user_robots(data")
        validate_at = source.index("_validate(name, data)")
        assert merge_at < validate_at, (
            "the user overlay must be merged before validation, or a driver typo in user_robots.json is never graded"
        )


class TestAKnownDriverWithNoRouteIsRefused:
    """A driver name added to the vocabulary must be routed, not assumed."""

    def test_an_unrouted_driver_name_is_refused_rather_than_built(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import strands_robots.robot as robot_mod

        monkeypatch.setattr(robot_mod, "resolve_driver", lambda canonical, explicit=None: "ros2")
        with pytest.raises(ValueError, match="no route in Robot"):
            Robot(_ROBOT, mode="real", port="/dev/null")


class TestTheSeamIsReachableAsDocumented:
    """The import path the module docstring tells a driver author to use."""

    def test_the_public_names_are_importable_from_the_package(self) -> None:
        import strands_robots.drivers as drivers_pkg

        for name in drivers_pkg.__all__:
            assert hasattr(drivers_pkg, name), f"{name} is exported but absent"

    def test_the_seam_costs_no_heavy_import(self) -> None:
        """``robot.py`` imports the seam at module scope, so it must stay cheap.

        The factory is the surface ``import strands_robots`` defers precisely
        because lerobot, torch and mujoco are expensive. A driver contract that
        dragged one of them in would move that cost back to import time for
        every caller, including the sim-only ones.
        """
        heavy = ("lerobot", "torch", "mujoco", "numpy", "cv2")
        for module_path in sorted(Path(__file__).parent.parent.glob("strands_robots/drivers/*.py")):
            tree = ast.parse(module_path.read_text(encoding="utf-8"))
            for node in tree.body:  # module level only; a lazy import is fine
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.module:
                    names = [node.module]
                for name in names:
                    assert not name.startswith(heavy), (
                        f"{module_path.name} imports {name!r} at module scope; the driver seam "
                        "is imported eagerly by strands_robots.robot and must stay light"
                    )
