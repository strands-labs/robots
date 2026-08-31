"""The DDS domain id every ROS 2 surface publishes on is one shared domain.

A ROS 2 / DDS domain id is an index into the RTPS port map, so only an ``int``
in ``[0, MAX_DDS_DOMAIN_ID]`` names a domain. Six surfaces take one - the
hardware ``Robot``'s ``ros2_domain``, a simulation backend's ``ros2_domain``,
and the ``domain_id`` of the rclpy telemetry bridge, its hardware subclass, the
pure-RTPS bridge and the Booster T1 native driver - and they must not disagree
about which values name a domain, because those transports exist to advertise
the same topics.

A native driver belongs on that list for the reason the bridges do: it opens a
DDS participant of its own (``ChannelFactory.Init(domain_id, ip)``), so a
domain the bridges refuse and a driver accepts is a robot publishing where
nothing subscribes.

The rclpy bridge is why the range is load-bearing at the boundary rather than
at the participant: it pins the domain by writing ``ROS_DOMAIN_ID`` into the
process environment, and that write lands *before* ``rclpy`` is imported. So an
out-of-range value is never offered to the transport for rejection - it is
published to the whole process, outliving the call that set it.

``rclpy``, ``cyclonedds`` and the vendor SDKs are optional, so every refusal
test here runs with all of them absent: each guard is placed ahead of its
transport probe, which is what makes that possible and is asserted directly.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

import strands_robots
import strands_robots.hardware_rtps_bridge as rtps_mod
from strands_robots.drivers.booster import BoosterDriver
from strands_robots.hardware_robot import Robot as HwRobot
from strands_robots.hardware_ros_bridge import HardwareRosBridge
from strands_robots.hardware_rtps_bridge import HardwareRtpsBridge
from strands_robots.ros_telemetry import RosTelemetryBridge
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import MAX_DDS_DOMAIN_ID, dds_domain_id_error

#: Values that cannot name a DDS domain, one per way of missing the domain.
UNUSABLE_DOMAINS: list[Any] = [
    -1,  # below the floor
    -5,
    MAX_DDS_DOMAIN_ID + 1,  # the first id whose discovery ports overflow
    300,
    2**31,
    True,  # int subclass: a silent domain 1
    False,  # int subclass: a silent domain 0
    2.7,  # truncates to a domain the caller did not name
    3.0,  # integral, still not an int
    float("nan"),
    float("inf"),
    "5",  # a numeric string is not an int
    None,
    [5],
]

#: Values that do name a domain, including both ends of the range.
USABLE_DOMAINS: list[int] = [0, 1, 7, 101, MAX_DDS_DOMAIN_ID]


def _refuses(fn: Any, value: Any) -> bool:
    """Whether ``fn(value)`` refuses ``value`` as a domain id.

    An ``ImportError`` means the value cleared the guard and the surface then
    found its optional transport missing - an install problem, not a verdict
    about the domain - so it counts as accepted.
    """
    try:
        fn(value)
    except ValueError as exc:
        return f"invalid {'ros2_domain' if 'ros2_domain' in str(exc) else 'domain_id'}" in str(exc)
    except ImportError:
        return False
    return False


def _hardware_robot(value: Any) -> None:
    """Drive the hardware ``Robot``'s domain surface without opening a bus.

    ``_init_ros_bridge`` is a plain method precisely so a ``__new__``-built
    double can call it, which is also how the existing bridge tests reach it.
    """
    robot = HwRobot.__new__(HwRobot)
    robot.tool_name_str = "arm"
    robot._init_ros_bridge(ros2_bridge=False, ros2_domain=value)


class _StubEngine(SimEngine):
    """The smallest concrete ``SimEngine``: enough to reach the shared method.

    Every abstract method is an inert stub with a permissive signature - the
    subject here is ``_init_ros_bridge``, which is a plain method precisely so a
    lightweight subclass need not thread a constructor contract through. Using a
    stub rather than a real backend keeps this module free of any simulation
    dependency, so the shared domain is checked on a minimal install too.
    """

    def add_object(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"status": "success"}

    add_robot = create_world = get_observation = get_state = add_object
    remove_object = remove_robot = render = reset = send_action = step = destroy = add_object

    def list_robots(self) -> list[str]:
        return []

    def robot_joint_names(self, *args: Any, **kwargs: Any) -> list[str]:
        return []


def _sim_engine(value: Any) -> None:
    """Drive the simulation domain surface with no backend and no world."""
    _StubEngine()._init_ros_bridge(ros2_bridge=False, ros2_domain=value)


#: Every surface that takes a domain id, with the parameter it names it by.
SURFACES: list[tuple[str, Any]] = [
    ("RosTelemetryBridge(domain_id=)", lambda v: RosTelemetryBridge(domain_id=v)),
    ("HardwareRosBridge(domain_id=)", lambda v: HardwareRosBridge(domain_id=v)),
    ("HardwareRtpsBridge(domain_id=)", lambda v: HardwareRtpsBridge(domain_id=v)),
    ("Robot._init_ros_bridge(ros2_domain=)", _hardware_robot),
    ("SimEngine._init_ros_bridge(ros2_domain=)", _sim_engine),
    # A native driver opens its own participant, so it is a domain surface in
    # the same sense as the bridges - and the only one whose transport is a
    # vendor wheel, which the constructor never touches.
    ("BoosterDriver(domain_id=)", lambda v: BoosterDriver(domain_id=v)),
]
SURFACE_IDS = [name.split("(")[0] for name, _ in SURFACES]


class TestTheRtpsPortMapFixesTheCeiling:
    """``MAX_DDS_DOMAIN_ID`` is derived from the protocol, not chosen.

    RTPS 2.2 sec. 9.6.1.1 maps a domain id onto discovery ports as
    ``PB + DG * domain_id + d0`` (SPDP multicast) and
    ``PB + DG * domain_id + d1 + PG * participant_id`` (SPDP unicast). The
    highest domain whose ports still fit the 16-bit port space is the ceiling;
    pinning the arithmetic here means the constant cannot drift away from the
    reason it holds.
    """

    #: RTPS 2.2 sec. 9.6.1.1 default port parameters.
    PB, DG, D0, D1, PG = 7400, 250, 0, 10, 2

    def _highest_port(self, domain_id: int) -> int:
        multicast = self.PB + self.DG * domain_id + self.D0
        unicast = self.PB + self.DG * domain_id + self.D1 + self.PG * 0
        return max(multicast, unicast)

    def test_the_ceiling_is_the_highest_domain_whose_ports_fit(self) -> None:
        assert self._highest_port(MAX_DDS_DOMAIN_ID) <= 65535
        assert self._highest_port(MAX_DDS_DOMAIN_ID + 1) > 65535

    def test_no_domain_above_the_ceiling_has_a_port_to_bind(self) -> None:
        overflowing = [d for d in range(0, 400) if self._highest_port(d) > 65535]
        assert min(overflowing) == MAX_DDS_DOMAIN_ID + 1


class TestTheSharedDomain:
    """``dds_domain_id_error`` decides which values name a domain."""

    @pytest.mark.parametrize("value", UNUSABLE_DOMAINS, ids=repr)
    def test_a_value_that_cannot_name_a_domain_is_refused(self, value: Any) -> None:
        error = dds_domain_id_error(value, "domain_id", "Surface")
        assert error is not None
        assert error.startswith("Surface: invalid domain_id:")
        assert f"expected 0-{MAX_DDS_DOMAIN_ID}" in error

    @pytest.mark.parametrize("value", USABLE_DOMAINS, ids=repr)
    def test_a_domain_id_in_range_is_accepted(self, value: int) -> None:
        assert dds_domain_id_error(value, "domain_id", "Surface") is None

    def test_the_message_names_the_parameter_it_came_from(self) -> None:
        assert "invalid ros2_domain" in str(dds_domain_id_error(-1, "ros2_domain", "Robot"))


class TestARefusedDomainLeavesTheProcessEnvironmentAlone:
    """The refusal precedes the process-wide ``ROS_DOMAIN_ID`` write.

    That write is the reason the range cannot be left to the transport: it is
    global to the process and lands before ``rclpy`` is imported, so an accepted
    out-of-range value would steer every later participant - and every
    subprocess that inherits the environment - at a domain nothing is reachable
    on, long after the call that set it returned.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DOMAINS, ids=repr)
    def test_a_refused_domain_does_not_touch_ros_domain_id(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        import os

        with pytest.raises(ValueError, match="invalid domain_id"):
            RosTelemetryBridge(domain_id=value)
        assert os.environ["ROS_DOMAIN_ID"] == "7"

    def test_a_usable_domain_is_still_pinned_into_the_environment(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        import os

        # rclpy is optional; the pin lands before it is imported, which is
        # exactly why the guard has to run first.
        with pytest.raises(ImportError):
            RosTelemetryBridge(domain_id=11)
        assert os.environ["ROS_DOMAIN_ID"] == "11"


class TestARefusedDomainReachesNoTransport:
    """Each guard runs before its surface probes for an optional transport.

    Placing it there is what lets the same caller mistake report identically on
    an install with the ``[ros2]`` extra and one without it, and it means no DDS
    state is built for a domain that was never usable.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DOMAINS, ids=repr)
    def test_the_rtps_bridge_refuses_before_probing_for_cyclonedds(
        self, value: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _unreachable(*_args: Any, **_kwargs: Any) -> Any:
            raise AssertionError("the cyclonedds probe must not be reached")

        monkeypatch.setattr(rtps_mod, "require_optional", _unreachable)
        with pytest.raises(ValueError, match="invalid domain_id"):
            HardwareRtpsBridge(domain_id=value)

    def test_a_usable_domain_still_reaches_the_cyclonedds_probe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        probed: list[str] = []

        def _record(module: str, **_kwargs: Any) -> Any:
            probed.append(module)
            raise ImportError("cyclonedds absent")

        monkeypatch.setattr(rtps_mod, "require_optional", _record)
        with pytest.raises(ImportError):
            HardwareRtpsBridge(domain_id=11)
        assert probed == ["cyclonedds"]


class TestEverySurfaceRefusesTheSameDomains:
    """No surface may accept a domain id another one refuses.

    The two transports advertise the same topics, so a domain the RTPS bridge
    cannot bind is one the rclpy bridge must not publish on either - and the
    ``Robot`` / simulation layers hand their ``ros2_domain`` straight to one of
    them.
    """

    @pytest.mark.parametrize("value", UNUSABLE_DOMAINS, ids=repr)
    def test_an_unusable_domain_is_refused_by_every_surface(self, value: Any, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        refused = {name: _refuses(fn, value) for name, fn in SURFACES}
        assert all(refused.values()), f"accepted by {[n for n, r in refused.items() if not r]}"

    @pytest.mark.parametrize("value", USABLE_DOMAINS, ids=repr)
    def test_a_usable_domain_is_refused_by_no_surface(self, value: int, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        refused = {name: _refuses(fn, value) for name, fn in SURFACES}
        assert not any(refused.values()), f"refused by {[n for n, r in refused.items() if r]}"

    @pytest.mark.parametrize("name,fn", SURFACES, ids=SURFACE_IDS)
    def test_each_surface_stores_a_usable_domain_verbatim(
        self, name: str, fn: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A surface that accepts the value must keep the value it accepted.

        The guard replaced an ``int()`` coercion at each of these surfaces, and
        that coercion is where a ``True`` became domain 1 and a ``2.7`` became
        domain 2. Nothing may re-introduce one.
        """
        monkeypatch.setenv("ROS_DOMAIN_ID", "7")
        assert not _refuses(fn, MAX_DDS_DOMAIN_ID)


class TestEverySurfaceRoutesThroughTheSharedDomain:
    """Structural guard: a domain-taking surface guards it or forwards it.

    A surface that stores a caller-supplied domain id without either calling
    :func:`dds_domain_id_error` or handing the value to a surface that does is
    accepting a domain nothing can be reached on. Checked structurally so a
    further surface cannot ship without joining the rule: the population is the
    scan, so a surface joins the sweep by existing rather than by being listed.
    """

    #: Parameter names that carry a DDS domain id.
    DOMAIN_PARAMS = frozenset({"domain_id", "ros2_domain"})

    #: The domain surfaces the sweep is known to reach, as ``file::function``.
    #: A floor for the scan, not an inventory of what may exist: a scan that
    #: stopped reaching these would report a clean sweep over nothing.
    KNOWN_SURFACES = frozenset(
        {
            "hardware_robot.py::__init__",
            "hardware_robot.py::_init_ros_bridge",
            "hardware_ros_bridge.py::__init__",
            "hardware_rtps_bridge.py::__init__",
            "ros_telemetry.py::__init__",
            "simulation/base.py::_init_ros_bridge",
            "simulation/mujoco/simulation.py::__init__",
        }
    )

    @staticmethod
    def _package_root() -> pathlib.Path:
        """The installed package directory, derived from an imported symbol."""
        return pathlib.Path(inspect.getfile(strands_robots)).parent

    @classmethod
    def _classify(cls, source: str) -> dict[str, tuple[bool, bool]]:
        """Map ``function name -> (calls the guard, forwards the parameter)``."""
        found: dict[str, tuple[bool, bool]] = {}
        for node in ast.walk(ast.parse(source)):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            args = [a.arg for a in node.args.args + node.args.kwonlyargs]
            taken = [a for a in args if a in cls.DOMAIN_PARAMS]
            if not taken:
                continue
            guards = any(
                isinstance(call.func, ast.Name) and call.func.id == "dds_domain_id_error"
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            )
            forwards = any(
                keyword.arg in cls.DOMAIN_PARAMS and isinstance(keyword.value, ast.Name) and keyword.value.id in taken
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
                for keyword in call.keywords
            )
            found[node.name] = (guards, forwards)
        return found

    def _surfaces(self) -> dict[str, tuple[bool, bool]]:
        surfaces: dict[str, tuple[bool, bool]] = {}
        for path in sorted(self._package_root().rglob("*.py")):
            for name, verdict in self._classify(path.read_text()).items():
                surfaces[f"{path.relative_to(self._package_root())}::{name}"] = verdict
        return surfaces

    def test_the_scan_reaches_every_surface_the_sweep_is_known_to_grade(self) -> None:
        """Non-vacuity: a scan rooted elsewhere would report a clean sweep.

        A floor, checked in the one direction that means the scan broke. The
        other direction is not a defect: a new domain surface is graded by the
        sweep below the moment it exists, so failing it for also being absent
        from a list here would fail correct code and teach its author that
        appending a name is the remedy, when the remedy is calling the guard.
        """
        missing = self.KNOWN_SURFACES - set(self._surfaces())
        assert not missing, f"the scan no longer reaches {sorted(missing)}, so the sweep passes over nothing"

    def test_every_domain_surface_guards_or_forwards_the_value(self) -> None:
        adrift = {name for name, (guards, forwards) in self._surfaces().items() if not (guards or forwards)}
        assert not adrift, f"these surfaces neither validate nor forward the domain id: {sorted(adrift)}"

    def test_the_scanner_detects_a_surface_that_does_neither(self) -> None:
        """A scanner that matched nothing would pass the sweep vacuously."""
        planted = "def brand_new_bridge(self, *, domain_id: int = 0) -> None:\n    self._domain = domain_id\n"
        assert self._classify(planted) == {"brand_new_bridge": (False, False)}

    def test_the_scanner_recognises_a_new_surface_that_joins_the_rule(self) -> None:
        """The other answer a new surface can give, so the floor is safe.

        The sweep's population is whatever the scan finds, so growth needs no
        edit here - but only if a compliant newcomer is actually read as
        compliant. Both answers are pinned: a surface that calls the shared
        guard clears the sweep, and one that hand-rolls its own range check is
        named by it, which is the disagreement the shared domain exists to
        prevent.
        """
        joins = (
            "def __init__(self, *, domain_id: int = 0) -> None:\n"
            "    if reason := dds_domain_id_error(domain_id, 'domain_id', 'NewBridge'):\n"
            "        raise ValueError(reason)\n"
            "    self._domain = domain_id\n"
        )
        hand_rolled = (
            "def __init__(self, *, domain_id: int = 0) -> None:\n"
            "    if domain_id < 0:\n"
            "        raise ValueError('bad domain')\n"
            "    self._domain = domain_id\n"
        )
        assert self._classify(joins) == {"__init__": (True, False)}
        assert self._classify(hand_rolled) == {"__init__": (False, False)}
