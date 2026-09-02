"""``ReachyMiniDriver``'s motion arguments are the shared signed-number domain.

The three movement RPCs -- ``look``, ``antennas`` and ``body`` -- are the only
surface in the package that actuates a physical robot, and between them they
carry nine caller-supplied numbers: six pose components in degrees and
millimetres, two antenna angles, one body yaw. Each is interpolated verbatim
into a command dict and handed to the active ``HardwareLink``, and nothing
downstream refuses it:

* Both links put the dict on the wire as JSON. Python's encoder emits ``nan``
  and ``inf`` as the bare tokens ``NaN`` / ``Infinity``, which RFC 8259 does not
  define, so the frame the daemon receives is not valid JSON. It must either
  reject the whole frame -- losing a command the RPC already reported as
  ``success`` -- or parse leniently and hand a non-finite target to the servos.
* ``rpy_to_pose`` is pure trigonometry with no domain of its own, so the same
  unusable argument behaved differently depending on which parameter it landed
  in: an ``inf`` angle raised ``ValueError: math domain error`` out of
  ``math.cos`` and past the ``@rpc`` wrapper, while an ``inf`` offset was
  divided by 1000 and reported as a successful move.

:func:`strands_robots.utils.finite_number_error` is the shared domain for
exactly this, and its docstring names the failure: a non-finite value
"serialize[s] into a wire message as a valid IEEE-754 float64, so the transport
accepts them and the receiving controller integrates them into its state
estimate - a silently poisoned pose rather than a rejected command". Twenty call
sites across ten modules apply it, including the ``drive(linear, angular)`` of
all three mesh bridges; this package applied it nowhere. The constructor's own
``api_port`` guard already argues the placement -- the RPC is where the caller
names the value, so it is the only point a caller can act on.

``TestWhyTheDriverOwnsTheDomain`` pins those premises rather than asserting
them in prose. Per-axis travel is bounded too, through the shared
:func:`~strands_robots.tools.reachy.envelope_error`: this file's original
scope note excused it as depending on hardware the library does not model,
which stopped being true when that envelope landed. What is still the
daemon's is whatever the envelope declares no limit for - ``look``'s
millimetre offsets and the antenna angles - and the cells below hold that
boundary from both sides.

``TestNoMotionRpcDrifts`` widens to every exported driver: an RPC carrying a
number must either validate it or forward it to something that does. The two
``execute`` RPCs take the second route -- ``duration`` reaches ``start_task`` /
``start_policy``, which refuse it and name it -- and ``TestTheDelegationIsReal``
measures that, so the exemption cannot become a hole.
"""

from __future__ import annotations

import ast
import asyncio
import json
import math
from pathlib import Path
from typing import Any

import pytest

import strands_robots
from strands_robots.hardware_robot import Robot as HardwareRobot
from strands_robots.simulation.base import SimEngine
from strands_robots.utils import finite_number_error
from tests.test_reachy_mini_driver import _force_real_device_connect_edge

# Values that cannot be carried to the robot as a signed physical quantity.
# ``True``/``False`` are included because ``bool`` is an ``int`` subclass: a bare
# ``float()`` coercion reads ``True`` as a silent one-degree command.
UNUSABLE_MOTION_VALUES: list[Any] = [
    float("nan"),
    float("inf"),
    float("-inf"),
    True,
    False,
    "30",
    None,
    [30],
    {"deg": 30},
    10**400,
]

# Values a motion RPC must still carry. Every one is inside the tightest axis
# in ``MOTION_ENVELOPE_DEG`` (head pitch/roll, +/-40 deg), because these drive
# every parameter of every RPC including the bounded ones - a value outside
# that would be refused on travel and read as a finiteness regression. The
# envelope's own suite grades the out-of-travel half; a cell there pins this
# constant against the live limits so tightening an axis fails loudly here.
USABLE_MOTION_VALUES: list[Any] = [0, 0.0, -15.0, 15, 32.5, -0.5]

# Every motion RPC, and the parameters it carries, in signature order.
MOTION_SURFACES: dict[str, list[str]] = {
    "look": ["pitch", "roll", "yaw", "x", "y", "z"],
    "antennas": ["left", "right"],
    "body": ["yaw"],
}


@pytest.fixture
def rmd():
    """The reachy_mini_driver module bound to the real device_connect_edge."""
    _force_real_device_connect_edge()
    import strands_robots.device_connect.reachy_mini_driver as module

    return module


class _RecordingLink:
    """A ``HardwareLink`` stand-in that records what reached the wire."""

    def __init__(self) -> None:
        self.commands: list[dict[str, Any]] = []

    async def start(self, on_joints: Any, on_imu: Any) -> None:
        return None

    async def stop(self) -> None:
        return None

    async def send_cmd(self, cmd: dict[str, Any]) -> None:
        self.commands.append(cmd)


def _driver(rmd: Any) -> tuple[Any, _RecordingLink]:
    """A driver with a recording link, built without dialing anything."""
    driver = rmd.ReachyMiniDriver.__new__(rmd.ReachyMiniDriver)
    driver._host = "bot.local"
    driver._prefix = "reachy_mini"
    driver._api_port = 8000
    driver._latest_joints = None
    driver._latest_imu = None
    link = _RecordingLink()
    driver._hw = link
    return driver, link


def _call(driver: Any, rpc_name: str, **kwargs: Any) -> Any:
    """Invoke a motion RPC as an authorized caller would, and report the result.

    Kept as one funnel because the values below are deliberately outside the
    declared ``float`` annotations - splatting them keeps that intent in one
    documented place instead of a suppression at every call site.
    """
    return asyncio.run(getattr(driver, rpc_name)(**kwargs))


class TestMotionArgumentDomain:
    """Each motion RPC accepts exactly what the shared domain accepts."""

    @pytest.mark.parametrize("value", UNUSABLE_MOTION_VALUES, ids=repr)
    @pytest.mark.parametrize("rpc_name", sorted(MOTION_SURFACES))
    def test_an_unusable_argument_is_refused_by_every_motion_rpc(self, rmd, rpc_name, value):
        """A value no link can carry is refused, not sent."""
        for param in MOTION_SURFACES[rpc_name]:
            driver, link = _driver(rmd)
            result = _call(driver, rpc_name, **{param: value})
            assert result["status"] == "error", f"{rpc_name}({param}={value!r}) -> {result}"
            assert link.commands == [], f"{rpc_name}({param}={value!r}) reached the wire"

    @pytest.mark.parametrize("value", UNUSABLE_MOTION_VALUES, ids=repr)
    @pytest.mark.parametrize("rpc_name", sorted(MOTION_SURFACES))
    def test_the_refusal_names_the_rpc_and_the_parameter(self, rmd, rpc_name, value):
        """A caller can act on the reason: it says which argument, and where."""
        for param in MOTION_SURFACES[rpc_name]:
            driver, _link = _driver(rmd)
            reason = _call(driver, rpc_name, **{param: value})["reason"]
            assert reason.startswith(f"{rpc_name}: "), reason
            assert f" {param} " in reason, reason

    @pytest.mark.parametrize("value", USABLE_MOTION_VALUES, ids=repr)
    @pytest.mark.parametrize("rpc_name", sorted(MOTION_SURFACES))
    def test_a_usable_argument_still_reaches_the_robot(self, rmd, rpc_name, value):
        """The finiteness guard is additive over values inside the envelope.

        Scoped deliberately: a value outside an axis's travel is refused now,
        which is the envelope's contract rather than a finiteness regression.
        ``USABLE_MOTION_VALUES`` is bounded so the two cannot be confused.
        """
        for param in MOTION_SURFACES[rpc_name]:
            driver, link = _driver(rmd)
            result = _call(driver, rpc_name, **{param: value})
            assert result["status"] == "success", f"{rpc_name}({param}={value!r}) -> {result}"
            assert len(link.commands) == 1
            assert json.dumps(link.commands[0], allow_nan=False)

    def test_the_accepted_domain_matches_the_shared_helper_exactly(self, rmd):
        """Neither wider nor narrower than :func:`finite_number_error`.

        Graded on ``body``, whose one parameter every value here is inside the
        travel of, so the verdicts that differ can only differ on finiteness.
        """
        for value in UNUSABLE_MOTION_VALUES + USABLE_MOTION_VALUES:
            shared_refuses = finite_number_error(value, "yaw", "body") is not None
            driver, _link = _driver(rmd)
            rpc_refuses = _call(driver, "body", yaw=value)["status"] == "error"
            assert rpc_refuses is shared_refuses, f"verdicts differ for {value!r}"

    def test_the_first_unusable_argument_is_the_one_reported(self, rmd):
        """Deterministic in signature order, so the reason is reproducible."""
        driver, link = _driver(rmd)
        result = _call(driver, "look", roll=float("nan"), y=float("inf"))
        assert result["reason"] == "look: roll must be a finite number, got nan."
        assert link.commands == []


class TestWhyTheDriverOwnsTheDomain:
    """The premises that make the RPC, and not the transport, the right owner."""

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")], ids=repr)
    def test_the_json_encoder_emits_a_token_rfc_8259_does_not_define(self, value):
        """A non-finite value is not refused by the wire format - it corrupts it."""
        encoded = json.dumps({"body_yaw": value})
        assert encoded.split(":")[1].strip(" }") in ("NaN", "Infinity", "-Infinity")
        with pytest.raises(ValueError, match="not JSON compliant"):
            json.dumps({"body_yaw": value}, allow_nan=False)

    def test_a_lenient_parser_accepts_the_corrupt_token_as_a_servo_target(self):
        """The other half of the harm: it may be parsed rather than rejected."""
        assert math.isnan(json.loads("NaN"))

    def test_the_pose_math_has_no_domain_of_its_own(self, rmd):
        """``rpy_to_pose`` is trigonometry: it cannot answer for its inputs."""
        pose = rmd.rpy_to_pose(float("nan"), 0, 0)
        assert all(math.isnan(v) for v in pose[0][:3])

    def test_the_same_bad_value_reached_two_different_failure_modes(self, rmd):
        """Why one shared helper: ``look`` carries six values, not one kind.

        An ``inf`` angle raises out of ``math.cos`` while an ``inf`` offset is a
        finite-looking division, so a per-parameter fix would have left the two
        halves of one signature disagreeing.
        """
        with pytest.raises(ValueError, match="math domain error"):
            rmd.rpy_to_pose(float("inf"), 0, 0)
        leaked = rmd.rpy_to_pose(0, 0, 0, x_mm=float("inf"))
        assert math.isinf(leaked[0][3])

    def test_a_boolean_would_otherwise_be_a_silent_one_degree_command(self):
        """``bool`` is an ``int`` subclass, so a bare coercion honors it."""
        assert math.radians(True) == math.radians(1)


class TestAuthorizationIsAnsweredFirst:
    """An unauthorized caller is told that, not which arguments were bad."""

    @pytest.mark.parametrize("rpc_name", sorted(MOTION_SURFACES))
    def test_an_unauthorized_caller_gets_the_authorization_reason(self, rmd, monkeypatch, rpc_name):
        """The value guard must not answer on behalf of the authz gate."""
        monkeypatch.setenv("DEVICE_CONNECT_RPC_ALLOW", "trusted-*")
        driver, link = _driver(rmd)
        param = MOTION_SURFACES[rpc_name][0]
        result = _call(driver, rpc_name, **{param: float("nan")})
        assert result["status"] == "error"
        assert "not authorized" in result["reason"]
        assert link.commands == []


class TestTheDelegationIsReal:
    """The forwarding surfaces really are answered downstream, not merely shaped so.

    Measured rather than inferred from the call shape: the structural guard above
    exempts a forwarder, and that exemption is only sound while the consumer it
    forwards to still refuses. Both consumers name the parameter in the reason,
    which is what makes the refusal actionable from the RPC's caller.
    """

    @pytest.mark.parametrize("value", [float("nan"), 0, -5, float("inf"), "30", None], ids=repr)
    def test_the_hardware_execute_duration_is_refused_downstream(self, value):
        """``RobotDeviceDriver.execute`` -> ``HardwareRobot.start_task``."""
        import strands_robots.device_connect.robot_driver as robot_driver

        recorded: list[Any] = []

        class _Robot:
            @staticmethod
            def start_task(instruction: str, **kwargs: Any) -> dict[str, Any]:
                recorded.append(kwargs["duration"])
                return {"status": "success"}

        driver = robot_driver.RobotDeviceDriver.__new__(robot_driver.RobotDeviceDriver)
        driver._robot = _Robot()
        asyncio.run(driver.execute("task", duration=value))
        assert recorded == [value], "the value must reach the guarded consumer verbatim"
        assert HardwareRobot._duration_error(value, "start_task") is not None

    @pytest.mark.parametrize("value", [float("nan"), 0, -5, float("inf"), "30", None], ids=repr)
    def test_the_simulation_execute_duration_is_refused_downstream(self, value):
        """``SimulationDeviceDriver.execute`` -> ``SimEngine.start_policy``."""
        import strands_robots.device_connect.sim_driver as sim_driver

        recorded: list[Any] = []

        class _Sim:
            _world = None

            @staticmethod
            def start_policy(**kwargs: Any) -> dict[str, Any]:
                recorded.append(kwargs["duration"])
                return {"status": "success"}

        driver = sim_driver.SimulationDeviceDriver.__new__(sim_driver.SimulationDeviceDriver)
        driver._sim = _Sim()
        asyncio.run(driver.execute("task", duration=value, robot_name="arm"))
        assert recorded == [value], "the value must reach the guarded consumer verbatim"
        assert SimEngine._validate_duration(value, "start_policy", 50.0) is not None


def _exported_names(init_py: Path) -> list[str]:
    """The package's own ``__all__``, read without importing it."""
    for node in ast.parse(init_py.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
            if isinstance(node.value, ast.List | ast.Tuple):
                return [e.value for e in node.value.elts if isinstance(e, ast.Constant) and isinstance(e.value, str)]
    return []


def _rpc_decorated(method: ast.AsyncFunctionDef | ast.FunctionDef) -> bool:
    """Is this method exposed as a Device Connect RPC?"""
    for decorator in method.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name) and target.id == "rpc":
            return True
    return False


def _motion_rpcs(source: str, exported: list[str]) -> dict[str, list[str]]:
    """Exported drivers' RPCs taking a ``float`` parameter, to their names.

    Scoped to ``float``-annotated parameters because those are the values a
    command carries to an actuator; a ``str`` motor-id list is a different
    domain with a different answer.
    """
    found: dict[str, list[str]] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef) or node.name not in exported:
            continue
        for method in node.body:
            if not isinstance(method, ast.AsyncFunctionDef | ast.FunctionDef) or not _rpc_decorated(method):
                continue
            numeric = [
                arg.arg
                for arg in method.args.args + method.args.kwonlyargs
                if arg.arg != "self" and isinstance(arg.annotation, ast.Name) and arg.annotation.id == "float"
            ]
            if numeric:
                found[f"{node.name}.{method.name}"] = numeric
    return found


def _method_node(source: str, qualified: str) -> ast.AsyncFunctionDef | ast.FunctionDef | None:
    """The AST node for ``Class.method``, or ``None`` when absent."""
    class_name, method_name = qualified.split(".")
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for method in node.body:
            if isinstance(method, ast.AsyncFunctionDef | ast.FunctionDef) and method.name == method_name:
                return method
    return None


def _routes_through_the_domain(source: str, qualified: str, params: list[str]) -> bool:
    """Does that RPC either validate its numbers, or hand them to something that does?

    Two ways to satisfy one rule, because two ways exist in the package and both
    are correct:

    * **Validate.** ``ReachyMiniDriver``'s movement RPCs build the command dict
      themselves, so they are the last surface that can answer for the value.
    * **Forward.** ``RobotDeviceDriver.execute`` and
      ``SimulationDeviceDriver.execute`` pass ``duration`` by keyword to
      ``start_task`` / ``start_policy``, which already refuse every unusable
      value and name the parameter doing it. Re-checking here would be a second
      copy of a rule that already has an owner.

    ``TestTheDelegationIsReal`` measures the forwarding rather than trusting the
    shape, so this clause cannot become a hole.
    """
    method = _method_node(source, qualified)
    if method is None:
        return False
    if any(
        isinstance(call.func, ast.Name) and call.func.id == "_motion_domain_error"
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
    ):
        return True
    forwarded = {
        keyword.arg
        for call in ast.walk(method)
        if isinstance(call, ast.Call)
        for keyword in call.keywords
        if keyword.arg in params and isinstance(keyword.value, ast.Name) and keyword.value.id == keyword.arg
    }
    return forwarded == set(params)


class TestNoMotionRpcDrifts:
    """Every exported RPC that commands a number routes it through one domain."""

    @staticmethod
    def _package_dir() -> Path:
        return Path(strands_robots.__file__).parent / "device_connect"

    def _surfaces(self) -> dict[str, tuple[Path, list[str]]]:
        package = self._package_dir()
        exported = _exported_names(package / "__init__.py")
        surfaces: dict[str, tuple[Path, list[str]]] = {}
        for module in sorted(package.rglob("*.py")):
            source = module.read_text(encoding="utf-8")
            for qualified, params in _motion_rpcs(source, exported).items():
                surfaces[qualified] = (module, params)
        return surfaces

    def test_the_scan_finds_every_known_numeric_rpc(self):
        """Non-vacuity: a scan resolving elsewhere would report nothing."""
        expected = {f"ReachyMiniDriver.{rpc}": params for rpc, params in MOTION_SURFACES.items()}
        expected["RobotDeviceDriver.execute"] = ["duration"]
        expected["SimulationDeviceDriver.execute"] = ["duration"]
        assert {name: params for name, (_, params) in self._surfaces().items()} == expected

    def test_every_motion_rpc_validates_its_arguments(self):
        """A future motion RPC cannot command an actuator unvalidated."""
        adrift = {
            name: params
            for name, (module, params) in self._surfaces().items()
            if not _routes_through_the_domain(module.read_text(encoding="utf-8"), name, params)
        }
        assert adrift == {}, f"exported RPCs commanding a number without the shared domain: {adrift}"

    def test_the_domain_is_not_re_implemented_locally(self):
        """One rule: the helper delegates rather than deciding for itself."""
        source = (self._package_dir() / "reachy_mini_driver.py").read_text(encoding="utf-8")
        helper = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name == "_motion_domain_error"
        )
        assert any(
            isinstance(call.func, ast.Name) and call.func.id == "finite_number_error"
            for call in ast.walk(helper)
            if isinstance(call, ast.Call)
        )

    def test_the_scan_detects_a_planted_unguarded_motion_rpc(self):
        """Meta: an empty result must mean clean sources, not a dead scanner."""
        planted = (
            "class Planted:\n"
            "    @rpc()\n"
            "    async def tilt(self, angle: float = 0) -> dict:\n"
            "        await self._send_cmd({'a': angle})\n"
        )
        assert _motion_rpcs(planted, ["Planted"]) == {"Planted.tilt": ["angle"]}
        assert not _routes_through_the_domain(planted, "Planted.tilt", ["angle"])

    def test_the_scan_accepts_a_planted_forwarder(self):
        """Meta: the delegation clause is reachable, so its pin is not vacuous."""
        planted = (
            "class Planted:\n"
            "    @rpc()\n"
            "    async def hold(self, duration: float = 1.0) -> dict:\n"
            "        return self._robot.start_task('t', duration=duration)\n"
        )
        assert _routes_through_the_domain(planted, "Planted.hold", ["duration"])

    def test_a_partial_forwarder_does_not_satisfy_the_rule(self):
        """Forwarding only some of the numbers leaves the rest unanswered."""
        planted = (
            "class Planted:\n"
            "    @rpc()\n"
            "    async def move(self, speed: float = 1.0, tilt: float = 0.0) -> dict:\n"
            "        return self._robot.go(speed=speed)\n"
        )
        assert not _routes_through_the_domain(planted, "Planted.move", ["speed", "tilt"])
