"""``gr00t_inference`` refuses a numeric option it cannot honor.

Three agent-supplied numbers reach places that never report back a value they
cannot use. ``port`` and ``denoising_steps`` are interpolated into a
``docker exec`` command line that runs detached, so a value the inference
server's own argument parser rejects surfaces minutes later inside the
container's log rather than as the tool call's result. ``timeout`` bounds the
poll loop that waits for the port to open, where a non-positive budget never
polls once - reporting a service that came up fine as one that "failed to
start" - and a non-finite budget never gives up at all.

The port half is a cross-surface contract rather than a local one: six places
in this package hold a caller-supplied port to ``1 <= port <= 65535``, so
:func:`~strands_robots.utils.tcp_port_error` owns that range and the surfaces
that take a port route through it. The parity and drift tests below are what
keep the domain from diverging again between one transport onto a service and
the next.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path
from typing import Any

import pytest

# The module object rather than its members: the tests below reach
# ``gi.subprocess`` to make a side effect fatal and ``gi.__file__`` to scan the
# shipped source, so the tool's own symbols are qualified through this alias
# instead of being imported a second time by name.
import strands_robots.tools.gr00t_inference as gi
from strands_robots.mesh.rosbridge_robot import RosbridgeRobot
from strands_robots.tools.use_rosbridge import use_rosbridge
from strands_robots.utils import tcp_port_error

# Ports outside the 16-bit TCP port space, or of a type that cannot index it.
# ``0`` asks the kernel for an ephemeral port instead of naming one, and ``True``
# is an ``int`` subclass that a bare range test reads as a silent port 1.
UNUSABLE_PORTS: list[Any] = [0, -1, 70000, 2.7, True, "5555", None, float("nan"), float("inf")]

# A denoising step count is consumed as a discrete ``--denoising-steps`` value,
# so only a true positive ``int`` can be honored.
# Ports every surface must accept. ``65535`` is a legal TCP port and the shared
# owner accepts it, but autobahn's URL builder asserts ``port in range(0, 65535)``
# before ``use_rosbridge`` can connect - a quirk of that transport, not of this
# domain - so the cross-surface list stops at 65534. The rosbridge surfaces now
# refuse 65535 up front for that reason, which is a deliberate divergence from
# the shared owner rather than a drift: it is pinned, with the transport premise
# it rests on, in ``test_rosbridge_transport_port_limit.py``. Adding 65535 here
# would assert the opposite of that file and fail.
USABLE_PORTS: list[Any] = [1, 5555, 65534]

UNUSABLE_STEPS: list[Any] = [0, -1, 2.7, True, "4", None, float("nan"), float("inf")]

# A wait budget in seconds. Fractional is fine (``2.5``); non-positive skips the
# poll loop and non-finite never leaves it.
UNUSABLE_TIMEOUTS: list[Any] = [0, -1, True, "60", None, float("nan"), float("inf")]


def _call(**kwargs: Any) -> dict[str, Any]:
    """Invoke the tool with deliberately off-type values.

    Routed through one ``**kwargs`` funnel because several tests pass values the
    signature's annotations forbid on purpose - that is the input class under
    test - and mypy does not narrow a splatted ``dict[str, Any]``.
    """
    return gi.gr00t_inference(**kwargs)


def _message(result: dict[str, Any]) -> str:
    return str(result.get("message", ""))


@pytest.fixture
def no_side_effects(monkeypatch: pytest.MonkeyPatch) -> dict[str, int]:
    """Fail the test if a refused call reaches docker or a socket.

    A guard that runs after the container work has already started cannot make
    the rejection a property of the request, so every refusal test asserts the
    counters below stayed at zero.
    """
    calls = {"subprocess": 0, "socket": 0}

    def _no_subprocess(*_a: Any, **_kw: Any) -> Any:
        calls["subprocess"] += 1
        raise AssertionError("a refused call reached subprocess.run")

    def _no_socket(_port: Any) -> bool:
        calls["socket"] += 1
        raise AssertionError("a refused call opened a socket")

    monkeypatch.setattr(gi.subprocess, "run", _no_subprocess)
    monkeypatch.setattr(gi, "_is_service_running", _no_socket)
    return calls


class TestPortDomain:
    """A port that cannot address a service is refused, not sent to docker."""

    @pytest.mark.parametrize("action", ["status", "stop", "start", "restart", "start_container"])
    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_an_unusable_port_is_refused_by_every_action_that_reads_one(
        self, action: str, port: Any, no_side_effects: dict[str, int]
    ) -> None:
        result = _call(action=action, port=port, checkpoint_path="/ckpt", hf_repo="org/model")
        assert result["status"] == "error"
        assert "invalid port" in _message(result)
        assert repr(port) in _message(result)
        assert action in _message(result)
        assert no_side_effects == {"subprocess": 0, "socket": 0}

    def test_the_message_names_the_accepted_range(self, no_side_effects: dict[str, int]) -> None:
        message = _message(_call(action="status", port=70000))
        assert "1-65535" in message


class TestDenoisingStepsDomain:
    """A denoising step count the server cannot parse is refused up front."""

    @pytest.mark.parametrize("action", ["start", "restart"])
    @pytest.mark.parametrize("steps", UNUSABLE_STEPS)
    def test_an_unusable_step_count_is_refused(self, action: str, steps: Any, no_side_effects: dict[str, int]) -> None:
        result = _call(action=action, checkpoint_path="/ckpt", denoising_steps=steps)
        assert result["status"] == "error"
        assert "denoising_steps" in _message(result)
        assert repr(steps) in _message(result)
        assert no_side_effects == {"subprocess": 0, "socket": 0}

    @pytest.mark.parametrize("steps", UNUSABLE_STEPS)
    def test_n17_ignores_the_step_count_so_it_is_not_refused(self, steps: Any) -> None:
        """The N1.7 entrypoint takes no ``--denoising-steps`` flag.

        Refusing a value that protocol never reads would be a false rejection,
        so the guard leaves it alone - the call fails (or succeeds) for whatever
        the orchestration itself decides.
        """
        result = _call(action="start", checkpoint_path="/ckpt", protocol="n1.7", denoising_steps=steps)
        assert "denoising_steps must be" not in _message(result)


class TestTimeoutDomain:
    """A wait budget that cannot bound the poll loop is refused."""

    @pytest.mark.parametrize("action", ["start", "restart"])
    @pytest.mark.parametrize("timeout", UNUSABLE_TIMEOUTS)
    def test_an_unusable_timeout_is_refused(self, action: str, timeout: Any, no_side_effects: dict[str, int]) -> None:
        result = _call(action=action, checkpoint_path="/ckpt", timeout=timeout)
        assert result["status"] == "error"
        assert "timeout" in _message(result)
        assert repr(timeout) in _message(result)
        assert no_side_effects == {"subprocess": 0, "socket": 0}

    def test_a_fractional_timeout_is_a_usable_budget(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``2.5`` seconds is a perfectly good wait, so the guard must not refuse it."""
        monkeypatch.setattr(gi, "_find_gr00t_containers", lambda: {"status": "success", "containers": []})
        result = _call(action="start", checkpoint_path="/ckpt", timeout=2.5)
        assert "timeout must be" not in _message(result)

    def test_a_healthy_service_is_no_longer_reported_as_failed_to_start(self, no_side_effects: dict[str, int]) -> None:
        """``timeout=0`` never polls, so the poll loop cannot see a live service.

        The pre-fix result was ``status="error"`` with "failed to start" for a
        container that had started and a port that was open. The budget is
        refused instead, which is the only answer that does not misreport the
        service's state.
        """
        result = _call(action="start", checkpoint_path="/ckpt", timeout=0)
        assert result["status"] == "error"
        assert "failed to start" not in _message(result)
        assert "timeout must be > 0" in _message(result)

    def test_an_unbounded_timeout_never_enters_the_poll_loop(self, no_side_effects: dict[str, int]) -> None:
        """``inf`` satisfies ``elapsed < timeout`` forever, so it must not reach the loop.

        ``no_side_effects`` is the assertion that matters here: reaching the loop
        means calling ``_is_service_running``, which this fixture makes fatal, so
        a regression fails instead of hanging the suite.
        """
        result = _call(action="start", checkpoint_path="/ckpt", timeout=math.inf)
        assert result["status"] == "error"
        assert "timeout must be > 0" in _message(result)


class TestEffectiveOptionsOnly:
    """A caller is never refused for a value the requested action ignores."""

    @pytest.mark.parametrize("action", ["list", "find_containers"])
    def test_an_action_that_reads_no_numeric_option_is_not_refused(
        self, action: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(gi, "_is_service_running", lambda _port: False)
        monkeypatch.setattr(gi, "_find_gr00t_containers", lambda: {"status": "success", "containers": []})
        result = _call(action=action, port=-1, denoising_steps=0, timeout=-1)
        assert "invalid port" not in _message(result)
        assert "denoising_steps must be" not in _message(result)
        assert "timeout must be" not in _message(result)

    def test_lifecycle_teardown_reads_no_numeric_option(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Teardown only removes a container; refusing a port it never opens is wrong."""
        monkeypatch.setattr(gi, "_remove_container", lambda **_kw: {"status": "success", "message": "removed"})
        result = _call(action="lifecycle", lifecycle="teardown", port=-1, denoising_steps=0, timeout=-1)
        assert result["status"] == "success"

    def test_lifecycle_full_reads_every_numeric_option(self, no_side_effects: dict[str, int]) -> None:
        """``lifecycle="full"`` starts a container and waits for the port."""
        result = _call(action="lifecycle", lifecycle="full", hf_repo="org/model", denoising_steps=0)
        assert result["status"] == "error"
        assert "denoising_steps must be" in _message(result)

    def test_the_table_only_names_actions_the_tool_dispatches(self) -> None:
        """Every key is a real action (or ``lifecycle:<phase>``), so nothing is dead."""
        source = Path(gi.__file__).read_text(encoding="utf-8")
        for key in gi._ACTION_NUMERIC_OPTIONS:
            action = key.split(":", 1)[0]
            assert f'action == "{action}"' in source, f"{key} names no dispatched action"


class TestSharedPortDomain:
    """The owner's accepted set is exactly the 16-bit port space."""

    @pytest.mark.parametrize("port", [1, 80, 5555, 65534, 65535])
    def test_the_whole_port_space_is_accepted(self, port: int) -> None:
        assert tcp_port_error(port, "port", "ctx") is None

    @pytest.mark.parametrize("port", UNUSABLE_PORTS)
    def test_nothing_outside_it_is(self, port: Any) -> None:
        assert tcp_port_error(port, "port", "ctx") is not None

    def test_the_message_names_the_surface_and_the_parameter(self) -> None:
        assert tcp_port_error(0, "port", "RosbridgeRobot") == ("RosbridgeRobot: invalid port: 0 (expected 1-65535)")


class TestPortDomainParity:
    """Every surface that takes a caller-supplied port shares one range.

    Parametrized over the shared owner's verdict rather than over a hand-written
    expectation, so the surfaces cannot drift apart from each other or from it.
    """

    @pytest.mark.parametrize("port", [*UNUSABLE_PORTS, *USABLE_PORTS])
    def test_the_tool_agrees_with_the_shared_owner(self, port: Any) -> None:
        owner_refuses = tcp_port_error(port, "port", "status") is not None
        message = _message(_call(action="status", port=port))
        assert ("invalid port" in message) is owner_refuses

    @pytest.mark.parametrize("port", [*UNUSABLE_PORTS, *USABLE_PORTS])
    def test_the_rosbridge_tool_agrees_with_the_shared_owner(self, port: Any) -> None:
        owner_refuses = tcp_port_error(port, "port", "status") is not None
        result = use_rosbridge(**{"action": "status", "host": "127.0.0.1", "port": port, "timeout": 0.05})
        texts = " ".join(block.get("text", "") for block in result["content"])
        assert ("invalid port" in texts) is owner_refuses

    @pytest.mark.parametrize("port", [*UNUSABLE_PORTS, *USABLE_PORTS])
    def test_the_rosbridge_robot_agrees_with_the_shared_owner(self, port: Any) -> None:
        owner_refuses = tcp_port_error(port, "port", "RosbridgeRobot") is not None
        try:
            RosbridgeRobot(
                **{
                    "node_name": "probe",
                    "cmd_vel_topic": "/cmd_vel",
                    "odom_topic": "/odom",
                    "host": "127.0.0.1",
                    "port": port,
                }
            )
            refused = False
        except ValueError as exc:
            refused = "invalid port" in str(exc)
        assert refused is owner_refuses


# The surfaces where a caller or an agent supplies a port: the agent tools and
# the mesh robot bridges. The other places this package mentions the port space
# are a CLI ``argparse`` check and a generic env-var range helper, which have
# their own failure channels and are not caller-facing entry points.
_PORT_TAKING_GLOBS = ("tools/*.py", "mesh/*_robot.py")
_ROUTED_MODULES = {"use_rosbridge.py", "rosbridge_robot.py", "gr00t_inference.py"}


def _port_taking_sources() -> dict[Path, str]:
    root = Path(gi.__file__).resolve().parent.parent
    return {path: path.read_text(encoding="utf-8") for glob in _PORT_TAKING_GLOBS for path in sorted(root.glob(glob))}


def _has_port_range_literal(source: str) -> bool:
    """True when the module spells the 16-bit port range out for itself."""
    return any(isinstance(node, ast.Constant) and node.value == 65535 for node in ast.walk(ast.parse(source)))


class TestPortRangeHasOneOwner:
    """The 16-bit range is not re-implemented beside a caller-facing port."""

    def test_no_caller_facing_module_spells_the_range_out(self) -> None:
        offenders = sorted(p.name for p, src in _port_taking_sources().items() if _has_port_range_literal(src))
        assert offenders == [], f"these modules re-implement the port range: {offenders}"

    def test_the_known_surfaces_route_through_the_shared_owner(self) -> None:
        routed = {p.name for p, src in _port_taking_sources().items() if "tcp_port_error" in src}
        assert _ROUTED_MODULES <= routed, f"missing: {sorted(_ROUTED_MODULES - routed)}"

    def test_the_scan_detects_a_planted_copy(self, tmp_path: Path) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def f(port):\n    return 1 <= port <= 65535\n"
        assert _has_port_range_literal(planted)
        assert not _has_port_range_literal("def f(port):\n    return port > 0\n")
