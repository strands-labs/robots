"""Behavior tests for the MoveIt2 ZMQ sidecar reference implementation.

Exercises :mod:`strands_robots.policies.moveit2.server.zmq_node` without a
ROS 2 environment. The sidecar keeps every ROS / ``moveit_py`` import lazy
(inside ``_build_moveit_py``, ``_plan``, and ``main``), so the module imports
cleanly on a plain venv and the heavy deps are supplied here as light fakes
through ``sys.modules`` / monkeypatch. ``zmq`` and ``msgpack`` are real (they
ship with the ``[moveit2]`` extra); the REP loop runs against a fake socket
that replays a fixed request sequence and then raises ``KeyboardInterrupt``
to break out exactly like a real Ctrl-C.

The wire protocol pinned here matches what the client
(:class:`strands_robots.policies.moveit2.MoveIt2Policy`) speaks: msgpack
request/response with ``ping`` / ``reset`` / ``plan`` endpoints and
``[time_from_start, q0..qN]`` trajectory rows.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

msgpack = pytest.importorskip(
    "msgpack",
    reason="msgpack not installed - pip install 'strands-robots[moveit2]'",
)

from strands_robots.policies.moveit2.server import zmq_node  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes for the ROS / moveit_py surface the sidecar touches.
# ---------------------------------------------------------------------------
class _FakePoint:
    def __init__(self, sec: int, nanosec: int, positions: list[float]) -> None:
        self.time_from_start = types.SimpleNamespace(sec=sec, nanosec=nanosec)
        self.positions = positions


class _FakeComponent:
    """Stand-in for a moveit_py PlanningComponent."""

    def __init__(self, *, plan_points: list[_FakePoint] | None, plan_raises: bool = False) -> None:
        self._plan_points = plan_points
        self._plan_raises = plan_raises
        self.goal: dict[str, Any] = {}
        self.start_state_set = False

    def set_start_state_to_current_state(self) -> None:
        self.start_state_set = True

    def set_goal_state(self, **kwargs: Any) -> None:
        self.goal = kwargs

    def plan(self) -> Any:
        if self._plan_raises:
            raise RuntimeError("ompl exploded")
        if self._plan_points is None:
            return None
        joint_traj = types.SimpleNamespace(points=self._plan_points)
        trajectory = types.SimpleNamespace(joint_trajectory=joint_traj)
        return types.SimpleNamespace(trajectory=trajectory)


class _FakeMoveItPy:
    # ``component`` is duck-typed: the doubles below differ in which moveit_py
    # interaction they fail at, and the sidecar only ever calls the three
    # methods they all provide.
    def __init__(self, component: Any, *, unknown_group: bool = False) -> None:
        self._component = component
        self._unknown_group = unknown_group

    def get_planning_component(self, group: str) -> _FakeComponent:
        if self._unknown_group:
            raise KeyError(group)
        assert self._component is not None  # only None in the unknown_group path
        return self._component


@pytest.fixture(autouse=True)
def fake_geometry_msgs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Install a fake ``geometry_msgs.msg`` exposing ``PoseStamped``.

    ``_plan`` imports ``PoseStamped`` at function entry for every goal branch
    (not just the pose path), so this is autouse for the whole module.
    """

    class _PoseStamped:
        def __init__(self) -> None:
            self.header = types.SimpleNamespace(frame_id="")
            self.pose = types.SimpleNamespace(
                position=types.SimpleNamespace(x=0.0, y=0.0, z=0.0),
                orientation=types.SimpleNamespace(w=0.0, x=0.0, y=0.0, z=0.0),
            )

    geometry_pkg = types.ModuleType("geometry_msgs")
    geometry_msg_mod = types.ModuleType("geometry_msgs.msg")
    geometry_msg_mod.PoseStamped = _PoseStamped  # type: ignore[attr-defined]
    geometry_pkg.msg = geometry_msg_mod  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "geometry_msgs", geometry_pkg)
    monkeypatch.setitem(sys.modules, "geometry_msgs.msg", geometry_msg_mod)


# ---------------------------------------------------------------------------
# _parse_args
# ---------------------------------------------------------------------------
def test_parse_args_defaults_match_client_protocol() -> None:
    args = zmq_node._parse_args([])
    assert args.host == "0.0.0.0"
    assert args.port == 5556  # MoveIt2Policy default port
    assert args.planning_group == "arm"
    assert args.log_level == "INFO"


def test_parse_args_overrides_are_applied() -> None:
    args = zmq_node._parse_args(
        ["--host", "127.0.0.1", "--port", "6000", "--planning-group", "left_arm", "--log-level", "DEBUG"]
    )
    assert args.host == "127.0.0.1"
    assert args.port == 6000
    assert args.planning_group == "left_arm"
    assert args.log_level == "DEBUG"


# ---------------------------------------------------------------------------
# _plan: every branch of the goal/result decision tree
# ---------------------------------------------------------------------------
def test_plan_unknown_planning_group_returns_structured_failure() -> None:
    moveit_py = _FakeMoveItPy(component=None, unknown_group=True)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="nope",
        joint_state=None,
        target_pose=None,
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["trajectory"] == []
    assert resp["status"].startswith("unknown_planning_group:")


def test_plan_missing_goal_is_rejected() -> None:
    moveit_py = _FakeMoveItPy(component=_FakeComponent(plan_points=[]))
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=[0.1, 0.2],  # hint accepted but unused -> exercises debug branch
        target_pose=None,
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"] == "missing_goal:expected_target_pose_or_target_joints"


def test_plan_with_target_joints_succeeds_and_serialises_rows() -> None:
    points = [
        _FakePoint(sec=0, nanosec=0, positions=[0.0, 0.0]),
        _FakePoint(sec=1, nanosec=500_000_000, positions=[0.5, 0.6]),
    ]
    component = _FakeComponent(plan_points=points)
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.5, "j1": 0.6},
        world_update={"depth_topic": "/camera/depth"},  # schema-free, ignored
    )
    assert resp["success"] is True
    assert resp["status"] == "ok"
    assert component.start_state_set is True
    assert component.goal == {"joint_values": {"j0": 0.5, "j1": 0.6}}
    # rows are [time_from_start_seconds, q0, q1]; nanosec folds into seconds.
    assert resp["trajectory"][0] == [0.0, 0.0, 0.0]
    assert resp["trajectory"][1][0] == pytest.approx(1.5)
    assert resp["trajectory"][1][1:] == [0.5, 0.6]


def test_plan_with_target_pose_builds_posestamped() -> None:
    component = _FakeComponent(plan_points=[_FakePoint(sec=2, nanosec=0, positions=[1.0])])
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=[0.1, 0.2, 0.3, 1.0, 0.0, 0.0, 0.0],
        target_joints=None,
        world_update=None,
    )
    assert resp["success"] is True
    pose = component.goal["pose_stamped_msg"]
    assert component.goal["pose_link"] == "end_effector_link"
    assert pose.header.frame_id == "base_link"
    assert (pose.pose.position.x, pose.pose.position.y, pose.pose.position.z) == (0.1, 0.2, 0.3)
    assert pose.pose.orientation.w == 1.0


def test_plan_planner_exception_is_caught() -> None:
    component = _FakeComponent(plan_points=None, plan_raises=True)
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.0},
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"].startswith("planner_exception:")


def test_plan_empty_result_reported() -> None:
    component = _FakeComponent(plan_points=None)  # plan() -> None
    moveit_py = _FakeMoveItPy(component=component)
    resp = zmq_node._plan(
        moveit_py,
        planning_group="arm",
        joint_state=None,
        target_pose=None,
        target_joints={"j0": 0.0},
        world_update=None,
    )
    assert resp["success"] is False
    assert resp["status"] == "planner_returned_empty"


# ---------------------------------------------------------------------------
# _build_moveit_py: builder wiring (moveit_py + config builder faked)
# ---------------------------------------------------------------------------
def test_build_moveit_py_wires_optional_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: dict[str, Any] = {}

    class _Builder:
        def robot_description(self, package: str) -> _Builder:
            calls["robot_description"] = package
            return self

        def moveit_cpp(self, file_path: str) -> _Builder:
            calls["moveit_cpp"] = file_path
            return self

        def to_moveit_configs(self) -> Any:
            return types.SimpleNamespace(to_dict=lambda: {"k": "v"})

    def _ConfigsBuilder(robot_name: str) -> _Builder:
        calls["robot_name"] = robot_name
        return _Builder()

    class _MoveItPy:
        def __init__(self, node_name: str, config_dict: dict) -> None:
            calls["node_name"] = node_name
            calls["config_dict"] = config_dict

        def get_planning_component_names(self) -> list[str]:
            return ["arm"]

    planning_mod = types.ModuleType("moveit.planning")
    planning_mod.MoveItPy = _MoveItPy  # type: ignore[attr-defined]
    moveit_pkg = types.ModuleType("moveit")
    moveit_pkg.planning = planning_mod  # type: ignore[attr-defined]
    configs_mod = types.ModuleType("moveit_configs_utils")
    configs_mod.MoveItConfigsBuilder = _ConfigsBuilder  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "moveit", moveit_pkg)
    monkeypatch.setitem(sys.modules, "moveit.planning", planning_mod)
    monkeypatch.setitem(sys.modules, "moveit_configs_utils", configs_mod)

    args = zmq_node._parse_args(
        ["--robot-description-package", "my_robot_desc", "--moveit-config-package", "my_moveit_cfg"]
    )
    result = zmq_node._build_moveit_py(args)

    assert isinstance(result, _MoveItPy)
    assert calls["robot_name"] == "moveit2_sidecar"
    assert calls["robot_description"] == "my_robot_desc"
    assert calls["moveit_cpp"] == "my_moveit_cfg"
    assert calls["config_dict"] == {"k": "v"}


# ---------------------------------------------------------------------------
# main: the REP dispatch loop (ping / reset / plan / unknown / malformed)
# ---------------------------------------------------------------------------
class _FakeSocket:
    """ZMQ REP socket replaying ``recv_queue`` then raising KeyboardInterrupt."""

    def __init__(self, recv_queue: list[bytes]) -> None:
        self._recv_queue = list(recv_queue)
        self.sent: list[bytes] = []
        self.bound_to: str | None = None
        self.closed = False

    def bind(self, addr: str) -> None:
        self.bound_to = addr

    def recv(self) -> bytes:
        if not self._recv_queue:
            raise KeyboardInterrupt
        return self._recv_queue.pop(0)

    def send(self, data: bytes) -> None:
        self.sent.append(data)

    def close(self) -> None:
        self.closed = True


def _install_fake_zmq_rclpy(monkeypatch: pytest.MonkeyPatch, socket: _FakeSocket) -> dict[str, bool]:
    flags = {"rclpy_init": False, "rclpy_shutdown": False, "ctx_term": False}

    class _Context:
        def socket(self, _kind: Any) -> _FakeSocket:
            return socket

        def term(self) -> None:
            flags["ctx_term"] = True

    zmq_mod = types.ModuleType("zmq")
    zmq_mod.REP = "REP"  # type: ignore[attr-defined]
    zmq_mod.Context = types.SimpleNamespace(instance=lambda: _Context())  # type: ignore[attr-defined]
    rclpy_mod = types.ModuleType("rclpy")
    rclpy_mod.init = lambda: flags.__setitem__("rclpy_init", True)  # type: ignore[attr-defined]
    rclpy_mod.shutdown = lambda: flags.__setitem__("rclpy_shutdown", True)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "zmq", zmq_mod)
    monkeypatch.setitem(sys.modules, "rclpy", rclpy_mod)
    return flags


def test_main_dispatches_endpoints_and_cleans_up(monkeypatch: pytest.MonkeyPatch) -> None:
    requests = [
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
        msgpack.packb({"endpoint": "reset", "data": {"options": {"seed": 7}}}, use_bin_type=True),
        msgpack.packb(
            {"endpoint": "plan", "data": {"planning_group": "arm", "target_joints": {"j0": 0.1}}},
            use_bin_type=True,
        ),
        msgpack.packb({"endpoint": "bogus"}, use_bin_type=True),
        b"\xc1",  # invalid msgpack -> malformed_request branch
    ]
    socket = _FakeSocket(requests)
    flags = _install_fake_zmq_rclpy(monkeypatch, socket)

    component = _FakeComponent(plan_points=[_FakePoint(sec=0, nanosec=0, positions=[0.1])])
    monkeypatch.setattr(zmq_node, "_build_moveit_py", lambda args: _FakeMoveItPy(component=component))

    rc = zmq_node.main(["--port", "5599"])

    assert rc == 0
    assert flags["rclpy_init"] and flags["rclpy_shutdown"] and flags["ctx_term"]
    assert socket.bound_to == "tcp://0.0.0.0:5599"
    assert socket.closed is True

    responses = [msgpack.unpackb(s, raw=False) for s in socket.sent]
    assert responses[0] == {"status": "ok"}  # ping
    assert responses[1] == {"status": "ok"}  # reset no-op
    assert responses[2]["success"] is True  # plan
    assert responses[2]["trajectory"] == [[0.0, 0.1]]
    assert responses[3] == {"error": "unknown_endpoint:bogus"}
    assert "malformed_request" in responses[4]["error"]


def test_main_returns_1_when_moveit_construction_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    socket = _FakeSocket([])
    flags = _install_fake_zmq_rclpy(monkeypatch, socket)

    def _boom(args: Any) -> Any:
        raise RuntimeError("no ros sourced")

    monkeypatch.setattr(zmq_node, "_build_moveit_py", _boom)

    rc = zmq_node.main([])

    assert rc == 1
    assert flags["rclpy_init"] is True
    assert flags["rclpy_shutdown"] is True
    # Socket was never bound because construction failed first.
    assert socket.bound_to is None


# ---------------------------------------------------------------------------
# A failing request is reported, not fatal.
#
# ``_plan``'s contract is that a failing plan comes back as
# ``{"success": False, "status": ...}``. REQ/REP is lockstep, so an exception
# escaping the handler does not merely lose one answer: it unwinds the REP
# loop, closes the socket with the in-flight reply never sent, and ends the
# sidecar process - so one bad request from one client stops planning for
# every client. ``component.plan()`` was the only guarded interaction; the
# start-state read, the goal set and the trajectory serialisation were not,
# and a joint name the planning group does not have reaches the goal set
# through the client's own validation (which checks name syntax, not
# existence).
# ---------------------------------------------------------------------------
class _StageFailingComponent:
    """PlanningComponent that fails at exactly one moveit_py interaction.

    ``raise_at`` selects the interaction that raises (``"start_state"`` or
    ``"goal"``); ``bad_trajectory`` instead returns a plan result whose
    trajectory does not serialise, which is how a moveit_py version skew
    surfaces. ``positions`` overrides the joint values so a non-numeric entry
    can be fed to the serialiser.
    """

    def __init__(
        self,
        *,
        raise_at: str | None = None,
        exc: Exception | None = None,
        bad_trajectory: bool = False,
        positions: list[Any] | None = None,
    ) -> None:
        self._raise_at = raise_at
        self._exc = exc or RuntimeError("moveit_py refused")
        self._bad_trajectory = bad_trajectory
        self._positions = positions if positions is not None else [0.1]

    def set_start_state_to_current_state(self) -> None:
        if self._raise_at == "start_state":
            raise self._exc

    def set_goal_state(self, **kwargs: Any) -> None:
        if self._raise_at == "goal":
            raise self._exc

    def plan(self) -> Any:
        if self._bad_trajectory:
            return types.SimpleNamespace(trajectory=None)
        point = _FakePoint(sec=0, nanosec=0, positions=self._positions)
        joint_traj = types.SimpleNamespace(points=[point])
        return types.SimpleNamespace(trajectory=types.SimpleNamespace(joint_trajectory=joint_traj))


def _plan_or_fail(moveit_py: Any, **goal: Any) -> dict[str, Any]:
    """Call ``_plan``, converting an escaping exception into a named failure."""
    payload: dict[str, Any] = {
        "planning_group": "arm",
        "joint_state": None,
        "target_pose": None,
        "target_joints": None,
        "world_update": None,
    }
    payload.update(goal)
    try:
        return zmq_node._plan(moveit_py, **payload)
    except Exception as exc:
        raise AssertionError(
            f"_plan raised {type(exc).__name__}: {exc} instead of reporting the failure. "
            "The exception unwinds the REP loop, so the sidecar closes its socket with "
            "the peer's reply never sent and stops serving every other client."
        ) from exc


# (raise_at, exception, goal kwargs, expected status prefix)
_UNGUARDED_FAILURE_SOURCES = [
    pytest.param(
        {"raise_at": "start_state", "exc": RuntimeError("no robot state received yet")},
        {"target_joints": {"j0": 0.1}},
        "start_state_error:",
        id="start-state-read-raises",
    ),
    pytest.param(
        {"raise_at": "goal", "exc": RuntimeError("Joint 'elbo' not found in group 'arm'")},
        {"target_joints": {"elbo": 0.1}},
        "invalid_goal:",
        id="joint-name-not-in-group",
    ),
    pytest.param(
        {"raise_at": "goal", "exc": RuntimeError("link 'end_effector_link' does not exist")},
        {"target_pose": [0.3, 0.0, 0.4, 1.0, 0.0, 0.0, 0.0]},
        "invalid_goal:",
        id="pose-link-unresolvable",
    ),
    pytest.param(
        {},
        {"target_pose": [0.3, 0.0, 0.4]},
        "invalid_goal:",
        id="target-pose-not-seven-values",
    ),
    pytest.param(
        {"bad_trajectory": True},
        {"target_joints": {"j0": 0.1}},
        "trajectory_error:",
        id="trajectory-shape-differs",
    ),
    pytest.param(
        {"positions": ["not-a-number"]},
        {"target_joints": {"j0": 0.1}},
        "trajectory_error:",
        id="joint-position-not-numeric",
    ),
]


@pytest.mark.parametrize(("component_kwargs", "goal", "status_prefix"), _UNGUARDED_FAILURE_SOURCES)
def test_plan_reports_every_failure_source_instead_of_raising(
    component_kwargs: dict[str, Any], goal: dict[str, Any], status_prefix: str
) -> None:
    """Every moveit_py interaction reports its failure in the response.

    Each case is a real moveit_py failure mode at a stage other than
    ``component.plan()``: the current-state read before any goal is set, the
    goal set (an unknown joint, an unresolvable pose link, a ``target_pose``
    that is not 7 values), and the serialisation of the planned trajectory.
    """
    component = _StageFailingComponent(**component_kwargs)

    result = _plan_or_fail(_FakeMoveItPy(component=component), **goal)

    assert result["success"] is False
    assert result["trajectory"] == []
    assert result["status"].startswith(status_prefix), result["status"]
    # The detail carries the planner's own complaint, so an operator can act
    # on it without reading the sidecar's log.
    assert len(result["status"]) > len(status_prefix)


def test_plan_status_kinds_are_documented() -> None:
    """Every status kind ``_plan`` can emit is named in its docstring.

    The kind is the part an operator matches on, so a kind the docstring does
    not name is one nobody can look up.
    """
    doc = zmq_node._plan.__doc__ or ""
    for kind in (
        "unknown_planning_group",
        "start_state_error",
        "missing_goal",
        "invalid_goal",
        "planner_exception",
        "planner_returned_empty",
        "trajectory_error",
    ):
        assert kind in doc, f"_plan can emit {kind!r} but its docstring does not name it"


def test_plan_already_guarded_statuses_are_unchanged() -> None:
    """The four failure kinds that already reported keep their exact status.

    Guarding the remaining interactions must not reword an existing verdict -
    operators and forks match on these strings.
    """
    unknown_group = _FakeMoveItPy(component=None, unknown_group=True)
    assert _plan_or_fail(unknown_group, target_joints={"j0": 0.1})["status"].startswith("unknown_planning_group:")

    component = _FakeComponent(plan_points=[_FakePoint(sec=0, nanosec=0, positions=[0.1])])
    assert (
        _plan_or_fail(_FakeMoveItPy(component=component))["status"]
        == "missing_goal:expected_target_pose_or_target_joints"
    )

    raising = _FakeComponent(plan_points=None, plan_raises=True)
    assert _plan_or_fail(_FakeMoveItPy(component=raising), target_joints={"j0": 0.1})["status"].startswith(
        "planner_exception:"
    )

    empty = _FakeComponent(plan_points=None)
    assert (
        _plan_or_fail(_FakeMoveItPy(component=empty), target_joints={"j0": 0.1})["status"] == "planner_returned_empty"
    )


def test_plan_happy_path_still_serialises_rows() -> None:
    """A plan that succeeds is unaffected by the new guards."""
    component = _StageFailingComponent(positions=[0.1, 0.2])

    result = _plan_or_fail(_FakeMoveItPy(component=component), target_joints={"j0": 0.1})

    assert result == {"trajectory": [[0.0, 0.1, 0.2]], "success": True, "status": "ok"}


# ---------------------------------------------------------------------------
# The REP loop owes its peer exactly one reply.
# ---------------------------------------------------------------------------
def test_loop_answers_every_request_when_a_handler_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """A raising handler is answered and the sidecar keeps serving.

    The module invites forks to edit the handlers (start-state override, pose
    frame, ``world_update`` schema), so the loop must not turn an edit into a
    process exit. Without the dispatch guard the exception unwinds ``main``:
    the socket closes with the second request unanswered and the third never
    read, so a single bad request stops planning for every client.
    """
    requests = [
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
        msgpack.packb(
            {"endpoint": "plan", "data": {"planning_group": "arm", "target_joints": {"j0": 0.1}}},
            use_bin_type=True,
        ),
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
    ]
    socket = _FakeSocket(requests)
    _install_fake_zmq_rclpy(monkeypatch, socket)
    monkeypatch.setattr(zmq_node, "_build_moveit_py", lambda args: _FakeMoveItPy(component=None))

    def _forked_handler_raises(*_args: Any, **_kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("a fork's start-state builder failed")

    monkeypatch.setattr(zmq_node, "_plan", _forked_handler_raises)

    rc = zmq_node.main(["--port", "5599"])

    assert rc == 0, "the sidecar must survive a failing request"
    assert len(socket.sent) == len(requests), (
        f"{len(requests)} requests received but {len(socket.sent)} replies sent - "
        "a REQ peer is left waiting and the sidecar has stopped serving"
    )

    responses = [msgpack.unpackb(s, raw=False) for s in socket.sent]
    assert responses[0] == {"status": "ok"}
    # ``error`` is the envelope the client turns into RuntimeError, and it names
    # the exception so the failure is diagnosable from the client side.
    assert responses[1]["error"].startswith("internal_error:RuntimeError:")
    assert "start-state builder failed" in responses[1]["error"]
    # The third request is served normally: one bad request is not terminal.
    assert responses[2] == {"status": "ok"}


def test_loop_answers_a_response_that_will_not_serialise(monkeypatch: pytest.MonkeyPatch) -> None:
    """A response msgpack cannot pack is still answered.

    A fork returning an un-packable value (a ROS message object, a set) would
    otherwise raise inside ``send`` with the reply never delivered.
    """
    requests = [
        msgpack.packb(
            {"endpoint": "plan", "data": {"planning_group": "arm", "target_joints": {"j0": 0.1}}},
            use_bin_type=True,
        ),
    ]
    socket = _FakeSocket(requests)
    _install_fake_zmq_rclpy(monkeypatch, socket)
    monkeypatch.setattr(zmq_node, "_build_moveit_py", lambda args: _FakeMoveItPy(component=None))
    monkeypatch.setattr(zmq_node, "_plan", lambda *a, **k: {"trajectory": {object()}})

    rc = zmq_node.main(["--port", "5599"])

    assert rc == 0
    assert len(socket.sent) == 1
    assert "internal_error:" in msgpack.unpackb(socket.sent[0], raw=False)["error"]


def test_loop_reports_the_invariant_it_holds() -> None:
    """The module docstring states the one-reply-per-request invariant.

    The wire protocol is what a fork implements against, so the guarantee that
    a failing request is answered rather than dropped belongs beside it.
    """
    doc = zmq_node.__doc__ or ""
    assert "exactly one reply" in doc


# ---------------------------------------------------------------------------
# A payload that decodes to something other than a map.
#
# ``msgpack.unpackb`` decodes any valid msgpack value, so the undecodable-bytes
# guard above it does not make ``request`` a mapping: the single byte ``0x2a``
# is the integer 42, and a string, list, nil, bool or float decode just as
# cleanly. Reading ``endpoint`` off one of those raises between the two guards,
# where nothing answers - so it unwinds the REP loop, runs the ``finally`` that
# closes the socket with the reply never sent, and ends the process. That is the
# same one-bad-request-stops-every-client failure the dispatch guard exists to
# close, reachable by any peer (the reference sidecar accepts any client).
# ---------------------------------------------------------------------------
def _serve_or_fail(
    monkeypatch: pytest.MonkeyPatch, requests: list[bytes], *, plan_points: list[_FakePoint] | None = None
) -> tuple[int, list[dict[str, Any]]]:
    """Run the REP loop over ``requests``; fail naming the consequence if it escapes.

    Returns the exit code and the decoded replies. An exception escaping
    ``main`` is the failure under test, not an error in the test: it is
    reported with the number of replies that made it out first.
    """
    socket = _FakeSocket(requests)
    _install_fake_zmq_rclpy(monkeypatch, socket)
    points = plan_points if plan_points is not None else [_FakePoint(sec=0, nanosec=0, positions=[0.1])]
    component = _FakeComponent(plan_points=points)
    monkeypatch.setattr(zmq_node, "_build_moveit_py", lambda args: _FakeMoveItPy(component=component))

    try:
        rc = zmq_node.main(["--port", "5599"])
    except Exception as exc:
        raise AssertionError(
            f"main() raised {type(exc).__name__}: {exc} instead of answering the request. "
            f"Only {len(socket.sent)} of {len(requests)} requests were answered: the exception "
            "unwinds the REP loop, so the socket closes with the peer's reply never sent and the "
            "sidecar stops serving every other client too."
        ) from exc
    return rc, [msgpack.unpackb(s, raw=False) for s in socket.sent]


# (payload bytes, the type name the reply must name)
_NON_MAP_PAYLOADS = [
    pytest.param(b"\x2a", "int", id="single-byte-integer"),
    pytest.param(msgpack.packb("plan", use_bin_type=True), "str", id="bare-string"),
    pytest.param(msgpack.packb(["plan"], use_bin_type=True), "list", id="bare-list"),
    pytest.param(msgpack.packb(None, use_bin_type=True), "NoneType", id="nil"),
    pytest.param(msgpack.packb(True, use_bin_type=True), "bool", id="bare-bool"),
    pytest.param(msgpack.packb(1.5, use_bin_type=True), "float", id="bare-float"),
]


@pytest.mark.parametrize(("payload", "type_name"), _NON_MAP_PAYLOADS)
def test_loop_answers_a_payload_that_is_not_a_map(
    monkeypatch: pytest.MonkeyPatch, payload: bytes, type_name: str
) -> None:
    """A decodable non-map payload is answered and the sidecar keeps serving.

    The request either side of it is a ``ping``, so the reply count shows
    whether the loop survived: without the guard the second request is
    unanswered and the third is never read.
    """
    requests = [
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
        payload,
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
    ]

    rc, responses = _serve_or_fail(monkeypatch, requests)

    assert rc == 0, "the sidecar must survive a payload that is not a request"
    assert len(responses) == len(requests), (
        f"{len(requests)} requests received but {len(responses)} replies sent - a REQ peer is left waiting"
    )
    assert responses[0] == {"status": "ok"}
    # The detail names what arrived, so a client author can see the payload was
    # not a map rather than guessing at the sidecar's internals.
    assert responses[1]["error"].startswith("malformed_request:"), responses[1]
    assert type_name in responses[1]["error"], responses[1]
    # The request after it is served normally: one bad payload is not terminal.
    assert responses[2] == {"status": "ok"}


def test_both_malformed_payload_kinds_share_one_error_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Undecodable bytes and a decodable non-map report the same error class.

    Both are the same problem from the client's side - the bytes it sent are
    not a request - so a client matching on ``malformed_request:`` catches both
    without knowing which. Neither is ``internal_error:``, which would blame
    the sidecar for the peer's payload.
    """
    requests = [
        b"\xc1",  # not valid msgpack at all
        b"\x2a",  # valid msgpack, decodes to the integer 42
    ]

    rc, responses = _serve_or_fail(monkeypatch, requests)

    assert rc == 0
    assert len(responses) == 2
    for response in responses:
        assert response["error"].startswith("malformed_request:"), response
        assert "internal_error" not in response["error"], response


def test_a_non_map_data_field_was_already_answered(monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-map ``data`` is answered by the dispatch guard, not by this check.

    ``data`` is read inside the dispatch ``try``, so it is reported there. That
    is what bounds the map check to ``request`` - the one read that happens
    where no guard can answer it.
    """
    requests = [
        msgpack.packb({"endpoint": "plan", "data": 42}, use_bin_type=True),
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
    ]

    rc, responses = _serve_or_fail(monkeypatch, requests)

    assert rc == 0
    assert len(responses) == 2
    assert responses[0]["error"].startswith("internal_error:"), responses[0]
    assert responses[1] == {"status": "ok"}


def test_a_well_formed_request_is_unaffected_by_the_map_check(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every endpoint still dispatches: the check only rejects non-maps."""
    requests = [
        msgpack.packb({"endpoint": "ping"}, use_bin_type=True),
        msgpack.packb({"endpoint": "reset", "data": {"options": {"seed": 7}}}, use_bin_type=True),
        msgpack.packb(
            {"endpoint": "plan", "data": {"planning_group": "arm", "target_joints": {"j0": 0.1}}},
            use_bin_type=True,
        ),
        msgpack.packb({"endpoint": "bogus"}, use_bin_type=True),
    ]

    rc, responses = _serve_or_fail(monkeypatch, requests)

    assert rc == 0
    assert responses[0] == {"status": "ok"}
    assert responses[1] == {"status": "ok"}
    assert responses[2] == {"trajectory": [[0.0, 0.1]], "success": True, "status": "ok"}
    assert responses[3] == {"error": "unknown_endpoint:bogus"}
