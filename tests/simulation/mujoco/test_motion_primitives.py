"""Analytic motion primitives - ``move_to`` / ``set_gripper`` / ``rotate_wrist``.

GH #1645: the agent-facing staging/transport/release vocabulary (Harness VLA,
arXiv:2607.08448) that positions the robot around a learned policy's competence
region. These tests pin the whole tool contract on a genuine MuJoCo scene (an
inline MJCF arm - no asset downloads):

* discovery: the actions are advertised in ``tool_spec.json`` and ``describe()``;
* dispatch: kwargs are forwarded end-to-end (no silent drops), vector params
  are validated at the router;
* guards: no-world, unknown-robot, and refuse-while-policy-running (per-robot,
  the ``_require_no_running_policy`` regression from the issue's acceptance
  criteria);
* semantics: ``move_to`` reaches a reachable target, returns a structured
  error with the IK residual for an unreachable one (never a hang or raise),
  rejects targets outside the workspace sanity box, and degrades to a
  structured error (not a raise) when ``mink`` is missing; ``set_gripper``
  drives toward the correct ctrlrange end per state; ``rotate_wrist`` reaches
  a set-point and rejects out-of-range targets;
* recording interplay (the #1498 bug class): primitive motion does NOT feed
  the dataset recorder - pinned explicitly so a silent zero-frame "recording"
  can never masquerade as a recorded episode.

IK-dependent tests ``importorskip`` on ``mink`` (the dev env ships it; a clean
base install skips them), everything else runs on bare ``mujoco``.
"""

import concurrent.futures
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("mujoco")

from strands_robots.simulation.mujoco.simulation import Simulation  # noqa: E402

# A stable 4-DOF positioner + jaw with conventional names so the primitives'
# heuristics resolve everything: an ``ee_site`` TCP site (EE-frame discovery),
# a ``wrist_roll`` hinge (rotate_wrist), and a ``jaw`` joint whose LOW range
# end is "closed" (set_gripper, the SO-101 convention). Position servos use
# dampratio + joint armature/damping so the servo loop is stable at the
# default 0.002 s timestep.
ARM_XML = """
<mujoco model="prim_arm">
  <compiler angle="radian" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 0"/>
  <default>
    <joint armature="0.05" damping="0.5"/>
    <geom density="2000"/>
  </default>
  <worldbody>
    <body name="base" pos="0 0 0">
      <geom type="cylinder" size="0.04 0.02"/>
      <body name="link1" pos="0 0 0.05">
        <joint name="shoulder_pan" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.2" size="0.02"/>
        <body name="link2" pos="0 0 0.2">
          <joint name="shoulder_lift" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.02"/>
          <body name="link3" pos="0.15 0 0">
            <joint name="elbow" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0.15 0 0" size="0.018"/>
            <body name="link4" pos="0.15 0 0">
              <joint name="wrist_roll" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
              <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015"/>
              <site name="ee_site" pos="0.05 0 0"/>
              <body name="jaw_body" pos="0.05 0 0">
                <joint name="jaw" type="hinge" axis="0 0 1" range="-0.2 1.5" armature="0.01" damping="0.1"/>
                <geom type="box" size="0.01 0.01 0.02" contype="0" conaffinity="0"/>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="shoulder_pan" joint="shoulder_pan" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="shoulder_lift" joint="shoulder_lift" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="elbow" joint="elbow" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="wrist_roll" joint="wrist_roll" kp="5" dampratio="1" ctrlrange="-3.0 3.0"/>
    <position name="jaw" joint="jaw" kp="2" dampratio="1" ctrlrange="-0.2 1.5"/>
  </actuator>
</mujoco>
"""

# Reachable for the arm above (the ee_site sweeps a shell around the shoulder
# at (0, 0, 0.25) with reach 0.35); verified against the real solver.
REACHABLE = [0.2, 0.1, 0.2]
UNREACHABLE = [1.5, 0.0, 0.2]  # inside the sanity box, outside the workspace


@pytest.fixture
def arm_path(tmp_path):
    path = tmp_path / "prim_arm.xml"
    path.write_text(ARM_XML)
    return str(path)


@pytest.fixture
def sim(arm_path):
    s = Simulation(tool_name="test_motion_primitives", mesh=False)
    assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
    assert s.add_robot("arm", urdf_path=arm_path)["status"] == "success"
    yield s
    s.cleanup(policy_stop_timeout=2.0)


def _dispatch(s: Simulation, action: str, **fields: Any) -> dict[str, Any]:
    return s._dispatch_action(action, {"action": action, **fields})


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _drain_policy_thread(s: Simulation, robot_name: str, timeout: float = 10.0) -> None:
    """Wait for a stopped policy's worker thread to exit before touching the robot.

    Uses :func:`concurrent.futures.wait` rather than ``fut.result()`` because a
    policy stopped mid-run may legitimately have raised inside the worker; these
    tests only require that the thread has finished (so the per-robot policy
    guard clears), and a thread that fails to stop is a hard test failure.
    """
    fut = s._policy_threads.get(robot_name)
    if fut is None:
        return
    done, _ = concurrent.futures.wait([fut], timeout=timeout)
    assert fut in done, f"policy thread for {robot_name!r} did not stop within {timeout}s"


def _joint_positions(s: Simulation) -> dict[str, float]:
    state = _json_block(s.get_robot_state("arm"))["state"]
    return {joint: float(entry["position"]) for joint, entry in state.items()}


def _jaw_pos(s: Simulation) -> float:
    return _joint_positions(s)["jaw"]


class TestDiscoverySurface:
    """tool_spec + describe() advertise the primitives (acceptance criterion)."""

    def test_tool_spec_enum_contains_primitives(self):
        s = Simulation(tool_name="spec_probe", mesh=False)
        try:
            actions = s.tool_spec["inputSchema"]["json"]["properties"]["action"]["enum"]
            for action in ("move_to", "set_gripper", "rotate_wrist"):
                assert action in actions
            description = s.tool_spec["description"]
            for action in ("move_to", "set_gripper", "rotate_wrist"):
                assert action in description
        finally:
            s.cleanup()

    def test_tool_spec_declares_primitive_params(self):
        s = Simulation(tool_name="spec_probe", mesh=False)
        try:
            props = s.tool_spec["inputSchema"]["json"]["properties"]
            assert props["tol"]["type"] == "number"
            assert props["state"]["enum"] == ["open", "close"]
            assert props["steps"]["type"] == "integer"
            assert props["target_yaw"]["type"] == "number"
        finally:
            s.cleanup()

    def test_describe_advertises_primitives(self, sim):
        methods = sim.describe()["methods"]
        for action in ("move_to", "set_gripper", "rotate_wrist"):
            assert action in methods, f"describe() missing {action}"
        assert "collision-aware" in methods["move_to"]


class TestDispatchForwarding:
    """Every advertised kwarg reaches the method (no silent drops)."""

    def _capture(self, captured: dict[str, Any], sim: Simulation, method_name: str):
        import inspect
        from functools import wraps

        original = getattr(sim, method_name)

        @wraps(original)
        def fake(*args: Any, **kwargs: Any) -> dict[str, Any]:
            bound = inspect.signature(original).bind_partial(*args, **kwargs)
            captured.clear()
            captured.update(bound.arguments)
            return {"status": "success", "content": [{"text": "ok"}]}

        return fake

    def test_move_to_kwargs_forwarded(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "move_to", self._capture(captured, sim, "move_to")):
            result = _dispatch(
                sim,
                "move_to",
                robot_name="arm",
                position=[0.1, 0.2, 0.3],
                orientation=[1, 0, 0, 0],
                tol=0.05,
                max_steps=7,
            )
        assert result["status"] == "success"
        assert captured == {
            "robot_name": "arm",
            "position": [0.1, 0.2, 0.3],
            "orientation": [1, 0, 0, 0],
            "tol": 0.05,
            "max_steps": 7,
        }

    def test_set_gripper_kwargs_forwarded(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "set_gripper", self._capture(captured, sim, "set_gripper")):
            result = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=5)
        assert result["status"] == "success"
        assert captured == {"robot_name": "arm", "state": "close", "steps": 5}

    def test_rotate_wrist_kwargs_forwarded(self, sim):
        captured: dict[str, Any] = {}
        with patch.object(sim, "rotate_wrist", self._capture(captured, sim, "rotate_wrist")):
            result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.5, tol=0.01, max_steps=9)
        assert result["status"] == "success"
        assert captured == {"robot_name": "arm", "target_yaw": 0.5, "tol": 0.01, "max_steps": 9}

    def test_move_to_position_vector_validated_at_router(self, sim):
        result = _dispatch(sim, "move_to", robot_name="arm", position=[0.1, 0.2])
        assert result["status"] == "error"
        assert "must be a list of 3 numbers" in result["content"][0]["text"]

    def test_move_to_orientation_vector_validated_at_router(self, sim):
        result = _dispatch(sim, "move_to", robot_name="arm", position=[0.1, 0.2, 0.3], orientation=[1, 0, 0])
        assert result["status"] == "error"
        assert "must be a list of 4 numbers" in result["content"][0]["text"]

    def test_unknown_param_rejected(self, sim):
        result = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, speed=2.0)
        assert result["status"] == "error"
        assert "Unknown parameter 'speed'" in result["content"][0]["text"]


class TestGuards:
    """World / robot / policy-running guards return structured errors."""

    def test_no_world_errors(self):
        s = Simulation(tool_name="no_world", mesh=False)
        try:
            for action, fields in (
                ("move_to", {"position": REACHABLE}),
                ("set_gripper", {"state": "open"}),
                ("rotate_wrist", {"target_yaw": 0.1}),
            ):
                result = _dispatch(s, action, **fields)
                assert result["status"] == "error", (action, result)
                assert "No world" in result["content"][0]["text"]
        finally:
            s.cleanup()

    def test_unknown_robot_errors(self, sim):
        for action, fields in (
            ("move_to", {"position": REACHABLE}),
            ("set_gripper", {"state": "open"}),
            ("rotate_wrist", {"target_yaw": 0.1}),
        ):
            result = _dispatch(sim, action, robot_name="nope", **fields)
            assert result["status"] == "error", (action, result)
            assert "Robot 'nope' not found" in result["content"][0]["text"]

    def test_refused_while_policy_running(self, sim):
        """Acceptance criterion: primitives refuse while a policy runs on the robot."""
        assert sim.start_policy("arm", policy_provider="mock", duration=10.0, fast_mode=True)["status"] == "success"
        try:
            for action, fields in (
                ("move_to", {"position": REACHABLE}),
                ("set_gripper", {"state": "open"}),
                ("rotate_wrist", {"target_yaw": 0.1}),
            ):
                result = _dispatch(sim, action, robot_name="arm", **fields)
                assert result["status"] == "error", (action, result)
                assert "policy is running" in result["content"][0]["text"], (action, result)
        finally:
            sim.stop_policy("arm")
            _drain_policy_thread(sim, "arm")

    def test_allowed_after_policy_stopped(self, sim):
        assert sim.start_policy("arm", policy_provider="mock", duration=10.0, fast_mode=True)["status"] == "success"
        sim.stop_policy("arm")
        _drain_policy_thread(sim, "arm")
        result = _dispatch(sim, "set_gripper", robot_name="arm", state="open", steps=2)
        assert result["status"] == "success", result


class TestMoveTo:
    """Cartesian transport semantics (real mink solver)."""

    def test_reaches_reachable_target(self, sim):
        pytest.importorskip("mink")
        result = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.02)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["position_error_m"] <= 0.02
        assert payload["frame"] == "arm/ee_site"
        assert payload["frame_type"] == "site"

    def test_unreachable_target_returns_structured_error_with_residual(self, sim):
        """Acceptance criterion: unreachable -> structured error + residual, no hang/raise."""
        pytest.importorskip("mink")
        result = _dispatch(sim, "move_to", robot_name="arm", position=UNREACHABLE, tol=0.01)
        assert result["status"] == "error", result
        assert "unreachable" in result["content"][0]["text"]
        payload = _json_block(result)
        assert payload["reached"] is False
        assert payload["ik_residual_m"] > 0.01

    def test_sanity_box_rejects_far_target(self, sim):
        result = _dispatch(sim, "move_to", robot_name="arm", position=[50.0, 0.0, 0.2])
        assert result["status"] == "error"
        assert "sanity box" in result["content"][0]["text"]

    def test_missing_position_errors(self, sim):
        result = _dispatch(sim, "move_to", robot_name="arm")
        assert result["status"] == "error"
        assert "position" in result["content"][0]["text"]

    def test_bad_tol_and_max_steps_rejected(self, sim):
        assert _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.0)["status"] == "error"
        assert _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=-1)["status"] == "error"
        assert _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, max_steps=0)["status"] == "error"
        assert _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, max_steps=10**9)["status"] == "error"

    def test_missing_mink_degrades_to_structured_error(self, sim, monkeypatch):
        """A missing IK stack is a structured error through the tool surface, not a raise."""
        monkeypatch.setitem(sys.modules, "mink", None)  # forces `import mink` to ImportError
        result = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE)
        assert result["status"] == "error"
        assert "IK bridge unavailable" in result["content"][0]["text"]
        assert "mink" in result["content"][0]["text"]


class TestGraspPreservation:
    """``move_to`` must never command the gripper (review on #1654).

    The gripper DOF is kinematically irrelevant to the EE task, so mink passes
    its seed value straight through; the pre-fix restart loop randomized the
    jaw over its full range and then servoed it there - the
    stage -> close -> transport sequence silently dropped whatever it held.
    These tests pin the fix: gripper actuators are excluded from the IK
    seeding/command set and HELD at their live position.
    """

    def test_restart_path_move_to_preserves_closed_gripper(self, sim):
        """A target the direct solve misses (restart branch engaged) must not
        move a closed jaw. [-0.3, 0.0, 0.25] has a direct-solve residual of
        ~0.65 m on this arm - the exact repro from the review."""
        pytest.importorskip("mink")
        r = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=30)
        assert r["status"] == "success", r
        jaw_before = _jaw_pos(sim)
        assert jaw_before < 0.0, f"jaw did not close: {jaw_before}"

        r = _dispatch(sim, "move_to", robot_name="arm", position=[-0.3, 0.0, 0.25], tol=0.03, max_steps=400)
        assert r["status"] == "success", r
        assert _json_block(r)["reached"] is True

        jaw_after = _jaw_pos(sim)
        assert abs(jaw_after - jaw_before) < 0.05, (
            f"move_to moved the gripper: jaw {jaw_before:.3f} -> {jaw_after:.3f} "
            f"(ctrlrange [-0.2, 1.5]; the restart path must not command the jaw)"
        )

    def test_direct_path_move_to_preserves_open_gripper(self, sim):
        """The direct-solve path must hold the jaw too (the gripper channel is
        written every tick with the LIVE position, not left to stale ctrl)."""
        pytest.importorskip("mink")
        r = _dispatch(sim, "set_gripper", robot_name="arm", state="open", steps=40)
        assert r["status"] == "success", r
        jaw_before = _jaw_pos(sim)
        assert jaw_before > 1.0, f"jaw did not open: {jaw_before}"

        r = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.03, max_steps=400)
        assert r["status"] == "success", r

        jaw_after = _jaw_pos(sim)
        assert abs(jaw_after - jaw_before) < 0.05, f"move_to moved the gripper: jaw {jaw_before:.3f} -> {jaw_after:.3f}"


class TestSetGripper:
    """Open/close set-point semantics (LOW end = closed, HIGH end = open)."""

    def test_close_drives_toward_low_end(self, sim):
        result = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=40)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["targets"]["jaw"] == pytest.approx(-0.2)
        assert _jaw_pos(sim) < -0.1  # traveled toward the low (closed) end

    def test_open_drives_toward_high_end(self, sim):
        _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=40)
        result = _dispatch(sim, "set_gripper", robot_name="arm", state="open", steps=80)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["targets"]["jaw"] == pytest.approx(1.5)
        assert _jaw_pos(sim) > 1.0  # traveled toward the high (open) end

    def test_invalid_state_rejected(self, sim):
        result = _dispatch(sim, "set_gripper", robot_name="arm", state="ajar")
        assert result["status"] == "error"
        assert '"open" or "close"' in result["content"][0]["text"]

    def test_missing_state_rejected(self, sim):
        result = _dispatch(sim, "set_gripper", robot_name="arm")
        assert result["status"] == "error"

    def test_bad_steps_rejected(self, sim):
        assert _dispatch(sim, "set_gripper", robot_name="arm", state="open", steps=0)["status"] == "error"


class TestRotateWrist:
    """Wrist-yaw set-point semantics."""

    def test_reaches_target_yaw(self, sim):
        result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.7)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["reached"] is True
        assert payload["wrist_joint"] == "wrist_roll"
        assert payload["final_yaw"] == pytest.approx(0.7, abs=0.03)

    def test_holds_other_joints(self, sim):
        before = _joint_positions(sim)
        result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.5)
        assert result["status"] == "success", result
        after = _joint_positions(sim)
        for joint in ("shoulder_pan", "shoulder_lift", "elbow"):
            assert abs(after[joint] - before[joint]) < 0.05, joint

    def test_out_of_range_target_rejected(self, sim):
        result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=10.0)
        assert result["status"] == "error"
        assert "outside joint" in result["content"][0]["text"]

    def test_missing_target_yaw_rejected(self, sim):
        result = _dispatch(sim, "rotate_wrist", robot_name="arm")
        assert result["status"] == "error"
        assert "target_yaw" in result["content"][0]["text"]


class TestRecordingInterplay:
    """Primitive motion does NOT feed the dataset recorder (documented + pinned).

    Only ``run_policy``'s per-frame hook records dataset episodes; a primitive
    stepping physics directly must not add frames (and must not pretend to).
    Camera MP4 recording is a separate live-sampling path and is unaffected.
    """

    def test_primitive_does_not_feed_dataset_recorder(self, sim):
        recorder = MagicMock()
        assert sim._world is not None
        sim._world._backend_state["recording"] = True
        sim._world._backend_state["dataset_recorder"] = recorder
        try:
            result = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=5)
            assert result["status"] == "success", result
            recorder.add_frame.assert_not_called()
        finally:
            sim._world._backend_state["recording"] = False
            sim._world._backend_state.pop("dataset_recorder", None)
