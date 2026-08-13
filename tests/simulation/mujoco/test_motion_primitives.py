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
  drives toward the correct set-point-range end per state (the range itself,
  including the driven-joint substitution for an MJCF that left the ctrlrange
  unset, is pinned by ``test_set_gripper_setpoint_range_sources.py``);
  ``rotate_wrist`` reaches a set-point and rejects out-of-range targets;
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

# The GH #1658 regression arm: same kinematics as ARM_XML, but the BASE pan
# actuator is named ``finger_camera_pan`` (contains the "finger" hint - the
# name heuristic misclassifies it as a gripper drive) and the real gripper
# matches the shipped so100 registry entry (actuator ``Jaw``). REACHABLE has a
# nonzero y component, so the EE target is only reachable when the pan DOF is
# available to the IK solve: with the heuristic in charge, move_to holds the
# pan at 0 and the target is unreachable.
HINT_COLLIDER_XML = ARM_XML.replace('"prim_arm"', '"hint_collider_arm"').replace("shoulder_pan", "finger_camera_pan")
assert HINT_COLLIDER_XML.count("finger_camera_pan") == 3  # joint def + actuator name/joint refs
HINT_COLLIDER_XML = HINT_COLLIDER_XML.replace('"jaw"', '"Jaw"')  # joint + actuator refs; so100's metadata name

# The GH #1661 regression arm: so101's shipped sim MJCF names its joints and
# actuators "1".."6". No wrist hint matches and no gripper hint matches, so
# the raw-heuristic fallback ("last non-gripper hinge") selected joint 6 -
# which IS the jaw: rotate_wrist would open/close the gripper instead of
# rotating the wrist. so101's registry gripper metadata (actuators: ["6"])
# is what excludes the jaw from the wrist candidate set. Same servo tuning
# as ARM_XML, one extra wrist-flex link so the DOF count matches the real
# robot (5 arm joints + jaw).
SO101_STYLE_XML = """
<mujoco model="so101_style_arm">
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
        <joint name="1" type="hinge" axis="0 0 1" range="-3.14 3.14"/>
        <geom type="capsule" fromto="0 0 0 0 0 0.1" size="0.02"/>
        <body name="link2" pos="0 0 0.1">
          <joint name="2" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
          <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.02"/>
          <body name="link3" pos="0.1 0 0">
            <joint name="3" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
            <geom type="capsule" fromto="0 0 0 0.1 0 0" size="0.018"/>
            <body name="link4" pos="0.1 0 0">
              <joint name="4" type="hinge" axis="0 1 0" range="-3.14 3.14"/>
              <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.016"/>
              <body name="link5" pos="0.05 0 0">
                <joint name="5" type="hinge" axis="1 0 0" range="-3.0 3.0"/>
                <geom type="capsule" fromto="0 0 0 0.05 0 0" size="0.015"/>
                <site name="ee_site" pos="0.05 0 0"/>
                <body name="jaw_body" pos="0.05 0 0">
                  <joint name="6" type="hinge" axis="0 0 1" range="-0.2 1.5" armature="0.01" damping="0.1"/>
                  <geom type="box" size="0.01 0.01 0.02" contype="0" conaffinity="0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>
  </worldbody>
  <actuator>
    <position name="1" joint="1" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="2" joint="2" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="3" joint="3" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="4" joint="4" kp="20" dampratio="1" ctrlrange="-3.14 3.14"/>
    <position name="5" joint="5" kp="5" dampratio="1" ctrlrange="-3.0 3.0"/>
    <position name="6" joint="6" kp="2" dampratio="1" ctrlrange="-0.2 1.5"/>
  </actuator>
</mujoco>
"""

# GH #1661's second failure mode (fallback shift): the most distal arm joint
# is named ``finger_camera_roll`` - the raw heuristic excluded it from the
# wrist candidate set, shifting the last-non-gripper-hinge fallback onto the
# elbow. With so100's registry metadata (gripper = ``Jaw``) the roll joint
# stays a candidate and the fallback picks it. No wrist hint matches (the
# name deliberately avoids "wrist"), so the fallback path is exercised.
FALLBACK_SHIFT_XML = ARM_XML.replace('"prim_arm"', '"fallback_shift_arm"').replace("wrist_roll", "finger_camera_roll")
assert FALLBACK_SHIFT_XML.count("finger_camera_roll") == 3  # joint def + actuator name/joint refs
FALLBACK_SHIFT_XML = FALLBACK_SHIFT_XML.replace('"jaw"', '"Jaw"')  # joint + actuator refs; so100's metadata name


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

    def test_servo_timeout_reports_the_residual_and_a_budget_that_fixes_it(self, sim):
        """A solvable pose the servo cannot reach inside the step budget.

        Distinct from ``test_unreachable_target_...`` above: there the IK cannot
        solve the pose at all and the pre-flight refuses before a single tick,
        so there is no measured position error to report. Here the IK residual
        is small - the pose IS solvable - and the servo simply runs out of
        steps, which is why the envelope's advice ("the servo may need more
        steps") is the actionable one. Verified by following it: the identical
        call with a real budget converges.
        """
        pytest.importorskip("mink")
        short = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.02, max_steps=2)
        assert short["status"] == "error", short
        text = short["content"][0]["text"]
        assert "did not reach" in text, text
        assert "max_steps=2" in text, text
        payload = _json_block(short)
        assert payload["reached"] is False
        assert payload["steps"] == 2
        # The two residuals are separate fields precisely so the agent can tell
        # "needs more steps" from "unreachable": here the IK solved the pose.
        assert payload["position_error_m"] > 0.02
        assert payload["ik_residual_m"] <= 0.02

        again = _dispatch(sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.02, max_steps=400)
        assert again["status"] == "success", again
        assert _json_block(again)["reached"] is True


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

    def test_servo_timeout_reports_the_residual_and_a_budget_that_fixes_it(self, sim):
        """An in-range yaw the servo cannot settle on inside the step budget.

        Distinct from ``test_out_of_range_target_rejected``: that target is
        refused up front against the joint range, so no tick runs. Here the
        target is legal and only the budget is short, so the refusal reports
        how far off the joint ended and stays retryable - verified by retrying.
        """
        short = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.3, tol=0.02, max_steps=1)
        assert short["status"] == "error", short
        text = short["content"][0]["text"]
        assert "did not reach" in text, text
        assert "max_steps=1" in text, text
        payload = _json_block(short)
        assert payload["reached"] is False
        assert payload["steps"] == 1
        assert payload["wrist_joint"] == "wrist_roll"
        assert payload["yaw_error_rad"] > 0.02

        again = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.3, tol=0.02, max_steps=300)
        assert again["status"] == "success", again
        assert _json_block(again)["reached"] is True


class TestGripperRegistryMetadata:
    """Registry gripper metadata beats the name heuristic (GH #1658).

    The heuristic's inverse failure mode is silent: an ARM actuator whose
    name happens to contain a hint (here ``finger_camera_pan``, the base pan
    joint) would be classified as a gripper - ``set_gripper`` would command
    it and ``move_to`` would exclude it from IK and hold it, quietly
    reducing the arm's usable DOF. Registry metadata for the robot's
    ``data_config`` is authoritative: only the named actuators are gripper
    drives. These tests use the REAL shipped ``so100`` registry entry
    (``actuators: ["Jaw"]``) against an inline arm - no asset downloads.
    """

    @pytest.fixture
    def collider_sim(self, tmp_path):
        path = tmp_path / "hint_collider_arm.xml"
        path.write_text(HINT_COLLIDER_XML)
        s = Simulation(tool_name="test_gripper_metadata", mesh=False)
        assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
        assert s.add_robot("arm", urdf_path=str(path), data_config="so100")["status"] == "success"
        yield s
        s.cleanup(policy_stop_timeout=2.0)

    def test_set_gripper_commands_only_metadata_actuators(self, collider_sim):
        """Regression (the issue's acceptance test): with metadata present,
        set_gripper commands ONLY the named actuator - the hint-colliding
        arm actuator is not driven. Pre-#1658 the heuristic classified
        ``finger_camera_pan`` as a gripper drive and slewed the whole base."""
        result = _dispatch(collider_sim, "set_gripper", robot_name="arm", state="close", steps=20)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["actuators"] == ["Jaw"], payload
        assert "finger_camera_pan" not in payload["targets"]

    def test_move_to_keeps_hint_colliding_arm_dof_in_ik(self, collider_sim):
        """The off-axis target NEEDS the base pan DOF. Pre-#1658 the heuristic
        excluded ``finger_camera_pan`` from the IK solve and held it, so the
        target was unreachable; metadata keeps the DOF usable while the real
        gripper stays held (grasp preservation intact)."""
        pytest.importorskip("mink")
        r = _dispatch(collider_sim, "set_gripper", robot_name="arm", state="close", steps=30)
        assert r["status"] == "success", r
        jaw_before = _json_block(r)["gripper_joint_positions"]["Jaw"]

        result = _dispatch(collider_sim, "move_to", robot_name="arm", position=REACHABLE, tol=0.03, max_steps=400)
        assert result["status"] == "success", result
        assert _json_block(result)["reached"] is True

        state = _json_block(collider_sim.get_robot_state("arm"))["state"]
        jaw_after = float(state["Jaw"]["position"])
        assert abs(jaw_after - jaw_before) < 0.05, f"move_to moved the gripper: {jaw_before:.3f} -> {jaw_after:.3f}"
        assert abs(float(state["finger_camera_pan"]["position"])) > 0.05, (
            "the hint-colliding base pan joint never moved - it was excluded from IK"
        )

    def test_alias_data_config_resolves_metadata(self, tmp_path):
        """data_config aliases (e.g. multi-cam configs) resolve to the canonical
        registry entry - so100_dualcam must not silently lose the metadata."""
        path = tmp_path / "hint_collider_arm.xml"
        path.write_text(HINT_COLLIDER_XML)
        s = Simulation(tool_name="test_gripper_metadata_alias", mesh=False)
        try:
            assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
            assert s.add_robot("arm", urdf_path=str(path), data_config="so100_dualcam")["status"] == "success"
            result = _dispatch(s, "set_gripper", robot_name="arm", state="open", steps=5)
            assert result["status"] == "success", result
            assert _json_block(result)["actuators"] == ["Jaw"]
        finally:
            s.cleanup(policy_stop_timeout=2.0)

    def test_set_gripper_honors_inverted_open_close_convention(self, collider_sim, monkeypatch):
        """The metadata ``closed``/``open`` fields override the open=HIGH /
        close=LOW convention (the SO-101-style sign trap): closed=high means
        'close' targets the HIGH ctrlrange end."""
        monkeypatch.setattr(
            "strands_robots.simulation.mujoco.motion_primitives.get_robot",
            lambda name: {"gripper": {"actuators": ["Jaw"], "closed": "high", "open": "low"}},
        )
        result = _dispatch(collider_sim, "set_gripper", robot_name="arm", state="close", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["targets"]["Jaw"] == pytest.approx(1.5)  # HIGH end
        result = _dispatch(collider_sim, "set_gripper", robot_name="arm", state="open", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["targets"]["Jaw"] == pytest.approx(-0.2)  # LOW end

    def test_stale_metadata_is_a_loud_error_not_a_heuristic_fallback(self, collider_sim, monkeypatch):
        """Metadata naming actuators absent from the model errors with the
        model's actual actuator list - silently degrading to the heuristic
        would reintroduce the misclassification the metadata prevents."""
        monkeypatch.setattr(
            "strands_robots.simulation.mujoco.motion_primitives.get_robot",
            lambda name: {"gripper": {"actuators": ["no_such_actuator"], "closed": "low", "open": "high"}},
        )
        for action, fields in (
            ("set_gripper", {"state": "close"}),
            ("move_to", {"position": REACHABLE}),
            ("rotate_wrist", {"target_yaw": 0.3}),
        ):
            result = _dispatch(collider_sim, action, robot_name="arm", **fields)
            assert result["status"] == "error", (action, result)
            text = result["content"][0]["text"]
            assert "no_such_actuator" in text and "Jaw" in text, (action, text)

    def test_malformed_metadata_is_a_loud_error(self, collider_sim, monkeypatch):
        monkeypatch.setattr(
            "strands_robots.simulation.mujoco.motion_primitives.get_robot",
            lambda name: {"gripper": {"actuators": [], "closed": "low", "open": "high"}},
        )
        for action, fields in (
            ("set_gripper", {"state": "close"}),
            ("rotate_wrist", {"target_yaw": 0.3}),
        ):
            result = _dispatch(collider_sim, action, robot_name="arm", **fields)
            assert result["status"] == "error", (action, result)
            assert "malformed" in result["content"][0]["text"], (action, result)

    def test_no_data_config_still_uses_heuristic(self, sim):
        """Zero-config fallback pinned: a robot without a data_config resolves
        the jaw by name hints exactly as before this feature."""
        result = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=5)
        assert result["status"] == "success", result
        assert _json_block(result)["actuators"] == ["jaw"]


class TestRotateWristRegistryMetadata:
    """rotate_wrist excludes gripper DOFs via the shared registry-first
    classification (GH #1661, follow-up to #1658).

    Both heuristic failure modes from #1658 applied to rotate_wrist's joint
    selection, and one was live on a shipped model: so101's sim MJCF names
    its joints ``1``..``6`` - no wrist hint matches, no gripper hint matches,
    so the last-non-gripper-hinge fallback selected joint ``6``, the JAW.
    rotate_wrist on so101 would open/close the gripper instead of rotating
    the wrist. These tests use the REAL shipped registry entries (so101:
    ``actuators: ["6"]``; so100: ``actuators: ["Jaw"]``) against inline
    arms - no asset downloads.
    """

    @pytest.fixture
    def so101_sim(self, tmp_path):
        path = tmp_path / "so101_style_arm.xml"
        path.write_text(SO101_STYLE_XML)
        s = Simulation(tool_name="test_rotate_wrist_metadata", mesh=False)
        assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
        assert s.add_robot("arm", urdf_path=str(path), data_config="so101")["status"] == "success"
        yield s
        s.cleanup(policy_stop_timeout=2.0)

    def test_jaw_is_never_selected_as_wrist_on_so101_style_model(self, so101_sim):
        """The issue's acceptance test: joints named 1..6 + so101 registry
        metadata - the jaw (joint 6) is excluded, the fallback picks the
        distal arm roll joint (5), and the jaw does not move."""
        result = _dispatch(so101_sim, "rotate_wrist", robot_name="arm", target_yaw=0.5)
        assert result["status"] == "success", result
        payload = _json_block(result)
        assert payload["wrist_joint"] == "5", payload
        assert payload["reached"] is True
        state = _json_block(so101_sim.get_robot_state("arm"))["state"]
        assert float(state["5"]["position"]) == pytest.approx(0.5, abs=0.05)
        assert abs(float(state["6"]["position"])) < 0.05, "rotate_wrist moved the jaw"

    def test_hint_colliding_distal_joint_stays_a_wrist_candidate(self, tmp_path):
        """The fallback-shift failure mode: the most distal arm joint is named
        ``finger_camera_roll``. The raw heuristic excluded it (finger hint)
        and the fallback shifted onto the elbow; with so100's metadata
        (gripper = Jaw) the roll joint stays a candidate and is picked."""
        path = tmp_path / "fallback_shift_arm.xml"
        path.write_text(FALLBACK_SHIFT_XML)
        s = Simulation(tool_name="test_rotate_wrist_fallback_shift", mesh=False)
        try:
            assert s.create_world(gravity=[0, 0, 0])["status"] == "success"
            assert s.add_robot("arm", urdf_path=str(path), data_config="so100")["status"] == "success"
            result = _dispatch(s, "rotate_wrist", robot_name="arm", target_yaw=0.5)
            assert result["status"] == "success", result
            payload = _json_block(result)
            assert payload["wrist_joint"] == "finger_camera_roll", payload
            assert payload["reached"] is True
            state = _json_block(s.get_robot_state("arm"))["state"]
            assert abs(float(state["elbow"]["position"])) < 0.05, "fallback shifted onto the elbow"
            assert abs(float(state["Jaw"]["position"])) < 0.05, "rotate_wrist moved the jaw"
        finally:
            s.cleanup(policy_stop_timeout=2.0)

    def test_no_metadata_heuristic_unchanged(self, sim):
        """Zero-config fallback pinned: without registry metadata the wrist
        hint match resolves ``wrist_roll`` exactly as before this change."""
        result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.4)
        assert result["status"] == "success", result
        assert _json_block(result)["wrist_joint"] == "wrist_roll"


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
