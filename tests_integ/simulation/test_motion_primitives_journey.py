"""Integration: analytic motion primitives drive a real SO-100 pick sequence.

GH #1645 acceptance criterion: an agent can execute
``move_to`` (above a cube) -> ``set_gripper("close")`` -> ``move_to`` (lift) ->
``set_gripper("open")`` on a registry SO-100 arm in MuJoCo **via tool-dispatch
actions only** (the exact `_dispatch_action` surface an LLM agent drives), with
real mink IK and real physics - no mocks.

Also journeys ``rotate_wrist`` on the real arm (the SO-100's ``Wrist_Roll``
joint) since the wrist heuristic is name-driven and must be pinned against a
real menagerie model, not just the synthetic unit-test arm.

Skips cleanly when the SO-100 asset or the mink/qpsolvers stack is absent.
"""

from __future__ import annotations

import os
from typing import Any

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mink")

os.environ.setdefault("MUJOCO_GL", "glfw")


@pytest.fixture
def sim():
    """A fresh MuJoCo world with one SO-100 arm and a graspable cube."""
    from strands_robots.simulation import Simulation

    s = Simulation()
    assert s.create_world(timestep=0.002, gravity=[0.0, 0.0, -9.81])["status"] == "success"
    result = s.add_robot("arm", data_config="so100", position=[0.0, 0.0, 0.0])
    if result["status"] != "success":
        s.destroy()
        pytest.skip(f"so100 asset unavailable: {result['content'][0]['text']}")
    s.add_object(name="cube", shape="box", size=[0.015, 0.015, 0.015], position=[0.2, 0.0, 0.015], color=[1, 0, 0, 1])
    s.step(n_steps=50)  # settle
    yield s
    s.destroy()


def _dispatch(s: Any, action: str, **fields: Any) -> dict[str, Any]:
    return s._dispatch_action(action, {"action": action, **fields})


def _json_block(result: dict[str, Any]) -> dict[str, Any]:
    for block in result["content"]:
        if isinstance(block, dict) and "json" in block:
            return block["json"]
    raise AssertionError(f"no json block in result: {result}")


def _jaw_position(s: Any) -> float:
    """Live position of the SO-100 jaw joint (grasp-preservation probe)."""
    state = _json_block(s.get_robot_state("arm"))["state"]
    for joint, entry in state.items():
        if "jaw" in joint.lower():
            return float(entry["position"])
    raise AssertionError(f"no jaw joint in robot state: {list(state)}")


def test_pick_sequence_via_dispatch_only(sim):
    """move_to above cube -> close -> move_to lift -> open, dispatch-only."""
    # The auto-discovered EE frame on SO-100 is the wrist body (no TCP site in
    # the menagerie model); the jaw hangs below it, so the staging height must
    # clear the jaw's extent above the cube.
    above_cube = [0.2, 0.0, 0.15]
    lift = [0.2, 0.0, 0.25]
    tol = 0.03

    # 1) Stage above the cube (position-only IK - SO-100 is a 5-DOF arm).
    result = _dispatch(sim, "move_to", robot_name="arm", position=above_cube, tol=tol, max_steps=400)
    assert result["status"] == "success", result
    stage = _json_block(result)
    assert stage["reached"] is True
    assert stage["position_error_m"] <= tol

    # 2) Close the gripper (SO-100 jaw closes toward the LOW end of ctrlrange).
    result = _dispatch(sim, "set_gripper", robot_name="arm", state="close", steps=30)
    assert result["status"] == "success", result
    jaw_closed = _jaw_position(sim)

    # 3) Lift.
    result = _dispatch(sim, "move_to", robot_name="arm", position=lift, tol=tol, max_steps=400)
    assert result["status"] == "success", result
    lifted = _json_block(result)
    assert lifted["reached"] is True
    # Targets are 0.10 m apart in z and each end pose is within tol of its
    # target, so the realized z gain is at least 0.10 - 2*tol.
    assert lifted["ee_position"][2] - stage["ee_position"][2] > 0.10 - 2 * tol
    # Grasp preservation on the real arm (review on #1654): transport must not
    # command the jaw - a closed gripper stays closed through move_to. (The
    # restart-path variant of this pin lives in the unit suite, where the
    # synthetic arm makes the restart branch deterministic.)
    jaw_after_lift = _jaw_position(sim)
    assert abs(jaw_after_lift - jaw_closed) < 0.15, (
        f"move_to moved the jaw during transport: {jaw_closed:.3f} -> {jaw_after_lift:.3f}"
    )

    # 4) Release.
    result = _dispatch(sim, "set_gripper", robot_name="arm", state="open", steps=30)
    assert result["status"] == "success", result
    payload = _json_block(result)
    assert payload["state"] == "open"


def test_unreachable_target_returns_residual_on_real_arm(sim):
    """A target far outside the SO-100 workspace errors with the IK residual."""
    result = _dispatch(sim, "move_to", robot_name="arm", position=[2.0, 0.0, 0.2], tol=0.01, max_steps=50)
    assert result["status"] == "error", result
    assert "unreachable" in result["content"][0]["text"]
    assert _json_block(result)["ik_residual_m"] > 0.01


def test_rotate_wrist_on_real_arm(sim):
    """The wrist heuristic resolves SO-100's Wrist_Roll and reaches a set-point."""
    result = _dispatch(sim, "rotate_wrist", robot_name="arm", target_yaw=0.5, tol=0.05, max_steps=400)
    assert result["status"] == "success", result
    payload = _json_block(result)
    assert payload["reached"] is True
    assert "wrist" in payload["wrist_joint"].lower()
