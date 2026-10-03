"""``grasped``'s ``gripper_prefix`` names the gripper body, and a prefix no body has is refused.

On the SO-101 the cube is touched by ``so101/so101/static_finger``, a named
geom on body ``so101/gripper``; the moving jaw is a child body. Matching the
prefix against geom labels missed it, so the documented ``"so101/gripper"``
never fired on a real pick, and an unmatchable ``"so100"`` armed silently and
pinned the clause ``False``.
"""

from __future__ import annotations

import pytest

pytest.importorskip("mujoco")

from strands_robots import Robot  # noqa: E402
from strands_robots.simulation.predicates import _grasped  # noqa: E402

CUBE_XY = (0.02, -0.34)  # examples/18_so101_pick_and_lift.py


def _ok(result: dict) -> dict:
    assert result.get("status") == "success", result
    return result


@pytest.fixture(scope="module")
def closed_on_cube():
    """examples/18 up to the jaws closing on the cube (no weld, so the touch is physical)."""
    sim = Robot("so101", mesh=False)
    _ok(sim.add_object(name="cube", shape="box", position=[*CUBE_XY, 0.0125], size=[0.025] * 3, mass=0.02))
    _ok(sim.step(200))
    z = sim.get_body_state("cube")["content"][-1]["json"]["position"][2]
    _ok(sim.set_gripper(robot_name="so101", state="open"))
    _ok(sim.move_to(robot_name="so101", position=[*CUBE_XY, z + 0.10], tol=0.02))
    _ok(sim.move_to(robot_name="so101", position=[*CUBE_XY, z + 0.005], tol=0.02))
    _ok(sim.set_gripper(robot_name="so101", state="close"))
    yield sim
    sim.destroy()


@pytest.mark.parametrize(
    ("prefix", "grasped"),
    [("so101/gripper", True), ("so101/moving_jaw", False), ("so100", False)],
)
def test_the_gripper_body_selects_every_geom_under_it(closed_on_cube, prefix, grasped):
    assert _grasped("cube", prefix)(closed_on_cube) is grasped


@pytest.mark.parametrize(("prefix", "refused"), [("so100", True), ("so101/gripper", False)])
def test_run_policy_refuses_a_gripper_prefix_no_body_has(closed_on_cube, prefix, refused):
    out = closed_on_cube.run_policy(
        robot_name="so101",
        policy_provider="mock",
        n_steps=1,
        stop_when={"predicate": "grasped", "body": "cube", "gripper_prefix": prefix},
    )
    text = " ".join(c.get("text", "") for c in out["content"] if isinstance(c, dict))
    assert (out["status"] == "error" and "gripper_prefix ['so100']" in text) is refused, text
