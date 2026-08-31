"""A UR action recorded in simulation reaches the real wire without a remap.

The driver's :data:`~strands_robots.drivers.ur.JOINT_NAMES` is the order RTDE
uses: ``getActualQ`` returns it and ``servoJ`` expects it. The MuJoCo assets for
both arms declare their joints in that same order, which is what makes a policy
trained in simulation deployable on the arm - the observation keys the simulator
reports *are* the action keys the controller accepts.

That agreement is a coincidence of two independent sources, so it is pinned
rather than assumed. If a future asset revision reorders its joints, or the
driver's tuple is edited, this fails here - where the answer is a renamed key -
instead of on a real arm, where the answer is a shoulder moving to a wrist's
setpoint.

Needs MuJoCo and the arm's asset, so it skips where neither is available; the
wire order itself is graded without either in
``test_ur_wire_encoding.py``.
"""

from __future__ import annotations

from typing import Any

import pytest

from strands_robots.drivers.ur import JOINT_NAMES, SUPPORTED_ROBOTS, targets_from_action


def _sim_or_skip(robot: str) -> Any:
    """Build the simulation for ``robot``, or skip when it cannot be built.

    Returns the value from inside its own ``try`` rather than binding it for a
    later read, so the name is never bound only on the non-skipping path.

    Args:
        robot: Registry name of the arm to load.

    Returns:
        A live simulation holding ``robot``.
    """
    pytest.importorskip("mujoco", reason="the sim asset is needed to read its joint order")
    from strands_robots import Robot

    try:
        return Robot(robot, mode="sim")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        pytest.skip(f"the {robot} asset is unavailable here: {exc}")


def _sim_joint_names(robot: str) -> list[str]:
    """Return the joint names the MuJoCo asset for ``robot`` declares, in order."""
    sim = _sim_or_skip(robot)
    try:
        return list(sim.robot_joint_names(robot))
    finally:
        sim.cleanup()


@pytest.mark.parametrize("robot", SUPPORTED_ROBOTS)
def test_the_sim_joint_order_is_the_rtde_wire_order(robot: str) -> None:
    """Both arms, because the driver serves both from one joint tuple."""
    assert _sim_joint_names(robot) == list(JOINT_NAMES)


def test_a_sim_observation_encodes_to_the_same_vector_the_sim_holds() -> None:
    """The end-to-end claim: sim keys in, an ordered controller setpoint out.

    Encodes the simulator's own joint reading as if it were a policy's action
    and checks the six-vector matches the reading element for element. A remap
    error anywhere between the two would show up as a transposition here.
    """
    names = _sim_joint_names("ur5e")
    reading = {name: 0.1 * (index + 1) for index, name in enumerate(names)}

    targets, reason = targets_from_action(reading, [0.0] * 6, model="ur5e")

    assert reason is None, reason
    assert targets == [reading[name] for name in JOINT_NAMES]
