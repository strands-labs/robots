"""A ``stop_policy`` stand-in that answers what the real verb answers.

Three test modules stand in for the simulation a
:class:`~strands_robots.device_connect.sim_driver.SimulationDeviceDriver` wraps,
and that driver's ``stop`` routes every robot through the simulation's own
``stop_policy`` and reads the ``was_running`` verdict out of the answer. A bare
``MagicMock`` answers ``hasattr("stop_policy")`` truthfully-by-fabrication,
absorbs the stop, and returns an envelope carrying no verdict -- so a stand-in
without this observes neither half of what the verb does.

One definition rather than three: the answer shape is a claim about production,
and three copies of a claim drift.
:class:`tests.test_device_connect_sim_stop_reports_the_rollouts_it_halted.TestTheStandInMatchesTheRealSimulation`
compares this function's answers to
:meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.stop_policy`
key for key, so the claim is graded rather than asserted.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any


def stop_policy_stand_in(world: Any) -> Callable[..., dict[str, Any]]:
    """Build a ``stop_policy`` over *world* answering the production envelope.

    The verdict is read off the robot record's own
    :meth:`~strands_robots.simulation.models.SimRobot.request_policy_stop`, so a
    stand-in cannot report a halt the record did not make. The real verb widens
    that with its rollout-Future registry, which a stand-in world does not have;
    for a world where the flag is the only source, the two agree.

    Args:
        world: A ``SimWorld``-like object with a ``robots`` mapping of records,
            or ``None`` for a simulation whose world is torn down.

    Returns:
        A callable with ``stop_policy``'s signature and envelope.
    """

    def stop_policy(robot_name: str = "") -> dict[str, Any]:
        robots = getattr(world, "robots", {}) if world is not None else {}
        if not robot_name or robot_name not in robots:
            return {"status": "error", "content": [{"text": f"Unknown robot '{robot_name}'."}]}
        was_running = bool(robots[robot_name].request_policy_stop())
        msg = f"Stopped on '{robot_name}'" if was_running else f"Was not running on '{robot_name}'"
        return {
            "status": "success",
            "content": [{"text": msg}, {"json": {"robot": robot_name, "was_running": was_running}}],
        }

    return stop_policy
