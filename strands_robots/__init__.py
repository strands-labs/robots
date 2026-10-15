"""strands-robots v1: a Simulation is a world, a Robot is a body in a world (or on a bus),
a Policy turns an Observation into an Action.

This is the ``dev`` branch: v1 is built here part by part (strands-labs/robots#4749, schedule #3818).
Each of the five concepts is a package you can read top to bottom; the top level exports the
classes a program names, and everything else is imported from its package
(``from strands_robots.simulation import Scene``):

    from strands_robots import Robot

    arm = Robot("so101")                    # joins Simulation.default(); mujoco is imported here, not before
    g1 = Robot("g1", position=[1, 0, 0])    # the SAME world: arm.sim is g1.sim
    arm.act({"shoulder_pan": 0.3})          # one control step for the world
    arm.run(lambda obs: {"elbow_flex": 0.4}, hz=50, steps=60)   # -> RunResult
    real = Robot("so101", port="/dev/ttyACM0", confirm=ask)     # a feetech bus, a twin in the default world
    real.act({"shoulder_pan": 0.3})         # writes the bus; the twin mirrors it and its world steps once
    lab = Mesh("zenoh://", tenant="lab-a")                   # every robot on the LAN; lab.mount(real)

Every refusal (the operator said no, a limit, the e-stop) raises ``RequestRefused``, a ``PermissionError``.
"""

from __future__ import annotations

from .description import Description
from .mesh import Mesh
from .policy import Policy
from .robot import Body, Robot
from .robot.safety import RequestRefused
from .simulation import Simulation
from .types import Action, JointState, Observation, Pose

__version__ = "1.0.0.dev0"

__all__ = [
    "Robot",
    "Simulation",
    "Policy",
    "Body",
    "Description",
    "Observation",
    "Action",
    "JointState",
    "Pose",
    "RequestRefused",
    "Mesh",
    "__version__",
]
