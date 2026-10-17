"""The public surface after 'one concept, one package': the top level exports eleven names, no more.

Everything else is imported from its package. The classes a program names still work exactly as before.
"""

from __future__ import annotations

import strands_robots
from strands_robots import Robot, Simulation

_TOP_LEVEL = {
    "Robot",
    "Simulation",
    "Policy",
    "Body",
    "Description",
    "Observation",
    "Action",
    "JointState",
    "Pose",
    "Mesh",
    "__version__",
}


def test_the_top_level_exports_exactly_the_eleven_names():
    assert set(strands_robots.__all__) == _TOP_LEVEL


def test_mesh_is_the_same_class_at_the_top_level():
    from strands_robots.mesh import Mesh

    assert strands_robots.Mesh is Mesh


def test_names_that_left_the_top_level_import_from_their_package():
    # each name is now reached through the package that owns it
    from strands_robots.description import Gripper, Hardware  # noqa: F401
    from strands_robots.policy import CallablePolicy, PolicyLike, Requirements, SequencePolicy, as_policy  # noqa: F401
    from strands_robots.robot import RunResult, SimBody, Task  # noqa: F401
    from strands_robots.robot.drivers import Calibration, Driver, make_driver  # noqa: F401
    from strands_robots.robot.drivers.calibration import JointCalibration  # noqa: F401
    from strands_robots.simulation import Backend, Contact, Object, Physics, Scene  # noqa: F401
    from strands_robots.units import Units  # noqa: F401

    # and they are NOT re-exported at the top level
    for gone in ("Scene", "Backend", "Calibration", "RunResult", "Requirements", "Units", "SimBody", "Task"):
        assert gone not in strands_robots.__all__
        assert not hasattr(strands_robots, gone)


def test_the_named_classes_still_work_as_before():
    assert type(Robot("so101")) is Robot
    sim = Simulation()
    try:
        arm = Robot("so101", sim=sim)
        assert arm.sim is sim
        res = arm.run(lambda obs: {"shoulder_pan": 0.1}, steps=3)
        assert res.steps == 3
        out = sim.run({arm: lambda obs: obs.joints}, steps=2)
        assert out[arm.id].steps == 2
    finally:
        sim.close()
