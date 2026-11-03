"""D09: observe() places the twin AT the measurement; a closed world refuses; the public types are values."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d09_twin_measured",
        "The twin shows what the arm measured",
        "real.observe() places so101_twin at the measured joints with no step; a closed Simulation refuses; "
        "== and hash work on every public type.",
        "Top left, real.observe() on a hardware SO-101 reads the Driver. A wire runs right to the green element, "
        "set_joints on the MuJoCo backend: qpos becomes the measurement, qvel zero, no physics step. A second wire "
        "runs right to so101_twin, which real.twin.sim.render() now draws in the real pose. Bottom left, a dashed "
        "box: a closed Simulation, whose add, step, control_step, reset and run raise RuntimeError. Bottom right, "
        "JointState, Pose, Observation and RunResult compare by value and hash.",
        h=560,
    )
    s.section(60, 122, "your code")
    s.box(60, 134, 320, 130, "real.observe()", "six joints read off the feetech bus", size=14, subsize=12, id="obs")
    s.chips(74, 222, ["Driver", "real.twin"])
    s.section(440, 122, "the backend")
    s.box(440, 134, 320, 130, "set_joints", "qpos := measured, qvel := 0; no physics step", accent=True,
          size=14, subsize=12, id="place")
    s.section(820, 122, "Simulation.default()")
    s.box(820, 134, 320, 130, "so101_twin", "real.twin.sim.render() draws the real pose", size=14, subsize=12,
          id="twin")
    s.arrow([(380, 199), (440, 199)], label="measured", label_dx=-30, label_dy=-8, id="w1")
    s.arrow([(760, 199), (820, 199)], id="w2")

    s.section(60, 322, "a closed world")
    s.box(60, 334, 500, 130, "sim.close()", "then add, step, control_step, reset, run raise RuntimeError",
          dashed=True, size=14, subsize=12, id="closed")
    s.section(640, 322, "values")
    s.box(640, 334, 500, 130, "== and hash", "JointState, Pose, Observation, RunResult compare their arrays",
          size=14, subsize=12, id="eq")
    s.chips(654, 422, ["JointState", "Pose", "Observation", "RunResult"])

    s.footnote(510, "Before, the twin stood at zero until its world stepped about two hundred times.")
    s.motion = [("place", "pulse"), ("w1", "flow"), ("w2", "flow")]
    return s
