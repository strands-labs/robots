"""V03: one loop for every body. Robot.run = observe, policy, act at hz."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v03_one_loop",
        "One loop, every body",
        "Robot.run is observe, policy, act at hz; written once, it drives a simulated, a real and a remote robot the same way.",
        "Left, a cycle of three cards lit one after another: robot.observe() returns an Observation, policy.act(obs, "
        "instruction) returns an Action, robot.act(action) writes targets and advances time; the wire returns to "
        "observe. Under it the one green element, robot.run(policy, instruction, hz, seconds, record) with its options. "
        "Right, what one act() costs on each body: in a world one control step, timestep times substeps, 20 ms by "
        "default; on hardware wall time; with sim.manual_time() the caller steps; sim.run with a robot to policy "
        "mapping steps many robots in lock-step.",
        h=660,
    )
    X = 60
    s.section(X, 122, "the loop")
    s.box(X, 134, 300, 92, "robot.observe()", "joints, base pose and twist, camera images, extras", size=14,
          subsize=12, id="l_obs")
    s.box(X + 380, 134, 300, 92, "policy.act(obs)", "with the instruction; a chunked policy replays its chunk",
          size=14, subsize=12, id="l_pol")
    s.box(X + 190, 290, 300, 92, "robot.act(action)", "validated against the Description, then one step of time",
          size=14, subsize=12, id="l_act")
    s.arrow([(X + 300, 180), (X + 380, 180)], label="Observation", label_dx=-36, label_dy=-8, id="f1")
    s.arrow([(X + 530, 226), (X + 530, 336), (X + 490, 336)], label="Action", label_dx=10, label_dy=4, id="f2")
    s.arrow([(X + 190, 336), (X + 150, 336), (X + 150, 226)], id="f3")
    s.motion = [("l_obs", "visit"), ("l_pol", "visit"), ("l_act", "visit"), ("f1", "flow"), ("f2", "flow"),
                ("f3", "flow")]

    s.section(X, 420, "written once", accent=True)
    s.box(X, 432, 680, 120, "robot.run(policy, instruction=, hz=30, seconds=)",
          "checks the policy's requirements before step one, so a missing camera is a one line refusal "
          "naming it; start() runs the same loop in the background", accent=True, size=14, subsize=12, id="run")
    s.chips(X + 14, 512, ["steps=", "stop_when=", "record=", "realtime=", "start()", "stop()", "RunResult"])

    R, RW = 800, 340
    s.section(R, 122, "what one act() costs")
    rows = [
        ("in a world", "one control step: timestep x substeps, 20 ms by default"),
        ("on hardware", "wall time; act() writes the bus and mirrors into the twin"),
        ("sim.manual_time()", "act() only writes targets; the caller steps the world"),
        ("sim.run({arm: p1, g1: p2})", "many robots, many policies, lock-step"),
    ]
    for i, (t, sub) in enumerate(rows):
        s.box(R, 134 + i * 106, RW, 92, t, sub, size=14, subsize=12)

    s.footnote(616, "act() costing sim time is what keeps robot.act(policy.act(robot.observe())) identical on a "
                    "simulated and a real body.")
    return s
