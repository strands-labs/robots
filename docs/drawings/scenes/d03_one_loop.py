"""D03: Robot is a class. Robot.run = observe, policy, act at hz, and returns a RunResult."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d03_one_loop",
        "One loop, written once",
        "Robot.run is observe, policy, act at hz; it drives a body in a world and returns what it measured.",
        "Left, a cycle of three cards lit one after another: robot.observe() returns an Observation, policy.act(obs) "
        "returns an Action, robot.act(action) writes targets and advances time; the wire returns to observe. Under "
        "it the one green element, robot.run(policy, hz, steps) with its options, which returns a RunResult. Right, "
        "what the loop accepts and what one act() costs: a Policy subclass or a bare callable; in a world one "
        "control step of 20 ms; with sim.manual_time() the caller steps; sim.run steps many robots in lock-step.",
        h=660,
    )
    X = 60
    s.section(X, 122, "the loop")
    s.box(X, 134, 300, 92, "robot.observe()", "joints, base pose and twist, camera images", size=14,
          subsize=12, id="l_obs")
    s.box(X + 380, 134, 300, 92, "policy.act(obs)", "an Action: a vector or a joint mapping",
          size=14, subsize=12, id="l_pol")
    s.box(X + 190, 290, 300, 92, "robot.act(action)", "checked against the Description, clipped to limits",
          size=14, subsize=12, id="l_act")
    s.arrow([(X + 300, 180), (X + 380, 180)], label="Observation", label_dx=-36, label_dy=-8, id="f1")
    s.arrow([(X + 530, 226), (X + 530, 336), (X + 490, 336)], label="Action", label_dx=10, label_dy=4, id="f2")
    s.arrow([(X + 190, 336), (X + 150, 336), (X + 150, 226)], id="f3")

    s.section(X, 420, "written once", accent=True)
    s.box(X, 432, 680, 120, "robot.run(policy, hz=50, steps=60)",
          "checks the policy's requirements before step one, so a missing camera is a one line refusal "
          "naming it; start() runs the same loop in the background", accent=True, size=14, subsize=12, id="run")
    s.chips(X + 14, 512, ["seconds=", "stop_when=", "units=", "realtime=", "start()", "stop()", "RunResult"])

    R, RW = 800, 340
    s.section(R, 122, "what it takes, what it costs")
    rows = [
        ("Policy or callable", "a subclass with act(obs), or any function of obs"),
        ("in a world", "one act() is one control step: 20 ms by default"),
        ("sim.manual_time()", "act() only writes targets; the caller steps the world"),
        ("sim.run({arm: p1, g1: p2})", "many robots, many policies, lock-step"),
    ]
    for i, (t, sub) in enumerate(rows):
        s.box(R, 134 + i * 106, RW, 92, t, sub, size=14, subsize=12, id=f"r{i}")

    s.footnote(616, "Measured headless: the so101 run above reaches 0.40 rad on shoulder_pan in 1.2 s of sim time.")
    s.motion = [("l_obs", "visit"), ("l_pol", "visit"), ("l_act", "visit"), ("f1", "flow"), ("f2", "flow"),
                ("f3", "flow"), ("run", "pulse")]
    return s
