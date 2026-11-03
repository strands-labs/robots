"""D18: a LeRobot leader arm is a policy; its bus declares the units, the follower converts through its calibration."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d18_leader_source",
        "A leader arm is a policy",
        "Hand a LeRobot Teleoperator to run(): the leader's bus declares the units, and the follower converts "
        "every reading through its own calibration, exactly as LeRobot's teleoperate does.",
        "Left, the green element: a LeRobot Teleoperator, the leader, with chips Teleoperator and degrees; its bus "
        "declares each motor's norm mode. A wire runs right to SourcePolicy, which orders the leader's reading by "
        "the follower's action_keys and declares the exact units; a chip action_keys. A wire runs right to run() "
        "and act() on the follower, which convert in the leader's units through the follower's own calibration; a "
        "chip calibration, and a note that it matches LeRobot to within one encoder count. Below SourcePolicy a "
        "dashed wire drops to a refused box: a source that declares no units raises a TypeError naming the fix.",
        h=540,
    )
    s.section(60, 150, "the leader", accent=True)
    s.box(60, 162, 320, 150, "leader", "a LeRobot Teleoperator; its bus declares each motor's norm mode",
          accent=True, size=14, subsize=12, id="leader")
    s.chips(74, 240, ["Teleoperator", "degrees"])

    s.section(450, 150, "as a policy")
    s.box(450, 162, 320, 150, "SourcePolicy", "orders the reading by the follower's action_keys; declares the "
          "exact units", size=14, subsize=12, id="src")
    s.chips(464, 252, ["action_keys"])
    s.arrow([(380, 237), (450, 237)], label="its reading", label_dx=-24, label_dy=-10, id="w1")

    s.section(840, 150, "the follower")
    s.box(840, 162, 300, 150, "run() / act()", "convert in the leader's units, through the follower's own "
          "calibration", size=14, subsize=12, id="run")
    s.chips(854, 252, ["calibration"])
    s.arrow([(770, 237), (840, 237)], label="one reading", label_dx=-20, label_dy=-10, id="w2")

    s.section(450, 384, "no declared units")
    s.box(450, 396, 690, 96, "TypeError", "a source with no .bus.motors: its units are unknown; pass units= to "
          "run(), or a LeRobot Teleoperator", dashed=True, size=14, subsize=12, id="bad")
    s.arrow([(610, 312), (610, 396)], dashed=True, label="no units", label_dx=10, label_dy=4, id="w3")

    s.footnote(518, "A degrees leader drives a range mode follower to within one encoder count, like teleoperate.")
    s.motion = [("leader", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow")]
    return s
