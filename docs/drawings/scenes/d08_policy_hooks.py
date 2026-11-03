"""D08: a policy's optional hooks are read once, before step one; a hook of the wrong type is a TypeError naming it."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d08_policy_hooks",
        "A policy's hooks, read once",
        "Before step one the loop reads every optional hook of the policy and checks its type; both loops run on "
        "what it read.",
        "Left, your policy with its seven optional hooks as chips: hz, realtime, gains, attach, detach, "
        "requirements, needs_observation. A wire runs right to the green element, before step one, where each "
        "hook is read once and its type checked. Two wires leave it: right to Robot.run and down to "
        "Simulation.run, both reading the same values, so one hz paces both. A dashed refused path runs to "
        "TypeError, which names the hook and the value, and nothing moves.",
        h=600,
    )
    s.section(60, 122, "your policy")
    s.box(60, 134, 360, 250, "policy", "act(obs) plus the optional hooks the loop honours", size=14, subsize=12,
          id="pol")
    s.chips(74, 214, ["hz", "realtime", "gains"])
    s.chips(74, 260, ["attach", "detach"])
    s.chips(74, 306, ["requirements", "needs_observation"])

    s.section(480, 122, "before step one")
    s.box(480, 134, 300, 250, "read once", "every hook read and type-checked; units and requirements too",
          accent=True, size=14, subsize=12, id="bind")
    s.arrow([(420, 259), (480, 259)], label="bind", label_dx=-14, label_dy=-8, id="w1")

    s.section(840, 122, "both loops")
    s.box(840, 134, 300, 110, "Robot.run", "paces at the hz it was given", size=14, subsize=12, id="run")
    s.box(840, 274, 300, 110, "Simulation.run", "the same hz, every robot in one world", size=14, subsize=12,
          id="sim")
    s.arrow([(780, 189), (840, 189)], id="w2")
    s.arrow([(780, 329), (840, 329)], id="w3")

    s.section(480, 430, "refused")
    s.box(480, 442, 660, 90, "TypeError", "names the hook and the value it got; nothing is written",
          dashed=True, size=14, subsize=12, id="no")
    s.arrow([(630, 384), (630, 442)], dashed=True, label="wrong type", label_dx=10, label_dy=4, id="w4")

    s.footnote(566, "Before, hz = 0 ran at 30 Hz and needs_observation = False rendered every step, silently.")
    s.motion = [("bind", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow")]
    return s
