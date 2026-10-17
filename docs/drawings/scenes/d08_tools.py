"""D08: a device's verbs become one strands tool; an agent's command passes the operator gate."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d08_tools",
        "The verbs become one tool",
        "robot.tool and sim.tool are generated from the device's rpcs; a model calls them, the gate still holds.",
        "Left, the device's manifest: the rpcs a body declares (observe, act, run, hold, estop), each with its "
        "Python signature. A wire labelled generated leads right to the one green element, robot.tool, a single "
        "strands tool with an action enum and one parameter object per rpc, its JSON schema read from the "
        "signatures with nothing written by hand. A second wire, one tool, leads to the agent, which observes "
        "then acts in one turn. Along the bottom, the gate: inside from_agent an act becomes confirm_act, so a "
        "command from a tool goes through the safety chain and the operator; resume and calibrate stay the "
        "operator's and a model never sees them.",
        h=560,
    )
    X = 60
    s.section(X, 122, "the verbs")
    s.box(X, 134, 320, 110, "device.manifest", "each rpc and its Python signature", size=14, subsize=12, id="verbs")
    s.chips(X + 14, 206, ["observe", "act", "run", "hold", "estop"])

    s.section(X + 410, 122, "one tool", accent=True)
    s.box(X + 410, 134, 320, 110, "robot.tool", "an action enum, one parameter object per rpc; schema generated",
          accent=True, size=14, subsize=12, id="tool")
    s.chips(X + 424, 206, ["action", "inputSchema", "image blocks"])

    s.section(X + 820, 122, "the agent")
    s.box(X + 820, 134, 260, 110, "Agent(tools=[arm.tool])", "observe then act, one turn", size=14, subsize=12,
          id="agent")
    s.chips(X + 834, 206, ["observe", "act"])

    s.arrow([(X + 320, 179), (X + 410, 179)], label="generated", label_dx=-30, label_dy=-8, id="w1")
    s.arrow([(X + 730, 179), (X + 820, 179)], label="one tool", label_dx=-24, label_dy=-8, id="w2")

    s.section(X, 320, "an agent's act passes the gate")
    s.box(X, 332, 1080, 110, "with from_agent(): act -> confirm_act",
          "a command from a tool is an agent's: the safety chain and the operator gate run on every step; "
          "resume and calibrate stay the operator's", size=14, subsize=12, id="gate")
    s.chips(X + 14, 412, ["from_agent()", "confirm_act", "RequestRefused"])

    s.footnote(516, "Measured headless: a real model turn observed, then act set shoulder_pan 0.0 to 0.469 rad.")
    s.motion = [("verbs", "visit"), ("tool", "pulse"), ("agent", "visit"), ("w1", "flow"), ("w2", "flow")]
    return s
