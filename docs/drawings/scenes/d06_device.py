"""D06: a Robot is a device; every agent write passes one safety chain, and the e-stop goes around it."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d06_device",
        "A Robot is a device",
        "One generated manifest says what it is and what it accepts. Every agent or dashboard write passes one "
        "safety chain; estop() goes around it and latches.",
        "Left, the one green element: robot.manifest, generated from the Description and the body, with sensors, "
        "controls (range, safe, on_disconnect), rpcs and events. A wire runs right into the safety chain, five "
        "checks in order: range, safe, interlocks, validators, the operator gate. A wire leaves the chain to the "
        "body, a SimBody or a Driver. A refused write drops down a dashed wire to RequestRefused, which names the "
        "joint, the value and the bound. Above the chain, estop() runs straight from the robot to the body, "
        "bypassing every check, and latches until resume(approved_by=). Below, the watchdog applies each "
        "control's on_disconnect when the command path goes quiet.",
        h=660,
    )
    s.section(60, 214, "what it is", accent=True)
    s.box(60, 226, 250, 196, "robot.manifest", "generated, never hand-written; describe() returns it",
          accent=True, size=14, subsize=12, id="manifest")
    s.chips(74, 300, ["sensors", "controls"])
    s.chips(74, 336, ["rpcs", "events"])
    s.chips(74, 372, ["device_type", "tenant"])

    s.section(370, 214, "the safety chain, in order")
    s.box(370, 226, 520, 196, None, None, dashed=True)
    s.box(390, 246, 150, 70, "range", "hard joint limits", size=13, subsize=11)
    s.box(560, 246, 150, 70, "safe", "soft bound: ask once", size=13, subsize=11)
    s.box(730, 246, 140, 70, "interlocks", "other values agree", size=13, subsize=11)
    s.box(390, 332, 230, 70, "validators", "pure checks, refusal verbatim", size=13, subsize=11)
    s.box(640, 332, 230, 70, "operator gate", "confirm= on a real body", size=13, subsize=11)
    s.arrow([(540, 281), (560, 281)], id="c1")
    s.arrow([(710, 281), (730, 281)], id="c2")
    s.arrow([(620, 367), (640, 367)], id="c3")

    s.section(950, 214, "the body")
    s.box(950, 226, 190, 196, "SimBody | Driver", "write() the targets; estop() is the fastest stop the bus has",
          size=14, subsize=12, id="body")
    s.chips(964, 336, ["hold()", "estop()"])
    s.chips(964, 372, ["position_hold"])
    s.arrow([(1045, 486), (1045, 422)], dashed=True, label="lapse", label_dx=10, label_dy=4, id="lapse")
    s.arrow([(310, 324), (370, 324)], label="act", label_dx=-14, label_dy=-8, id="w_in")
    s.arrow([(890, 324), (950, 324)], label="targets", label_dx=-24, label_dy=-8, id="w_out")

    s.arrow([(185, 226), (185, 150), (1045, 150), (1045, 226)], label="estop(): no chain, no lock; latches",
            label_dx=10, label_dy=4, id="estop")
    s.section(370, 120, "around the chain")

    s.arrow([(630, 422), (630, 486)], dashed=True, label="refused", label_dx=10, label_dy=4, id="refused")
    s.section(370, 474, "refused")
    s.box(370, 486, 520, 84, "RequestRefused", "elbow_flex=3.1000 rad is outside its range [-1.7453, 1.5708]: an agent "
                                               "can correct its next command", size=14, subsize=12)
    s.section(950, 474, "the watchdog")
    s.box(950, 486, 190, 84, "on_disconnect", "hold, zero, safe or apply", size=14, subsize=12)
    s.box(60, 486, 250, 84, "resume(approved_by=)", "only a named operator releases the latch", size=13,
          subsize=11)

    s.footnote(610, "Validators are advisory command checks, not functional safety. The developer's own act() is "
                    "the operator's and skips the chain; the e-stop never does.")
    s.motion = [("manifest", "pulse"), ("w_in", "flow"), ("c1", "flow"), ("c2", "flow"), ("c3", "flow"),
                ("w_out", "flow"), ("estop", "flow"), ("refused", "flow"), ("lapse", "flow")]
    return s
