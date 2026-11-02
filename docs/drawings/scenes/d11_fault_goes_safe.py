"""D11: when a run ends in a fault, the body goes to its declared on_disconnect state, not frozen at the last command."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d11_fault_goes_safe",
        "A fault leaves the arm somewhere safe",
        "A policy that can no longer act ends the run with stopped_by fault; every control is then written to "
        "its declared on_disconnect target, so the body is left somewhere safe, not frozen at the last command.",
        "Left, a box run() ends in a fault: a lost server, a NaN target, a stale remote frame. A wire labelled "
        "on fault runs right to the green box on_disconnect, which writes each control once. A wire runs on to "
        "the body, left somewhere safe, not frozen. Under the green box four chips name the modes: hold, zero, "
        "safe, apply. A dashed box at the bottom notes that in lock-step only the robot that faulted is moved; "
        "the rest keep running.",
        h=600,
    )
    s.section(60, 122, "a run ends")
    s.box(60, 134, 320, 130, 'stopped_by = "fault"', "a lost server, a NaN target, a stale frame",
          size=14, subsize=12, id="fault")
    s.section(470, 122, "then", accent=True)
    s.box(470, 134, 340, 130, "on_disconnect", "each control's declared target, once", accent=True,
          size=14, subsize=12, id="apply")
    s.section(900, 122, "the body")
    s.box(900, 134, 240, 130, "left safe", "not frozen at the last command", size=14, subsize=12, id="body")
    s.arrow([(380, 199), (470, 199)], label="on fault", label_dx=-30, label_dy=-8, id="w1")
    s.arrow([(810, 199), (900, 199)], label="write", label_dx=-22, label_dy=-8, id="w2")

    s.chips(470, 320, ["hold", "zero", "safe", "apply"])
    s.arrow([(640, 264), (640, 314)], dashed=True, label="per control", label_dx=12, label_dy=4, id="w3")

    s.section(60, 440, "many robots, one world")
    s.box(60, 452, 1080, 100, "lock-step", "only the robot that faulted is moved to its on_disconnect state; the "
          "rest keep running", dashed=True, size=14, subsize=12, id="lockstep")

    s.footnote(580, "A real arm is never left holding wherever a failed policy happened to put it.")
    s.motion = [("apply", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow"), ("lockstep", "visit")]
    return s
