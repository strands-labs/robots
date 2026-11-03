"""D11: the e-stop names its stopper and its clock; a resume takes the operator who saw that stop."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d11_estop_audit",
        "Who stopped it, who may resume",
        "An e-stop names its stopper and carries a wall clock; a resume pinned to the stop count "
        "clears only the stop the operator saw, never a later one.",
        "Left, a box estop(by=) latches the arm. A wire runs right into the green element, the Latch "
        "record, which holds chips stopped_by, reason, count and engaged_at, engaged_wall. A wire runs "
        "right from the Latch into resume(approved_by=, stop=n). From resume a solid wire goes down to a "
        "released box when stop equals count, and a dashed wire goes left to a refused box when stop does "
        "not equal count, a stale yes for a stop the operator never saw.",
        h=560,
    )
    s.section(60, 122, "the stop")
    s.box(60, 140, 300, 120, "estop(by=)", "latches now, bypasses the chain", size=14, subsize=12, id="estop")

    s.section(440, 122, "the latch record", accent=True)
    s.box(440, 140, 330, 200, "Latch", "who, why and when: an audit trail", accent=True, size=15, subsize=12,
          id="latch")
    s.chips(456, 232, ["stopped_by", "reason", "count"], accent=True)
    s.chips(456, 276, ["engaged_at", "engaged_wall"], accent=True)
    s.arrow([(360, 200), (440, 200)], label="by", label_dx=-18, label_dy=-8, id="w1")

    s.section(850, 122, "resume")
    s.box(850, 140, 290, 120, "resume(approved_by=, stop=n)", "the operator who saw that stop", size=13,
          subsize=12, id="resume")
    s.arrow([(770, 200), (850, 200)], label="count", label_dx=-30, label_dy=-8, id="w2")

    s.box(850, 400, 290, 100, "released", "stop == count: the stop they approved", size=14, subsize=12, id="ok")
    s.arrow([(995, 260), (995, 400)], label="stop == count", label_dx=12, label_dy=4, id="w3")

    s.box(440, 400, 330, 100, "refused", "stop != count: a stale yes is turned away", dashed=True, size=14,
          subsize=12, id="refused")
    s.arrow([(900, 260), (900, 360), (605, 360), (605, 400)], dashed=True, label="stop != count",
            label_dx=-120, label_dy=-6, id="w4")

    s.footnote(520, "Before, an approver's yes cleared whatever stop was latched, including one they never saw.")
    s.motion = [("latch", "pulse"), ("w1", "flow"), ("w3", "flow"), ("w4", "flow"), ("ok", "visit"),
                ("refused", "visit")]
    return s
