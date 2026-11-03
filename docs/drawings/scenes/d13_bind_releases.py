"""D13: bind checks what needs no session first, and every refusal after attach detaches before it propagates."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d13_bind_releases",
        "A refused run leaves nothing attached",
        "What needs no session is refused before attach; once attach has run, every refusal calls detach "
        "before it reaches you.",
        "Top row, left to right: a box checks, the hook types and a named units scheme, refused before anything "
        "opens; a wire to a box attach, which opens a session; a wire to a box hz, units, requirements, reset; "
        "a wire to a box the loop, step one onward. Under the row a green box detach, reached by dashed refused "
        "wires from the last two boxes, then a dashed wire to a box you get the error, with nothing left open.",
        h=560,
    )
    s.section(60, 122, "no session yet")
    s.box(60, 134, 240, 130, "checks", "hook types, a named units scheme", size=14, subsize=12, id="check")
    s.section(340, 122, "acquire")
    s.box(340, 134, 220, 130, "attach", "opens a session", size=14, subsize=12, id="attach")
    s.section(600, 122, "can still refuse")
    s.box(600, 134, 260, 130, "hz, units", "requirements, reset", size=14, subsize=12, id="ready")
    s.section(900, 122, "run")
    s.box(900, 134, 240, 130, "the loop", "step one onward", size=14, subsize=12, id="loop")
    s.arrow([(300, 199), (340, 199)], id="w1")
    s.arrow([(560, 199), (600, 199)], id="w2")
    s.arrow([(860, 199), (900, 199)], id="w3")

    s.section(600, 330, "every exit path", accent=True)
    s.box(600, 342, 540, 110, "detach", "called once, before the error propagates", accent=True, size=14,
          subsize=12, id="detach")
    s.arrow([(730, 264), (730, 342)], dashed=True, label="refused", label_dx=10, label_dy=4, id="w4")
    s.arrow([(1020, 264), (1020, 342)], dashed=True, label="raised", label_dx=10, label_dy=4, id="w5")
    s.box(60, 342, 500, 110, "you get the error", "and nothing stays open", dashed=True, size=14, subsize=12,
          id="you")
    s.arrow([(600, 397), (560, 397)], dashed=True, id="w6")

    s.footnote(500, "Before, 30 runs refused after attach left 15 sessions open; now none.")
    s.motion = [("detach", "pulse"), ("w4", "flow"), ("w5", "flow"), ("w6", "flow")]
    return s
