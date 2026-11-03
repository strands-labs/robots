"""D12: one tick of a run is one read, one decision and one write, paced on one clock, shared by lockstep."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d12_one_tick",
        "One read, one write, one clock",
        "Each tick reads the body once, decides, and writes once; writes are spaced one period apart on the "
        "robot's clock, and a late one is counted, never caught up.",
        "Top row, left to right: a box observe() reads the bus once; a wire carries the observation to a box "
        "decide, where joints the policy leaves out hold that same read; a wire carries targets to the green box "
        "command(), the write, which never reads. Under the green box, chips name what a run counts: "
        "deadline_misses, worst_late_s, on_miss. A dashed box below the row says the clock: robot.clock, the "
        "world's time in a world, the monotonic clock on a bus. A dashed box at the bottom says lockstep runs "
        "the same tick for every robot, and one fault ends the tick for all.",
        h=620,
    )
    s.section(60, 122, "one tick")
    s.box(60, 134, 300, 130, "observe()", "the only read of the body", size=14, subsize=12, id="obs")
    s.box(450, 134, 300, 130, "decide", "joints left out hold that read", size=14, subsize=12, id="decide")
    s.section(840, 122, "the write", accent=True)
    s.box(840, 134, 300, 130, "command()", "the write; it never reads", accent=True, size=14, subsize=12,
          id="write")
    s.arrow([(360, 199), (450, 199)], label="obs", label_dx=-14, label_dy=-8, id="w1")
    s.arrow([(750, 199), (840, 199)], label="targets", label_dx=-30, label_dy=-8, id="w2")
    s.chips(828, 284, ["deadline_misses", "worst_late_s", "on_miss"])

    s.section(60, 330, "the clock")
    s.box(60, 342, 1080, 100, "robot.clock", "a world's time in a world; the monotonic clock on a bus",
          dashed=True, size=14, subsize=12, id="clock")

    s.section(60, 470, "many robots, one world")
    s.box(60, 482, 1080, 90, "lockstep", "the same tick for every robot; one fault ends the tick for all",
          dashed=True, size=14, subsize=12, id="lockstep")

    s.footnote(600, "After a stall the next write still waits one period: no burst of stale commands.")
    s.motion = [("obs", "visit"), ("decide", "visit"), ("write", "pulse"), ("w1", "flow"), ("w2", "flow")]
    return s
