"""D15: a closed mesh refuses every verb, so nothing it starts outlives it."""

from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d15_closed_mesh",
        "A closed mesh stays closed",
        "After close(), mount, connect, invoke, broadcast and subscribe raise RuntimeError instead of "
        "starting a watchdog and a state thread that the second close never stops.",
        "Left, before, a dashed card: a mount after close is accepted and two threads, the watchdog and "
        "the state stream, stay alive after a second close. A dashed wire labelled was runs right to the "
        "green card, now: close, then any verb raises RuntimeError and no new thread starts. Right, a card "
        "of the five refused verbs as chips: mount, connect, invoke, broadcast, subscribe.",
        h=400,
    )
    s.section(60, 122, "before")
    s.box(
        60, 134, 340, 150,
        "2 threads outlive it",
        "a mount after close is accepted; the watchdog and the state stream run after a second close",
        dashed=True, size=14, subsize=12, id="before",
    )
    s.section(430, 122, "now", accent=True)
    s.box(
        430, 134, 360, 150,
        "RuntimeError, 0 threads",
        "close, then any verb is refused by name and starts nothing",
        accent=True, size=14, subsize=12, id="now",
    )
    s.arrow([(400, 209), (430, 209)], dashed=True, label="was", label_dx=6, label_dy=-6, id="w1")
    s.section(830, 122, "refused after close")
    s.box(830, 134, 310, 150, "five verbs", "the same refusal a closed Simulation gives", size=14, subsize=12, id="verbs")
    s.chips(844, 200, ["mount", "connect", "invoke"])
    s.chips(844, 238, ["broadcast", "subscribe"])
    s.footnote(344, "close() stays idempotent; a second call is still a no-op.")
    s.motion = [("now", "pulse"), ("w1", "flow")]
    return s
