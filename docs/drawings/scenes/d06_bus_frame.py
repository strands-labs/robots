"""D06: the feetech bus reads the reply it was told about, not a guess that waits out the timeout."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d06_bus_frame",
        "Read what the frame says",
        "FeetechBus reads FF FF ID LEN, then exactly LEN bytes: a read costs the reply, not the timeout.",
        "Left, the host writes a request frame to the servos on the half-duplex wire. Centre, the one green "
        "element: the bus reads the four-byte head FF FF ID LEN, then exactly LEN more bytes, and the six "
        "servos of a sync_read answer one frame each in id order, in about a millisecond. Right, a dashed "
        "refused path: asking for two bytes more than the frame holds waits out the whole timeout, 50 ms a "
        "read. Bottom, two wires the bus tolerates: an adapter that echoes the request, and a stray byte "
        "ahead of the header; both are skipped.",
        h=640,
    )
    s.section(60, 122, "the host writes")
    s.box(60, 134, 320, 150, "request", "FF FF ID LEN INSTR params CHK: ping, read, write or sync_read",
          size=14, subsize=12, id="req")
    s.chips(74, 240, ["FeetechBus", "pyserial"])

    s.section(460, 122, "the bus reads", accent=True)
    s.box(460, 134, 340, 150, "FF FF ID LEN, then LEN bytes", "one status frame per servo, in id order; "
          "six servos answer in about a millisecond", accent=True, size=14, subsize=12, id="frame")
    s.chips(474, 240, ["ping", "read", "sync_read"])
    s.arrow([(380, 210), (460, 210)], label="reply", label_dx=-20, label_dy=-8, id="w1")

    s.section(880, 122, "a guess")
    s.box(880, 134, 260, 150, "LEN + 2 bytes", "two bytes that never come: the read waits out the timeout, "
          "50 ms each", dashed=True, size=14, subsize=12, id="guess")
    s.arrow([(800, 210), (880, 210)], dashed=True, id="w2")

    s.section(60, 388, "wires it tolerates")
    s.box(60, 400, 520, 120, "an echoing adapter", "hears its own request first; a frame equal to the request "
          "is skipped", size=14, subsize=12, id="echo")
    s.box(620, 400, 520, 120, "a stray byte", "ahead of FF FF; the bus slides one byte at a time to the "
          "header", size=14, subsize=12, id="noise")
    s.arrow([(320, 400), (320, 340), (560, 340), (560, 284)], label="skipped", label_dx=8, label_dy=-8,
            id="w3")
    s.arrow([(880, 400), (880, 340), (700, 340), (700, 284)], dashed=False, id="w4")

    s.footnote(590, "The timeout only bounds a servo that does not answer: 20 ms, against a healthy reply under 1 ms.")
    s.motion = [("frame", "pulse"), ("w1", "flow"), ("w3", "flow"), ("w4", "flow"),
                ("req", "visit"), ("echo", "visit"), ("noise", "visit")]
    return s
