"""D05: acceptance check 9, a feetech Driver on a physical SO-101, four checks in one file."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d05_bench",
        "The arm on the bench",
        "Acceptance check 9 drives a physical SO-101 through the public Robot API: four checks, one hold, torque off "
        "at exit.",
        "Left, the one green element: STRANDS_SO101_PORT, the follower arm's serial port, opened as "
        "Robot(\"so101\", port=...). Wires run right to four checks visited in turn: observe() reads six measured "
        "joints; confirm_act() refused by the operator raises PermissionError and writes nothing, drawn as a "
        "dashed refused path; an allowed hold stays within 20 counts and real.twin mirrors it; calibrate("
        "interactive=False) returns LeRobot's calibration file unchanged. Bottom, without an arm every check "
        "skips and says why.",
        h=640,
    )
    s.section(60, 122, "the arm", accent=True)
    s.box(60, 134, 340, 150, "STRANDS_SO101_PORT", 'the follower\'s serial port, opened as Robot("so101", '
                                                  "port=...)", accent=True, size=14, subsize=12, id="arm")
    s.chips(74, 240, ["tests_integ/acceptance", "pyserial"])

    s.section(520, 122, "four checks, one file")
    s.box(520, 134, 300, 96, "observe()", "six measured joints, in action_keys order", size=14, subsize=12,
          id="c1")
    s.box(840, 134, 300, 96, "confirm_act(): refused", "PermissionError; no frame reaches the bus, torque "
                                                      "stays off", size=14, subsize=12, id="c2")
    s.box(520, 250, 300, 96, "confirm_act(): allowed", "a hold at the measured pose, within 20 counts",
          size=14, subsize=12, id="c3")
    s.box(840, 250, 300, 96, "calibrate(interactive=False)", "LeRobot's calibration file, read back unchanged",
          size=14, subsize=12, id="c4")
    s.arrow([(400, 182), (520, 182)], label="read", label_dx=-14, label_dy=-8, id="w1")
    s.arrow([(400, 260), (460, 260), (460, 298), (520, 298)], label="hold", label_dx=-40, label_dy=-8, id="w3")
    s.arrow([(820, 182), (840, 182)], dashed=True, id="w2")

    s.section(60, 388, "the twin")
    s.box(60, 400, 500, 120, "real.twin", "a SimBody in Simulation.default(); after the hold it shows the "
                                          "measured joints within 0.05 rad", size=14, subsize=12, id="twin")
    s.arrow([(670, 346), (670, 460), (560, 460)], label="mirrors", label_dx=8, label_dy=-8, id="mirror")
    s.section(700, 388, "no arm")
    s.box(700, 400, 440, 120, "every check skips", "and names the reason: set STRANDS_SO101_PORT; torque off "
                                                  "at exit on every arm the file opened", dashed=True, size=14,
          subsize=12)

    s.footnote(590, "The same four checks run against a fake bus in tests/; here the wire and the servos are real.")
    s.motion = [("arm", "pulse"), ("w1", "flow"), ("w3", "flow"), ("mirror", "flow"),
                ("c1", "visit"), ("c2", "visit"), ("c3", "visit"), ("c4", "visit")]
    return s
