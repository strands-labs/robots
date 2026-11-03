"""D10: what is refused before the bus: a NaN target, a driver whose names are not the action keys."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d10_what_never_reaches_the_arm",
        "What never reaches the arm",
        "A NaN target, a mislabelled Driver and a calibrate() with no human are refused before one byte is "
        "written; a missing LeRobot file says where it looked.",
        "Top left, act(a) and run(policy) feed one box, coerce(), the green element. A wire runs right to the "
        "Driver, which writes a servo goal on the bus. A dashed wire runs down from coerce() to a dashed "
        "refused box: NonFiniteTarget, a ValueError, and a run that ends with stopped_by fault. Bottom left, "
        "Robot(driver=) checks names against action_keys in order. Bottom middle, calibration= and the LeRobot "
        "0.6 search path so_follower. Bottom right, calibrate() with no terminal refuses before torque goes off.",
        h=600,
    )
    s.section(60, 122, "your code or a policy")
    s.box(60, 134, 300, 120, "act(a) / run(policy)", "every target, from any caller", size=14, subsize=12, id="src")
    s.section(440, 122, "before the bus")
    s.box(440, 134, 320, 120, "coerce()", "a NaN or inf is refused here, not clipped", accent=True, size=14,
          subsize=12, id="coerce")
    s.section(840, 122, "the bus")
    s.box(840, 134, 300, 120, "Driver: feetech", "a servo goal written only when finite", size=14, subsize=12,
          id="bus")
    s.arrow([(360, 194), (440, 194)], label="targets", label_dx=-28, label_dy=-8, id="w1")
    s.arrow([(760, 194), (840, 194)], label="finite", label_dx=-22, label_dy=-8, id="w2")

    s.box(440, 300, 320, 110, "NonFiniteTarget", "a ValueError; run() ends as a fault", dashed=True,
          size=14, subsize=12, id="nan")
    s.arrow([(600, 254), (600, 300)], dashed=True, label="NaN", label_dx=10, label_dy=4, id="w3")

    s.section(60, 440, "refused or found at connect")
    s.box(60, 452, 320, 100, "Robot(driver=)", "names must be action_keys, in order", size=14, subsize=12,
          id="names")
    s.box(440, 452, 320, 100, "calibration=", "or so_follower/<id>.json, LeRobot 0.6", size=14, subsize=12,
          id="cal")
    s.box(820, 452, 320, 100, "calibrate()", "no terminal: refused, nothing written", size=14,
          subsize=12, id="human")

    s.footnote(580, "Before, NaN rounded to range_min: a full-speed move of the arm to its end stop.")
    s.motion = [("coerce", "pulse"), ("w1", "flow"), ("w3", "flow"), ("names", "visit"), ("cal", "visit"),
                ("human", "visit")]
    return s
