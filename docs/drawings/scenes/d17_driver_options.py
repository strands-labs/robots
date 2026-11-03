"""D17: Robot(**driver_options) reaches every driver parameter; a misspelt one is refused, a world's robot takes none."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d17_driver_options",
        "An option reaches its driver",
        "Robot(**driver_options) hands feetech robot_id= and baud= to the driver unchanged. A misspelt option is "
        "refused naming what the driver takes, and a robot in a world takes none.",
        "Left, the green element: Robot(\"so101\", port=, robot_id=, baud=) in your code, with chips robot_id, baud, "
        "cameras. A wire runs right to make_driver, which forwards the options and checks them against "
        "driver_parameters. A wire leaves it to FeetechDriver, which receives robot_id= and baud= unchanged. From "
        "make_driver a dashed wire drops to a refused box: baudrate= raises a TypeError that names what the driver "
        "takes. Bottom left, a dashed box for a body in a world: Robot(\"so101\", baud=) with no port raises a "
        "TypeError that says a world's robot takes no driver options.",
        h=540,
    )
    s.section(60, 150, "your code", accent=True)
    s.box(60, 162, 300, 150, "Robot(\"so101\", ...)", "port=, robot_id=, baud= on a real body",
          accent=True, size=14, subsize=12, id="call")
    s.chips(74, 236, ["robot_id=", "baud="])
    s.chips(74, 272, ["cameras="])

    s.section(430, 150, "forward and check")
    s.box(430, 162, 300, 150, "make_driver", "forwards the options; checks them against driver_parameters",
          size=14, subsize=12, id="make")
    s.arrow([(360, 237), (430, 237)], label="**driver_options", label_dx=-32, label_dy=-10, id="w1")

    s.section(800, 150, "the driver")
    s.box(800, 162, 340, 150, "FeetechDriver", "receives robot_id= and baud= unchanged", size=14, subsize=12,
          id="driver")
    s.chips(814, 240, ["robot_id", "baud", "cameras"])
    s.arrow([(730, 237), (800, 237)], label="options", label_dx=-18, label_dy=-10, id="w2")

    s.section(430, 384, "misspelt")
    s.box(430, 396, 710, 96, "TypeError", "the feetech driver takes no ['baudrate']; it takes ['baud', 'cameras', "
                                          "'calibration', 'port', 'robot_id']", size=14, subsize=12, id="bad")
    s.arrow([(580, 312), (580, 396)], dashed=True, label="baudrate=", label_dx=10, label_dy=4, id="w3")

    s.section(60, 384, "a body in a world")
    s.box(60, 396, 340, 96, "Robot(\"so101\", baud=)", "no port, no bus: TypeError, a world's robot takes no driver "
                                                       "options", dashed=True, size=13, subsize=11, id="world")

    s.footnote(518, "A world's robot has no bus to pass an option to; the message says so instead of swallowing it.")
    s.motion = [("call", "pulse"), ("w1", "flow"), ("w2", "flow"), ("w3", "flow")]
    return s
