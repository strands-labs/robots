"""D08: the first snippet on a fresh install; without MuJoCo the error names the extra to install."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d08_missing_extra",
        "A missing MuJoCo says how to get it",
        "The first Robot() loads the MuJoCo backend; without the mujoco extra it raises an ImportError that "
        "names the install line, not a bare ModuleNotFoundError.",
        "Left, Robot(\"so101\") in your code on a fresh install. A wire runs right to Simulation.default(), "
        "which loads the MuJoCo backend lazily. From there a dashed wire runs up to the old outcome, a bare "
        "ModuleNotFoundError raised from inside the backend with no install line. A solid wire runs down to "
        "the green element, an ImportError whose message reads Simulation() needs MuJoCo and gives "
        "pip install 'strands-robots[mujoco]'.",
        h=470,
    )
    s.section(60, 122, "your code")
    s.box(60, 134, 300, 120, "Robot(\"so101\")", "the first snippet, on a fresh install", size=14, subsize=12,
          id="code")
    s.section(440, 122, "the world")
    s.box(440, 134, 300, 120, "Simulation.default()", "loads the MuJoCo backend the first time", size=14,
          subsize=12, id="world")
    s.arrow([(360, 194), (440, 194)], id="w1")

    s.section(820, 122, "before")
    s.box(820, 134, 320, 120, "ModuleNotFoundError", "a bare traceback from inside the backend; no install line",
          dashed=True, size=14, subsize=12, id="old")
    s.arrow([(740, 170), (820, 170)], dashed=True, label="was", label_dx=-14, label_dy=-8, id="w2")

    s.section(440, 300, "now")
    s.box(440, 312, 700, 100, "ImportError", "Simulation() needs MuJoCo: pip install 'strands-robots[mujoco]'",
          accent=True, size=14, subsize=12, id="new")
    s.arrow([(590, 254), (590, 312)], label="no mujoco", label_dx=10, label_dy=4, id="w3")

    s.footnote(442, "Every other extra already said what to type; now the one the first snippet needs does too.")
    s.motion = [("new", "pulse"), ("w1", "flow"), ("w3", "flow")]
    return s
