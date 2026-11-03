"""D05: one concept, one package; a type lives with its owner and import-linter guards the edges."""

from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d05_packages",
        "One concept, one package",
        "Each of the five concepts is a package you can read top to bottom; its implementations live "
        "inside it, and a type lives with its owner.",
        "Top, the one green element: robot/, the package a program drives, holding robot/run/ (the one "
        "loop) and robot/drivers/ (feetech, unitree, microduck). Below it a middle band of simulation/ "
        "(with backends/mujoco/ inside), policy/ and units/. A bottom band of description/, types/ and "
        "geometry.py. Arrows run only down the allowed import edges: robot/ to simulation/ and policy/, "
        "policy/ to units/, units/ and simulation/ to description/, description/ and geometry.py to "
        "types/. types/ imports nothing else, the leaf every arrow points at. import-linter keeps these "
        "edges on CI.",
        h=640,
    )
    s.section(60, 118, "the package a program drives", accent=True)
    s.box(60, 130, 1080, 118, "robot/", "a body in a world or on a bus: observe, act, run",
          accent=True, id="robot")
    s.box(620, 152, 240, 80, "robot/run/", "the one loop", dashed=True, size=13, subsize=11, id="run")
    s.box(884, 152, 240, 80, "robot/drivers/", "feetech, unitree, microduck", dashed=True,
          size=13, subsize=11, id="drivers")

    s.section(60, 288, "the three middle concepts")
    s.box(60, 300, 440, 128, "simulation/", "one world; owns time", id="sim")
    s.box(300, 330, 184, 78, "backends/mujoco/", "the first backend", dashed=True, size=12, subsize=10, id="mj")
    s.box(540, 300, 268, 128, "policy/", "Observation to Action", id="pol")
    s.box(848, 300, 292, 128, "units/", "rad and lerobot schemes", id="units")

    s.section(60, 466, "a type lives with its owner")
    s.box(60, 478, 372, 104, "description/", "what a robot is: registry data", id="desc")
    s.box(472, 478, 288, 104, "types/", "JointState, Action, Pose, Observation", id="types")
    s.box(800, 478, 340, 104, "geometry.py", "quaternions and frames", id="geom")

    s.arrow([(300, 248), (300, 300)], label="run()", label_dx=8, label_dy=4, id="e_rs")
    s.arrow([(680, 248), (680, 300)], id="e_rp")
    s.arrow([(808, 360), (848, 360)], label="units", label_dx=6, label_dy=-8, id="e_pu")
    s.arrow([(240, 428), (240, 478)], id="e_sd")
    s.arrow([(900, 428), (660, 478)], id="e_ut")
    s.arrow([(432, 530), (472, 530)], label="types", label_dx=-42, label_dy=-8, id="e_dt")
    s.arrow([(800, 530), (760, 530)], id="e_gt")

    s.footnote(606, "types/ is the leaf: it imports nothing else in the package, and five import-linter "
                    "contracts fail CI on a wrong edge.")
    s.motion = [("robot", "pulse"), ("e_rs", "flow"), ("e_rp", "flow"), ("e_pu", "flow"),
                ("e_sd", "flow"), ("e_ut", "flow")]
    return s
