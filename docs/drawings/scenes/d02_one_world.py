"""D02: Simulation.default(): one world holds every robot, owns time, and a Backend steps it."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d02_one_world",
        "One world, many robots",
        "Every robot given no world joins Simulation.default(); one act() is one control step for all of them.",
        "Left, two robot cards, so101 and unitree_g1, each with its Description; a wire labelled add() runs from "
        "each into the one green element, Simulation.default(), which holds the Scene and owns time, with robots, "
        "revision, lock, manual_time() and describe(). Right, three cards lit in turn: control_step(), one control "
        "step of 20 ms for the whole world; set_pose, a teleport that lands; render(), one overview frame that "
        "shows both robots. Under the world a dashed layer, the Backend contract, with load, step, joint_state, "
        "set_targets, body_pose, set_pose and render, implemented by mujoco.",
        h=640,
    )
    s.section(60, 122, "robots")
    s.box(60, 134, 260, 110, "so101", "a fixed-base arm; 6 action_keys", size=15, subsize=12, id="arm")
    s.chips(74, 200, ["Description"])
    s.box(60, 264, 260, 110, "unitree_g1", "a floating humanoid; 29 joints", size=15, subsize=12, id="g1")
    s.chips(74, 330, ["Description"])

    s.section(400, 122, "the world", accent=True)
    s.box(400, 134, 380, 240, "Simulation.default()", "holds the Scene, owns time; surviving bodies keep "
                                                      "their joint state on every rebuild",
          accent=True, size=15, subsize=12, id="world")
    s.chips(414, 236, ["sim.robots", "revision", "lock"])
    s.chips(414, 268, ["manual_time()", "describe()"])
    s.chips(414, 300, ["add_object", "camera"])

    s.arrow([(320, 189), (400, 189)], label="add()", label_dx=0, label_dy=-8, label_anchor="middle", id="a1")
    s.arrow([(320, 319), (400, 319)], label="add()", label_dx=0, label_dy=-8, label_anchor="middle", id="a2")

    s.section(860, 122, "what it does")
    s.box(860, 134, 280, 70, "control_step()", "20 ms for every robot at once", size=14, subsize=12, id="c1")
    s.box(860, 219, 280, 70, "set_pose", "a teleport that lands; time stays", size=14, subsize=12, id="c2")
    s.box(860, 304, 280, 70, "render()", "one overview frame shows both", size=14, subsize=12, id="c3")
    s.arrow([(780, 254), (860, 254)], id="w3")

    s.down(590, 374, 430, label="Scene", label_dx=10, label_dy=4, id="scene")
    s.box(60, 430, 1080, 110, None, None, dashed=True)
    s.text(74, 462, "Backend", cls="mono fg", size=15, weight=500)
    s.text(170, 462, "an abstract class; mujoco is the first: one MjSpec, each robot under an id/ prefix",
           cls="grot text", size=12.5)
    s.chips(74, 486, ["load", "step", "joint_state", "set_targets", "body_pose", "set_pose", "render", "mujoco"])
    s.footnote(590, "Measured headless: 2000 physics steps of so101 plus unitree_g1 take 85 us each.")
    s.motion = [("world", "pulse"), ("a1", "flow"), ("a2", "flow"), ("scene", "flow"),
                ("c1", "visit"), ("c2", "visit"), ("c3", "visit")]
    return s
