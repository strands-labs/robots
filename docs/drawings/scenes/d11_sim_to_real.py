"""D11: the sim-to-real world: edits batched into one rebuild, a wrist camera, per-body randomization, Perturb."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "d11_sim_to_real",
        "A world built for sim to real",
        "Edits rebuild once, a camera rides the gripper, one body is randomized, and the policy sees a real arm's noise.",
        "Left, a card editing() holding four scene verbs, add_object, camera, randomize and remove, with a wire "
        "labelled one rebuild into the one green element, the world, which holds so101 with a wrist camera on "
        "so101/gripper and a cube randomized alone with target. Right, three cards lit in turn: Perturb adds "
        "joint_noise, image_noise and action_delay around any policy; the policy acts; the arm holds its measured "
        "pose until the first late action lands. A dashed return wire carries the true observation back. Under "
        "the world a dashed layer, describe, with links and deterministic, true for mujoco and newton with one "
        "env, false for 1024 envs.",
        h=640,
    )
    s.section(60, 122, "scene edits")
    s.box(60, 134, 270, 240, "sim.editing()", "the scene changes inside; the world rebuilds on exit",
          size=15, subsize=12, id="edit")
    s.chips(74, 236, ["add_object", "camera"])
    s.chips(74, 268, ["randomize", "remove"])
    s.chips(74, 300, ["set_pose"])
    s.arrow([(330, 254), (410, 254)], label="one rebuild", label_dx=0, label_dy=-8, label_anchor="middle", id="a1")

    s.section(410, 122, "the world", accent=True)
    s.box(410, 134, 360, 240, "Simulation", "so101 with a wrist camera; one cube randomized alone",
          accent=True, size=15, subsize=12, id="world")
    s.chips(424, 236, ['attach="so101/gripper"'])
    s.chips(424, 268, ['randomize(target="cube")'])
    s.chips(424, 300, ["camera=0.01", "lighting"])

    s.section(850, 122, "what the policy sees")
    s.box(850, 134, 290, 70, "Perturb", "joint_noise, image_noise around any policy", size=14, subsize=12, id="p1")
    s.box(850, 219, 290, 70, "action_delay", "each action lands that many steps late", size=14, subsize=12,
          id="p2")
    s.box(850, 304, 290, 70, "a hold", "the measured pose until the first one lands", size=14, subsize=12,
          id="p3")
    s.arrow([(770, 169), (850, 169)], label="observe", label_dx=0, label_dy=-8, label_anchor="middle", id="w1")
    s.arrow([(850, 339), (770, 339)], dashed=True, label="act", label_dx=0, label_dy=-8, label_anchor="middle",
            id="w2")

    s.down(590, 374, 430, label="describe()", label_dx=10, label_dy=4, id="d")
    s.box(60, 430, 1080, 110, None, None, dashed=True)
    s.text(74, 462, "describe()", cls="mono fg", size=15, weight=500)
    s.text(190, 462, "the links a camera can ride, and whether a replay is bit-exact",
           cls="grot text", size=12.5)
    s.chips(74, 486, ["links", "deterministic", "native", "newton", "1024 envs"])
    s.footnote(590, "Measured: newton compiles a new scene shape in 42.8 s cold and 0.63 s warm, so batch the edits.")
    s.motion = [("world", "pulse"), ("a1", "flow"), ("w1", "flow"), ("w2", "flow"), ("d", "flow"),
                ("p1", "visit"), ("p2", "visit"), ("p3", "visit")]
    return s
