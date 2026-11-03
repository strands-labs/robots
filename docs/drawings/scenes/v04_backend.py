"""V04: the Backend contract. Simulation is written once; a backend is 13 methods."""
from scene import Scene

X, W = 60, 1080


def scene() -> Scene:
    s = Scene(
        "v04_backend",
        "Thirteen methods make a simulator",
        "Simulation is written once; a backend implements the Backend contract and nothing else.",
        "Top, a dashed layer: Simulation, written once for every backend, with default, manual_time, add, remove, "
        "add_object, camera, set_pose, randomize, run, describe and tool. A wire labelled Scene, plain data, runs down to "
        "the one green element: the Backend abstract class and its thirteen methods, load, step, reset, time, "
        "joint_state, set_targets, body_pose, set_pose, set_gains, render, contacts, physics and close. Bottom, three "
        "cards: mujoco (MjSpec attach with an id prefix, state carried by joint name, 29k steps per second for so101 "
        "plus g1), isaac (the one SimulationApp per process, USD import cache, 228 PhysX steps per second on an L40S) "
        "and physics newton, an axis inside the world rather than a world (mujoco_warp with a CUDA graph, 1.75M "
        "env-steps per second at num_envs 1024).",
        h=680,
    )
    s.section(X, 122, "written once")
    s.box(X, 134, W, 96, None, None, dashed=True)
    s.text(X + 14, 160, "Simulation", cls="mono fg", size=15, weight=500)
    s.text(X + 130, 160, "world-level behaviour: default world, time, ids, scene bookkeeping, locking, the agent tool",
           cls="grot text", size=12.5)
    s.chips(X + 14, 180, ["default()", "manual_time()", "add", "remove", "add_object", "camera", "set_pose",
                          "randomize", "run", "describe", "tool"])
    s.down(600, 230, 270, label="Scene: robots, objects, cameras, gravity, timestep", label_dx=10, label_dy=8,
           id="scene")

    s.section(X, 262, "the contract", accent=True)
    s.box(X, 274, W, 130, "Backend", "an abstract class; adding a robot appends to the Scene and calls load() again "
                                     "with joint state carried over", accent=True, size=15, subsize=12, id="backend")
    s.chips(X + 14, 334, ["load", "step", "reset", "time", "joint_state", "set_targets", "body_pose"])
    s.chips(X + 14, 366, ["set_pose", "set_gains", "render", "contacts", "physics", "close"])

    s.down(600, 404, 430, id="impl")
    s.arrow([(X + 180, 430), (X + W - 180, 430)], head=False)
    cards = [
        ("mujoco", "MjSpec attach with an id/ prefix; state carried by joint name across recompiles",
         ["29k steps/s", "the default"]),
        ("isaac", "the one SimulationApp per process; USD import cache; authored drives",
         ["228 PhysX steps/s", "L40S"]),
        ('physics="newton"', "an axis inside the world, not a world: mujoco_warp with a CUDA graph",
         ["num_envs=1024", "1.75M env-steps/s"]),
    ]
    bw = (W - 32) / 3
    for i, (name, sub, chips) in enumerate(cards):
        x = X + i * (bw + 16)
        s.down(x + bw / 2, 430, 454, id=f"to_{i}")
        s.box(x, 454, bw, 132, name, sub, size=14, subsize=12)
        s.chips(x + 14, 546, chips)
        s.motion.append((f"to_{i}", "flow"))
    s.footnote(628, "mjlab is not a backend: it is Simulation(\"mujoco\", physics=\"newton\", num_envs=N); "
                    "its RL scaffolding returns inside strands_robots after 1.0.")
    s.motion[:0] = [("backend", "pulse"), ("scene", "flow"), ("impl", "flow")]
    return s
