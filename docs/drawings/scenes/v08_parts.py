"""V08: the refactor, part by part. Twelve parts, each one PR with a gate."""
from scene import Scene

PARTS = [
    ("1", "types + Description", "Observation, Action, Pose, JointState; registry JSON loaded once"),
    ("2", "Simulation + MuJoCo", "Backend contract, default world, shared-world test"),
    ("3", "Robot is a class", "one class for sim, hardware and remote bodies"),
    ("4", "Policy", "one factory, ChunkedPolicy, lerobot, onnx, remote"),
    ("5", "drivers + twin + gate", "feetech, unitree, microduck, Robot(confirm=)"),
    ("6", "isaac + newton", "Isaac singleton, physics=\"newton\", mjlab folds in"),
    ("7", "the mesh", "Mesh, use_mesh, zenoh and AWS IoT"),
    ("8", "agent tools", "robot.tool and sim.tool generated from the verbs"),
    ("9", "recording", "run(record=) writes LeRobot v3"),
    ("10", "dashboard", "one world, four routes, MuJoCo WASM twin"),
    ("11", "dev becomes main", "one release PR; v0.5.x kept on its own branch"),
    ("12", "docs + release", "the site, notebooks, v1.0.0"),
]


def scene() -> Scene:
    s = Scene(
        "v08_parts",
        "The refactor, part by part",
        "Each part is one pull request with a gate; it merges into dev when the gate is green, and the next part starts there.",
        "Twelve numbered cards in three rows of four, lit one after another: 1 types and Description, 2 Simulation and "
        "the MuJoCo backend, 3 Robot becomes a class, 4 Policy, 5 drivers, twin and operator gate, 6 Isaac and Newton, "
        "7 the mesh, 8 generated agent tools, 9 recording to LeRobot v3, 10 the dashboard, 11 dev becoming "
        "main, 12 docs and the v1.0.0 release. Part 1, the one green element, is where the first "
        "agent starts. Wires run left to right along each row and down to the next row. Footnote: parts 4, 5 and 6 "
        "can run in parallel once part 3 has merged.",
        h=660,
    )
    X, W, GAP = 60, 1080, 24
    bw = (W - GAP * 3) / 4
    bh = 100
    for i, (n, t, sub) in enumerate(PARTS):
        r, c = divmod(i, 4)
        x, y = X + c * (bw + GAP), 134 + r * (bh + 48)
        s.box(x, y, bw, bh, f"{n}. {t}", sub, accent=(i == 0), size=14, subsize=12, id=f"p{n}")
        s.motion.append((f"p{n}", "visit"))
        if c < 3:
            s.arrow([(x + bw, y + bh / 2), (x + bw + GAP, y + bh / 2)])
        elif r < 2:
            s.arrow([(x + bw / 2, y + bh), (x + bw / 2, y + bh + 14), (X + bw / 2, y + bh + 14),
                     (X + bw / 2, y + bh + 48)], dashed=True)
    s.section(X, 560, "rules for every part")
    s.chips(X, 570, ["one part, one PR", "failing test first", "gate green on dev", "no **kwargs",
                          "core imports nothing heavy", "deviations written down"])
    s.footnote(622, "Parts 4, 5 and 6 can run in parallel once part 3 has merged; 7 to 10 need only 3; 11 waits for "
                    "all of them.")
    return s
