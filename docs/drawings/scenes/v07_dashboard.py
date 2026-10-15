"""V07: the dashboard is a twin of one Simulation."""
from scene import Scene


def scene() -> Scene:
    s = Scene(
        "v07_dashboard",
        "The dashboard is a twin of one world",
        "Physics stays on the server; the browser loads the same merged MJCF and gets joint values by name at 30 Hz.",
        "Left, the server: one Simulation with every robot in it, and one strands Agent whose tools are sim.tool, each "
        "robot.tool and use_mesh. Middle, four routes: GET /api/scene (merged MJCF plus assets plus revision), "
        "WS /ws/state (joint values by name at 30 Hz, about 1.5 KB a frame), WS /ws/chat (the agent's stream), POST "
        "/api/robots/{id}/act (X-Confirm for a real robot). Right, the one green element: the browser, MuJoCo WASM "
        "running mj_forward on the received joints and three.js drawing. Under it standalone mode, no server: the "
        "browser runs mj_step and an ONNX policy itself, the duck balancing on the docs site.",
        h=640,
    )
    s.section(60, 122, "server, one process")
    s.box(60, 134, 340, 150, "Simulation", "every robot in one world; the dashboard is a reader of it, "
                                           "not a process per robot", size=14, subsize=12)
    s.chips(74, 222, ["sim.describe()", "sim.lock"])
    s.chips(74, 252, ["revision"])
    s.box(60, 304, 340, 150, "Agent(tools=[...])", "one strands agent for the whole world", size=14, subsize=12)
    s.chips(74, 368, ["sim.tool", "robot.tool"])
    s.chips(74, 398, ["use_mesh"])

    s.section(440, 122, "four routes")
    routes = [
        ("GET /api/scene", "merged MJCF, assets, revision", "r1"),
        ("WS /ws/state", "joints by name, 30 Hz, 1.5 KB a frame", "r2"),
        ("WS /ws/chat", "text, tool_use, tool_result", "r3"),
        ("POST /api/robots/{id}/act", "X-Confirm for a real robot", "r4"),
    ]
    for i, (t, sub, rid) in enumerate(routes):
        y = 134 + i * 82
        s.box(440, y, 320, 70, t, sub, size=13, subsize=12)
        s.arrow([(400, y + 35), (440, y + 35)], head=False)
        s.arrow([(760, y + 35), (820, y + 35)], head="both" if i > 1 else True, id=rid)
        s.motion.append((rid, "flow"))

    s.section(820, 122, "browser", accent=True)
    s.box(820, 134, 320, 320, "MuJoCo WASM twin", "loads the same MJCF, writes the received joint values into qpos, "
                                                  "calls mj_forward, three.js draws; no video", accent=True,
          size=14, subsize=12, id="browser")
    s.chips(834, 240, ["@mujoco/mujoco", "three.js"])
    s.chips(834, 270, ["policy picker", "chat"])
    s.para(834, 320, "a runtime add recompiles the twin without a reload; a real robot appears through its twin "
                     "with no dashboard code", 292, size=12, cls="grot muted")

    s.section(60, 486, "standalone, no server")
    s.box(60, 498, 1080, 70, "the docs hero", "the browser runs mj_step and an ONNX policy on onnxruntime-web at 50 "
                                              "Hz itself: the duck stands and walks with no Python anywhere",
          size=14, subsize=12)
    s.footnote(606, "Strands Robots v0.5.3 needed 20,470 lines and a process per card for this; the v1 dashboard is "
                    "623 lines of Python.")
    s.motion.insert(0, ("browser", "pulse"))
    return s
