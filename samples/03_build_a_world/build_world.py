#!/usr/bin/env python3
"""Sample 03: Build a World — Programmatic 3D Environment Generation.

Creates a complete tabletop manipulation scene from scratch using the
Simulation API.  Every step mirrors what an AI agent does when told:
"Set up a tabletop with a robot and some objects."

Pipeline
--------
1. Initialise physics world (gravity, timestep)
2. Add an SO-100 robot arm from the asset registry
3. Spawn 5 objects of varying shapes and colours
4. Mount 3 cameras (front, side, top-down)
5. Step physics to let objects settle
6. Render the scene from each camera and save PNGs

Hardware : CPU only (MuJoCo runs on CPU)
Time     : ~10 seconds
Outputs  : output/world_front.png, output/world_side.png, output/world_top.png
"""
from __future__ import annotations

import os

# ── Output directory ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Object definitions ───────────────────────────────────────────────
#  Each dict maps directly to Simulation.add_object() keyword arguments.
#  Positions are in metres; colours are RGBA floats.
SCENE_OBJECTS = [
    {
        "name": "red_cube",
        "shape": "box",
        "size": [0.04, 0.04, 0.04],
        "position": [0.25, 0.00, 0.55],
        "color": [1.0, 0.2, 0.2, 1.0],
        "mass": 0.05,
    },
    {
        "name": "green_sphere",
        "shape": "sphere",
        "size": [0.03],
        "position": [0.20, 0.10, 0.55],
        "color": [0.2, 0.9, 0.3, 1.0],
        "mass": 0.03,
    },
    {
        "name": "blue_cylinder",
        "shape": "cylinder",
        "size": [0.02, 0.02, 0.06],
        "position": [0.30, -0.05, 0.55],
        "color": [0.2, 0.4, 1.0, 1.0],
        "mass": 0.04,
    },
    {
        "name": "yellow_cube",
        "shape": "box",
        "size": [0.03, 0.03, 0.03],
        "position": [0.15, -0.08, 0.55],
        "color": [1.0, 0.9, 0.1, 1.0],
        "mass": 0.03,
    },
    {
        "name": "orange_sphere",
        "shape": "sphere",
        "size": [0.025],
        "position": [0.28, 0.08, 0.55],
        "color": [1.0, 0.6, 0.1, 1.0],
        "mass": 0.02,
    },
]

# ── Camera definitions ───────────────────────────────────────────────
CAMERAS = [
    {"name": "front", "position": [0.8, 0.0, 0.6], "target": [0.2, 0.0, 0.4]},
    {"name": "side", "position": [0.2, 0.8, 0.6], "target": [0.2, 0.0, 0.4]},
    {"name": "top", "position": [0.2, 0.0, 1.2], "target": [0.2, 0.0, 0.4]},
]


def _save_render(result: dict, path: str) -> bool:
    """Extract PNG bytes from a render result and save to *path*.

    Returns ``True`` on success, ``False`` otherwise.
    """
    if result.get("status") != "success":
        return False
    for item in result.get("content", []):
        if "image" in item:
            with open(path, "wb") as fh:
                fh.write(item["image"]["source"]["bytes"])
            return True
    return False


def build_world(sim=None):
    """Build a tabletop scene and return the Simulation instance.

    If *sim* is ``None`` a fresh ``Simulation`` is created.  Accepts an
    existing instance so tests can inject a mock.
    """
    if sim is None:
        from strands_robots.simulation import Simulation

        sim = Simulation(tool_name="build_world")

    # ── Step 1: Create physics world ─────────────────────────────────
    print("🌍 Step 1 — Creating physics world …")
    result = sim.create_world(timestep=0.002, gravity=[0, 0, -9.81])
    print(f"   {result['content'][0]['text'].splitlines()[0]}")

    # ── Step 2: Add a robot arm ──────────────────────────────────────
    print("\n🤖 Step 2 — Adding SO-100 robot arm …")
    result = sim.add_robot(name="arm", data_config="so100", position=[0, 0, 0])
    print(f"   {result['content'][0]['text'].splitlines()[0]}")

    # ── Step 3: Spawn objects ────────────────────────────────────────
    print("\n📦 Step 3 — Spawning 5 objects …")
    for obj in SCENE_OBJECTS:
        result = sim.add_object(**obj)
        icon = "✅" if result["status"] == "success" else "⚠️"
        print(f"   {icon} {obj['name']}: {obj['shape']} at {obj['position']}")

    # ── Step 4: Mount cameras ────────────────────────────────────────
    print("\n📷 Step 4 — Mounting cameras …")
    for cam in CAMERAS:
        result = sim.add_camera(**cam)
        icon = "✅" if result["status"] == "success" else "⚠️"
        print(f"   {icon} {cam['name']} at {cam['position']}")

    # ── Step 5: Let physics settle ───────────────────────────────────
    print("\n⏩ Step 5 — Stepping physics (200 steps) …")
    sim.step(n_steps=200)
    print("   ✅ Physics settled")

    return sim


def render_cameras(sim, cameras=None, out_dir=None):
    """Render each camera and save PNGs.  Returns list of saved paths."""
    cameras = cameras or CAMERAS
    out_dir = out_dir or OUT_DIR
    saved = []

    print("\n🖼️  Step 6 — Rendering scene from each camera …")
    for cam in cameras:
        result = sim.render(camera_name=cam["name"], width=640, height=480)
        path = os.path.join(out_dir, f"world_{cam['name']}.png")
        if _save_render(result, path):
            size = os.path.getsize(path)
            print(f"   ✅ {path} ({size:,} bytes)")
            saved.append(path)
        else:
            print(f"   ⚠️  Camera '{cam['name']}': render failed or no image")

    return saved


def main() -> None:
    """Entry-point: build the world, render, print summary."""
    print("=" * 60)
    print("  Sample 03 — Build a World (Programmatic Scene Generation)")
    print("=" * 60)
    print()

    sim = build_world()
    saved = render_cameras(sim)

    # ── Summary ──────────────────────────────────────────────────────
    state = sim.get_state()
    print("\n📊 World summary:")
    for line in state["content"][0]["text"].splitlines():
        print(f"   {line}")

    print(f"\n{'─' * 60}")
    print(f"✅ Done! {len(saved)} images saved to {OUT_DIR}/")
    if not saved:
        print("   (No images rendered — this is normal in headless CI without EGL)")

    # Clean up
    sim.destroy()


if __name__ == "__main__":
    main()
