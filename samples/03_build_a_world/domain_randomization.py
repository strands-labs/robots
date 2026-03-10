#!/usr/bin/env python3
"""Sample 03: Domain Randomization — Scene Variation for Robust Training.

Creates a base scene then applies ``sim.randomize()`` with different seeds
to produce visually diverse variants of the same environment.

Why domain randomization?
    A policy trained on a *single* scene memorises pixel-level features
    (exact colour of the table, shadow direction, object placement).
    By training on hundreds of randomised variants the policy instead
    learns *invariant* features — which are exactly the features that
    transfer to the real world.

This script demonstrates the ``Simulation.randomize()`` API:

    sim.randomize(
        randomize_colors=True,       # random RGBA per geom
        randomize_lighting=True,     # light position ± 0.5 m, intensity 0.3–1.0
        randomize_physics=True,      # friction × [0.5, 1.5], mass × [0.8, 1.2]
        randomize_positions=True,    # position ± noise_m
        position_noise=0.02,         # metres
        seed=42,                     # reproducible
    )

Hardware : CPU only
Time     : ~15 seconds
Outputs  : output/randomized_000_baseline.png … randomized_010.png
"""
from __future__ import annotations

import os
import time

# ── Configuration ────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "output")
os.makedirs(OUT_DIR, exist_ok=True)

NUM_VARIANTS = 10
RENDER_WIDTH = 480
RENDER_HEIGHT = 360
CAMERA_NAME = "overhead"

# Objects placed in the base scene (randomization perturbs them)
BASE_OBJECTS = [
    {"name": "cube_a", "shape": "box", "size": [0.04, 0.04, 0.04],
     "position": [0.25, 0.00, 0.55], "color": [0.8, 0.2, 0.2, 1.0]},
    {"name": "sphere_b", "shape": "sphere", "size": [0.03],
     "position": [0.20, 0.10, 0.55], "color": [0.2, 0.8, 0.3, 1.0]},
    {"name": "cylinder_c", "shape": "cylinder", "size": [0.02, 0.02, 0.05],
     "position": [0.30, -0.05, 0.55], "color": [0.3, 0.3, 0.9, 1.0]},
]


def _save_render(result: dict, path: str) -> bool:
    """Extract PNG bytes from a render result and save to *path*."""
    if result.get("status") != "success":
        return False
    for item in result.get("content", []):
        if "image" in item:
            with open(path, "wb") as fh:
                fh.write(item["image"]["source"]["bytes"])
            return True
    return False


def create_base_scene(sim=None):
    """Build the base tabletop scene.

    Accepts an optional *sim* instance for test injection.
    """
    if sim is None:
        from strands_robots.simulation import Simulation

        sim = Simulation(tool_name="domain_rand")

    sim.create_world(timestep=0.002, gravity=[0, 0, -9.81])
    sim.add_robot(name="arm", data_config="so100", position=[0, 0, 0])

    for obj in BASE_OBJECTS:
        sim.add_object(**obj)

    sim.add_camera(
        name=CAMERA_NAME,
        position=[0.2, 0.0, 1.0],
        target=[0.2, 0.0, 0.4],
    )

    # Let physics settle
    sim.step(n_steps=100)

    return sim


def randomize_and_render(sim, variant_index: int, seed: int) -> dict:
    """Apply one randomization pass and render.

    Returns a dict with keys ``variant``, ``seed``, ``saved``,
    ``changes``, ``filename``.
    """
    # Decide which randomization axes to enable for this variant
    do_physics = (variant_index % 3 == 0)
    do_positions = (variant_index % 2 == 0)

    result = sim.randomize(
        randomize_colors=True,
        randomize_lighting=True,
        randomize_physics=do_physics,
        randomize_positions=do_positions,
        position_noise=0.02,
        color_range=(0.1, 1.0),
        friction_range=(0.5, 1.5),
        mass_range=(0.8, 1.2),
        seed=seed,
    )

    # Step physics so position changes take effect
    sim.step(n_steps=50)

    # Summarise what was randomised
    changes_text = result.get("content", [{}])[0].get("text", "")
    short_changes = " | ".join(
        line.strip()
        for line in changes_text.splitlines()
        if line.strip() and not line.startswith("🎲")
    )

    filename = f"randomized_{variant_index:03d}.png"
    render_result = sim.render(
        camera_name=CAMERA_NAME,
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
    )
    saved = _save_render(render_result, os.path.join(OUT_DIR, filename))

    return {
        "variant": variant_index,
        "seed": seed,
        "saved": saved,
        "changes": short_changes,
        "filename": filename,
    }


def main() -> None:
    """Entry-point: build scene, render baseline, generate variants."""
    print("=" * 60)
    print("  Sample 03 — Domain Randomization")
    print("=" * 60)
    print()

    # ── Build baseline ───────────────────────────────────────────────
    print("🎯 Creating base scene (SO-100 + 3 objects) …")
    sim = create_base_scene()

    baseline_file = "randomized_000_baseline.png"
    baseline_result = sim.render(
        camera_name=CAMERA_NAME,
        width=RENDER_WIDTH,
        height=RENDER_HEIGHT,
    )
    if _save_render(baseline_result, os.path.join(OUT_DIR, baseline_file)):
        print(f"   ✅ Baseline saved: {baseline_file}")
    else:
        print("   ⚠️  Baseline render failed (headless without EGL?)")

    # ── Generate randomised variants ─────────────────────────────────
    print(f"\n🎲 Generating {NUM_VARIANTS} randomised variants …\n")

    t0 = time.time()
    results = []

    for i in range(1, NUM_VARIANTS + 1):
        info = randomize_and_render(sim, i, seed=42 + i)
        results.append(info)

        icon = "✅" if info["saved"] else "⚠️"
        print(
            f"   {icon} Variant {i:2d}/{NUM_VARIANTS}: "
            f"{info['filename']}  [seed={info['seed']}  {info['changes']}]"
        )

    elapsed = time.time() - t0
    n_saved = sum(1 for r in results if r["saved"])

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'─' * 60}")
    print(f"📊 Generated {n_saved + 1} images in {elapsed:.1f}s")
    print(f"   📁 {OUT_DIR}/")
    print()
    print("💡 In real training you'd generate 100–1 000 variants per epoch.")
    print("   The policy learns features that are *invariant* to these changes,")
    print("   which is exactly what transfers to the real world.")
    print()
    print("✅ Done!")

    # Clean up
    sim.destroy()


if __name__ == "__main__":
    main()
