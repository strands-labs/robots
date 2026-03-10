#!/usr/bin/env python3
"""Sample 03: Marble World — Text-to-3D World Generation.

Demonstrates the Marble (World Labs) pipeline for generating full 3D
training environments from natural-language descriptions.

Pipeline stages:

    Text Prompt ──→ Marble API ──→ .ply + .glb
                                        │
                       Convert PLY → USDZ
                                        │
                       Compose Robot + Table + Objects → scene.usd

This script gracefully handles three scenarios:

1. **Full demo** (WLT_API_KEY set): generates a scene, composes with robot.
2. **Offline exploration** (strands-robots[marble] installed, no key):
   lists presets, configures the pipeline, shows what's available.
3. **Minimal fallback** (marble not installed): explains how to install.

Hardware : CPU (API call to World Labs cloud)
Time     : ~30 seconds with API key, ~2 seconds without
Outputs  : marble_output/ directory with generated assets (or metadata JSON)
"""
from __future__ import annotations

import json
import os

# ── Output directory ─────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(SCRIPT_DIR, "marble_output")
os.makedirs(OUT_DIR, exist_ok=True)

# ── Default prompt for generation ────────────────────────────────────
DEFAULT_PROMPT = (
    "A modern kitchen with wooden countertops, a stainless steel sink, "
    "and a cutting board with fruit on the counter"
)


def _print_header() -> None:
    print("=" * 60)
    print("  Sample 03 — Marble World (Text → 3D Environment)")
    print("=" * 60)
    print()


def list_presets() -> dict | None:
    """Import and display MARBLE_PRESETS.  Returns the dict or None."""
    try:
        from strands_robots.marble import MARBLE_PRESETS

        print("🏠 Step 1 — Available Marble presets\n")
        for name, preset in MARBLE_PRESETS.items():
            prompt = preset.get("prompt", "")
            short = prompt[:65] + " …" if len(prompt) > 65 else prompt
            category = preset.get("category", "—")
            print(f"   • {name:20s}  [{category:10s}]  {short}")
        print(f"\n   ✅ {len(MARBLE_PRESETS)} presets available\n")
        return MARBLE_PRESETS
    except ImportError:
        print("⚠️  strands-robots[marble] not installed.")
        print("   Install with:  pip install strands-robots[marble]")
        print("   Then re-run this script.\n")
        return None


def configure_pipeline():
    """Create and return a (MarblePipeline, MarbleConfig) tuple."""
    from strands_robots.marble import MarbleConfig, MarblePipeline

    config = MarbleConfig(
        model="Marble 0.1-mini",   # Fast model for demos
        output_format="ply",       # Gaussian splat (convert to USDZ later)
        robot="so101",             # SO-101 for composition
    )
    pipeline = MarblePipeline(config)

    print("⚙️  Step 2 — Pipeline configured\n")
    print(f"   Model  : {config.model}")
    print(f"   Format : {config.output_format}")
    print(f"   Robot  : {config.robot}")
    print(f"   Seed   : {config.seed}")
    print()
    return pipeline, config


def generate_scene(pipeline, prompt: str = DEFAULT_PROMPT):
    """Call the Marble API and return the first MarbleScene, or None."""
    print("🌍 Step 3 — Generating 3D world from text …\n")
    print(f'   Prompt: "{prompt}"')
    print("   ⏳ Calling Marble API (this may take 20–30 seconds) …\n")

    try:
        result = pipeline.generate_world(prompt=prompt, output_dir=OUT_DIR)

        # generate_world returns a list of MarbleScene objects
        scenes = result if isinstance(result, list) else [result]
        if not scenes:
            print("   ⚠️  No scenes returned\n")
            return None

        scene = scenes[0]
        is_placeholder = scene.metadata.get("placeholder", False)

        if is_placeholder:
            print("   ⚠️  API returned a placeholder (check billing / quota)")
        else:
            print(f"   ✅ World ID : {scene.world_id}")
            print(f"   ✅ Caption  : {scene.caption}")
            if scene.glb_path:
                print(f"   ✅ GLB      : {scene.glb_path}")
            if scene.spz_path:
                print(f"   ✅ SPZ      : {scene.spz_path}")
            if scene.ply_path:
                print(f"   ✅ PLY      : {scene.ply_path}")

        # Persist metadata
        scene_info = {
            "scene_id": scene.scene_id,
            "world_id": scene.world_id,
            "caption": scene.caption,
            "marble_url": scene.world_marble_url,
            "glb_path": str(scene.glb_path) if scene.glb_path else None,
            "spz_path": str(scene.spz_path) if scene.spz_path else None,
            "ply_path": str(scene.ply_path) if scene.ply_path else None,
            "placeholder": is_placeholder,
        }
        info_path = os.path.join(OUT_DIR, "scene_info.json")
        with open(info_path, "w") as fh:
            json.dump(scene_info, fh, indent=2)
        print(f"   💾 Metadata : {info_path}\n")

        return scene

    except Exception as exc:
        print(f"   ❌ Generation failed: {exc}\n")
        return None


def compose_with_robot(pipeline, scene):
    """Compose a generated scene with a robot.  Returns result dict or None."""
    asset_path = scene.glb_path or scene.spz_path or scene.ply_path
    is_placeholder = scene.metadata.get("placeholder", False)

    if not asset_path or is_placeholder:
        print("🤖 Step 4 — Skipped (no real scene to compose)\n")
        return None

    print("🤖 Step 4 — Composing scene with SO-101 robot …\n")
    try:
        composed = pipeline.compose_scene(
            scene_path=str(asset_path),
            robot="so101",
            task_objects=["orange", "mug"],
        )
        print(f"   ✅ Composed scene: {composed}\n")
        return composed
    except Exception as exc:
        print(f"   ⚠️  Composition failed: {exc}\n")
        return None


def main() -> None:
    _print_header()

    # ── Step 1: List presets ─────────────────────────────────────────
    presets = list_presets()
    if presets is None:
        return  # marble not installed

    # ── Step 2: Configure pipeline ───────────────────────────────────
    pipeline, config = configure_pipeline()

    # ── Step 3: Generate (only if API key is available) ──────────────
    api_key = os.getenv("WLT_API_KEY") or os.getenv("MARBLE_API_KEY")

    if not api_key:
        print("🔑 Step 3 — Skipped (no API key)\n")
        print("   Set WLT_API_KEY or MARBLE_API_KEY to generate scenes.")
        print("   Get a key at: https://marble.worldlabs.ai")
        print()
        print("   📸 Pre-generated outputs available at:")
        print("      presentation/output/10_marble_kitchen/")
        print()
        print(f"{'─' * 60}")
        print("✅ Done (exploration mode — no API call made).")
        return

    scene = generate_scene(pipeline)
    if scene is None:
        return

    # ── Step 4: Compose ──────────────────────────────────────────────
    compose_with_robot(pipeline, scene)

    print(f"{'─' * 60}")
    print(f"✅ Done! Assets saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
