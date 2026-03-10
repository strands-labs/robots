#!/usr/bin/env python3
"""
Marble 3D World Generation Tool for Strands Agents.

This AgentTool wraps the MarblePipeline to give AI agents natural language
control over 3D environment generation for robot training.

Uses the World Labs Marble API v1 at api.worldlabs.ai.
Auth: WLT-Api-Key header (set WLT_API_KEY or MARBLE_API_KEY env var).
Models: Marble 0.1-mini, Marble 0.1-plus

Usage via Agent:
    "Generate a kitchen scene for robot training"
    "List available Marble scene presets"
    "Compose a workshop scene with an SO101 robot and task objects"
    "Generate 50 diverse training scenes from the kitchen preset"
    "Convert a PLY Gaussian splat to USDZ for Isaac Sim"
    "List my generated Marble worlds"
    "Get world details for world_id xyz"

Actions:
    generate    — Generate a 3D world from text/image/video prompt
    compose     — Compose a scene with robot + task objects
    presets     — List available scene presets
    robots      — List supported robots for composition
    batch       — Generate batch training scenes (domain randomisation)
    convert     — Convert PLY → USDZ
    info        — Get details about a preset or generated scene
    list_worlds — List worlds from your Marble account
    get_world   — Get a specific world by ID
"""

import json
import logging
from typing import Any, Dict

from strands import tool

logger = logging.getLogger(__name__)

# Global pipeline instance (reused across calls)
_pipeline = None


def _get_pipeline(**kwargs):
    """Get or create the global MarblePipeline instance."""
    global _pipeline

    if _pipeline is not None:
        return _pipeline

    from strands_robots.marble import MarbleConfig, MarblePipeline

    config = MarbleConfig(**{k: v for k, v in kwargs.items() if hasattr(MarbleConfig, k)})
    _pipeline = MarblePipeline(config)
    return _pipeline


@tool
def marble_tool(
    action: str = "presets",
    # Generation
    prompt: str = "",
    preset: str = "",
    input_image: str = "",
    input_video: str = "",
    num_variations: int = 1,
    output_dir: str = "./marble_scenes",
    model: str = "Marble 0.1-plus",
    # Composition
    scene_path: str = "",
    robot: str = "",
    task_objects: str = "",
    table_replacement: bool = True,
    robot_position: str = "",
    # Batch
    num_per_prompt: int = 10,
    # Conversion
    ply_path: str = "",
    output_path: str = "",
    # World management
    world_id: str = "",
    page_size: int = 20,
    status: str = "",
    tags: str = "",
    is_public: str = "",
    # Config
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Marble 3D World Generation — create diverse training environments for robots.

    Generate photorealistic 3D worlds from text, images, or video using World Labs'
    Marble API (api.worldlabs.ai), convert to Isaac Sim-compatible USD, and compose
    with robots for training in LeIsaac environments.

    Auth: Set WLT_API_KEY or MARBLE_API_KEY env var for real generation.
    Models: "Marble 0.1-mini" (faster) or "Marble 0.1-plus" (higher quality).

    Args:
        action: Action to perform:
            - "presets": List available scene presets (kitchen, office, etc.)
            - "robots": List supported robots for scene composition
            - "generate": Generate a 3D world from text/image/video prompt
            - "compose": Compose a generated scene with robot + objects
            - "batch": Generate batch training scenes for domain randomisation
            - "convert": Convert PLY Gaussian splat to USDZ
            - "info": Get details about a preset
            - "list_worlds": List worlds from your Marble account
            - "get_world": Get a specific world by ID
        prompt: Text prompt for world generation (for generate/batch actions)
        preset: Scene preset name (kitchen, office_desk, workshop, etc.)
        input_image: Path or URL to input image for image-to-3D generation
        input_video: Path or URL to input video for video-to-3D generation
        num_variations: Number of scene variations to generate
        output_dir: Output directory for generated scenes
        model: Marble model ("Marble 0.1-mini" or "Marble 0.1-plus")
        scene_path: Path to scene file (for compose/convert actions)
        robot: Robot key (so101, bi_so101, panda, ur5e, xarm7, lekiwi)
        task_objects: Comma-separated task object names (orange, mug, plate, etc.)
        table_replacement: Replace detected table with physics-enabled mesh
        robot_position: Robot position as JSON "[x, y, z]"
        num_per_prompt: Number of variations per prompt (for batch action)
        ply_path: PLY file path (for convert action)
        output_path: Output file path (for convert action)
        world_id: World ID (for get_world action)
        page_size: Results per page (for list_worlds, 1-100)
        status: Filter by status (SUCCEEDED, PENDING, FAILED, RUNNING)
        tags: Comma-separated tags filter (for list_worlds)
        is_public: Filter by visibility ("true"/"false", for list_worlds)
        seed: Random seed for reproducibility

    Returns:
        Dict with status and content

    Examples:
        marble_tool(action="presets")
        marble_tool(action="generate", prompt="A modern kitchen with fruits",
                   robot="so101", num_variations=5)
        marble_tool(action="generate", prompt="Recreate this scene",
                   input_image="photo.jpg", robot="so101")
        marble_tool(action="batch", preset="kitchen", num_per_prompt=50,
                   robot="so101")
        marble_tool(action="convert", ply_path="scene.ply")
        marble_tool(action="list_worlds", status="SUCCEEDED", page_size=10)
        marble_tool(action="get_world", world_id="abc123")
    """
    try:
        from strands_robots.marble import (
            MARBLE_PRESETS,
            SUPPORTED_ROBOTS,
            MarbleConfig,
            MarblePipeline,
            list_presets,
        )

        # ── presets ───────────────────────────────────────────────
        if action == "presets":
            presets = list_presets()
            lines = ["🌍 Marble Scene Presets:\n"]
            lines.append(f"{'Preset':<18s} {'Category':<12s} {'Description'}")
            lines.append("─" * 75)
            for p in presets:
                lines.append(f"{p['name']:<18s} {p['category']:<12s} {p['description']}")
                lines.append(f"  Objects: {p['objects']}")
            lines.append(f"\nTotal: {len(presets)} presets")
            lines.append("\n💡 Use: marble_tool(action='generate', preset='kitchen', robot='so101')")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── robots ────────────────────────────────────────────────
        elif action == "robots":
            lines = ["🤖 Supported Robots for Marble Scene Composition:\n"]
            lines.append(f"{'Robot':<15s} {'Type':<22s} {'Description'}")
            lines.append("─" * 65)
            for key, info in sorted(SUPPORTED_ROBOTS.items()):
                lines.append(f"{key:<15s} {info['type']:<22s} {info['description']}")
            lines.append(f"\nTotal: {len(SUPPORTED_ROBOTS)} robots")
            lines.append("\n💡 Use: marble_tool(action='generate', robot='so101')")

            return {"status": "success", "content": [{"text": "\n".join(lines)}]}

        # ── info ──────────────────────────────────────────────────
        elif action == "info":
            if preset and preset in MARBLE_PRESETS:
                info = MARBLE_PRESETS[preset]
                text = (
                    f"🌍 Marble Preset: {preset}\n"
                    f"  Description: {info['description']}\n"
                    f"  Category: {info['category']}\n"
                    f"  Prompt: {info['prompt']}\n"
                    f"  Task Objects: {', '.join(info.get('task_objects', []))}\n"
                    f"\n  Usage: marble_tool(action='generate', preset='{preset}', robot='so101')"
                )
            elif preset:
                text = f"❌ Unknown preset: '{preset}'.\n" f"Available: {', '.join(sorted(MARBLE_PRESETS.keys()))}"
            else:
                text = (
                    "🌍 Marble 3D World Generation\n\n"
                    "Marble by World Labs generates full 3D worlds from text, images, "
                    "video, or 3D layouts. These worlds are converted to Isaac Sim-compatible "
                    "USD scenes and composed with robots for training.\n\n"
                    "Pipeline: Prompt → Marble API → PLY/GLB → 3DGrut → USDZ → "
                    "Compose(Robot + Objects) → LeIsaac/Isaac Sim\n\n"
                    f"Presets: {len(MARBLE_PRESETS)} | "
                    f"Robots: {len(SUPPORTED_ROBOTS)} | "
                    f"Outputs: PLY, GLB, USDZ, Video\n\n"
                    "💡 Actions: presets, robots, generate, compose, batch, convert, info"
                )
            return {"status": "success", "content": [{"text": text}]}

        # ── generate ──────────────────────────────────────────────
        elif action == "generate":
            # Resolve prompt from preset if not provided directly
            effective_prompt = prompt
            obj_list = [o.strip() for o in task_objects.split(",") if o.strip()] if task_objects else []

            if not effective_prompt and preset and preset in MARBLE_PRESETS:
                effective_prompt = MARBLE_PRESETS[preset]["prompt"]
                if not obj_list:
                    obj_list = MARBLE_PRESETS[preset].get("task_objects", [])

            if not effective_prompt:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "❌ Either 'prompt' or a valid 'preset' is required.\n"
                                f"Available presets: {', '.join(sorted(MARBLE_PRESETS.keys()))}"
                            )
                        }
                    ],
                }

            # Use the global singleton pipeline
            pipeline = _get_pipeline(
                robot=robot or None,
                auto_compose=bool(robot),
                num_variations=num_variations,
                seed=seed,
                convert_to_usdz=True,
                model=model,
            )

            scenes = pipeline.generate_world(
                prompt=effective_prompt,
                output_dir=output_dir,
                input_image=input_image or None,
                input_video=input_video or None,
                num_variations=num_variations,
            )

            # Compose scenes if robot specified and not auto-composed
            if robot and not pipeline.config.auto_compose:
                for scene in scenes:
                    bg = scene.best_background
                    if bg:
                        try:
                            composed = pipeline.compose_scene(
                                scene_path=bg,
                                robot=robot,
                                task_objects=obj_list,
                                table_replacement=table_replacement,
                                robot_position=json.loads(robot_position) if robot_position else None,
                            )
                            scene.scene_usd = composed.get("scene_usd")
                            scene.robot = robot
                            scene.composed = True
                        except Exception as e:
                            logger.warning("Composition failed: %s", e)

            # Format response
            lines = [
                "🌍 Marble World Generation Complete\n",
                f"  Prompt: {effective_prompt[:80]}{'...' if len(effective_prompt) > 80 else ''}",
                f"  Variations: {len(scenes)}",
                f"  Model: {model}",
                f"  Robot: {robot or 'none'}",
                f"  Output: {output_dir}",
            ]

            for i, scene in enumerate(scenes):
                lines.append(f"\n  Scene {i+1}: {scene.scene_id}")
                if scene.world_id:
                    lines.append(f"    World ID: {scene.world_id}")
                if scene.world_marble_url:
                    lines.append(f"    Marble URL: {scene.world_marble_url}")
                if scene.caption:
                    lines.append(f"    Caption: {scene.caption}")
                if scene.spz_path:
                    lines.append(f"    SPZ: {scene.spz_path}")
                if scene.ply_path and scene.ply_path != scene.spz_path:
                    lines.append(f"    PLY: {scene.ply_path}")
                if scene.glb_path:
                    lines.append(f"    GLB: {scene.glb_path}")
                if scene.pano_path:
                    lines.append(f"    Panorama: {scene.pano_path}")
                if scene.usdz_path:
                    lines.append(f"    USDZ: {scene.usdz_path}")
                if scene.scene_usd:
                    lines.append(f"    Composed USD: {scene.scene_usd}")
                if scene.pano_url:
                    lines.append(f"    Pano URL: {scene.pano_url}")
                if scene.spz_urls:
                    lines.append(
                        f"    SPZ splats: {len(scene.spz_urls)} variant(s) ({', '.join(scene.spz_urls.keys())})"
                    )
                if scene.metadata.get("placeholder"):
                    lines.append("    ⚠️ Placeholder (set WLT_API_KEY for real generation)")

            return {
                "status": "success",
                "content": [{"text": "\n".join(lines)}],
                "scenes": [s.to_dict() for s in scenes],
            }

        # ── compose ───────────────────────────────────────────────
        elif action == "compose":
            if not scene_path:
                return {"status": "error", "content": [{"text": "❌ scene_path required for compose action"}]}
            if not robot:
                return {"status": "error", "content": [{"text": "❌ robot required for compose action"}]}

            obj_list = [o.strip() for o in task_objects.split(",") if o.strip()] if task_objects else []

            pipeline = _get_pipeline(robot=robot)
            result = pipeline.compose_scene(
                scene_path=scene_path,
                robot=robot,
                task_objects=obj_list,
                table_replacement=table_replacement,
                robot_position=json.loads(robot_position) if robot_position else None,
            )

            text = (
                f"🔧 Scene Composed\n"
                f"  Background: {scene_path}\n"
                f"  Robot: {robot}\n"
                f"  Objects: {', '.join(obj_list) if obj_list else 'none'}\n"
                f"  Output: {result.get('scene_usd', 'N/A')}"
            )
            return {"status": "success", "content": [{"text": text}], "result": result}

        # ── batch ─────────────────────────────────────────────────
        elif action == "batch":
            config = MarbleConfig(
                robot=robot or None,
                seed=seed,
                convert_to_usdz=True,
            )
            pipeline = MarblePipeline(config)

            prompts_list = None
            if prompt:
                prompts_list = [p.strip() for p in prompt.split("|") if p.strip()]

            result = pipeline.generate_training_scenes(
                prompts=prompts_list,
                preset=preset or None,
                num_per_prompt=num_per_prompt,
                robot=robot or None,
                output_dir=output_dir,
            )

            text = (
                f"🏭 Batch Training Scene Generation Complete\n"
                f"  Total scenes: {result['total_scenes']}\n"
                f"  Composed: {result['composed_scenes']}\n"
                f"  Preset: {result.get('preset', 'custom')}\n"
                f"  Robot: {result.get('robot', 'none')}\n"
                f"  Objects: {', '.join(result.get('task_objects', []))}\n"
                f"  Output: {result['output_dir']}\n"
                f"\n  💡 Use these scenes with LeIsaac for diverse training"
            )

            pipeline.cleanup()
            return {"status": "success", "content": [{"text": text}], "result": result}

        # ── convert ───────────────────────────────────────────────
        elif action == "convert":
            if not ply_path:
                return {"status": "error", "content": [{"text": "❌ ply_path required for convert action"}]}

            pipeline = _get_pipeline()
            usdz = pipeline.convert_ply_to_usdz(
                ply_path=ply_path,
                output_path=output_path or None,
            )

            text = (
                f"🔄 PLY → USDZ Conversion Complete\n"
                f"  Input: {ply_path}\n"
                f"  Output: {usdz}\n"
                f"  💡 Load in Isaac Sim or compose with a robot"
            )
            return {"status": "success", "content": [{"text": text}]}

        # ── list_worlds ───────────────────────────────────────────
        elif action == "list_worlds":
            pipeline = _get_pipeline(model=model)
            if not pipeline.config.api_key:
                return {
                    "status": "error",
                    "content": [{"text": "❌ WLT_API_KEY or MARBLE_API_KEY required for list_worlds"}],
                }

            tags_list = [t.strip() for t in tags.split(",") if t.strip()] if tags else None
            pub_filter = None
            if is_public.lower() == "true":
                pub_filter = True
            elif is_public.lower() == "false":
                pub_filter = False

            result = pipeline.list_worlds(
                page_size=page_size,
                status=status or None,
                model=model if model != "Marble 0.1-plus" else None,
                tags=tags_list,
                is_public=pub_filter,
            )

            worlds = result.get("worlds", [])
            lines = [f"🌍 Marble Worlds ({len(worlds)} results)\n"]
            for w in worlds:
                w_id = w.get("world_id", "?")
                w_name = w.get("display_name", "Untitled")
                w_model = w.get("model", "?")
                w_url = w.get("world_marble_url", "")
                lines.append(f"  • {w_name} (id: {w_id})")
                lines.append(f"    Model: {w_model} | URL: {w_url}")

            next_token = result.get("next_page_token")
            if next_token:
                lines.append("\n  📄 More results available (next_page_token exists)")

            return {"status": "success", "content": [{"text": "\n".join(lines)}], "result": result}

        # ── get_world ─────────────────────────────────────────────
        elif action == "get_world":
            if not world_id:
                return {"status": "error", "content": [{"text": "❌ world_id required for get_world action"}]}

            pipeline = _get_pipeline(model=model)
            if not pipeline.config.api_key:
                return {
                    "status": "error",
                    "content": [{"text": "❌ WLT_API_KEY or MARBLE_API_KEY required for get_world"}],
                }

            world = pipeline.get_world(world_id)
            assets = world.get("assets") or {}

            lines = [
                f"🌍 World: {world.get('display_name', 'Untitled')}",
                f"  ID: {world.get('world_id')}",
                f"  Model: {world.get('model', '?')}",
                f"  Marble URL: {world.get('world_marble_url', 'N/A')}",
                f"  Created: {world.get('created_at', '?')}",
            ]

            if assets.get("caption"):
                lines.append(f"  Caption: {assets['caption']}")
            if assets.get("thumbnail_url"):
                lines.append(f"  Thumbnail: {assets['thumbnail_url']}")
            imagery = assets.get("imagery") or {}
            if imagery.get("pano_url"):
                lines.append(f"  Pano: {imagery['pano_url']}")
            mesh = assets.get("mesh") or {}
            if mesh.get("collider_mesh_url"):
                lines.append(f"  Collider Mesh: {mesh['collider_mesh_url']}")
            splats = assets.get("splats") or {}
            spz = splats.get("spz_urls") or {}
            if spz:
                lines.append(f"  SPZ Splats: {len(spz)} file(s)")
                for k, v in spz.items():
                    lines.append(f"    {k}: {v}")

            tags_list = world.get("tags") or []
            if tags_list:
                lines.append(f"  Tags: {', '.join(tags_list)}")

            return {"status": "success", "content": [{"text": "\n".join(lines)}], "result": world}

        # ── unknown ───────────────────────────────────────────────
        else:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown action: '{action}'\n"
                            "Valid: presets, robots, info, generate, compose, batch, convert, "
                            "list_worlds, get_world"
                        )
                    }
                ],
            }

    except Exception as e:
        logger.error("marble_tool error: %s", e, exc_info=True)
        return {"status": "error", "content": [{"text": f"❌ Error: {str(e)}"}]}
