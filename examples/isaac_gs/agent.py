#!/usr/bin/env python3
"""Strands agent that drives the Isaac Sim + 3DGS demo by natural language.

Unlike ``examples/mujoco_gs`` -- which registers the real ``strands_robots``
``Simulation`` AgentTool directly -- ``IsaacSimulation`` is a ``SimEngine``,
not a Strands tool, and Isaac's RTX renderer is **main-thread-affine**. So we
expose a few high-level custom ``@tool`` functions that mutate the app's shared
state and marshal any render onto the app's main-thread queue (via
``IsaacGsApp.render_once``). The agent chooses tools; the chat handler then
shows a freshly composited frame (the GS compositing stays the *display* layer,
mirroring mujoco_gs).

Optional: if ``strands-agents`` or an LLM backend (e.g. Bedrock) isn't
available, :func:`build_agent` returns ``None`` and the app runs without chat.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

logger = logging.getLogger("isaac_gs.agent")


def _scene_names() -> list:
    try:
        from examples.mujoco_gs.backgrounds import gsplat_skybox_scene_names

        return gsplat_skybox_scene_names()
    except Exception:  # noqa: BLE001
        return []


def build_agent(app, model_id: Optional[str] = None, robot_label: str = "robot arm") -> Any:
    """Build a Strands ``Agent`` bound to an :class:`IsaacGsApp`.

    ``robot_label`` names the loaded arm (e.g. "SO-101" or "Franka") so the
    agent describes the scene truthfully.

    Returns ``None`` (chat disabled) if ``strands-agents`` or the LLM backend
    isn't available, so the rest of the app still runs.
    """
    try:
        from strands import Agent, tool
    except Exception as exc:  # noqa: BLE001
        logger.warning("strands-agents unavailable (%s); agent chat disabled.", exc)
        return None

    from examples.isaac_gs.app import PANORAMA_CHOICE
    from examples.isaac_gs.scene import CAMERA_PRESETS

    cameras = list(CAMERA_PRESETS.keys())
    scenes = _scene_names()

    @tool
    def move_camera(view: str) -> str:
        """Switch the preview camera angle. ``view`` is one of the preset names."""
        v = (view or "").strip().lower()
        if v not in cameras:
            return f"Unknown camera {view!r}. Choose one of: {', '.join(cameras)}."
        app.set_camera(v)
        app.render_once(camera=v)
        return f"Camera set to {v}."

    @tool
    def wave_arm() -> str:
        """Make the robot arm wave (swing its base joint), then re-render."""
        app.render_once(camera=app.current_camera, wave=True)
        return "The arm waved."

    @tool
    def change_background(scene: str) -> str:
        """Swap the photoreal backdrop to a 3DGS scene name or 'procedural panorama'."""
        s = (scene or "").strip()
        choice = PANORAMA_CHOICE if s.lower() in ("panorama", "procedural panorama", "procedural") else s
        msg = app.set_background(choice)
        app.render_once(camera=app.current_camera)
        return msg

    @tool
    def spawn_cube(color: str = "red") -> str:
        """Add a small colored cube to the scene in front of the arm, then re-render.

        ``color`` is a common name: red, green, blue, yellow, orange, purple,
        white, or black. The cube is static (it stays put in the workspace).
        """
        msg = app.spawn_cube(color=color)
        app.render_once(camera=app.current_camera)
        return msg

    @tool
    def describe_scene() -> str:
        """Describe what's in the scene (robot, cameras, available backgrounds)."""
        return (
            f"A simulated {robot_label} on a tabletop, composited live over a photoreal "
            f"3D Gaussian Splatting room. Cameras: {', '.join(cameras)}. "
            f"3DGS scenes: {', '.join(scenes) or 'none'}. Current camera: {app.current_camera}."
        )

    system_prompt = (
        "You control an Isaac Sim + 3D Gaussian Splatting (3DGS) robotics demo. A "
        f"simulated {robot_label} sits on a tabletop, composited LIVE over a photoreal "
        "3DGS room. The scene is ALREADY built -- never rebuild, reset, or destroy it "
        "(you may add a cube with spawn_cube, but do not recreate the world or arm).\n\n"
        "Drive it with these tools ONLY:\n"
        f"  - move_camera(view): one of {cameras}.\n"
        "  - wave_arm(): the arm waves.\n"
        f"  - change_background(scene): a 3DGS scene {scenes} or 'procedural panorama'.\n"
        "  - spawn_cube(color): drop a small static colored cube into the scene.\n"
        "  - describe_scene(): a short summary.\n\n"
        "Keep replies SHORT (one or two sentences). Call at most one or two tools per "
        f"turn. Refer to the robot as the {robot_label}. If a request isn't supported by "
        "a tool, say so briefly. Be concise."
    )

    kwargs: dict = {
        "tools": [move_camera, wave_arm, change_background, spawn_cube, describe_scene],
        "system_prompt": system_prompt,
    }
    if model_id:
        kwargs["model"] = model_id
    try:
        agent = Agent(**kwargs)
        logger.info("Isaac-GS agent ready (model=%s).", model_id or "default")
        return agent
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to build agent (%s); agent chat disabled.", exc)
        return None


def extract_text(result: Any) -> str:
    """Best-effort extraction of the assistant's text from a Strands result."""
    if result is None:
        return ""
    msg = getattr(result, "message", None)
    if msg is not None:
        content = msg.get("content") if isinstance(msg, dict) else getattr(msg, "content", None)
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    if isinstance(result, str):
        return result
    return str(result)
