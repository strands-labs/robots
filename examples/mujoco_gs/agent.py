# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strands Agent driving the MuJoCo + 3DGS hybrid demo.

Wires a :class:`strands_robots.simulation.Simulation` (MuJoCo backend) to a
:class:`HybridCompositor` (panorama or gsplat background) and exposes the
Simulation through a Strands :class:`Agent`. The agent is given **only the
real** ``Simulation`` AgentTool (58 actions: world composition, physics,
``run_policy``, ``render``, recording, …) — no custom tools — so it
showcases the genuine strands-robots API (e.g. "have the arm wave" → the
agent calls ``run_policy``). The 3DGS compositing is the example's *display*
layer (the live MJPEG view + still preview render through the
``HybridCompositor``), not an agent tool.

Designed to be driven from :mod:`app` (Gradio chat) or from a plain REPL —
:meth:`MujocoGsAgent.chat` is the single user-facing entry point.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np

from .backgrounds import BackgroundRenderer, GsplatBackground, PanoramaBackground
from .compositor import HybridCompositor
from .scene import SCENE_DESCRIPTION, build_default_scene, make_scene_description

if TYPE_CHECKING:
    from strands_robots.simulation import Simulation

logger = logging.getLogger(__name__)


def make_system_prompt(scene_description: str) -> str:
    """Build the agent system prompt around a concrete scene description.

    Takes the *actual* scene description (with the robot config that really
    loaded) so the agent is never misinformed about what's in the world.
    """
    return f"""\
You are the orchestrator for a MuJoCo + 3D Gaussian Splatting (3DGS) hybrid
robotics demo. You control a small simulated arm through the real
`strands-robots` Simulation AgentTool — the same `Simulation` actions the
package ships (no custom helpers). The UI composites your scene against a
3DGS / panorama background for display.

>>> THE SCENE IS ALREADY FULLY BUILT AND READY. <<<
The world, the `arm` robot, the `cube`, and the cameras below already exist.
You do NOT need to set anything up. NEVER call these actions — they will
error or wipe the demo, and make you loop:
  - `create_world`, `load_scene`   (world already exists)
  - `add_robot`, `add_object`, `add_camera`   (everything already added)
  - `reset`, `destroy`             (these wipe the running demo)
If a `Simulation` call returns an error, do NOT try to "fix" it by rebuilding
the scene or adding robots/objects. Just report the error briefly and stop.

{scene_description}

How to drive the scene (real `Simulation` actions only):
  * Move / wave / "do a demo": call `run_policy` ONCE — the real
    strands-robots policy engine — e.g.
    `run_policy(robot_name="arm", policy_provider="mock", duration=4.0,
    control_frequency=20.0)`. It steps the arm in real time; the user watches
    it in the live view. (The `mock` policy emits exploratory actions — the
    arm moves but won't perform a trained skill.)
  * Pose the arm precisely: `set_joint_positions(robot_name="arm",
    positions={{...}})` then `step(n_steps=...)`.
  * Move the cube: `move_object(name="cube", position=[x, y, z])`.
  * Other physics: `apply_force`, `step`, `get_state`, `get_body_state`.
  * Show a still frame: `render(camera_name="front")` (or "topdown" /
    "oblique"). The UI re-composites the current scene for the panels too.

Keep every turn SHORT: usually ONE action, then a one- or two-sentence reply.
Do not explore or verify with extra calls — trust the scene description
above. Be concise.
"""


# Default prompt (used only if a caller builds an agent without a live scene).
SYSTEM_PROMPT = make_system_prompt(SCENE_DESCRIPTION)


@dataclass
class MujocoGsAgent:
    """High-level wrapper holding the sim, compositor, and Strands agent.

    Construct via :func:`build`, not directly — it owns lifecycle resources
    (a MuJoCo simulation and a thread-pool executor) that should be created
    in one place.

    Attributes:
        sim: the underlying ``Simulation`` (MuJoCo backend).
        compositor: hybrid renderer.
        agent: Strands ``Agent`` with the real ``Simulation`` AgentTool
            registered (no custom tools).
        history: list of (user_msg, assistant_msg, last_frame_or_None)
            tuples — handy to drive a Gradio chat panel.
    """

    sim: Simulation
    compositor: HybridCompositor
    agent: Any  # strands.Agent — typed Any to keep import optional at module load
    history: list[tuple[str, str, np.ndarray | None]] = field(default_factory=list)

    # ----- user-facing API ----- #

    def chat(self, message: str):
        """Run one agent turn; return ``(text, frame)``.

        ``text`` is the assistant's reply; ``frame`` is a freshly composited
        view of the current scene (so the UI can show the result of whatever
        real ``Simulation`` actions the agent just ran).
        """
        result = self.agent(message)
        text = _extract_agent_text(result)
        try:
            frame = self.compositor.render(camera_name="front").rgb
        except Exception:  # pragma: no cover
            frame = None
        return text, frame

    def render_now(
        self,
        camera_name: str = "front",
        width: int | None = None,
        height: int | None = None,
    ) -> np.ndarray:
        """Render a composite frame *without* going through the agent.

        Useful for the Gradio "live preview" panel that polls outside of
        chat turns. ``width``/``height`` let the live stream use a smaller,
        bandwidth-friendly frame over a remote share tunnel.
        """
        return self.compositor.render(camera_name=camera_name, width=width, height=height).rgb

    def set_background(self, background: BackgroundRenderer) -> None:
        """Hot-swap the background (UI dropdown, etc.)."""
        self.compositor.set_background(background)

    def reset_scene(self) -> None:
        """Tear down and rebuild the default scene."""
        try:
            self.sim.destroy()
        except Exception as e:  # pragma: no cover  — best-effort cleanup
            logger.warning("Simulation.destroy() raised: %s", e)
        build_default_scene(self.sim)
        # Cached renderers/backgrounds reference the old model — drop them.
        self.compositor.clear_caches()
        self.history.clear()

    def close(self) -> None:
        try:
            self.compositor.close()
        except Exception:  # pragma: no cover
            pass
        try:
            self.sim.destroy()
        except Exception:  # pragma: no cover
            pass


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #


def build(
    background: BackgroundRenderer | None = None,
    panorama_path: str | None = None,
    gsplat_ply: str | None = None,
    model_id: str | None = None,
) -> MujocoGsAgent:
    """Construct a fully wired :class:`MujocoGsAgent`.

    Args:
        background: pre-built renderer. If ``None``, one is chosen from the
            other args (``gsplat_ply`` > ``panorama_path`` > procedural).
        panorama_path: path to an equirectangular ``.jpg``/``.png``.
        gsplat_ply: path to a 3DGS ``.ply`` (requires ``gsplat`` extra).
        model_id: optional Strands model id (e.g. an Anthropic / Bedrock
            model). If ``None``, defaults to whatever Strands is configured
            with.

    Returns:
        A ready-to-chat :class:`MujocoGsAgent`.
    """
    # Lazy imports so the module stays importable for static analysis even
    # when strands deps aren't installed yet.
    try:
        from strands import Agent
    except ImportError as e:  # pragma: no cover
        raise ImportError("strands-agents is required. Run `pip install strands-agents`.") from e
    try:
        from strands_robots.simulation import Simulation
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "strands-robots[sim-mujoco] is required. Run `pip install 'strands-robots[sim-mujoco]'`."
        ) from e

    # 1. Pick / build the background.
    if background is None:
        if gsplat_ply:
            background = GsplatBackground(ply_path=gsplat_ply)
        elif panorama_path:
            background = PanoramaBackground(image_path=panorama_path)
        else:
            background = _default_live_background()

    # 2. Build sim + compositor + scene. Capture the robot config that
    #    actually loaded so the system prompt reflects reality.
    sim = Simulation(tool_name="sim", mesh=False)
    scene_summary = build_default_scene(sim)
    compositor = HybridCompositor(sim, background=background)
    scene_description = make_scene_description(
        robot_config=scene_summary.get("robot_config", "so101"),
        robot_name=scene_summary.get("robot_name", "arm"),
    )

    # 3. Build the Strands agent with ONLY the real `Simulation` AgentTool.
    #    Motion ("wave") goes through the genuine `run_policy` action; stills
    #    through `render`; etc. The GS compositing is the example's *display*
    #    layer (the live MJPEG view + still preview render through the
    #    HybridCompositor), not an agent tool.
    agent_kwargs: dict = {
        "tools": [sim],
        "system_prompt": make_system_prompt(scene_description),
    }
    if model_id is not None:
        agent_kwargs["model"] = model_id
    agent = Agent(**agent_kwargs)

    return MujocoGsAgent(sim=sim, compositor=compositor, agent=agent)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


_DEFAULT_LIVE_SCENE = "tabletop (indoor room)"


def _default_live_background() -> BackgroundRenderer:
    """Startup default background: the curated **live 'tabletop' 3DGS skybox**
    (MuJoCo-GS-Web's purpose-built room — clean from every angle).

    Eagerly downloads + loads the splats so any problem (missing ``gsplat``/
    ``torch``, no network, bad file) surfaces *here* and we fall back to the
    dependency-free procedural panorama — startup never hard-fails.
    """
    try:
        import torch  # noqa: F401 — ensure the gsplat runtime is present
        from gsplat import rasterization  # noqa: F401

        from .backgrounds import download_gsplat_scene, gsplat_skybox_align_for

        ply = download_gsplat_scene(_DEFAULT_LIVE_SCENE)
        bg = GsplatBackground(ply_path=str(ply), skybox=True, **gsplat_skybox_align_for(_DEFAULT_LIVE_SCENE))
        bg._load()  # warm + validate now (not mid-render)
        logger.info("Default background → live 3DGS skybox (%s).", _DEFAULT_LIVE_SCENE)
        return bg
    except Exception as e:  # noqa: BLE001 — any failure → safe fallback
        logger.warning(
            "Live 3DGS default unavailable (%s); falling back to procedural panorama. "
            "Install the gsplat extra (`pip install '.[gsplat]'`) for the photoreal default.",
            e,
        )
        return PanoramaBackground()


def _extract_agent_text(result: Any) -> str:
    """Best-effort extraction of the assistant's text reply from a Strands
    ``Agent.__call__`` result.

    Strands' agent return shape evolved across versions; this tries the most
    common shapes and falls back to ``str(result)``.
    """
    if result is None:
        return ""
    # Strands AgentResult.message.content -> list of {"text": ...} dicts
    msg = getattr(result, "message", None)
    if msg is not None:
        # NB: must branch explicitly — `getattr(...) or msg.get(...) if
        # isinstance(msg, dict) else None` binds as `(... or ...) if dict`, so
        # the attribute-style message (the common, non-dict case) would return
        # None and the chat panel would fall through to a raw `str(result)`.
        if isinstance(msg, dict):
            content = msg.get("content")
        else:
            content = getattr(msg, "content", None)
        if isinstance(content, list):
            texts = [c.get("text", "") for c in content if isinstance(c, dict) and "text" in c]
            if texts:
                return "\n".join(t for t in texts if t)
    # Plain string return
    if isinstance(result, str):
        return result
    return str(result)
