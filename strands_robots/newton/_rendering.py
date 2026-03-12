"""Rendering and observation retrieval."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def render(
    backend: NewtonBackend,
    camera_name: Optional[str] = None,
    width: int = 1024,
    height: int = 768,
) -> Dict[str, Any]:
    """Render current scene state."""
    if backend._model is None:
        return {"success": False, "message": "No model to render."}

    newton = backend._newton
    render_backend = backend._config.render_backend

    if backend._renderer is None:
        try:
            viewer_mod = getattr(newton, "viewer", None)
            if viewer_mod is None:
                viewer_mod = newton

            if render_backend == "opengl":
                backend._renderer = viewer_mod.ViewerGL()
            elif render_backend == "rerun":
                backend._renderer = viewer_mod.ViewerRerun()
            elif render_backend == "viser":
                backend._renderer = viewer_mod.ViewerViser()
            else:
                backend._renderer = viewer_mod.ViewerNull()

            if hasattr(backend._renderer, "set_model"):
                backend._renderer.set_model(backend._model)
            logger.info("Renderer initialised: %s", render_backend)
        except Exception as exc:
            logger.warning("Renderer init failed: %s", exc)
            return {"success": False, "message": str(exc)}

    try:
        if hasattr(backend._renderer, "begin_frame"):
            backend._renderer.begin_frame(backend._sim_time)
        if hasattr(backend._renderer, "log_state"):
            backend._renderer.log_state(backend._model, backend._state_0)
        if hasattr(backend._renderer, "end_frame"):
            backend._renderer.end_frame()

        image = None
        if hasattr(backend._renderer, "get_pixels"):
            image = backend._renderer.get_pixels(width=width, height=height)
            logger.debug("Rendered frame %dx%d", width, height)

        return {
            "success": True,
            "backend": render_backend,
            "image": image,
            "sim_time": backend._sim_time,
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def get_observation(
    backend: NewtonBackend, robot_name: Optional[str] = None
) -> Dict[str, Any]:
    """Return current observation for one or all robots."""
    if backend._model is None:
        return {}

    observations = {}

    if robot_name:
        robots = {robot_name: backend._robots.get(robot_name, {})}
    else:
        robots = backend._robots

    try:
        for rname, rinfo in robots.items():
            obs: Dict[str, Any] = {}
            offset = rinfo.get("joint_offset", 0)
            n_joints = rinfo.get("num_joints", backend._joints_per_world)

            try:
                jq = backend._state_0.joint_q.numpy().copy()
                obs["joint_q"] = (
                    jq[offset : offset + n_joints] if n_joints > 0 else jq
                )
            except Exception:
                pass

            try:
                jqd = backend._state_0.joint_qd.numpy().copy()
                obs["joint_qd"] = (
                    jqd[offset : offset + n_joints] if n_joints > 0 else jqd
                )
            except Exception:
                pass

            try:
                obs["body_q"] = backend._state_0.body_q.numpy().copy()
            except Exception:
                pass

            try:
                obs["body_qd"] = backend._state_0.body_qd.numpy().copy()
            except Exception:
                pass

            # Particle state (for cloth/MPM)
            if (
                hasattr(backend._model, "particle_count")
                and backend._model.particle_count > 0
            ):
                try:
                    obs["particle_q"] = backend._state_0.particle_q.numpy().copy()
                    obs["particle_qd"] = backend._state_0.particle_qd.numpy().copy()
                except Exception:
                    pass

            observations[rname] = obs

        return {
            "success": True,
            "sim_time": backend._sim_time,
            "step_count": backend._step_count,
            "observations": observations,
        }
    except Exception as exc:
        return {"success": False, "observations": observations, "error": str(exc)}
