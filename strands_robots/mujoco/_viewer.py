"""Viewer management — open, close interactive 3D viewer."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict

from ._registry import get_mujoco_viewer

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def open_viewer(sim: MujocoBackend) -> Dict[str, Any]:
    """Open interactive 3D viewer window."""
    if sim._world is None or sim._world._model is None:
        return {
            "status": "error",
            "content": [{"text": "❌ No simulation to view."}],
        }

    viewer = get_mujoco_viewer()
    if viewer is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": "❌ mujoco.viewer not available. Install: pip install mujoco"
                }
            ],
        }

    if sim._viewer_handle is not None:
        return {
            "status": "success",
            "content": [{"text": "👁️ Viewer already open."}],
        }

    try:
        sim._viewer_handle = viewer.launch_passive(
            sim._world._model, sim._world._data
        )
        return {
            "status": "success",
            "content": [
                {
                    "text": "👁️ Interactive viewer opened. Close window or use action='close_viewer'."
                }
            ],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Viewer failed: {e}"}]}


def close_viewer_internal(sim: MujocoBackend):
    """Internal: close viewer if open."""
    if sim._viewer_handle is not None:
        try:
            sim._viewer_handle.close()
        except Exception:
            pass
        sim._viewer_handle = None


def close_viewer(sim: MujocoBackend) -> Dict[str, Any]:
    close_viewer_internal(sim)
    return {"status": "success", "content": [{"text": "👁️ Viewer closed."}]}
