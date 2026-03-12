"""Robot management — add robot from URDF, MJCF, USD, or procedural fallback."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ._registry import (
    _build_procedural_robot,
    _try_get_newton_asset,
    _try_resolve_model_path,
)

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def add_robot(
    backend: NewtonBackend,
    name: str,
    urdf_path: Optional[str] = None,
    usd_path: Optional[str] = None,
    data_config: Optional[str] = None,
    position: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0,
) -> Dict[str, Any]:
    """Add a robot from URDF, MJCF, or USD.

    Phase 1.4: Now supports USD loading via add_usd().
    """
    backend._ensure_world()

    # Resolve model path
    model_path = urdf_path or usd_path or data_config
    if model_path is None:
        model_path = name

    # Try to resolve path
    if not Path(str(model_path)).exists():
        resolved = _try_resolve_model_path(model_path)
        if resolved:
            model_path = resolved
        else:
            asset = _try_get_newton_asset(model_path)
            if asset:
                model_path = asset

    model_path_obj = Path(str(model_path))
    if not model_path_obj.exists():
        # Try as a known robot name
        asset = _try_get_newton_asset(name)
        if asset:
            model_path = asset
            model_path_obj = Path(model_path)

    suffix = model_path_obj.suffix.lower() if model_path_obj.exists() else ""
    logger.info("Adding robot '%s' from %s", name, model_path)

    wp = backend._wp
    xform = None
    if position is not None:
        xform = wp.transform(position, wp.quat_identity())

    try:
        joint_count_before = getattr(backend._builder, "joint_count", 0)
        body_count_before = getattr(backend._builder, "body_count", 0)

        fmt = suffix
        used_procedural = False

        if suffix in (".usda", ".usd"):
            backend._builder.add_usd(
                str(model_path), xform=xform, scale=scale,
            )
        elif suffix == ".urdf":
            used_procedural, fmt = _try_load_urdf(
                backend, name, model_path, xform, scale, position
            )
        elif suffix == ".xml":
            used_procedural, fmt = _try_load_mjcf(
                backend, name, model_path, xform, scale, position
            )
        else:
            used_procedural, fmt = _try_load_any(
                backend, name, model_path, xform, scale, position
            )

        if used_procedural:
            robot_info = backend._robots[name]  # set by _build_procedural fallback
        else:
            joint_count_after = getattr(backend._builder, "joint_count", 0)
            body_count_after = getattr(backend._builder, "body_count", 0)

            robot_info = {
                "name": name,
                "model_path": str(model_path),
                "format": fmt,
                "position": position,
                "scale": scale,
                "num_joints": joint_count_after - joint_count_before,
                "num_bodies": body_count_after - body_count_before,
                "joint_offset": joint_count_before,
                "body_offset": body_count_before,
            }
            backend._robots[name] = robot_info

        # Invalidate model so it gets rebuilt
        backend._model = None
        backend._replicated = False
        backend._joints_per_world = getattr(backend._builder, "joint_count", 0)
        backend._bodies_per_world = getattr(backend._builder, "body_count", 0)

        num_j = robot_info.get("num_joints", 0)
        num_b = robot_info.get("num_bodies", 0)
        return {
            "success": True,
            "num_joints": num_j,
            "robot_info": robot_info,
            "message": f"Robot '{name}' added ({num_j} joints, {num_b} bodies)",
        }
    except Exception as exc:
        return {"success": False, "message": str(exc)}


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _try_load_urdf(
    backend: NewtonBackend, name, model_path, xform, scale, position
) -> Tuple[bool, str]:
    """Try URDF load, fall back to procedural."""
    from ._scene import recreate_builder

    try:
        backend._builder.add_urdf(
            str(model_path), xform=xform, scale=scale, collapse_fixed_joints=True,
        )
        return False, ".urdf"
    except Exception as urdf_exc:
        logger.warning(
            "URDF load failed for '%s': %s — trying procedural", model_path, urdf_exc,
        )
        recreate_builder(backend)
        return _procedural_or_raise(backend, name, position, scale, urdf_exc)


def _try_load_mjcf(
    backend: NewtonBackend, name, model_path, xform, scale, position
) -> Tuple[bool, str]:
    """Try MJCF load, fall back to procedural."""
    from ._scene import recreate_builder

    try:
        backend._builder.add_mjcf(str(model_path), xform=xform, scale=scale)
        return False, ".xml"
    except Exception as mjcf_exc:
        logger.warning(
            "MJCF load failed for '%s': %s — trying procedural", model_path, mjcf_exc,
        )
        recreate_builder(backend)
        return _procedural_or_raise(backend, name, position, scale, mjcf_exc)


def _try_load_any(
    backend: NewtonBackend, name, model_path, xform, scale, position
) -> Tuple[bool, str]:
    """Try URDF then MJCF, fall back to procedural."""
    from ._scene import recreate_builder

    for loader_name, loader_fn in [
        ("urdf", backend._builder.add_urdf),
        ("mjcf", backend._builder.add_mjcf),
    ]:
        try:
            loader_fn(str(model_path), xform=xform, scale=scale)
            return False, loader_name
        except Exception:
            continue

    # All loaders failed — recreate builder and try procedural
    recreate_builder(backend)
    proc_info = _build_procedural_robot(
        backend._builder, backend._wp, name, position=position, scale=scale,
    )
    if proc_info is not None:
        backend._robots[name] = proc_info
        return True, "procedural"

    raise FileNotFoundError(
        f"No model file found for '{name}' and no procedural "
        f"definition available. Provide urdf_path, usd_path, or "
        f"add '{name}' to _PROCEDURAL_ROBOTS registry."
    )


def _procedural_or_raise(
    backend: NewtonBackend, name, position, scale, original_exc
) -> Tuple[bool, str]:
    """Try procedural build or re-raise the original exception."""
    proc_info = _build_procedural_robot(
        backend._builder, backend._wp, name, position=position, scale=scale,
    )
    if proc_info is not None:
        backend._robots[name] = proc_info
        return True, "procedural"
    raise original_exc
