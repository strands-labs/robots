"""Newton/Warp lazy loading and model path resolution."""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, Optional, Tuple

from ._types import _PROCEDURAL_ROBOTS, _ROBOT_ALIAS_MAP

logger = logging.getLogger(__name__)

# Lazily imported
_newton_module = None
_warp_module = None


def _ensure_newton():
    """Lazily import Newton and Warp."""
    global _newton_module, _warp_module
    if _newton_module is not None:
        return

    try:
        import warp as wp

        _warp_module = wp
        logger.debug("Warp %s loaded.", getattr(wp, "__version__", "unknown"))
    except ImportError as exc:
        raise ImportError(
            "warp-lang is required for the Newton backend. "
            "Install with: pip install warp-lang"
        ) from exc

    try:
        import newton

        _newton_module = newton
    except ImportError as exc:
        raise ImportError(
            "newton-sim is required for the Newton backend. "
            "Install with: pip install newton-sim"
        ) from exc


def get_newton():
    """Return the cached newton module (call _ensure_newton first)."""
    return _newton_module


def get_warp():
    """Return the cached warp module (call _ensure_newton first)."""
    return _warp_module


def _try_resolve_model_path(name: str) -> Optional[str]:
    """Attempt to resolve a model path via strands_robots.assets."""
    try:
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path(name)
        logger.debug("Resolved '%s' → %s", name, path)
        return str(path)
    except Exception:
        return None


def _try_get_newton_asset(name: str) -> Optional[str]:
    """Try to resolve a Newton bundled asset."""
    try:
        _ensure_newton()
        from newton import examples as ne

        path = ne.get_asset(name)
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return None


def _build_procedural_robot(
    builder: Any,
    wp: Any,
    name: str,
    position: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Build a robot procedurally using Newton ModelBuilder API.

    Creates a serial-chain manipulator from the _PROCEDURAL_ROBOTS registry
    when no URDF/MJCF/USD file is available.

    Returns robot_info dict on success, None if the robot name is not in the registry.
    """
    canonical = _ROBOT_ALIAS_MAP.get(name.lower())
    if canonical is None:
        # Also try stripping common prefixes/suffixes
        for prefix in ("trs_", "bi_"):
            stripped = name.lower().removeprefix(prefix)
            canonical = _ROBOT_ALIAS_MAP.get(stripped)
            if canonical:
                break
    if canonical is None:
        return None

    spec = _PROCEDURAL_ROBOTS[canonical]
    links = spec["links"]

    base_pos = position or (0.0, 0.0, 0.0)
    joint_count_before = getattr(builder, "joint_count", 0)
    body_count_before = getattr(builder, "body_count", 0)

    # Accumulate height for each link
    current_height = base_pos[1]
    parent_body = -1  # -1 = world

    for i, (link_len, radius, axis) in enumerate(links):
        link_len_scaled = link_len * scale
        radius_scaled = radius * scale

        # Position the body at the centre of the link
        body_y = current_height + link_len_scaled * 0.5
        body_pos = (base_pos[0], body_y, base_pos[2])

        body_idx = builder.add_body(
            xform=wp.transform(body_pos, wp.quat_identity()),
            armature=0.01,
        )

        # Add a capsule shape for the link
        builder.add_shape_capsule(
            body=body_idx,
            radius=radius_scaled,
            half_height=link_len_scaled * 0.5,
        )

        # Joint connecting to parent
        joint_pos = (base_pos[0], current_height, base_pos[2])
        builder.add_joint_revolute(
            parent=parent_body,
            child=body_idx,
            parent_xform=wp.transform(joint_pos, wp.quat_identity()),
            child_xform=wp.transform(
                (0.0, -link_len_scaled * 0.5, 0.0), wp.quat_identity()
            ),
            axis=axis,
            limit_lower=-3.14159,
            limit_upper=3.14159,
            target_ke=100.0,
            target_kd=10.0,
        )

        parent_body = body_idx
        current_height += link_len_scaled

    joint_count_after = getattr(builder, "joint_count", 0)
    body_count_after = getattr(builder, "body_count", 0)
    num_new_joints = joint_count_after - joint_count_before
    num_new_bodies = body_count_after - body_count_before

    # Register all new joints as a single articulation (required by Newton)
    if num_new_joints > 0:
        joint_indices = list(range(joint_count_before, joint_count_after))
        try:
            builder.add_articulation(joint_indices, label=name)
        except Exception as exc:
            logger.debug("add_articulation for '%s' failed: %s", name, exc)

    logger.info(
        "Procedurally built robot '%s' (canonical: %s) — %d joints, %d bodies",
        name,
        canonical,
        num_new_joints,
        num_new_bodies,
    )

    return {
        "name": name,
        "model_path": f"procedural:{canonical}",
        "format": "procedural",
        "position": position,
        "scale": scale,
        "num_joints": num_new_joints,
        "num_bodies": num_new_bodies,
        "joint_offset": joint_count_before,
        "body_offset": body_count_before,
    }
