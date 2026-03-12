"""Deformable objects — cloth, cable, and particle management."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def add_cloth(
    backend: NewtonBackend,
    name: str,
    vertices: Optional[np.ndarray] = None,
    indices: Optional[np.ndarray] = None,
    usd_path: Optional[str] = None,
    usd_prim_path: Optional[str] = None,
    position: Optional[Tuple[float, float, float]] = None,
    rotation: Optional[Tuple[float, float, float, float]] = None,
    velocity: Optional[Tuple[float, float, float]] = None,
    density: float = 100.0,
    scale: float = 1.0,
    tri_ke: float = 10000.0,
    tri_ka: float = 10000.0,
    tri_kd: float = 10.0,
    edge_ke: float = 100.0,
    edge_kd: float = 0.0,
    particle_radius: float = 0.01,
) -> Dict[str, Any]:
    """Add a cloth mesh to the simulation.

    Supports either raw vertices/indices or loading from USD.
    Uses VBD solver for cloth simulation.
    """
    backend._ensure_world()
    wp = backend._wp

    # Load from USD if path provided
    if usd_path and vertices is None:
        try:
            from newton import usd as newton_usd
            from pxr import Usd

            stage = Usd.Stage.Open(usd_path)
            if usd_prim_path:
                prim = stage.GetPrimAtPath(usd_prim_path)
            else:
                prim = stage.GetDefaultPrim()
                usd_prim_path = str(prim.GetPath())
            mesh = newton_usd.get_mesh(stage, prim)
            vertices = mesh.vertices
            indices = mesh.indices
        except Exception as exc:
            return {"success": False, "message": f"USD cloth load failed: {exc}"}

    if vertices is None or indices is None:
        return {
            "success": False,
            "message": "vertices and indices required for cloth.",
        }

    # Convert vertices to warp vec3
    v = [wp.vec3(*vert) for vert in vertices]

    rot = rotation if rotation is not None else wp.quat_identity()
    if isinstance(rotation, (list, tuple)) and not isinstance(
        rotation, type(wp.quat_identity())
    ):
        if hasattr(wp, "quat"):
            rot = wp.quat(*rotation)

    backend._builder.add_cloth_mesh(
        pos=position or (0.0, 0.0, 0.0),
        rot=rot,
        scale=scale,
        vel=velocity or (0.0, 0.0, 0.0),
        vertices=v,
        indices=indices.flatten() if hasattr(indices, "flatten") else indices,
        density=density,
        tri_ke=tri_ke,
        tri_ka=tri_ka,
        tri_kd=tri_kd,
        edge_ke=edge_ke,
        edge_kd=edge_kd,
        particle_radius=particle_radius,
    )

    cloth_info = {
        "name": name,
        "num_vertices": len(vertices),
        "num_triangles": len(indices) // 3,
        "density": density,
        "particle_radius": particle_radius,
    }
    backend._cloths[name] = cloth_info
    backend._model = None  # Invalidate

    logger.info(
        "Cloth '%s' added — %d vertices, %d triangles",
        name,
        len(vertices),
        len(indices) // 3,
    )
    return {
        "success": True,
        "message": f"Cloth '{name}' added",
        "cloth_info": cloth_info,
    }


def add_cable(
    backend: NewtonBackend,
    name: str,
    start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    end: Tuple[float, float, float] = (1.0, 0.0, 0.0),
    num_segments: int = 10,
    radius: float = 0.01,
    density: float = 100.0,
    bend_stiffness: float = 2.0,
    bend_damping: float = 0.5,
) -> Dict[str, Any]:
    """Add a cable/rope to the simulation using chained capsule bodies."""
    backend._ensure_world()
    wp = backend._wp

    try:
        start_vec = np.array(start)
        end_vec = np.array(end)
        direction = end_vec - start_vec
        length = np.linalg.norm(direction)
        seg_length = length / num_segments

        for i in range(num_segments):
            t = (i + 0.5) / num_segments
            pos = tuple(start_vec + direction * t)

            backend._builder.add_body(
                xform=wp.transform(pos, wp.quat_identity()),
                armature=0.01,
            )
            backend._builder.add_shape_capsule(
                body=backend._builder.body_count - 1,
                radius=radius,
                half_height=seg_length * 0.5,
            )

            if i > 0:
                backend._builder.add_joint_ball(
                    parent=backend._builder.body_count - 2,
                    child=backend._builder.body_count - 1,
                )

        backend._cables[name] = {
            "name": name,
            "num_segments": num_segments,
            "length": length,
            "radius": radius,
        }
        backend._model = None
        logger.info(
            "Cable '%s' added — %d segments, length=%.2f.",
            name,
            num_segments,
            length,
        )
        return {"success": True, "message": f"Cable '{name}' added"}
    except Exception as exc:
        return {"success": False, "message": f"Cable creation failed: {exc}"}


def add_particles(
    backend: NewtonBackend,
    name: str,
    positions: Union[np.ndarray, List],
    velocities: Optional[Union[np.ndarray, List]] = None,
    mass: float = 1.0,
    radius: float = 0.01,
) -> Dict[str, Any]:
    """Add particle-based entities (for MPM granular/fluid simulation)."""
    backend._ensure_world()
    wp = backend._wp

    if isinstance(positions, np.ndarray):
        positions = positions.tolist()

    count = 0
    for i, pos in enumerate(positions):
        vel = (
            velocities[i]
            if velocities is not None and hasattr(velocities, "__iter__")
            else (0.0, 0.0, 0.0)
        )
        if isinstance(vel, np.ndarray):
            vel = tuple(vel)
        try:
            backend._builder.add_particle(
                pos=wp.vec3(*tuple(pos)),
                vel=wp.vec3(*tuple(vel)),
                mass=mass,
            )
            count += 1
        except Exception as exc:
            logger.debug("add_particle failed at %d: %s", i, exc)

    backend._model = None
    return {"success": True, "message": f"{count} particles added", "count": count}
