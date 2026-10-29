"""What the world looks through and with: a camera moved where it stands, and the lights ``Simulation.light()`` adds.
A move the engine can make in place (MuJoCo writes the compiled camera) costs no rebuild; otherwise, and inside
``with sim.editing():``, the scene changes and the world rebuilds, putting the old entry back if that fails."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..geometry import vec
from .scene import SceneCamera, SceneLight


def light_vectors(position: Any, direction: Any, color: Any) -> tuple[Any, Any, Any]:
    pos, dirn = vec(position, 3, "position"), vec(direction, 3, "direction")
    rgb = vec(color, 3, "color (RGB 0-1)")
    if not np.isfinite([*pos, *dirn, *rgb]).all() or float(np.linalg.norm(dirn)) < 1e-9:
        raise ValueError(f"a light needs finite numbers and a non-zero direction, got {pos}, {dirn}, {rgb}")
    if not all(0.0 <= c <= 1.0 for c in rgb):
        raise ValueError(f"color is RGB 0-1, got {rgb}")
    return pos, dirn, rgb


def _swap(ops: Any, items: list[Any], old: Any, new: Any) -> None:
    """Put ``new`` where ``old`` stands and rebuild; a rebuild that fails leaves ``old`` there."""
    i = items.index(old)
    items[i] = new
    try:
        ops._reload()
    except Exception:
        items[i] = old
        raise


def put_camera(ops: Any, old: SceneCamera | None, cam: SceneCamera) -> None:
    with ops.lock:
        if old is None:
            ops._commit(ops.scene.cameras, cam)
        elif not (ops._editing or ops._pending) and ops.backend.move_camera(old, cam):
            ops.scene.cameras[ops.scene.cameras.index(old)] = cam
            ops.revision += 1
        else:
            _swap(ops, ops.scene.cameras, old, cam)


def light(ops: Any, lit: SceneLight) -> SceneLight:
    s = ops.scene
    if lit.name in ops._robots or any(x.name == lit.name for x in [*s.objects, *s.cameras]):
        raise ValueError(f"name {lit.name!r} is already used in this world")
    with ops.lock:
        old = next((li for li in s.lights if li.name == lit.name), None)
        if old is None:
            ops._commit(s.lights, lit)
        else:
            _swap(ops, s.lights, old, lit)
    return lit
