"""The draw behind ``Simulation.randomize``: which scales change, from a factor or a (lo, hi) range, and how the
scene is put back when the engine refuses the result. Pure scene arithmetic; the lock and the rebuild stay with the
world."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from .scene import Scene

Range = float | tuple[float, float] | None
Saved = tuple[float, float, float, dict[str, tuple[float, float]], dict[str, Any]]


def check(scene: Scene, robots: Mapping[str, Any], target: str | None, lighting: Range) -> None:
    if target is not None and target not in robots and not any(o.name == target for o in scene.objects):
        raise KeyError(f"target={target!r} is not a robot or object in this world")
    if target is not None and lighting is not None:
        raise ValueError("lighting is world-wide; randomize(lighting=) takes no target")


def draw(
    scene: Scene,
    *,
    friction: Range,
    mass: Range,
    lighting: Range,
    camera: float | None,
    target: str | None,
    seed: int | None,
) -> Saved:
    """Write the new scales into ``scene`` and return what they were (for ``restore``)."""
    rng = np.random.default_rng(seed)

    def pick(v: Range, current: float) -> float:
        return current if v is None else float(v) if isinstance(v, int | float) else float(rng.uniform(*v))

    was: Saved = (
        scene.friction_scale,
        scene.mass_scale,
        scene.light_scale,
        dict(scene.scales),
        dict(scene.camera_jitter),
    )
    if target is None:
        scene.friction_scale, scene.mass_scale = pick(friction, was[0]), pick(mass, was[1])
    else:
        f, m = scene.scales.get(target, (1.0, 1.0))
        scene.scales[target] = (pick(friction, f), pick(mass, m))
    scene.light_scale = pick(lighting, was[2])
    if camera is not None:
        scene.camera_jitter = {c.name: tuple(rng.uniform(-camera, camera, 3).tolist()) for c in scene.cameras}  # type: ignore[misc]
    return was


def restore(scene: Scene, was: Saved) -> None:
    scene.friction_scale, scene.mass_scale, scene.light_scale, scene.scales, scene.camera_jitter = was


def scales(scene: Scene) -> dict[str, Any]:
    """The world-wide factors, plus ``scales`` per targeted body and ``camera_jitter`` per camera: everything a
    ``randomize`` call can have drawn, so an episode can record exactly what it ran on."""
    return {
        "friction_scale": scene.friction_scale,
        "mass_scale": scene.mass_scale,
        "light_scale": scene.light_scale,
        "scales": {name: {"friction_scale": f, "mass_scale": m} for name, (f, m) in scene.scales.items()},
        "camera_jitter": {name: list(j) for name, j in scene.camera_jitter.items()},
    }
