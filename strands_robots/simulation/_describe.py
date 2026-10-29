"""``Simulation.describe()``'s document: robots (spawn frame, live pose, link names), objects where they are now,
cameras, physics, time. JSON-safe, built under the world lock by the caller. ``known`` names what ``pose()`` answers."""

from __future__ import annotations

from typing import Any


def describe(sim: Any) -> dict[str, Any]:
    p = sim.physics
    return {
        "backend": sim.backend_name,
        "physics": {
            "name": p.name,
            "timestep": p.timestep,
            "substeps": p.substeps,
            "envs": sim.num_envs,
            "deterministic": sim.backend.deterministic,  # same scene + actions -> bit-identical trajectory
        },
        "time": sim.time,
        "revision": sim.revision,
        "manual_time": sim.time_is_manual,
        "robots": {
            rid: r.description.describe()
            | {"position": list(sim.scene.robot(rid).position), "pose": where(sim, rid)}
            | {"links": sim.backend.links(rid)}
            for rid, r in sim._robots.items()
        },
        "objects": [{"name": o.name, "shape": o.shape or o.path} | where(sim, o.name) for o in sim.scene.objects],
        "cameras": [{"name": c.name, "size": list(c.size), "attach": c.attach} for c in sim.scene.cameras],
    }


def where(sim: Any, body: str) -> dict[str, list[float]]:
    try:
        p = sim.backend.body_pose(body)
    except KeyError:  # added inside ``with sim.editing():`` and not built yet: the spawn frame it will take
        spawn = next(
            x for x in [*sim.scene.robots, *sim.scene.objects] if getattr(x, "id", getattr(x, "name", None)) == body
        )
        return {"position": [float(v) for v in spawn.position], "orientation": [float(v) for v in spawn.orientation]}
    return {"position": p.position.tolist(), "orientation": p.orientation.tolist()}


def known(sim: Any, body: str) -> str:
    """``body`` if ``pose()`` can answer it: a robot id, an object, or ``"<robot>/<link>"`` from describe()'s links."""
    rid, _, link = body.partition("/")
    if body in sim._robots or any(o.name == body for o in sim.scene.objects):
        return body
    if rid in sim._robots and link in sim.backend.links(rid):
        return body
    name = link or body
    hint = [f"{r}/{name}" for r in sim._robots if name in sim.backend.links(r)]
    raise KeyError(f"no body {body!r} in this world" + (f"; a robot link is '<robot>/<link>': {hint}" if hint else ""))
