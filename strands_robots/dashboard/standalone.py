"""The standalone bundle: one robot's world and its ONNX skills, packaged so a browser can run them alone.

``python -m strands_robots.dashboard.standalone [out]`` writes (default ``static/scenes/<robot>/``, gitignored):

    scene.json            the shape ``GET /api/scene`` returns, plus a ``standalone`` block
    assets/<robot>/...    the meshes the merged MJCF references
    policy/<skill>.onnx   every skill of the preset, with its ``<skill>.json`` sidecar (onnxruntime-web cannot read
                          ``custom_metadata_map``, so the page reads the layout from the sidecar)

The preset that declares a ``browser`` block in ``policy/onnx/presets.json`` is the one bundled. The MJCF is the
same ``scene_to_mjcf`` the server sends, plus the physics options the MuJoCo backend sets (gravity, implicitfast), so
the browser steps the world the backend would have stepped.
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any

from ..policy.onnx.graph import _session, resolve_weights, write_sidecar
from ..policy.onnx.presets import _table
from .twin_scene import robot_cards, scene_to_mjcf

STATIC = Path(__file__).parent / "static"


def browser_preset() -> tuple[str, dict[str, Any]]:
    """(preset name, facts) of the one preset that declares a ``browser`` block."""
    found = [(name, facts) for name, facts in _table().items() if "browser" in facts]
    if len(found) != 1:
        raise LookupError(f"expected one preset with a 'browser' block in presets.json, found {len(found)}")
    return found[0]


def with_physics(xml: str, gravity: tuple[float, float, float]) -> str:
    """The merged MJCF with the options the MuJoCo backend compiles with."""
    opt = f'gravity="{gravity[0]} {gravity[1]} {gravity[2]}" integrator="implicitfast"'
    if re.search(r"<option\b", xml):
        return re.sub(r"<option\b([^>]*?)/?>", rf"<option\1 {opt}/>", xml, count=1)
    return re.sub(r"(<mujoco\b[^>]*>)", rf"\1<option {opt}/>", xml, count=1)


def _skills(facts: dict[str, Any], dst: Path) -> list[dict[str, Any]]:
    out = []
    for skill, name in facts["files"].items():
        weights = resolve_weights(name, repo=facts["repo"], sha256=facts["sha256"].get(name))
        side = weights.with_name(weights.name + ".json")  # the sidecar write_sidecar keeps next to the graph
        doc = json.loads(side.read_text()) if side.is_file() else None
        if doc is None or "metadata" not in doc:
            doc = json.loads(write_sidecar(weights, _session(weights, ("CPUExecutionProvider",))).read_text())
        shutil.copyfile(weights, dst / name)
        (dst / f"{skill}.json").write_text(json.dumps(doc), encoding="utf-8")
        out.append(
            {
                "name": skill,
                "model": f"policy/{name}",
                "sidecar": f"policy/{skill}.json",
                "commands": doc["metadata"].get("command_names", "twist"),
            }
        )
    return out


def build(out: Path | None = None, *, force: bool = False) -> Path:
    """Write the bundle (atomically) and return its directory; an existing bundle is kept unless ``force``."""
    from .. import Robot, Simulation

    preset, facts = browser_preset()
    spec = facts["browser"]
    out = Path(out) if out is not None else STATIC / "scenes" / spec["robot"]
    if (out / "scene.json").is_file() and not force:
        return out
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    (tmp / "policy").mkdir(parents=True)
    with Simulation("mujoco") as sim:
        body = Robot(spec["robot"], sim=sim)
        p = sim.physics
        if abs(p.timestep * p.substeps * spec["hz"] - 1.0) > 1e-9:
            raise ValueError(
                f"{body.id}: one control step is {p.timestep * p.substeps} s, the skills run at {spec['hz']} Hz"
            )
        sx = scene_to_mjcf(sim.scene)
        for served, src in sx.assets.items():
            (tmp / served).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, tmp / served)
        scene = {
            "xml": with_physics(sx.xml, sim.scene.gravity),
            "assets": [{"path": k, "url": k} for k in sx.assets],
            "robots": robot_cards(sim.scene, sx.roots, {}),
            "objects": [],
            "physics": {"name": p.name, "timestep": p.timestep, "substeps": p.substeps, "backend": "browser"},
            "time": 0.0,
            "revision": sim.revision,
            "stats": sx.stats,
            "standalone": {
                "robot": body.id,
                "root": sx.roots[body.id],
                "hz": spec["hz"],
                "fall_z": spec["fall_z"],
                "stand_z": spec["stand_z"],
                "repo": facts["repo"],
                "preset": preset,
                "skill": spec["skill"],
                "skills": _skills(facts, tmp / "policy"),
            },
        }
    (tmp / "scene.json").write_text(json.dumps(scene), encoding="utf-8")
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    return out


def size_mb(out: Path) -> float:
    return round(sum(f.stat().st_size for f in out.rglob("*") if f.is_file()) / 1048576, 1)


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    where = build(Path(args[0]) if args else None, force="--force" in sys.argv)
    print(json.dumps({"out": str(where), "mb": size_mb(where)}))
