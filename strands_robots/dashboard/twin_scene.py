"""The world as the browser needs it: one merged MJCF for kinematics, an asset table, robot cards.

``scene_to_mjcf`` attaches every robot of the world's ``Scene`` with prefix ``<id>/`` (the same names the
MuJoCo backend uses), with each asset ``file`` rewritten to a served name ``assets/<owner>/<relpath>``.
The browser only does kinematics, so this works for any backend.

Two differences from the server's model, both so that a pose streams instead of recompiling the page: a
fixed-base robot hangs under a free root body ``<id>`` (joint ``<id>/__root``), and the revision is the
world's own ``Simulation.revision``.
"""

from __future__ import annotations

import re
import warnings
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .._gl import ensure_gl
from ..simulation import Scene
from ..simulation.backends.mujoco.build import add_object, robot_spec_xml

ensure_gl()
import mujoco  # noqa: E402

_FILE_ATTR = re.compile(r'(\bfile=)"([^"]+)"')
BROWSER_MUJOCO = "3.14.0"  # static/vendor/mujoco.{js,wasm}; NOTICE carries the sha256 (tests/dashboard/test_vendor.py)
_ACTUATOR_INPUT = re.compile(r'\s+input="[a-z0-9]+"')  # newer MuJoCo writes it; the vendored browser build rejects it


@dataclass(frozen=True)
class SceneXml:
    xml: str
    assets: dict[str, str]  # served path -> absolute file path
    roots: dict[str, str]  # robot/object id -> the browser body whose free joint carries its pose
    stats: dict[str, int]


def _robot_spec(path: str) -> mujoco.MjSpec:
    rs = mujoco.MjSpec.from_string(robot_spec_xml(path))
    for g in [g for g in rs.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]:
        rs.delete(g)
    for light in [li for li in rs.lights if li.parent.name == "world"]:
        rs.delete(light)
    return rs


def scene_to_mjcf(scene: Scene) -> SceneXml:
    spec = mujoco.MjSpec()
    spec.option.timestep = scene.timestep
    spec.nkey = sum(len(r.description.keyframes) for r in scene.robots)
    if scene.ground:
        spec.worldbody.add_geom(
            name="ground", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], rgba=[0.75, 0.78, 0.82, 1]
        )
    owners: dict[str, Path] = {}
    for r in scene.robots:
        rs = _robot_spec(str(r.description.model_path))
        if r.description.floating:
            frame = spec.worldbody.add_frame(pos=list(r.position), quat=list(r.orientation))
        else:
            root = spec.worldbody.add_body(name=r.id, pos=list(r.position), quat=list(r.orientation))
            root.mass, root.inertia, root.explicitinertial = 1.0, [0.01, 0.01, 0.01], True
            root.add_freejoint(name=f"{r.id}/__root")
            frame = root.add_frame()
            for b in rs.worldbody.bodies:  # the wrapper carries the placement; the server streams it as a pose
                b.pos, b.quat = [0, 0, 0], [1, 0, 0, 0]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            spec.attach(rs, prefix=f"{r.id}/", frame=frame)
        owners[r.id] = r.description.package_dir
    for o in scene.objects:
        add_object(spec, o)
        if o.path is not None:
            owners[o.name] = Path(o.path).expanduser().resolve().parent
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = spec.compile()
    roots: dict[str, str] = {}
    for b in range(1, model.nbody):
        if model.body_parentid[b] == 0:
            name = model.body(b).name
            roots.setdefault(name.split("/", 1)[0], name)
    assets: dict[str, str] = {}

    def rewrite(m: re.Match[str]) -> str:
        p = Path(m.group(2))
        if not p.is_absolute():
            return m.group(0)
        owner = next((k for k, d in owners.items() if p.is_relative_to(d)), None)
        rel = p.relative_to(owners[owner]).as_posix() if owner else p.name
        served = f"assets/{owner or 'misc'}/{rel}"
        assets[served] = str(p)
        return f'{m.group(1)}"{served}"'

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        xml = _FILE_ATTR.sub(rewrite, spec.to_xml())
    xml = _ACTUATOR_INPUT.sub("", xml)  # the browser only does kinematics: gains/bias are enough
    xml = re.sub(r"<compiler[^>]*/>", '<compiler angle="radian" autolimits="true"/>', xml, count=1)
    stats = {"nq": model.nq, "nu": model.nu, "ngeom": model.ngeom, "nmesh": model.nmesh, "nbody": model.nbody}
    return SceneXml(xml, assets, roots, stats)


def robot_cards(scene: Scene, roots: Mapping[str, str], real: Mapping[str, Any]) -> list[dict[str, Any]]:
    """One card per robot in the scene; ``real`` maps a card id to the hardware robot it stands for."""
    out = []
    for sr in scene.robots:
        rid, d = sr.id, sr.description
        owner = real.get(rid)
        out.append(
            {
                "id": rid,
                "name": d.name,
                "joints": [f"{rid}/{d.unlabel(k)}" for k in d.action_keys],
                "labels": list(d.action_keys),
                "action_keys": list(d.action_keys),
                "limits": {k: [float(a), float(b)] for k, (a, b) in d.limits.items()},
                "floating": d.floating,
                "root": roots.get(rid),
                "position": list(sr.position),
                "is_real": owner is not None,
                "is_twin_of": owner.id if owner is not None and owner.id != rid else None,
            }
        )
    return out
