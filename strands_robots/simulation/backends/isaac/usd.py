"""MJCF/URDF to USD with Isaac's own importers, cached by content, with the MuJoCo servo gains authored on.

Without the authored drives every joint of a converted MJCF hangs limp: the importer leaves the PhysX
drive stiffness at 0. Gains come from the COMPILED MuJoCo model (``dampratio`` resolves only there).
"""

from __future__ import annotations

import hashlib
import math
import os
from pathlib import Path

from ...._extras import hint


def usd_cache_dir() -> Path:
    return Path(os.environ.get("STRANDS_ROBOTS_USD_CACHE", "~/.cache/strands_robots/usd")).expanduser()


def demangle(usd_name: str) -> str:
    """Isaac renames USD-invalid prim names (joints called "1".."6") as ``tn__1_``; undo that."""
    return usd_name[4:-1] if usd_name.startswith("tn__") and usd_name.endswith("_") else usd_name


def model_format(path: Path) -> str:
    """``"mjcf"`` for ``.xml`` / ``.mjcf``, ``"urdf"`` for ``.urdf``; anything else (a mesh) is refused with the fix."""
    if path.suffix == ".urdf":
        return "urdf"
    if path.suffix in (".xml", ".mjcf"):
        return "mjcf"
    raise ValueError(
        f"{path.name}: Simulation('isaac') imports a model file, MJCF (.xml) or URDF (.urdf); "
        "a mesh goes inside an MJCF (<asset><mesh file=.../></asset> and a geom of type mesh)"
    )


def object_model(path: Path, *, free: bool) -> tuple[Path, str]:
    """The file to convert for an ``add_object(path=)`` body, and its format.

    A ``free=True`` MJCF is first wrapped in one body with a free joint, as the MuJoCo backend attaches it, so a file
    of bare world geoms is one rigid body here too (Isaac's importer floats a root link, never a world geom). The
    wrapped MJCF is written once into the USD cache, keyed on the file's path and bytes. A URDF has one root link.
    """
    fmt = model_format(path)
    if not free or fmt != "mjcf":
        return path, fmt
    try:
        import mujoco

        from ..mujoco.build import robot_spec_xml
    except ImportError as exc:
        raise ImportError(
            f"a free add_object(path=) on Simulation('isaac') wraps the MJCF with mujoco: {hint('mujoco')}"
        ) from exc
    digest = hashlib.sha256(str(path).encode() + path.read_bytes()).hexdigest()[:16]  # the wrap holds absolute paths
    out = usd_cache_dir() / f"{path.stem}-free-{digest}.xml"
    if not out.is_file():
        spec = mujoco.MjSpec()
        spec.modelname = path.stem
        body = spec.worldbody.add_body(name="body")
        body.add_freejoint(name="free")
        spec.attach(mujoco.MjSpec.from_string(robot_spec_xml(str(path))), prefix="model/", frame=body.add_frame())
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(spec.to_xml())
    return out, "mjcf"


def convert_to_usd(model_path: Path, fmt: str, *, fix_base: bool) -> Path:
    """The USD stage for a model file, converted once and keyed on the file bytes + options."""
    digest = hashlib.sha256(model_path.read_bytes() + f"|{fmt}|fix_base={fix_base}|v1".encode()).hexdigest()[:16]
    out_dir = usd_cache_dir() / f"{model_path.stem}-{digest}"
    marker = out_dir / "USD_PATH"
    if marker.is_file() and Path(marker.read_text()).is_file():
        return Path(marker.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    produced = _import(model_path, fmt, out_dir, fix_base)
    usd = Path(str(produced)) if produced and str(produced).endswith((".usd", ".usda", ".usdc")) else None
    if usd is None or not usd.is_file():
        found = [p for p in sorted(out_dir.rglob("*.usd*")) if p.suffix in (".usd", ".usda", ".usdc")]
        if not found:
            raise RuntimeError(f"Isaac importer produced no USD for {model_path} (returned {produced!r})")
        usd = max(found, key=lambda p: p.stat().st_size)  # the main stage, not a payload layer
    if fmt == "mjcf":
        author_drives(usd, model_path)
    marker.write_text(str(usd))
    return usd


def _import(model_path: Path, fmt: str, out_dir: Path, fix_base: bool) -> object:
    if fmt == "mjcf":
        from isaacsim.asset.importer.mjcf import MJCFImporter, MJCFImporterConfig

        cfg = MJCFImporterConfig()
        cfg.mjcf_path = str(model_path)
        cfg.usd_path = str(out_dir)
        cfg.fix_base = fix_base
        cfg.import_scene = False
        return MJCFImporter(config=cfg).import_mjcf()
    from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

    ucfg = URDFImporterConfig()
    ucfg.urdf_path = str(model_path)
    ucfg.usd_path = str(out_dir)
    ucfg.fix_base = fix_base
    return URDFImporter(config=ucfg).import_urdf()


def servo_gains(model_path: Path) -> dict[str, tuple[float, float, float | None]]:
    """``{joint: (kp, kd, force_limit)}`` for every MuJoCo position servo on a joint."""
    try:
        import mujoco
    except ImportError:
        return {}
    m = mujoco.MjModel.from_xml_path(str(model_path))
    out: dict[str, tuple[float, float, float | None]] = {}
    for i in range(m.nu):
        if int(m.actuator_trntype[i]) != mujoco.mjtTrn.mjTRN_JOINT:
            continue
        kp, bias = float(m.actuator_gainprm[i][0]), m.actuator_biasprm[i]
        if kp <= 0 or float(bias[1]) != -kp:
            continue
        j = int(m.actuator_trnid[i, 0])
        passive = float(m.dof_damping[m.jnt_dofadr[j]])
        force = float(m.actuator_forcerange[i][1]) if m.actuator_forcelimited[i] else None
        out[m.joint(j).name] = (kp, max(0.0, -float(bias[2])) + passive, force)
    return out


def author_drives(usd: Path, model_path: Path) -> list[str]:
    """Write the servo gains onto the converted joints' PhysX drives (angular drives are per DEGREE)."""
    gains = servo_gains(model_path)
    if not gains:
        return []
    from pxr import Usd, UsdPhysics

    stage = Usd.Stage.Open(str(usd))
    default = stage.GetDefaultPrim()
    if default and default.GetVariantSets().HasVariantSet("Physics"):
        vset = default.GetVariantSets().GetVariantSet("Physics")
        if not vset.GetVariantSelection():
            vset.SetVariantSelection("physx" if "physx" in vset.GetVariantNames() else "physics")
    stage.SetEditTarget(stage.GetRootLayer())
    written = []
    for prim in stage.Traverse():
        name = demangle(prim.GetName())
        if name not in gains:
            continue
        if prim.IsA(UsdPhysics.RevoluteJoint):
            inst, scale = "angular", math.pi / 180.0
        elif prim.IsA(UsdPhysics.PrismaticJoint):
            inst, scale = "linear", 1.0
        else:
            continue
        kp, kd, force = gains[name]
        drive = UsdPhysics.DriveAPI.Apply(prim, inst)
        drive.CreateStiffnessAttr().Set(kp * scale)
        drive.CreateDampingAttr().Set(kd * scale)
        if force is not None:
            drive.CreateMaxForceAttr().Set(force)
        written.append(name)
    if written:
        stage.GetRootLayer().Save()
    return written
