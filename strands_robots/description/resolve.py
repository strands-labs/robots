"""Resolve a registry name / alias / robot_descriptions name / file path to a model file."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

from .registry import urdf_registry


def _import_description(module: str) -> Any:
    """Import a ``robot_descriptions`` module (clones the asset on first use)."""
    try:
        return importlib.import_module(f"robot_descriptions.{module}")
    except ModuleNotFoundError as exc:
        if exc.name == "robot_descriptions":
            raise ImportError("robot_descriptions is not installed: pip install robot_descriptions") from exc
        raise


def _discovered_module(norm: str) -> tuple[str, str] | None:
    """(module, format) from robot_descriptions' static table, MJCF preferred — no import, no network."""
    try:
        from robot_descriptions._descriptions import DESCRIPTIONS, Format
    except ImportError:
        return None
    for suffix, fmt in (("_mj_description", "mjcf"), ("_description", "urdf")):
        mod = f"{norm}{suffix}"
        desc = DESCRIPTIONS.get(mod)
        if desc is not None and (Format.MJCF if fmt == "mjcf" else Format.URDF) in desc.formats:
            return mod, fmt
    return None


def _resolve_asset(norm: str, entry: dict[str, Any] | None) -> tuple[Path, str, dict[str, Any]]:
    """Return (model_path, format, entry) for a registry entry, a discoverable name or a file path."""
    if entry is not None and "asset" in entry:
        module = entry["asset"].get("robot_descriptions_module")
        if module:
            mod = _import_description(module)
            path = getattr(mod, "MJCF_PATH", None) or getattr(mod, "URDF_PATH", None)
            if path is None:
                raise FileNotFoundError(f"robot_descriptions.{module} has neither MJCF_PATH nor URDF_PATH")
            return Path(path), ("mjcf" if module.endswith("_mj_description") else "urdf"), entry
        if (entry["asset"].get("source") or {}).get("type") == "github":
            from . import assets

            path = assets.resolve(entry)
            return path, ("urdf" if path.suffix == ".urdf" else "mjcf"), entry
    if norm in urdf_registry():
        u = urdf_registry()[norm]
        mod = _import_description(u["module"])
        return Path(mod.URDF_PATH), "urdf", {"name": norm, "category": u.get("category", "urdf"), **(entry or {})}
    found = _discovered_module(norm)
    if found is not None:
        module, fmt = found
        mod = _import_description(module)
        path = mod.MJCF_PATH if fmt == "mjcf" else mod.URDF_PATH
        return Path(path), fmt, {"name": norm, "category": "discovered", "discovered": True}
    if entry is not None:
        hw = (entry.get("hardware") or {}).get("lerobot_type")
        raise FileNotFoundError(
            f"robot {norm!r} is in the registry but ships no fetchable simulation model"
            + (f" (hardware only: lerobot_type {hw!r})" if hw else "")
        )
    raise KeyError(
        f"unknown robot {norm!r}: not in the registry, not a robot_descriptions module "
        f"({norm}_mj_description / {norm}_description) and not a path to an MJCF/URDF file"
    )
