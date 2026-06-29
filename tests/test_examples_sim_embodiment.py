"""Sim examples must not pass a hardware-only embodiment to ``create_policy``.

A recurring documentation defect: a MuJoCo sim example is copied from a hardware
example and keeps ``embodiment="so_real"``. The ``*_real`` embodiments declare
the lerobot driver's ``<motor>.pos`` joint keys (e.g. ``shoulder_pan.pos``),
which never match the bare-numeric MuJoCo joint names (``"1".."6"``). The
``PackStateProcessorStep`` then finds zero state keys, never composes
``observation.state``, and a state-conditioned policy (MolmoAct2) fails deep
inside the lerobot processor pipeline with ``requires observation.state``.

This statically scans top-level example scripts: any example that constructs a
SIM robot (``Robot(...)`` without ``mode="real"``, or ``create_simulation(...)``)
and passes a hardware (``*_real``) embodiment to ``create_policy`` is a defect.
Pure-hardware examples (``Robot(..., mode="real")``) are exempt - ``so_real`` is
correct there.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"
_EMBODIMENTS_JSON = _REPO_ROOT / "strands_robots" / "policies" / "lerobot_local" / "embodiments.json"


def _hardware_embodiments() -> set[str]:
    """Names (configs + aliases) that resolve to a hardware (``*_real``) config.

    Derived from ``embodiments.json`` so the rule tracks the registry rather than
    a hand-maintained list. A config is hardware iff its name ends in ``_real``;
    an alias is hardware iff its target config is.
    """
    raw = json.loads(_EMBODIMENTS_JSON.read_text(encoding="utf-8"))
    configs = raw.get("configs", {})
    aliases = raw.get("aliases", {})
    hardware = {name for name in configs if name.endswith("_real")}
    hardware |= {alias for alias, target in aliases.items() if target in hardware}
    return hardware


def _string_value(node: ast.AST) -> str | None:
    """Return the literal string value of ``node``, else ``None``."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return None


def _is_sim_robot_example(tree: ast.AST) -> bool:
    """True if the script constructs a sim robot.

    Sim = a ``Robot(...)`` call WITHOUT ``mode="real"`` (sim is the default), or a
    ``create_simulation(...)`` call. A script that only ever builds
    ``Robot(..., mode="real")`` robots is treated as hardware-only.
    """
    has_sim = False
    has_real_only_robot = False
    saw_robot = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name == "create_simulation":
            return True
        if name == "Robot":
            saw_robot = True
            mode = None
            for kw in node.keywords:
                if kw.arg == "mode":
                    mode = _string_value(kw.value)
            if mode == "real":
                has_real_only_robot = True
            else:
                has_sim = True
    if has_sim:
        return True
    # Only real robots constructed -> not a sim example.
    if saw_robot and has_real_only_robot:
        return False
    return False


def _create_policy_embodiments(tree: ast.AST) -> list[str]:
    """Collect literal ``embodiment=`` string kwargs passed to ``create_policy``."""
    found: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.id if isinstance(func, ast.Name) else getattr(func, "attr", None)
        if name != "create_policy":
            continue
        for kw in node.keywords:
            if kw.arg == "embodiment":
                val = _string_value(kw.value)
                if val is not None:
                    found.append(val)
    return found


def _example_scripts() -> list[Path]:
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize("script", _example_scripts(), ids=[p.name for p in _example_scripts()])
def test_sim_example_uses_sim_embodiment(script: Path) -> None:
    """Sim examples must not pass a hardware (``*_real``) embodiment."""
    tree = ast.parse(script.read_text(encoding="utf-8"))
    if not _is_sim_robot_example(tree):
        pytest.skip(f"{script.name} is not a sim example")
    hardware = _hardware_embodiments()
    used = _create_policy_embodiments(tree)
    offending = [e for e in used if e in hardware]
    assert not offending, (
        f"{script.name} builds a sim robot but passes hardware embodiment(s) "
        f"{offending} to create_policy. Hardware ('*_real') embodiments declare "
        f"'<motor>.pos' joint keys that never match the MuJoCo bare-numeric joint "
        f"names; observation.state ends up empty. Use the sim embodiment (e.g. "
        f'"so101"/"so100") for MuJoCo.'
    )
