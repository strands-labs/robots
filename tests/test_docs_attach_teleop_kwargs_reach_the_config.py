"""Every kwarg a documented ``attach_teleop("<type>", ...)`` passes must land.

``TeleopMixin.attach_teleop`` reads ``name``, ``method`` and ``map_fn`` and
forwards every other keyword to ``Teleoperator(type, **kwargs)``, whose
``_build_teleop_config`` refuses a kwarg that is neither on the resolved lerobot
config dataclass nor on the cross-device allowlist. That refusal is a
``ValueError`` raised before any device is touched, so a fence that passes a
``teleoperate()`` kwarg (``robot_name=``) to ``attach_teleop`` fails for every
reader on the first line, hardware or not.

This grades every ``python`` fence in ``docs/**/*.md`` and ``README.md``: for
each ``attach_teleop`` call whose type is a string literal, every keyword must be
one the mixin reads or one ``_build_teleop_config`` would accept for that type.
The dataclass is resolved through lerobot's own registry, so the oracle is the
one the factory consults.
"""

from __future__ import annotations

import ast
import dataclasses
import re
from pathlib import Path

import pytest

import strands_robots
from strands_robots.teleop_mixin import TeleopMixin
from strands_robots.teleoperator import (
    _FORWARDABLE_TELEOP_KWARGS,
    _ensure_lerobot_teleoperators_registered,
)

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent
_PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)


def _documentation_files() -> list[Path]:
    files = sorted((_REPO_ROOT / "docs").rglob("*.md"))
    readme = _REPO_ROOT / "README.md"
    return [*files, readme] if readme.is_file() else files


def _mixin_keywords() -> set[str]:
    """The keywords ``attach_teleop`` consumes itself (everything else is forwarded)."""
    signature = ast.parse(Path(TeleopMixin.attach_teleop.__code__.co_filename).read_text())
    for node in ast.walk(signature):
        if isinstance(node, ast.FunctionDef) and node.name == "attach_teleop":
            return {arg.arg for arg in node.args.kwonlyargs}
    raise AssertionError("premise: TeleopMixin.attach_teleop is no longer a def with keyword-only args")


def _accepted_by_factory(teleop_type: str) -> set[str]:
    """What ``_build_teleop_config`` recognises for ``teleop_type``: dataclass fields + allowlist + id."""
    from lerobot.teleoperators.config import TeleoperatorConfig

    _ensure_lerobot_teleoperators_registered()
    config_class = TeleoperatorConfig.get_choice_class(teleop_type)
    fields = {f.name for f in dataclasses.fields(config_class)}
    return fields | set(_FORWARDABLE_TELEOP_KWARGS) | {"id"}


def _attach_calls(source: str) -> list[tuple[str, list[str]]]:
    """``(teleop_type, keyword names)`` for every literal-typed ``attach_teleop`` call."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return []  # a fence with ``...`` placeholders in statement position is prose
    calls: list[tuple[str, list[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        callee = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if callee != "attach_teleop" or not node.args:
            continue
        first = node.args[0]
        if not (isinstance(first, ast.Constant) and isinstance(first.value, str)):
            continue  # a pre-built device: its kwargs are refused by the mixin itself
        calls.append((first.value, [kw.arg for kw in node.keywords if kw.arg is not None]))
    return calls


def _fenced_attach_calls() -> list[tuple[str, str, list[str]]]:
    found: list[tuple[str, str, list[str]]] = []
    for path in _documentation_files():
        for fence in _PYTHON_FENCE.findall(path.read_text(encoding="utf-8")):
            for teleop_type, keywords in _attach_calls(fence):
                found.append((str(path.relative_to(_REPO_ROOT)), teleop_type, keywords))
    return found


_CASES = _fenced_attach_calls()


def test_the_docs_still_attach_a_teleoperator_by_type() -> None:
    """Premise guard: an empty case list would pass the grader below vacuously."""
    assert _CASES, "no documented attach_teleop('<type>', ...) call found under docs/ or README.md"


@pytest.mark.parametrize(
    ("page", "teleop_type", "keywords"),
    _CASES,
    ids=[f"{page}:{teleop_type}" for page, teleop_type, _ in _CASES],
)
def test_every_documented_attach_teleop_kwarg_is_accepted(page: str, teleop_type: str, keywords: list[str]) -> None:
    """A kwarg the mixin does not read must be one the factory accepts for that type."""
    forwarded = set(keywords) - _mixin_keywords()
    refused = sorted(forwarded - _accepted_by_factory(teleop_type))
    assert not refused, (
        f"{page}: attach_teleop({teleop_type!r}, ...) passes {refused}, which neither "
        f"attach_teleop reads nor Teleoperator({teleop_type!r}) accepts; the fence raises "
        "ValueError before any device is touched (robot_name= belongs on teleoperate())"
    )
