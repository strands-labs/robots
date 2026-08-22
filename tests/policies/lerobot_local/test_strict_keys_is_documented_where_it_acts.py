# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A flag documented for one of the surfaces it gates licenses a caller's wrong model of it.

``strict_keys`` is a public constructor parameter. Its ``Args:`` entry defined it
as a camera concern - "if any camera name cannot be matched to a declared policy
image key by exact name" - which was accurate when the flag was added (#614,
2026-06-23): it gated two refusals then, both on the camera path. Seven days
later a joint-state refusal was gated on the same flag (#897), and twelve days
after that a second one (#1099). The entry never followed. So a caller who read
it, saw every camera bind by exact name, and set ``strict_keys=True`` expecting a
no-op instead converted a joint-state *warning* into a hard ``ValueError`` -
measured on an aloha-style mimic gripper, one configured key the sim does not
report: ``strict_keys=False`` warns once and returns a 7-value vector, and
``strict_keys=True`` refuses.

The sibling that got the flag in the same PR is the counter-evidence.
``Gr00tPolicy``'s entry says "if auto-inferred observation/action keys cannot be
matched", naming the whole key surface, and is still accurate. Both joint-state
methods document their own ``strict_keys`` condition (``_resolve_state_order``
ships an explicit two-bullet list of the True and False branches), and
``docs/policies/lerobot-local.md`` documents the joint-state raise in its body -
so the code, the methods and the page agreed, and only the flag's own definition
and that page's one-line gloss did not.

Both rules here derive what they grade from the tree rather than listing it:

* :class:`TestEveryGateDocumentsTheFlag` - a method whose behaviour the flag
  changes must name it in its own docstring, because that docstring is where a
  caller learns what the method does. Two of the four gates did not.
* :class:`TestTheFlagsEntryNamesEverySurfaceItGoverns` - for each gating method,
  the public attribute it reads alongside the flag is the surface that gate is
  about (``camera_key_map`` for the camera gates, ``robot_state_keys`` for the
  joint-state ones). Every such attribute must be named in the flag's own entry.
  ``camera_key_map`` was; ``robot_state_keys`` was not.

The class set is derived too - any class that assigns ``self.strict_keys`` is
graded - so a second class taking the flag is held to the same rule on arrival.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots as package

_PACKAGE_ROOT = Path(package.__file__).resolve().parent
_FLAG = "strict_keys"


def _classes_carrying_the_flag() -> list[tuple[Path, ast.ClassDef]]:
    """Every class that assigns ``self.strict_keys``, wherever it lives."""
    found: list[tuple[Path, ast.ClassDef]] = []
    for path in sorted(_PACKAGE_ROOT.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a parse failure is its own bug
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue
            if any(
                isinstance(n, ast.Attribute)
                and n.attr == _FLAG
                and isinstance(n.ctx, ast.Store)
                and isinstance(n.value, ast.Name)
                and n.value.id == "self"
                for n in ast.walk(node)
            ):
                found.append((path.relative_to(_PACKAGE_ROOT.parent), node))
    return found


def _own_nodes(fn: ast.FunctionDef | ast.AsyncFunctionDef) -> list[ast.AST]:
    """Nodes belonging to ``fn`` itself, not to a function nested inside it."""
    nested = {
        id(inner)
        for child in ast.walk(fn)
        if child is not fn and isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        for inner in ast.walk(child)
    }
    return [n for n in ast.walk(fn) if id(n) not in nested]


def _self_reads(nodes: list[ast.AST]) -> set[str]:
    return {
        n.attr
        for n in nodes
        if isinstance(n, ast.Attribute)
        and isinstance(n.ctx, ast.Load)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
    }


def _public_attributes(cls: ast.ClassDef) -> set[str]:
    return {
        n.attr
        for n in ast.walk(cls)
        if isinstance(n, ast.Attribute)
        and isinstance(n.ctx, ast.Store)
        and isinstance(n.value, ast.Name)
        and n.value.id == "self"
        and not n.attr.startswith("_")
    }


def _gates(cls: ast.ClassDef) -> list[tuple[str, ast.FunctionDef | ast.AsyncFunctionDef, set[str]]]:
    """Methods that READ the flag, with the public attributes they read beside it.

    The constructor's ``self.strict_keys = strict_keys`` is a write, not a gate,
    so it is excluded by requiring a Load context.
    """
    out = []
    for fn in cls.body:
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        nodes = _own_nodes(fn)
        reads = _self_reads(nodes)
        if _FLAG not in reads:
            continue
        subjects = (reads & _public_attributes(cls)) - {_FLAG}
        out.append((fn.name, fn, subjects))
    return out


def _args_entry(doc: str, param: str) -> str:
    """The ``Args:`` entry for ``param``, whitespace-normalised.

    ``ast.get_docstring`` dedents, so an entry sits at one indent level and its
    continuation lines one deeper; the entry ends at the next entry or block.
    """
    parts = doc.split("Args:", 1)
    if len(parts) < 2:
        return ""
    match = re.search(rf"^\s*{re.escape(param)}:(.*?)(?=^\s{{0,8}}\w+:|\Z)", parts[1], re.S | re.M)
    return " ".join((match.group(1) if match else "").split())


_CARRIERS = _classes_carrying_the_flag()
_GATE_CASES = [
    pytest.param(path, cls, name, fn, subjects, id=f"{cls.name}.{name}")
    for path, cls in _CARRIERS
    for name, fn, subjects in _gates(cls)
]


class TestTheScanIsNonVacuous:
    """A clean sweep must mean the rules held, not that they graded nothing."""

    def test_at_least_one_class_carries_the_flag(self) -> None:
        assert _CARRIERS, f"no class in {_PACKAGE_ROOT.name} assigns self.{_FLAG} - the scan found nothing to grade"

    def test_every_carrier_has_gates_and_an_entry(self) -> None:
        for path, cls in _CARRIERS:
            gates = _gates(cls)
            assert gates, f"{path}::{cls.name} assigns self.{_FLAG} but no method reads it"
            entry = _args_entry(ast.get_docstring(cls) or "", _FLAG)
            assert entry, f"{path}::{cls.name} carries {_FLAG} but documents no Args: entry for it"

    def test_a_gate_carries_the_subject_it_is_about(self) -> None:
        """Each gate reads exactly the public attribute its refusal is about.

        This is what makes the entry rule derivable rather than a listed table:
        if a gate read no public attribute beside the flag, that rule would pass
        vacuously for it.
        """
        for _path, cls in _CARRIERS:
            for name, _fn, subjects in _gates(cls):
                assert subjects, f"{cls.name}.{name} gates on {_FLAG} but reads no public attribute beside it"


class TestEveryGateDocumentsTheFlag:
    """A method whose behaviour the flag changes names it in its own docstring."""

    @pytest.mark.parametrize(("path", "cls", "name", "fn", "subjects"), _GATE_CASES)
    def test_the_gate_names_the_flag(
        self,
        path: Path,
        cls: ast.ClassDef,
        name: str,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        subjects: set[str],
    ) -> None:
        doc = ast.get_docstring(fn) or ""
        assert doc, f"{path}::{cls.name}.{name} gates on {_FLAG} and has no docstring at all"
        assert _FLAG in doc, (
            f"{path}::{cls.name}.{name} behaves differently under {_FLAG} "
            f"(it reads self.{_FLAG} and self.{sorted(subjects)[0]}) but its docstring never names the flag, "
            "so a caller reading this method cannot tell that setting it changes what this method does"
        )


class TestTheFlagsEntryNamesEverySurfaceItGoverns:
    """The flag's own entry names every surface its gates are about."""

    @pytest.mark.parametrize(("path", "cls", "name", "fn", "subjects"), _GATE_CASES)
    def test_the_entry_names_this_gates_subject(
        self,
        path: Path,
        cls: ast.ClassDef,
        name: str,
        fn: ast.FunctionDef | ast.AsyncFunctionDef,
        subjects: set[str],
    ) -> None:
        entry = _args_entry(ast.get_docstring(cls) or "", _FLAG)
        missing = sorted(s for s in subjects if s not in entry)
        assert not missing, (
            f"{path}::{cls.name}.{name} refuses on {missing} under {_FLAG}, but the {_FLAG} "
            f"Args: entry never names {missing} - a caller who reads the entry cannot anticipate "
            f"that refusal. Entry reads: {entry!r}"
        )


class TestTheJointStateHalfIsWhatTheEntryOmitted:
    """The measured behaviour the entry did not describe, pinned directly.

    The camera half was already documented, so it is the control: if the entry is
    ever narrowed back the other way, this class still passes while the rule
    above fails, which keeps the two questions separate.
    """

    def test_the_entry_names_both_halves(self) -> None:
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        entry = _args_entry(inspect.getdoc(LerobotLocalPolicy) or "", _FLAG)
        assert "camera" in entry, f"the camera half is no longer described: {entry!r}"
        assert "robot_state_keys" in entry, (
            "the joint-state half is undescribed: strict_keys also refuses when configured "
            f"robot_state_keys are missing from the observation. Entry reads: {entry!r}"
        )

    def test_a_no_op_claim_is_only_made_for_a_robot_that_binds_everything(self) -> None:
        """The entry may call the flag a no-op only for names that all bind.

        Claiming it is a no-op "when cameras resolve" would be the same defect in
        the other direction, since the joint-state gates are still live.
        """
        from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy

        entry = _args_entry(inspect.getdoc(LerobotLocalPolicy) or "", _FLAG)
        if "no-op" not in entry:
            return
        claim = entry[entry.index("no-op") - 160 : entry.index("no-op") + 40]
        assert "joint" in claim or "robot_state_keys" in claim, (
            "the entry calls strict_keys a no-op without qualifying the joint-state gates: " + repr(claim)
        )
