"""A documented ``Attributes:`` block is the only field list a caller reads.

A configuration dataclass in this package is a caller-facing surface, not an
internal record: :class:`~strands_robots.policies.lerobot_local.embodiment.EmbodimentMap`
is exported and is built straight from a caller-supplied dict, and
:class:`~strands_robots.policies.protomotions.config.ProtoMotionsConfig` is
resolved from a checkpoint's sidecar. So the class's ``Attributes:`` block is
where a caller learns which keys that dict may carry - and a field missing from
it is a field the caller does not know to set.

That is not a cosmetic gap. ``EmbodimentMap`` documented five of its ten fields,
and the five it omitted were the unit-conversion group: ``state_units`` /
``action_units`` decide whether the map converts at all, and ``gripper_index`` /
``gripper_joint_range`` / ``joint_mids`` parameterise the conversion. Their
defaults are ``"native"`` / ``-1`` / empty, so a caller writing an embodiment
from that block got a map that converts nothing, ``validate()`` accepted it, and
a degrees-trained SO-arm checkpoint's raw ``90.0`` reached a sim joint whose
range is ``+/-1.9199`` radians. ``ProtoMotionsConfig`` omitted ``onnx_in_names``
- while another docstring in the same package cites
``:attr:`ProtoMotionsConfig.onnx_in_names``` as where its shapes are documented.

``tests/test_args_docstring_completeness.py`` pins exactly this comparison for
``Args:`` blocks, and its scope is a function signature, so an attribute list
was graded by nothing. These tests close that: for every dataclass whose
docstring *already* has an ``Attributes:`` block, every field it declares must
have an entry. A class with no ``Attributes:`` block at all is out of scope -
this checks a block that exists for completeness rather than demanding one,
which is the same boundary the ``Args:`` guard draws.

A field documented on a base class counts, so a subclass may document only what
it adds; and one entry may name several fields that share a description
(``actor_obs_keys / critic_obs_keys: ...``), matching the ``Args:`` guard's own
label handling. The scan is AST-only, so no optional backend has to be
importable for a config dataclass behind one to be graded.
"""

from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import pytest

import strands_robots as package

# Derived from an imported symbol rather than a path literal, so a moved package
# cannot leave this scanning an empty tree while reporting success.
_PACKAGE_ROOT = Path(inspect.getfile(package)).parent

# Same entry / label conventions as the ``Args:`` guard, so the two blocks are
# read the same way: ``name (type):`` / ``name:`` / ``a / b:``.
_ENTRY = re.compile(r"^\s*([A-Za-z_][\w ,/]*?)\s*(\([^)]*\))?\s*:")
_LABEL_SPLIT = re.compile(r"\s*(?:/|,|\bor\b)\s*")

# A block that follows ``Attributes:`` ends at the next section header.
_OTHER_SECTIONS = frozenset(
    {
        "Args:",
        "Arguments:",
        "Parameters:",
        "Returns:",
        "Raises:",
        "Yields:",
        "Example:",
        "Examples:",
        "Note:",
        "Notes:",
    }
)

# Fewer than this many graded dataclasses means the scan stopped reaching the
# package rather than that the package stopped documenting its fields.
_MINIMUM_GRADED_DATACLASSES = 15
_MINIMUM_GRADED_FIELDS = 120


def documented_attributes(doc: str) -> tuple[frozenset[str], bool]:
    """Names documented in ``doc``'s ``Attributes:`` section, and whether it has one.

    Args:
        doc: A dedented docstring (as :func:`ast.get_docstring` returns).

    Returns:
        ``(names, has_section)``. ``names`` is empty when there is no
        ``Attributes:`` section, so callers must consult ``has_section`` to tell
        "documents nothing" from "has no block to check". Combined labels are
        split on ``/``, ``,`` and ``or``.
    """
    lines = doc.splitlines()
    header_indent = None
    body: list[str] = []
    for index, line in enumerate(lines):
        if line.strip() == "Attributes:":
            header_indent = len(line) - len(line.lstrip())
            body = lines[index + 1 :]
            break
    if header_indent is None:
        return frozenset(), False

    names: set[str] = set()
    entry_indent = None
    for line in body:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= header_indent:
            break  # section ended (a dedented paragraph)
        if line.strip() in _OTHER_SECTIONS:
            break  # section ended (the next header, at the entry indent)
        if entry_indent is None:
            entry_indent = indent
        if indent != entry_indent:
            continue  # continuation line of the entry above
        match = _ENTRY.match(line)
        if not match:
            continue
        for part in _LABEL_SPLIT.split(match.group(1)):
            cleaned = part.strip()
            if cleaned:
                names.add(cleaned)
    return frozenset(names), True


def _is_dataclass(cls: ast.ClassDef) -> bool:
    """Whether ``cls`` carries a ``@dataclass`` decorator, bare or called."""
    for decorator in cls.decorator_list:
        node = decorator.func if isinstance(decorator, ast.Call) else decorator
        name = node.attr if isinstance(node, ast.Attribute) else getattr(node, "id", "")
        if name == "dataclass":
            return True
    return False


def declared_fields(cls: ast.ClassDef) -> list[str]:
    """Public fields ``cls`` declares in its own body, in declaration order.

    ``ClassVar`` annotations are class-level constants rather than fields, so
    they are excluded the way :func:`dataclasses.fields` excludes them.
    """
    names = []
    for node in cls.body:
        if not isinstance(node, ast.AnnAssign) or not isinstance(node.target, ast.Name):
            continue
        if node.target.id.startswith("_"):
            continue
        if "ClassVar" in ast.unparse(node.annotation):
            continue
        names.append(node.target.id)
    return names


def documented_dataclasses(root: Path) -> list[tuple[str, list[str], frozenset[str]]]:
    """Every dataclass under ``root`` whose docstring has an ``Attributes:`` block.

    Args:
        root: Package directory to walk. Every ``*.py`` beneath it is parsed by
            AST, so no optional backend needs to be importable.

    Returns:
        ``(surface_id, declared_fields, documented_names)`` triples, where
        ``surface_id`` is ``"<relative path>::<Class>"`` and ``documented_names``
        includes the names any base class documents, so a subclass may document
        only the fields it adds.
    """
    by_name: dict[str, frozenset[str]] = {}
    classes: list[tuple[str, ast.ClassDef]] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for cls in (n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)):
            doc = ast.get_docstring(cls)
            documented, has_section = documented_attributes(doc) if doc else (frozenset(), False)
            if has_section:
                by_name[cls.name] = documented | by_name.get(cls.name, frozenset())
            if _is_dataclass(cls) and has_section:
                classes.append((str(path.relative_to(root.parent)), cls))

    surfaces = []
    for rel, cls in classes:
        credited: set[str] = set(by_name.get(cls.name, frozenset()))
        for base in cls.bases:
            base_name = base.attr if isinstance(base, ast.Attribute) else getattr(base, "id", "")
            credited |= by_name.get(base_name, frozenset())
        surfaces.append((f"{rel}::{cls.name}", declared_fields(cls), frozenset(credited)))
    return surfaces


_DATACLASSES = documented_dataclasses(_PACKAGE_ROOT)


@pytest.mark.parametrize(
    ("surface", "fields", "documented"),
    _DATACLASSES,
    ids=[surface for surface, _, _ in _DATACLASSES],
)
def test_every_field_has_an_attributes_entry(surface: str, fields: list[str], documented: frozenset[str]) -> None:
    """A dataclass documenting its attributes documents every field it declares."""
    missing = [name for name in fields if name not in documented]
    assert not missing, (
        f"{surface} has an Attributes: block that omits {missing}. A caller "
        "building this config reads that block for the keys it may set, so an "
        "omitted field is one they do not know exists - and its default decides "
        "the behaviour instead. Document it, or document it on a base class."
    )


class TestTheScanIsNonVacuous:
    """A clean sweep has to mean the package is documented, not that nothing was read."""

    def test_the_scan_reaches_the_package(self) -> None:
        assert len(_DATACLASSES) >= _MINIMUM_GRADED_DATACLASSES, (
            f"only {len(_DATACLASSES)} dataclasses carry an Attributes: block; the scan "
            "is no longer reaching the package"
        )

    def test_the_scan_grades_real_fields(self) -> None:
        total = sum(len(fields) for _, fields, _ in _DATACLASSES)
        assert total >= _MINIMUM_GRADED_FIELDS, (
            f"only {total} declared fields were graded across {len(_DATACLASSES)} dataclasses"
        )

    def test_the_unit_conversion_group_is_graded(self) -> None:
        """The fields whose omission started this are in scope, by name."""
        embodiment = next(
            (entry for entry in _DATACLASSES if entry[0].endswith("::EmbodimentMap")),
            None,
        )
        assert embodiment is not None, "EmbodimentMap is no longer graded"
        _, fields, _ = embodiment
        for name in ("state_units", "action_units", "gripper_index", "gripper_joint_range", "joint_mids"):
            assert name in fields, f"EmbodimentMap.{name} is no longer a graded field"


class TestABaseClassEntryCounts:
    """A subclass may document only the fields it adds."""

    def test_an_inherited_field_documented_on_the_base_is_accepted(self) -> None:
        source = "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class Base:",
                '    """Base.',
                "",
                "    Attributes:",
                "        shared: Documented here.",
                '    """',
                "",
                "    shared: int = 0",
                "",
                "@dataclass",
                "class Child(Base):",
                '    """Child.',
                "",
                "    Attributes:",
                "        extra: Documented here.",
                '    """',
                "",
                "    shared: int = 1",
                "    extra: int = 2",
            ]
        )
        surfaces = self._scan(source)
        child = next(entry for entry in surfaces if entry[0].endswith("::Child"))
        _, fields, documented = child
        assert fields == ["shared", "extra"]
        assert set(fields) <= documented, "a base-documented field was not credited to the subclass"

    def test_a_field_documented_nowhere_is_still_reported(self) -> None:
        source = "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass",
                "class Solo:",
                '    """Solo.',
                "",
                "    Attributes:",
                "        documented: Yes.",
                '    """',
                "",
                "    documented: int = 0",
                "    forgotten: int = 1",
            ]
        )
        _, fields, documented = self._scan(source)[0]
        missing = [name for name in fields if name not in documented]
        assert missing == ["forgotten"]

    @staticmethod
    def _scan(source: str) -> list[tuple[str, list[str], frozenset[str]]]:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "pkg"
            root.mkdir()
            (root / "mod.py").write_text(source, encoding="utf-8")
            return documented_dataclasses(root)


class TestCombinedEntryLabelsCount:
    """One entry may name several fields that share a description."""

    @pytest.mark.parametrize(
        "label",
        ["lora_r / lora_alpha", "lora_r, lora_alpha", "lora_r or lora_alpha"],
    )
    def test_both_names_are_credited(self, label: str) -> None:
        doc = "\n".join(["Summary.", "", "Attributes:", f"    {label}: Shared description."])
        names, has_section = documented_attributes(doc)
        assert has_section
        assert {"lora_r", "lora_alpha"} <= names

    def test_the_shipped_combined_entry_is_read(self) -> None:
        """``TrainSpec`` documents three LoRA fields in one entry."""
        surface = next(entry for entry in _DATACLASSES if entry[0].endswith("::TrainSpec"))
        _, _, documented = surface
        assert {"lora_r", "lora_alpha", "lora_target_modules"} <= documented


class TestTheParserBoundsTheSection:
    """An ``Attributes:`` block ends where the next section begins."""

    def test_a_following_section_is_not_read_as_entries(self) -> None:
        doc = "\n".join(
            [
                "Summary.",
                "",
                "Attributes:",
                "    documented: Yes.",
                "",
                "Args:",
                "    not_a_field: A parameter.",
            ]
        )
        names, has_section = documented_attributes(doc)
        assert has_section
        assert names == frozenset({"documented"})

    def test_a_continuation_line_is_not_read_as_an_entry(self) -> None:
        doc = "\n".join(
            [
                "Summary.",
                "",
                "Attributes:",
                "    documented: A description that wraps and whose",
                "        continuation: contains a colon.",
            ]
        )
        names, _ = documented_attributes(doc)
        assert names == frozenset({"documented"})

    def test_a_docstring_with_no_block_is_out_of_scope(self) -> None:
        names, has_section = documented_attributes("Summary only.")
        assert not has_section
        assert names == frozenset()


class TestTheScannerDetectsAPlantedDefect:
    """A clean result must mean the fields are documented, not that nothing is checked."""

    def test_a_planted_undocumented_field_is_reported(self) -> None:
        source = "\n".join(
            [
                "from dataclasses import dataclass",
                "",
                "@dataclass(frozen=True)",
                "class Planted:",
                '    """Planted.',
                "",
                "    Attributes:",
                "        named: Documented.",
                '    """',
                "",
                "    named: int = 0",
                "    unnamed: int = 1",
            ]
        )
        surfaces = TestABaseClassEntryCounts._scan(source)
        assert len(surfaces) == 1
        _, fields, documented = surfaces[0]
        with pytest.raises(AssertionError, match="unnamed"):
            test_every_field_has_an_attributes_entry(surfaces[0][0], fields, documented)

    def test_a_class_that_is_not_a_dataclass_is_out_of_scope(self) -> None:
        """The field list of a plain class is not an annotation list."""
        source = "\n".join(
            [
                "class Plain:",
                '    """Plain.',
                "",
                "    Attributes:",
                "        documented: Yes.",
                '    """',
                "",
                "    documented: int = 0",
                "    undocumented: int = 1",
            ]
        )
        assert TestABaseClassEntryCounts._scan(source) == []

    def test_a_classvar_is_not_graded_as_a_field(self) -> None:
        source = "\n".join(
            [
                "from dataclasses import dataclass",
                "from typing import ClassVar",
                "",
                "@dataclass",
                "class WithConstant:",
                '    """WithConstant.',
                "",
                "    Attributes:",
                "        value: Documented.",
                '    """',
                "",
                "    LIMIT: ClassVar[int] = 5",
                "    value: int = 0",
            ]
        )
        _, fields, _ = TestABaseClassEntryCounts._scan(source)[0]
        assert fields == ["value"]
