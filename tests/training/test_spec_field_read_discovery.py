"""Every field-scoped shared-domain guard sees a table-driven read of its field.

Each shared numeric domain on :class:`~strands_robots.training.base.Trainer`
documents a biconditional - a backend that reads the field MUST route it
through the gate, one that ignores it MUST NOT report on it - and the guard for
each domain pins the first half with a scope *derived from the tree*, so that
"a new backend that starts reading the field fails this test until it does".

That promise rests entirely on the guard's notion of "reads the field". A
backend can read a spec field two ways: by name (``spec.seed``) or through a
forwarding table (``getattr(spec, field)`` over a tuple of field names, which
is how a transport-only provider serializes every field it passes on). A scan
keyed on the first form alone certifies a complete sweep while a table-driven
reader sits outside it, and the biconditional is then unenforced for exactly
that backend - silently, because the guard reports a clean tree.

This grades the guards from the outside rather than trusting each to grade
itself: the set of field-scoped guards is discovered structurally, so a new
domain guard is held to the same rule the moment it lands.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots.training.base import Trainer
from strands_robots.training.sagemaker import _FORWARDED_FIELDS
from tests.training._spec_field_reads import reads_spec_field

# The gates whose scope is a field rather than every backend, mapped to the
# TrainSpec fields each owns. The learning-rate gate is deliberately absent: no
# backend may skip it, so its guard scans Trainer subclasses rather than field
# reads and needs no notion of "reads the field" at all.
FIELD_SCOPED_GATES: dict[str, tuple[str, ...]] = {
    "_seed_problems": ("seed",),
    "_validation_episodes_problems": ("val_episodes",),
    "_lora_hyperparameter_problems": ("lora_r", "lora_alpha"),
    "_launch_topology_problems": ("num_gpus", "num_nodes"),
}


def _guard_modules() -> dict[str, Any]:
    """The field-scoped domain guards, discovered by structure not by name.

    A guard qualifies when it scans the backend tree (``_trainer_modules``),
    tests membership of one gate (``_calls_the_gate``) and derives its scope
    from a reader scan (a ``_reads...`` helper). Discovering them rather than
    listing them means a new domain guard is held to this rule on arrival.
    """
    here = pathlib.Path(__file__).parent
    guards: dict[str, Any] = {}
    for path in sorted(here.glob("test_*_domain.py")):
        source = path.read_text()
        names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        readers = sorted(n for n in names if n.startswith("_reads"))
        if not readers or "_trainer_modules" not in names or "_calls_the_gate" not in names:
            continue
        guards[path.name] = importlib.import_module(f"tests.training.{path.stem}")
    return guards


def _reader_helper(module: Any) -> Any:
    """The single ``_reads...`` helper a field-scoped guard derives its scope from."""
    helpers = [
        getattr(module, name) for name in dir(module) if name.startswith("_reads") and callable(getattr(module, name))
    ]
    assert len(helpers) == 1, f"{module.__name__} has {len(helpers)} reader helpers"
    return helpers[0]


def _table_driven_reader(field: str) -> str:
    """A backend that reads *field* only through a forwarding table."""
    return f'FIELDS = ("{field}",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'


class TestEveryFieldScopedGuardSeesBothFormsOfARead:
    """The headline: a reader scan must recognize a table-driven read."""

    def test_the_scan_finds_the_field_scoped_guards(self) -> None:
        """Non-vacuity: a scan that matched nothing would grade nothing."""
        assert set(_guard_modules()) == {
            "test_launch_topology_domain.py",
            "test_lora_hyperparameter_domain.py",
            "test_seed_domain.py",
            "test_validation_episodes_domain.py",
        }

    @pytest.mark.parametrize("guard_name", sorted(_guard_modules()))
    def test_it_sees_a_table_driven_read(self, guard_name: str) -> None:
        module = _guard_modules()[guard_name]
        reads = _reader_helper(module)
        gate = next(g for g in FIELD_SCOPED_GATES if any(g in line for line in inspect.getsource(module).splitlines()))
        for field in FIELD_SCOPED_GATES[gate]:
            assert reads(_table_driven_reader(field)), (
                f"{guard_name} does not see a table-driven read of spec.{field}, "
                "so a backend that forwards the field by name is outside its derived scope"
            )

    @pytest.mark.parametrize("guard_name", sorted(_guard_modules()))
    def test_it_still_sees_a_read_by_name(self, guard_name: str) -> None:
        """The form it already recognized must keep being recognized."""
        module = _guard_modules()[guard_name]
        reads = _reader_helper(module)
        gate = next(g for g in FIELD_SCOPED_GATES if any(g in line for line in inspect.getsource(module).splitlines()))
        for field in FIELD_SCOPED_GATES[gate]:
            assert reads(f"def validate(self, spec):\n    return [spec.{field}]\n")


class TestTheForwardingProviderIsInScopeForEveryGateItReads:
    """The reader the literal-only scans could not see, on the real tree."""

    @pytest.mark.parametrize(("gate", "fields"), sorted(FIELD_SCOPED_GATES.items()))
    def test_it_is_discovered_as_a_reader(self, gate: str, fields: tuple[str, ...]) -> None:
        source = pathlib.Path(inspect.getfile(Trainer)).parent.joinpath("sagemaker.py").read_text()
        forwarded = [f for f in fields if f in _FORWARDED_FIELDS]
        assert forwarded, f"{gate}'s fields are no longer forwarded: {fields}"
        assert reads_spec_field(source, forwarded)

    @pytest.mark.parametrize(("gate", "fields"), sorted(FIELD_SCOPED_GATES.items()))
    def test_it_routes_that_read_through_the_shared_gate(self, gate: str, fields: tuple[str, ...]) -> None:
        """Being in scope is only useful if the gate is then enforced on it."""
        source = pathlib.Path(inspect.getfile(Trainer)).parent.joinpath("sagemaker.py").read_text()
        calls = {
            node.func.attr
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert gate in calls, f"sagemaker.py forwards {fields} without calling {gate}"


class TestTheSharedRuleIsPrecise:
    """Both halves of the table form are required, so the rule cannot over-reach."""

    def test_a_field_name_in_a_string_alone_is_not_a_read(self) -> None:
        """A message or a docstring naming the field reads nothing."""
        source = 'def validate(self, spec):\n    return ["seed must be positive"]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_a_getattr_on_spec_for_other_fields_is_not_a_read(self) -> None:
        """Forwarding a table that does not contain the field reads nothing."""
        source = 'FIELDS = ("steps",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_a_getattr_on_something_else_is_not_a_read(self) -> None:
        source = 'def validate(self, spec):\n    return [getattr(self, "seed")]\n'
        assert not reads_spec_field(source, ("seed",))

    def test_an_unrelated_module_reads_nothing(self) -> None:
        assert not reads_spec_field("x = 1\n", ("seed",))
