"""The launch topology a TrainSpec asks for is one shared positive-count domain.

:attr:`~strands_robots.training.base.TrainSpec.num_gpus` and
:attr:`~strands_robots.training.base.TrainSpec.num_nodes` are the two process
counts a distributed run is sized from. Each is read in three places by the
three supervised backends: a ``spec.num_gpus > 1`` / ``spec.num_nodes > 1`` test
that selects between the single-process and the multi-process launch path, a
``nproc_per_node`` / ``nnodes`` argument to torch's ``elastic_launch``, and a
``--nproc_per_node=`` / ``--nnodes=`` / ``--num_gpus=`` argv token.

Before this contract neither field had a domain, while their run-size
neighbours ``steps`` and ``global_batch_size`` shared one - and every way a bad
process count failed was silent or late:

* ``0``, a negative, ``nan`` and ``True`` all compare as *not* greater than one
  (``nan`` compares false against everything), so the selector routed them to
  the single-process path and the run proceeded on one process under a
  successful result. For ``num_nodes`` that also slipped past the multi-node
  refusal LeRobot and GR00T raise for a topology they cannot run.
* ``2.7`` and ``inf`` *are* greater than one, so they selected the multi-process
  path and reached ``elastic_launch`` as the worker count.
  ``torch.distributed.launcher.api.LaunchConfig`` accepts both without
  complaint, so nothing downstream rejected them either.
* A string, ``None`` or a list raised ``TypeError`` out of the comparison
  itself - from inside a :meth:`Trainer.validate` documented to *return*
  problems.

The tests below pin the contract in both directions: every backend that reads
either field refuses the same unusable values through the one shared domain with
a message naming itself, a usable topology is untouched (including the
multi-node refusal, which must still fire for a *usable* count), a backend that
ignores the fields reports nothing about them, and the refused values are
grounded in what the comparison and the real launcher do with them.
"""

from __future__ import annotations

import ast
import inspect
import math
import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import launch_topology_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from tests.training._spec_field_reads import reads_spec_field

# The two fields, and the backends that launch from them.
TOPOLOGY_FIELDS = ("num_gpus", "num_nodes")
LAUNCHING_BACKENDS = (LerobotTrainer, Gr00tTrainer, Cosmos3Trainer)

# Values no launcher can honor, split by how each one failed before the gate.

# Read as "not more than one" by the selector -> a silent single-process run.
SILENT_SINGLE_PROCESS = (0, -4, True, False, float("nan"))

# Read as "more than one" -> reached elastic_launch as the worker count.
REACHED_THE_LAUNCHER = (2.7, float("inf"))

# Raised TypeError out of the comparison, from inside validate().
RAISED_OUT_OF_VALIDATE = ("4", None, [2])

UNUSABLE = SILENT_SINGLE_PROCESS + REACHED_THE_LAUNCHER + tuple(RAISED_OUT_OF_VALIDATE)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose topology fields are the only thing under test.

    ``validate`` may well report unrelated problems (GR00T wants a checkout,
    Cosmos a recipe TOML); every assertion below filters for the field name, so
    an unrelated problem cannot mask - or fake - a topology verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


def _problems_about(trainer: Trainer, spec: TrainSpec, field: str) -> list[str]:
    """``validate`` problems that name *field*."""
    return [p for p in trainer.validate(spec) if field in p]


class TestEveryLaunchingBackendRefusesAnUnusableProcessCount:
    """Each backend that launches from a field refuses every unusable value."""

    @pytest.mark.parametrize("trainer_cls", LAUNCHING_BACKENDS)
    @pytest.mark.parametrize("field", TOPOLOGY_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str, value: Any
    ) -> None:
        setattr(spec, field, value)
        problems = _problems_about(trainer_cls(), spec, field)
        assert problems, f"{trainer_cls.__name__} accepted {field}={value!r}"
        assert any("must be a positive integer" in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", LAUNCHING_BACKENDS)
    @pytest.mark.parametrize("field", TOPOLOGY_FIELDS)
    def test_the_problem_names_the_backend_that_refused_it(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str
    ) -> None:
        trainer = trainer_cls()
        setattr(spec, field, 0)
        assert any(p.startswith(f"{trainer.provider_name}: {field} ") for p in _problems_about(trainer, spec, field))

    @pytest.mark.parametrize("trainer_cls", LAUNCHING_BACKENDS)
    @pytest.mark.parametrize("field", TOPOLOGY_FIELDS)
    @pytest.mark.parametrize("value", RAISED_OUT_OF_VALIDATE)
    def test_a_non_numeric_count_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str, value: Any
    ) -> None:
        """``validate`` returns problems; it must not raise out of a comparison."""
        setattr(spec, field, value)
        problems = trainer_cls().validate(spec)  # must not raise
        assert any(field in p for p in problems), problems


class TestAUsableTopologyIsUntouched:
    """A usable process count raises no topology problem on any backend."""

    @pytest.mark.parametrize("trainer_cls", LAUNCHING_BACKENDS)
    @pytest.mark.parametrize("value", (1, 2, 8))
    def test_a_usable_gpu_count_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: int) -> None:
        spec.num_gpus = value
        assert _problems_about(trainer_cls(), spec, "num_gpus") == []

    @pytest.mark.parametrize("trainer_cls", LAUNCHING_BACKENDS)
    def test_the_single_node_default_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        spec.num_nodes = 1
        assert _problems_about(trainer_cls(), spec, "num_nodes") == []

    @pytest.mark.parametrize("trainer_cls", (LerobotTrainer, Gr00tTrainer))
    def test_the_multi_node_refusal_still_fires_for_a_usable_count(
        self, spec: TrainSpec, trainer_cls: type[Trainer]
    ) -> None:
        """The guarded comparison must still run once the count IS a count.

        LeRobot and GR00T cannot launch multi-node in-process and say so. That
        comparison is now reached only when the shared domain has established
        ``num_nodes`` is a count - so this pins that gating it did not disable
        it.
        """
        spec.num_nodes = 8
        problems = _problems_about(trainer_cls(), spec, "num_nodes")
        assert any("multi-node" in p for p in problems), problems
        assert not any("must be a positive integer" in p for p in problems), problems


class TestABackendThatIgnoresTheFieldsReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports
    and ignores the rest", so this gate is scoped to the launching backends
    rather than made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("field", TOPOLOGY_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_mock_backend_launches_from_neither_field(self, spec: TrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _problems_about(MockTrainer(), spec, field) == []

    @pytest.mark.parametrize("field", TOPOLOGY_FIELDS)
    def test_an_rl_backend_launches_from_neither_field(self, tmp_path: pathlib.Path, field: str) -> None:
        pytest.importorskip("torch")
        from strands_robots.training.rl.base_algo import RLTrainSpec
        from strands_robots.training.rl.fast_sac import FastSacTrainer

        rl_spec = RLTrainSpec(
            output_dir=str(tmp_path),
            env_factory=lambda: None,  # type: ignore[arg-type,return-value]
        )
        setattr(rl_spec, field, 0)
        assert [p for p in FastSacTrainer().validate(rl_spec) if field in p] == []


class TestTheRefusedValuesAreOnesTheLauncherCannotHonor:
    """Ground the domain in what the selector and the real launcher do."""

    @pytest.mark.parametrize("value", SILENT_SINGLE_PROCESS)
    def test_a_silent_value_reads_as_not_more_than_one(self, value: Any) -> None:
        """The ``> 1`` selector routes each of these to the single-process path."""
        assert not value > 1

    def test_nan_compares_false_against_every_bound(self) -> None:
        """Why ``nan`` slipped through a comparison-based guard in both directions."""
        nan = float("nan")
        assert not nan > 1
        assert not nan <= 0
        assert math.isnan(nan)

    @pytest.mark.parametrize("value", REACHED_THE_LAUNCHER)
    def test_a_launcher_bound_value_reads_as_more_than_one(self, value: Any) -> None:
        assert value > 1

    @pytest.mark.parametrize("value", SILENT_SINGLE_PROCESS + REACHED_THE_LAUNCHER)
    def test_the_torch_launcher_does_not_reject_it_either(self, value: Any) -> None:
        """Nothing downstream catches an unusable worker count.

        ``LaunchConfig`` is the value's first destination once the selector
        chooses the multi-process path, and it accepts every one of these - so
        the preflight is the only place the caller can be told.
        """
        pytest.importorskip("torch")
        from torch.distributed.launcher.api import LaunchConfig

        config = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=value)
        assert config.nproc_per_node == value or (
            isinstance(value, float) and math.isnan(value) and math.isnan(config.nproc_per_node)
        )

    @pytest.mark.parametrize("value", RAISED_OUT_OF_VALIDATE)
    def test_a_non_numeric_count_cannot_be_compared_at_all(self, value: Any) -> None:
        """Which is why the old comparison raised rather than reporting."""
        with pytest.raises(TypeError):
            _ = value > 1  # type: ignore[operator]


def _trainer_modules() -> list[pathlib.Path]:
    """Every backend module, INCLUDING the ``rl`` subpackage.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The module that *defines* the shared gate
    is excluded - derived from the gate itself rather than named, so the
    exclusion cannot drift - because it reads both fields as their owner rather
    than as a consumer of them.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(launch_topology_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_a_topology_field(source: str) -> bool:
    """Does *source* read either field, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, TOPOLOGY_FIELDS)


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_launch_topology_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheLaunchTopologyDomain:
    """No backend may re-implement the domain, and none may skip it.

    The set of backends in scope is derived from the tree rather than listed:
    a module that *reads* either field must route it through the shared gate, so
    a fifth backend that starts launching from ``num_gpus`` fails this test
    until it does.
    """

    def test_the_scan_finds_the_launching_backends(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if _reads_a_topology_field(p.read_text())}
        assert readers == {"cosmos3.py", "groot.py", "lerobot.py", "sagemaker.py"}

    def test_every_backend_that_launches_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if _reads_a_topology_field(source := p.read_text()) and not _calls_the_gate(source)
        )
        assert adrift == [], f"modules reading a topology field without the shared gate: {adrift}"

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local ``<= 0`` / ``< 1`` test on either field is the hole this closed.

        A ``> 1`` comparison is a different question - "is this topology one I
        can launch" - and stays where it is.
        """
        offenders: list[str] = []
        for path in _trainer_modules():
            for line in path.read_text().splitlines():
                if any(f"spec.{field}" in line for field in TOPOLOGY_FIELDS) and (
                    "<= 0" in line or "< 1" in line or "!= int" in line
                ):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on a topology field: {offenders}"

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [] if spec.num_gpus > 1 else []\n"
        assert _reads_a_topology_field(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards either field by name is a reader too.

        The form a transport-only provider takes: no attribute access mentions
        either field, so a scan keyed on ``spec.num_gpus`` alone reports a clean
        sweep while this backend skips the gate.
        """
        planted = 'F = ("num_gpus",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in F]\n'
        assert _reads_a_topology_field(planted)
        assert not _calls_the_gate(planted)


class TestTheGateIsUsableOnItsOwn:
    """The shared gate's own contract, independent of any backend."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.num_gpus = 0
        assert launch_topology_problems(spec, context="acme") == ["acme: num_gpus must be a positive integer, got 0."]

    def test_a_usable_topology_reports_nothing(self, spec: TrainSpec) -> None:
        spec.num_gpus, spec.num_nodes = 8, 2
        assert launch_topology_problems(spec, context="acme") == []

    def test_both_fields_are_reported_together(self, spec: TrainSpec) -> None:
        spec.num_gpus, spec.num_nodes = 0, "4"  # type: ignore[assignment]
        problems = launch_topology_problems(spec, context="acme")
        assert len(problems) == 2
        assert "num_gpus" in problems[0] and "num_nodes" in problems[1]
