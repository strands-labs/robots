"""The run size a spec asks for is one shared positive-count domain.

:attr:`~strands_robots.training.base.TrainSpec.steps` and
:attr:`~strands_robots.training.base.TrainSpec.global_batch_size` are the two
factors of how much training a spec asks for, and every supervised backend
reads both straight into a discrete consumer: ``steps`` bounds the optimizer
loop (lerobot iterates ``range(step, cfg.steps)``) and ``global_batch_size``
becomes a ``DataLoader`` batch size / a ``--global_batch_size`` flag.

Before this contract each backend re-implemented ``if spec.steps <= 0`` and
none checked ``global_batch_size`` at all, so the four copies agreed with each
other and shared one hole: a comparison admits every value that is not
comparably non-positive. ``True`` became a silent run of one optimizer step,
a fractional or non-finite value reached ``range()`` and raised there - after
the dataset and the model were already loaded - and a non-numeric value raised
out of the comparison itself, from a :meth:`Trainer.validate` documented to
*return* problems rather than raise.

The tests below pin the contract in both directions: every backend that reads
the run size refuses the same unusable values through the one shared domain,
a usable run size is untouched, and a backend that drives training from other
fields (the RL trainers, on ``total_timesteps`` / ``batch_size``) reports
nothing for a field it never reads.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import run_size_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.sagemaker import SagemakerTrainer

# The two factors of the run size, and the values no backend can honor. Split
# by failure mode so a test can say which contract each value belongs to.
RUN_SIZE_FIELDS = ("steps", "global_batch_size")

# Non-positive: already refused before this change (by four separate copies of
# the same comparison). Kept so the fix cannot regress what worked.
NON_POSITIVE = (0, -5)

# Slipped through a comparison, then misbehaved in the consumer.
WRONG_TYPE = (True, False, 2.7, float("nan"), float("inf"))

# Raised out of the comparison itself, i.e. out of validate().
NOT_COMPARABLE = ("1000", None, [4])

UNUSABLE = NON_POSITIVE + WRONG_TYPE + NOT_COMPARABLE

# Every backend that reads the run size. The RL trainers are deliberately
# absent - see TestTheRLTrainersIgnoreAFieldTheyDoNotRead.
SUPERVISED_TRAINERS = (MockTrainer, Cosmos3Trainer, Gr00tTrainer, LerobotTrainer, SagemakerTrainer)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose non-numeric fields are all fine, so only the run size varies."""
    root = tmp_path / "ds"
    (root / "meta").mkdir(parents=True)
    (root / "meta" / "info.json").write_text(json.dumps({"total_episodes": 10, "fps": 30}))
    out = tmp_path / "out"
    out.mkdir()
    return TrainSpec(
        dataset_root=str(root),
        base_model="lerobot/act",
        output_dir=str(out),
        embodiment="so101",
    )


def _mutate(spec: TrainSpec, field: str, value: Any) -> TrainSpec:
    """Set one run-size field to a value its annotation does not describe."""
    setattr(spec, field, value)
    return spec


class TestEveryBackendRefusesAnUnusableRunSize:
    """One domain, five backends, one verdict per value."""

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_unusable_run_size_is_reported(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str, value: Any
    ) -> None:
        trainer = trainer_cls()
        problems = trainer.validate(_mutate(spec, field, value))
        named = [p for p in problems if field in p]
        assert named, f"{trainer_cls.__name__} accepted {field}={value!r}: {problems}"
        assert "must be a positive integer" in named[0]
        assert repr(value) in named[0], named[0]

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """A shared domain must still say which backend rejected the spec."""
        trainer = trainer_cls()
        problems = trainer.validate(_mutate(spec, "steps", 0))
        named = [p for p in problems if "steps" in p]
        assert named and named[0].startswith(f"{trainer.provider_name}: "), named


class TestValidateReportsInsteadOfRaising:
    """``validate`` returns problems - a non-numeric run size must not raise."""

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    @pytest.mark.parametrize("value", NOT_COMPARABLE)
    def test_a_non_comparable_value_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str, value: Any
    ) -> None:
        problems = trainer_cls().validate(_mutate(spec, field, value))
        assert isinstance(problems, list)
        assert any(field in p for p in problems), problems


class TestAUsableRunSizeIsUntouched:
    """The change is additive: every honorable run size still validates."""

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize(("steps", "batch"), [(10000, 32), (1, 1), (2, 4096)])
    def test_a_usable_run_size_raises_no_run_size_problem(
        self, spec: TrainSpec, trainer_cls: type[Trainer], steps: int, batch: int
    ) -> None:
        spec.steps = steps
        spec.global_batch_size = batch
        problems = trainer_cls().validate(spec)
        assert not [p for p in problems if "positive integer" in p], problems


class TestTheRLTrainersIgnoreAFieldTheyDoNotRead:
    """A backend must not report on a spec field it never consumes.

    :class:`TrainSpec` documents that a backend "reads the fields it supports
    and ignores the rest". The RL trainers drive training from
    ``total_timesteps`` / ``batch_size`` and never read ``steps`` or
    ``global_batch_size``, so the run-size gate must stay off their path -
    otherwise an RL spec carrying the inherited defaults would be refused for
    a reason that has no effect on the run.
    """

    @pytest.mark.parametrize("field", RUN_SIZE_FIELDS)
    def test_an_rl_spec_is_not_refused_for_an_unread_field(self, field: str) -> None:
        pytest.importorskip("torch")
        from strands_robots.training.rl.base_algo import RLTrainSpec
        from strands_robots.training.rl.ppo import PpoTrainer

        rl_spec = RLTrainSpec(output_dir="/tmp/rl-out", env_factory=lambda: None)  # type: ignore[arg-type,return-value]
        setattr(rl_spec, field, 0)
        problems = PpoTrainer().validate(rl_spec)
        assert not [p for p in problems if field in p], problems


class TestTheRefusedValuesAreOnesTheConsumerCannotHonor:
    """Ground the domain in what the backends actually do with the values."""

    def test_a_fractional_or_non_finite_step_count_cannot_bound_a_loop(self) -> None:
        for value in (2.7, float("nan"), float("inf")):
            # Bound through an ``Any`` local: the point is what the *runtime*
            # does with the value a spec carried, which is what lerobot's train
            # loop does with ``cfg.steps``.
            bound: Any = value
            with pytest.raises(TypeError):
                range(0, bound)

    def test_a_boolean_step_count_is_a_silent_run_of_one_step(self) -> None:
        assert len(range(0, True)) == 1

    def test_an_unusable_batch_size_cannot_build_a_dataloader(self) -> None:
        torch = pytest.importorskip("torch")
        from torch.utils.data import DataLoader, TensorDataset

        dataset = TensorDataset(torch.zeros(4, 2))
        for value in (0, -8, True, 2.7, float("nan"), "32"):
            size: Any = value
            with pytest.raises(ValueError):
                DataLoader(dataset, batch_size=size)


def _training_modules() -> list[pathlib.Path]:
    """Top-level training backend modules (the ``rl`` subpackage is excluded).

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    return sorted(p for p in root.glob("*.py") if p.name != "__init__.py")


def _concrete_trainer_validators(source: str) -> dict[str, ast.FunctionDef]:
    """Map ``ClassName -> its validate() node`` for each ``Trainer`` subclass."""
    found: dict[str, ast.FunctionDef] = {}
    for node in ast.parse(source).body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(isinstance(b, ast.Name) and b.id == "Trainer" for b in node.bases):
            continue
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "validate":
                found[node.name] = item
    return found


def _calls_run_size_gate(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_run_size_problems"
        for n in ast.walk(fn)
    )


def _local_run_size_comparisons(source: str) -> list[str]:
    """Comparisons of a run-size field against a numeric literal.

    That is the shape of a re-implemented domain (``if spec.steps <= 0``). A
    comparison against another *field* is a different question and is allowed.
    """
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr in RUN_SIZE_FIELDS):
            continue
        if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) for c in node.comparators):
            hits.append(ast.unparse(node))
    return hits


class TestOneOwnerForTheRunSizeDomain:
    """A fifth backend cannot ship with its own copy of the rule."""

    def test_every_supervised_backend_routes_through_the_shared_gate(self) -> None:
        adrift: dict[str, list[str]] = {}
        seen: set[str] = set()
        for path in _training_modules():
            for cls_name, fn in _concrete_trainer_validators(path.read_text()).items():
                seen.add(cls_name)
                if not _calls_run_size_gate(fn):
                    adrift.setdefault(path.name, []).append(cls_name)
        assert not adrift, f"validate() does not call _run_size_problems: {adrift}"
        # Non-vacuity: the scan really reached every backend, not an empty tree.
        assert seen == {t.__name__ for t in SUPERVISED_TRAINERS}, seen

    def test_no_backend_re_implements_the_domain(self) -> None:
        copies = {p.name: hits for p in _training_modules() if (hits := _local_run_size_comparisons(p.read_text()))}
        assert not copies, f"run size compared against a literal instead of the shared gate: {copies}"

    def test_the_scanners_detect_a_planted_copy(self) -> None:
        """An empty result must mean clean sources, not a scanner matching nothing."""
        planted = (
            "class RogueTrainer(Trainer):\n"
            "    def validate(self, spec):\n"
            "        problems = []\n"
            "        if spec.steps <= 0:\n"
            "            problems.append('bad')\n"
            "        return problems\n"
        )
        validators = _concrete_trainer_validators(planted)
        assert set(validators) == {"RogueTrainer"}
        assert not _calls_run_size_gate(validators["RogueTrainer"])
        assert _local_run_size_comparisons(planted) == ["spec.steps <= 0"]


class TestTheGateIsUsableOnItsOwn:
    """The shared gate is a plain function - the trainers only bind a context."""

    def test_it_reports_both_factors_at_once(self, spec: TrainSpec) -> None:
        spec.steps = 0
        spec.global_batch_size = -1
        problems = run_size_problems(spec, context="probe")
        assert len(problems) == 2
        assert all(p.startswith("probe: ") for p in problems)

    def test_a_usable_spec_reports_nothing(self, spec: TrainSpec) -> None:
        assert run_size_problems(spec, context="probe") == []
