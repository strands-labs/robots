"""The optimizer learning rate is one shared positive-finite domain.

:attr:`~strands_robots.training.base.TrainSpec.learning_rate` is the one
numeric on a spec that decides whether a run *learns* rather than how much work
it does, and unlike the run-size fields **every** backend reads it: the three
supervised trainers assign it to their config's optimizer field (LeRobot
``policy.optimizer_lr``, GR00T ``FinetuneConfig.learning_rate``, Cosmos
``optimizer.lr``) and the RL trainers pass it straight to
``torch.optim.Adam(..., lr=...)``. ``rl/base_algo.py`` calls it one of the
"universal" spec fields.

Before this contract no backend checked it, while each bounded its neighbours:
``FastSacTrainer.validate`` compares eight sibling numerics against literals
(``total_timesteps``, ``rollout_steps``, ``num_envs``, ``buffer_size``,
``batch_size``, ``gradient_steps``, ``tau``, ``learning_starts``) and skipped
the one that decides whether any of that work updates a weight, and the
supervised backends gate the two run-size factors and skipped it too.

The two ends of the domain fail *silently*, which is why this is a preflight
rather than something a backend can be left to notice:

* ``0`` (and ``False``) runs the full ``steps`` x ``global_batch_size`` of work
  and updates no weight - the checkpoint equals the initialisation and the run
  reports success. That is the pathology the run-size gate exists to prevent,
  reached by a different route and at full cost.
* ``inf`` diverges on the first step, so the checkpoint is all ``NaN``, again
  under a successful result.
* ``True`` is a silent learning rate of ``1.0``.

A negative or ``nan`` value *is* refused - by ``torch.optim.Adam`` itself - but
only once the dataset and model are loaded, past the point
:meth:`Trainer.validate` documents itself as running before.

The tests below pin the contract in both directions: every backend refuses the
same unusable values through the one shared domain with a message naming itself,
a usable rate and the documented ``None`` sentinel are untouched, and the
refused values are grounded in what a real optimizer does with them.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import learning_rate_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.training.sagemaker import SagemakerTrainer

# Values no backend can honor, split by how each one failed before the gate.

# Ran the whole schedule and updated nothing, under a successful result.
SILENT_NO_OP = (0, 0.0, False)

# Completed and wrote a checkpoint of NaN, also under a successful result.
SILENT_DIVERGENCE = (float("inf"),)

# Honored as 1.0 - four orders of magnitude above a fine-tuning preset.
SILENT_MISREAD = (True,)

# Refused by the optimizer, but only after the dataset and model were loaded.
LOUD_BUT_LATE = (-1e-3, float("nan"))

# Never reached an optimizer at all: a config field of the wrong type.
NOT_A_NUMBER = ("1e-4", [1e-4], {"lr": 1e-4})

UNUSABLE = SILENT_NO_OP + SILENT_DIVERGENCE + SILENT_MISREAD + LOUD_BUT_LATE + NOT_A_NUMBER

SUPERVISED_TRAINERS = (MockTrainer, Cosmos3Trainer, Gr00tTrainer, LerobotTrainer, SagemakerTrainer)
RL_TRAINER_NAMES = ("FastSacTrainer", "PpoTrainer")
ALL_TRAINER_NAMES = tuple(t.__name__ for t in SUPERVISED_TRAINERS) + RL_TRAINER_NAMES


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose other fields are all fine, so only the rate varies."""
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


def _rl_trainers() -> list[Trainer]:
    """The two RL trainers, which read the rate straight into ``Adam``."""
    pytest.importorskip("torch")
    from strands_robots.training.rl.fast_sac import FastSacTrainer
    from strands_robots.training.rl.ppo import PpoTrainer

    return [FastSacTrainer(), PpoTrainer()]


def _rl_spec(tmp_path: pathlib.Path, rate: Any) -> Any:
    """An otherwise-valid RL spec carrying ``rate``."""
    pytest.importorskip("torch")
    from strands_robots.training.rl.base_algo import RLTrainSpec

    spec = RLTrainSpec(
        output_dir=str(tmp_path),
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )
    spec.learning_rate = rate
    return spec


class TestEveryBackendRefusesAnUnusableLearningRate:
    """One domain, seven backends, one verdict per value."""

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_a_supervised_backend_reports_it(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.learning_rate = value
        problems = trainer_cls().validate(spec)
        named = [p for p in problems if "learning_rate" in p]
        assert named, f"{trainer_cls.__name__} accepted learning_rate={value!r}: {problems}"
        assert "must be > 0" in named[0], named[0]
        assert repr(value) in named[0], named[0]

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_an_rl_backend_reports_it(self, tmp_path: pathlib.Path, value: Any) -> None:
        for trainer in _rl_trainers():
            problems = trainer.validate(_rl_spec(tmp_path, value))
            named = [p for p in problems if "learning_rate" in p]
            assert named, f"{trainer.provider_name} accepted learning_rate={value!r}: {problems}"
            assert "must be > 0" in named[0], named[0]

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """A shared domain must still say which backend rejected the spec."""
        trainer = trainer_cls()
        spec.learning_rate = 0
        named = [p for p in trainer.validate(spec) if "learning_rate" in p]
        assert named and named[0].startswith(f"{trainer.provider_name}: "), named

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize("value", NOT_A_NUMBER)
    def test_a_non_numeric_rate_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        """``validate`` returns problems - it must not raise on a bad type."""
        spec.learning_rate = value
        problems = trainer_cls().validate(spec)
        assert isinstance(problems, list)
        assert any("learning_rate" in p for p in problems), problems


class TestAUsableLearningRateIsUntouched:
    """The change is additive: nothing honorable becomes a problem."""

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    @pytest.mark.parametrize("value", [1e-5, 1e-4, 3e-4, 0.1, 1.0])
    def test_a_usable_rate_raises_no_learning_rate_problem(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: float
    ) -> None:
        spec.learning_rate = value
        assert not [p for p in trainer_cls().validate(spec) if "learning_rate" in p]

    @pytest.mark.parametrize("trainer_cls", SUPERVISED_TRAINERS)
    def test_the_none_sentinel_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """``None`` is the documented "use the backend's own default" spelling."""
        spec.learning_rate = None
        assert not [p for p in trainer_cls().validate(spec) if "learning_rate" in p]

    def test_an_rl_spec_at_its_own_default_is_not_a_problem(self, tmp_path: pathlib.Path) -> None:
        """``RLTrainSpec`` pins a concrete rate instead of the sentinel."""
        pytest.importorskip("torch")
        from strands_robots.training.rl.base_algo import RLTrainSpec

        spec = RLTrainSpec(
            output_dir=str(tmp_path),
            env_factory=lambda: None,  # type: ignore[arg-type,return-value]
        )
        assert spec.learning_rate > 0
        for trainer in _rl_trainers():
            assert not [p for p in trainer.validate(spec) if "learning_rate" in p]


class TestTheRefusedValuesAreOnesTheConsumerCannotHonor:
    """Ground the domain in what a real optimizer does with each value."""

    @staticmethod
    def _run(rate: Any, steps: int = 20) -> tuple[float, bool]:
        """Train a 1-layer model for ``steps`` - returns (max weight delta, any NaN)."""
        import torch

        torch.manual_seed(0)
        model = torch.nn.Linear(4, 1)
        before = model.weight.detach().clone()
        optimizer = torch.optim.Adam(model.parameters(), lr=rate)
        inputs, targets = torch.randn(8, 4), torch.randn(8, 1)
        for _ in range(steps):
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(model(inputs), targets).backward()
            optimizer.step()
        after = model.weight.detach()
        return float((after - before).abs().max()), bool(torch.isnan(after).any())

    @pytest.mark.parametrize("value", SILENT_NO_OP)
    def test_a_zero_rate_completes_the_run_and_updates_nothing(self, value: Any) -> None:
        pytest.importorskip("torch")
        delta, has_nan = self._run(value)
        assert delta == 0.0, f"lr={value!r} moved the weights by {delta}"
        assert not has_nan

    def test_a_usable_rate_does_move_the_weights(self) -> None:
        """Non-vacuity: the probe can tell learning from no learning."""
        pytest.importorskip("torch")
        delta, has_nan = self._run(1e-3)
        assert delta > 0.0
        assert not has_nan

    @pytest.mark.parametrize("value", SILENT_DIVERGENCE)
    def test_an_infinite_rate_completes_the_run_with_nan_weights(self, value: Any) -> None:
        pytest.importorskip("torch")
        _, has_nan = self._run(value)
        assert has_nan, f"lr={value!r} was expected to diverge"

    @pytest.mark.parametrize("value", SILENT_MISREAD)
    def test_a_boolean_rate_is_read_as_one(self, value: Any) -> None:
        pytest.importorskip("torch")
        assert self._run(value)[0] == pytest.approx(self._run(1.0)[0])

    @pytest.mark.parametrize("value", LOUD_BUT_LATE)
    def test_a_negative_or_nan_rate_is_refused_by_the_optimizer(self, value: Any) -> None:
        """Refused, but only once a model exists to build an optimizer over."""
        torch = pytest.importorskip("torch")
        with pytest.raises(ValueError, match="[Ll]earning rate"):
            torch.optim.Adam(torch.nn.Linear(2, 1).parameters(), lr=value)

    def test_the_backend_config_layer_does_not_catch_it_either(self) -> None:
        """lerobot's own optimizer config accepts every unusable rate."""
        pytest.importorskip("lerobot")
        from lerobot.optim.optimizers import AdamWConfig

        for value in (0.0, -1e-3, float("nan"), float("inf")):
            assert AdamWConfig(lr=value).lr == value or value != value


def _trainer_sources() -> dict[str, str]:
    """``filename -> source`` for every backend module in the tree."""
    return {p.name: p.read_text() for p in _trainer_modules()}


def _trainer_modules() -> list[pathlib.Path]:
    """Every backend module, INCLUDING the ``rl`` subpackage.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The rl subpackage is in scope here
    (unlike for the run-size gate) precisely because there is no backend that
    may skip this rule.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py")


def _trainer_class_defs(sources: dict[str, str]) -> dict[str, ast.ClassDef]:
    """Every class in the backend tree that descends from :class:`Trainer`.

    The base chain is followed to a fixpoint rather than matched against one
    name: the RL trainers reach :class:`Trainer` through ``BaseRLAlgo``, so a
    scanner keyed on ``class X(Trainer)`` alone would silently skip the two
    backends that hand the rate straight to ``torch.optim.Adam``.
    """
    classes: dict[str, ast.ClassDef] = {}
    for source in sources.values():
        for node in ast.parse(source).body:
            if isinstance(node, ast.ClassDef):
                classes[node.name] = node

    descendants = {"Trainer"}
    grew = True
    while grew:
        grew = False
        for name, node in classes.items():
            if name in descendants:
                continue
            if any(isinstance(b, ast.Name) and b.id in descendants for b in node.bases):
                descendants.add(name)
                grew = True
    return {n: c for n, c in classes.items() if n in descendants - {"Trainer"}}


def _concrete_trainer_validators(sources: dict[str, str]) -> dict[str, ast.FunctionDef]:
    """Map ``ClassName -> its validate() node`` for each concrete backend.

    A class that does not define ``validate`` of its own (the abstract
    ``BaseRLAlgo``) has no gate to call and is not a backend.
    """
    found: dict[str, ast.FunctionDef] = {}
    for name, node in _trainer_class_defs(sources).items():
        for item in node.body:
            if isinstance(item, ast.FunctionDef) and item.name == "validate":
                found[name] = item
    return found


def _calls_learning_rate_gate(fn: ast.FunctionDef) -> bool:
    return any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "_learning_rate_problems"
        for n in ast.walk(fn)
    )


def _local_learning_rate_comparisons(source: str) -> list[str]:
    """Comparisons of ``learning_rate`` against a numeric literal.

    That is the shape of a re-implemented domain (``if spec.learning_rate <= 0``).
    """
    hits: list[str] = []
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.Compare):
            continue
        left = node.left
        if not (isinstance(left, ast.Attribute) and left.attr == "learning_rate"):
            continue
        if any(isinstance(c, ast.Constant) and isinstance(c.value, (int, float)) for c in node.comparators):
            hits.append(ast.unparse(node))
    return hits


class TestOneOwnerForTheLearningRateDomain:
    """A seventh backend cannot ship without the rule."""

    def test_every_backend_routes_through_the_shared_gate(self) -> None:
        validators = _concrete_trainer_validators(_trainer_sources())
        adrift = sorted(n for n, fn in validators.items() if not _calls_learning_rate_gate(fn))
        assert not adrift, f"validate() does not call _learning_rate_problems: {adrift}"
        # Non-vacuity: the scan really reached every backend, not an empty tree.
        assert set(validators) == set(ALL_TRAINER_NAMES), sorted(validators)

    def test_no_backend_re_implements_the_domain(self) -> None:
        copies = {p.name: hits for p in _trainer_modules() if (hits := _local_learning_rate_comparisons(p.read_text()))}
        assert not copies, f"learning_rate compared against a literal instead of the shared gate: {copies}"

    def test_the_scanners_detect_a_planted_copy(self) -> None:
        """An empty result must mean clean sources, not a scanner matching nothing."""
        planted = (
            "class RogueTrainer(Trainer):\n"
            "    def validate(self, spec):\n"
            "        problems = []\n"
            "        if spec.learning_rate <= 0:\n"
            "            problems.append('bad')\n"
            "        return problems\n"
        )
        validators = _concrete_trainer_validators({"planted.py": planted})
        assert set(validators) == {"RogueTrainer"}
        assert not _calls_learning_rate_gate(validators["RogueTrainer"])
        assert _local_learning_rate_comparisons(planted) == ["spec.learning_rate <= 0"]


class TestTheGateIsUsableOnItsOwn:
    """The shared gate is a plain function - the trainers only bind a context."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.learning_rate = 0
        problems = learning_rate_problems(spec, context="probe")
        assert len(problems) == 1
        assert problems[0].startswith("probe: ")

    def test_a_usable_spec_reports_nothing(self, spec: TrainSpec) -> None:
        spec.learning_rate = 1e-4
        assert learning_rate_problems(spec, context="probe") == []
        spec.learning_rate = None
        assert learning_rate_problems(spec, context="probe") == []
