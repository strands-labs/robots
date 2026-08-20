"""The reproducibility seed a TrainSpec asks for is one shared non-negative-count domain.

:attr:`~strands_robots.training.base.TrainSpec.seed` is the field a caller sets
to make a run reproducible, and five backends read it - LeRobot (``cfg.seed`` and
a ``--seed=`` argv token), Cosmos (a ``trainer.seed=`` Hydra override), the two
RL trainers (``torch.manual_seed``) and SageMaker (a string hyperparameter the
job's container seeds from). Before this contract none of them checked it,
and the appliers do not agree about a single value:

* ``torch.manual_seed`` reduces the value modulo ``2**64``, so a negative seed is
  *silently a different seed*: ``manual_seed(-1)`` and ``manual_seed(2**64 - 1)``
  draw the identical stream. Two seeds a caller means to be distinct collapse
  onto one, and the run is reproducible under a number nobody asked for. ``True``
  is a silent seed of ``1`` and ``2.7`` a silent seed of ``2``.
* LeRobot's ``set_seed`` reseeds Python's ``random`` and only *then* hands the
  value to NumPy, which refuses a negative or a float. A refused seed therefore
  leaves the process RNG reseeded by a call that failed, and the message that
  surfaces is NumPy's rather than one naming the field.
* Every value renders into a Hydra override or an argv token, so on those paths a
  bad seed fails - if at all - inside the run, after the dataset and model are
  loaded.

The tests below pin the contract in both directions: every backend that reads the
field refuses the same unusable values through the one shared domain with a
message naming itself, a usable seed (``0`` included) is untouched, a backend that
ignores the field reports nothing about it, and the refused values are grounded in
what the real appliers do with them.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
from typing import Any

import pytest

from strands_robots.training._validate import seed_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from tests.training._spec_field_reads import reads_spec_field

# The backends that seed from the field. The RL trainers are exercised through
# their own spec type further down (their validate() needs an RLTrainSpec).
SEEDING_BACKENDS = (LerobotTrainer, Cosmos3Trainer)

# Values no applier can honor, split by how each one failed before the gate.

# Silently rewritten by torch's modulo, and refused by NumPy - so no applier
# honors the value the caller supplied.
SILENTLY_REWRITTEN = (-1, -5, True, False, 2.7, 3.0)

# Raised out of the applier: torch refuses nan/inf/list, NumPy refuses the float
# and the string.
RAISED_IN_THE_APPLIER = (float("nan"), float("inf"), "42", [7])

UNUSABLE = SILENTLY_REWRITTEN + RAISED_IN_THE_APPLIER

# Seeds every applier honors as themselves. ``0`` is a seed, not a degenerate
# value, which is why the domain's floor is zero rather than one.
USABLE = (0, 1, 42, 1000, 2**31 - 1)


@pytest.fixture
def spec(tmp_path: pathlib.Path) -> TrainSpec:
    """A spec whose ``seed`` is the only thing under test.

    ``validate`` may well report unrelated problems (Cosmos wants a recipe TOML,
    LeRobot a dataset); every assertion below filters for the field name, so an
    unrelated problem can neither mask nor fake a seed verdict.
    """
    return TrainSpec(
        dataset_root=str(tmp_path / "ds"),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="new_embodiment",
    )


# The shape the shared domain emits: ``"{context}: seed must be ..."``. Matched
# rather than the bare word because pytest derives ``tmp_path`` from the test
# name, so a path in an unrelated problem can contain "seed" too - a filter that
# picked that up could both mask and fake a verdict.
_NAMES_THE_SEED = ": seed "


def _seed_problems_of(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """``validate`` problems about ``seed``."""
    return [p for p in trainer.validate(spec) if _NAMES_THE_SEED in p]


def _rl_spec(tmp_path: pathlib.Path) -> Any:
    """A minimal :class:`RLTrainSpec` for the two RL trainers."""
    from strands_robots.training.rl.base_algo import RLTrainSpec

    return RLTrainSpec(
        output_dir=str(tmp_path),
        env_factory=lambda: None,  # type: ignore[arg-type,return-value]
    )


def _rl_trainers() -> list[Any]:
    """The two RL trainers, or skip when torch is absent."""
    pytest.importorskip("torch")
    from strands_robots.training.rl.fast_sac import FastSacTrainer
    from strands_robots.training.rl.ppo import PpoTrainer

    return [FastSacTrainer(), PpoTrainer()]


class TestEverySeedingBackendRefusesAnUnusableSeed:
    """Each backend that seeds from the field refuses every unusable value."""

    @pytest.mark.parametrize("trainer_cls", SEEDING_BACKENDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.seed = value
        problems = _seed_problems_of(trainer_cls(), spec)
        assert problems, f"{trainer_cls.__name__} accepted seed={value!r}"
        assert any("must be a non-negative integer" in p for p in problems), problems

    @pytest.mark.parametrize("trainer_cls", SEEDING_BACKENDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        trainer = trainer_cls()
        spec.seed = -1
        assert any(p.startswith(f"{trainer.provider_name}: seed ") for p in _seed_problems_of(trainer, spec))

    @pytest.mark.parametrize("trainer_cls", SEEDING_BACKENDS)
    @pytest.mark.parametrize("value", RAISED_IN_THE_APPLIER)
    def test_an_unusable_seed_is_a_problem_not_an_exception(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        """``validate`` returns problems; it must not raise out of the check."""
        spec.seed = value
        problems = trainer_cls().validate(spec)  # must not raise
        assert any(_NAMES_THE_SEED in p for p in problems), problems

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_both_rl_trainers_refuse_it_too(self, tmp_path: pathlib.Path, value: Any) -> None:
        rl_spec = _rl_spec(tmp_path)
        rl_spec.seed = value
        for trainer in _rl_trainers():
            problems = [p for p in trainer.validate(rl_spec) if _NAMES_THE_SEED in p]
            assert problems, f"{type(trainer).__name__} accepted seed={value!r}"
            assert any(p.startswith(f"{trainer.provider_name}: seed ") for p in problems), problems


class TestAUsableSeedIsUntouched:
    """A seed every applier honors raises no problem, and zero is one of them."""

    @pytest.mark.parametrize("trainer_cls", SEEDING_BACKENDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_seed_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer], value: int) -> None:
        spec.seed = value
        assert _seed_problems_of(trainer_cls(), spec) == []

    @pytest.mark.parametrize("trainer_cls", SEEDING_BACKENDS)
    def test_an_unset_seed_is_not_a_problem(self, spec: TrainSpec, trainer_cls: type[Trainer]) -> None:
        """``None`` is the documented "use the backend's own default" sentinel."""
        assert spec.seed is None
        assert _seed_problems_of(trainer_cls(), spec) == []

    @pytest.mark.parametrize("value", USABLE)
    def test_the_rl_trainers_accept_it_too(self, tmp_path: pathlib.Path, value: int) -> None:
        rl_spec = _rl_spec(tmp_path)
        rl_spec.seed = value
        for trainer in _rl_trainers():
            assert [p for p in trainer.validate(rl_spec) if _NAMES_THE_SEED in p] == []


class TestABackendThatIgnoresTheFieldReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports and
    ignores the rest", so this gate is scoped to the seeding backends rather than
    made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("trainer_cls", (MockTrainer, Gr00tTrainer))
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_seeds_from_nothing(self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any) -> None:
        spec.seed = value
        assert _seed_problems_of(trainer_cls(), spec) == []


class TestTheRefusedValuesAreOnesNoApplierCanHonor:
    """Ground the domain in what the real appliers do with each value."""

    @pytest.mark.parametrize(("supplied", "actual"), ((-1, 2**64 - 1), (-5, 2**64 - 5), (True, 1), (2.7, 2)))
    def test_torch_silently_seeds_a_different_stream(self, supplied: Any, actual: int) -> None:
        """The headline: a refused value is not merely rejected downstream.

        ``torch.manual_seed`` reduces its argument, so the run IS reproducible -
        under a seed the caller did not name, and one another caller could have.
        """
        torch = pytest.importorskip("torch")
        torch.manual_seed(supplied)
        as_supplied = torch.rand(8).tolist()
        torch.manual_seed(actual)
        as_rewritten = torch.rand(8).tolist()
        assert as_supplied == as_rewritten, f"seed={supplied!r} no longer collapses onto {actual}"

    @pytest.mark.parametrize("value", (-1, -5, 2.7, 3.0, float("nan"), float("inf"), "42"))
    def test_numpys_legacy_seeder_refuses_it(self, value: Any) -> None:
        """The narrowest applier on the LeRobot path refuses what torch rewrites."""
        np = pytest.importorskip("numpy")
        with pytest.raises((ValueError, TypeError)):
            np.random.seed(value)

    def test_a_refused_seed_still_reseeds_pythons_random_first(self) -> None:
        """Why the check must precede the applier rather than trust it.

        ``set_seed`` reseeds ``random`` before NumPy refuses, so the failed call
        mutates process state a caller never asked it to touch.
        """
        pytest.importorskip("numpy")
        set_seed = pytest.importorskip("lerobot.utils.random_utils").set_seed
        import random

        random.seed(999)
        untouched = [random.random() for _ in range(2)]
        random.seed(999)
        with pytest.raises((ValueError, TypeError)):
            set_seed(-1)
        after_the_failure = [random.random() for _ in range(2)]
        assert untouched != after_the_failure


def _trainer_modules() -> list[pathlib.Path]:
    """Every trainer module, minus the one that defines the shared gate.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The module that *defines* the gate is
    excluded - derived from the gate itself rather than named, so the exclusion
    cannot drift - because it reads the field as its owner, not as a consumer.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(seed_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_seed(source: str) -> bool:
    """Does *source* read ``spec.seed``, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("seed",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "_seed_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheSeedDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.seed`` must route it through the shared gate, so a
    fifth backend that starts seeding from the field fails this test until it does.
    """

    def test_the_scan_finds_the_seeding_backends(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if _reads_the_seed(p.read_text())}
        assert readers == {"cosmos3.py", "lerobot.py", "fast_sac.py", "ppo.py", "sagemaker.py"}

    def test_every_backend_that_seeds_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if _reads_the_seed(source := p.read_text()) and not _calls_the_gate(source)
        )
        assert adrift == [], f"modules reading spec.seed without the shared gate: {adrift}"

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local sign or type test on the field is the hole this closed."""
        offenders: list[str] = []
        for path in _trainer_modules():
            for line in path.read_text().splitlines():
                if "spec.seed" in line and ("< 0" in line or ">= 0" in line or "int(" in line):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on spec.seed: {offenders}"

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [] if spec.seed is None else []\n"
        assert _reads_the_seed(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards the field by name is a reader too.

        The form a transport-only provider takes: no attribute access mentions
        the field, so a scan keyed on ``spec.seed`` alone reports a clean sweep
        while this backend skips the gate.
        """
        planted = 'FIELDS = ("seed",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in FIELDS]\n'
        assert _reads_the_seed(planted)
        assert not _calls_the_gate(planted)


class TestTheGateIsUsableOnItsOwn:
    """The shared gate's own contract, independent of any backend."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.seed = -1
        assert seed_problems(spec, context="acme") == ["acme: seed must be a non-negative integer, got -1."]

    def test_a_usable_seed_reports_nothing(self, spec: TrainSpec) -> None:
        spec.seed = 0
        assert seed_problems(spec, context="acme") == []

    def test_an_unset_seed_reports_nothing(self, spec: TrainSpec) -> None:
        assert seed_problems(spec, context="acme") == []
