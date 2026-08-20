"""The held-out validation episode count is one shared positive-count domain.

:attr:`~strands_robots.training.base.TrainSpec.val_episodes` is the number of
episodes a caller reserves from the tail of the dataset to validate on, and it is
consumed by two independent writers of the same lerobot flag: the LeRobot
trainer's :meth:`~strands_robots.training.lerobot.LerobotTrainer.validate` /
config path, and the ``lerobot_train`` tool's ``build_train_command``. Before
this contract each writer compared the value itself, and the two comparisons
disagreed about the same input.

Neither comparison is safe, because the count is not read into a loop bound - it
is converted into lerobot's real-valued ``dataset.eval_split`` fraction, and
lerobot then holds out ``ceil(episodes_in_task * eval_split)``:

* A non-positive count is **silently dropped**. The fraction is only computed for
  a count in ``(0, total)``, so ``val_episodes=0`` produced no ``eval_split`` and
  no ``eval_steps`` at all: the run trained on the whole dataset, recorded no
  validation loss, and ``validate`` reported no problem. The tool refused the
  same value, so one field had opposite verdicts on the two paths.
* A value that merely *compares* as positive is silently rewritten: ``True``
  reserved 1 episode and ``2.7`` reserved 3 - a whole number nobody asked for -
  on both paths, and ``nan`` rendered as the literal ``--dataset.eval_split=nan``.
* A non-numeric value raised ``TypeError`` out of the comparison itself, from a
  ``validate`` documented to *return* problems.

The tests below pin the contract in both directions: every unusable value is
reported (never raised) by the backend that reads the field and refused by the
tool that writes the same flag, a usable count still produces the split and the
evaluation cadence, the dataset-dependent bounds that already worked still fire,
and a backend that ignores the field reports nothing about it.
"""

from __future__ import annotations

import ast
import inspect
import json
import math
import pathlib
from typing import Any

import pytest

from strands_robots.tools.lerobot_train import build_train_command
from strands_robots.training._validate import validation_episodes_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from strands_robots.utils import validation_split_fraction
from tests.training._spec_field_reads import reads_spec_field

TOTAL_EPISODES = 10

# Values that were silently dropped: the split fraction is only computed for a
# count in ``(0, total)``, so the run held nothing out and said nothing.
SILENTLY_DROPPED = (0, -1, -5, float("nan"))

# Values that compared as positive and were silently rewritten by the ceiling
# lerobot applies to the fraction, mapped to the count each one really reserved.
REWRITTEN_TO = ((True, 1), (2.7, 3), (0.5, 0))
SILENTLY_REWRITTEN = tuple(supplied for supplied, _ in REWRITTEN_TO)

# Values that raised out of the comparison, from a method documented to return.
RAISED_IN_THE_COMPARISON = ("5", [5], {"n": 5})

UNUSABLE = SILENTLY_DROPPED + SILENTLY_REWRITTEN + RAISED_IN_THE_COMPARISON

# Counts that reserve exactly what they name on a 10-episode single-task dataset.
USABLE = (1, 2, 3, 9)


def _write_dataset(root: pathlib.Path, total_episodes: int = TOTAL_EPISODES, total_tasks: int = 1) -> pathlib.Path:
    """A minimal LeRobot v3 dataset stub the episode-count checks can read."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    meta.joinpath("info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": total_episodes,
                "total_tasks": total_tasks,
                "total_frames": total_episodes * 120,
                "fps": 30,
                "features": {},
            }
        )
    )
    return root


@pytest.fixture
def dataset(tmp_path: pathlib.Path) -> pathlib.Path:
    return _write_dataset(tmp_path / "ds")


@pytest.fixture
def spec(tmp_path: pathlib.Path, dataset: pathlib.Path) -> TrainSpec:
    """A spec whose ``val_episodes`` is the only thing under test.

    ``validate`` may report unrelated problems (a missing policy config, an
    absent lerobot); every assertion below filters for the field name, so an
    unrelated problem can neither mask nor fake a verdict.
    """
    return TrainSpec(
        dataset_root=str(dataset),
        output_dir=str(tmp_path / "out"),
        base_model="lerobot/act",
        embodiment="so101",
        steps=100,
        global_batch_size=8,
        extra={"policy_type": "act"},
    )


# The shape the shared domain emits: ``"{context}: val_episodes must be ..."``.
# Matched rather than the bare word because pytest derives ``tmp_path`` from the
# test name, so a path in an unrelated problem can contain "val_episodes" too - a
# filter that picked that up could both mask and fake a verdict.
_NAMES_THE_COUNT = ": val_episodes "


def _count_problems_of(trainer: Trainer, spec: TrainSpec) -> list[str]:
    """``validate`` problems emitted by the shared count domain."""
    return [p for p in trainer.validate(spec) if _NAMES_THE_COUNT in p]


def _eval_flags(cmd: list[str]) -> list[str]:
    """The flags that make lerobot hold data out and evaluate on it."""
    return [c for c in cmd if "eval_split" in c or "eval_steps" in c]


def _raising_tool(exc: BaseException) -> Any:
    """A ``build_train_command`` stand-in that raises, so the width is observable."""

    def tool(**_kwargs: Any) -> list[str]:
        raise exc

    return tool


def _tool_verdict(dataset_root: pathlib.Path, value: Any) -> str:
    """``refused`` / ``accepted`` for the tool that writes the same flag.

    The returned string is *compared* as an answer about ``val_episodes``, so
    the handler catches :class:`Exception` and stops there. A library failure is
    one of the answers being collected - the tool raising instead of refusing is
    part of what this file pins - so it must not abort the comparison. An
    interrupt is not an answer about the field, though: recording one as "the
    tool did not refuse this value" and comparing it against the backend turns an
    operator's Ctrl-C into a verdict. ``pytest``'s own ``skip`` and ``fail``
    outcomes derive from ``BaseException`` for the same reason, so they have to
    reach the runner rather than becoming a verdict.
    """
    try:
        build_train_command(dataset_root=str(dataset_root), policy_type="act", val_episodes=value)
    except ValueError as exc:
        return "refused" if "val_episodes" in str(exc) else f"other-error: {exc}"
    except Exception as exc:  # noqa: BLE001 - a library failure is a verdict, control flow is not
        return f"raised {type(exc).__name__}: {exc}"
    return "accepted"


class TestTheBackendRefusesAnUnusableValidationCount:
    """Every value no split can honor is reported as a problem, never raised."""

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: TrainSpec, value: Any) -> None:
        spec.val_episodes = value
        problems = _count_problems_of(LerobotTrainer(), spec)
        assert problems, f"LerobotTrainer accepted val_episodes={value!r}"
        assert any("must be a positive integer" in p for p in problems), problems

    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec) -> None:
        trainer = LerobotTrainer()
        spec.val_episodes = 0
        assert any(p.startswith(f"{trainer.provider_name}: val_episodes ") for p in _count_problems_of(trainer, spec))

    @pytest.mark.parametrize("value", RAISED_IN_THE_COMPARISON)
    def test_a_non_numeric_count_is_a_problem_not_an_exception(self, spec: TrainSpec, value: Any) -> None:
        """The comparison the gate replaced raised ``TypeError`` for these."""
        spec.val_episodes = value
        problems = LerobotTrainer().validate(spec)  # must not raise
        assert any(_NAMES_THE_COUNT in p for p in problems), problems

    @pytest.mark.parametrize("value", RAISED_IN_THE_COMPARISON)
    def test_the_dataset_dependent_comparison_is_not_reached(self, spec: TrainSpec, value: Any) -> None:
        """Guard order: the domain gate runs before anything compares the value.

        The ``>= total_episodes`` refusal below it is only a meaningful comparison
        once the value IS a count, so a non-numeric must be reported by the gate
        and stop there rather than reaching - and raising out of - the bound.
        """
        spec.val_episodes = value
        problems = LerobotTrainer().validate(spec)
        assert not [p for p in problems if "total_episodes=" in p], problems


class TestANonPositiveCountWasSilentlyDropped:
    """The headline: the request produced a run with no validation at all.

    A count outside ``(0, total)`` never became an ``eval_split``, so lerobot was
    given no held-out set and no ``eval_steps`` cadence - the run trained on every
    episode and recorded no validation loss, and nothing reported it.
    """

    @pytest.mark.parametrize("value", SILENTLY_DROPPED)
    def test_no_split_and_no_evaluation_cadence_is_produced(self, spec: TrainSpec, value: Any) -> None:
        spec.val_episodes = value
        assert _eval_flags(LerobotTrainer().build_command(spec)) == []

    @pytest.mark.parametrize("value", SILENTLY_DROPPED)
    def test_so_the_request_is_refused_instead(self, spec: TrainSpec, value: Any) -> None:
        spec.val_episodes = value
        assert _count_problems_of(LerobotTrainer(), spec)


class TestACountThatComparesAsPositiveWasSilentlyRewritten:
    """A value the comparison admitted reserved a count nobody asked for."""

    @pytest.mark.parametrize(("supplied", "reserved"), REWRITTEN_TO)
    def test_the_fraction_holds_out_a_count_that_was_never_asked_for(self, supplied: Any, reserved: int) -> None:
        """Ground the refusal in what lerobot really does with the fraction.

        lerobot holds out ``ceil(episodes_in_task * eval_split)``, so the
        real-valued fraction a non-integer count maps to reserves a whole number
        the caller never named as one: ``2.7`` reserves 3, a ``bool`` reserves 1,
        and ``0.5`` reserves nothing at all.
        """
        fraction = validation_split_fraction(supplied, TOTAL_EPISODES)
        assert math.ceil(TOTAL_EPISODES * fraction) == reserved
        assert not (type(supplied) is int and supplied == reserved), (
            f"{supplied!r} is an integer count of {reserved}, so it is not a rewrite"
        )

    def test_a_fractional_count_below_one_reserves_nothing_yet_still_evaluates(self, spec: TrainSpec) -> None:
        """The sharpest of the three: an evaluation pass over an empty set.

        ``0.5`` clears the ``0 < count < total`` test, so unlike a non-positive
        count it DOES emit a split - ``eval_split=0.0``, which holds out zero
        episodes - together with an ``eval_steps`` cadence. lerobot is asked to
        validate periodically on nothing.
        """
        # Bound through Any because the point is what the RUNTIME did with a
        # value outside the field's declared ``int | None``, which is exactly
        # what the gate now refuses.
        fractional: Any = 0.5
        assert math.ceil(TOTAL_EPISODES * validation_split_fraction(fractional, TOTAL_EPISODES)) == 0
        spec.val_episodes = fractional
        assert _count_problems_of(LerobotTrainer(), spec)

    @pytest.mark.parametrize("value", SILENTLY_REWRITTEN)
    def test_so_the_request_is_refused_instead(self, spec: TrainSpec, value: Any) -> None:
        spec.val_episodes = value
        assert _count_problems_of(LerobotTrainer(), spec)


class TestAUsableCountIsUntouched:
    """A count the split reproduces exactly still produces the split."""

    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_count_is_not_a_problem(self, spec: TrainSpec, value: int) -> None:
        spec.val_episodes = value
        assert _count_problems_of(LerobotTrainer(), spec) == []

    @pytest.mark.parametrize("value", USABLE)
    def test_it_still_produces_the_split_and_the_evaluation_cadence(self, spec: TrainSpec, value: int) -> None:
        spec.val_episodes = value
        flags = _eval_flags(LerobotTrainer().build_command(spec))
        expected = validation_split_fraction(value, TOTAL_EPISODES)
        assert f"--dataset.eval_split={expected}" in flags
        assert any(f.startswith("--eval_steps=") for f in flags), flags

    @pytest.mark.parametrize("value", USABLE)
    def test_the_split_reserves_exactly_what_was_asked_for(self, value: int) -> None:
        fraction = validation_split_fraction(value, TOTAL_EPISODES)
        assert math.ceil(TOTAL_EPISODES * fraction) == value

    def test_an_unset_count_is_not_a_problem(self, spec: TrainSpec) -> None:
        """``None`` is the documented "train on every episode" sentinel."""
        assert spec.val_episodes is None
        assert _count_problems_of(LerobotTrainer(), spec) == []
        assert _eval_flags(LerobotTrainer().build_command(spec)) == []


class TestTheDatasetDependentBoundsStillApply:
    """Over-reach controls: the checks that already worked are untouched.

    The gate decides the type and the floor; the upper bound and the per-task
    fraction refusal stay with the backend that reads the dataset metadata.
    """

    def test_a_count_that_leaves_no_training_data_is_still_refused(self, spec: TrainSpec) -> None:
        spec.val_episodes = TOTAL_EPISODES
        problems = LerobotTrainer().validate(spec)
        assert any("total_episodes=" in p for p in problems), problems

    def test_a_multi_task_dataset_still_refuses_a_global_count(self, tmp_path: pathlib.Path) -> None:
        root = _write_dataset(tmp_path / "multi", total_tasks=3)
        spec = TrainSpec(
            dataset_root=str(root),
            output_dir=str(tmp_path / "out"),
            base_model="lerobot/act",
            embodiment="so101",
            steps=100,
            global_batch_size=8,
            extra={"policy_type": "act"},
            val_episodes=2,
        )
        problems = LerobotTrainer().validate(spec)
        assert any("cannot be reserved exactly" in p for p in problems), problems


class TestBothWritersOfTheSplitShareOneDomain:
    """The trainer and the tool build the same flag, so they refuse the same set.

    An implication rather than an equivalence: the tool additionally reads the
    dataset itself and may refuse a count the backend defers on, so the pinned
    property is that nothing the backend refuses is accepted by the tool.
    """

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_tool_refuses_every_value_the_backend_refuses(
        self, spec: TrainSpec, dataset: pathlib.Path, value: Any
    ) -> None:
        spec.val_episodes = value
        assert _count_problems_of(LerobotTrainer(), spec), f"backend accepted {value!r}"
        assert _tool_verdict(dataset, value) == "refused", f"tool accepted {value!r}"

    @pytest.mark.parametrize("value", USABLE)
    def test_the_tool_accepts_every_value_the_backend_accepts(
        self, spec: TrainSpec, dataset: pathlib.Path, value: int
    ) -> None:
        spec.val_episodes = value
        assert _count_problems_of(LerobotTrainer(), spec) == []
        assert _tool_verdict(dataset, value) == "accepted"

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_tool_names_the_field_and_the_domain(self, dataset: pathlib.Path, value: Any) -> None:
        with pytest.raises(ValueError, match="val_episodes must be a positive integer"):
            build_train_command(dataset_root=str(dataset), policy_type="act", val_episodes=value)

    @pytest.mark.parametrize("value", SILENTLY_REWRITTEN + RAISED_IN_THE_COMPARISON)
    def test_the_tool_refuses_before_it_reads_the_dataset(self, tmp_path: pathlib.Path, value: Any) -> None:
        """The domain does not depend on the metadata, so it is checked first."""
        with pytest.raises(ValueError, match="val_episodes must be a positive integer"):
            build_train_command(dataset_root=str(tmp_path / "absent"), policy_type="act", val_episodes=value)


class TestTheParityClassifierCollectsFailuresWithoutSwallowingControlFlow:
    """``_tool_verdict`` must catch a library failure and only a library failure.

    The classifier turns "what did the tool do with this value" into the string
    the two parity tests above compare against ``refused`` / ``accepted``. A raise
    is one of the answers it has to collect - a non-numeric count really does
    raise out of the tool - so a library failure cannot be allowed to abort the
    comparison. An interrupt is not an answer about the field, though: recording
    one as a verdict and then comparing it against the backend turns an operator's
    Ctrl-C into a claim about ``val_episodes``. Both halves are pinned here
    because the correct handler is the one that satisfies both.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            TypeError("'>=' not supported between instances of 'str' and 'int'"),
            FileNotFoundError("meta/info.json"),
        ],
        ids=["TypeError", "FileNotFoundError"],
    )
    def test_a_library_failure_is_collected_as_the_verdict(
        self, monkeypatch: pytest.MonkeyPatch, dataset: pathlib.Path, exc: Exception
    ) -> None:
        """Dropping the catch-all would let these escape instead of being recorded."""
        monkeypatch.setattr(f"{__name__}.build_train_command", _raising_tool(exc))
        assert _tool_verdict(dataset, 3) == f"raised {type(exc).__name__}: {exc}"

    @pytest.mark.parametrize(
        "exc",
        [
            KeyboardInterrupt(),
            SystemExit(1),
            pytest.skip.Exception("the lerobot extra is not installed"),
            pytest.fail.Exception("the dataset fixture is incomplete"),
        ],
        ids=["KeyboardInterrupt", "SystemExit", "pytest.skip", "pytest.fail"],
    )
    def test_control_flow_reaches_the_runner_instead_of_becoming_a_verdict(
        self, monkeypatch: pytest.MonkeyPatch, dataset: pathlib.Path, exc: BaseException
    ) -> None:
        # Executable premise: each of these derives from BaseException without
        # deriving from Exception, which is the whole reason the handler width
        # is observable at all.
        assert not isinstance(exc, Exception), f"{type(exc).__name__} no longer tests the handler width"
        monkeypatch.setattr(f"{__name__}.build_train_command", _raising_tool(exc))
        with pytest.raises(type(exc)):
            _tool_verdict(dataset, 3)


class TestABackendThatIgnoresTheFieldReportsNothing:
    """A backend must not report on a field it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports and
    ignores the rest", so this gate is scoped to the LeRobot backend rather than
    made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("trainer_cls", (MockTrainer, Gr00tTrainer, Cosmos3Trainer))
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_validates_nothing_about_the_count(
        self, spec: TrainSpec, trainer_cls: type[Trainer], value: Any
    ) -> None:
        spec.val_episodes = value
        assert _count_problems_of(trainer_cls(), spec) == []


def _trainer_modules() -> list[pathlib.Path]:
    """Every trainer module, minus the one that defines the shared gate.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The module that *defines* the gate is
    excluded - derived from the gate itself rather than named, so the exclusion
    cannot drift - because it reads the field as its owner, not as a consumer.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(validation_episodes_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_the_count(source: str) -> bool:
    """Does *source* read ``spec.val_episodes``, by name or through a table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, ("val_episodes",))


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_validation_episodes_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheValidationEpisodesDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* ``spec.val_episodes`` must route it through the shared
    gate, so a third backend that starts reserving a validation set fails this
    test until it does.
    """

    def test_the_scan_finds_the_backend_that_reads_the_field(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if _reads_the_count(p.read_text())}
        assert readers == {"lerobot.py", "sagemaker.py"}

    def test_every_backend_that_reads_it_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if _reads_the_count(source := p.read_text()) and not _calls_the_gate(source)
        )
        assert adrift == [], f"modules reading spec.val_episodes without the shared gate: {adrift}"

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local sign or type test on the field is the hole this closed."""
        offenders: list[str] = []
        for path in _trainer_modules():
            for line in path.read_text().splitlines():
                if "spec.val_episodes" in line and ("<= 0" in line or "> 0" in line or "int(" in line):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on spec.val_episodes: {offenders}"

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [] if spec.val_episodes is None else []\n"
        assert _reads_the_count(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards the field by name is a reader too.

        The form a transport-only provider takes: no attribute access mentions
        the field, so a scan keyed on ``spec.val_episodes`` alone reports a clean
        sweep while this backend skips the gate.
        """
        planted = 'F = ("val_episodes",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in F]\n'
        assert _reads_the_count(planted)
        assert not _calls_the_gate(planted)


class TestTheGateIsUsableOnItsOwn:
    """The shared gate's own contract, independent of any backend."""

    def test_it_reports_the_context_it_was_given(self, spec: TrainSpec) -> None:
        spec.val_episodes = 0
        (problem,) = validation_episodes_problems(spec, context="anything")
        assert problem.startswith("anything: val_episodes ")

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_reports_exactly_one_problem_per_unusable_value(self, spec: TrainSpec, value: Any) -> None:
        spec.val_episodes = value
        assert len(validation_episodes_problems(spec, context="ctx")) == 1

    @pytest.mark.parametrize("value", USABLE)
    def test_it_reports_nothing_for_a_usable_count(self, spec: TrainSpec, value: int) -> None:
        spec.val_episodes = value
        assert validation_episodes_problems(spec, context="ctx") == []

    def test_an_unset_count_is_not_a_problem(self, spec: TrainSpec) -> None:
        assert spec.val_episodes is None
        assert validation_episodes_problems(spec, context="ctx") == []
