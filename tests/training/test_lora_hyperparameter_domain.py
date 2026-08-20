# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""The LoRA adapter hyperparameters are one shared positive-count domain.

``lora_r`` and ``lora_alpha`` are the rank and the scaling numerator of a LoRA
fine-tune: peft builds a rank-``r`` adapter and applies its update scaled by
``lora_alpha / r``. Three places carried them and none judged them - the
in-process ``peft_kwargs`` the LeRobot trainer builds, that trainer's
argv-parity command, and :func:`build_train_command`, a second independent
writer of the same two flags in the tools layer.

Only one of the two fields fails loudly. peft refuses a non-positive ``r`` from
inside ``get_peft_model``, after the base model is downloaded and loaded, and a
``bool``/float ``r`` raises out of torch's tensor allocation. ``lora_alpha`` is a
bare numerator that nothing compares, so every unusable value is *accepted*:
``lora_alpha=0`` builds the adapter, reports its trainable parameters and trains
them with a scaling of ``0.0``, so the adapter provably cannot change the
model's output - the run completes, writes checkpoints, and has learned nothing
that can ever be applied.

The two paths also disagreed about a fractional value: peft accepts
``lora_alpha=2.7`` in-process while lerobot's ``PeftConfig`` declares both fields
``int``, so the argv spelling of the same run is refused by draccus.

A positive integer is what both paths can honor, so both fields are checked
against the same shared :func:`~strands_robots.utils.positive_count_error`
domain the run-size knobs use, in one gate that every writer routes through.
"""

from __future__ import annotations

import ast
import inspect
import json
import pathlib
from collections import OrderedDict
from typing import Any

import pytest

from strands_robots.tools.lerobot_train import build_train_command
from strands_robots.training._validate import lora_hyperparameter_problems
from strands_robots.training.base import Trainer, TrainSpec
from strands_robots.training.cosmos3 import Cosmos3Trainer
from strands_robots.training.groot import Gr00tTrainer
from strands_robots.training.lerobot import LerobotTrainer
from strands_robots.training.mock import MockTrainer
from tests.training._spec_field_reads import reads_spec_field

TOTAL_EPISODES = 10

# Values peft refuses, but only from inside get_peft_model - after the base model
# has been downloaded and loaded, with a message naming neither field nor run.
REFUSED_LATE_BY_PEFT = (0, -8)

# Values that reached torch's tensor allocation and raised there instead.
RAISED_IN_TORCH = (2.7, float("nan"), float("inf"))

# Values silently honored as a different number: bool is an int subclass, so a
# bare ``value < 1`` test reads True as a request for a rank/alpha of one.
SILENTLY_ONE = (True,)

# Values that render into an argv token and fail, if at all, inside the run.
NON_NUMERIC = ("8", [8], {"r": 8})

UNUSABLE = REFUSED_LATE_BY_PEFT + RAISED_IN_TORCH + SILENTLY_ONE + NON_NUMERIC + (False,)

# Ranks and alphas that name exactly what they mean.
USABLE = (1, 8, 16, 64)

FIELDS = ("lora_r", "lora_alpha")


def _write_dataset(root: pathlib.Path) -> pathlib.Path:
    """A minimal LeRobot v3 dataset stub ``validate`` can read."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    meta.joinpath("info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "total_episodes": TOTAL_EPISODES,
                "total_tasks": 1,
                "total_frames": TOTAL_EPISODES * 120,
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
    """A LoRA spec whose adapter hyperparameters are the only thing under test.

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
        method="lora",
        extra={"policy_type": "act"},
    )


def _problems_of(trainer: Trainer, spec: TrainSpec, field: str) -> list[str]:
    """``validate`` problems the shared domain emitted about *field*.

    Matched on the ``"{context}: {field} "`` shape the shared domain emits rather
    than the bare field name: pytest derives ``tmp_path`` from the test name, so a
    path in an unrelated problem can contain the word too - a filter that picked
    that up could both mask and fake a verdict.
    """
    return [p for p in trainer.validate(spec) if f": {field} " in p]


def _build(**kwargs: Any) -> list[str]:
    """Call ``build_train_command`` with the field under test named dynamically.

    Every test here parametrizes over WHICH of the two hyperparameters carries
    the value, so the field name is a variable. Splatting it through one
    ``**kwargs: Any`` funnel keeps that dynamism in a single documented place
    rather than in a suppression at each call site.
    """
    return build_train_command(**kwargs)


def _peft_flags(cmd: list[str]) -> list[str]:
    """The flags that configure the adapter."""
    return [c for c in cmd if c.startswith("--peft.")]


def _raising_tool(exc: BaseException) -> Any:
    """A ``build_train_command`` stand-in that raises, so the width is observable."""

    def tool(**_kwargs: Any) -> list[str]:
        raise exc

    return tool


def _tool_verdict(dataset_root: pathlib.Path, field: str, value: Any) -> str:
    """``refused`` / ``accepted`` for the tool that writes the same two flags.

    The returned string is *compared* as an answer about *field*, so the handler
    catches :class:`Exception` and stops there. A library failure is one of the
    answers being collected - the tool raising instead of refusing is part of what
    this file pins - so it must not abort the comparison. An interrupt is not an
    answer about the field, though: recording one as "the tool did not refuse this
    value" and comparing it against the backend turns an operator's Ctrl-C into a
    verdict. ``pytest``'s own ``skip`` and ``fail`` outcomes derive from
    ``BaseException`` for the same reason, so they have to reach the runner rather
    than becoming a verdict.
    """
    try:
        _build(dataset_root=str(dataset_root), policy_type="act", lora=True, **{field: value})
    except ValueError as exc:
        return "refused" if field in str(exc) else f"other-error: {exc}"
    except Exception as exc:  # noqa: BLE001 - a library failure is a verdict, control flow is not
        return f"raised {type(exc).__name__}: {exc}"
    return "accepted"


def _lora_target(torch: Any) -> Any:
    """A one-linear-layer module whose ``q_proj`` a LoRA adapter can target.

    Built from ``OrderedDict`` rather than declared as a subclass because
    ``torch`` arrives from :func:`pytest.importorskip` - the type checker cannot
    resolve a base class that only exists once the optional dependency is
    present, and peft only needs a named submodule to adapt.
    """
    return torch.nn.Sequential(OrderedDict(q_proj=torch.nn.Linear(8, 8, bias=False)))


class TestAScalingOfZeroTrainsAnAdapterThatCannotBeApplied:
    """The measured consequence of the value nothing refused.

    ``lora_alpha`` is only ever the numerator of ``lora_alpha / r``, so peft has
    nothing to compare it against and accepts it. These pin what that produced,
    because it is the reason a bare numerator needs a domain of its own rather
    than being left to the library.
    """

    def test_a_zero_alpha_adapter_provably_cannot_change_the_model_output(self) -> None:
        torch = pytest.importorskip("torch")
        peft = pytest.importorskip("peft")

        def effect_of(alpha: int) -> tuple[float, int]:
            wrapped = peft.get_peft_model(
                _lora_target(torch), peft.LoraConfig(r=8, lora_alpha=alpha, target_modules=["q_proj"])
            )
            trainable = sum(p.numel() for p in wrapped.parameters() if p.requires_grad)
            # Perturb the adapter's own weights: an adapter with any scaling at
            # all then changes the output, so a zero delta isolates the scaling.
            for name, param in wrapped.named_parameters():
                if "lora_B" in name and param.numel():
                    with torch.no_grad():
                        param.add_(1.0)
            probe = torch.ones(1, 8)
            with torch.no_grad():
                with wrapped.disable_adapter():
                    base = wrapped(probe)
                adapted = wrapped(probe)
            return float((adapted - base).abs().sum()), trainable

        zero_effect, zero_trainable = effect_of(0)
        usable_effect, usable_trainable = effect_of(8)

        # Non-vacuity: the two runs built the same adapter, so the only thing
        # that differs is whether its update can ever be applied.
        assert zero_trainable == usable_trainable > 0
        assert usable_effect > 0.0, "the probe cannot detect an adapter at all"
        assert zero_effect == 0.0, "a zero-alpha adapter would have changed the output"

    @pytest.mark.parametrize("value", REFUSED_LATE_BY_PEFT)
    def test_a_non_positive_rank_is_refused_only_once_the_model_is_built(self, value: int) -> None:
        """peft judges the rank, but not until the base model is already loaded."""
        torch = pytest.importorskip("torch")
        peft = pytest.importorskip("peft")
        config = peft.LoraConfig(r=value, lora_alpha=8, target_modules=["q_proj"])

        # Building the config is not where it is caught - that is the point.
        assert config.r == value
        with pytest.raises(ValueError, match="positive integer"):
            peft.get_peft_model(_lora_target(torch), config)


class TestTheBackendRefusesAnUnusableAdapterHyperparameter:
    """Every value neither path can honor is reported as a problem, never raised."""

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_it_is_reported_as_a_problem(self, spec: TrainSpec, field: str, value: Any) -> None:
        setattr(spec, field, value)
        assert _problems_of(LerobotTrainer(), spec, field), f"{field}={value!r} was accepted"

    @pytest.mark.parametrize("field", FIELDS)
    def test_the_problem_names_the_backend_that_refused_it(self, spec: TrainSpec, field: str) -> None:
        setattr(spec, field, 0)
        problems = _problems_of(LerobotTrainer(), spec, field)
        assert problems and problems[0].startswith("lerobot_local: ")

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", NON_NUMERIC)
    def test_a_non_numeric_value_is_a_problem_not_an_exception(self, spec: TrainSpec, field: str, value: Any) -> None:
        """``validate`` is documented to *return* problems, so it must not raise."""
        setattr(spec, field, value)
        assert _problems_of(LerobotTrainer(), spec, field)

    def test_both_unusable_values_are_reported_together(self, spec: TrainSpec) -> None:
        """One pass reports both, rather than making the caller re-run to find the second."""
        spec.lora_r = 0
        spec.lora_alpha = -8
        problems = lora_hyperparameter_problems(spec, context="lerobot_local")
        assert len(problems) == 2
        assert any("lora_r" in p for p in problems) and any("lora_alpha" in p for p in problems)


class TestAUsableAdapterHyperparameterIsUntouched:
    """The guard refuses exactly the values that cannot be honored."""

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_a_usable_value_is_not_a_problem(self, spec: TrainSpec, field: str, value: int) -> None:
        setattr(spec, field, value)
        assert _problems_of(LerobotTrainer(), spec, field) == []

    @pytest.mark.parametrize("field", FIELDS)
    def test_an_unset_value_is_not_a_problem(self, spec: TrainSpec, field: str) -> None:
        """``None`` is the documented "keep peft's own default" sentinel."""
        setattr(spec, field, None)
        assert _problems_of(LerobotTrainer(), spec, field) == []

    def test_an_alpha_larger_than_the_rank_is_usable(self, spec: TrainSpec) -> None:
        """The common LoRA setting: alpha above r scales the update up."""
        spec.lora_r = 8
        spec.lora_alpha = 32
        assert lora_hyperparameter_problems(spec, context="lerobot_local") == []


class TestOnlyTheStrategyThatReadsThemIsChecked:
    """The fields are read on the ``method == "lora"`` branch and nowhere else.

    Refusing a value a run's own strategy never reads would be a false rejection,
    for the same reason a backend that ignores a field must not report on it.
    """

    @pytest.mark.parametrize("method", ("full", "expert_only", "frozen_backbone"))
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_another_strategy_reports_nothing(self, spec: TrainSpec, method: str, value: Any) -> None:
        spec.method = method
        spec.lora_alpha = value
        assert lora_hyperparameter_problems(spec, context="lerobot_local") == []

    @pytest.mark.parametrize("value", UNUSABLE)
    def test_but_the_lora_strategy_does(self, spec: TrainSpec, value: Any) -> None:
        """Non-vacuity: the scoping above is not just an always-empty gate."""
        spec.method = "lora"
        spec.lora_alpha = value
        assert lora_hyperparameter_problems(spec, context="lerobot_local")


class TestBothWritersOfThePeftFlagsShareOneDomain:
    """The tools-layer writer and the backend agree about every value.

    :func:`build_train_command` writes ``--peft.r`` / ``--peft.lora_alpha``
    without going through a :class:`TrainSpec`, so a domain on the backend alone
    would leave one of the two writers judging nothing.
    """

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_tool_refuses_every_value_the_backend_refuses(
        self, spec: TrainSpec, dataset: pathlib.Path, field: str, value: Any
    ) -> None:
        setattr(spec, field, value)
        assert _problems_of(LerobotTrainer(), spec, field), "fixture drift: the backend accepted it"
        assert _tool_verdict(dataset, field, value) == "refused"

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_the_tool_accepts_every_value_the_backend_accepts(
        self, spec: TrainSpec, dataset: pathlib.Path, field: str, value: int
    ) -> None:
        setattr(spec, field, value)
        assert _problems_of(LerobotTrainer(), spec, field) == []
        assert _tool_verdict(dataset, field, value) == "accepted"

    @pytest.mark.parametrize("field", FIELDS)
    def test_the_tool_names_the_field_and_the_domain(self, dataset: pathlib.Path, field: str) -> None:
        with pytest.raises(ValueError, match=f"{field} must be a positive integer"):
            _build(dataset_root=str(dataset), policy_type="act", lora=True, **{field: 0})

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", USABLE)
    def test_an_honored_value_still_reaches_the_argv_unchanged(
        self, dataset: pathlib.Path, field: str, value: int
    ) -> None:
        flag = "--peft.r" if field == "lora_r" else "--peft.lora_alpha"
        cmd = _build(dataset_root=str(dataset), policy_type="act", lora=True, **{field: value})
        assert f"{flag}={value}" in _peft_flags(cmd)

    @pytest.mark.parametrize("field", FIELDS)
    @pytest.mark.parametrize("value", UNUSABLE)
    def test_the_tool_emits_nothing_when_lora_was_not_requested(
        self, dataset: pathlib.Path, field: str, value: Any
    ) -> None:
        """Neither flag is written without ``lora``, so neither value is read."""
        cmd = _build(dataset_root=str(dataset), policy_type="act", lora=False, **{field: value})
        assert _peft_flags(cmd) == []


class TestTheParityClassifierCollectsFailuresWithoutSwallowingControlFlow:
    """``_tool_verdict`` must catch a library failure and only a library failure.

    The classifier turns "what did the tool do with this value" into the string
    the two parity tests above compare against ``refused`` / ``accepted``. A raise
    is one of the answers it has to collect, so a library failure cannot be
    allowed to abort the comparison. An interrupt is not an answer about the
    field, though: recording one as a verdict and then comparing it against the
    backend turns an operator's Ctrl-C into a claim about the adapter. Both
    halves are pinned here because the correct handler is the one that satisfies
    both.
    """

    @pytest.mark.parametrize(
        "exc",
        [
            TypeError("'<' not supported between instances of 'str' and 'int'"),
            FileNotFoundError("meta/info.json"),
        ],
        ids=["TypeError", "FileNotFoundError"],
    )
    def test_a_library_failure_is_collected_as_the_verdict(
        self, monkeypatch: pytest.MonkeyPatch, dataset: pathlib.Path, exc: Exception
    ) -> None:
        """Dropping the catch-all would let these escape instead of being recorded."""
        monkeypatch.setattr(f"{__name__}.build_train_command", _raising_tool(exc))
        assert _tool_verdict(dataset, "lora_r", 8) == f"raised {type(exc).__name__}: {exc}"

    @pytest.mark.parametrize(
        "exc",
        [
            KeyboardInterrupt(),
            SystemExit(1),
            pytest.skip.Exception("peft is not installed"),
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
            _tool_verdict(dataset, "lora_r", 8)


class TestABackendThatIgnoresTheFieldsReportsNothing:
    """A backend must not report on fields it never reads.

    :class:`TrainSpec` documents that a backend "reads the fields it supports and
    ignores the rest", so this gate is scoped to the LeRobot backend rather than
    made universal like the learning-rate one.
    """

    @pytest.mark.parametrize("trainer_cls", (MockTrainer, Gr00tTrainer, Cosmos3Trainer))
    @pytest.mark.parametrize("field", FIELDS)
    def test_it_validates_nothing_about_the_adapter(
        self, spec: TrainSpec, trainer_cls: type[Trainer], field: str
    ) -> None:
        setattr(spec, field, 0)
        assert _problems_of(trainer_cls(), spec, field) == []


def _trainer_modules() -> list[pathlib.Path]:
    """Every trainer module, minus the one that defines the shared gate.

    Rooted at the module that defines :class:`Trainer` so the scan cannot
    silently point at the wrong tree. The module that *defines* the gate is
    excluded - derived from the gate itself rather than named, so the exclusion
    cannot drift - because it reads the fields as their owner, not as a consumer.
    """
    root = pathlib.Path(inspect.getfile(Trainer)).parent
    owner = pathlib.Path(inspect.getfile(lora_hyperparameter_problems)).resolve()
    return sorted(p for p in root.rglob("*.py") if p.name != "__init__.py" and p.resolve() != owner)


def _reads_a_hyperparameter(source: str) -> bool:
    """Does *source* read either field, by name or through a forwarding table?

    Delegated to the shared rule so this guard and its siblings cannot disagree
    about what counts as a read - a transport-only provider reads every field it
    forwards through ``getattr(spec, field)`` and names none of them in an
    attribute access.
    """
    return reads_spec_field(source, FIELDS)


def _calls_the_gate(source: str) -> bool:
    """Does *source* route through the shared gate?"""
    return any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "_lora_hyperparameter_problems"
        for node in ast.walk(ast.parse(source))
    )


class TestOneOwnerForTheAdapterHyperparameterDomain:
    """No backend may skip the domain, and none may re-implement it.

    The set of backends in scope is derived from the tree rather than listed: a
    module that *reads* either field must route it through the shared gate, so a
    third backend that starts building a LoRA adapter fails this test until it
    does.
    """

    def test_the_scan_finds_the_backend_that_reads_the_fields(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        readers = {p.name for p in _trainer_modules() if _reads_a_hyperparameter(p.read_text())}
        assert readers == {"lerobot.py", "sagemaker.py"}

    def test_every_backend_that_reads_them_routes_through_the_shared_gate(self) -> None:
        adrift = sorted(
            p.name
            for p in _trainer_modules()
            if _reads_a_hyperparameter(source := p.read_text()) and not _calls_the_gate(source)
        )
        assert adrift == [], f"modules reading a LoRA hyperparameter without the shared gate: {adrift}"

    def test_no_backend_re_implements_the_domain(self) -> None:
        """A local sign or type test on either field is the hole this closed."""
        offenders: list[str] = []
        for path in _trainer_modules():
            for line in path.read_text().splitlines():
                if any(f"spec.{f}" in line for f in FIELDS) and ("< 1" in line or "<= 0" in line or "int(" in line):
                    offenders.append(f"{path.name}: {line.strip()}")
        assert offenders == [], f"local domain checks on a LoRA hyperparameter: {offenders}"

    def test_the_scanners_detect_a_planted_defect(self) -> None:
        """A scanner that silently matched nothing would look like a clean tree."""
        planted = "def validate(self, spec):\n    return [] if spec.lora_alpha > 0 else ['bad']\n"
        assert _reads_a_hyperparameter(planted)
        assert not _calls_the_gate(planted)

    def test_the_scanners_detect_a_table_driven_defect(self) -> None:
        """A backend that forwards either field by name is a reader too.

        The form a transport-only provider takes: no attribute access mentions
        either field, so a scan keyed on ``spec.lora_r`` alone reports a clean
        sweep while this backend skips the gate.
        """
        planted = 'F = ("lora_alpha",)\ndef validate(self, spec):\n    return [getattr(spec, f) for f in F]\n'
        assert _reads_a_hyperparameter(planted)
        assert not _calls_the_gate(planted)


def _peft_flag_writers() -> list[pathlib.Path]:
    """Every module that writes a ``--peft.r`` / ``--peft.lora_alpha`` token."""
    root = pathlib.Path(inspect.getfile(Trainer)).parents[1]
    return sorted(
        p for p in root.rglob("*.py") if "--peft.r=" in (source := p.read_text()) or "--peft.lora_alpha=" in source
    )


class TestEveryWriterOfThePeftFlagsValidatesFirst:
    """A second writer of the same two flags must not judge nothing.

    Derived from the tree rather than listed, so a third module that starts
    emitting either flag fails this until it applies the domain too.
    """

    def test_the_scan_finds_both_known_writers(self) -> None:
        """Non-vacuity: a mis-rooted scan cannot report a clean sweep of nothing."""
        assert {p.name for p in _peft_flag_writers()} == {"lerobot.py", "lerobot_train.py"}

    def test_each_writer_applies_the_shared_count_domain(self) -> None:
        adrift = sorted(
            p.name
            for p in _peft_flag_writers()
            if not ("_lora_hyperparameter_problems" in (source := p.read_text()) or "positive_count_error" in source)
        )
        assert adrift == [], f"modules writing a --peft flag without the shared domain: {adrift}"
