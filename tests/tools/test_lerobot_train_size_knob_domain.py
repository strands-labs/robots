"""``lerobot_train`` refuses a run size it cannot honor, before launching it.

``build_train_command`` writes ``--steps`` and ``--batch_size`` into the argv of
a process the tool then launches DETACHED, so an unusable value is not reported
to the caller at all: the tool returns a pid and only the training log, minutes
later, records that the run could not proceed. Two of those values are worse
than late - they are silent:

* ``steps=0`` and ``steps=-5`` parse as valid ints and make lerobot's training
  loop (``for _ in range(step, cfg.steps)``) EMPTY, so the run completes having
  trained nothing.
* ``steps=True`` / ``batch_size=True`` parse as ``1``, so a boolean silently
  becomes a one-step / batch-of-one run.

The trainer surface for the same lerobot run already refused a non-positive
``steps`` (``LerobotTrainer.validate`` reports "steps must be > 0"), so one
parameter had two contracts depending on which surface built the flag; and
``build_train_command`` already refused ``num_gpus < 1``, a non-positive
``val_episodes`` and two mutually-exclusive tuning strategies, leaving the batch
size as the one knob sizing the run with no domain at all.

These tests pin the corrected contract on the shared positive-count domain, that
the refusal reaches the caller as an error envelope before any process starts,
the premises the domain rests on, and that the scope is no wider than the defect
(``save_freq`` is deliberately untouched - lerobot documents a non-positive value
there as "disables periodic saving").
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod
from tests.tools.test_lerobot_train import _FakeProc, _write_dataset

build_train_command = train_mod.build_train_command

# Values no lerobot run can be sized by, and why each one matters.
UNUSABLE_COUNTS: tuple[Any, ...] = (
    0,  # an empty training loop / a DataLoader torch refuses
    -5,  # likewise, and reported by the trainer surface as "must be > 0"
    True,  # int subclass: parses as 1, so it silently becomes a run of one
    False,  # parses as 0
    2.7,  # lerobot's int field cannot decode it
    2.0,  # integral but not an int - same decoding failure
    float("nan"),
    float("inf"),
    "20000",  # the trainer surface leaks a bare TypeError on this one
    [8],
    object(),  # a value with no numeric meaning at all
)

SIZE_PARAMS = ("steps", "batch_size")


def _build(**kwargs: Any) -> list[str]:
    """Build an argv with a minimal usable base, overridden by ``kwargs``.

    Funnelled so the deliberately off-type values below reach the runtime guard
    the way an agent supplies them, without a type checker objecting at each
    call site.
    """
    base: dict[str, Any] = {"dataset_root": "/data/cubes", "policy_type": "act"}
    base.update(kwargs)
    return build_train_command(**base)


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the on-disk session store inside the test's own tmp_path."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class TestARunSizeThatCannotBeHonoredIsRefused:
    """Each size knob is checked against the shared positive-count domain."""

    @pytest.mark.parametrize("param", SIZE_PARAMS)
    @pytest.mark.parametrize("value", UNUSABLE_COUNTS)
    def test_an_unusable_size_never_reaches_the_argv(self, param: str, value: Any) -> None:
        with pytest.raises(ValueError, match=f"{param} must be a positive integer"):
            _build(**{param: value})

    @pytest.mark.parametrize("param", SIZE_PARAMS)
    def test_the_refusal_names_the_parameter_and_quotes_the_value(self, param: str) -> None:
        with pytest.raises(ValueError) as excinfo:
            _build(**{param: -5})
        message = str(excinfo.value)
        assert param in message
        assert "-5" in message
        assert "lerobot_train" in message

    @pytest.mark.parametrize(("param", "value"), [("steps", 20000), ("batch_size", 8)])
    def test_a_usable_size_still_reaches_the_argv(self, param: str, value: int) -> None:
        assert f"--{param}={value}" in _build(**{param: value})

    @pytest.mark.parametrize("param", SIZE_PARAMS)
    def test_none_omits_the_flag_and_keeps_lerobots_own_default(self, param: str) -> None:
        """``None`` means "do not pass the flag" and must stay accepted.

        Omitting the flag leaves lerobot's own documented default in place, which
        is a working call today; the guard checks only a supplied value.
        """
        cmd = _build(**{param: None})
        assert not [flag for flag in cmd if flag.startswith(f"--{param}=")]

    def test_both_knobs_are_checked_not_just_the_first(self) -> None:
        """A usable ``steps`` must not mask an unusable ``batch_size``."""
        with pytest.raises(ValueError, match="batch_size must be a positive integer"):
            _build(steps=20000, batch_size=0)


class TestTheBuilderNowMeetsTheTrainerSurfaceContract:
    """The two surfaces that size the same lerobot run cannot disagree.

    ``LerobotTrainer.validate`` already refused a non-positive ``steps``. Any
    value it will not accept must not be turned into a ``--steps`` flag by the
    builder either, or the accepted domain depends on which surface the caller
    reached for.
    """

    @staticmethod
    def _trainer_accepts_steps(value: Any) -> bool:
        from strands_robots.training import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        spec = TrainSpec(dataset_root="/data/cubes", output_dir="/tmp/out", steps=value)
        try:
            problems = LerobotTrainer().validate(spec)
        except TypeError:
            # The hand-rolled comparison cannot even order this value.
            return False
        return not any("steps" in problem for problem in problems)

    @staticmethod
    def _builder_accepts_steps(value: Any) -> bool:
        try:
            _build(steps=value)
        except ValueError:
            return False
        return True

    @pytest.mark.parametrize("value", [*UNUSABLE_COUNTS, 20000, 1])
    def test_the_builder_accepts_no_step_budget_the_trainer_would_reject(self, value: Any) -> None:
        if not self._trainer_accepts_steps(value):
            assert not self._builder_accepts_steps(value), (
                f"steps={value!r} is refused by the trainer surface but built into the argv"
            )

    def test_the_trainer_surface_really_does_refuse_a_non_positive_step_budget(self) -> None:
        """Non-vacuity: the contract being matched is actually enforced there."""
        assert not self._trainer_accepts_steps(0)
        assert not self._trainer_accepts_steps(-5)
        assert self._trainer_accepts_steps(20000)


class TestTheRefusalReachesTheCallerBeforeAnyProcessStarts:
    """A rejected size must be reported, not launched and then discovered."""

    def test_the_tool_reports_an_error_envelope_rather_than_raising(self, tmp_path: Path) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        result = train_mod.lerobot_train(action="start", dataset_root=str(dataset), steps=0)
        assert result["status"] == "error"
        text = "\n".join(item["text"] for item in result["content"] if "text" in item)
        assert "steps must be a positive integer" in text

    def test_a_refused_size_launches_no_process(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        dataset = _write_dataset(tmp_path / "cubes")
        launched: list[Any] = []

        def _fail_if_launched(*args: Any, **kwargs: Any) -> _FakeProc:
            launched.append(args)
            return _FakeProc()

        monkeypatch.setattr(train_mod.subprocess, "Popen", _fail_if_launched)
        result = train_mod.lerobot_train(action="start", dataset_root=str(dataset), batch_size=-8)
        assert result["status"] == "error"
        assert launched == [], "a refused run size still spawned a training process"


class TestThePremisesTheDomainRestsOn:
    """Executable premises, so the reasoning cannot silently become wrong."""

    @pytest.mark.parametrize("steps", [0, -5])
    def test_a_non_positive_step_budget_is_an_empty_training_loop(self, steps: int) -> None:
        """lerobot iterates ``range(step, cfg.steps)``; that is empty here."""
        assert list(range(0, steps)) == []

    def test_a_boolean_step_budget_would_have_been_a_single_step(self) -> None:
        """``True`` is an int subclass, so it sizes the loop at exactly one."""
        assert list(range(0, True)) == [0]

    @pytest.mark.parametrize("batch_size", [0, -8])
    def test_a_non_positive_batch_size_cannot_build_a_dataloader(self, batch_size: int) -> None:
        torch = pytest.importorskip("torch")
        dataset = torch.utils.data.TensorDataset(torch.zeros(4, 2))
        with pytest.raises(ValueError, match="batch_size should be a positive integer"):
            torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    def test_lerobot_declares_both_knobs_as_plain_int_fields(self) -> None:
        """A fractional value cannot be decoded into either field."""
        import dataclasses

        train_config = pytest.importorskip("lerobot.configs.train")
        declared = {field.name: field.type for field in dataclasses.fields(train_config.TrainPipelineConfig)}
        assert declared["steps"] is int
        assert declared["batch_size"] is int


class TestTheScopeIsNoWiderThanTheDefect:
    """``save_freq`` is a mode selector for lerobot, not a run size."""

    @pytest.mark.parametrize("save_freq", [0, -1])
    def test_a_non_positive_save_freq_is_still_accepted(self, save_freq: int) -> None:
        """lerobot documents it as "disables periodic saving".

        ``should_save_checkpoint`` treats a non-positive ``save_freq`` as "write
        only the final checkpoint", so refusing it here would refuse a documented
        capability rather than an unusable value.
        """
        assert f"--save_freq={save_freq}" in _build(save_freq=save_freq)

    def test_lerobot_really_honors_a_non_positive_save_freq(self) -> None:
        """Non-vacuity for the scope decision above."""
        lerobot_train_script = pytest.importorskip("lerobot.scripts.lerobot_train")
        should_save = lerobot_train_script.should_save_checkpoint
        assert should_save(50, 0, 100) is False, "no periodic checkpoint"
        assert should_save(100, 0, 100) is True, "but the final one is still written"

    def test_the_other_numeric_knobs_keep_their_own_guards(self) -> None:
        """The guards that already existed are untouched."""
        with pytest.raises(ValueError, match="num_gpus must be >= 1"):
            _build(num_gpus=0)
        with pytest.raises(ValueError, match="val_episodes must be a positive integer"):
            _build(val_episodes=0)


def test_nan_and_inf_are_covered_by_the_probe_set() -> None:
    """Guards the probe set itself against a future edit dropping a case."""
    floats = [value for value in UNUSABLE_COUNTS if isinstance(value, float)]
    assert any(math.isnan(value) for value in floats)
    assert any(math.isinf(value) for value in floats)
