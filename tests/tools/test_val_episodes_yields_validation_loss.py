"""``val_episodes`` must produce a validation signal, not just a smaller train set.

lerobot expresses a held-out validation split as ``dataset.eval_split`` (a
per-task fraction) and evaluates it every ``eval_steps`` training steps. The two
are coupled in lerobot's own ``TrainPipelineConfig.validate``: an ``eval_steps``
with no ``eval_split`` to draw held-out data from is a hard error, and an
``eval_split`` with ``eval_steps == 0`` builds the split but never evaluates it.

So a surface that reserves episodes by restricting ``dataset.episodes`` shrinks
the TRAINING set and produces no validation loss at all, and one that emits
``eval_steps`` alone refuses to launch. These tests pin that every surface taking
``val_episodes`` emits the pair, that the fraction reproduces the requested
episode COUNT exactly, and that a count which cannot be honored is refused
rather than silently rounded.
"""

import dataclasses
import inspect
import json
import math
import re
from pathlib import Path
from typing import Any

import pytest

from strands_robots.tools.lerobot_train import build_train_command
from strands_robots.utils import validation_split_error, validation_split_fraction


def _write_dataset(root: Path, total_episodes: int = 10, total_tasks: int = 1) -> Path:
    """Minimal LeRobot v3 dataset stub carrying episode and task counts."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": total_episodes, "total_tasks": total_tasks}))
    return root


def _flag(cmd: list[str], name: str) -> str | None:
    """The single value emitted for ``--name=``, or None when absent."""
    hits = [c.split("=", 1)[1] for c in cmd if c.startswith(f"--{name}=")]
    assert len(hits) <= 1, f"--{name} emitted {len(hits)} times: {hits}"
    return hits[0] if hits else None


class TestFractionReproducesRequestedCount:
    """The emitted fraction must hold out exactly the requested episode count."""

    # (total, N) pairs where the obvious ``N / total`` fraction is NOT float-safe:
    # 25 * (7 / 25) == 7.000000000000001, whose ceiling is 8 - one episode more
    # than asked. These are the cases a boundary fraction gets wrong.
    @pytest.mark.parametrize(("total", "n"), [(25, 7), (25, 14), (29, 15), (35, 29), (38, 21), (41, 7)])
    def test_boundary_unsafe_pairs_still_reserve_exactly_n(self, total: int, n: int) -> None:
        assert math.ceil(total * (n / total)) != n, "fixture no longer exercises the float hazard"
        assert math.ceil(total * validation_split_fraction(n, total)) == n

    @pytest.mark.parametrize("total", [2, 3, 10, 50, 206, 397])
    def test_every_count_reserves_exactly_n(self, total: int) -> None:
        """Sweep every reservable count, not just the hand-picked hazards."""
        for n in range(1, total):
            assert math.ceil(total * validation_split_fraction(n, total)) == n, (total, n)

    def test_command_fraction_reserves_exactly_n(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=5)
        assert math.ceil(50 * float(_flag(cmd, "dataset.eval_split") or "0")) == 5


class TestEmittedFlagsCanProduceAValidationLoss:
    """Both halves of lerobot's coupled pair must be emitted together."""

    def test_val_episodes_emits_split_and_a_nonzero_eval_cadence(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", save_freq=250, val_episodes=2)
        assert float(_flag(cmd, "dataset.eval_split") or "0") > 0.0
        assert int(_flag(cmd, "eval_steps") or "0") == 250

    def test_eval_cadence_is_never_emitted_without_a_split_to_evaluate(self, tmp_path: Path) -> None:
        """lerobot rejects eval_steps > 0 when eval_split is 0.0, so never pair them that way."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        # Annotated so mypy does not narrow the splat to the inferred value type.
        cases: list[dict[str, Any]] = [{}, {"val_episodes": 3}]
        for kwargs in cases:
            cmd = build_train_command(dataset_root=str(root), policy_type="act", **kwargs)
            steps = int(_flag(cmd, "eval_steps") or "0")
            split = float(_flag(cmd, "dataset.eval_split") or "0")
            assert (steps > 0) == (split > 0.0), f"{kwargs}: eval_steps={steps} eval_split={split}"

    def test_no_val_episodes_leaves_evaluation_untouched(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act")
        assert _flag(cmd, "dataset.eval_split") is None
        assert _flag(cmd, "eval_steps") is None

    def test_disabled_periodic_saving_still_evaluates_once(self, tmp_path: Path) -> None:
        """save_freq <= 0 disables checkpointing, so fall back to a pass at the end."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", steps=4000, save_freq=0, val_episodes=2)
        assert int(_flag(cmd, "eval_steps") or "0") == 4000

    def test_reserved_episodes_are_not_also_cut_from_training_by_hand(self, tmp_path: Path) -> None:
        """A hand-rolled --dataset.episodes restriction would hide the split from lerobot."""
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2)
        assert _flag(cmd, "dataset.episodes") is None


class TestCountThatCannotBeHonoredIsRefused:
    """A per-task fraction cannot express a global count on a multi-task dataset."""

    def test_multi_task_dataset_is_refused_with_the_fraction_to_use(self, tmp_path: Path) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10, total_tasks=3)
        with pytest.raises(ValueError, match="cannot be reserved exactly"):
            build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2)

    @pytest.mark.parametrize("tasks", [0, 1, None, True, "2"])
    def test_absent_or_single_task_count_is_honored(self, tasks: object) -> None:
        """0 / None mean 'no task count recorded', which lerobot treats as one task."""
        assert validation_split_error(2, tasks, "ctx", passthrough_param="extra_flags") is None

    def test_refusal_names_the_task_count_and_the_direct_knobs(self) -> None:
        err = validation_split_error(1, 4, "lerobot_train", passthrough_param="extra_flags")
        assert err is not None
        assert err.startswith("lerobot_train: ")
        assert "4 tasks" in err
        assert "dataset.eval_split" in err and "eval_steps" in err


class TestCallerOverridesTakePrecedence:
    """extra_flags is the documented escape hatch; it must not be duplicated."""

    @pytest.mark.parametrize("key", ["eval_steps", "dataset.eval_split"])
    def test_explicit_value_replaces_the_derived_one(self, tmp_path: Path, key: str) -> None:
        root = _write_dataset(tmp_path / "ds", total_episodes=10)
        cmd = build_train_command(dataset_root=str(root), policy_type="act", val_episodes=2, extra_flags={key: 7})
        # _flag asserts the flag appears at most once, so a duplicate fails here.
        assert _flag(cmd, key) == "7"


class TestEverySurfaceAgrees:
    """The tool and the TrainSpec backend must not drift on one knob."""

    def test_trainspec_path_emits_the_same_pair(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        spec = TrainSpec(dataset_root=str(root), output_dir=str(tmp_path / "out"), val_episodes=5, save_freq=250)
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert math.ceil(50 * float(_flag(cmd, "dataset.eval_split") or "0")) == 5
        assert int(_flag(cmd, "eval_steps") or "0") == 250
        assert _flag(cmd, "dataset.episodes") is None

    def test_trainspec_validate_refuses_a_multi_task_count(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=10, total_tasks=3)
        spec = TrainSpec(dataset_root=str(root), output_dir=str(tmp_path / "out"), val_episodes=2)
        assert any("cannot be reserved exactly" in p for p in LerobotTrainer().validate(spec))


def _spec(**kwargs: Any) -> Any:
    """A launchable :class:`TrainSpec`, overridden per case."""
    from strands_robots.training.base import TrainSpec

    return TrainSpec(output_dir="/tmp/strands-val-episodes-out", steps=10, save_freq=5, **kwargs)


def _passthrough_keyword(message: str) -> str:
    """The passthrough parameter a refusal tells the reader to use instead."""
    match = re.search(r"(\w+)=\{'dataset\.eval_split'", message)
    assert match is not None, f"refusal names no dataset.eval_split passthrough: {message}"
    return match.group(1)


class TestAnUnreadableEpisodeCountIsRefused:
    """A count that cannot become a fraction must be refused, not dropped.

    The split is ``ceil(episodes_in_task * eval_split)``, so turning an episode
    COUNT into lerobot's fraction needs the dataset's ``total_episodes``. That
    number is only in a local ``meta/info.json``, which a Hub source need not
    have: ``dataset_repo_id`` with no ``dataset_root``, or a ``dataset_root``
    that is a Hub cache directory nothing has been downloaded into yet. Emitting
    no fraction there is indistinguishable from "no validation was asked for" -
    the run trains on every episode and records no validation loss, which is the
    outcome this module exists to prevent.
    """

    def test_a_hub_source_with_no_local_root_is_refused(self) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        spec = _spec(dataset_repo_id="lerobot/svla_so101_pickplace", val_episodes=2)
        trainer = LerobotTrainer()
        problems = trainer.validate(spec)
        assert any("val_episodes=2" in p for p in problems), (
            "a Hub source drops val_episodes silently: validate() reported "
            f"{problems} while build_command() emits "
            f"{[c for c in trainer.build_command(spec) if 'eval' in c]}"
        )

    def test_a_hub_cache_root_with_nothing_downloaded_is_refused(self, tmp_path: Path) -> None:
        """The documented Hub-cache shape: repo id plus an empty local cache dir."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        cache = tmp_path / "hf-cache"
        cache.mkdir()
        spec = _spec(dataset_repo_id="lerobot/svla_so101_pickplace", dataset_root=str(cache), val_episodes=2)
        assert any("val_episodes=2" in p for p in LerobotTrainer().validate(spec))

    def test_the_refusal_is_load_bearing_so_no_run_launches_without_the_split(self) -> None:
        """``train`` fails closed on validate(), so the request cannot be dropped."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        result = LerobotTrainer().train(_spec(dataset_repo_id="lerobot/svla_so101_pickplace", val_episodes=2))
        assert result.status != "success"
        assert "val_episodes=2" in result.message


class TestARefusalNamesAPassthroughItsOwnSurfaceAccepts:
    """Every remedy pointing at raw flags must name a keyword that surface has.

    The passthrough is spelled ``extra_flags`` on the ``lerobot_train`` tool and
    ``extra`` on :class:`TrainSpec` (and the ``train_policy`` tool), so one
    hardcoded spelling makes the remedy a dead end on the other surface: applied
    verbatim it raises ``TypeError`` for an unexpected keyword.
    """

    def test_the_trainer_names_a_real_trainspec_field(self) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.base import TrainSpec
        from strands_robots.training.lerobot import LerobotTrainer

        problems = LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x", val_episodes=2))
        keyword = _passthrough_keyword(" ".join(problems))
        assert keyword in {f.name for f in dataclasses.fields(TrainSpec)}

    def test_the_tool_names_a_real_lerobot_train_parameter(self) -> None:
        from strands_robots.tools.lerobot_train import lerobot_train

        err = validation_split_error(1, 4, "lerobot_train", passthrough_param="extra_flags")
        assert err is not None
        target = getattr(lerobot_train, "__wrapped__", lerobot_train)
        assert _passthrough_keyword(err) in inspect.signature(target).parameters

    def test_applying_the_trainer_remedy_verbatim_produces_the_pair(self) -> None:
        """Parse the remedy out of the refusal, apply it, and it must work."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        trainer = LerobotTrainer()
        keyword = _passthrough_keyword(" ".join(trainer.validate(_spec(dataset_repo_id="lerobot/x", val_episodes=2))))
        remedied = _spec(dataset_repo_id="lerobot/x", **{keyword: {"dataset.eval_split": 0.1, "eval_steps": 5}})
        assert trainer.validate(remedied) == []
        cmd = trainer.build_command(remedied)
        assert _flag(cmd, "dataset.eval_split") == "0.1"
        assert _flag(cmd, "eval_steps") == "5"

    def test_pointing_dataset_root_at_a_local_copy_is_the_other_named_remedy(self, tmp_path: Path) -> None:
        """The refusal's first remedy: a populated cache root makes the count readable."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "cache", total_episodes=10)
        spec = _spec(dataset_repo_id="lerobot/x", dataset_root=str(root), val_episodes=2)
        trainer = LerobotTrainer()
        assert trainer.validate(spec) == []
        assert math.ceil(10 * float(_flag(trainer.build_command(spec), "dataset.eval_split") or "0")) == 2


class TestTheRefusalDoesNotReachPastTheDefect:
    """Controls: only an unhonorable request may be refused."""

    def test_a_local_root_still_emits_the_pair(self, tmp_path: Path) -> None:
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        root = _write_dataset(tmp_path / "ds", total_episodes=50)
        spec = _spec(dataset_root=str(root), val_episodes=5)
        trainer = LerobotTrainer()
        assert trainer.validate(spec) == []
        assert math.ceil(50 * float(_flag(trainer.build_command(spec), "dataset.eval_split") or "0")) == 5

    def test_a_hub_source_without_val_episodes_is_not_refused(self) -> None:
        """None is the documented "train on everything" sentinel, not a problem."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        assert LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x")) == []

    def test_an_unusable_count_still_names_the_count_domain_first(self) -> None:
        """A non-positive count is a domain problem wherever the data lives."""
        pytest.importorskip("lerobot")
        from strands_robots.training.lerobot import LerobotTrainer

        problems = LerobotTrainer().validate(_spec(dataset_repo_id="lerobot/x", val_episodes=0))
        assert any("must be a positive integer" in p for p in problems)
