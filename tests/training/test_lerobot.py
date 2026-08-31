"""Tests for LerobotTrainer: factory wiring, validate, and command building.

These are pure/offline (no GPU, no actual lerobot_train launch). The real
end-to-end sim->train->load is exercised separately.
"""

import ast
import inspect
import json
import pathlib
import sys
from types import SimpleNamespace

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.lerobot import LerobotTrainer


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="",
        output_dir=str(tmp_path / "out"),
        steps=200,
        global_batch_size=8,
        save_freq=100,
        extra={"policy_type": "act"},
    )


class TestFactoryWiring:
    def test_resolves_from_registry(self):
        t = create_trainer("lerobot_local")
        assert isinstance(t, LerobotTrainer)
        assert t.provider_name == "lerobot_local"

    def test_alias_resolves(self):
        # 'lerobot' is a policies.json alias of lerobot_local
        t = create_trainer("lerobot")
        assert isinstance(t, LerobotTrainer)


class TestValidate:
    def test_clean(self, spec):
        assert LerobotTrainer().validate(spec) == []

    @pytest.mark.parametrize("mode", ["absent", "probe_raises"])
    def test_flags_missing_lerobot_install(self, spec, monkeypatch, mode):
        """validate() must surface an actionable problem when lerobot itself is
        not importable, rather than deferring the failure to the training launch.

        Both an absent module (find_spec returns None) and a broken probe
        (find_spec raises) resolve to the same actionable message.
        """
        import importlib.util as importlib_util

        real_find_spec = importlib_util.find_spec

        def fake_find_spec(name, *args, **kwargs):
            if name == "lerobot.scripts.lerobot_train":
                if mode == "probe_raises":
                    raise ImportError("boom")
                return None
            return real_find_spec(name, *args, **kwargs)

        monkeypatch.setattr(importlib_util, "find_spec", fake_find_spec)
        problems = LerobotTrainer().validate(spec)
        assert any("lerobot is not installed" in problem for problem in problems)

    def test_non_native_policy_type(self, spec):
        spec.extra["policy_type"] = "openvla"
        problems = LerobotTrainer().validate(spec)
        assert any("not LeRobot-native" in p for p in problems)

    def test_lora_expert_clash(self, spec):
        spec.method = "lora"
        spec.tune = {"expert_only": True}
        problems = LerobotTrainer().validate(spec)
        assert any("mutually exclusive" in p for p in problems)

    def test_val_episodes_too_large(self, spec):
        spec.val_episodes = 99  # total is 10
        problems = LerobotTrainer().validate(spec)
        assert any("val_episodes" in p for p in problems)


_AUGMENT_HINT = "augment_dataset_quantile_stats"


def _write_stats(dataset_root, *, with_quantiles):
    """Write a minimal v3 meta/stats.json with or without quantile keys."""
    import os

    stats = {
        "observation.state": {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]},
        "action": {"mean": [0.0], "std": [1.0], "min": [-1.0], "max": [1.0]},
    }
    if with_quantiles:
        for feat in stats.values():
            feat["q01"] = [-0.9]
            feat["q99"] = [0.9]
    meta = os.path.join(dataset_root, "meta")
    with open(os.path.join(meta, "stats.json"), "w", encoding="utf-8") as fh:
        json.dump(stats, fh)


class TestQuantileStatsPreflight:
    """A QUANTILES-normalizing policy (molmoact2, pi05) needs the dataset stats
    to carry quantile keys; validate() must flag a definite miss at spec time
    instead of letting it fail deep inside lerobot's normalization.
    """

    def test_quantile_policy_missing_stats_flagged(self, spec):
        spec.extra["policy_type"] = "molmoact2"
        _write_stats(spec.dataset_root, with_quantiles=False)
        problems = LerobotTrainer().validate(spec)
        offending = [p for p in problems if _AUGMENT_HINT in p]
        assert offending, f"expected augment-script guidance, got {problems}"
        assert "QUANTILES" in offending[0]
        assert spec.dataset_root in offending[0]

    def test_quantile_policy_with_stats_is_clean(self, spec):
        spec.extra["policy_type"] = "molmoact2"
        _write_stats(spec.dataset_root, with_quantiles=True)
        problems = LerobotTrainer().validate(spec)
        assert not any(_AUGMENT_HINT in p for p in problems), problems

    def test_pi05_missing_stats_flagged(self, spec):
        spec.extra["policy_type"] = "pi05"
        _write_stats(spec.dataset_root, with_quantiles=False)
        problems = LerobotTrainer().validate(spec)
        assert any(_AUGMENT_HINT in p for p in problems), problems

    def test_non_quantile_policy_not_flagged(self, spec):
        # act normalizes with MEAN_STD/MIN_MAX; a pre-quantile dataset is fine.
        spec.extra["policy_type"] = "act"
        _write_stats(spec.dataset_root, with_quantiles=False)
        problems = LerobotTrainer().validate(spec)
        assert not any(_AUGMENT_HINT in p for p in problems), problems

    def test_no_local_stats_json_not_flagged(self, spec):
        # meta/stats.json absent -> unknown (e.g. Hub dataset) -> no false positive.
        spec.extra["policy_type"] = "molmoact2"
        problems = LerobotTrainer().validate(spec)
        assert not any(_AUGMENT_HINT in p for p in problems), problems


class TestQuantileHelpers:
    def test_policy_uses_quantile_norm_live_registry(self):
        from strands_robots.training.lerobot import _policy_uses_quantile_norm

        assert _policy_uses_quantile_norm("molmoact2") is True
        assert _policy_uses_quantile_norm("pi05") is True
        assert _policy_uses_quantile_norm("act") is False

    def test_stats_have_quantiles(self):
        from strands_robots.training.lerobot import _stats_have_quantiles

        assert _stats_have_quantiles({"action": {"mean": [0.0], "q01": [-1.0]}}) is True
        assert _stats_have_quantiles({"action": {"mean": [0.0], "std": [1.0]}}) is False
        assert _stats_have_quantiles(None) is False

    def test_dataset_quantile_stats_present_tristate(self, tmp_path):
        import os

        from strands_robots.training.lerobot import _dataset_quantile_stats_present

        root = str(tmp_path)
        assert _dataset_quantile_stats_present(root) is None  # no stats.json
        os.makedirs(os.path.join(root, "meta"))
        _write_stats(root, with_quantiles=False)
        assert _dataset_quantile_stats_present(root) is False
        _write_stats(root, with_quantiles=True)
        assert _dataset_quantile_stats_present(root) is True


class TestBuildCommand:
    def test_single_gpu_core_flags(self, spec):
        spec.base_model = "lerobot/act_aloha_sim"  # argv-parity only; no load
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        # build_command is now a PURE argv-parity helper (no launcher prefix);
        # the module path is the first token.
        assert cmd[0] == "lerobot.scripts.lerobot_train"
        assert "--dataset.repo_id=local" in cmd
        assert f"--dataset.root={spec.dataset_root}" in cmd
        assert "--policy.type=act" in cmd
        assert "--policy.device=cpu" in cmd
        assert "--policy.push_to_hub=false" in cmd
        assert "--steps=200" in cmd
        assert "--batch_size=8" in cmd
        assert "--save_freq=100" in cmd
        assert "--wandb.enable=false" in cmd
        assert "--policy.pretrained_path=lerobot/act_aloha_sim" in cmd

    def test_build_command_is_launcher_free(self, spec):
        # build_command is parity-only: it never prepends accelerate/torchrun/
        # python. Multi-GPU is driven by elastic_launch in train(), not here.
        spec.num_gpus = 4
        cmd = LerobotTrainer(device="cuda").build_command(spec)
        assert cmd[0] == "lerobot.scripts.lerobot_train"
        assert "accelerate" not in cmd
        assert "torchrun" not in cmd
        assert "python" not in cmd

    def test_lora_flags(self, spec):
        spec.method = "lora"
        spec.lora_r = 16
        spec.lora_alpha = 32
        spec.lora_target_modules = "q_proj,v_proj"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--peft.method_type=LORA" in cmd
        assert "--peft.r=16" in cmd
        # lora_alpha (the LoRA scaling numerator; scaling = lora_alpha / r) is a
        # real PeftConfig field that build_config wires; the argv-parity helper
        # must emit it too or the documented "equivalent CLI" trains with a
        # different LoRA scale than the in-process train(cfg).
        assert "--peft.lora_alpha=32" in cmd
        assert "--peft.target_modules=q_proj,v_proj" in cmd

    def test_lora_alpha_omitted_when_unset(self, spec):
        # Only emitted when requested, mirroring --peft.r / --peft.target_modules.
        spec.method = "lora"
        spec.lora_r = 16
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert not any(c.startswith("--peft.lora_alpha") for c in cmd)

    def test_expert_only_flag(self, spec):
        spec.method = "expert_only"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--policy.train_expert_only=true" in cmd

    def test_val_split_emits_an_evaluated_split_flag(self, spec):
        """The reserved tail must be evaluated, not merely withheld from training.

        This previously asserted ``--dataset.episodes=[0..7]``; lerobot draws its
        eval dataloader from ``dataset.eval_split``, so an episode restriction
        left the reserved episodes unused and no eval loss was ever computed.
        """
        spec.val_episodes = 2  # total 10 -> ceil(10 * 0.15) == 2 reserved
        spec.save_freq = 400
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--dataset.eval_split=0.15" in cmd
        assert "--eval_steps=400" in cmd
        assert not [c for c in cmd if c.startswith("--dataset.episodes=")]

    def test_seed_and_jobname_and_passthrough(self, spec):
        spec.seed = 42
        spec.extra["job_name"] = "my_run"
        spec.extra["num_workers"] = 4  # arbitrary passthrough
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--seed=42" in cmd
        assert "--job_name=my_run" in cmd
        assert "--num_workers=4" in cmd
        # consumed keys must NOT leak as flags
        assert not any(c.startswith("--policy_type=") for c in cmd)
        assert not any(c.startswith("--job_name=strands_ft") for c in cmd)


class TestBuildConfig:
    """build_config() yields lerobot's typed TrainPipelineConfig (the real lib path)."""

    def test_builds_typed_config(self, spec):
        pytest.importorskip("lerobot")
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.repo_id == "local"
        assert cfg.dataset.root == spec.dataset_root
        assert cfg.policy.type == "act"
        assert cfg.policy.device == "cpu"
        assert cfg.policy.push_to_hub is False
        assert cfg.policy.pretrained_path is None
        assert cfg.steps == 200
        assert cfg.batch_size == 8
        assert cfg.save_freq == 100
        assert cfg.wandb.enable is False
        assert cfg.peft is None

    def test_lora_builds_peft(self, spec):
        pytest.importorskip("lerobot")
        spec.method = "lora"
        spec.lora_r = 16
        spec.lora_target_modules = "q_proj,v_proj"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.peft is not None
        assert cfg.peft.method_type == "LORA"
        assert cfg.peft.r == 16
        assert cfg.peft.target_modules == "q_proj,v_proj"
        # strands must NOT pre-set use_peft: make_policy would then misread the
        # base checkpoint as a PEFT adapter repo. cfg.peft alone drives wrapping.
        assert cfg.policy.use_peft is False

    def test_val_split_sets_eval_split_and_a_nonzero_eval_cadence(self, spec):
        """In-process config must carry BOTH halves of lerobot's coupled pair.

        This previously asserted ``cfg.dataset.episodes == [0..7]``. lerobot's own
        validate() rejects ``eval_steps > 0`` without ``eval_split > 0``, and an
        ``eval_split`` with ``eval_steps == 0`` is built but never evaluated - so
        only the pair yields a validation loss.
        """
        pytest.importorskip("lerobot")
        spec.val_episodes = 2  # total 10 -> ceil(10 * 0.15) == 2 reserved
        spec.save_freq = 400
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.episodes is None
        assert cfg.dataset.eval_split == pytest.approx(0.15)
        assert cfg.eval_steps == 400


class TestParseLog:
    """_parse_log against lerobot's real MetricsTracker line format."""

    def test_expand_big_number(self):
        from strands_robots.training.lerobot import _expand_big_number

        assert _expand_big_number("1.2K") == 1200.0
        assert _expand_big_number("2") == 2.0
        assert _expand_big_number("3M") == 3_000_000.0
        assert _expand_big_number("1.5B") == 1.5e9
        assert _expand_big_number("nope") is None
        assert _expand_big_number("") is None

    def test_parses_real_metricstracker_line(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text(
            "INFO 2026-06-23 ot_train.py:419 Start offline training\n"
            "step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123\n"
            "step:1.3K smpl:5.0K ep:9 epch:2.10 loss:0.087\n"
        )
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["latest_step"] == 1300  # newest, K-expanded
        assert abs(m["latest_loss"] - 0.087) < 1e-9
        assert m["latest_epoch"] == 2.10
        assert m["learning"] is True
        assert m["liveness_ok"] is True

    def test_plain_integer_step(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("step:2 smpl:4 ep:1 epch:1.00 loss:0.5\n")
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["latest_step"] == 2
        assert m["latest_loss"] == 0.5

    def test_parses_metric_line_with_logging_prefix(self, tmp_path):
        """A captured lerobot log prefixes the metric line with the logging
        formatter's level/timestamp/logger tokens; the parser must skip those
        colon-free tokens and still read the metrics. The MetricsTracker's bare
        ``str`` is emitted via ``logging.info``, so the line actually written to
        the log file carries this prefix - not the bare form.
        """
        log = tmp_path / "run.log"
        log.write_text(
            "INFO 2026-06-23 12:00:00 lerobot_train.py:210 "
            "step:1.2K smpl:4.9K ep:8 epch:2.00 loss:0.123 grad_norm:0.5\n"
        )
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["latest_step"] == 1200  # prefix tokens ignored, K-expanded
        assert abs(m["latest_loss"] - 0.123) < 1e-9
        assert m["latest_epoch"] == 2.00
        assert m["learning"] is True
        assert m["liveness_ok"] is True

    def test_no_metrics_line_means_not_live(self, tmp_path):
        log = tmp_path / "run.log"
        log.write_text("INFO booting...\nCreating dataset\n")
        m = LerobotTrainer(device="cpu")._parse_log(str(log))
        assert m["liveness_ok"] is False
        assert "latest_step" not in m

    def test_unreadable_log_returns_empty(self):
        assert LerobotTrainer(device="cpu")._parse_log("/no/such/log") == {}


class TestDatasetTotalEpisodes:
    """_dataset_total_episodes reads meta/info.json defensively."""

    def test_reads_total_episodes(self, dataset_root):
        assert LerobotTrainer()._dataset_total_episodes(dataset_root) == 10

    def test_missing_info_json_returns_none(self, tmp_path):
        assert LerobotTrainer()._dataset_total_episodes(str(tmp_path)) is None

    def test_malformed_info_json_returns_none(self, tmp_path):
        meta = tmp_path / "meta"
        meta.mkdir()
        (meta / "info.json").write_text("{not valid json")
        assert LerobotTrainer()._dataset_total_episodes(str(tmp_path)) is None


class TestCheckpointResolution:
    """_resume_config_path (FILE) and latest_checkpoint (DIR) walk the lerobot
    checkpoint layout ``<out>/checkpoints/<step|last>/pretrained_model/``."""

    def test_no_checkpoints_dir(self, tmp_path):
        out = str(tmp_path / "out")
        assert LerobotTrainer()._resume_config_path(out) is None
        assert LerobotTrainer().latest_checkpoint(out) is None

    def test_prefers_last_symlink_dir(self, tmp_path):
        out = tmp_path / "out"
        last = out / "checkpoints" / "last" / "pretrained_model"
        last.mkdir(parents=True)
        (last / "train_config.json").write_text("{}")
        cfg_file = LerobotTrainer()._resume_config_path(str(out))
        assert cfg_file == str(last / "train_config.json")
        # latest_checkpoint returns the loadable DIRECTORY (parent of the file)
        assert LerobotTrainer().latest_checkpoint(str(out)) == str(last)

    def test_falls_back_to_highest_numbered_step(self, tmp_path):
        out = tmp_path / "out"
        for step in ("000100", "000200"):
            pm = out / "checkpoints" / step / "pretrained_model"
            pm.mkdir(parents=True)
            (pm / "train_config.json").write_text("{}")
        # No "last" dir -> newest by sorted name wins (000200).
        cfg_file = LerobotTrainer()._resume_config_path(str(out))
        assert cfg_file.endswith("000200/pretrained_model/train_config.json")

    def test_checkpoints_dir_without_configs_returns_none(self, tmp_path):
        out = tmp_path / "out"
        (out / "checkpoints" / "000100").mkdir(parents=True)
        assert LerobotTrainer()._resume_config_path(str(out)) is None


class TestValidateAdditionalBranches:
    """Cover the remaining fail-closed validate() branches."""

    def test_missing_dataset_root(self, spec):
        spec.dataset_root = ""
        problems = LerobotTrainer().validate(spec)
        assert any("a data source is required" in p for p in problems)

    def test_dataset_root_not_v3(self, spec, tmp_path):
        spec.dataset_root = str(tmp_path / "empty")
        (tmp_path / "empty").mkdir()
        problems = LerobotTrainer().validate(spec)
        assert any("not a LeRobotDataset v3 root" in p for p in problems)

    def test_missing_output_dir(self, spec):
        spec.output_dir = ""
        problems = LerobotTrainer().validate(spec)
        assert any("output_dir is required" in p for p in problems)

    def test_unsupported_method(self, spec):
        spec.method = "frozen_backbone"
        problems = LerobotTrainer().validate(spec)
        assert any("unsupported method" in p for p in problems)

    def test_non_positive_steps(self, spec):
        spec.steps = 0
        problems = LerobotTrainer().validate(spec)
        assert any("steps must be a positive integer" in p for p in problems)

    def test_multinode_rejected(self, spec):
        spec.num_nodes = 2
        problems = LerobotTrainer().validate(spec)
        assert any("multi-node lerobot" in p for p in problems)


class TestBuildCommandResume:
    def test_resume_appends_config_path(self, spec, tmp_path):
        last = tmp_path / "out" / "checkpoints" / "last" / "pretrained_model"
        last.mkdir(parents=True)
        (last / "train_config.json").write_text("{}")
        spec.output_dir = str(tmp_path / "out")
        spec.resume = True
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--resume=true" in cmd
        assert any(c.startswith("--config_path=") for c in cmd)

    def test_resume_without_checkpoint_omits_flags(self, spec):
        spec.resume = True  # no checkpoint on disk
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--resume=true" not in cmd


class TestBuildConfigAdditionalBranches:
    def test_expert_only_sets_flag(self, spec):
        pytest.importorskip("lerobot")
        spec.method = "expert_only"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # ACT has no train_expert_only attr; build must not crash and peft stays None.
        assert cfg.peft is None

    def test_lora_options_passed_through_when_supported(self, spec):
        pytest.importorskip("lerobot")
        import dataclasses

        from lerobot.configs.default import PeftConfig

        supported = {f.name for f in dataclasses.fields(PeftConfig)}
        if "lora_alpha" not in supported:
            pytest.skip("installed lerobot PeftConfig has no lora_alpha field")
        spec.method = "lora"
        spec.lora_r = 8
        spec.lora_alpha = 32
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.peft.r == 8
        assert cfg.peft.lora_alpha == 32

    def test_lora_alpha_build_command_matches_build_config(self, spec):
        """build_command's argv-parity for --peft.lora_alpha must agree with the
        value build_config wires into cfg.peft.lora_alpha (drift guard, the
        contract build_command exists to uphold)."""
        pytest.importorskip("lerobot")
        import dataclasses

        from lerobot.configs.default import PeftConfig

        if "lora_alpha" not in {f.name for f in dataclasses.fields(PeftConfig)}:
            pytest.skip("installed lerobot PeftConfig has no lora_alpha field")
        spec.method = "lora"
        spec.lora_r = 8
        spec.lora_alpha = 32
        trainer = LerobotTrainer(device="cpu")
        cmd = trainer.build_command(spec)
        cfg = trainer.build_config(spec)
        emitted = [c for c in cmd if c.startswith("--peft.lora_alpha=")]
        assert emitted == [f"--peft.lora_alpha={cfg.peft.lora_alpha}"]

    def test_unsupported_lora_option_raises_actionable_error(self, spec, monkeypatch):
        """A LoRA option the installed PeftConfig rejects must raise a clear
        ValueError, not an opaque TypeError from the dataclass constructor.

        Older lerobot releases in the supported range (e.g. 0.5.1) lack the
        ``lora_alpha`` field, so forwarding it crashed build_config. Simulate
        that drift by stripping the field, independent of the installed version.
        """
        pytest.importorskip("lerobot")
        import dataclasses

        from lerobot.configs.default import PeftConfig

        kept = [f for f in dataclasses.fields(PeftConfig) if f.name != "lora_alpha"]

        class _LegacyPeftConfig:
            _names = {f.name for f in kept}

            def __init__(self, **kwargs):
                bad = set(kwargs) - self._names
                if bad:
                    raise TypeError(f"unexpected keyword argument {sorted(bad)}")
                for k, v in kwargs.items():
                    setattr(self, k, v)

        _LegacyPeftConfig.__dataclass_fields__ = {f.name: f for f in kept}
        monkeypatch.setattr("lerobot.configs.default.PeftConfig", _LegacyPeftConfig, raising=True)

        spec.method = "lora"
        spec.lora_r = 8
        spec.lora_alpha = 32
        with pytest.raises(ValueError, match="lora_alpha"):
            LerobotTrainer(device="cpu").build_config(spec)

    def test_seed_set_on_config(self, spec):
        pytest.importorskip("lerobot")
        spec.seed = 123
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.seed == 123

    def test_dotted_extra_passthrough_sets_field(self, spec):
        pytest.importorskip("lerobot")
        spec.extra["num_workers"] = 0  # a real top-level TrainPipelineConfig field
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.num_workers == 0

    def test_unknown_extra_is_ignored(self, spec, caplog):
        pytest.importorskip("lerobot")
        spec.extra["definitely_not_a_field"] = "x"
        with caplog.at_level("WARNING"):
            cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert not hasattr(cfg, "definitely_not_a_field")
        assert any("ignoring extra" in r.message for r in caplog.records)

    def test_resume_sets_checkpoint_path(self, spec, tmp_path):
        pytest.importorskip("lerobot")
        last = tmp_path / "out" / "checkpoints" / "last" / "pretrained_model"
        last.mkdir(parents=True)
        # A real checkpoint's train_config.json is a fully-serialized
        # TrainPipelineConfig, never an empty stub: _build_resume_config rebuilds
        # the config via TrainPipelineConfig.from_pretrained (draccus), which
        # rejects a config missing required fields like `dataset`. Serialize a
        # real fresh-built config so resume exercises the true round-trip.
        trainer = LerobotTrainer(device="cpu")
        trainer._build_policy_config(spec)._save_pretrained(last)
        spec.output_dir = str(tmp_path / "out")
        spec.resume = True
        cfg = trainer.build_config(spec)
        # checkpoint_path is pretrained_model.parent.parent == the "last" dir.
        assert str(cfg.checkpoint_path) == str(last.parent)
        # The checkpoint's config survives the from_pretrained round-trip (not a
        # defaults-only fresh build): the dataset field is restored verbatim. For
        # the local-root spec fixture, _dataset_source resolves repo_id="local".
        assert cfg.dataset.repo_id == "local"


class TestResolveDotted:
    def test_plain_key(self):
        from strands_robots.training.lerobot import _resolve_dotted

        class C:
            pass

        c = C()
        assert _resolve_dotted(c, "steps") == (c, "steps")

    def test_single_level_dotted(self):
        from strands_robots.training.lerobot import _resolve_dotted

        class Sub:
            pass

        class C:
            pass

        c = C()
        c.dataset = Sub()
        assert _resolve_dotted(c, "dataset.root") == (c.dataset, "root")

    def test_missing_head_returns_none(self):
        from strands_robots.training.lerobot import _resolve_dotted

        class C:
            pass

        assert _resolve_dotted(C(), "nope.root") == (None, "root")

    def test_multi_level_dotted_unsupported(self):
        from strands_robots.training.lerobot import _resolve_dotted

        class Sub:
            pass

        class C:
            pass

        c = C()
        c.a = Sub()
        # Only single-level dotting is wired; deeper paths bail out.
        assert _resolve_dotted(c, "a.b.c") == (None, "b.c")


class TestAutoDevice:
    def test_cuda_preferred(self, monkeypatch):
        import torch

        from strands_robots.training.lerobot import _auto_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        assert _auto_device() == "cuda"

    def test_mps_when_no_cuda(self, monkeypatch):
        import torch

        from strands_robots.training.lerobot import _auto_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
        assert _auto_device() == "mps"

    def test_cpu_fallback(self, monkeypatch):
        import torch

        from strands_robots.training.lerobot import _auto_device

        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)
        assert _auto_device() == "cpu"

    def test_no_torch_falls_back_to_cpu(self, monkeypatch):
        import builtins

        from strands_robots.training.lerobot import _auto_device

        real_import = builtins.__import__

        def fake_import(name, *args, **kwargs):
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", fake_import)
        assert _auto_device() == "cpu"


class TestTrainOrchestration:
    """train() control flow with lerobot's train() stubbed out (no real run)."""

    def test_validation_failure_short_circuits(self, spec):
        spec.steps = -1  # invalid
        result = LerobotTrainer(device="cpu").train(spec)
        assert result.status == "error"
        assert "validation failed" in result.message
        assert result.job_id == ""

    def test_build_config_failure_is_caught(self, spec, monkeypatch):
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])

        def boom(_s):
            raise ValueError("bad config")

        monkeypatch.setattr(trainer, "build_config", boom)
        result = trainer.train(spec)
        assert result.status == "error"
        assert "failed to build lerobot TrainPipelineConfig" in result.message

    def test_success_path_parses_metrics(self, spec, monkeypatch):
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())

        import lerobot.scripts.lerobot_train as lt

        def fake_train(cfg, **kw):
            # lerobot tees stdout to the log; emit one MetricsTracker line.
            print("step:2 smpl:4 ep:1 epch:1.00 loss:0.42")

        monkeypatch.setattr(lt, "train", fake_train)
        result = trainer.train(spec)
        assert result.status == "success"
        assert result.metrics["latest_step"] == 2
        assert result.metrics["learning"] is True

    def test_train_error_is_converted_to_result(self, spec, monkeypatch):
        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())

        import lerobot.scripts.lerobot_train as lt

        def fake_train(cfg, **kw):
            raise RuntimeError("CUDA OOM")

        monkeypatch.setattr(lt, "train", fake_train)
        result = trainer.train(spec)
        assert result.status == "error"
        assert "lerobot train raised RuntimeError" in result.message
        assert "CUDA OOM" in result.message

    def test_fresh_start_clears_stale_output_dir(self, spec, monkeypatch, tmp_path):
        out = tmp_path / "stale_out"
        out.mkdir()
        sentinel = out / "leftover.txt"
        sentinel.write_text("old")
        spec.output_dir = str(out)
        spec.resume = False

        trainer = LerobotTrainer(device="cpu")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())

        import lerobot.scripts.lerobot_train as lt

        monkeypatch.setattr(lt, "train", lambda cfg, **kw: None)
        trainer.train(spec)
        # Stale dir (no resumable checkpoint) is wiped before training.
        assert not sentinel.exists()

    def test_multi_gpu_uses_elastic_launch(self, spec, monkeypatch):
        spec.num_gpus = 2
        trainer = LerobotTrainer(device="cuda")
        monkeypatch.setattr(trainer, "validate", lambda s: [])
        monkeypatch.setattr(trainer, "build_config", lambda s: object())

        calls = {}

        def fake_elastic(fn, **kw):
            calls.update(kw)

        monkeypatch.setattr("strands_robots.training.lerobot.elastic_launch_callable", fake_elastic)
        result = trainer.train(spec)
        assert calls["nproc_per_node"] == 2
        assert result.status == "success"


class TestStreamingAndHubSource:
    """Hub-repo + streaming data source (the 50-500 GB no-download fix).

    A LeRobotDataset can be trained either from a local v3 root (the
    record->train loop) OR streamed from the Hub by ``dataset_repo_id`` without
    a full local download. ``streaming`` selects lerobot's
    ``StreamingLeRobotDataset``. These tests pin both the argv-parity helper and
    the typed-config path against real lerobot.
    """

    def test_hub_repo_id_validates_without_local_root(self, tmp_path):
        # No local v3 root present; a Hub repo id is a sufficient data source.
        spec = TrainSpec(
            dataset_root="",
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            extra={"policy_type": "act"},
        )
        assert LerobotTrainer().validate(spec) == []

    def test_invalid_hub_repo_id_rejected(self, tmp_path):
        spec = TrainSpec(
            dataset_repo_id="not a repo id!!",
            base_model="",
            output_dir=str(tmp_path / "out"),
            extra={"policy_type": "act"},
        )
        problems = LerobotTrainer().validate(spec)
        assert any("not a valid Hub id" in p for p in problems)

    def test_no_data_source_is_rejected(self, tmp_path):
        spec = TrainSpec(
            dataset_root="",
            base_model="",
            output_dir=str(tmp_path / "out"),
            extra={"policy_type": "act"},
        )
        problems = LerobotTrainer().validate(spec)
        assert any("a data source is required" in p for p in problems)

    def test_build_command_streams_from_hub(self, tmp_path):
        spec = TrainSpec(
            dataset_root="",
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            extra={"policy_type": "act"},
        )
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--dataset.repo_id=lerobot/aloha_sim_transfer_cube_human" in cmd
        # No local root flag when streaming purely from the Hub.
        assert not any(c.startswith("--dataset.root=") for c in cmd)
        assert "--dataset.streaming=true" in cmd

    def test_build_command_local_root_keeps_repo_id_local(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            extra={"policy_type": "act"},
        )
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--dataset.repo_id=local" in cmd
        assert f"--dataset.root={dataset_root}" in cmd
        # streaming defaults off -> no flag.
        assert not any("streaming" in c for c in cmd)

    def test_build_config_streams_from_hub(self, tmp_path):
        pytest.importorskip("lerobot")
        spec = TrainSpec(
            dataset_root="",
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            extra={"policy_type": "act"},
        )
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.repo_id == "lerobot/aloha_sim_transfer_cube_human"
        assert cfg.dataset.root is None
        assert cfg.dataset.streaming is True

    def test_build_config_local_streaming(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            extra={"policy_type": "act"},
        )
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.repo_id == "local"
        assert cfg.dataset.root == dataset_root
        assert cfg.dataset.streaming is True

    def test_local_cache_root_with_hub_repo_id(self, tmp_path):
        # Hub id + a local cache root: repo_id is the Hub id, root is the cache.
        cache = str(tmp_path / "cache")
        spec = TrainSpec(
            dataset_root=cache,
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            extra={"policy_type": "act"},
        )
        repo_id, root = LerobotTrainer()._dataset_source(spec)
        assert repo_id == "lerobot/aloha_sim_transfer_cube_human"
        assert root == cache

    def test_val_episodes_noop_without_local_root(self, tmp_path):
        # No local meta/info.json to count episodes -> use the full Hub dataset.
        spec = TrainSpec(
            dataset_root="",
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            val_episodes=2,
            extra={"policy_type": "act"},
        )
        assert LerobotTrainer()._val_eval_split(spec) is None


class TestRelativeActions:
    """relative_actions wiring: extra['relative_actions'] -> policy.use_relative_actions.

    Relative-action training (predict deltas from current state) is part of the
    strongest manipulation ablations. lerobot implements it as a matched
    processor pair built from ``config.use_relative_actions`` and saved into the
    checkpoint's pre/post processors, so the inference-side inverse decode is
    restored automatically by lerobot_local. Before the fix the flag had no
    wiring: passing it via extra fell through the generic passthrough (no
    matching top-level config field) and was silently dropped, so relative-action
    training was unreachable and unsupported policies failed silently.
    """

    def _pi0_spec(self, dataset_root, tmp_path, ptype="pi0"):
        return TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": ptype, "relative_actions": True},
        )

    def test_build_config_sets_use_relative_actions(self, dataset_root, tmp_path):
        cfg = LerobotTrainer(device="cpu").build_config(self._pi0_spec(dataset_root, tmp_path))
        assert cfg.policy.use_relative_actions is True

    def test_build_command_emits_flag(self, dataset_root, tmp_path):
        cmd = LerobotTrainer(device="cpu").build_command(self._pi0_spec(dataset_root, tmp_path))
        assert "--policy.use_relative_actions=true" in cmd
        # Must not leak as a bare top-level flag.
        assert not any(c.startswith("--relative_actions=") for c in cmd)

    def test_default_off_leaves_flag_false(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "pi0"},
        )
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.policy.use_relative_actions is False
        assert not any("use_relative_actions" in c for c in LerobotTrainer(device="cpu").build_command(spec))

    def test_validate_rejects_unsupported_policy(self, dataset_root, tmp_path):
        spec = self._pi0_spec(dataset_root, tmp_path, ptype="act")
        problems = LerobotTrainer().validate(spec)
        assert any("relative_actions is not supported" in p for p in problems)

    def test_validate_accepts_pi_family(self, dataset_root, tmp_path):
        for ptype in ("pi0", "pi05", "pi0_fast"):
            spec = self._pi0_spec(dataset_root, tmp_path, ptype=ptype)
            assert LerobotTrainer().validate(spec) == []


class TestExpertOnlyMethod:
    """expert_only training method -> policy.train_expert_only on the built config.

    ``method="expert_only"`` trains only the action expert of a VLA while the
    (V)LM backbone stays frozen - the standard cheap-finetune recipe for the pi0
    family. build_command emits ``--policy.train_expert_only=true`` for the CLI
    launch path, but an in-process run consumes the ``build_config`` object
    directly, so the flag must also be applied to the policy config there.
    Before this was pinned only the CLI path was exercised; a build_config
    regression that dropped the flag would silently full-finetune the backbone
    (a far more expensive, different run) while reporting success.
    """

    def _expert_spec(self, dataset_root, tmp_path, ptype="pi0"):
        return TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            method="expert_only",
            extra={"policy_type": ptype},
        )

    def test_build_config_sets_train_expert_only(self, dataset_root, tmp_path):
        cfg = LerobotTrainer(device="cpu").build_config(self._expert_spec(dataset_root, tmp_path))
        assert cfg.policy.train_expert_only is True

    def test_default_method_leaves_train_expert_only_at_preset(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "pi0"},
        )
        assert spec.method == "full"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # A plain (non-expert_only) run must not silently flip the flag on.
        assert cfg.policy.train_expert_only is False

    def test_validate_rejects_expert_only_for_unsupported_policy(self, dataset_root, tmp_path):
        # pi0_fast's lerobot config has NO train_expert_only field: build_config
        # silently full-finetunes the backbone (hasattr guard) while reporting
        # success, so validate() must surface it up front.
        spec = self._expert_spec(dataset_root, tmp_path, ptype="pi0_fast")
        problems = LerobotTrainer().validate(spec)
        assert any("method 'expert_only' is not supported" in p for p in problems)

    def test_validate_accepts_expert_only_for_supported_policy(self, dataset_root, tmp_path):
        for ptype in ("pi0", "pi05", "smolvla"):
            spec = self._expert_spec(dataset_root, tmp_path, ptype=ptype)
            assert LerobotTrainer().validate(spec) == []

    def test_policy_supports_expert_only_tracks_config_field(self):
        # Drift guard: the capability must reflect each config's ACTUAL
        # train_expert_only field, not a hardcoded copy - so a lerobot policy
        # that gains/loses the field is tracked with zero maintenance here.
        import dataclasses

        from strands_robots.training.lerobot import _policy_registry, _policy_supports_expert_only

        reg = _policy_registry()
        if reg is None:
            pytest.skip("lerobot registry unavailable offline")
        for ptype, cfg_cls in reg.items():
            has_field = any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
            assert _policy_supports_expert_only(ptype) is has_field, ptype


class TestSampleWeightingRABC:
    """RA-BC sample-weighting wiring: extra['sample_weighting'] -> nested SampleWeightingConfig.

    Regression for the folding recipe's headline ablation (HQ + RA-BC + relative
    actions). lerobot >= 0.6.0 configures RA-BC through a NESTED
    ``SampleWeightingConfig`` on ``TrainPipelineConfig`` (``cfg.sample_weighting``,
    fields ``type`` / ``progress_path`` / ``head_mode`` / ``kappa`` / ``epsilon``),
    replacing the flat ``use_rabc`` / ``rabc_*`` fields of earlier 0.5.x. The
    trainer forwards the friendly ``sample_weighting`` dict (whose keys match
    those fields 1:1) into that config. Before this migration the trainer set the
    removed flat fields and raised "no 'use_rabc'" against lerobot 0.6.0, so RA-BC
    was unreachable.
    """

    def _rabc_spec(self, dataset_root, tmp_path):
        return TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={
                "policy_type": "act",
                "sample_weighting": {"type": "rabc", "kappa": 0.02, "head_mode": "sparse"},
            },
        )

    def test_build_config_sets_nested_sample_weighting(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.utils.sample_weighting")
        cfg = LerobotTrainer(device="cpu").build_config(self._rabc_spec(dataset_root, tmp_path))
        assert cfg.sample_weighting is not None
        assert cfg.sample_weighting.type == "rabc"
        assert cfg.sample_weighting.kappa == 0.02
        assert cfg.sample_weighting.head_mode == "sparse"

    def test_build_config_forwards_progress_path(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.utils.sample_weighting")
        spec = self._rabc_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"]["progress_path"] = "/tmp/sarm_progress.parquet"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.sample_weighting.progress_path == "/tmp/sarm_progress.parquet"

    def test_build_config_old_lerobot_raises_actionable(self, dataset_root, tmp_path, monkeypatch):
        # On a lerobot without the nested sample-weighting surface, build_config
        # must raise an actionable ValueError ("requires lerobot >= 0.6.0"), not
        # leak the raw ImportError from the internal SampleWeightingConfig import.
        pytest.importorskip("lerobot.utils.sample_weighting")
        import sys

        monkeypatch.setitem(sys.modules, "lerobot.utils.sample_weighting", None)
        with pytest.raises(ValueError, match="requires lerobot >= 0.6.0"):
            LerobotTrainer(device="cpu").build_config(self._rabc_spec(dataset_root, tmp_path))

    def test_build_config_missing_sample_weighting_field_raises_actionable(self, dataset_root, tmp_path, monkeypatch):
        # A lerobot whose TrainPipelineConfig predates the nested sample-weighting
        # field (no ``cfg.sample_weighting`` attribute at all) must raise the
        # actionable "does not expose sample weighting" ValueError -- distinct from
        # the ImportError path above, where the field exists but the helper module
        # is gone. Shadow TrainPipelineConfig with a subclass that hides the
        # attribute to stand in for that older lerobot.
        pytest.importorskip("lerobot.utils.sample_weighting")
        import lerobot.configs.train as lerobot_train_cfg

        class _NoSampleWeightingConfig(lerobot_train_cfg.TrainPipelineConfig):
            def __getattribute__(self, name):
                if name == "sample_weighting":
                    raise AttributeError(name)
                return super().__getattribute__(name)

        monkeypatch.setattr(lerobot_train_cfg, "TrainPipelineConfig", _NoSampleWeightingConfig)
        with pytest.raises(ValueError, match="does not expose sample weighting"):
            LerobotTrainer(device="cpu").build_config(self._rabc_spec(dataset_root, tmp_path))

    def test_build_command_emits_nested_flags(self, dataset_root, tmp_path):
        cmd = LerobotTrainer(device="cpu").build_command(self._rabc_spec(dataset_root, tmp_path))
        assert "--sample_weighting.type=rabc" in cmd
        assert "--sample_weighting.kappa=0.02" in cmd
        assert "--sample_weighting.head_mode=sparse" in cmd
        # The dict must NOT leak through as one top-level flag, and the removed
        # flat <= 0.5.1 fields must NOT be emitted.
        assert not any(c == "--sample_weighting" or c.startswith("--sample_weighting=") for c in cmd)
        assert not any(c.startswith("--use_rabc") or c.startswith("--rabc_") for c in cmd)

    def test_no_sample_weighting_leaves_it_unset(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.utils.sample_weighting")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "act"},
        )
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.sample_weighting is None

    def test_unsupported_field_raises_actionable_error(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.utils.sample_weighting")
        spec = self._rabc_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"] = {"type": "rabc", "bogus_field": 1}
        with pytest.raises(ValueError, match="does not support field"):
            LerobotTrainer(device="cpu").build_config(spec)

    def test_unsupported_type_raises_actionable_error(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.utils.sample_weighting")
        spec = self._rabc_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"] = {"type": "boltzmann", "kappa": 0.02}
        with pytest.raises(ValueError, match="must be one of"):
            LerobotTrainer(device="cpu").build_config(spec)

    def test_validate_rejects_non_dict(self, dataset_root, tmp_path):
        spec = self._rabc_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"] = "rabc"
        problems = LerobotTrainer().validate(spec)
        assert any("sample_weighting" in p and "dict" in p for p in problems)

    def test_validate_rejects_leading_dash_value(self, dataset_root, tmp_path):
        spec = self._rabc_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"] = {"type": "rabc", "progress_path": "-x"}
        problems = LerobotTrainer().validate(spec)
        assert any("must not start with '-'" in p for p in problems)


class TestRewardModelTraining:
    """SARM reward-model training: extra['reward_model'] -> cfg.reward_model.

    The *producing* half of RA-BC. A reward model (SARM) trains through the SAME
    ``lerobot_train.train(cfg)`` entry point as a policy, but populates
    ``cfg.reward_model`` (and leaves ``cfg.policy`` unset) so lerobot follows its
    ``is_reward_model_training`` path. Requires lerobot >= 0.6.0 (the
    ``lerobot.rewards`` package). Before this, ``sarm`` was rejected outright -
    there was no reward-model path in ``LerobotTrainer`` at all.
    """

    def _sarm_spec(self, dataset_root, tmp_path, **rm):
        reward_model = {"type": "sarm", "annotation_mode": "single_stage"}
        reward_model.update(rm)
        return TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "sarm_out"),
            steps=200,
            extra={"reward_model": reward_model},
        )

    def test_validate_accepts_sarm(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.rewards")
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        assert LerobotTrainer().validate(spec) == []

    def test_build_config_targets_reward_model(self, dataset_root, tmp_path):
        pytest.importorskip("lerobot.rewards")
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # cfg.reward_model is set, cfg.policy is not -> lerobot's reward path.
        assert cfg.is_reward_model_training is True
        assert cfg.policy is None
        assert cfg.reward_model.type == "sarm"
        assert cfg.reward_model.annotation_mode == "single_stage"
        assert cfg.reward_model.image_key == "observation.images.base"

    def test_build_config_forwards_base_model_to_reward_pretrained_path(self, dataset_root, tmp_path):
        """A reward-model spec's base_model warm-starts cfg.reward_model.pretrained_path.

        A reward model can resume from a pretrained checkpoint the same way a
        policy does. When TrainSpec.base_model is set on a reward-model run, it
        must land on cfg.reward_model.pretrained_path so the checkpoint is
        actually loaded - never silently dropped (which would train from scratch
        despite the caller asking to warm-start).
        """
        pytest.importorskip("lerobot.rewards")
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        spec.base_model = "lerobot/sarm_pretrained"
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert str(cfg.reward_model.pretrained_path) == "lerobot/sarm_pretrained"

    def test_build_command_emits_reward_model_flags(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--reward_model.type=sarm" in cmd
        assert "--reward_model.annotation_mode=single_stage" in cmd
        assert "--reward_model.image_key=observation.images.base" in cmd
        # A reward-model run does not train a policy -> no --policy.* flags.
        assert not any(c.startswith("--policy.") for c in cmd)

    def test_build_command_emits_reward_model_pretrained_path(self, dataset_root, tmp_path, monkeypatch):
        """A reward-model spec's base_model must reach --reward_model.pretrained_path.

        build_config warm-starts cfg.reward_model.pretrained_path from base_model
        (test_build_config_forwards_base_model_to_reward_pretrained_path), but the
        argv-parity helper only emitted --policy.pretrained_path (the policy
        branch). For a warm-started reward-model run the documented "equivalent
        CLI" therefore trained from scratch (pretrained_path defaults to None)
        instead of loading base_model -- the exact build_command<->build_config
        drift the policy path already avoids.

        Forces the offline reward-registry fallback so the argv parity is asserted
        without importing lerobot (the pretrained_path emission is
        registry-independent), keeping the check fast and deterministic.
        """
        import sys

        monkeypatch.setitem(sys.modules, "lerobot.rewards", None)
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        spec.base_model = "lerobot/sarm_pretrained"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--reward_model.pretrained_path=lerobot/sarm_pretrained" in cmd
        # A reward-model run never trains a policy -> no --policy.pretrained_path.
        assert not any(c.startswith("--policy.pretrained_path=") for c in cmd)

    def test_validate_rejects_unknown_reward_type(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path, type="not_a_reward_model")
        problems = LerobotTrainer().validate(spec)
        assert any("is not LeRobot-native" in p for p in problems)

    def test_validate_rejects_bad_annotation_mode(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path, annotation_mode="bogus")
        problems = LerobotTrainer().validate(spec)
        assert any("annotation_mode" in p and "invalid" in p for p in problems)

    def test_validate_rejects_unknown_reward_field(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path)
        spec.extra["reward_model"]["bogus"] = 1
        problems = LerobotTrainer().validate(spec)
        assert any("does not support field" in p for p in problems)

    def test_validate_rejects_sample_weighting_combo(self, dataset_root, tmp_path):
        # RA-BC weights POLICY training; pairing it with a reward-model run is a
        # pipeline-ordering mistake (train SARM first, THEN weight a policy).
        spec = self._sarm_spec(dataset_root, tmp_path)
        spec.extra["sample_weighting"] = {"type": "rabc"}
        problems = LerobotTrainer().validate(spec)
        assert any("RA-BC" in p and "POLICY" in p for p in problems)

    def test_validate_rejects_relative_actions_combo(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path)
        spec.extra["relative_actions"] = True
        problems = LerobotTrainer().validate(spec)
        assert any("relative_actions applies to policy training" in p for p in problems)

    def test_validate_rejects_non_full_method(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path)
        spec.method = "lora"
        problems = LerobotTrainer().validate(spec)
        assert any("reward-model training uses method='full'" in p for p in problems)

    def test_validate_rejects_non_dict_reward_model(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"reward_model": "sarm"},
        )
        problems = LerobotTrainer().validate(spec)
        assert any("reward_model" in p and "dict" in p for p in problems)

    def test_validate_rejects_leading_dash_value(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="-x")
        problems = LerobotTrainer().validate(spec)
        assert any("must not start with '-'" in p for p in problems)


class TestBuilderEscapeHatchValidation:
    """The builders re-validate ``extra`` escape-hatch types independently of validate().

    ``build_config`` / ``build_command`` are public entry points a caller can
    reach without first running :meth:`LerobotTrainer.validate` (for example a
    programmatic caller that already trusts its inputs). Each escape hatch that
    ``validate()`` guards - ``extra['reward_model']``, ``extra['sample_weighting']``,
    and ``extra['relative_actions']`` - is therefore re-checked at build time so
    a malformed value fails fast with an actionable ValueError instead of being
    silently coerced into a stray flag or a config missing the intended wiring.
    """

    def test_build_config_rejects_non_dict_reward_model(self, dataset_root, tmp_path):
        # build_config resolves the reward-model escape hatch itself (before any
        # lerobot config is built), so a non-dict value must raise here too.
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"reward_model": "sarm"},  # str, not a dict of fields
        )
        with pytest.raises(ValueError, match="must be a dict"):
            LerobotTrainer(device="cpu").build_config(spec)

    def test_build_command_rejects_non_dict_sample_weighting(self, dataset_root, tmp_path):
        # A non-dict sample_weighting must raise, never be flattened into stray
        # --sample_weighting.* flags.
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "act", "sample_weighting": "rabc"},
        )
        with pytest.raises(ValueError, match="must be a dict"):
            LerobotTrainer(device="cpu").build_command(spec)

    def test_build_config_rejects_relative_actions_for_unsupported_policy(self, dataset_root, tmp_path):
        # act does not expose use_relative_actions (only the pi0 family and
        # groot do); build_config must fail fast for such a policy rather than
        # drop the flag silently (which would train an ordinary absolute-action
        # policy unnoticed).
        pytest.importorskip("lerobot")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "act", "relative_actions": True},
        )
        with pytest.raises(ValueError, match="use_relative_actions"):
            LerobotTrainer(device="cpu").build_config(spec)


class TestLearningRate:
    """An explicit ``TrainSpec.learning_rate`` reaches lerobot's optimizer config.

    Regression for the gap where ``learning_rate`` was the only universal spec
    field with zero references in ``training/lerobot.py``: the policy training
    preset was always used and a caller-set value was silently dropped. The
    opt-in shape mirrors ``seed`` -- ``None`` keeps the policy preset, an
    explicit value maps to ``policy.optimizer_lr``.
    """

    def test_explicit_lr_reaches_policy_optimizer_config(self, dataset_root, tmp_path):
        # Pre-fix: cfg.policy.optimizer_lr stayed at the ACT preset (1e-5),
        # silently ignoring the requested 5e-5.
        pytest.importorskip("lerobot")
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            learning_rate=5e-5,
            extra={"policy_type": "act"},
        )
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.policy.optimizer_lr == 5e-5

    def test_default_lr_keeps_policy_preset(self, dataset_root, tmp_path):
        # learning_rate defaults to None -> the ACT preset (1e-5) is untouched,
        # so a plain run is unchanged by this feature.
        pytest.importorskip("lerobot")
        from lerobot.policies.factory import make_policy_config

        preset_lr = make_policy_config("act").optimizer_lr
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "act"},
        )
        assert spec.learning_rate is None
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.policy.optimizer_lr == preset_lr

    def test_build_command_emits_optimizer_lr_when_set(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            learning_rate=5e-5,
            extra={"policy_type": "act"},
        )
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--policy.optimizer_lr=5e-05" in cmd

    def test_build_command_omits_optimizer_lr_by_default(self, dataset_root, tmp_path):
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            extra={"policy_type": "act"},
        )
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert not any(c.startswith("--policy.optimizer_lr=") for c in cmd)

    def test_explicit_lr_on_policy_without_optimizer_lr_raises(self, dataset_root, tmp_path, monkeypatch):
        # A policy whose optimizer preset is not an Adam-style single LR has no
        # optimizer_lr field; an explicit learning_rate must fail loudly rather
        # than be dropped.
        pytest.importorskip("lerobot")
        import lerobot.policies.factory as factory

        class _NoLrPolicyConfig:
            type = "act"
            device = "cpu"
            push_to_hub = False

        monkeypatch.setattr(factory, "make_policy_config", lambda ptype: _NoLrPolicyConfig())
        spec = TrainSpec(
            dataset_root=dataset_root,
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=200,
            learning_rate=5e-5,
            extra={"policy_type": "act"},
        )
        with pytest.raises(ValueError, match="optimizer_lr"):
            LerobotTrainer(device="cpu").build_config(spec)


class TestOfflineRegistryFallbacks:
    """Graceful degradation when lerobot's config ChoiceRegistry is unavailable.

    Type/field discovery reads live off lerobot's draccus registries, which are
    populated as an import side effect of ``lerobot.policies`` / ``lerobot.rewards``.
    When those imports fail (lerobot not installed, or lerobot < 0.6.0 with no
    ``lerobot.rewards``), discovery must fall back to the documented static sets
    instead of raising, so ``validate`` still produces an actionable message
    offline. Import failure is simulated by binding the submodule to ``None`` in
    ``sys.modules`` (makes ``import`` raise ``ImportError``).
    """

    def test_policy_registry_is_none_when_lerobot_policies_unimportable(self, monkeypatch):
        import sys

        from strands_robots.training import lerobot as mod

        monkeypatch.setitem(sys.modules, "lerobot.policies", None)
        assert mod._policy_registry() is None

    def test_policy_types_use_static_fallback_offline(self, monkeypatch):
        import sys

        from strands_robots.training import lerobot as mod

        monkeypatch.setitem(sys.modules, "lerobot.policies", None)
        assert mod._lerobot_policy_types() == set(mod._LEROBOT_POLICY_TYPES_FALLBACK)

    def test_relative_action_support_uses_static_fallback_offline(self, monkeypatch):
        import sys

        from strands_robots.training import lerobot as mod

        monkeypatch.setitem(sys.modules, "lerobot.policies", None)
        # The pi0 family is in the documented fallback set; act is not.
        assert mod._policy_supports_relative_actions("pi0") is True
        assert mod._policy_supports_relative_actions("act") is False

    def test_reward_registry_is_none_when_lerobot_rewards_unimportable(self, monkeypatch):
        import sys

        from strands_robots.training import lerobot as mod

        monkeypatch.setitem(sys.modules, "lerobot.rewards", None)
        assert mod._reward_registry() is None

    def test_reward_model_types_use_static_fallback_offline(self, monkeypatch):
        import sys

        from strands_robots.training import lerobot as mod

        monkeypatch.setitem(sys.modules, "lerobot.rewards", None)
        assert mod._reward_model_types() == set(mod._REWARD_MODEL_TYPES_FALLBACK)


class TestRunTypeLabel:
    """``_run_type_label`` distinguishes a reward-model run from a policy run."""

    def test_labels_reward_model_run(self):
        spec = TrainSpec(extra={"reward_model": {"type": "sarm"}})
        assert LerobotTrainer(device="cpu")._run_type_label(spec) == "reward_model:sarm"

    def test_labels_policy_run(self):
        spec = TrainSpec(extra={"policy_type": "diffusion"})
        assert LerobotTrainer(device="cpu")._run_type_label(spec) == "policy:diffusion"


class TestValSplitEpisodesFallthrough:
    """The held-out split is a no-op (use the full dataset) when it can't be computed.

    ``_val_eval_split`` returns ``None`` -- meaning "train on every episode" --
    rather than raising or emitting a malformed ``episodes`` range when the episode
    count is unknown (no readable ``meta/info.json``) or the requested holdout is
    not strictly inside ``(0, total)``.
    """

    def test_no_op_when_episode_count_unknown(self, tmp_path):
        # dataset_root set but no meta/info.json -> total unknown -> no split.
        spec = TrainSpec(dataset_root=str(tmp_path), val_episodes=2)
        assert LerobotTrainer(device="cpu")._val_eval_split(spec) is None

    def test_no_op_when_holdout_not_smaller_than_total(self, dataset_root):
        # info.json reports total_episodes=10; a holdout >= total is out of range.
        spec = TrainSpec(dataset_root=dataset_root, val_episodes=10)
        assert LerobotTrainer(device="cpu")._val_eval_split(spec) is None


class TestHardwareFloor:
    """``hardware_floor`` advertises the advisory minimum compute for LeRobot tuning."""

    def test_advisory_single_consumer_gpu(self):
        floor = LerobotTrainer(device="cpu").hardware_floor
        assert floor == {"min_gpus": 1, "min_vram_gb": 8, "multinode": False}


class TestLerobotDdpWorker:
    """``_lerobot_worker`` is the per-GPU entry torch's elastic launcher spawns.

    It rebuilds the typed config in-worker and calls lerobot's ``train`` inline
    (no argv, no nested interpreter). Only local rank 0 tees output to the shared
    log so parallel workers do not interleave writes into one file.
    """

    def _install_fake_train(self, monkeypatch, recorder):
        import sys
        import types

        module = types.ModuleType("lerobot.scripts.lerobot_train")

        def _train(cfg, **kwargs):
            recorder["cfg"] = cfg
            print("TRAINING_RAN")

        module.train = _train
        monkeypatch.setitem(sys.modules, "lerobot.scripts.lerobot_train", module)

    def _spec(self, tmp_path):
        return TrainSpec(
            dataset_root=str(tmp_path),
            base_model="",
            output_dir=str(tmp_path / "out"),
            steps=1,
            extra={"policy_type": "act"},
        )

    def test_rank0_worker_trains_and_writes_shared_log(self, tmp_path, monkeypatch):
        from strands_robots.training import lerobot as lerobot_module

        recorder: dict = {}
        self._install_fake_train(monkeypatch, recorder)
        sentinel = object()
        monkeypatch.setattr(LerobotTrainer, "build_config", lambda self, spec: sentinel)
        monkeypatch.setenv("LOCAL_RANK", "0")

        log_path = tmp_path / "train.log"
        lerobot_module._lerobot_worker("act", "cpu", self._spec(tmp_path), str(log_path))

        # The config built in-worker is exactly what lerobot's train() receives.
        assert recorder["cfg"] is sentinel
        # Rank 0 owns the shared log, so train output lands in it.
        assert log_path.is_file()
        assert "TRAINING_RAN" in log_path.read_text()

    def test_nonzero_rank_worker_trains_without_writing_shared_log(self, tmp_path, monkeypatch):
        from strands_robots.training import lerobot as lerobot_module

        recorder: dict = {}
        self._install_fake_train(monkeypatch, recorder)
        sentinel = object()
        monkeypatch.setattr(LerobotTrainer, "build_config", lambda self, spec: sentinel)
        monkeypatch.setenv("LOCAL_RANK", "1")

        log_path = tmp_path / "train.log"
        lerobot_module._lerobot_worker("act", "cpu", self._spec(tmp_path), str(log_path))

        # Training still runs on every rank...
        assert recorder["cfg"] is sentinel
        # ...but only rank 0 writes the shared log, so a non-zero rank leaves it absent.
        assert not log_path.exists()


class TestInProcessTrainerCorrectness:
    """Regression tests for the in-process trainer's resume / LoRA / warm-start.

    Each test fails on the pre-fix code and passes after:
      * resume: cfg.validate() needs --config_path on sys.argv (argv shim);
      * LoRA: strands must not pre-set policy_cfg.use_peft (inverts lerobot);
      * warm start: base_model must load the checkpoint's own saved config.
    """

    def _act_checkpoint(self, tmp_path, **overrides):
        """Write a local ACT checkpoint config.json with non-default fields."""
        from lerobot.policies.factory import make_policy_config

        ckpt = tmp_path / "base_ckpt"
        ckpt.mkdir()
        base = make_policy_config("act")
        for key, value in overrides.items():
            setattr(base, key, value)
        base.save_pretrained(ckpt)
        return ckpt

    def test_base_model_warm_start_reads_checkpoint_config(self, spec, tmp_path):
        pytest.importorskip("lerobot")
        # A base checkpoint trained with a non-default chunk_size (default 100).
        ckpt = self._act_checkpoint(tmp_path, chunk_size=42, n_action_steps=42)
        spec.base_model = str(ckpt)
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # Pre-fix: make_policy_config defaults -> chunk_size 100 (checkpoint ignored).
        assert cfg.policy.chunk_size == 42
        assert cfg.policy.n_action_steps == 42
        assert str(cfg.policy.pretrained_path) == str(ckpt)
        # Managed overrides still applied on top of the loaded config.
        assert cfg.policy.device == "cpu"
        assert cfg.policy.push_to_hub is False

    def test_lora_base_model_does_not_preset_use_peft(self, spec, tmp_path):
        pytest.importorskip("lerobot")
        ckpt = self._act_checkpoint(tmp_path)
        spec.base_model = str(ckpt)
        spec.method = "lora"
        spec.lora_r = 8
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # cfg.peft drives wrap_with_peft on the loaded base; use_peft must stay
        # False so make_policy loads the plain base checkpoint, not an adapter repo.
        assert cfg.peft is not None
        assert cfg.peft.method_type == "LORA"
        assert cfg.policy.use_peft is False
        assert str(cfg.policy.pretrained_path) == str(ckpt)

    def _train_checkpoint(self, tmp_path, output_dir, **policy_overrides):
        """Write a resumable checkpoint: <out>/checkpoints/last/pretrained_model.

        Persists a full ``TrainPipelineConfig`` (via draccus ``save_pretrained``)
        exactly as lerobot writes ``train_config.json`` at save time -- so the
        serialized ``optimizer``/``scheduler``/``policy`` are present, which is
        the whole point of resuming from the checkpoint rather than the spec.
        """
        from lerobot.configs.default import DatasetConfig
        from lerobot.configs.train import TrainPipelineConfig
        from lerobot.policies.factory import make_policy_config

        policy = make_policy_config("act")
        for key, value in policy_overrides.items():
            setattr(policy, key, value)
        # Force push_to_hub off, mirroring the production build_config path
        # (lerobot.py sets policy_cfg.push_to_hub = False). Otherwise lerobot's
        # validate() raises "'repo_id' argument missing" because the policy config
        # defaults push_to_hub truthy without a repo_id.
        if hasattr(policy, "push_to_hub"):
            policy.push_to_hub = False
        # A minimal but VALID pipeline config: validate() fills optimizer/scheduler
        # from the policy preset on a fresh (non-resume) build, so this is exactly
        # what lerobot serializes into a real checkpoint's train_config.json.
        # ``dataset`` is a required field on TrainPipelineConfig (no default), so
        # pass a minimal DatasetConfig mirroring the production build path.
        ckpt_cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="dummy/resume-fixture"),
            policy=policy,
            output_dir=output_dir,
            steps=100,
        )
        ckpt_cfg.validate()
        last = output_dir / "checkpoints" / "last" / "pretrained_model"
        last.mkdir(parents=True)
        ckpt_cfg.save_pretrained(last)
        return last

    def test_resume_rebuilds_config_from_checkpoint(self, spec, tmp_path):
        pytest.importorskip("lerobot")
        from strands_robots.training._inproc import resume_argv

        out = tmp_path / "out"
        # Checkpoint trained with a non-default chunk_size (default 100).
        last = self._train_checkpoint(out, out, chunk_size=42, n_action_steps=42)
        spec.output_dir = str(out)
        spec.resume = True
        spec.steps = 500  # resume to a larger step target
        trainer = LerobotTrainer(device="cpu")
        cfg = trainer.build_config(spec)

        # MUST FIX regression: a spec-built resume cfg leaves optimizer/scheduler
        # unset (validate() skips the preset when resuming) and make_optimizer_
        # and_scheduler raises before the train loop. Rebuilding from the
        # checkpoint carries them, so the run can actually start.
        assert cfg.resume is True
        assert cfg.optimizer is not None
        # Policy config comes from the checkpoint, not make_policy_config defaults.
        assert cfg.policy.chunk_size == 42
        assert cfg.policy.n_action_steps == 42
        # Managed run-control overrides reapplied on top of the loaded config.
        assert cfg.steps == 500
        assert str(cfg.output_dir) == str(out)

        cfg_path = trainer._resume_config_path(spec.output_dir)
        assert cfg_path is not None
        # validate() still resolves the checkpoint via the argv shim (the flag
        # lerobot recovers off sys.argv), now on a cfg that also carries optimizer.
        with resume_argv(cfg_path):
            cfg.validate()
        assert str(cfg.policy.pretrained_path) == str(last)

    def test_resume_without_checkpoint_falls_back_to_fresh_build(self, spec, tmp_path):
        pytest.importorskip("lerobot")
        # resume=True but no checkpoint on disk: build_config must NOT raise;
        # it falls through to a fresh build so lerobot reports its own resume error.
        spec.output_dir = str(tmp_path / "out")
        spec.resume = True
        trainer = LerobotTrainer(device="cpu")
        assert trainer._resume_config_path(spec.output_dir) is None
        cfg = trainer.build_config(spec)  # no AttributeError / no crash
        assert cfg.resume is True

    def test_resume_argv_noop_when_config_path_falsy(self):
        from strands_robots.training._inproc import resume_argv

        saved = list(sys.argv)
        with resume_argv(None):
            assert sys.argv == saved
        assert sys.argv == saved

    def test_resume_carries_checkpoint_optimizer_values(self, spec, tmp_path):
        """Resume must inherit the checkpoint's optimizer VALUES, not just a preset.

        lerobot's validate() applies the optimizer preset only when NOT resuming,
        so a spec-built resume config leaves optimizer=None and
        make_optimizer_and_scheduler raises before the train loop. Rebuilding
        from the checkpoint must carry the serialized optimizer verbatim -- a
        run resumed with a hand-tuned learning rate has to keep it, otherwise
        the resumed schedule silently differs from the interrupted one.
        """
        pytest.importorskip("lerobot")

        out = tmp_path / "out"
        last = self._train_checkpoint(out, out)
        # Rewrite the serialized optimizer with a non-preset learning rate, the
        # way a run launched with a tuned lr persists it into train_config.json.
        saved = json.loads((last / "train_config.json").read_text())
        assert saved["optimizer"]["type"] == "adamw"
        saved["optimizer"]["lr"] = 3.7e-6
        (last / "train_config.json").write_text(json.dumps(saved))

        spec.output_dir = str(out)
        spec.resume = True
        trainer = LerobotTrainer(device="cpu")
        cfg = trainer.build_config(spec)

        assert cfg.optimizer is not None, (
            "cfg.optimizer must be populated from the checkpoint's "
            "train_config.json on resume, otherwise make_optimizer_and_scheduler "
            "raises ValueError before the training loop starts."
        )
        assert cfg.optimizer.lr == pytest.approx(3.7e-6), (
            "the resumed run must keep the checkpoint's learning rate, not fall back to the policy preset default"
        )

    def test_resume_with_undecodable_checkpoint_config_raises_actionable_error(self, spec, tmp_path):
        """A checkpoint config that will not decode must name the file it came from.

        A truncated / hand-edited / version-skewed train_config.json used to
        surface draccus' bare DecodingError, which names neither the offending
        path nor a way forward. Resume is fatal in that case (falling back to a
        fresh build re-enters the optimizer=None crash), so it must raise an
        error that points at the file and at resume=False.
        """
        pytest.importorskip("lerobot")

        last = tmp_path / "out" / "checkpoints" / "last" / "pretrained_model"
        last.mkdir(parents=True)
        cfg_file = last / "train_config.json"
        # 'AdamW' is not a registered optimizer choice name (lerobot registers
        # 'adamw'), so this decodes no further than the optimizer field -- the
        # shape a config written by an incompatible version arrives in.
        cfg_file.write_text(json.dumps({"optimizer": {"type": "AdamW", "lr": 1e-4}}))

        spec.output_dir = str(tmp_path / "out")
        spec.resume = True
        trainer = LerobotTrainer(device="cpu")

        with pytest.raises(ValueError) as excinfo:
            trainer.build_config(spec)
        message = str(excinfo.value)
        assert str(cfg_file) in message, "the error must name the checkpoint config path"
        assert "resume=False" in message, "the error must offer a way forward"


class TestStreamingAndValidationSplitAreMutuallyExclusive:
    """``streaming`` and ``val_episodes`` cannot both be honored by lerobot.

    ``val_episodes`` becomes lerobot's ``dataset.eval_split``, and any non-zero
    ``eval_split`` routes lerobot into ``make_train_eval_datasets``, which
    rebuilds both splits as map-style ``LeRobotDataset`` objects without
    consulting ``dataset.streaming``. The whole dataset is materialized - the
    outcome ``streaming`` exists to avoid - and nothing reports it, because an
    annulled stream looks exactly like ``streaming=False``. Preflight refuses
    the pair instead, the same way an unreadable episode count is refused rather
    than allowed to drop a requested split.
    """

    def _spec(self, dataset_root, tmp_path, **kw):
        fields = {
            "dataset_root": dataset_root,
            "base_model": "",
            "output_dir": str(tmp_path / "out"),
            "extra": {"policy_type": "act"},
        }
        fields.update(kw)
        return TrainSpec(**fields)

    def test_asking_for_both_is_refused(self, dataset_root, tmp_path):
        spec = self._spec(dataset_root, tmp_path, streaming=True, val_episodes=2)
        problems = LerobotTrainer().validate(spec)
        assert problems, "streaming + val_episodes delivers neither field, so it is not launchable"
        assert any("streaming=True cannot be combined with val_episodes=2" in p for p in problems)

    def test_the_refusal_names_the_cost_of_the_silent_outcome(self, dataset_root, tmp_path):
        spec = self._spec(dataset_root, tmp_path, streaming=True, val_episodes=2)
        (message,) = [p for p in LerobotTrainer().validate(spec) if "streaming=True" in p]
        # A caller who reads only the message must learn WHY the pair is refused:
        # not a policy choice, but that the stream is dropped and the dataset lands
        # on disk/in RAM in full.
        assert "materialized" in message
        assert "stream is dropped" in message

    @pytest.mark.parametrize(
        ("remedy", "keeps_the_split"),
        [
            ({"streaming": False}, True),
            ({"val_episodes": None}, False),
        ],
    )
    def test_either_remedy_the_message_offers_clears_it(self, dataset_root, tmp_path, remedy, keeps_the_split):
        # Both remedies are quoted in the refusal, so both must actually work -
        # and each must deliver the field it keeps, not merely silence the message.
        trainer = LerobotTrainer()
        asked = {"streaming": True, "val_episodes": 2, **remedy}
        spec = self._spec(dataset_root, tmp_path, **asked)
        assert trainer.validate(spec) == []
        split = trainer._val_eval_split(spec)
        assert (split is not None and split > 0.0) is keeps_the_split
        assert spec.streaming is not keeps_the_split

    def test_streaming_alone_is_launchable(self, dataset_root, tmp_path):
        # The control that keeps the refusal from swallowing the feature itself.
        spec = self._spec(dataset_root, tmp_path, streaming=True)
        assert LerobotTrainer().validate(spec) == []

    def test_validation_split_alone_is_launchable(self, dataset_root, tmp_path):
        spec = self._spec(dataset_root, tmp_path, val_episodes=2)
        assert LerobotTrainer().validate(spec) == []

    def test_a_hub_source_with_no_readable_count_keeps_its_own_refusal(self, tmp_path):
        # With no local meta/info.json no split is emitted at all, so the pair is
        # not what is wrong here - the episode count is. Reporting both would name
        # a conflict that this spec does not have.
        spec = TrainSpec(
            dataset_root="",
            dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
            base_model="",
            output_dir=str(tmp_path / "out"),
            streaming=True,
            val_episodes=2,
            extra={"policy_type": "act"},
        )
        problems = LerobotTrainer().validate(spec)
        assert any("episode count is unavailable" in p for p in problems)
        assert not any("cannot be combined with" in p for p in problems)

    def test_lerobot_split_path_still_drops_the_stream(self, monkeypatch):
        """The measured constraint the refusal stands on, pinned against lerobot.

        Asserts the property that makes the pair unsatisfiable: lerobot's
        eval-split path constructs a map-style dataset even when
        ``dataset.streaming`` is set. If lerobot starts honoring the stream
        there, this fails and the refusal above should be lifted rather than
        left to reject a combination that has become supportable.
        """
        pytest.importorskip("lerobot")
        import lerobot.datasets.factory as factory

        built: list[str] = []

        class _StubDataset:
            def __init__(self, *args, **kwargs):
                built.append(type(self).__name__)
                self.episodes = kwargs.get("episodes")
                self.num_episodes = 4
                self.meta = SimpleNamespace(
                    episodes={"tasks": [["t"], ["t"], ["t"], ["t"]]},
                    camera_keys=[],
                    depth_keys=[],
                    stats={},
                )

        class _StubMapStyle(_StubDataset):
            pass

        class _StubStreaming(_StubDataset):
            pass

        monkeypatch.setattr(factory, "LeRobotDataset", _StubMapStyle)
        monkeypatch.setattr(factory, "StreamingLeRobotDataset", _StubStreaming)
        monkeypatch.setattr(factory, "make_dataset", lambda cfg: _StubStreaming())
        monkeypatch.setattr(factory, "resolve_delta_timestamps", lambda *a, **k: None)

        cfg = SimpleNamespace(
            dataset=SimpleNamespace(
                repo_id="local",
                root=None,
                revision=None,
                streaming=True,
                eval_split=0.25,
                video_backend=None,
                # lerobot's factory reads this on all three of its dataset
                # constructions (stream, train split, eval split). Mirror its
                # own default so the stand-in config carries the surface the
                # real DatasetConfig does.
                depth_output_unit="mm",
                image_transforms=SimpleNamespace(enable=False),
                use_imagenet_stats=False,
            ),
            trainable_config=None,
            rename_map=None,
            tolerance_s=1e-4,
            num_workers=1,
        )
        train_dataset, eval_dataset = factory.make_train_eval_datasets(cfg)
        assert built.count("_StubStreaming") == 1, "the stream is built once, to read metadata"
        assert type(train_dataset) is _StubMapStyle, "lerobot rebuilds the TRAIN split map-style, discarding the stream"
        assert type(eval_dataset) is _StubMapStyle


class TestTheStandInConfigTracksLerobotsOwnFactory:
    """A field lerobot's factory starts reading fails here, not inside lerobot.

    ``make_train_eval_datasets`` reads ``cfg.dataset.<field>`` straight off the
    config it is handed, so a stand-in missing one raises ``AttributeError``
    from inside lerobot rather than failing an assertion in this suite. lerobot
    is tracked from source, so that field set grows: ``depth_output_unit``
    arrived with its depth-map support and the stand-in did not carry it, which
    surfaced as this file's own split assertion appearing to break.

    The expectation is derived from the function the split test actually calls,
    not from the module. ``make_dataset`` also reads ``episodes``,
    ``exclude_episodes`` and ``repo_type`` on paths the split test never
    reaches, so a module-wide rule would demand three fields the stand-in has
    no reason to carry.
    """

    @staticmethod
    def _factory_dataset_fields(function_name: str) -> set[str]:
        """Every ``cfg.dataset.<field>`` lerobot reads inside one function."""
        factory = pytest.importorskip("lerobot.datasets.factory")
        tree = ast.parse(inspect.getsource(factory))
        target = next(
            (n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == function_name),
            None,
        )
        assert target is not None, f"lerobot.datasets.factory has no {function_name}"
        return {
            node.attr
            for node in ast.walk(target)
            if isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Attribute)
            and node.value.attr == "dataset"
            and isinstance(node.value.value, ast.Name)
            and node.value.value.id == "cfg"
        }

    @staticmethod
    def _stand_in_dataset_fields() -> set[str]:
        """The keys this file's own stand-in ``cfg.dataset`` namespace carries."""
        tree = ast.parse(pathlib.Path(__file__).read_text(encoding="utf-8"))
        namespaces = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "SimpleNamespace"
            and any(keyword.arg == "repo_id" for keyword in node.keywords)
        ]
        assert len(namespaces) == 1, f"expected one stand-in dataset namespace, found {len(namespaces)}"
        return {keyword.arg for keyword in namespaces[0].keywords if keyword.arg}

    def test_the_stand_in_carries_every_field_the_split_path_reads(self) -> None:
        needed = self._factory_dataset_fields("make_train_eval_datasets")
        carried = self._stand_in_dataset_fields()
        missing = sorted(needed - carried)
        assert not missing, (
            f"lerobot's make_train_eval_datasets reads cfg.dataset.{missing} and the stand-in "
            "config does not carry it, so the split test raises AttributeError from inside "
            "lerobot instead of grading the split. Add the field to the stand-in, mirroring "
            "lerobot's own default for it."
        )

    def test_both_derived_sets_are_populated(self) -> None:
        needed = self._factory_dataset_fields("make_train_eval_datasets")
        carried = self._stand_in_dataset_fields()
        assert len(needed) >= 5, f"only {len(needed)} fields derived - the scan is looking in the wrong place"
        assert len(carried) >= 5, f"only {len(carried)} stand-in keys derived - the scan is looking in the wrong place"

    def test_the_module_wide_set_is_wider_than_the_path_the_split_test_drives(self) -> None:
        """Why the expectation is function-scoped and not module-scoped."""
        split_path = self._factory_dataset_fields("make_train_eval_datasets")
        other_path = self._factory_dataset_fields("make_dataset")
        assert other_path - split_path, (
            "make_dataset no longer reads any field make_train_eval_datasets does not, so the "
            "function scoping above no longer buys anything and could be widened to the module"
        )
