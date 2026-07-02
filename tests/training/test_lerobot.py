"""Tests for LerobotTrainer: factory wiring, validate, and command building.

These are pure/offline (no GPU, no actual lerobot_train launch). The real
end-to-end sim->train->load is exercised separately.
"""

import json

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
        base_model="lerobot/act_aloha_sim",
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


class TestBuildCommand:
    def test_single_gpu_core_flags(self, spec):
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
        spec.lora_target_modules = "q_proj,v_proj"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--peft.method_type=LORA" in cmd
        assert "--peft.r=16" in cmd
        assert "--peft.target_modules=q_proj,v_proj" in cmd

    def test_expert_only_flag(self, spec):
        spec.method = "expert_only"
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--policy.train_expert_only=true" in cmd

    def test_val_split_episodes_flag(self, spec):
        spec.val_episodes = 2  # total 10 -> train on [0..7]
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        ep_flags = [c for c in cmd if c.startswith("--dataset.episodes=")]
        assert ep_flags
        assert ep_flags[0] == "--dataset.episodes=[0, 1, 2, 3, 4, 5, 6, 7]"

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
        assert str(cfg.policy.pretrained_path) == "lerobot/act_aloha_sim"
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
        assert cfg.policy.use_peft is True

    def test_val_split_episodes(self, spec):
        pytest.importorskip("lerobot")
        spec.val_episodes = 2  # total 10 -> [0..7]
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        assert cfg.dataset.episodes == [0, 1, 2, 3, 4, 5, 6, 7]


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
        assert any("steps must be > 0" in p for p in problems)

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
        (last / "train_config.json").write_text("{}")
        spec.output_dir = str(tmp_path / "out")
        spec.resume = True
        cfg = LerobotTrainer(device="cpu").build_config(spec)
        # checkpoint_path is pretrained_model.parent.parent == the "last" dir.
        assert str(cfg.checkpoint_path) == str(last.parent)


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
        assert LerobotTrainer()._val_split_episodes(spec) is None


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


class TestSampleWeightingRABC:
    """RA-BC sample-weighting wiring: extra['sample_weighting'] -> nested SampleWeightingConfig.

    Regression for the folding recipe's headline ablation (HQ + RA-BC + relative
    actions). lerobot >= 0.5.2 configures RA-BC through a NESTED
    ``SampleWeightingConfig`` on ``TrainPipelineConfig`` (``cfg.sample_weighting``,
    fields ``type`` / ``progress_path`` / ``head_mode`` / ``kappa`` / ``epsilon``),
    replacing the flat ``use_rabc`` / ``rabc_*`` fields of earlier 0.5.x. The
    trainer forwards the friendly ``sample_weighting`` dict (whose keys match
    those fields 1:1) into that config. Before this migration the trainer set the
    removed flat fields and raised "no 'use_rabc'" against lerobot 0.5.2, so RA-BC
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
        # must raise an actionable ValueError ("requires lerobot >= 0.5.2"), not
        # leak the raw ImportError from the internal SampleWeightingConfig import.
        pytest.importorskip("lerobot.utils.sample_weighting")
        import sys

        monkeypatch.setitem(sys.modules, "lerobot.utils.sample_weighting", None)
        with pytest.raises(ValueError, match="requires lerobot >= 0.5.2"):
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
    ``is_reward_model_training`` path. Requires lerobot >= 0.5.2 (the
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

    def test_build_command_emits_reward_model_flags(self, dataset_root, tmp_path):
        spec = self._sarm_spec(dataset_root, tmp_path, image_key="observation.images.base")
        cmd = LerobotTrainer(device="cpu").build_command(spec)
        assert "--reward_model.type=sarm" in cmd
        assert "--reward_model.annotation_mode=single_stage" in cmd
        assert "--reward_model.image_key=observation.images.base" in cmd
        # A reward-model run does not train a policy -> no --policy.* flags.
        assert not any(c.startswith("--policy.") for c in cmd)

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
        # Only the pi0 family exposes use_relative_actions; build_config must
        # fail fast for any other policy rather than drop the flag silently
        # (which would train an ordinary absolute-action policy unnoticed).
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
