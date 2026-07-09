"""Tests for Gr00tTrainer: factory wiring, validate, command building.

Offline/pure - does not require an Isaac-GR00T checkout to run (uses a fake
groot_root with a stub launch_finetune.py for the happy-path command tests).
"""

import importlib
import json
import os
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.groot import Gr00tTrainer


@pytest.fixture
def dataset_root(tmp_path):
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"total_episodes": 10}))
    return str(tmp_path)


@pytest.fixture
def fake_groot_root(tmp_path):
    """A fake Isaac-GR00T checkout with a stub launch_finetune.py."""
    script = tmp_path / "gr00t" / "experiment" / "launch_finetune.py"
    script.parent.mkdir(parents=True)
    script.write_text("# stub\n")
    return str(tmp_path)


@pytest.fixture
def spec(dataset_root, tmp_path, fake_groot_root):
    return TrainSpec(
        dataset_root=dataset_root,
        base_model="nvidia/GR00T-N1.5-3B",
        output_dir=str(tmp_path / "out"),
        embodiment="GR1",
        steps=500,
        global_batch_size=32,
        learning_rate=1e-4,
        save_freq=100,
        extra={"groot_root": fake_groot_root},
    )


class TestFactoryWiring:
    def test_resolves_from_registry(self):
        t = create_trainer("groot")
        assert isinstance(t, Gr00tTrainer)
        assert t.provider_name == "groot"

    def test_hardware_floor(self):
        assert create_trainer("groot").hardware_floor["min_gpus"] == 1


class TestValidate:
    def test_clean(self, spec):
        assert Gr00tTrainer().validate(spec) == []

    def test_multi_node_rejected(self, spec):
        spec.num_nodes = 2
        problems = Gr00tTrainer().validate(spec)
        assert any("multi-node" in p for p in problems)

    def test_embodiment_required(self, spec):
        spec.embodiment = None
        problems = Gr00tTrainer().validate(spec)
        assert any("embodiment is required" in p for p in problems)

    def test_missing_groot_root(self, spec, monkeypatch):
        monkeypatch.delenv("GR00T_ROOT", raising=False)
        spec.extra.pop("groot_root")
        problems = Gr00tTrainer().validate(spec)
        assert any("Isaac-GR00T checkout not found" in p for p in problems)

    def test_bad_modality_config_path(self, spec):
        spec.extra["modality_config_path"] = "/does/not/exist.py"
        problems = Gr00tTrainer().validate(spec)
        assert any("modality_config_path does not exist" in p for p in problems)

    @pytest.mark.parametrize(
        ("mutate", "expected"),
        [
            (lambda s: setattr(s, "dataset_root", ""), "dataset_root is required"),
            (lambda s: setattr(s, "base_model", ""), "base_model is required"),
            (lambda s: setattr(s, "output_dir", ""), "output_dir is required"),
            (lambda s: setattr(s, "method", "bogus"), "unsupported method 'bogus'"),
            (lambda s: setattr(s, "steps", 0), "steps must be > 0"),
        ],
    )
    def test_required_field_branch(self, spec, mutate, expected):
        """Each missing/invalid core field surfaces its own problem string."""
        mutate(spec)
        problems = Gr00tTrainer().validate(spec)
        assert any(expected in p for p in problems), problems

    def test_dataset_root_without_info_json_rejected(self, spec, tmp_path):
        """A dataset_root dir lacking meta/info.json is not a v3 root."""
        empty = tmp_path / "not_a_dataset"
        empty.mkdir()
        spec.dataset_root = str(empty)
        problems = Gr00tTrainer().validate(spec)
        assert any("is not a LeRobotDataset v3 root" in p for p in problems), problems


class TestBuildCommand:
    def test_single_gpu_core_flags(self, spec):
        cmd = Gr00tTrainer().build_command(spec)
        joined = " ".join(cmd)
        assert "launch_finetune.py" in joined
        assert "--base_model_path=nvidia/GR00T-N1.5-3B" in cmd
        assert f"--dataset_path={spec.dataset_root}" in cmd
        assert "--embodiment_tag=GR1" in cmd
        assert "--max_steps=500" in cmd
        assert "--global_batch_size=32" in cmd
        assert "--save_steps=100" in cmd
        assert "--num_gpus=1" in cmd

    def test_default_tune_flags(self, spec):
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=false" in cmd
        assert "--tune_visual=false" in cmd
        assert "--tune_projector=true" in cmd
        assert "--tune_diffusion_model=true" in cmd

    def test_custom_tune_dict(self, spec):
        spec.tune = {"llm": True, "visual": True, "projector": False, "diffusion": False}
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=true" in cmd
        assert "--tune_visual=true" in cmd
        assert "--tune_projector=false" in cmd
        assert "--tune_diffusion_model=false" in cmd

    def test_frozen_backbone_method(self, spec):
        spec.method = "frozen_backbone"
        spec.tune = {"llm": True, "visual": True}  # should be forced off
        cmd = Gr00tTrainer().build_command(spec)
        assert "--tune_llm=false" in cmd
        assert "--tune_visual=false" in cmd

    def test_multi_gpu_uses_torchrun(self, spec):
        spec.num_gpus = 4
        cmd = Gr00tTrainer().build_command(spec)
        assert cmd[0] == "torchrun"
        assert "--nproc_per_node=4" in cmd
        assert "--num_gpus=4" in cmd

    def test_resume_flag(self, spec):
        spec.resume = True
        cmd = Gr00tTrainer().build_command(spec)
        assert "--resume_from_checkpoint" in cmd

    def test_modality_config_and_passthrough(self, spec, tmp_path):
        mcfg = tmp_path / "modality.py"
        mcfg.write_text("# modality\n")
        spec.extra["modality_config_path"] = str(mcfg)
        spec.extra["weight_decay"] = 1e-5
        cmd = Gr00tTrainer().build_command(spec)
        assert f"--modality_config_path={mcfg}" in cmd
        assert "--weight_decay=1e-05" in cmd
        # consumed keys must not leak
        assert not any(c.startswith("--groot_root=") for c in cmd)

    def test_no_augmentation_emits_no_augmentation_flags(self, spec):
        """With no augmentation, none of the augmentation flags appear."""
        cmd = Gr00tTrainer().build_command(spec)
        assert not any(c.startswith("--random_rotation_angle=") for c in cmd)
        assert not any(c.startswith("--color_jitter_params=") for c in cmd)
        assert not any(c.startswith("--extra_augmentation_config=") for c in cmd)

    def test_augmentation_native_and_extra_map_to_distinct_flags(self, spec):
        """build_command mirrors build_finetune_config: random_rotation_angle
        and color_jitter_params are native flags; any other key is bundled into
        --extra_augmentation_config JSON that excludes those native keys.
        """
        spec.augmentation = {
            "random_rotation_angle": 30,
            "color_jitter_params": [0.3, 0.3, 0.3, 0.1],
            "mixup_alpha": 0.4,
        }
        cmd = Gr00tTrainer().build_command(spec)
        assert "--random_rotation_angle=30" in cmd
        # color_jitter_params is a native field: its own flag, NOT folded into
        # --extra_augmentation_config.
        assert "--color_jitter_params=[0.3, 0.3, 0.3, 0.1]" in cmd
        extra_flags = [c for c in cmd if c.startswith("--extra_augmentation_config=")]
        assert len(extra_flags) == 1
        extra = json.loads(extra_flags[0].split("=", 1)[1])
        assert extra == {"mixup_alpha": 0.4}
        # native keys must NOT leak into the extra config bundle.
        assert "random_rotation_angle" not in extra
        assert "color_jitter_params" not in extra

    def test_extra_only_augmentation_is_not_dropped(self, spec):
        """An augmentation dict with only non-native keys still emits
        --extra_augmentation_config (regression: it used to be silently dropped
        unless color_jitter_params was also present).
        """
        spec.augmentation = {"mixup_alpha": 0.4}
        cmd = Gr00tTrainer().build_command(spec)
        extra_flags = [c for c in cmd if c.startswith("--extra_augmentation_config=")]
        assert len(extra_flags) == 1
        assert json.loads(extra_flags[0].split("=", 1)[1]) == {"mixup_alpha": 0.4}

    def test_color_jitter_only_emits_native_flag_no_extra(self, spec):
        """color_jitter_params alone emits its native flag and no extra bundle
        (regression: --extra_augmentation_config used to dump the whole dict).
        """
        spec.augmentation = {"color_jitter_params": [0.2, 0.2, 0.2, 0.05]}
        cmd = Gr00tTrainer().build_command(spec)
        assert "--color_jitter_params=[0.2, 0.2, 0.2, 0.05]" in cmd
        assert not any(c.startswith("--extra_augmentation_config=") for c in cmd)


# ---------------------------------------------------------------------------
# Fake Isaac-GR00T package tree so the config-lowering + in-process launch
# paths run with NO checkout, NO GPU, NO real dependency. Mirrors only the
# narrow surface Gr00tTrainer touches: FinetuneConfig (a dataclass), a default
# Config with mutable .data/.model/.training bags + .load_dict(), the
# EmbodimentTag resolver, and experiment.run().
# ---------------------------------------------------------------------------
@dataclass
class _FakeFinetuneConfig:
    base_model_path: str = ""
    dataset_path: str = ""
    embodiment_tag: Any = None
    output_dir: str = ""
    max_steps: int = 0
    global_batch_size: int = 0
    learning_rate: float = 0.0
    save_steps: int = 0
    num_gpus: int = 1
    tune_llm: bool = False
    tune_visual: bool = False
    tune_projector: bool = True
    tune_diffusion_model: bool = True
    resume_from_checkpoint: bool = False
    random_rotation_angle: float = 0.0
    color_jitter_params: Any = None
    extra_augmentation_config: str | None = None
    modality_config_path: str | None = None
    state_dropout_prob: float = 0.0
    experiment_name: str = "exp"
    dataloader_num_workers: int = 4
    gradient_accumulation_steps: int = 1
    save_total_limit: int = 3
    use_wandb: bool = False
    weight_decay: float = 0.0
    warmup_ratio: float = 0.0
    wandb_project: str = "groot"
    shard_size: int = 1
    episode_sampling_rate: float = 1.0
    num_shards_per_epoch: int = 1
    save_only_model: bool = False
    skip_weight_loading: bool = False
    lr_scheduler_type: str = "cosine"  # a real field, for passthrough-allowlist


class _Bucket:
    """Mutable attribute bag; unset attributes read as None."""

    def __getattr__(self, name: str) -> Any:
        return None


class _FakeRunConfig:
    def __init__(self) -> None:
        self.data = _Bucket()
        self.model = _Bucket()
        self.training = _Bucket()
        self.load_config_path: Any = "unset"
        self.loaded_dict: dict[str, Any] | None = None

    def load_dict(self, d: dict[str, Any]) -> "_FakeRunConfig":
        self.loaded_dict = d
        return self


class _FakeEmbodimentTag:
    def __init__(self, value: str) -> None:
        self.value = value

    @classmethod
    def resolve(cls, tag: Any) -> "_FakeEmbodimentTag":
        return tag if isinstance(tag, _FakeEmbodimentTag) else cls(str(tag))


def _install_fake_gr00t(monkeypatch, run_recorder=None):
    """Register a fake ``gr00t`` package tree; return the FinetuneConfig class."""

    def _mod(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        monkeypatch.setitem(sys.modules, name, m)
        return m

    gr00t = _mod("gr00t")
    _mod("gr00t.configs")
    ft_mod = _mod("gr00t.configs.finetune_config")
    base_cfg = _mod("gr00t.configs.base_config")
    _mod("gr00t.data")
    emb_mod = _mod("gr00t.data.embodiment_tags")
    _mod("gr00t.experiment")
    exp_mod = _mod("gr00t.experiment.experiment")

    ft_mod.FinetuneConfig = _FakeFinetuneConfig
    base_cfg.get_default_config = _FakeRunConfig
    emb_mod.EmbodimentTag = _FakeEmbodimentTag

    def _run(config):
        if run_recorder is not None:
            run_recorder.append(config)

    exp_mod.run = _run
    return gr00t


class TestBuildFinetuneConfig:
    """build_finetune_config() lowers a TrainSpec into GR00T's FinetuneConfig."""

    def test_core_fields_mapped(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        ft = Gr00tTrainer().build_finetune_config(spec)
        assert ft.base_model_path == "nvidia/GR00T-N1.5-3B"
        assert ft.dataset_path == spec.dataset_root
        assert ft.embodiment_tag == "GR1"
        assert ft.max_steps == 500
        assert ft.global_batch_size == 32
        assert ft.save_steps == 100
        # default tune: projector + diffusion on, llm/visual off
        assert ft.tune_projector is True
        assert ft.tune_diffusion_model is True
        assert ft.tune_llm is False
        assert ft.tune_visual is False

    def test_augmentation_split_into_native_and_extra(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        spec.augmentation = {
            "random_rotation_angle": 15,
            "color_jitter_params": {"brightness": 0.2},
            "mixup_alpha": 0.4,  # neither native field -> extra_augmentation_config JSON
        }
        ft = Gr00tTrainer().build_finetune_config(spec)
        assert ft.random_rotation_angle == 15
        assert ft.color_jitter_params == {"brightness": 0.2}
        assert json.loads(ft.extra_augmentation_config) == {"mixup_alpha": 0.4}

    def test_extra_passthrough_allowlisted_by_dataclass_fields(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        spec.extra["lr_scheduler_type"] = "linear"  # real FinetuneConfig field
        spec.extra["not_a_real_field"] = "ignored"  # silently dropped
        ft = Gr00tTrainer().build_finetune_config(spec)
        assert ft.lr_scheduler_type == "linear"
        assert not hasattr(ft, "not_a_real_field")

    def test_modality_config_path_forwarded(self, spec, tmp_path, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        mcfg = tmp_path / "modality.py"
        mcfg.write_text("# modality\n")
        spec.extra["modality_config_path"] = str(mcfg)
        ft = Gr00tTrainer().build_finetune_config(spec)
        assert ft.modality_config_path == str(mcfg)


class TestBuildRunConfig:
    """_build_run_config() lowers a FinetuneConfig into the run Config."""

    def test_dataset_and_embodiment_lowered(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        t = Gr00tTrainer()
        ft = t.build_finetune_config(spec)
        cfg = t._build_run_config(ft)
        ds = cfg.loaded_dict["data"]["datasets"][0]
        assert ds["dataset_paths"] == [spec.dataset_root]
        assert ds["embodiment_tag"] == "GR1"
        assert cfg.load_config_path is None

    def test_training_knobs_forwarded(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        t = Gr00tTrainer()
        cfg = t._build_run_config(t.build_finetune_config(spec))
        assert cfg.training.max_steps == 500
        assert cfg.training.global_batch_size == 32
        assert cfg.training.learning_rate == spec.learning_rate
        assert cfg.training.start_from_checkpoint == "nvidia/GR00T-N1.5-3B"
        assert cfg.training.output_dir == spec.output_dir

    def test_multi_path_dataset_split_on_pathsep(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        t = Gr00tTrainer()
        ft = t.build_finetune_config(spec)
        ft.dataset_path = os.pathsep.join(["/a/ds1", "/b/ds2"])
        cfg = t._build_run_config(ft)
        assert cfg.loaded_dict["data"]["datasets"][0]["dataset_paths"] == ["/a/ds1", "/b/ds2"]

    def test_extra_augmentation_json_decoded(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        t = Gr00tTrainer()
        ft = t.build_finetune_config(spec)
        ft.extra_augmentation_config = json.dumps({"mixup_alpha": 0.4})
        cfg = t._build_run_config(ft)
        assert cfg.model.extra_augmentation_config == {"mixup_alpha": 0.4}


class TestLoadModalityConfig:
    def test_imports_py_module_and_extends_sys_path(self, tmp_path):
        mod = tmp_path / "my_modality_cfg.py"
        mod.write_text("LOADED = True\n")
        Gr00tTrainer._load_modality_config(str(mod))
        assert str(tmp_path) in sys.path
        loaded = importlib.import_module("my_modality_cfg")
        assert loaded.LOADED is True

    def test_missing_path_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            Gr00tTrainer._load_modality_config(str(tmp_path / "nope.py"))

    def test_non_py_suffix_raises(self, tmp_path):
        bad = tmp_path / "cfg.json"
        bad.write_text("{}")
        with pytest.raises(FileNotFoundError):
            Gr00tTrainer._load_modality_config(str(bad))


class TestLatestCheckpoint:
    def test_none_when_dir_absent(self, tmp_path):
        assert Gr00tTrainer().latest_checkpoint(str(tmp_path / "missing")) is None

    def test_none_when_no_checkpoints(self, tmp_path):
        (tmp_path / "logs").mkdir()
        assert Gr00tTrainer().latest_checkpoint(str(tmp_path)) is None

    def test_picks_highest_step(self, tmp_path):
        for step in (100, 2000, 500):
            (tmp_path / f"checkpoint-{step}").mkdir()
        (tmp_path / "checkpoint-notanumber").mkdir()  # unparsable -> step -1
        latest = Gr00tTrainer().latest_checkpoint(str(tmp_path))
        assert latest == str(tmp_path / "checkpoint-2000")


class TestTrainInProcess:
    """train() single-GPU path: validate -> build configs -> call run()."""

    def test_validation_failure_short_circuits(self, spec):
        spec.embodiment = None  # GR00T requires it
        result = Gr00tTrainer().train(spec)
        assert result.status == "error"
        assert "validation failed" in result.message
        assert result.job_id == ""

    def test_success_calls_run_and_reports_checkpoint(self, spec, monkeypatch):
        recorder: list = []
        _install_fake_gr00t(monkeypatch, run_recorder=recorder)
        os.makedirs(spec.output_dir, exist_ok=True)
        (Path(spec.output_dir) / "checkpoint-300").mkdir()
        result = Gr00tTrainer().train(spec)
        assert result.status == "success"
        assert result.job_id.startswith("groot-")
        assert result.checkpoint_dir == str(Path(spec.output_dir) / "checkpoint-300")
        assert len(recorder) == 1  # experiment.run() was invoked exactly once

    def test_run_exception_converted_to_error_result(self, spec, monkeypatch):
        _install_fake_gr00t(monkeypatch)
        import gr00t.experiment.experiment as exp  # noqa: PLC0415

        def _boom(config):
            raise RuntimeError("CUDA OOM")

        monkeypatch.setattr(exp, "run", _boom)
        result = Gr00tTrainer().train(spec)
        assert result.status == "error"
        assert "RuntimeError" in result.message
        assert "CUDA OOM" in result.message

    def test_missing_gr00t_package_returns_install_hint(self, spec, monkeypatch):
        # Ensure no fake gr00t is registered and a real import fails.
        for name in list(sys.modules):
            if name == "gr00t" or name.startswith("gr00t."):
                monkeypatch.delitem(sys.modules, name, raising=False)
        monkeypatch.setattr(
            "strands_robots.training.groot.importlib.import_module",
            lambda name: (_ for _ in ()).throw(ImportError(f"no {name}")),
        )
        result = Gr00tTrainer().train(spec)
        assert result.status == "error"
        assert "Isaac-GR00T is not importable" in result.message


class TestResolveTune:
    def test_unknown_keys_ignored(self, spec):
        spec.tune = {"llm": True, "bogus": True}
        merged = Gr00tTrainer()._resolve_tune(spec)
        assert merged["llm"] is True
        assert "bogus" not in merged

    def test_frozen_backbone_forces_backbone_off(self, spec):
        spec.method = "frozen_backbone"
        spec.tune = {"llm": True, "visual": True, "diffusion": True}
        merged = Gr00tTrainer()._resolve_tune(spec)
        assert merged["llm"] is False
        assert merged["visual"] is False
        assert merged["diffusion"] is True
