"""GR00T trainer - wrapper over Isaac-GR00T's ``launch_finetune``.

GR00T N1.7 ships its own post-training pipeline (NOT lerobot): a
``FinetuneConfig`` dataclass. ``launch_finetune.py`` is only a thin
``__main__`` shim (build ``FinetuneConfig`` via tyro → lower to a ``Config``
→ call ``experiment.run(config)``); it has NO reusable function. So we
reproduce that translation here and call ``gr00t.experiment.experiment.run``
DIRECTLY as a library - same interpreter, no argv, no nested ``python``.
Multi-GPU uses torch's programmatic ``elastic_launch`` (the engine behind
``torchrun``); each worker builds the ``Config`` and calls ``run`` - no
``torchrun`` binary, no shell.

This adapter translates a provider-agnostic
:class:`~strands_robots.training.base.TrainSpec` into the ``launch_finetune``
argv. Mapping highlights:

* ``base_model``        → ``--base_model_path``
* ``dataset_root``      → ``--dataset_path``
* ``embodiment``        → ``--embodiment_tag`` (REQUIRED by GR00T)
* ``steps``             → ``--max_steps``
* ``global_batch_size`` → ``--global_batch_size``
* ``learning_rate``     → ``--learning_rate``
* ``save_freq``         → ``--save_steps``
* ``resume``            → ``--resume_from_checkpoint``
* ``tune`` dict         → ``--tune_llm/--tune_visual/--tune_projector/--tune_diffusion_model``
* ``augmentation``      → ``--random_rotation_angle`` / ``--color_jitter_params``
                          / ``--extra_augmentation_config`` (JSON)
* ``extra['modality_config_path']`` → ``--modality_config_path``

GR00T checkpoints are HF-native, so :meth:`export` is the default passthrough.
The Isaac-GR00T checkout is resolved from the ``GR00T_ROOT`` env var or
``extra['groot_root']`` - needed so we can ``chdir`` there for relative configs.
Install Isaac-GR00T from source (per its README) and ensure ``gr00t`` is
importable from the active interpreter; this trainer drives it as a Python
library, NOT by invoking another interpreter.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import logging
import os
import time
from typing import Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

# GR00T's tune flags - the model-tuning surface lerobot does NOT have.
# Sensible default mirrors FinetuneConfig defaults (projector + diffusion on).
_DEFAULT_TUNE = {"llm": False, "visual": False, "projector": True, "diffusion": True}

_SUPPORTED_METHODS = {"full", "frozen_backbone", "expert_only"}

_INSTALL_HINT = (
    "Isaac-GR00T is not importable from this interpreter. Install it from source "
    "(see https://github.com/NVIDIA/Isaac-GR00T or the path passed via "
    "groot_root / GR00T_ROOT) into the *same* Python that imports strands_robots "
    "- e.g. `pip install -e $GR00T_ROOT`."
)


def _import_groot_module(qualname: str) -> Any:
    """Import ``gr00t.<qualname>`` or raise a helpful ImportError."""
    full = f"gr00t.{qualname}"
    try:
        return importlib.import_module(full)
    except ImportError as e:  # pragma: no cover - exercised in integration
        raise ImportError(f"{_INSTALL_HINT} (failed to import {full})") from e


class Gr00tTrainer(Trainer):
    """Post-tune an NVIDIA GR00T N1.x policy via Isaac-GR00T launch_finetune.

    Args:
        groot_root: Path to the Isaac-GR00T checkout (used to ``chdir`` so
            relative configs/datasets resolve, and as the validation target
            for ``launch_finetune.py``). Falls back to the ``GR00T_ROOT`` env
            var, then ``TrainSpec.extra['groot_root']``. The package itself
            is loaded via :func:`importlib.import_module` from the active
            interpreter - install from source; ``GR00T_ROOT`` is for runtime
            config resolution, not the interpreter path.
    """

    def __init__(
        self,
        groot_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.groot_root = groot_root or os.environ.get("GR00T_ROOT")

    @property
    def provider_name(self) -> str:
        return "groot"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # N1.x fine-tune fits one modern GPU (lite); multi-GPU recommended.
        return {"min_gpus": 1, "min_vram_gb": 24, "multinode": False}

    def _resolve_groot_root(self, spec: TrainSpec) -> str | None:
        return self.groot_root or spec.extra.get("groot_root")

    def _launch_script(self, root: str) -> str:
        return os.path.join(root, "gr00t", "experiment", "launch_finetune.py")

    def _resolve_tune(self, spec: TrainSpec) -> dict[str, bool]:
        merged = dict(_DEFAULT_TUNE)
        merged.update({k: bool(v) for k, v in (spec.tune or {}).items() if k in _DEFAULT_TUNE})
        # method=frozen_backbone => freeze llm+visual, keep projector/diffusion.
        if spec.method == "frozen_backbone":
            merged["llm"] = False
            merged["visual"] = False
        return merged

    def validate(self, spec: TrainSpec) -> list[str]:
        problems: list[str] = self._security_problems(spec)

        if not spec.dataset_root:
            problems.append("dataset_root is required")
        elif not os.path.isfile(os.path.join(spec.dataset_root, "meta", "info.json")):
            problems.append(
                f"dataset_root is not a LeRobotDataset v3 root "
                f"(missing {os.path.join(spec.dataset_root, 'meta', 'info.json')})"
            )

        if not spec.base_model:
            problems.append("base_model is required (--base_model_path)")
        if not spec.output_dir:
            problems.append("output_dir is required")
        if not spec.embodiment:
            problems.append("embodiment is required for GR00T (--embodiment_tag)")

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(
                f"unsupported method '{spec.method}' for GR00T "
                f"(expected one of {sorted(_SUPPORTED_METHODS)}); "
                f"use tune={{...}} for fine-grained control"
            )
        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if spec.num_nodes > 1:
            problems.append(
                f"num_nodes={spec.num_nodes}: multi-node GR00T needs a per-node "
                "rendezvous launcher; this in-process trainer runs single-node only. "
                "Run one Gr00tTrainer per node under your own launcher, or use num_nodes=1."
            )

        root = self._resolve_groot_root(spec)
        if not root:
            problems.append(
                "Isaac-GR00T checkout not found; set GR00T_ROOT, pass groot_root=..., or extra['groot_root']"
            )
        elif not os.path.isfile(self._launch_script(root)):
            problems.append(
                f"launch_finetune.py not found under groot_root={root} (expected {self._launch_script(root)})"
            )

        mcfg = spec.extra.get("modality_config_path")
        if mcfg and not os.path.isfile(mcfg):
            problems.append(f"modality_config_path does not exist: {mcfg}")

        return problems

    def build_command(self, spec: TrainSpec) -> list[str]:
        """PURE argv-parity helper - the launch_finetune CLI the config maps to.

        NOT used to launch training (``train()`` builds a ``FinetuneConfig`` +
        ``Config`` and calls ``gr00t.experiment.experiment.run`` directly).
        Retained so ``test_native_parity`` can assert our flag mapping against
        the real ``FinetuneConfig`` dataclass fields, and as a readable
        description of the equivalent command.
        """
        root = self._resolve_groot_root(spec)
        script = self._launch_script(root) if root else "launch_finetune.py"

        if spec.num_gpus > 1:
            launcher = [
                "torchrun",
                f"--nproc_per_node={spec.num_gpus}",
                f"--master_port={spec.extra.get('master_port', 29500)}",
                script,
            ]
        else:
            launcher = ["python", script]

        cmd = [
            *launcher,
            f"--base_model_path={spec.base_model}",
            f"--dataset_path={spec.dataset_root}",
            f"--embodiment_tag={spec.embodiment}",
            f"--output_dir={spec.output_dir}",
            f"--max_steps={spec.steps}",
            f"--global_batch_size={spec.global_batch_size}",
            f"--save_steps={spec.save_freq}",
            f"--num_gpus={spec.num_gpus}",
        ]
        if spec.learning_rate is not None:
            cmd.append(f"--learning_rate={spec.learning_rate}")

        tune = self._resolve_tune(spec)
        cmd.append(f"--tune_llm={'true' if tune['llm'] else 'false'}")
        cmd.append(f"--tune_visual={'true' if tune['visual'] else 'false'}")
        cmd.append(f"--tune_projector={'true' if tune['projector'] else 'false'}")
        cmd.append(f"--tune_diffusion_model={'true' if tune['diffusion'] else 'false'}")

        if spec.augmentation:
            # Mirror build_finetune_config exactly: random_rotation_angle and
            # color_jitter_params are native FinetuneConfig fields; every other
            # augmentation key is bundled into --extra_augmentation_config JSON.
            if "random_rotation_angle" in spec.augmentation:
                cmd.append(f"--random_rotation_angle={spec.augmentation['random_rotation_angle']}")
            if "color_jitter_params" in spec.augmentation:
                cmd.append(f"--color_jitter_params={spec.augmentation['color_jitter_params']}")
            extra_aug = {
                k: v for k, v in spec.augmentation.items() if k not in ("random_rotation_angle", "color_jitter_params")
            }
            if extra_aug:
                cmd.append(f"--extra_augmentation_config={json.dumps(extra_aug)}")

        mcfg = spec.extra.get("modality_config_path")
        if mcfg:
            cmd.append(f"--modality_config_path={mcfg}")

        if spec.resume:
            cmd.append("--resume_from_checkpoint")

        # Passthrough: remaining extra.* as --key=value (skip consumed keys).
        _consumed = {"groot_root", "modality_config_path", "master_port"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            cmd.append(f"--{key}={value}")

        return cmd

    def build_finetune_config(self, spec: TrainSpec) -> Any:
        """Build GR00T's own ``FinetuneConfig`` object from a TrainSpec (pure).

        Returns an instance of ``gr00t.configs.finetune_config.FinetuneConfig``
        - the SAME typed object ``launch_finetune.py`` builds via tyro, but
        constructed directly from Python values (no argv). Requires the gr00t
        package importable.
        """
        FinetuneConfig = _import_groot_module("configs.finetune_config").FinetuneConfig

        import dataclasses

        tune = self._resolve_tune(spec)
        kwargs: dict[str, Any] = {
            "base_model_path": spec.base_model,
            "dataset_path": spec.dataset_root,
            "embodiment_tag": spec.embodiment,
            "output_dir": spec.output_dir,
            "max_steps": spec.steps,
            "global_batch_size": spec.global_batch_size,
            "save_steps": spec.save_freq,
            "num_gpus": spec.num_gpus,
            "tune_llm": tune["llm"],
            "tune_visual": tune["visual"],
            "tune_projector": tune["projector"],
            "tune_diffusion_model": tune["diffusion"],
            "resume_from_checkpoint": spec.resume,
        }
        if spec.learning_rate is not None:
            kwargs["learning_rate"] = spec.learning_rate
        if spec.augmentation:
            if "random_rotation_angle" in spec.augmentation:
                kwargs["random_rotation_angle"] = spec.augmentation["random_rotation_angle"]
            if "color_jitter_params" in spec.augmentation:
                kwargs["color_jitter_params"] = spec.augmentation["color_jitter_params"]
            extra_aug = {
                k: v for k, v in spec.augmentation.items() if k not in ("random_rotation_angle", "color_jitter_params")
            }
            if extra_aug:
                kwargs["extra_augmentation_config"] = json.dumps(extra_aug)
        if spec.extra.get("modality_config_path"):
            kwargs["modality_config_path"] = spec.extra["modality_config_path"]

        # Passthrough: any other extra.* that is a REAL FinetuneConfig field
        # (typed allowlist - an unknown key can never set an attribute).
        valid_fields = {f.name for f in dataclasses.fields(FinetuneConfig)}
        _consumed = {"groot_root", "master_port", "modality_config_path"}
        for key, value in spec.extra.items():
            if key in _consumed or key in kwargs:
                continue
            if key in valid_fields:
                kwargs[key] = value
            else:
                logger.warning("Gr00tTrainer: ignoring extra '%s' (not a FinetuneConfig field).", key)
        return FinetuneConfig(**kwargs)

    def _build_run_config(self, ft_config: Any) -> Any:
        """Lower a ``FinetuneConfig`` into the ``Config`` ``experiment.run`` consumes.

        Mirrors the body of ``launch_finetune.py``'s ``__main__`` exactly
        (verified against the Isaac-GR00T checkout), so calling ``run(config)``
        is behaviourally identical to launching the script - minus the process
        spawn and the tyro argv parse.
        """
        get_default_config = _import_groot_module("configs.base_config").get_default_config
        EmbodimentTag = _import_groot_module("data.embodiment_tags").EmbodimentTag

        ft_config.embodiment_tag = EmbodimentTag.resolve(ft_config.embodiment_tag)
        embodiment_tag = ft_config.embodiment_tag.value

        if ft_config.modality_config_path is not None:
            self._load_modality_config(ft_config.modality_config_path)

        dataset_paths = [p for p in ft_config.dataset_path.split(os.pathsep) if p]

        config = get_default_config().load_dict(
            {
                "data": {
                    "download_cache": False,
                    "datasets": [
                        {
                            "dataset_paths": dataset_paths,
                            "mix_ratio": 1.0,
                            "embodiment_tag": embodiment_tag,
                        }
                    ],
                }
            }
        )
        config.load_config_path = None

        config.model.tune_llm = ft_config.tune_llm
        config.model.tune_visual = ft_config.tune_visual
        config.model.tune_projector = ft_config.tune_projector
        config.model.tune_diffusion_model = ft_config.tune_diffusion_model
        config.model.state_dropout_prob = ft_config.state_dropout_prob
        config.model.random_rotation_angle = ft_config.random_rotation_angle
        config.model.color_jitter_params = ft_config.color_jitter_params
        config.model.extra_augmentation_config = (
            json.loads(ft_config.extra_augmentation_config) if ft_config.extra_augmentation_config else None
        )
        config.model.load_bf16 = False
        config.model.reproject_vision = False
        config.model.model_name = "nvidia/Cosmos-Reason2-2B"
        config.model.backbone_trainable_params_fp32 = True
        config.model.use_relative_action = True

        config.training.experiment_name = ft_config.experiment_name
        config.training.start_from_checkpoint = ft_config.base_model_path
        config.training.optim = "adamw_torch"
        config.training.global_batch_size = ft_config.global_batch_size
        config.training.dataloader_num_workers = ft_config.dataloader_num_workers
        config.training.learning_rate = ft_config.learning_rate
        config.training.gradient_accumulation_steps = ft_config.gradient_accumulation_steps
        config.training.output_dir = ft_config.output_dir
        config.training.save_steps = ft_config.save_steps
        config.training.save_total_limit = ft_config.save_total_limit
        config.training.num_gpus = ft_config.num_gpus
        config.training.use_wandb = ft_config.use_wandb
        config.training.max_steps = ft_config.max_steps
        config.training.weight_decay = ft_config.weight_decay
        config.training.warmup_ratio = ft_config.warmup_ratio
        config.training.wandb_project = ft_config.wandb_project

        config.data.shard_size = ft_config.shard_size
        config.data.episode_sampling_rate = ft_config.episode_sampling_rate
        config.data.num_shards_per_epoch = ft_config.num_shards_per_epoch

        config.training.save_only_model = ft_config.save_only_model
        config.training.resume_from_checkpoint = ft_config.resume_from_checkpoint
        config.training.skip_weight_loading = ft_config.skip_weight_loading

        return config

    @staticmethod
    def _load_modality_config(modality_config_path: str) -> None:
        """Register a user modality config (.py), mirroring launch_finetune.py."""
        import sys
        from pathlib import Path

        path = Path(modality_config_path)
        if path.exists() and path.suffix == ".py":
            if str(path.parent) not in sys.path:
                sys.path.append(str(path.parent))
            importlib.import_module(path.stem)
            logger.info("Loaded modality config: %s", path)
        else:
            raise FileNotFoundError(f"Modality config path does not exist: {modality_config_path}")

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        self.prepare(spec)
        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        job_id = f"groot-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")

        # Single-GPU: pin one device so HF Trainer doesn't DataParallel-wrap
        # (the StopIteration crash documented in examples/finetune.sh).
        if spec.num_gpus <= 1:
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
        os.environ.setdefault("LOGURU_LEVEL", "INFO")

        logger.info(
            "Gr00tTrainer launching GR00T experiment.run() in-process: num_gpus=%d steps=%d",
            spec.num_gpus,
            spec.steps,
        )

        train_error: Exception | None = None
        try:
            if spec.num_gpus and spec.num_gpus > 1:
                # Multi-GPU: torch elastic agent spawns workers; each builds the
                # FinetuneConfig + Config and calls experiment.run() - Python
                # objects, no argv, no torchrun binary.
                groot_root = self._resolve_groot_root(spec)
                elastic_launch_callable(
                    _groot_worker,
                    nproc_per_node=spec.num_gpus,
                    nnodes=1,
                    run_id=job_id,
                    fn_args=(groot_root, spec, log_path),
                )
            else:
                ft_config = self.build_finetune_config(spec)
                run_config = self._build_run_config(ft_config)
                run = _import_groot_module("experiment.experiment").run
                call_callable(run, run_config, log_path=log_path)
        except ImportError as e:
            return TrainResult(status="error", job_id=job_id, message=str(e))
        except Exception as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Gr00tTrainer in-process run failed: %s", e)

        ckpt = self.latest_checkpoint(spec.output_dir)
        if train_error is not None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt,
                message=f"GR00T experiment.run() raised {type(train_error).__name__}: {train_error}; see {log_path}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt,
            message=f"GR00T fine-tune complete (in-process); log: {log_path}",
        )

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """GR00T (HF Trainer) writes ``checkpoint-<step>`` dirs in output_dir."""
        if not os.path.isdir(output_dir):
            return None
        ckpts = [
            d
            for d in os.listdir(output_dir)
            if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d))
        ]
        if not ckpts:
            return None

        def _step(name: str) -> int:
            try:
                return int(name.split("-", 1)[1])
            except (IndexError, ValueError):
                return -1

        best = max(ckpts, key=_step)
        return os.path.join(output_dir, best)


def _groot_worker(groot_root: str | None, spec: TrainSpec, log_path: str) -> None:
    """elastic_launch worker: build the GR00T Config and call run() in this worker.

    Runs in a torch-spawned worker (one per GPU). torch sets RANK / LOCAL_RANK /
    WORLD_SIZE; GR00T's experiment.run() + HF Trainer read those to shard. We do
    NOT pin CUDA_VISIBLE_DEVICES here - each worker sees all devices and selects
    by LOCAL_RANK. Only local rank 0 tees to the shared log file.
    """
    import os as _os

    trainer = Gr00tTrainer(groot_root=groot_root)
    ft_config = trainer.build_finetune_config(spec)
    run_config = trainer._build_run_config(ft_config)
    run = _import_groot_module("experiment.experiment").run

    is_rank0 = _os.environ.get("LOCAL_RANK", "0") == "0"
    call_callable(run, run_config, log_path=log_path if is_rank0 else None)
