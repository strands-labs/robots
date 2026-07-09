"""Train (post-tune) a policy - Strands Agent ``@tool`` wrapper.

Exposes the :class:`~strands_robots.training.base.Trainer` abstraction to an
agent. One tool, provider-agnostic: the ``provider`` argument selects the
backend (``lerobot_local`` / ``groot`` / ``cosmos3`` / ``mock``) and the SAME
arguments map onto each backend's native pipeline via ``create_trainer`` +
``TrainSpec``.

This closes the physical-AI data loop from natural language:

    record  ->  train_policy(...)  ->  create_policy(<checkpoint>)  ->  run

All training logic lives in ``strands_robots.training``; this file only parses
agent input and formats output in the Strands ``{status, content:[...]}``
convention (structured fields live in a ``{"json": ...}`` content block).
"""

from __future__ import annotations

import logging
from typing import Any

from strands.tools.decorator import tool

from strands_robots.training import TrainSpec, create_trainer, list_trainers

logger = logging.getLogger(__name__)


def _ok(text: str, data: dict[str, Any] | None = None) -> dict[str, Any]:
    """Canonical Strands tool result: ``{status, content:[...]}`` only.

    Structured data goes INSIDE the content list as a ``{"json": ...}`` block -
    NEVER as a sibling key of ``status``/``content`` (the result dict must not be
    extended beyond the two-key convention).
    """
    content: list[dict[str, Any]] = [{"text": text}]
    if data is not None:
        content.append({"json": data})
    return {"status": "success", "content": content}


def _err(text: str) -> dict[str, Any]:
    return {"status": "error", "content": [{"text": text}]}


@tool
def train_policy(
    action: str = "train",
    provider: str = "lerobot_local",
    dataset_root: str | None = None,
    dataset_repo_id: str | None = None,
    streaming: bool = False,
    base_model: str = "",
    output_dir: str | None = None,
    embodiment: str | None = None,
    steps: int = 10000,
    batch_size: int = 32,
    learning_rate: float | None = None,
    save_freq: int = 1000,
    num_gpus: int = 1,
    num_nodes: int = 1,
    resume: bool = False,
    seed: int | None = None,
    method: str = "full",
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_target_modules: str | None = None,
    tune: dict[str, bool] | None = None,
    val_episodes: int | None = None,
    augmentation: dict[str, Any] | None = None,
    fps: int | None = None,
    extra: dict[str, Any] | None = None,
    job_id: str | None = None,
) -> dict[str, Any]:
    """Post-tune (fine-tune) a robot policy on a recorded LeRobotDataset.

    Provider-agnostic: ``provider`` picks the training backend and the same
    arguments map onto its native pipeline. Closes the record -> train -> deploy
    loop - the produced checkpoint loads back via ``create_policy``.

    Args:
        action: One of:
            - ``"train"``    : validate + launch training (default).
            - ``"validate"`` : pure preflight only; report problems, launch nothing.
            - ``"status"``   : "RUNNING != learning" verdict for a job (needs ``job_id``).
            - ``"export"``   : produce a loadable artifact from a checkpoint
                               (needs ``output_dir``; uses the run's last checkpoint).
            - ``"list"``     : list available training providers.
        provider: Training backend / policy family - ``"lerobot_local"`` (act,
            diffusion, smolvla, pi0, pi05, ...), ``"groot"`` (NVIDIA GR00T),
            ``"cosmos3"`` (NVIDIA Cosmos3), or ``"mock"``. Same name as the
            inference provider in ``create_policy``.
        dataset_root: Path to a LeRobotDataset v3 root (has ``meta/info.json``) -
            exactly what ``Robot.stop_recording`` writes. Optional when
            ``dataset_repo_id`` is set (then it is the local cache root).
        dataset_repo_id: Hugging Face Hub dataset id (``org/name``) to train from
            the Hub instead of a local root - required to ``streaming`` a large
            (50-500 GB) Hub dataset without downloading it in full. lerobot only.
        streaming: Stream frames instead of materializing the dataset (lerobot
            ``StreamingLeRobotDataset``). With ``dataset_repo_id`` this streams
            Hub shards with bounded disk; with a local ``dataset_root`` it
            streams from disk with bounded RAM. lerobot only; ignored elsewhere.
        base_model: HF id or local checkpoint to post-tune from. For GR00T this
            is required (``--base_model_path``); ACT-from-scratch leaves it "".
        output_dir: Where checkpoints + logs go.
        embodiment: Embodiment tag (REQUIRED for GR00T; inferred by lerobot).
        steps: Total optimizer steps.
        batch_size: Global batch size (summed across GPUs).
        learning_rate: Optimizer learning rate. ``None`` (default) uses the
            backend's own default (the policy training preset for lerobot,
            GR00T's FinetuneConfig default, Cosmos's TOML default); an explicit
            value is honored by every backend.
        save_freq: Checkpoint cadence in steps.
        num_gpus: GPUs on this node (``>1`` -> accelerate/torchrun multi-GPU).
        num_nodes: Nodes (Cosmos HSDP / torchrun ``--nnodes``).
        resume: Resume from the latest checkpoint under ``output_dir``.
        seed: Master seed.
        method: Tuning strategy - ``"full"`` | ``"lora"`` | ``"expert_only"`` |
            ``"frozen_backbone"``. ``lora`` and ``expert_only`` are mutually
            exclusive.
        lora_r / lora_alpha / lora_target_modules: LoRA hyperparameters.
        tune: Fine-grained component toggles for GR00T
            (``{"llm","visual","projector","diffusion"}``).
        val_episodes: Hold out the LAST N episodes for validation.
        augmentation: Backend-specific augmentation dict.
        fps: Dataset control rate (when a backend needs it).
        extra: Backend-specific passthrough. lerobot: ``policy_type``,
            ``job_name``, any ``--key=value``. GR00T: ``groot_root``,
            ``modality_config_path``. Cosmos: ``cosmos_root``, ``sft_toml``.
        job_id: Job identifier for ``action="status"``.

    Returns:
        Canonical Strands result ``{status, content:[...]}`` (no sibling keys).
        For ``train``/``status``/``export`` the structured fields
        (``job_id``, ``checkpoint_dir``, ``exported_model``, ``metrics``) are in
        a ``{"json": ...}`` block inside ``content``, alongside a human-readable
        ``{"text": ...}`` block.

    Dependencies (per provider - the base ``[lerobot]`` extra is not always
    enough):
        - ``lerobot_local`` + ACT/diffusion: ``pip install 'strands-robots[lerobot]'``.
        - ``lerobot_local`` + ``smolvla``/``pi0``/``pi05``: add lerobot's
          ``[smolvla]``/``[pi]`` extra on top of ``strands-robots[lerobot]``
          (which pins ``lerobot>=0.6.0``). Those extras layer
          ``transformers>=5.4.0,<5.6.0`` (plus num2words / scipy); do NOT pin
          ``transformers==5.3.0`` - it conflicts with lerobot 0.6's transformers
          floor.
        - ``groot``/``cosmos3``: install the upstream package into THIS
          interpreter (the trainer imports it and calls its library functions
          in-process - no subprocess). Point ``extra['groot_root']``/``GR00T_ROOT``
          or ``extra['cosmos_root']``/``COSMOS_ROOT`` at the checkout for runtime
          config/recipe resolution.
        - torchcodec's ``.so`` must match the installed torch build exactly; a
          torch nightly load-fails a stable torchcodec (``undefined symbol``)
          and lerobot silently yields zero frames. See docs/training/overview.md.
    """
    try:
        if action == "list":
            return _ok("Available training providers:\n  " + "\n  ".join(list_trainers()))

        if action == "status":
            if not job_id:
                return _err("action='status' requires job_id")
            trainer = create_trainer(provider)
            res = trainer.status(job_id)
            return {
                "status": "success" if res.status != "error" else "error",
                "content": [
                    {"text": f"[{provider}] job {job_id}: {res.status}\n{res.message}\nmetrics: {res.metrics}"},
                    {
                        "json": {
                            "job_id": job_id,
                            "provider": provider,
                            "status": res.status,
                            "metrics": dict(res.metrics),
                        }
                    },
                ],
            }

        # All remaining actions need a spec.
        if not (dataset_root or dataset_repo_id) or not output_dir:
            return _err("a data source (dataset_root or dataset_repo_id) and output_dir are required")

        trainer = create_trainer(provider)
        spec = TrainSpec(
            dataset_root=dataset_root or "",
            dataset_repo_id=dataset_repo_id,
            streaming=streaming,
            base_model=base_model,
            output_dir=output_dir or "",
            embodiment=embodiment,
            steps=steps,
            global_batch_size=batch_size,
            learning_rate=learning_rate,
            save_freq=save_freq,
            num_gpus=num_gpus,
            num_nodes=num_nodes,
            resume=resume,
            seed=seed,
            method=method,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            lora_target_modules=lora_target_modules,
            tune=tune or {},
            val_episodes=val_episodes,
            augmentation=augmentation,
            fps=fps,
            extra=extra or {},
        )

        if action == "validate":
            problems = trainer.validate(spec)
            if problems:
                return _err("validation problems:\n  - " + "\n  - ".join(problems))
            return _ok(f"[{provider}] spec is valid and launchable.")

        if action == "export":
            # Gate on validate() FIRST, like train/validate - export consumes the
            # same agent-supplied spec (output_dir + extra reach trainer.export),
            # so it must not run with unvalidated input (path traversal, a leading
            # '-', or an arbitrary extra key). validate() runs _security_problems
            # plus the backend's own spec checks, fail-closed.
            problems = trainer.validate(spec)
            if problems:
                return _err("validation problems (nothing exported):\n  - " + "\n  - ".join(problems))
            ckpt = trainer.latest_checkpoint(output_dir)
            if not ckpt:
                return _err(f"no checkpoint found under {output_dir} to export")
            exported = trainer.export(spec, ckpt)
            return _ok(
                f"[{provider}] exported loadable artifact:\n{exported}\nLoad it with: create_policy('{exported}')",
                data={"provider": provider, "exported_model": exported},
            )

        if action == "train":
            problems = trainer.validate(spec)
            if problems:
                return _err("validation problems (nothing launched):\n  - " + "\n  - ".join(problems))
            res = trainer.train(spec)
            if res.status == "error":
                return _err(f"[{provider}] training failed: {res.message}")
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"[{provider}] {res.message}\n"
                            f"job_id: {res.job_id}\n"
                            f"checkpoint_dir: {res.checkpoint_dir}\n"
                            f"metrics: {res.metrics}\n"
                            f"Load the result with: create_policy('{res.checkpoint_dir}')"
                        )
                    },
                    {
                        "json": {
                            "provider": provider,
                            "job_id": res.job_id,
                            "status": res.status,
                            "checkpoint_dir": res.checkpoint_dir,
                            "exported_model": res.exported_model,
                            "metrics": dict(res.metrics),
                        }
                    },
                ],
            }

        return _err(f"Unknown action: {action}. Valid: train, validate, status, export, list")

    except Exception as e:  # noqa: BLE001 - tool boundary: report, don't crash the agent
        logger.exception("train_policy failed")
        return _err(f"train_policy error: {e}")
