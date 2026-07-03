"""Cosmos3 trainer - wrapper over cosmos_framework's SFT pipeline.

Cosmos3 has the most distinct pipeline of the three backends, and is the reason
the :class:`Trainer` ABC has optional ``prepare``/``export`` hooks:

* **prepare()** - the base HF checkpoint MUST be converted to PyTorch DCP
  (``cosmos_framework.scripts.convert_model_to_dcp``) before training.
  LeRobot/GR00T need no such step.
* **train()** - builds the cosmos ``Config`` via
  ``cosmos_framework.configs.toml_config.sft_config.load_experiment_from_toml``
  (TOML recipe + a Hydra ``key.path=value`` override LIST) and calls
  ``cosmos_framework.scripts.train.launch(config, args)`` DIRECTLY. (train.py
  has no reusable ``main()`` - only ``launch`` - verified against upstream.)
  Multi-GPU uses torch's programmatic ``elastic_launch`` (the engine behind
  ``torchrun``); each worker calls ``launch`` - no shell, no torchrun binary.
* **export()** - the trained DCP is converted back to HF safetensors via
  ``cosmos_framework.scripts.export_model`` so ``create_policy`` can consume
  it.

Multi-node HSDP maps ``num_nodes`` →
``model.config.parallelism.data_parallel_replicate_degree`` (intra-node shard
stays at ``nproc_per_node``). 8×H100 80 GB is the tested floor.

The cosmos_framework checkout is resolved from ``COSMOS_ROOT`` env var or
``extra['cosmos_root']``; the SFT recipe TOML from ``extra['sft_toml']``.

Install cosmos-framework from source (per its README) and ensure
``cosmos_framework`` is importable in the active Python - this trainer drives
it as a Python library, NOT as a subprocess invoking another interpreter.
"""

from __future__ import annotations

import importlib
import logging
import os
import time
from typing import Any

from strands_robots.training._inproc import call_callable, elastic_launch_callable
from strands_robots.training.base import Trainer, TrainResult, TrainSpec

logger = logging.getLogger(__name__)

_SUPPORTED_METHODS = {"full", "lora"}

_INSTALL_HINT = (
    "cosmos-framework is not importable from this interpreter. "
    "Install it from source (see https://github.com/nvidia-cosmos/cosmos-framework "
    "or the path passed via cosmos_root / COSMOS_ROOT) into the *same* Python "
    "that imports strands_robots - e.g. `pip install -e $COSMOS_ROOT`."
)


def _import_cosmos_module(qualname: str) -> Any:
    """Import ``cosmos_framework.<qualname>`` or raise a helpful ImportError.

    ``qualname`` is e.g. ``scripts.convert_model_to_dcp`` /
    ``scripts.train`` / ``scripts.export_model``. We resolve as a Python
    library so the trainer runs in-process - no nested ``python`` invocation.
    """
    full = f"cosmos_framework.{qualname}"
    try:
        return importlib.import_module(full)
    except ImportError as e:  # pragma: no cover - exercised in integration
        raise ImportError(f"{_INSTALL_HINT} (failed to import {full})") from e


class Cosmos3Trainer(Trainer):
    """Post-tune an NVIDIA Cosmos3 policy via cosmos_framework SFT.

    Args:
        cosmos_root: Path to the cosmos-framework checkout (used for
            ``torchrun``'s ``cwd`` so relative recipe/config paths resolve).
            Falls back to the ``COSMOS_ROOT`` env var, then
            ``TrainSpec.extra['cosmos_root']``. The package itself is loaded
            as a Python library via :func:`importlib.import_module` from the
            active interpreter - install from source per cosmos-framework's
            README; ``COSMOS_ROOT`` is for runtime config resolution, not the
            interpreter path.
    """

    def __init__(
        self,
        cosmos_root: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.cosmos_root = cosmos_root or os.environ.get("COSMOS_ROOT")

    @property
    def provider_name(self) -> str:
        return "cosmos3"

    @property
    def hardware_floor(self) -> dict[str, Any]:
        # SFT tested on 8xH100 80GB; HSDP multi-node beyond.
        return {"min_gpus": 8, "min_vram_gb": 80, "multinode": True}

    def _resolve_cosmos_root(self, spec: TrainSpec) -> str | None:
        return self.cosmos_root or spec.extra.get("cosmos_root")

    def _dcp_path(self, spec: TrainSpec) -> str:
        """Where prepare() writes (and train() reads) the DCP base checkpoint."""
        return str(spec.extra.get("dcp_path", os.path.join(spec.output_dir, "_dcp_base")))

    def _nproc(self, spec: TrainSpec) -> int:
        return max(1, spec.num_gpus)

    def latest_checkpoint(self, output_dir: str) -> str | None:
        """Return the trained DCP checkpoint directory, or None.

        cosmos_framework writes its training output (DCP shards) under
        ``output_dir``; that directory is what ``export`` converts to HF
        safetensors. We treat ``output_dir`` as the checkpoint once it exists
        and is non-empty (the DCP base lives in the ``_dcp_base`` sibling, which
        we exclude). Returns None before any training output appears.
        """
        if not os.path.isdir(output_dir):
            return None
        # Anything other than our own _dcp_base / _exported scratch dirs means
        # training has written checkpoint state here.
        entries = [e for e in os.listdir(output_dir) if e not in ("_dcp_base", "_exported") and not e.endswith(".log")]
        return output_dir if entries else None

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
            problems.append("base_model is required (HF checkpoint to convert to DCP)")
        if not spec.output_dir:
            problems.append("output_dir is required")

        if spec.method not in _SUPPORTED_METHODS:
            problems.append(
                f"unsupported method '{spec.method}' for Cosmos3 (expected one of {sorted(_SUPPORTED_METHODS)})"
            )
        if spec.steps <= 0:
            problems.append(f"steps must be > 0, got {spec.steps}")

        if not spec.extra.get("sft_toml"):
            problems.append(
                "Cosmos3 needs a recipe TOML; pass extra['sft_toml']=<path> "
                "(selects the registered experiment + scalar knobs)"
            )
        elif not os.path.isfile(spec.extra["sft_toml"]):
            problems.append(f"sft_toml does not exist: {spec.extra['sft_toml']}")

        root = self._resolve_cosmos_root(spec)
        if not root:
            problems.append(
                "cosmos-framework checkout not found; set COSMOS_ROOT, pass cosmos_root=..., or extra['cosmos_root']"
            )
        elif not os.path.isdir(os.path.join(root, "cosmos_framework")):
            problems.append(f"cosmos_framework package not found under cosmos_root={root}")

        return problems

    def prepare(self, spec: TrainSpec) -> None:
        """Convert the base HF checkpoint to DCP (required before training).

        Skips if the DCP target already exists (idempotent). Calls
        ``cosmos_framework.scripts.convert_model_to_dcp.convert_model_to_dcp``
        DIRECTLY with a typed ``Args`` object - no subprocess, no argv.
        Verified against github.com/NVIDIA/cosmos-framework:
        ``convert_model_to_dcp(Args(checkpoint=CheckpointOverrides(
        checkpoint_path=<hf>), output_path=<dcp>))``.
        """
        root = self._resolve_cosmos_root(spec)
        if not root:
            return
        dcp = self._dcp_path(spec)
        if os.path.isdir(os.path.join(dcp, "model")):
            logger.info("Cosmos3 DCP base already present at %s; skipping convert", dcp)
            return

        convert_mod = _import_cosmos_module("scripts.convert_model_to_dcp")
        CheckpointOverrides = _import_cosmos_module("inference.common.args").CheckpointOverrides

        os.makedirs(os.path.dirname(os.path.abspath(dcp)) or ".", exist_ok=True)
        args = convert_mod.Args(
            checkpoint=CheckpointOverrides(checkpoint_path=spec.base_model),
            output_path=dcp,
        )
        logger.info("Cosmos3Trainer converting base->DCP in-process (library call)")
        call_callable(convert_mod.convert_model_to_dcp, args)

    def build_command(self, spec: TrainSpec) -> list[str]:
        """PURE argv-parity helper - the torchrun CLI the library call maps to.

        NOT used to launch training (``train()`` builds the cosmos ``Config`` via
        ``load_experiment_from_toml`` and calls ``train.launch(config, args)``
        directly, under ``elastic_launch`` for multi-GPU). Retained so
        ``test_native_parity`` can assert ``--sft-toml`` + module path against
        the real cosmos ``train.py``, and as a human-readable description.
        """
        nproc = self._nproc(spec)
        cmd = [
            "torchrun",
            f"--nproc_per_node={nproc}",
            f"--nnodes={max(1, spec.num_nodes)}",
            "-m",
            "cosmos_framework.scripts.train",
            f"--sft-toml={spec.extra['sft_toml']}",
            "--",
            *self.build_overrides(spec),
        ]
        return cmd

    def build_overrides(self, spec: TrainSpec) -> list[str]:
        """Hydra ``key.path=value`` override LIST passed to load_experiment_from_toml.

        A LIST of ``key=value`` strings (NOT argv flags / a shell line), applied
        after the TOML so they win. Caller ``extra.*`` keys are gated by
        ``validate()``'s allowlist so a stray entry can't inject tokens.
        """
        overrides = [
            f"trainer.max_iter={spec.steps}",
            f"checkpoint.save_iter={spec.save_freq}",
            f"checkpoint.load_path={self._dcp_path(spec)}",
            f"dataloader_train.max_samples_per_batch={spec.global_batch_size}",
        ]
        if spec.learning_rate is not None:
            overrides.append(f"optimizer.lr={spec.learning_rate}")
        if spec.num_nodes > 1:
            overrides.append(f"model.config.parallelism.data_parallel_replicate_degree={spec.num_nodes}")
        if spec.seed is not None:
            overrides.append(f"trainer.seed={spec.seed}")
        _consumed = {"cosmos_root", "sft_toml", "dcp_path", "export_dir", "rdzv_endpoint"}
        for key, value in spec.extra.items():
            if key in _consumed:
                continue
            overrides.append(f"{key}={value}")
        return overrides

    def export(self, spec: TrainSpec, checkpoint_dir: str) -> str:
        """Convert the trained DCP back to HF safetensors (in-process library call).

        Calls ``cosmos_framework.scripts.export_model.export_model`` DIRECTLY
        with a typed ``Args`` object. Verified against upstream:
        ``export_model(Args(checkpoint=CheckpointOverrides(checkpoint_path=<dcp>),
        output_dir=<hf>))``. Falls back to the passthrough if cosmos_root absent.
        """
        root = self._resolve_cosmos_root(spec)
        out: str = str(spec.extra.get("export_dir") or os.path.join(spec.output_dir, "_exported"))
        if not root:
            return checkpoint_dir
        os.makedirs(os.path.dirname(os.path.abspath(out)) or ".", exist_ok=True)
        try:
            export_mod = _import_cosmos_module("scripts.export_model")
            CheckpointOverrides = _import_cosmos_module("inference.common.args").CheckpointOverrides

            args = export_mod.Args(
                checkpoint=CheckpointOverrides(checkpoint_path=checkpoint_dir),
                output_dir=out,
            )
            logger.info("Cosmos3Trainer exporting DCP->safetensors in-process (library call)")
            call_callable(export_mod.export_model, args)
        except Exception as e:  # noqa: BLE001 - export is best-effort; fall back
            logger.error("Cosmos3 export failed (%s); returning DCP checkpoint dir", e)
            return checkpoint_dir
        return out

    def train(self, spec: TrainSpec) -> TrainResult:
        problems = self.validate(spec)
        if problems:
            return TrainResult(
                status="error",
                job_id="",
                message="validation failed: " + "; ".join(problems),
            )

        parent = os.path.dirname(os.path.abspath(spec.output_dir)) or "."
        os.makedirs(parent, exist_ok=True)

        # prepare(): convert base -> DCP (idempotent, in-process library call).
        try:
            self.prepare(spec)
        except Exception as e:  # noqa: BLE001 - surface convert failure as result
            return TrainResult(
                status="error",
                job_id="",
                message=f"DCP conversion (prepare) failed: {e}",
            )

        # Verify the train entrypoint imports BEFORE spinning up workers.
        try:
            _import_cosmos_module("scripts.train")
        except ImportError as e:
            return TrainResult(status="error", job_id=f"cosmos3-{int(time.time())}", message=str(e))

        sft_toml = str(spec.extra["sft_toml"])
        overrides = self.build_overrides(spec)
        job_id = f"cosmos3-{int(time.time())}"
        log_path = os.path.join(parent, f"{os.path.basename(spec.output_dir)}.{job_id}.log")
        nproc = self._nproc(spec)

        logger.info(
            "Cosmos3Trainer launching train.launch() in-process: nproc=%d nnodes=%d steps=%d",
            nproc,
            max(1, spec.num_nodes),
            spec.steps,
        )

        train_error: Exception | None = None
        try:
            if nproc > 1 or spec.num_nodes > 1:
                # Multi-GPU/-node: torch elastic agent spawns workers; each builds
                # the cosmos Config and calls train.launch() - Python objects, no
                # argv, no torchrun binary.
                rdzv = str(spec.extra.get("rdzv_endpoint", "")) if spec.num_nodes > 1 else ""
                elastic_launch_callable(
                    _cosmos_worker,
                    nproc_per_node=nproc,
                    nnodes=max(1, spec.num_nodes),
                    rdzv_endpoint=rdzv,
                    run_id=job_id,
                    fn_args=(sft_toml, overrides, log_path),
                )
            else:
                _run_cosmos_launch(sft_toml, overrides, log_path=log_path)
        except Exception as e:  # noqa: BLE001 - convert ANY failure to a result
            train_error = e
            logger.error("Cosmos3Trainer in-process train failed: %s", e)

        ckpt = spec.output_dir
        if train_error is not None:
            return TrainResult(
                status="error",
                job_id=job_id,
                checkpoint_dir=ckpt,
                message=f"cosmos_framework train.launch() raised {type(train_error).__name__}: {train_error}; see {log_path}",
            )
        return TrainResult(
            status="success",
            job_id=job_id,
            checkpoint_dir=ckpt,
            message=f"Cosmos3 SFT complete (in-process); log: {log_path}",
        )


def _run_cosmos_launch(sft_toml: str, overrides: list[str], *, log_path: str | None = None) -> None:
    """Build the cosmos Config from TOML + overrides and call train.launch(config, args).

    Mirrors ``cosmos_framework/scripts/train.py``'s ``__main__`` (which has NO
    reusable ``main()``): it builds ``config`` via ``load_experiment_from_toml``
    and calls ``launch(config, args)``. We construct the same argparse-shaped
    ``args`` namespace with the non-deterministic, non-debug defaults the script
    uses for a real run, so calling ``launch`` is behaviourally identical to the
    CLI - minus the process spawn and argv parse.
    """
    import argparse

    load_experiment_from_toml = _import_cosmos_module("configs.toml_config.sft_config").load_experiment_from_toml
    launch = _import_cosmos_module("scripts.train").launch

    config = load_experiment_from_toml(sft_toml, extra_overrides=overrides)
    args = argparse.Namespace(
        sft_toml=sft_toml,
        opts=list(overrides),
        deterministic=False,
        attach_vscode_debugger=False,
        dryrun=False,
        config=sft_toml,
    )
    call_callable(launch, config, args, log_path=log_path)


def _cosmos_worker(sft_toml: str, overrides: list[str], log_path: str) -> None:
    """elastic_launch worker: build the cosmos Config and call train.launch() here.

    Runs in a torch-spawned worker (one per GPU). torch sets RANK / LOCAL_RANK /
    WORLD_SIZE; cosmos's distributed.init() reads them. Only local rank 0 tees to
    the shared log file to avoid interleaved writes.
    """
    import os as _os

    is_rank0 = _os.environ.get("LOCAL_RANK", "0") == "0"
    _run_cosmos_launch(sft_toml, overrides, log_path=log_path if is_rank0 else None)
