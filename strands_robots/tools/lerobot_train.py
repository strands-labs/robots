#!/usr/bin/env python3
"""LeRobot training tool: a thin local wrapper over ``lerobot-train``.

This tool closes the strands-robots data loop locally: record a LeRobot v3
dataset (see ``lerobot_teleoperate`` or ``Robot.start_recording``), then
fine-tune a policy on it here, then deploy the resulting checkpoint with
``create_policy("lerobot_local", ...)``. No cloud orchestration is involved;
the command this builds is the same ``python -m lerobot.scripts.lerobot_train``
invocation a user would run by hand, plus a few ergonomic guardrails.

Process lifecycle mirrors ``lerobot_teleoperate``: ``start`` launches a detached
background process tracked in an on-disk session store, and ``status``/``stop``/
``list`` manage it.
"""

import dataclasses
import json
import logging
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

import psutil
from strands import tool
from strands.types.tools import ToolContext

from strands_robots.utils import validation_split_error, validation_split_fraction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reuse the teleoperate session store so all robot sessions live together.
SESSION_DIR = Path.cwd() / ".strands_robots/.sessions"
SESSION_DIR.mkdir(parents=True, exist_ok=True)

# Policy families that train an action expert on top of a frozen VLM. Only these
# accept ``--policy.train_expert_only``; emitting it on any other policy is a hard
# error in lerobot, so callers that pass it on an unsupported policy are told why.
# The supported set is sourced LIVE from lerobot (the policy configs that declare a
# ``train_expert_only`` field) so it tracks lerobot instead of drifting against a
# hardcoded copy; the static snapshot below is the FALLBACK when lerobot is not
# importable. pi0_fast is intentionally absent - its config has no such field.
_EXPERT_ONLY_POLICIES_FALLBACK = frozenset({"pi0", "pi05", "smolvla"})


def _expert_only_policy_types() -> frozenset[str]:
    """LeRobot policy types whose config declares ``train_expert_only``.

    Read live from lerobot's ``PreTrainedConfig`` registry so this tracks
    lerobot (a policy that gains or loses expert-only support is picked up with
    no change here) instead of drifting against a hardcoded copy. Falls back to
    the documented static snapshot when lerobot is not importable.
    """
    try:
        import lerobot.policies  # noqa: F401  (register_subclass side effect)
        from lerobot.configs.policies import PreTrainedConfig

        return frozenset(
            name
            for name, cfg_cls in PreTrainedConfig.get_known_choices().items()
            if any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
        )
    except Exception:  # noqa: BLE001 - offline / lerobot missing -> static fallback
        return _EXPERT_ONLY_POLICIES_FALLBACK


def _policy_config_field_names(policy_type: str) -> frozenset[str] | None:
    """Field names declared by lerobot's config class for ``policy_type``.

    Read live from lerobot's ``PreTrainedConfig`` registry so per-policy config
    fields (e.g. ``dtype``, ``gradient_checkpointing``) are gated against the
    ACTUAL policy config instead of a hardcoded guess. Returns ``None`` when
    lerobot is not importable or ``policy_type`` is unknown, signaling callers to
    pass the corresponding flag through unguarded rather than guess.
    """
    try:
        import lerobot.policies  # noqa: F401  (register_subclass side effect)
        from lerobot.configs.policies import PreTrainedConfig

        cfg_cls = PreTrainedConfig.get_known_choices().get(policy_type)
        if cfg_cls is None:
            return None
        return frozenset(f.name for f in dataclasses.fields(cfg_cls))
    except Exception:  # noqa: BLE001 - offline / lerobot missing -> skip field gating
        return None


# Security: flags that must not be overridden via extra_flags passthrough.
# These control file output paths, remote telemetry, and code-loading paths
# that an LLM agent (or prompt injection) could abuse. Gated by a HIL
# interrupt; operators can pre-approve individual flags via
# STRANDS_TRAIN_EXTRA_FLAGS_ALLOW or bypass entirely with BYPASS_TOOL_CONSENT.
_BLOCKED_EXTRA_FLAGS = frozenset(
    {
        "output_dir",
        "config_path",
        "wandb.enable",
        "wandb.project",
        "wandb.entity",
        "wandb.api_key",
        "dataset.root",
        "policy.pretrained_path",
        "push_to_hub",
        "policy.push_to_hub",
        "hub_repo_id",
    }
)

_EXTRA_FLAGS_ALLOW_ENV = "STRANDS_TRAIN_EXTRA_FLAGS_ALLOW"
_BYPASS_CONSENT_ENV = "BYPASS_TOOL_CONSENT"

_APPROVE_RESPONSES = frozenset({"y", "yes", "approve", "approved"})


def _approve_response(response: object) -> bool:
    """Accept affirmative operator responses from the HIL interrupt."""
    return isinstance(response, str) and response.strip().lower() in _APPROVE_RESPONSES


def _normalize_hydra_key(key: str) -> str:
    """Strip Hydra prefixes (--key, +key, ~key, ++key) for comparison."""
    return key.lstrip("-+~")


def _validate_extra_flags(extra_flags: dict[str, Any]) -> list[tuple[str, str]]:
    """Return list of (raw_key, normalized_key) pairs that are blocked."""
    blocked_pairs = []
    for key in extra_flags:
        normalized = _normalize_hydra_key(key)
        if normalized in _BLOCKED_EXTRA_FLAGS:
            blocked_pairs.append((key, normalized))
    return blocked_pairs


def _gate_extra_flags(
    extra_flags: dict[str, Any],
    tool_context: ToolContext | None,
) -> dict[str, Any] | None:
    """HIL gate for blocked extra_flags.

    Returns an error dict to halt the training, or None to proceed.
    Three modes:
    - STRANDS_TRAIN_EXTRA_FLAGS_ALLOW contains the flag -> allow silently
    - BYPASS_TOOL_CONSENT=true -> allow with WARNING log
    - Otherwise -> prompt the operator via tool_context.interrupt()
    """
    blocked = _validate_extra_flags(extra_flags)
    if not blocked:
        return None

    allow_raw = os.environ.get(_EXTRA_FLAGS_ALLOW_ENV)
    allowed = frozenset(f.strip() for f in allow_raw.split(",") if f.strip()) if allow_raw else frozenset()

    needs_approval = [(raw, norm) for raw, norm in blocked if norm not in allowed]
    if not needs_approval:
        logger.debug("all blocked flags allowed via %s", _EXTRA_FLAGS_ALLOW_ENV)
        return None

    if os.environ.get(_BYPASS_CONSENT_ENV, "").lower() == "true":
        flag_names = ", ".join(raw for raw, _ in needs_approval)
        logger.warning("BYPASS_TOOL_CONSENT: allowing blocked extra_flags: %s", flag_names)
        return None

    flag_names = ", ".join(raw for raw, _ in needs_approval)
    block_msg = (
        f"extra_flags {flag_names} blocked for security reasons (controls output paths, telemetry, or code loading)."
    )

    if tool_context is None:
        return {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{block_msg} No tool_context available for operator approval. "
                        f"Set {_EXTRA_FLAGS_ALLOW_ENV} or {_BYPASS_CONSENT_ENV}=true "
                        f"to allow in headless mode."
                    )
                }
            ],
        }

    try:
        response = tool_context.interrupt(
            "lerobot_train-extra_flags-approval",
            reason={
                "action": "train",
                "blocked_flags": {raw: str(extra_flags[raw]) for raw, _ in needs_approval},
                "warning": f"{block_msg} Reply 'y' to approve, anything else to deny.",
            },
        )
    except RuntimeError as exc:
        return {
            "status": "error",
            "content": [
                {"text": (f"blocked extra_flags require operator approval, but interrupts are not available: {exc}")}
            ],
        }

    if not _approve_response(response):
        return {
            "status": "error",
            "content": [{"text": f"extra_flags {flag_names} declined by the operator."}],
        }

    logger.info("blocked extra_flags %s approved via operator interrupt", flag_names)
    return None


class SessionManager:
    """Track detached training sessions with on-disk persistence.

    Sessions are keyed by name and stored as JSON. Dead processes are pruned on
    every load so ``list``/``status`` never report a stale PID as running.
    """

    def __init__(self) -> None:
        self.sessions_file = SESSION_DIR / "active_sessions.json"

    def _load_sessions(self) -> dict[str, Any]:
        if not self.sessions_file.exists():
            return {}
        try:
            with open(self.sessions_file) as f:
                sessions = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logger.error(f"Error loading sessions: {e}")
            return {}

        active: dict[str, Any] = {}
        for name, info in sessions.items():
            pid = info.get("pid")
            if pid and psutil.pid_exists(pid):
                try:
                    if psutil.Process(pid).is_running():
                        active[name] = info
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            else:
                # Keep finished training sessions so status can report the
                # final log tail; only the running flag is derived from the PID.
                active[name] = info
        return active

    def _save_sessions(self, sessions: dict[str, Any]) -> None:
        try:
            with open(self.sessions_file, "w") as f:
                json.dump(sessions, f, indent=2)
        except OSError as e:
            logger.error(f"Error saving sessions: {e}")

    def add_session(self, name: str, info: dict[str, Any]) -> None:
        """Persist a training session record under ``name``.

        Loads the current on-disk sessions, upserts ``name`` -> ``info``
        (an existing entry with the same name is overwritten), and writes the
        map back to disk.

        Args:
            name: session key (e.g. the run/job name) to store the record under.
            info: session metadata to persist (typically ``pid``, ``log_file``,
                ``dataset``, and start timestamp).
        """
        sessions = self._load_sessions()
        sessions[name] = info
        self._save_sessions(sessions)

    def remove_session(self, name: str) -> None:
        """Delete the session stored under ``name`` if one exists.

        A no-op when ``name`` is not tracked, so callers need not check first.

        Args:
            name: session key to remove.
        """
        sessions = self._load_sessions()
        if name in sessions:
            del sessions[name]
            self._save_sessions(sessions)

    def get_session(self, name: str) -> dict[str, Any] | None:
        """Return the stored metadata for a single session.

        Args:
            name: session key to look up.

        Returns:
            The session's info dict, or ``None`` if no session is tracked under
            ``name``. Dead processes are pruned on load, so a returned record is
            one whose PID either is still running or belonged to a finished run.
        """
        return self._load_sessions().get(name)

    def list_sessions(self) -> dict[str, Any]:
        """Return every currently-tracked session keyed by name.

        Returns:
            A ``name -> info`` map. Sessions whose PID is no longer a running
            process are not dropped -- they are retained so ``status`` can still
            report the final log tail -- but the load step is what derives the
            live/finished distinction, so this never reports a stale PID as
            running.
        """
        return self._load_sessions()


def _read_total_tasks(dataset_root: str) -> int:
    """Return ``total_tasks`` from a LeRobot v3 dataset's ``meta/info.json``.

    lerobot's own field defaults to 0, and older datasets may omit it entirely;
    both mean "no task count recorded" and are returned as 0, which callers
    treat as single-task.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        return 0
    with open(info_path) as f:
        info = json.load(f)
    total = info.get("total_tasks")
    return total if isinstance(total, int) and not isinstance(total, bool) else 0


def _read_total_episodes(dataset_root: str) -> int:
    """Return ``total_episodes`` from a LeRobot v3 dataset's ``meta/info.json``.

    Raises:
        FileNotFoundError: if ``<dataset_root>/meta/info.json`` does not exist.
        ValueError: if the file lacks a positive integer ``total_episodes``.
    """
    info_path = Path(dataset_root) / "meta" / "info.json"
    if not info_path.exists():
        raise FileNotFoundError(f"Dataset metadata not found: {info_path}")
    with open(info_path) as f:
        info = json.load(f)
    total = info.get("total_episodes")
    if not isinstance(total, int) or total <= 0:
        raise ValueError(f"info.json has no usable 'total_episodes' (got {total!r})")
    return total


def _has_resumable_checkpoint(output_dir: str) -> Path | None:
    """Return the ``train_config.json`` to resume from, or None if none exists.

    lerobot writes checkpoints under ``<output_dir>/checkpoints/`` with a ``last``
    symlink to the newest one; the resumable config lives at
    ``checkpoints/last/pretrained_model/train_config.json``. Resuming requires
    pointing ``--config_path`` at that FILE (not the directory).
    """
    last = Path(output_dir) / "checkpoints" / "last" / "pretrained_model" / "train_config.json"
    return last if last.exists() else None


def build_train_command(
    dataset_root: str,
    policy_type: str = "act",
    pretrained_path: str | None = None,
    output_dir: str | None = None,
    job_name: str = "strands_ft",
    steps: int = 20000,
    batch_size: int = 8,
    save_freq: int = 5000,
    device: str = "cuda",
    dtype: str | None = None,
    gradient_checkpointing: bool = False,
    lora: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_target_modules: str | None = None,
    train_expert_only: bool = False,
    val_episodes: int | None = None,
    num_gpus: int = 1,
    push_to_hub: bool = False,
    resume: bool = False,
    extra_flags: dict[str, Any] | None = None,
) -> list[str]:
    """Build the ``lerobot-train`` argv for the given arguments.

    Single-GPU runs invoke ``python -m lerobot.scripts.lerobot_train``;
    ``num_gpus > 1`` prepends ``accelerate launch --multi_gpu`` and runs the
    module via ``-m``. Resuming from a checkpoint emits the two-flag
    ``--config_path=<ckpt>/train_config.json --resume=true`` form lerobot's
    validate() requires, instead of the from-scratch flags.

    Raises:
        ValueError: if ``lora`` and ``train_expert_only`` are both set (both
            freeze the VLM and are mutually exclusive), if ``train_expert_only``
            is requested for a non-expert policy, or if ``num_gpus < 1``.
    """
    if lora and train_expert_only:
        raise ValueError(
            "lora and train_expert_only are mutually exclusive (both freeze the VLM). Pick one fine-tuning strategy."
        )
    expert_only_policies = _expert_only_policy_types()
    if train_expert_only and policy_type not in expert_only_policies:
        raise ValueError(
            f"train_expert_only is only valid for {sorted(expert_only_policies)} policies, not '{policy_type}'."
        )
    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    resume_config = _has_resumable_checkpoint(output_dir) if (resume and output_dir) else None

    # Launcher prefix: multi-GPU goes through accelerate, single-GPU runs the
    # module directly. Both end at the lerobot_train entrypoint.
    if num_gpus > 1:
        cmd = [
            "accelerate",
            "launch",
            "--multi_gpu",
            f"--num_processes={num_gpus}",
            "--num_machines=1",
            "--mixed_precision=bf16",
            "-m",
            "lerobot.scripts.lerobot_train",
        ]
    else:
        cmd = ["python", "-m", "lerobot.scripts.lerobot_train"]

    if resume_config is not None:
        # Resume path: lerobot loads the full config from the checkpoint file and
        # only honors --config_path + --resume. Other flags are ignored on resume.
        cmd.extend([f"--config_path={resume_config}", "--resume=true"])
        return cmd

    # Fresh-run flags.
    cmd.extend(
        [
            "--dataset.repo_id=local",
            f"--dataset.root={dataset_root}",
            f"--policy.type={policy_type}",
            f"--policy.device={device}",
            f"--policy.push_to_hub={str(push_to_hub).lower()}",
            f"--job_name={job_name}",
            "--wandb.enable=false",
        ]
    )
    if output_dir:
        cmd.append(f"--output_dir={output_dir}")
    if pretrained_path:
        cmd.append(f"--policy.pretrained_path={pretrained_path}")
    if steps is not None:
        cmd.append(f"--steps={steps}")
    if batch_size is not None:
        cmd.append(f"--batch_size={batch_size}")
    if save_freq is not None:
        cmd.append(f"--save_freq={save_freq}")
    # dtype and gradient_checkpointing are PER-POLICY config fields in lerobot:
    # only some policy configs declare them (e.g. the pi0 family / xvla / eo1 for
    # dtype; the pi0 family / diffusion / molmoact2 for gradient_checkpointing).
    # Emitting --policy.dtype for a policy whose config lacks it (like the default
    # ACT) makes draccus abort with "unrecognized arguments" before training even
    # starts. Gate each flag on the resolved config's fields, sourced live so it
    # tracks lerobot; when lerobot is not importable the field set is unknown and
    # the flag passes through unguarded.
    policy_fields = _policy_config_field_names(policy_type)
    if dtype:
        if policy_fields is not None and "dtype" not in policy_fields:
            raise ValueError(
                f"policy_type '{policy_type}' has no 'dtype' config field in lerobot; "
                "drop dtype= (only policies whose config declares it, e.g. the pi0 "
                "family, accept --policy.dtype)."
            )
        cmd.append(f"--policy.dtype={dtype}")
    if gradient_checkpointing:
        if policy_fields is not None and "gradient_checkpointing" not in policy_fields:
            raise ValueError(
                f"policy_type '{policy_type}' has no 'gradient_checkpointing' config "
                "field in lerobot; drop gradient_checkpointing= (only policies whose "
                "config declares it accept --policy.gradient_checkpointing)."
            )
        cmd.append("--policy.gradient_checkpointing=true")
    if train_expert_only:
        cmd.append("--policy.train_expert_only=true")

    if lora:
        cmd.append("--peft.method_type=LORA")
        if lora_r is not None:
            cmd.append(f"--peft.r={lora_r}")
        if lora_alpha is not None:
            cmd.append(f"--peft.lora_alpha={lora_alpha}")
        if lora_target_modules:
            cmd.append(f"--peft.target_modules={lora_target_modules}")

    if val_episodes is not None:
        if val_episodes <= 0:
            raise ValueError(f"val_episodes must be positive, got {val_episodes}")
        total = _read_total_episodes(dataset_root)
        if val_episodes >= total:
            raise ValueError(
                f"val_episodes={val_episodes} leaves no training data (dataset has {total} episodes); reserve fewer."
            )
        split_err = validation_split_error(val_episodes, _read_total_tasks(dataset_root), "lerobot_train")
        if split_err:
            raise ValueError(split_err)
        # Hand the split to lerobot instead of restricting --dataset.episodes
        # ourselves: it holds out the tail AND computes an eval loss on it, where
        # an episode restriction only shrinks the TRAINING set and leaves the
        # reserved episodes unused by either half.
        supplied = {key.lstrip("-") for key in (extra_flags or {})}
        if "dataset.eval_split" not in supplied:
            cmd.append(f"--dataset.eval_split={validation_split_fraction(val_episodes, total)}")
        if "eval_steps" not in supplied:
            # Validate on the caller's own checkpoint cadence, so every saved
            # checkpoint has a validation loss recorded beside it. A non-positive
            # save_freq disables periodic saving, so evaluate once at the end.
            cmd.append(f"--eval_steps={save_freq if save_freq > 0 else steps}")

    if extra_flags:
        for key, value in extra_flags.items():
            flag = key if key.startswith("--") else f"--{key}"
            cmd.append(f"{flag}={value}")

    return cmd


@tool(context=True)
def lerobot_train(
    dataset_root: str,
    tool_context: ToolContext | None = None,
    policy_type: str = "act",
    pretrained_path: str | None = None,
    output_dir: str | None = None,
    job_name: str = "strands_ft",
    steps: int = 20000,
    batch_size: int = 8,
    save_freq: int = 5000,
    device: str = "cuda",
    dtype: str | None = None,
    gradient_checkpointing: bool = False,
    lora: bool = False,
    lora_r: int | None = None,
    lora_alpha: int | None = None,
    lora_target_modules: str | None = None,
    train_expert_only: bool = False,
    val_episodes: int | None = None,
    num_gpus: int = 1,
    push_to_hub: bool = False,
    resume: bool = False,
    action: str = "start",
    session_name: str | None = None,
    extra_flags: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fine-tune a LeRobot policy on a local dataset by wrapping ``lerobot-train``.

    This closes the local record -> train -> deploy loop. After recording a
    LeRobot v3 dataset, call this with ``dataset_root`` pointing at the dataset
    directory (the one containing ``meta/info.json``). On ``start`` it launches
    ``python -m lerobot.scripts.lerobot_train`` (or ``accelerate launch`` for
    ``num_gpus > 1``) as a detached background process and tracks it in the same
    on-disk session store used by ``lerobot_teleoperate``.

    Memory-fit levers:
        ``lora`` and ``train_expert_only`` both freeze the VLM and are mutually
        exclusive; setting both fails fast. ``lora`` emits ``--peft.method_type=LORA``
        plus the supplied ``--peft.*`` overrides. ``train_expert_only`` applies
        only to policies whose lerobot config exposes it (currently pi0/pi05/
        smolvla; sourced live so it tracks lerobot).

    Overfit guard:
        ``val_episodes=N`` reserves the LAST N episodes as a validation set by
        emitting ``--dataset.eval_split`` (the fraction that makes lerobot hold
        out exactly N) together with ``--eval_steps``, so lerobot both keeps the
        tail out of training AND logs an eval loss over it at the checkpoint
        cadence. The episode and task counts are read from ``meta/info.json``;
        a dataset with several tasks is refused because lerobot applies the
        split fraction per task, where a global count is not expressible.
        Passing ``dataset.eval_split`` or ``eval_steps`` in ``extra_flags``
        overrides the derived value.

    Resume:
        ``resume=True`` emits ``--config_path=<ckpt>/train_config.json --resume=true``
        only when a checkpoint exists under ``<output_dir>/checkpoints/last``. If no
        resumable checkpoint exists, a fresh run starts and a stale empty
        ``output_dir`` is cleared so lerobot's "already exists" guard does not trip.

    Actions:
        start: launch a new training run (default).
        status: report a run's PID, uptime, running flag, and recent log tail.
        stop: terminate a running session by name (SIGTERM then SIGKILL).
        list: list tracked training sessions.

    Args:
        dataset_root: Local LeRobot v3 dataset directory (must contain meta/info.json).
        policy_type: Policy architecture (act, diffusion, vqbet, tdmpc, smolvla,
            pi0, pi05, pi0_fast, groot, xvla, ...).
        pretrained_path: HF id or local path to initialize weights from (gated
            checkpoints need HF_TOKEN in the environment).
        output_dir: Where to write run outputs; defaults to
            ``<dataset_root>/../train_out/<job_name>``.
        job_name: Run name used in the default output_dir and lerobot logs.
        steps: Number of training steps.
        batch_size: Training batch size.
        save_freq: Checkpoint save frequency in steps.
        device: Torch device (cuda, cuda:0, cpu, mps).
        dtype: Policy dtype (bfloat16, float32) for policies whose lerobot
            config declares a dtype field (e.g. the pi0 family, xvla). Default
            None lets lerobot pick; ACT and most policies have no dtype field,
            and passing dtype= for them raises before launch.
        gradient_checkpointing: Trade compute for memory on supported policies.
        lora: Enable LoRA/PEFT fine-tuning (full-VLM fit on one GPU).
        lora_r: LoRA rank.
        lora_alpha: LoRA alpha (scaling = lora_alpha / r).
        lora_target_modules: PEFT target module spec (e.g. "all-linear").
        train_expert_only: Freeze the VLM, train only the action expert
            (policies exposing train_expert_only: pi0/pi05/smolvla).
        val_episodes: Reserve the LAST N episodes as a held-out validation
            split, evaluated every ``save_freq`` steps so each checkpoint has
            a validation loss beside it.
        num_gpus: Number of GPUs; >1 launches via accelerate --multi_gpu.
        push_to_hub: Push the trained checkpoint to the HF Hub at the end.
        resume: Resume from the latest checkpoint under output_dir when present.
        action: One of start, status, stop, list.
        session_name: Session identifier (auto-generated on start; required for
            status/stop).
        extra_flags: Passthrough dict of additional lerobot-train flags, e.g.
            ``{"policy.optimizer_lr": 1e-4}`` -> ``--policy.optimizer_lr=0.0001``.

    Returns:
        Dict with ``status`` ("success" or "error") and a ``content`` list of
        ``{"text": ...}`` items, plus action-specific keys (``session_name``,
        ``pid``, ``command``, ``log_file``, ``output_dir``, ``sessions``,
        ``is_running``, ``uptime``).
    """
    session_manager = SessionManager()

    try:
        if action == "start":
            # Preflight: lerobot must be importable and the dataset must exist.
            try:
                import lerobot  # noqa: F401
            except ImportError as e:
                return {
                    "status": "error",
                    "content": [{"text": f"lerobot is not importable: {e}. Install it to train."}],
                }

            info_path = Path(dataset_root) / "meta" / "info.json"
            if not info_path.exists():
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"Dataset metadata not found: {info_path}. "
                            "dataset_root must point at a LeRobot v3 dataset directory."
                        }
                    ],
                }

            if not session_name:
                session_name = f"train_{int(time.time())}"
            if session_manager.get_session(session_name):
                return {
                    "status": "error",
                    "content": [{"text": f"Session '{session_name}' already exists"}],
                }

            # Default output_dir lives next to the dataset so artifacts are colocated.
            resolved_output_dir = output_dir or str(Path(dataset_root).resolve().parent / "train_out" / job_name)

            # Clear a stale EMPTY output_dir on a fresh (non-resumable) start so
            # lerobot's "already exists" guard does not crash. Never delete a dir
            # that holds checkpoints.
            out_path = Path(resolved_output_dir)
            if out_path.is_dir() and not _has_resumable_checkpoint(resolved_output_dir):
                if not any(out_path.iterdir()):
                    shutil.rmtree(out_path, ignore_errors=True)

            if extra_flags:
                gate_err = _gate_extra_flags(extra_flags, tool_context)
                if gate_err:
                    return gate_err

            if pretrained_path:
                gate_err = _gate_extra_flags({"policy.pretrained_path": pretrained_path}, tool_context)
                if gate_err:
                    return gate_err

            try:
                cmd = build_train_command(
                    dataset_root=dataset_root,
                    policy_type=policy_type,
                    pretrained_path=pretrained_path,
                    output_dir=resolved_output_dir,
                    job_name=job_name,
                    steps=steps,
                    batch_size=batch_size,
                    save_freq=save_freq,
                    device=device,
                    dtype=dtype,
                    gradient_checkpointing=gradient_checkpointing,
                    lora=lora,
                    lora_r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_target_modules=lora_target_modules,
                    train_expert_only=train_expert_only,
                    val_episodes=val_episodes,
                    num_gpus=num_gpus,
                    push_to_hub=push_to_hub,
                    resume=resume,
                    extra_flags=extra_flags,
                )
            except (ValueError, FileNotFoundError) as e:
                return {"status": "error", "content": [{"text": f"Command build failed: {e}"}]}

            env = os.environ.copy()
            env["PYTHONUNBUFFERED"] = "1"
            env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

            log_file = SESSION_DIR / f"{session_name}.log"
            with open(log_file, "w") as f:
                proc = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    start_new_session=True,
                    env=env,
                )

            session_info: dict[str, Any] = {
                "action": "train",
                "pid": proc.pid,
                "command": " ".join(cmd),
                "log_file": str(log_file),
                "start_time": time.time(),
                "policy_type": policy_type,
                "dataset_root": dataset_root,
                "output_dir": resolved_output_dir,
            }
            session_manager.add_session(session_name, session_info)

            return {
                "status": "success",
                "content": [
                    {
                        "text": f"**Training Session Started**\n"
                        f"Session: `{session_name}`\n"
                        f"Process ID: {proc.pid}\n"
                        f"Policy: {policy_type}\n"
                        f"Output dir: `{resolved_output_dir}`\n"
                        f"Command: `{' '.join(cmd)}`\n"
                        f"Log file: `{log_file}`\n"
                        f"Running in background"
                    },
                    {
                        "json": {
                            "session_name": session_name,
                            "pid": proc.pid,
                            "command": " ".join(cmd),
                            "log_file": str(log_file),
                            "output_dir": resolved_output_dir,
                        }
                    },
                ],
            }

        elif action == "stop":
            if not session_name:
                return {"status": "error", "content": [{"text": "Session name required for stop action"}]}
            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}
            pid = session_info.get("pid")
            if not pid:
                return {"status": "error", "content": [{"text": f"No PID found for session '{session_name}'"}]}

            pid_int = int(pid)
            try:
                os.kill(pid_int, signal.SIGTERM)
                time.sleep(2)
                if psutil.pid_exists(pid_int):
                    os.kill(pid_int, signal.SIGKILL)
                session_manager.remove_session(session_name)
                return {
                    "status": "success",
                    "content": [
                        {"text": f"**Session Stopped**\nSession: `{session_name}`\nPID: {pid}"},
                        {"json": {"session_name": session_name, "session_info": session_info}},
                    ],
                }
            except ProcessLookupError:
                session_manager.remove_session(session_name)
                return {
                    "status": "success",
                    "content": [
                        {"text": f"Session '{session_name}' was already stopped"},
                        {"json": {"session_name": session_name}},
                    ],
                }

        elif action == "list":
            sessions = session_manager.list_sessions()
            lines = [f"**Active Training Sessions** ({len(sessions)})", ""]
            if sessions:
                for name, info in sessions.items():
                    uptime_min = (time.time() - info.get("start_time", 0)) / 60
                    pid = info.get("pid")
                    is_running = bool(pid and psutil.pid_exists(pid))
                    lines.extend(
                        [
                            f"**{name}**",
                            f"   - Action: {info.get('action', 'Unknown')}",
                            f"   - PID: {pid}",
                            f"   - Uptime: {uptime_min:.1f} min",
                            f"   - Status: {'Running' if is_running else 'Stopped'}",
                            f"   - Policy: {info.get('policy_type', 'Unknown')}",
                            f"   - Output: {info.get('output_dir', 'Unknown')}",
                            "",
                        ]
                    )
            else:
                lines.append("No active sessions")
            return {
                "status": "success",
                "content": [
                    {"text": "\n".join(lines)},
                    {"json": {"sessions": sessions, "count": len(sessions)}},
                ],
            }

        elif action == "status":
            if not session_name:
                return {"status": "error", "content": [{"text": "Session name required for status action"}]}
            session_info = session_manager.get_session(session_name)  # type: ignore[assignment]  # narrow Optional
            if not session_info:
                return {"status": "error", "content": [{"text": f"Session '{session_name}' not found"}]}

            pid = session_info.get("pid")
            uptime = time.time() - float(session_info.get("start_time") or 0)
            is_running = bool(pid and psutil.pid_exists(int(pid)))
            lines = [
                f"**Session Status: `{session_name}`**",
                f"PID: {pid}",
                f"Action: {session_info.get('action', 'Unknown')}",
                f"Uptime: {uptime / 60:.1f} min",
                f"Status: {'Running' if is_running else 'Stopped'}",
                f"Policy: {session_info.get('policy_type', 'Unknown')}",
                f"Output dir: {session_info.get('output_dir', 'Unknown')}",
            ]
            log_file_path = session_info.get("log_file")
            if log_file_path and Path(str(log_file_path)).exists():
                lines.append(f"Log file: `{log_file_path}`")
                try:
                    with open(str(log_file_path)) as f:
                        tail = f.readlines()[-15:]
                    if tail:
                        lines.extend(["", "**Recent Log Output:**", "```", "".join(tail).strip(), "```"])
                except OSError as e:
                    lines.append(f"Error reading log: {e}")
            return {
                "status": "success",
                "content": [
                    {"text": "\n".join(lines)},
                    {
                        "json": {
                            **session_info,
                            "session_name": session_name,
                            "pid": pid,
                            "uptime": uptime,
                            "is_running": is_running,
                        }
                    },
                ],
            }

        else:
            return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

    except Exception as e:
        logger.error(f"LeRobot train error: {e}")
        return {"status": "error", "content": [{"text": f"Tool execution failed: {e}"}]}
