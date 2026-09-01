"""Behavior tests for the ``lerobot_train`` agent tool.

The train tool wraps LeRobot's ``lerobot-train`` script behind a single
agent-facing dispatcher with on-disk session tracking, mirroring
``lerobot_teleoperate``. These tests run hardware- and GPU-free by exercising
the pure command builder directly and by substituting fakes for ``subprocess``
and ``psutil`` in the session lifecycle. They pin:

1. The command builder maps each arg set to the correct ``lerobot-train`` argv
   (single-GPU, multi-GPU accelerate, LoRA, optional tuning flags, held-out val
   split, resume) and coerces numeric extra_flags to plain decimals.
2. The mutual-exclusion / preflight guards fail with clear errors instead of
   launching a doomed run (missing lerobot, duplicate session, build failure).
3. The session lifecycle (start -> list -> status -> stop) round-trips through
   the persisted session store, including stop's SIGTERM->SIGKILL escalation,
   the already-dead and missing-PID branches, and the status log tail.
4. The on-disk session store degrades gracefully on corrupt/unwritable files and
   the dispatcher wraps unexpected errors as structured results rather than
   raising past dispatch.
5. Every user-facing ``text`` field is plain ASCII (the project's no-emoji rule).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod
from tests.tool_result_contract import tool_json

build_train_command = train_mod.build_train_command
lerobot_train = train_mod.lerobot_train
SessionManager = train_mod.SessionManager


def _texts(result: dict[str, Any]) -> str:
    return "\n".join(item.get("text", "") for item in result.get("content", []) if "text" in item)


def _assert_ascii(text: str) -> None:
    offenders = {hex(ord(c)) for c in text if ord(c) > 127}
    assert not offenders, f"non-ASCII characters in tool output: {offenders}"


def _write_dataset(root: Path, total_episodes: int = 10) -> Path:
    """Create a minimal LeRobot v3 dataset stub with meta/info.json."""
    meta = root / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(json.dumps({"total_episodes": total_episodes}))
    return root


@pytest.fixture(autouse=True)
def _isolate_session_dir(tmp_path, monkeypatch: pytest.MonkeyPatch):
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class _FakeProc:
    def __init__(self, pid: int = 4242) -> None:
        self.pid = pid


# ---------------------------------------------------------------------------
# build_train_command - argv mapping
# ---------------------------------------------------------------------------
def test_build_single_gpu_command_emits_core_flags() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out/act",
        job_name="cube_ft",
        steps=5000,
        batch_size=16,
        save_freq=1000,
        device="cuda",
    )
    assert cmd[:3] == ["python", "-m", "lerobot.scripts.lerobot_train"]
    assert "--dataset.repo_id=local" in cmd
    assert "--dataset.root=/data/cubes" in cmd
    assert "--policy.type=act" in cmd
    assert "--policy.device=cuda" in cmd
    assert "--policy.push_to_hub=false" in cmd
    assert "--output_dir=/out/act" in cmd
    assert "--job_name=cube_ft" in cmd
    assert "--wandb.enable=false" in cmd
    assert "--steps=5000" in cmd
    assert "--batch_size=16" in cmd
    assert "--save_freq=1000" in cmd
    # ACT's lerobot config has no dtype field, so no --policy.dtype is emitted.
    assert not any(tok.startswith("--policy.dtype") for tok in cmd)
    # No accelerate prefix on a single-GPU run.
    assert "accelerate" not in cmd


def test_build_multi_gpu_command_prepends_accelerate_launch() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out/act",
        num_gpus=4,
    )
    assert cmd[:2] == ["accelerate", "launch"]
    assert "--multi_gpu" in cmd
    assert "--num_processes=4" in cmd
    assert "--num_machines=1" in cmd
    assert "--mixed_precision=bf16" in cmd
    # The module is launched via -m after the accelerate flags.
    assert cmd[cmd.index("-m") + 1] == "lerobot.scripts.lerobot_train"
    # Core training flags still present downstream.
    assert "--dataset.root=/data/cubes" in cmd
    assert "--policy.type=act" in cmd


def test_build_lora_command_emits_peft_flags() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="smolvla",
        output_dir="/out/smolvla",
        lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_target_modules="all-linear",
    )
    assert "--peft.method_type=LORA" in cmd
    assert "--peft.r=32" in cmd
    assert "--peft.lora_alpha=64" in cmd
    assert "--peft.target_modules=all-linear" in cmd


def test_build_expert_only_emits_flag_for_pi_family() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="pi05",
        output_dir="/out/pi05",
        train_expert_only=True,
    )
    assert "--policy.train_expert_only=true" in cmd


def test_lora_and_expert_only_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="pi05",
            lora=True,
            train_expert_only=True,
        )


def test_expert_only_rejected_for_non_expert_policy() -> None:
    with pytest.raises(ValueError, match="train_expert_only is only valid"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="act",
            train_expert_only=True,
        )


def test_expert_only_rejected_for_pi0_fast() -> None:
    # pi0_fast's lerobot config exposes NO train_expert_only field, so the CLI
    # flag is a hard draccus error. The supported set is sourced live from
    # lerobot (not a hardcoded copy that once wrongly listed pi0_fast).
    with pytest.raises(ValueError, match="train_expert_only is only valid"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="pi0_fast",
            train_expert_only=True,
        )


def test_expert_only_policy_types_tracks_lerobot_registry() -> None:
    # Drift guard: the live supported set must equal the lerobot policy configs
    # that actually declare a train_expert_only field, so it tracks lerobot.
    import dataclasses

    # importorskip both registers the policy configs (side-effect import) and
    # binds PreTrainedConfig unconditionally, so it is always initialized on the
    # path that uses it below (CodeQL: no use-before-init).
    pytest.importorskip("lerobot.policies")
    PreTrainedConfig = pytest.importorskip("lerobot.configs.policies").PreTrainedConfig
    expected = {
        name
        for name, cfg_cls in PreTrainedConfig.get_known_choices().items()
        if any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
    }
    assert set(train_mod._expert_only_policy_types()) == expected
    # pi0_fast has no such field on current lerobot.
    assert "pi0_fast" not in expected


def test_expert_only_fallback_is_a_faithful_snapshot_of_lerobot() -> None:
    """The static offline snapshot stays in sync with lerobot; drift trips this.

    ``_EXPERT_ONLY_POLICIES_FALLBACK`` is only consulted when lerobot is not
    importable, so a policy that gains or loses a ``train_expert_only`` field in
    lerobot would silently diverge from the snapshot and hand offline callers a
    stale set. Pinning equality here surfaces that drift the moment it happens.
    """
    import dataclasses

    pytest.importorskip("lerobot.policies")
    PreTrainedConfig = pytest.importorskip("lerobot.configs.policies").PreTrainedConfig
    expected = {
        name
        for name, cfg_cls in PreTrainedConfig.get_known_choices().items()
        if any(f.name == "train_expert_only" for f in dataclasses.fields(cfg_cls))
    }
    assert set(train_mod._EXPERT_ONLY_POLICIES_FALLBACK) == expected, (
        "_EXPERT_ONLY_POLICIES_FALLBACK drifted from lerobot's expert-only policy "
        "configs; update the fallback frozenset to match the current lerobot."
    )


def test_expert_only_policy_types_falls_back_when_lerobot_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft: an unimportable lerobot yields the static snapshot, not a crash.

    ``_expert_only_policy_types`` sources the supported set live from lerobot's
    ``PreTrainedConfig`` registry. If lerobot moves or renames those modules (a
    real risk when tracking lerobot from source), the tool must degrade to
    :data:`_EXPERT_ONLY_POLICIES_FALLBACK` rather than raise - otherwise a stale
    lerobot would break ``train_expert_only`` validation for every policy. The
    ``None`` sentinel in ``sys.modules`` forces ``import lerobot.policies`` to
    raise ``ImportError`` even though lerobot is installed here.
    """
    import sys

    monkeypatch.setitem(sys.modules, "lerobot.policies", None)
    assert train_mod._expert_only_policy_types() == train_mod._EXPERT_ONLY_POLICIES_FALLBACK


# ---------------------------------------------------------------------------
# Per-policy dtype / gradient_checkpointing field gating
#
# dtype and gradient_checkpointing are PER-POLICY config fields in lerobot: only
# some policy configs declare them (the pi0 family, xvla, eo1 for dtype; the pi0
# family, diffusion, molmoact2 for gradient_checkpointing). The default policy,
# ACT, declares NEITHER, so emitting --policy.dtype / --policy.gradient_checkpointing
# for it makes draccus abort with "unrecognized arguments" before training starts.
# ---------------------------------------------------------------------------
def test_act_default_omits_dtype_and_gradient_checkpointing() -> None:
    """The default (ACT) invocation must not emit fields ACT's config lacks.

    Regression: the tool used to default dtype="bfloat16" and emit --policy.dtype
    unconditionally, so the DEFAULT lerobot_train(policy_type="act") built a
    command draccus rejects with "unrecognized arguments: --dtype" before a step
    ever ran. Defaults must produce a launchable command.
    """
    cmd = build_train_command(dataset_root="/data/cubes", policy_type="act")
    assert not any(tok.startswith("--policy.dtype") for tok in cmd)
    assert "--policy.gradient_checkpointing=true" not in cmd


def test_dtype_emitted_for_policy_that_declares_it() -> None:
    """A policy whose config declares dtype (pi05) still gets --policy.dtype."""
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="pi05",
        dtype="bfloat16",
    )
    assert "--policy.dtype=bfloat16" in cmd


def test_explicit_dtype_on_policy_without_field_raises_clear_error() -> None:
    """Explicit dtype= on a policy lacking the field fails strands-side, clearly.

    Better a targeted ValueError naming the policy than a draccus stack trace
    from a doomed subprocess.
    """
    pytest.importorskip("lerobot.policies")
    with pytest.raises(ValueError, match="has no 'dtype' config field"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="act",
            dtype="bfloat16",
        )


def test_gradient_checkpointing_on_policy_without_field_raises_clear_error() -> None:
    """gradient_checkpointing on a policy lacking the field fails strands-side."""
    pytest.importorskip("lerobot.policies")
    with pytest.raises(ValueError, match="has no 'gradient_checkpointing' config field"):
        build_train_command(
            dataset_root="/data/cubes",
            policy_type="act",
            gradient_checkpointing=True,
        )


def test_policy_config_field_names_tracks_lerobot_registry() -> None:
    """The live field probe matches lerobot's actual config dataclass fields."""
    import dataclasses

    pytest.importorskip("lerobot.policies")
    PreTrainedConfig = pytest.importorskip("lerobot.configs.policies").PreTrainedConfig
    for name, cfg_cls in PreTrainedConfig.get_known_choices().items():
        expected = frozenset(f.name for f in dataclasses.fields(cfg_cls))
        assert train_mod._policy_config_field_names(name) == expected


def test_policy_config_field_names_unknown_policy_is_none() -> None:
    """An unknown policy type resolves to None (gating is skipped, not crashed)."""
    pytest.importorskip("lerobot.policies")
    assert train_mod._policy_config_field_names("definitely_not_a_policy") is None


def test_field_gating_passes_through_when_lerobot_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When lerobot cannot be imported the field set is unknown, so flags pass
    through unguarded rather than raising - the runtime start path preflights
    lerobot separately, so this only affects offline command building."""
    import sys

    monkeypatch.setitem(sys.modules, "lerobot.policies", None)
    assert train_mod._policy_config_field_names("act") is None
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        dtype="bfloat16",
        gradient_checkpointing=True,
    )
    assert "--policy.dtype=bfloat16" in cmd
    assert "--policy.gradient_checkpointing=true" in cmd


def test_val_episodes_reserves_last_n_episodes_as_an_evaluated_split(tmp_path: Path) -> None:
    """Reserving episodes must yield a validation loss, not only a smaller train set.

    This previously asserted ``--dataset.episodes=[0..6]``, which restricts the
    TRAINING set and leaves the reserved episodes unused by either half: lerobot
    builds its eval dataloader from ``dataset.eval_split``, so an episode
    restriction produces no validation signal at all. The corrected contract
    emits the split plus the cadence that evaluates it.
    """
    root = _write_dataset(tmp_path / "ds", total_episodes=10)
    cmd = build_train_command(
        dataset_root=str(root),
        policy_type="act",
        output_dir=str(tmp_path / "out"),
        save_freq=500,
        val_episodes=3,
    )
    # 10 total, reserve last 3: ceil(10 * 0.25) == 3.
    assert "--dataset.eval_split=0.25" in cmd
    assert "--eval_steps=500" in cmd
    assert not [c for c in cmd if c.startswith("--dataset.episodes=")]


def test_val_episodes_rejects_reserving_whole_dataset(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds", total_episodes=3)
    with pytest.raises(ValueError, match="leaves no training data"):
        build_train_command(
            dataset_root=str(root),
            policy_type="act",
            val_episodes=3,
        )


def test_resume_emits_two_flag_form_when_checkpoint_exists(tmp_path: Path) -> None:
    out = tmp_path / "out"
    ckpt = out / "checkpoints" / "last" / "pretrained_model"
    ckpt.mkdir(parents=True)
    (ckpt / "train_config.json").write_text("{}")

    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir=str(out),
        resume=True,
    )
    assert f"--config_path={ckpt / 'train_config.json'}" in cmd
    assert "--resume=true" in cmd
    # Resume path ignores the from-scratch flags.
    assert "--dataset.repo_id=local" not in cmd
    assert "--policy.type=act" not in cmd


def test_resume_without_checkpoint_starts_fresh(tmp_path: Path) -> None:
    out = tmp_path / "out"  # no checkpoints dir
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir=str(out),
        resume=True,
    )
    # No resumable checkpoint -> normal fresh-run flags, no --config_path.
    assert not any(c.startswith("--config_path=") for c in cmd)
    assert "--dataset.repo_id=local" in cmd
    assert "--policy.type=act" in cmd


def test_extra_flags_passthrough_normalizes_leading_dashes() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out",
        extra_flags={"policy.optimizer_lr": "1e-4", "--num_workers": 8},
    )
    assert "--policy.optimizer_lr=1e-4" in cmd
    assert "--num_workers=8" in cmd


def test_push_to_hub_true_sets_flag() -> None:
    cmd = build_train_command(
        dataset_root="/data/cubes",
        policy_type="act",
        output_dir="/out",
        push_to_hub=True,
    )
    assert "--policy.push_to_hub=true" in cmd


def test_num_gpus_zero_rejected() -> None:
    with pytest.raises(ValueError, match="num_gpus must be >= 1"):
        build_train_command(dataset_root="/data/cubes", num_gpus=0)


# ---------------------------------------------------------------------------
# Preflight (start action)
# ---------------------------------------------------------------------------
def test_start_errors_when_dataset_missing_info(tmp_path: Path) -> None:
    # Directory exists but has no meta/info.json.
    empty = tmp_path / "empty"
    empty.mkdir()
    result = lerobot_train(action="start", dataset_root=str(empty))
    assert result["status"] == "error"
    text = _texts(result)
    assert "info.json" in text
    _assert_ascii(text)


# ---------------------------------------------------------------------------
# Session lifecycle: start -> list -> status -> stop
# ---------------------------------------------------------------------------
def test_session_lifecycle_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = _write_dataset(tmp_path / "ds", total_episodes=8)

    # Fake Popen so no real training process spawns.
    def _fake_popen(cmd, **kwargs):  # noqa: ANN001
        return _FakeProc(pid=9999)

    monkeypatch.setattr(train_mod.subprocess, "Popen", _fake_popen)
    # Report the fake PID as a live, running process.
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _FakeProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

        def wait(self, timeout: float | None = None) -> int:
            # Exits as soon as it is signalled, so the stop below confirms it.
            return 0

    monkeypatch.setattr(train_mod.psutil, "Process", _FakeProcess)

    start = lerobot_train(
        action="start",
        dataset_root=str(root),
        policy_type="act",
        output_dir=str(tmp_path / "out"),
        session_name="t1",
    )
    assert start["status"] == "success"
    assert tool_json(start)["session_name"] == "t1"
    assert tool_json(start)["pid"] == 9999
    _assert_ascii(_texts(start))

    listed = lerobot_train(action="list", dataset_root=str(root))
    assert listed["status"] == "success"
    assert "t1" in tool_json(listed)["sessions"]
    _assert_ascii(_texts(listed))

    status = lerobot_train(action="status", dataset_root=str(root), session_name="t1")
    assert status["status"] == "success"
    assert tool_json(status)["is_running"] is True
    assert tool_json(status)["pid"] == 9999
    _assert_ascii(_texts(status))

    # Stop: capture the kill calls instead of touching a real process.
    killed: list[int] = []
    monkeypatch.setattr(train_mod.os, "kill", lambda pid, sig: killed.append(pid))

    stop = lerobot_train(action="stop", dataset_root=str(root), session_name="t1")
    assert stop["status"] == "success"
    assert 9999 in killed
    _assert_ascii(_texts(stop))

    # Session store no longer tracks it.
    gone = lerobot_train(action="status", dataset_root=str(root), session_name="t1")
    assert gone["status"] == "error"


def test_status_requires_session_name(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds")
    result = lerobot_train(action="status", dataset_root=str(root))
    assert result["status"] == "error"
    _assert_ascii(_texts(result))


def test_unknown_action_errors(tmp_path: Path) -> None:
    root = _write_dataset(tmp_path / "ds")
    result = lerobot_train(action="frobnicate", dataset_root=str(root))
    assert result["status"] == "error"
    assert "Unknown action" in _texts(result)


# ---------------------------------------------------------------------------
# build_train_command - optional flag pass-through
# ---------------------------------------------------------------------------
def test_build_command_emits_optional_tuning_flags() -> None:
    """Optional steps/batch/save_freq/dtype/grad-ckpt/pretrained flags all appear.

    dtype and gradient_checkpointing are per-policy config fields, so this uses
    pi05 - a policy whose lerobot config declares both - to exercise the emission
    path. ACT, the tool default, has neither field (see the dedicated gate tests).
    """
    cmd = build_train_command(
        dataset_root="/data/ds",
        policy_type="pi05",
        pretrained_path="lerobot/pi05_base",
        output_dir="/out",
        steps=500,
        batch_size=16,
        save_freq=100,
        dtype="float32",
        gradient_checkpointing=True,
    )
    assert "--steps=500" in cmd
    assert "--batch_size=16" in cmd
    assert "--save_freq=100" in cmd
    assert "--policy.dtype=float32" in cmd
    assert "--policy.gradient_checkpointing=true" in cmd
    assert "--policy.pretrained_path=lerobot/pi05_base" in cmd
    assert "--output_dir=/out" in cmd


def test_build_command_extra_flags_coerce_floats_without_scientific_notation() -> None:
    """Numeric extra_flags render as plain decimals lerobot's parser accepts."""
    cmd = build_train_command(
        dataset_root="/data/ds",
        extra_flags={"policy.optimizer_lr": 1e-4},
    )
    assert "--policy.optimizer_lr=0.0001" in cmd
    assert not any("e-" in tok for tok in cmd)


# ---------------------------------------------------------------------------
# SessionManager - on-disk store resilience
# ---------------------------------------------------------------------------
def test_session_manager_recovers_from_corrupt_store(tmp_path: Path) -> None:
    """A truncated/garbage sessions file degrades to empty, never raises."""
    mgr = SessionManager()
    mgr.sessions_file.parent.mkdir(parents=True, exist_ok=True)
    mgr.sessions_file.write_text("{not json")
    assert mgr.list_sessions() == {}


def test_session_manager_retains_finished_session_with_dead_pid(tmp_path: Path) -> None:
    """A session whose PID is gone is kept (so status can report the final log)."""
    mgr = SessionManager()
    mgr.add_session("done", {"pid": 1, "action": "train"})
    # PID 1 is not one of ours; _load_sessions keeps it but it is not "running".
    sessions = mgr.list_sessions()
    assert "done" in sessions


@pytest.mark.parametrize(
    "exc_factory",
    [
        pytest.param(train_mod.psutil.NoSuchProcess, id="no-such-process"),
        pytest.param(train_mod.psutil.AccessDenied, id="access-denied"),
    ],
)
def test_session_whose_probe_raises_keeps_its_record(monkeypatch: pytest.MonkeyPatch, exc_factory: Any) -> None:
    """A probe that raises must not cost the session its record, and must not raise.

    A PID that exists at check time whose process object then raises is either a
    finished run reaped between the two probes (``NoSuchProcess``) or a live
    process this user may not inspect (``AccessDenied``). Neither is grounds for
    deleting the record: the store is the only place a detached PID is written
    down, and ``add_session``/``remove_session`` write the loaded map back, so an
    omission here is erased from disk by the next session started or stopped.

    This is the same rule as the dead-PID case above rather than its complement:
    "cannot be confirmed running" is not a reason to drop the record, because
    being in the store is not the running claim - ``list`` and ``status`` derive
    that from the PID when asked. The consequences are pinned in
    ``tests.tools.test_train_session_store_keeps_a_live_pid``.
    """
    mgr = SessionManager()
    mgr.add_session("racy", {"pid": 4242, "action": "train"})

    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _VanishingProcess:
        def __init__(self, pid: int) -> None:
            self._pid = pid

        def is_running(self) -> bool:
            raise exc_factory(self._pid)

    monkeypatch.setattr(train_mod.psutil, "Process", _VanishingProcess)

    assert "racy" in mgr.list_sessions(), "a session whose probe raised must keep its record"


def test_session_manager_get_and_remove_round_trip(tmp_path: Path) -> None:
    mgr = SessionManager()
    mgr.add_session("s", {"pid": 4242})
    assert mgr.get_session("s") == {"pid": 4242}
    mgr.remove_session("s")
    assert mgr.get_session("s") is None
    # Removing a non-existent session is a no-op, not an error.
    mgr.remove_session("s")


# ---------------------------------------------------------------------------
# dispatcher - preflight + guard error paths
# ---------------------------------------------------------------------------
def test_start_errors_when_lerobot_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Start preflight fails cleanly (no Popen) when lerobot is not importable."""
    root = _write_dataset(tmp_path / "ds")
    monkeypatch.setitem(__import__("sys").modules, "lerobot", None)

    def _boom_popen(*a: Any, **k: Any):  # noqa: ANN401
        raise AssertionError("Popen must not run when lerobot is missing")

    monkeypatch.setattr(train_mod.subprocess, "Popen", _boom_popen)
    result = lerobot_train(action="start", dataset_root=str(root), policy_type="act")
    assert result["status"] == "error"
    assert "lerobot is not importable" in _texts(result)


def test_start_rejects_duplicate_session_name(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A second start with the same name is refused instead of clobbering the first."""
    root = _write_dataset(tmp_path / "ds")
    SessionManager().add_session("dup", {"pid": 4242})

    def _boom_popen(*a: Any, **k: Any):  # noqa: ANN401
        raise AssertionError("Popen must not run for a duplicate session")

    monkeypatch.setattr(train_mod.subprocess, "Popen", _boom_popen)
    result = lerobot_train(action="start", dataset_root=str(root), policy_type="act", session_name="dup")
    assert result["status"] == "error"
    assert "already exists" in _texts(result)


def test_start_surfaces_command_build_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A bad arg combo is reported as a build error, not a crash, and no Popen runs."""
    root = _write_dataset(tmp_path / "ds")

    def _boom_popen(*a: Any, **k: Any):  # noqa: ANN401
        raise AssertionError("Popen must not run when command build fails")

    monkeypatch.setattr(train_mod.subprocess, "Popen", _boom_popen)
    # lora + train_expert_only are mutually exclusive -> ValueError inside builder.
    result = lerobot_train(
        action="start",
        dataset_root=str(root),
        policy_type="smolvla",
        lora=True,
        train_expert_only=True,
        session_name="bad",
    )
    assert result["status"] == "error"
    assert "Command build failed" in _texts(result)


def test_stop_without_session_name_errors(tmp_path: Path) -> None:
    result = lerobot_train(action="stop", dataset_root=str(_write_dataset(tmp_path / "ds")))
    assert result["status"] == "error"
    assert "Session name required" in _texts(result)


def test_stop_unknown_session_errors(tmp_path: Path) -> None:
    result = lerobot_train(action="stop", dataset_root=str(_write_dataset(tmp_path / "ds")), session_name="ghost")
    assert result["status"] == "error"
    assert "not found" in _texts(result)


def test_stop_already_dead_process_clears_session(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A process that exits just before the signal lands: success, session dropped.

    ``os.kill`` reports that as ``ProcessLookupError``; the sibling spelling -
    gone before the identity capture, which raises ``psutil.NoSuchProcess`` - is
    pinned in ``tests.tools.test_session_stop_confirms_the_process_exited``. The
    process is alive for every probe here so the exit is decided by the signal
    and not by whether PID 5555 happens to exist on the host.
    """
    SessionManager().add_session("gone", {"pid": 5555, "action": "train"})
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _LiveProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

    monkeypatch.setattr(train_mod.psutil, "Process", _LiveProcess)

    def _raise_no_such(pid: int, sig: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(train_mod.os, "kill", _raise_no_such)
    result = lerobot_train(action="stop", dataset_root="/x", session_name="gone")
    assert result["status"] == "success"
    assert "already stopped" in _texts(result)
    assert SessionManager().get_session("gone") is None


def test_status_reports_log_tail(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """status surfaces the recent log tail for a tracked session."""
    log = tmp_path / "run.log"
    log.write_text("epoch 1\nepoch 2\nloss 0.01\n")
    SessionManager().add_session("live", {"pid": 7777, "action": "train", "start_time": 0.0, "log_file": str(log)})
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _RunningProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

    monkeypatch.setattr(train_mod.psutil, "Process", _RunningProcess)
    result = lerobot_train(action="status", dataset_root="/x", session_name="live")
    assert result["status"] == "success"
    assert tool_json(result)["is_running"] is True
    assert "loss 0.01" in _texts(result)
    assert "Recent Log Output" in _texts(result)


def test_list_reports_no_active_sessions_when_empty(tmp_path: Path) -> None:
    result = lerobot_train(action="list", dataset_root="/x")
    assert result["status"] == "success"
    assert tool_json(result)["count"] == 0
    assert "No active sessions" in _texts(result)


def test_build_command_rejects_a_val_episodes_that_is_not_a_count(tmp_path: Path) -> None:
    """The shared positive-count domain is applied before the dataset is read.

    Was ``val_episodes <= 0``, whose message ("must be positive") was false for
    the values that slipped through it: ``2.7`` and ``True`` both compare as
    positive and reserved a whole number the caller never named.
    """
    root = str(_write_dataset(tmp_path / "ds"))
    for value in (0, -5, True, 2.7, float("nan"), "5"):
        with pytest.raises(ValueError, match="val_episodes must be a positive integer"):
            build_train_command(dataset_root=root, val_episodes=value)  # type: ignore[arg-type]


def test_read_total_episodes_raises_on_missing_and_bad_metadata(tmp_path: Path) -> None:
    """_read_total_episodes guards both a missing file and a non-positive count."""
    missing = tmp_path / "nope"
    with pytest.raises(FileNotFoundError):
        train_mod._read_total_episodes(str(missing))

    bad = tmp_path / "bad"
    (bad / "meta").mkdir(parents=True)
    (bad / "meta" / "info.json").write_text(json.dumps({"total_episodes": 0}))
    with pytest.raises(ValueError, match="total_episodes"):
        train_mod._read_total_episodes(str(bad))


def test_start_autogenerates_session_name_and_clears_stale_empty_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No session_name -> auto name; a stale EMPTY output_dir is removed first."""
    root = _write_dataset(tmp_path / "ds")
    out = tmp_path / "out"
    out.mkdir()  # stale, empty, no checkpoints -> should be cleared
    monkeypatch.setattr(train_mod.subprocess, "Popen", lambda *a, **k: _FakeProc(pid=321))
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    rmtree_calls: list[str] = []
    real_rmtree = train_mod.shutil.rmtree

    def _spy_rmtree(path, **kwargs):  # noqa: ANN001
        rmtree_calls.append(str(path))
        return real_rmtree(path, **kwargs)

    monkeypatch.setattr(train_mod.shutil, "rmtree", _spy_rmtree)
    result = lerobot_train(action="start", dataset_root=str(root), policy_type="act", output_dir=str(out))
    assert result["status"] == "success"
    assert tool_json(result)["session_name"].startswith("train_")
    assert str(out) in rmtree_calls


# The SIGTERM -> SIGKILL escalation, and the verdict each outcome earns, are
# pinned for both session tools in tests.tools.test_session_stop_confirms_the_process_exited.


def test_stop_session_without_pid_errors(tmp_path: Path) -> None:
    """A tracked session missing its PID reports an error instead of crashing."""
    SessionManager().add_session("nopid", {"action": "train"})
    result = lerobot_train(action="stop", dataset_root="/x", session_name="nopid")
    assert result["status"] == "error"
    assert "No PID found" in _texts(result)


def test_status_handles_unreadable_log_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An OSError reading the log tail is reported inline, not raised."""
    log = tmp_path / "run.log"
    log.write_text("x\n")
    SessionManager().add_session("live", {"pid": 7777, "action": "train", "start_time": 0.0, "log_file": str(log)})
    monkeypatch.setattr(train_mod.psutil, "pid_exists", lambda pid: True)

    class _RunningProcess:
        def __init__(self, pid: int) -> None:
            self.pid = pid

        def is_running(self) -> bool:
            return True

    monkeypatch.setattr(train_mod.psutil, "Process", _RunningProcess)

    real_open = open

    def _boom_open(path, *a, **k):  # noqa: ANN001
        if str(path) == str(log):
            raise OSError("permission denied")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", _boom_open)
    result = lerobot_train(action="status", dataset_root="/x", session_name="live")
    assert result["status"] == "success"
    assert "Error reading log" in _texts(result)


def test_session_manager_save_error_is_swallowed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A failed write to the session store logs but does not raise."""
    mgr = SessionManager()

    def _boom_open(*a: Any, **k: Any):  # noqa: ANN401
        raise OSError("read-only filesystem")

    monkeypatch.setattr("builtins.open", _boom_open)
    # Should not raise despite the unwritable store.
    mgr._save_sessions({"x": {"pid": 1}})


def test_dispatcher_wraps_unexpected_errors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """An unexpected internal failure is returned as a structured error, never raised."""

    def _boom(self):  # noqa: ANN001
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(train_mod.SessionManager, "list_sessions", _boom)
    result = lerobot_train(action="list", dataset_root="/x")
    assert result["status"] == "error"
    assert "Tool execution failed" in _texts(result)
