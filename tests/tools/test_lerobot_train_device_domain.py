"""``lerobot_train`` refuses a device torch cannot parse, before launching it.

``build_train_command`` writes ``--policy.device`` into the argv of a process
the tool then launches DETACHED, so an unusable value was not reported to the
caller at all: the tool returned ``status="success"`` with a pid and a log path,
and only the training log recorded that lerobot aborted before the first step.
``device`` was the one token in that fresh-run argv with no domain - the run-size
numerics beside it (``steps``, ``batch_size``, ``lora_r``, ``lora_alpha``,
``val_episodes``) are refused up front for exactly this reason, and ``dtype`` /
``gradient_checkpointing`` / ``policy_type`` are each gated against a live
lerobot registry.

``device="gpu"`` is the ordinary mistake: it is the word the rest of the world
uses and it is not a torch device type, so it aborted the detached run while the
tool reported it started.

The admitted domain is torch's own, read by handing the value to
``torch.device`` rather than by comparing against a copied list, so a torch build
that gains a backend is admitted here with no change. These tests pin that only
the *spelling* is graded and never availability - ``cuda`` on a CPU-only box, and
the exotic-but-valid types below, must keep building an argv, because a queued or
containerised run legitimately names a device the dispatching machine lacks - and
that the scope is no wider than the defect: the resume path never emits
``--policy.device``, so a resumed run is not refused for a value its argv does
not carry.
"""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

import strands_robots.tools.lerobot_train as train_mod
from tests.tools.test_lerobot_train import _FakeProc, _write_dataset

build_train_command = train_mod.build_train_command

# Spellings no torch build can parse, and why each one matters.
UNUSABLE_DEVICES: tuple[Any, ...] = (
    "gpu",  # the ordinary mistake: the word everyone else uses
    "GPU",
    "CUDA",  # torch device types are lowercase
    "nvidia",
    "cuda:x",  # a non-numeric index
    "cuda:-1",
    "cuda:",  # an index promised and not given
    "cuda:0:1",
    "cuda 0",  # a space instead of the separator
    "",  # names nothing
    "   ",
    "cpu; rm -rf /",  # injection shapes, defended even though the argv is argv-style
    "$(whoami)",
    "`id`",
    "cpu\nInjected",
    0,  # a non-str: torch would read the accelerator inventory for this one
    None,
    True,
    3.5,
    ["cuda"],
)

# Spellings torch parses. The first four are what the tool's own docstring names;
# ``cpu:0`` is an explicitly indexed host device; the rest are valid device types
# that essentially no test machine has, and are here to prove availability is not
# what is being graded.
USABLE_DEVICES: tuple[str, ...] = ("cuda", "cuda:0", "cpu", "mps", "cpu:0", "xla", "vulkan", "hpu", "meta", "xpu")


def _build(**kwargs: Any) -> list[str]:
    """Build an argv with a minimal usable base, overridden by ``kwargs``.

    Funnelled so the deliberately off-type values above reach the runtime guard
    the way an agent supplies them, without a type checker objecting at each
    call site.
    """
    base: dict[str, Any] = {"dataset_root": "/data/cubes", "policy_type": "act"}
    base.update(kwargs)
    return build_train_command(**base)


def _write_resumable_checkpoint(output_dir: Path) -> Path:
    """Write the ``train_config.json`` ``--resume`` requires, and return the dir."""
    pretrained = output_dir / "checkpoints" / "last" / "pretrained_model"
    pretrained.mkdir(parents=True, exist_ok=True)
    (pretrained / "train_config.json").write_text(json.dumps({"steps": 20000}))
    return output_dir


@pytest.fixture(autouse=True)
def _isolated_sessions(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep the on-disk session store inside the test's own tmp_path."""
    session_dir = tmp_path / ".sessions"
    session_dir.mkdir()
    monkeypatch.setattr(train_mod, "SESSION_DIR", session_dir)
    return session_dir


class TestADeviceTorchCannotParseIsRefused:
    """The spelling is graded against torch's own parser before the argv is built."""

    @pytest.mark.parametrize("device", UNUSABLE_DEVICES)
    def test_an_unusable_device_never_reaches_the_argv(self, device: Any) -> None:
        pytest.importorskip("torch")
        with pytest.raises(ValueError) as excinfo:
            _build(device=device)
        assert "device" in str(excinfo.value)

    def test_the_refusal_names_the_parameter_the_tool_and_the_value(self) -> None:
        pytest.importorskip("torch")
        with pytest.raises(ValueError) as excinfo:
            _build(device="gpu")
        message = str(excinfo.value)
        assert "device" in message
        assert "lerobot_train" in message
        assert "gpu" in message

    def test_the_refusal_names_a_spelling_the_caller_can_use_instead(self) -> None:
        """A refusal that does not say what to pass costs the caller a round trip."""
        pytest.importorskip("torch")
        with pytest.raises(ValueError) as excinfo:
            _build(device="gpu")
        assert "cuda" in str(excinfo.value)

    def test_a_non_string_is_refused_for_its_type_not_by_torch(self) -> None:
        """The type gate runs first, so the verdict cannot vary by machine.

        ``torch.device(0)`` reads the accelerator inventory rather than a
        spelling, so consulting torch for a non-string would build this argv on a
        GPU box and refuse it on a CPU box.
        """
        with pytest.raises(ValueError) as excinfo:
            _build(device=0)
        message = str(excinfo.value)
        assert "must be a torch device string" in message
        assert "int" in message


class TestOnlyTheSpellingIsGradedNeverAvailability:
    """A run may name a device the dispatching machine does not have."""

    @pytest.mark.parametrize("device", USABLE_DEVICES)
    def test_a_parseable_device_still_reaches_the_argv(self, device: str) -> None:
        assert f"--policy.device={device}" in _build(device=device)

    def test_cuda_builds_on_a_machine_without_cuda(self) -> None:
        """The premise the whole guard rests on, asserted where it is false."""
        torch = pytest.importorskip("torch")
        if torch.cuda.is_available():
            pytest.skip("this machine has CUDA, so it cannot witness the CPU-only case")
        assert "--policy.device=cuda" in _build(device="cuda")

    def test_the_default_device_is_accepted(self) -> None:
        """``device`` defaults to ``"cuda"``; the guard must not refuse the default."""
        assert "--policy.device=cuda" in _build()


class TestTheDomainIsSourcedLiveRatherThanCopied:
    """Executable premises, so the reasoning cannot silently become wrong."""

    @pytest.mark.parametrize("device", [value for value in UNUSABLE_DEVICES if isinstance(value, str)])
    def test_torch_really_rejects_every_unusable_spelling(self, device: str) -> None:
        """Non-vacuity: the probe set is refused by torch, not by a local list."""
        torch = pytest.importorskip("torch")
        with pytest.raises((RuntimeError, ValueError)):
            torch.device(device)

    @pytest.mark.parametrize("device", USABLE_DEVICES)
    def test_torch_really_accepts_every_usable_spelling(self, device: str) -> None:
        """So the admitted half is torch's verdict too, not a copied allow-list."""
        torch = pytest.importorskip("torch")
        assert torch.device(device) is not None

    def test_torchs_own_message_enumerates_the_admitted_device_types(self) -> None:
        """Which is why the refusal quotes it instead of restating the set."""
        torch = pytest.importorskip("torch")
        with pytest.raises(RuntimeError) as excinfo:
            torch.device("gpu")
        assert "cuda" in str(excinfo.value)

    def test_an_unimportable_torch_leaves_the_value_alone(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """With no torch the domain is unknown, so the value passes through.

        The same contract ``_policy_config_field_names`` documents for its own
        field set: an absent source of truth must not become a refusal.
        """
        real_import = builtins.__import__

        def _no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch" or name.startswith("torch."):
                raise ImportError("torch is not installed")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        assert "--policy.device=gpu" in _build(device="gpu")


class TestTheRefusalReachesTheCallerBeforeAnyProcessStarts:
    """A rejected device must be reported, not launched and then discovered."""

    def test_the_tool_reports_an_error_envelope_rather_than_raising(self, tmp_path: Path) -> None:
        pytest.importorskip("torch")
        dataset = _write_dataset(tmp_path / "cubes")
        result = train_mod.lerobot_train(action="start", dataset_root=str(dataset), device="gpu")
        assert result["status"] == "error"
        text = "\n".join(item["text"] for item in result["content"] if "text" in item)
        assert "device" in text

    def test_a_refused_device_launches_no_process(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        pytest.importorskip("torch")
        dataset = _write_dataset(tmp_path / "cubes")
        launched: list[Any] = []

        def _fail_if_launched(*args: Any, **kwargs: Any) -> _FakeProc:
            launched.append(args)
            return _FakeProc()

        monkeypatch.setattr(train_mod.subprocess, "Popen", _fail_if_launched)
        result = train_mod.lerobot_train(action="start", dataset_root=str(dataset), device="gpu")
        assert result["status"] == "error"
        assert launched == [], "a refused device still spawned a training process"


class TestTheScopeIsNoWiderThanTheDefect:
    """A resumed run's argv carries no device, so it must not be refused for one."""

    def test_a_resumed_run_is_not_refused_for_an_unusable_device(self, tmp_path: Path) -> None:
        output_dir = _write_resumable_checkpoint(tmp_path / "train_out")
        cmd = _build(device="gpu", output_dir=str(output_dir), resume=True)
        assert "--resume=true" in cmd

    def test_a_resumed_run_really_emits_no_device_flag(self, tmp_path: Path) -> None:
        """Non-vacuity for the scoping decision above."""
        output_dir = _write_resumable_checkpoint(tmp_path / "train_out")
        cmd = _build(device="cuda", output_dir=str(output_dir), resume=True)
        assert not [flag for flag in cmd if flag.startswith("--policy.device=")]

    def test_the_other_argv_guards_are_untouched(self) -> None:
        with pytest.raises(ValueError, match="num_gpus must be >= 1"):
            _build(num_gpus=0)
        with pytest.raises(ValueError, match="steps must be a positive integer"):
            _build(steps=0)


def test_the_probe_sets_are_disjoint_and_cover_both_kinds() -> None:
    """Guards the probe sets themselves against a future edit collapsing them."""
    assert not set(USABLE_DEVICES) & {value for value in UNUSABLE_DEVICES if isinstance(value, str)}
    assert any(not isinstance(value, str) for value in UNUSABLE_DEVICES), "no non-string probe"
    assert any(isinstance(value, str) for value in UNUSABLE_DEVICES), "no unparseable-string probe"
