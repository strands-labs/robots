"""``LerobotTrainer.device`` is held to torch's device-string domain by ``validate``.

``device`` is the trainer's one constructor knob and it reaches lerobot twice:
:meth:`~strands_robots.training.lerobot.LerobotTrainer.build_command`
interpolates it into ``--policy.device=`` (and ``--reward_model.device=``), and
:meth:`~strands_robots.training.lerobot.LerobotTrainer.build_config` assigns it
onto ``policy_cfg.device``. Neither refuses an unusable spelling, so before this
gate ``validate`` returned ``[]`` for a run that cannot start - and an empty list
is the one thing :meth:`~strands_robots.training.base.Trainer.validate`
documents as meaning the spec IS launchable, from a preflight whose stated job is
to run "before anything expensive starts".

Every other caller-supplied value that preflight sees is already held to a
domain - the run-size counts, ``learning_rate``, ``seed``, the launch topology,
``val_episodes``, and ``dataset_repo_id`` against a Hub-id pattern. ``device``
was the one knob beside them with none.

Two properties are deliberately NOT graded, and both have a control below:

* **availability** - ``torch.device("cuda")`` constructs on a CPU-only box, and a
  spec legitimately names a device the machine writing it does not have, because
  a queued or containerised run is dispatched from one host and executed on
  another.
* **the tool surface's own answer** - ``tools.lerobot_train`` holds the same
  value to the same domain for its own detached argv. The two live in layers that
  cannot import each other, so the rule is stated twice; the parity test here is
  what keeps the two admitted sets from drifting, rather than an alias.
"""

import json
from typing import Any

import pytest

from strands_robots.training import TrainSpec, create_trainer
from strands_robots.training.lerobot import LerobotTrainer

# Measured against torch's own ``torch.device`` on this tree. ``'gpu'`` is the
# ordinary mistake - it is the word the rest of the world uses and it is not a
# torch device type.
UNUSABLE = ["gpu", "CUDA", "nvidia", "cuda:abc", "  "]

# A device type, optionally with an index. ``'mps'`` and ``'cuda'`` are here on a
# box that has neither: the spelling is the domain, not the inventory.
USABLE = ["cpu", "cuda", "cuda:0", "mps"]


def _dataset(tmp_path) -> str:
    """A LeRobotDataset v3 root complete enough for ``validate`` to pass it."""
    meta = tmp_path / "ds" / "meta"
    meta.mkdir(parents=True, exist_ok=True)
    (meta / "info.json").write_text(
        json.dumps({"total_episodes": 12, "codebase_version": "v3.0", "fps": 30}),
        encoding="utf-8",
    )
    return str(tmp_path / "ds")


def _spec(tmp_path) -> TrainSpec:
    return TrainSpec(
        dataset_root=_dataset(tmp_path),
        base_model="lerobot/act",
        output_dir=str(tmp_path / "out"),
        steps=10,
    )


class TestAnUnusableDeviceIsReportedByThePreflight:
    """The headline: ``validate`` no longer calls an unlaunchable spec launchable."""

    @pytest.mark.parametrize("device", UNUSABLE)
    def test_a_device_no_torch_build_can_parse_is_reported(self, tmp_path, device: str) -> None:
        pytest.importorskip("torch")
        trainer = LerobotTrainer(policy_type="act", device=device)
        spec = _spec(tmp_path)
        problems = trainer.validate(spec)
        argv = [a for a in trainer.build_command(spec) if a.startswith("--policy.device=")]
        assert problems, (
            f"validate() returned [] for device={device!r}, which torch cannot parse; "
            f"build_command still emits {argv} and build_config assigns the same string "
            "to policy_cfg.device, so the run aborts after the preflight called it launchable"
        )

    @pytest.mark.parametrize("device", UNUSABLE)
    def test_the_report_names_the_value_and_the_admitted_shape(self, tmp_path, device: str) -> None:
        """A refusal an operator can act on: the value, and what a device looks like."""
        pytest.importorskip("torch")
        problems = LerobotTrainer(policy_type="act", device=device).validate(_spec(tmp_path))
        joined = " ".join(problems)
        assert "device" in joined
        assert repr(device) in joined or device in joined
        assert "cuda:0" in joined, f"the refusal does not show an indexed device: {joined}"

    def test_a_non_string_device_is_reported_without_consulting_torch(self, tmp_path, monkeypatch) -> None:
        """``torch.device(0)`` reads the accelerator inventory, so a non-str is refused first.

        Consulting torch for it would make one spec validate on a GPU box and fail
        on a CPU box, which is the availability coupling this gate rules out. The
        stub below is what proves the order: it raises if the gate reaches
        ``torch.device`` at all. It is asserted over the gate rather than over
        ``validate``, because other checks in that preflight legitimately read
        torch for their own reasons.
        """
        torch = pytest.importorskip("torch")

        def _explode(*_a: Any, **_k: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("torch was consulted for a non-str device")

        monkeypatch.setattr(torch, "device", _explode)
        trainer = LerobotTrainer(policy_type="act", device=1)  # type: ignore[arg-type]
        problems = trainer._device_problems()
        assert any("must be a torch device string" in p for p in problems), problems
        assert any("int" in p for p in problems), problems

    def test_a_non_string_device_reaches_the_preflight(self, tmp_path) -> None:
        """And the gate above is wired into ``validate``, not only callable."""
        pytest.importorskip("torch")
        trainer = LerobotTrainer(policy_type="act", device=1)  # type: ignore[arg-type]
        assert any("must be a torch device string" in p for p in trainer.validate(_spec(tmp_path)))


class TestTheAdmittedDomainIsTorchsOwn:
    """Sourced live from ``torch.device``, and identical to the tool surface's."""

    @pytest.mark.parametrize("device", UNUSABLE + USABLE)
    def test_the_trainer_and_the_tool_agree_on_every_probe(self, tmp_path, device: str) -> None:
        """The rule is stated in two layers that cannot import each other.

        ``strands_robots.tools.lerobot_train`` grades the same value for its own
        detached argv. This is the parity that keeps the two admitted sets from
        drifting - the alternative, one of them importing the other, would invert
        the tool/library layering.
        """
        pytest.importorskip("torch")
        from strands_robots.tools.lerobot_train import _torch_device_error

        tool_refuses = _torch_device_error(device) is not None
        trainer_refuses = bool(LerobotTrainer(policy_type="act", device=device).validate(_spec(tmp_path)))
        assert trainer_refuses == tool_refuses, (
            f"device={device!r}: the tool surface {'refuses' if tool_refuses else 'admits'} it "
            f"and the trainer {'refuses' if trainer_refuses else 'admits'} it"
        )

    def test_the_domain_is_read_from_torch_rather_than_a_copied_list(self) -> None:
        """No device-type vocabulary is restated here, so a new backend needs no edit."""
        import inspect

        src = inspect.getsource(LerobotTrainer._device_problems)
        assert "torch.device(device)" in src
        for copied in ("xpu", "mkldnn", "opengl", "vulkan", "hpu"):
            assert copied not in src, f"the domain restates a torch device type: {copied}"


class TestWhatThisDeliberatelyDoesNotGrade:
    """Controls: each fails for a specific over-tight fix."""

    @pytest.mark.parametrize("device", USABLE)
    def test_a_usable_spelling_still_validates_clean(self, tmp_path, device: str) -> None:
        pytest.importorskip("torch")
        assert LerobotTrainer(policy_type="act", device=device).validate(_spec(tmp_path)) == []

    def test_availability_is_not_graded(self, tmp_path, monkeypatch) -> None:
        """A device this machine does not have is still a valid spec.

        Fails for a fix that reaches for ``torch.cuda.is_available()``.
        """
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert LerobotTrainer(policy_type="act", device="cuda").validate(_spec(tmp_path)) == []
        assert LerobotTrainer(policy_type="act", device="cuda:7").validate(_spec(tmp_path)) == []

    def test_the_auto_resolved_default_validates_clean(self, tmp_path) -> None:
        """``device=None`` resolves through ``_auto_device``; it must not be refused."""
        pytest.importorskip("torch")
        trainer = LerobotTrainer(policy_type="act")
        assert trainer.validate(_spec(tmp_path)) == []
        assert isinstance(trainer.device, str) and trainer.device

    @pytest.mark.parametrize("falsy", [None, "", 0])
    def test_a_falsy_device_still_resolves_through_the_documented_default(self, tmp_path, falsy: Any) -> None:
        """``device or _auto_device()`` predates this gate and still owns the falsy case.

        So a falsy value never reaches the domain check at all - it is replaced by
        the auto-resolved device first, which is what the constructor documents.
        Fails for a fix that refuses ``device=None`` or an empty string.
        """
        pytest.importorskip("torch")
        trainer = LerobotTrainer(policy_type="act", device=falsy)  # type: ignore[arg-type]
        assert isinstance(trainer.device, str) and trainer.device
        assert trainer.validate(_spec(tmp_path)) == []

    def test_the_argv_for_a_usable_device_is_unchanged(self, tmp_path) -> None:
        """The gate adds a verdict, never a different command."""
        cmd = LerobotTrainer(policy_type="act", device="cpu").build_command(_spec(tmp_path))
        assert "--policy.device=cpu" in cmd
        assert sum(1 for a in cmd if a.startswith("--policy.device=")) == 1

    def test_an_absent_torch_leaves_the_value_unguarded(self, tmp_path, monkeypatch) -> None:
        """With no torch the domain is unknown, so the value passes through.

        Fails for a fix that refuses every device when torch cannot be imported,
        which would make the preflight unusable on a dispatch-only host.
        """
        import builtins

        real_import = builtins.__import__

        def _no_torch(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "torch":
                raise ImportError("no torch")
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", _no_torch)
        assert LerobotTrainer(policy_type="act", device="gpu")._device_problems() == []


class TestTheGateIsReachedFromTheDocumentedRoute:
    """``create_trainer`` is how a caller builds this trainer, and it forwards ``device``."""

    def test_the_registry_route_carries_device_into_the_gate(self, tmp_path) -> None:
        pytest.importorskip("torch")
        # Annotated loose: create_trainer is declared to return the Trainer ABC,
        # which carries no ``device`` - that is the point of asserting the
        # concrete trainer received it.
        trainer: Any = create_trainer("lerobot_local", device="gpu")
        assert trainer.device == "gpu"
        assert trainer.validate(_spec(tmp_path)), "the registry route bypasses the device gate"
