"""``RLTrainSpec.device`` is held to torch's device-string domain by ``validate``.

All three from-scratch RL backends resolve the device the same way in
:meth:`setup`::

    self.device = torch.device(spec.device or ("cuda" if torch.cuda.is_available() else "cpu"))

so ``device`` is spent by ``torch.device`` itself - which judges nothing on the
spec's behalf - and every network, replay buffer and rollout tensor the run
allocates is placed on the result. ``device`` was the one caller-supplied knob on
this spec with no domain, while the counts, the interval coefficients, the loss
weights, the learning rate, the seed, the launch topology and the network width
beside it are all held to one. So
:meth:`~strands_robots.training.base.Trainer.validate` returned ``[]`` - the one
thing its contract says an empty list does not mean - for a spec that cannot
launch.

Measured on a 6-DoF SO-101 MuJoCo env with PPO before this gate, each under
``validate() == []``:

* ``"gpu"`` - the ordinary mistake, the word the rest of the world uses - and
  ``"cuda:abc"`` raise ``RuntimeError`` out of ``setup`` from the
  ``torch.device`` line itself.
* ``1`` is worse, and is why a non-``str`` is refused before torch is consulted:
  ``torch.device(1)`` constructs on ANY host, so the run reached the first
  ``.to()`` and died with ``CUDA error: invalid device ordinal`` raised from a
  ``torch/nn/modules/module.py`` frame naming neither the field nor the run - and
  the same spec would train on a host with more GPUs. One spec, two answers,
  decided by the machine's inventory.

The domain is :func:`~strands_robots.utils.torch_device_error`, the one owner the
``lerobot_train`` tool and
:class:`~strands_robots.training.lerobot.LerobotTrainer` also consult. Two
properties are deliberately NOT graded, and both have a control below:
**availability** (a spec legitimately names a device the machine writing it does
not have, because a queued run is dispatched from one host and executed on
another) and **an unstated device** (``spec.device or ...`` documents a falsy
value as "resolve the default", so it never reaches ``torch.device`` as written).
"""

from __future__ import annotations

import pathlib
from typing import Any

import pytest

from strands_robots.training.base import Trainer
from strands_robots.training.rl.base_algo import RLTrainSpec
from strands_robots.training.rl.fast_sac import FastSacTrainer
from strands_robots.training.rl.fast_td3 import FastTd3Trainer
from strands_robots.training.rl.ppo import PpoTrainer

#: The three backends that carry ``RLTrainSpec.device`` into ``torch.device``.
BACKENDS = (PpoTrainer, FastSacTrainer, FastTd3Trainer)

#: Measured against torch's own ``torch.device`` on this tree. ``'gpu'`` is the
#: ordinary mistake; ``1`` and ``2.0`` are the non-``str`` case torch would
#: answer by reading the accelerator inventory.
UNUSABLE: tuple[Any, ...] = ("gpu", "CUDA", "nvidia", "cuda:abc", "  ", 1, 2.0, ["cuda"])

#: A device type, optionally with an index. ``'mps'`` is here on a box that does
#: not have it: the spelling is the domain, not the inventory.
USABLE = ("cpu", "cuda", "cuda:0", "mps")


def _spec(tmp_path: pathlib.Path, **overrides: Any) -> RLTrainSpec:
    """A spec every backend's ``validate`` passes clean.

    Deliberately loose in its annotation: the probes below pass values whose type
    the field does not declare, which is the point of the gate under test.
    """
    fields: dict[str, Any] = {
        "env_factory": lambda: None,
        "total_timesteps": 64,
        "rollout_steps": 4,
        "num_envs": 1,
        "num_mini_batches": 1,
        "batch_size": 8,
        "learning_starts": 8,
        "output_dir": str(tmp_path / "out"),
    }
    fields.update(overrides)
    return RLTrainSpec(**fields)


def _device_problems(trainer: Trainer, spec: RLTrainSpec) -> list[str]:
    return [p for p in trainer.validate(spec) if "device" in p]


class TestAnUnusableDeviceIsReportedByThePreflight:
    """The headline: ``validate`` no longer calls an unlaunchable spec launchable."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", UNUSABLE)
    def test_a_device_no_torch_build_can_parse_is_reported(
        self, tmp_path: pathlib.Path, backend: type[Trainer], device: Any
    ) -> None:
        pytest.importorskip("torch")
        problems = _device_problems(backend(), _spec(tmp_path, device=device))
        assert problems, (
            f"{backend.__name__}.validate() returned no device problem for device={device!r}, "
            "which torch cannot parse; setup() hands it straight to torch.device, so the run "
            "aborts after the preflight called the spec launchable"
        )

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_the_report_names_the_backend_the_value_and_the_admitted_shape(
        self, tmp_path: pathlib.Path, backend: type[Trainer]
    ) -> None:
        """A refusal an operator can act on: whose preflight, which value, what a device looks like."""
        pytest.importorskip("torch")
        trainer = backend()
        (problem,) = _device_problems(trainer, _spec(tmp_path, device="gpu"))
        assert problem.startswith(f"{trainer.provider_name}:"), problem
        assert "'gpu'" in problem, problem
        assert "cuda:0" in problem, f"the refusal does not show an indexed device: {problem}"

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_a_non_string_device_is_reported_without_consulting_torch(
        self, tmp_path: pathlib.Path, backend: type[Trainer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``torch.device(1)`` constructs on any host and dies at the first ``.to()``.

        Consulting torch for a non-``str`` would make one spec validate on a GPU
        box and fail on a CPU box, which is the availability coupling this gate
        rules out. The stub below is what proves the order: it raises if the gate
        reaches ``torch.device`` at all.
        """
        torch = pytest.importorskip("torch")

        def _explode(*_a: Any, **_k: Any) -> Any:  # pragma: no cover - must not run
            raise AssertionError("torch was consulted for a non-str device")

        monkeypatch.setattr(torch, "device", _explode)
        problems = _device_problems(backend(), _spec(tmp_path, device=1))
        assert any("must be a torch device string" in p for p in problems), problems
        assert any("int" in p for p in problems), problems


class TestTheDomainHasOneOwner:
    """The tool, the supervised trainer and the RL preflight admit the same set."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", UNUSABLE + USABLE)
    def test_every_surface_that_spends_the_value_agrees(
        self, tmp_path: pathlib.Path, backend: type[Trainer], device: Any
    ) -> None:
        """One quantity reaching torch from three layers must have one admitted set.

        ``lerobot_train`` builds a detached argv from it, ``LerobotTrainer``
        assigns it onto a lerobot config in-process, and these backends hand it to
        ``torch.device``. A device one refuses and another accepts is the drift
        this shared owner removes.
        """
        pytest.importorskip("torch")
        from strands_robots.tools.lerobot_train import _torch_device_error

        tool_refuses = _torch_device_error(device) is not None
        rl_refuses = bool(_device_problems(backend(), _spec(tmp_path, device=device)))
        assert rl_refuses == tool_refuses, (
            f"device={device!r}: the lerobot_train tool {'refuses' if tool_refuses else 'admits'} it "
            f"and {backend.__name__} {'refuses' if rl_refuses else 'admits'} it"
        )

    @staticmethod
    def _calls(source: str) -> set[str]:
        """Every call in ``source``, as dotted names, read from the AST.

        The AST rather than the text because these functions *document* the
        ``torch.device(spec.device or ...)`` line they exist to bound, and a text
        scan would read that prose as the implementation.
        """
        import ast
        import textwrap

        names: set[str] = set()
        for node in ast.walk(ast.parse(textwrap.dedent(source))):
            if isinstance(node, ast.Call):
                names.add(ast.unparse(node.func))
        return names

    def test_no_consumer_re_implements_the_domain(self) -> None:
        """Exactly one function calls ``torch.device``; the rest ask it.

        The tool's guard is imported as a *function* rather than off its module:
        a tool and its module share a name on the ``tools`` package, and
        ``import strands_robots.tools.lerobot_train as mod`` binds by attribute
        lookup on that package - so in a session that has already touched the
        tool, ``mod`` is the ``DecoratedFunctionTool``, not the module.
        """
        import inspect

        from strands_robots import utils
        from strands_robots.tools.lerobot_train import _torch_device_error as tool_device_error
        from strands_robots.training import _validate
        from strands_robots.training.lerobot import LerobotTrainer

        owner_src = inspect.getsource(utils.torch_device_error)
        assert "torch.device" in self._calls(owner_src), "the owner does not read the domain from torch"

        for consumer in (
            _validate.torch_device_problems,
            tool_device_error,
            LerobotTrainer._device_problems,
        ):
            calls = self._calls(inspect.getsource(consumer))
            assert "torch_device_error" in calls, f"{consumer.__qualname__} does not route through the owner"
            assert "torch.device" not in calls, f"{consumer.__qualname__} re-implements the domain"

    def test_the_owner_states_no_device_vocabulary_of_its_own(self) -> None:
        """A torch build that gains a backend is admitted with no edit here."""
        import inspect

        from strands_robots import utils

        owner = inspect.getsource(utils.torch_device_error)
        for copied in ("xpu", "mkldnn", "opengl", "vulkan", "hpu"):
            assert copied not in owner, f"the domain restates a torch device type: {copied}"


class TestWhatThisDeliberatelyDoesNotGrade:
    """Controls: each fails for a specific over-tight fix."""

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("device", USABLE)
    def test_a_usable_spelling_still_validates_clean(
        self, tmp_path: pathlib.Path, backend: type[Trainer], device: str
    ) -> None:
        pytest.importorskip("torch")
        assert backend().validate(_spec(tmp_path, device=device)) == []

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_availability_is_not_graded(
        self, tmp_path: pathlib.Path, backend: type[Trainer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A device this machine does not have is still a valid spec.

        Fails for a fix that reaches for ``torch.cuda.is_available()``.
        """
        torch = pytest.importorskip("torch")
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        assert backend().validate(_spec(tmp_path, device="cuda")) == []
        assert backend().validate(_spec(tmp_path, device="cuda:7")) == []

    @pytest.mark.parametrize("backend", BACKENDS)
    @pytest.mark.parametrize("falsy", [None, "", 0])
    def test_an_unstated_device_resolves_through_the_documented_default(
        self, tmp_path: pathlib.Path, backend: type[Trainer], falsy: Any
    ) -> None:
        """``spec.device or (...)`` predates this gate and still owns the falsy case.

        A falsy value never reaches ``torch.device`` as written - it is replaced by
        the auto-resolved device - so it is not this gate's to judge. Fails for a
        fix that refuses ``device=None``.
        """
        pytest.importorskip("torch")
        assert backend().validate(_spec(tmp_path, device=falsy)) == []

    @pytest.mark.parametrize("backend", BACKENDS)
    def test_an_absent_torch_leaves_the_value_unguarded(
        self, tmp_path: pathlib.Path, backend: type[Trainer], monkeypatch: pytest.MonkeyPatch
    ) -> None:
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
        assert _device_problems(backend(), _spec(tmp_path, device="gpu")) == []
