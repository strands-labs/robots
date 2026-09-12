"""``IsaacSimulation.is_available`` must not make torch a precondition.

Isaac Sim's own runtime is PhysX + Warp. It does not import torch, and the
official image ``strands_robots.simulation.isaac._install.ISAAC_SIM_DOCKER_IMAGE``
names (``nvcr.io/nvidia/isaac-sim:6.0.1``) ships no torch at all. Measured in
that container: ``import isaacsim`` and ``from isaacsim import SimulationApp``
both succeed, ``create_world`` steps physics, ``send_action`` drives a joint to
its commanded target and ``render`` returns a 640x480 RTX frame -- while
``is_available()`` answered
``(False, "PyTorch not installed. Isaac Sim requires torch with CUDA support.")``.

``[sim-isaac]`` does not declare torch either (it ships ``usd-core`` and
``imageio``), so the requirement could not be satisfied by installing the extra
that is supposed to enable this backend.

The pre-existing ``test_is_available_returns_tuple`` could not catch it: it only
asserts ``"Isaac Sim" in reason`` when the verdict is False, and the wrong reason
contains that phrase.

Two directions are pinned here, because the fix is not "ignore torch":

  * torch ABSENT -> no verdict. Absence says nothing about the GPU.
  * torch PRESENT and reporting no CUDA -> still False. That is real evidence.

A third case is pinned separately: an ``ImportError`` raised *by CUDA
initialisation* rather than by the import statement must not be reported as
"PyTorch not installed", which is the wrong diagnosis for a torch that is
installed and cannot reach the GPU (AGENTS.md: a ``try`` covers only the
operation whose exception it classifies).
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import pytest


def _kit_entry_point_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """Make the SimulationApp probe succeed without installing Isaac Sim.

    ``is_available`` resolves the entry point with ``importlib.util.find_spec``,
    which for a module already in ``sys.modules`` returns that module's
    ``__spec__``. ``types.ModuleType`` leaves ``__spec__`` as ``None`` and
    ``find_spec`` raises ``ValueError`` on that, so the stand-in carries a real
    :class:`importlib.machinery.ModuleSpec`.
    """
    isaacsim = types.ModuleType("isaacsim")
    isaacsim.SimulationApp = object  # type: ignore[attr-defined]
    isaacsim.__spec__ = importlib.machinery.ModuleSpec("isaacsim", loader=None)
    monkeypatch.setitem(sys.modules, "isaacsim", isaacsim)


class TestAbsentTorchIsNotAVerdict:
    """The configuration that ships no torch is the one measured to work."""

    def test_available_when_torch_is_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        _kit_entry_point_present(monkeypatch)
        # ``setitem(..., None)`` makes ``import torch`` raise ImportError AND
        # restores the real entry on teardown, per the AGENTS.md sys.modules rule.
        monkeypatch.setitem(sys.modules, "torch", None)

        ok, reason = IsaacSimulation.is_available()

        assert ok is True, f"an absent torch must not make Isaac unavailable; got reason={reason!r}"
        assert reason is None

    def test_reason_never_blames_torch_for_being_absent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The exact string the container reported must be unreachable."""
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        _kit_entry_point_present(monkeypatch)
        monkeypatch.setitem(sys.modules, "torch", None)

        _ok, reason = IsaacSimulation.is_available()

        assert reason is None or "PyTorch not installed" not in reason


class TestPresentTorchStillReportsAMissingGpu:
    """torch is a useful signal where it exists; only its absence is not."""

    def test_unavailable_when_torch_reports_no_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        _kit_entry_point_present(monkeypatch)
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: False)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)

        ok, reason = IsaacSimulation.is_available()

        assert ok is False
        assert reason is not None and "CUDA" in reason

    def test_available_when_torch_reports_cuda(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        _kit_entry_point_present(monkeypatch)
        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=lambda: True)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)

        ok, reason = IsaacSimulation.is_available()

        assert ok is True
        assert reason is None


class TestCudaInitFailureIsNotReportedAsAnAbsentTorch:
    """A torch that is installed and cannot reach the GPU is a different fault."""

    def test_import_error_from_cuda_probe_is_not_blamed_on_the_import(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        _kit_entry_point_present(monkeypatch)

        def _raise_import_error() -> bool:
            raise ImportError("libcudart.so.12: cannot open shared object file")

        torch = types.ModuleType("torch")
        torch.cuda = types.SimpleNamespace(is_available=_raise_import_error)  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "torch", torch)

        # The ImportError must escape rather than be swallowed and re-reported as
        # "PyTorch not installed": torch IS installed here, and a caller told
        # otherwise would go and install it again.
        with pytest.raises(ImportError, match="libcudart"):
            IsaacSimulation.is_available()


class TestNoEntryPointStillRefuses:
    """The fix must not soften the refusal that matters: no Isaac Sim at all."""

    def test_unavailable_without_a_simulationapp_entry_point(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from strands_robots.simulation.isaac.simulation import IsaacSimulation

        monkeypatch.setitem(sys.modules, "isaacsim", None)
        monkeypatch.setitem(sys.modules, "omni.isaac.kit", None)

        ok, reason = IsaacSimulation.is_available()

        assert ok is False
        assert reason is not None and "Isaac Sim" in reason
