"""Unit tests for ``_MotionBricksAgentAdapter.build`` - the checkpoint-load seam.

``build`` is the heavy classmethod that turns a checkpoint tree plus the
``motionbricks`` package into a ready generator wrapped in the adapter. The
real package + git-LFS weights only exist in ``tests_integ``, but the
*orchestration* build performs is pure Python and has real correctness
contracts worth pinning on any machine (no GPU, no checkpoints, no
``motionbricks`` install):

* It ``os.chdir`` s into the checkpoint tree's parent (upstream configs
  reference the skeleton relative to CWD) and MUST restore the previous CWD
  afterwards - a leaked ``chdir`` would corrupt every later path resolution in
  the process.
* On a ``cpu`` device it temporarily neutralises ``torch.cuda.set_device`` so
  the upstream loader cannot pin a GPU, and MUST restore the original callable
  afterwards - a leaked stub would silently break real CUDA use elsewhere in
  the process.
* It loads BOTH the ``pose`` and ``root`` state dicts and derives the adapter's
  clip keys, per-clip token masks, and min/max token counts from the generator.

Both process-global restorations happen in a ``finally``, so they must survive a
mid-build failure too. These tests inject stub ``motionbricks.*`` modules into
``sys.modules`` to exercise that orchestration without the real dependency.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from strands_robots.policies.motionbricks.policy import (
    MotionBricksPolicy,
    _MotionBricksAgentAdapter,
)


class _FakeModel:
    """Upstream sub-model stub: records the state dict loaded into it."""

    def __init__(self) -> None:
        self.args = SimpleNamespace(model="fake")
        self.loaded_state_dict: dict[str, Any] | None = None

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        self.loaded_state_dict = state_dict


class _FakeInferencer:
    def __init__(self) -> None:
        self._args = {"min_tokens": 4, "max_tokens": 13}


class _FakeFullAgent:
    """Stand-in for ``full_navigation_agent``; records its construction kwargs."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self.init_args = args
        self.init_kwargs = kwargs

    def to(self, device: str) -> _FakeFullAgent:
        self.to_device = device
        return self


class _FakeClipHolder:
    # "idle" declares no token mask (-> None); "walk" declares an explicit one.
    CLIPS = {
        "idle": {"speed": 0.0},
        "walk": {"allowed_pred_num_tokens": [1, 1, 1]},
    }


def _install_fake_motionbricks(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pose_model: _FakeModel,
    root_model: _FakeModel,
    pose_ckpt: Path,
    root_ckpt: Path,
    on_test: Any = None,
) -> _FakeFullAgent:
    """Register stub ``motionbricks.*`` modules and return the fake agent build() yields."""
    fake_agent = _FakeFullAgent()

    def fake_test(args: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        if on_test is not None:
            on_test(args)
        models = {"pose": pose_model, "root": root_model}
        confs = {
            "pose": SimpleNamespace(ckpt_path=str(pose_ckpt)),
            "root": SimpleNamespace(ckpt_path=str(root_ckpt)),
        }
        return models, confs

    def fake_full_navigation_agent(*args: Any, **kwargs: Any) -> _FakeFullAgent:
        fake_agent.init_args = args
        fake_agent.init_kwargs = kwargs
        return fake_agent

    def fake_motion_inference(*args: Any, **kwargs: Any) -> _FakeInferencer:
        return _FakeInferencer()

    modules: dict[str, ModuleType] = {}
    for name in (
        "motionbricks",
        "motionbricks.exp_setup",
        "motionbricks.exp_setup.experiment",
        "motionbricks.motion_backbone",
        "motionbricks.motion_backbone.demo",
        "motionbricks.motion_backbone.demo.clips",
        "motionbricks.motion_backbone.demo.full_agent",
        "motionbricks.motion_backbone.inference",
        "motionbricks.motion_backbone.inference.motion_inference",
    ):
        modules[name] = ModuleType(name)

    modules["motionbricks.exp_setup.experiment"].test = fake_test  # type: ignore[attr-defined]
    modules["motionbricks.motion_backbone.demo.clips"].clip_holder_G1 = _FakeClipHolder  # type: ignore[attr-defined]
    modules["motionbricks.motion_backbone.demo.full_agent"].full_navigation_agent = fake_full_navigation_agent  # type: ignore[attr-defined]
    modules["motionbricks.motion_backbone.inference.motion_inference"].motion_inference = fake_motion_inference  # type: ignore[attr-defined]

    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    # require_optional would otherwise fail (real package absent).
    monkeypatch.setattr(
        "strands_robots.policies.motionbricks.policy.require_optional",
        lambda *a, **k: None,
    )
    return fake_agent


def _write_ckpt(path: Path) -> None:
    torch.save({"state_dict": {"w": torch.zeros(1)}}, path)


def test_build_wires_checkpoints_into_adapter_and_restores_process_state(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    result_dir = tmp_path / "out"
    result_dir.mkdir()
    pose_ckpt = result_dir / "pose.ckpt"
    root_ckpt = result_dir / "root.ckpt"
    _write_ckpt(pose_ckpt)
    _write_ckpt(root_ckpt)

    pose_model = _FakeModel()
    root_model = _FakeModel()
    fake_agent = _install_fake_motionbricks(
        monkeypatch,
        pose_model=pose_model,
        root_model=root_model,
        pose_ckpt=pose_ckpt,
        root_ckpt=root_ckpt,
    )

    cwd_before = os.getcwd()
    set_device_before = torch.cuda.set_device

    pol = MotionBricksPolicy(result_dir=str(result_dir), device="cpu")

    # The generator was wrapped in the adapter with fields derived from the model.
    agent = pol._agent
    assert isinstance(agent, _MotionBricksAgentAdapter)
    assert agent.clip_keys == ["idle", "walk"]
    assert agent.clip_token_specs == [None, [1, 1, 1]]
    assert agent.min_token == 4
    assert agent.max_token == 13
    assert agent._device == "cpu"
    assert agent._fa is fake_agent

    # Both sub-model state dicts were loaded from their checkpoint files.
    assert pose_model.loaded_state_dict is not None
    assert root_model.loaded_state_dict is not None

    # Process-global side effects were fully restored (the finally contract).
    assert os.getcwd() == cwd_before
    assert torch.cuda.set_device is set_device_before


def test_build_restores_process_state_when_load_fails(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_dir = tmp_path / "out"
    result_dir.mkdir()
    pose_ckpt = result_dir / "pose.ckpt"
    root_ckpt = result_dir / "root.ckpt"
    _write_ckpt(pose_ckpt)
    _write_ckpt(root_ckpt)

    def boom(_args: Any) -> None:
        raise RuntimeError("checkpoint layout mismatch")

    _install_fake_motionbricks(
        monkeypatch,
        pose_model=_FakeModel(),
        root_model=_FakeModel(),
        pose_ckpt=pose_ckpt,
        root_ckpt=root_ckpt,
        on_test=boom,
    )

    cwd_before = os.getcwd()
    set_device_before = torch.cuda.set_device

    with pytest.raises(RuntimeError, match="checkpoint layout mismatch"):
        MotionBricksPolicy(result_dir=str(result_dir), device="cpu")

    # Even on a mid-build failure the CWD and cuda.set_device are restored.
    assert os.getcwd() == cwd_before
    assert torch.cuda.set_device is set_device_before
