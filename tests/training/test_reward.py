"""Tests for the SARM reward-model production helpers (the producing half of RA-BC).

Covers :mod:`strands_robots.training.reward`:

* :func:`compute_rabc_weights` - the in-process wrapper that runs a trained SARM
  over a dataset and emits the ``sarm_progress.parquet`` RA-BC consumes. lerobot
  is mocked at the seam (``_require_sarm_progress``) so the test asserts argument
  forwarding + input validation without GPU, a real model, or a real dataset.
* :func:`_require_sarm_progress` - the matplotlib gate, the success import, and
  the too-old-lerobot ImportError, all driven deterministically by swapping the
  ``require_optional`` seam and ``sys.modules`` entries so the behavior is the
  same whether or not the optional deps are installed in the test environment.
* :func:`reward_progress` - tensor / ndarray / list / scalar return
  normalization to plain floats.
* :func:`load_reward_model` - reward-config build + load delegation to lerobot,
  plus the too-old-lerobot ImportError, both mocked at ``sys.modules`` so the
  test never silently skips on environments lacking ``lerobot.rewards``.
"""

from __future__ import annotations

import importlib.machinery
import sys
import types

import numpy as np
import pytest

from strands_robots.training import reward as reward_mod
from strands_robots.training.reward import (
    compute_rabc_weights,
    load_reward_model,
    reward_progress,
)

# Submodule path lerobot exposes SARM's in-process progress function under.
_SARM_MOD = "lerobot.rewards.sarm.compute_rabc_weights"


def _fake_module(name: str, **attrs: object) -> types.ModuleType:
    """Build an importable stand-in module with a real ``__spec__``.

    A bare ``types.ModuleType`` has ``__spec__ = None``, which makes
    ``importlib.util.find_spec`` raise ``ValueError`` instead of resolving the
    module, so we attach a minimal loader-less spec to keep it discoverable.
    """
    mod = types.ModuleType(name)
    mod.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    for key, value in attrs.items():
        setattr(mod, key, value)
    return mod


class TestComputeRabcWeights:
    """compute_rabc_weights() argument forwarding + input validation."""

    def _patch_compute(self, monkeypatch):
        calls: dict = {}

        def fake_compute_sarm_progress(**kwargs):
            calls.update(kwargs)
            return "/tmp/ds/sarm_progress.parquet"

        monkeypatch.setattr(reward_mod, "_require_sarm_progress", lambda: fake_compute_sarm_progress)
        return calls

    def test_forwards_local_root_and_returns_path(self, monkeypatch):
        calls = self._patch_compute(monkeypatch)
        out = compute_rabc_weights("/ckpt/sarm", dataset_root="/tmp/ds", head_mode="sparse", device="cpu", stride=5)
        assert out == "/tmp/ds/sarm_progress.parquet"
        # A local root is passed through as the dataset arg (lerobot accepts a path).
        assert calls["dataset_repo_id"] == "/tmp/ds"
        assert calls["reward_model_path"] == "/ckpt/sarm"
        assert calls["head_mode"] == "sparse"
        assert calls["device"] == "cpu"
        assert calls["stride"] == 5
        # Visualizations are always disabled by the wrapper.
        assert calls["num_visualizations"] == 0

    def test_forwards_hub_repo_and_output_path(self, monkeypatch):
        calls = self._patch_compute(monkeypatch)
        compute_rabc_weights("/ckpt", dataset_repo_id="org/ds", output_path="/out/p.parquet", device="cpu")
        assert calls["dataset_repo_id"] == "org/ds"
        assert calls["output_path"] == "/out/p.parquet"

    def test_requires_exactly_one_data_source(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="exactly one data source"):
            compute_rabc_weights("/ckpt")
        with pytest.raises(ValueError, match="exactly one data source"):
            compute_rabc_weights("/ckpt", dataset_root="/d", dataset_repo_id="org/ds")

    def test_rejects_bad_head_mode(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="head_mode"):
            compute_rabc_weights("/ckpt", dataset_root="/d", head_mode="bogus")

    def test_rejects_bad_stride(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="stride"):
            compute_rabc_weights("/ckpt", dataset_root="/d", stride=0)

    def test_rejects_leading_dash_value(self, monkeypatch):
        self._patch_compute(monkeypatch)
        with pytest.raises(ValueError, match="must not start with '-'"):
            compute_rabc_weights("-x", dataset_root="/d")


class TestRequireSarmProgress:
    """_require_sarm_progress() gates on matplotlib then imports lerobot's SARM."""

    def test_returns_lerobot_compute_function(self, monkeypatch):
        # Pass the matplotlib gate, then resolve to an injected SARM module so
        # the success path is exercised without a real matplotlib/lerobot build.
        monkeypatch.setattr(reward_mod, "require_optional", lambda *a, **k: None)
        sentinel = object()
        monkeypatch.setitem(sys.modules, _SARM_MOD, _fake_module(_SARM_MOD, compute_sarm_progress=sentinel))
        assert reward_mod._require_sarm_progress() is sentinel

    def test_requires_matplotlib(self, monkeypatch):
        def boom(name, **kwargs):
            raise ImportError(f"matplotlib missing for {name}")

        monkeypatch.setattr(reward_mod, "require_optional", boom)
        with pytest.raises(ImportError, match="matplotlib"):
            reward_mod._require_sarm_progress()

    def test_old_lerobot_raises_clear_error(self, monkeypatch):
        # matplotlib present, but the SARM module is absent (too-old lerobot).
        monkeypatch.setattr(reward_mod, "require_optional", lambda *a, **k: None)
        monkeypatch.setitem(sys.modules, _SARM_MOD, None)
        with pytest.raises(ImportError, match=r"lerobot >= 0\.6") as excinfo:
            reward_mod._require_sarm_progress()
        # lerobot 0.6 (incl. the rewards package) ships from PyPI, so the hint
        # must NOT send the caller chasing a from-source / git+ install.
        msg = str(excinfo.value)
        assert "from source" not in msg
        assert "git+" not in msg
        assert "0.5.2" not in msg


class TestRewardProgress:
    """reward_progress() normalizes the model's compute_reward() return to floats."""

    def test_normalizes_tensor(self):
        torch = pytest.importorskip("torch")

        class Model:
            def compute_reward(self, batch):
                return torch.tensor([[0.1], [0.9]])

        assert reward_progress(Model(), {}) == pytest.approx([0.1, 0.9], abs=1e-6)

    def test_normalizes_ndarray(self):
        # ndarray has .flatten() but no .detach(): exercises the elif branch.
        class Model:
            def compute_reward(self, batch):
                return np.array([[0.25], [0.75]])

        assert reward_progress(Model(), {}) == pytest.approx([0.25, 0.75], abs=1e-6)

    def test_passes_through_plain_list(self):
        class Model:
            def compute_reward(self, batch):
                return [0.2, 0.8]

        assert reward_progress(Model(), {}) == [0.2, 0.8]

    def test_wraps_scalar(self):
        class Model:
            def compute_reward(self, batch):
                return 0.5

        assert reward_progress(Model(), {}) == [0.5]

    def test_torch_mock_models_the_detach_chain(self):
        # reward_progress() normalizes a torch return via the chain
        # ``rewards.detach().to("cpu").flatten().tolist()``. When real torch is
        # absent, tests/conftest.py installs tests/mocks/torch_mock.MockTensor as
        # ``torch``; ``pytest.importorskip("torch")`` above therefore resolves to
        # the mock rather than skipping (the mock IS a torch module). The mock
        # must implement every method in that chain or the "run all unit tests
        # without PyTorch installed" contract in conftest breaks -- a
        # torch-return test would fail with AttributeError only in the mocked
        # (no-real-torch) environment. Pin that ``flatten``/``tolist`` (the two
        # links the mock previously lacked) survive on MockTensor and that the
        # full chain a torch tensor takes through reward_progress still yields
        # the right floats.
        torch = pytest.importorskip("torch")
        t = torch.tensor([[0.1], [0.9]])
        assert t.detach().to("cpu").flatten().tolist() == pytest.approx([0.1, 0.9], abs=1e-6)


class TestLoadRewardModel:
    """load_reward_model() builds the reward config and delegates the load to lerobot."""

    def test_builds_config_and_loads(self, monkeypatch):
        captured: dict = {}

        def fake_make_cfg(reward_type, **kw):
            captured["type"] = reward_type
            captured.update(kw)
            return "CFG"

        def fake_make_model(cfg):
            captured["model_cfg"] = cfg
            return "MODEL"

        # Inject a fully-formed lerobot.rewards so both find_spec and the inner
        # import resolve to the fakes regardless of the real lerobot install.
        fake_rewards = _fake_module(
            "lerobot.rewards",
            make_reward_model_config=fake_make_cfg,
            make_reward_model=fake_make_model,
        )
        monkeypatch.setitem(sys.modules, "lerobot.rewards", fake_rewards)

        model = load_reward_model("/ckpt/sarm", device="cpu")
        assert model == "MODEL"
        assert captured["type"] == "sarm"
        assert captured["pretrained_path"] == "/ckpt/sarm"
        assert captured["device"] == "cpu"
        assert captured["model_cfg"] == "CFG"

    def test_old_lerobot_raises_clear_error(self, monkeypatch):
        monkeypatch.setattr(reward_mod.importlib.util, "find_spec", lambda name: None)
        with pytest.raises(ImportError, match=r"lerobot >= 0\.6") as excinfo:
            load_reward_model("/ckpt/sarm", device="cpu")
        msg = str(excinfo.value)
        assert "from source" not in msg
        assert "git+" not in msg
        assert "0.5.2" not in msg
