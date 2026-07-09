"""Fail-soft contract for the single-file checkpoint state-dict loader.

``_load_checkpoint_state_dict`` backs the in-model normalization recovery path
(``strands_robots.policies.lerobot_local.processor``). Its caller invokes it
UNWRAPPED, so the function must be best-effort and never raise into the ACT /
diffusion load path: on any unreadable local or Hub checkpoint it returns
``None`` and lets the policy degrade to passthrough.

A truncated / corrupt ``model.safetensors`` (e.g. an interrupted Hub download)
raises safetensors' own ``SafetensorError`` -- which is neither ``OSError`` nor
``ValueError`` -- so it must be caught explicitly. These tests pin that the
loader returns outputs (a state dict or ``None``) rather than propagating.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("safetensors")

from safetensors.torch import save_file  # noqa: E402

from strands_robots.policies.lerobot_local import processor as processor_mod  # noqa: E402
from strands_robots.policies.lerobot_local.processor import (  # noqa: E402
    _load_checkpoint_state_dict,
)


def _write_valid_checkpoint(directory) -> str:
    """Write a real single-file model.safetensors and return the file path."""
    path = directory / "model.safetensors"
    save_file({"w": torch.zeros(3)}, str(path))
    return str(path)


class TestLocalCheckpoint:
    def test_valid_local_checkpoint_returns_state_dict(self, tmp_path):
        _write_valid_checkpoint(tmp_path)
        result = _load_checkpoint_state_dict(str(tmp_path))
        assert result is not None
        assert "w" in result

    def test_corrupt_local_checkpoint_degrades_to_none(self, tmp_path):
        # A truncated header raises SafetensorError; the loader must swallow it.
        (tmp_path / "model.safetensors").write_bytes(b"corrupt-not-safetensors")
        assert _load_checkpoint_state_dict(str(tmp_path)) is None

    def test_no_local_file_unreachable_hub_returns_none(self, tmp_path, monkeypatch):
        # No local model.safetensors and the Hub is unreachable (offline / IO
        # error) -> None, never a failure escaping into the load path.
        def _raise(*_a, **_k):
            raise OSError("offline")

        monkeypatch.setattr("huggingface_hub.hf_hub_download", _raise)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None


class TestHubCheckpoint:
    def test_hub_download_returns_state_dict(self, tmp_path, monkeypatch):
        # tmp_path has no local checkpoint; the Hub yields a valid file.
        weights_dir = tmp_path / "hub"
        weights_dir.mkdir()
        hub_path = _write_valid_checkpoint(weights_dir)
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_a, **_k: hub_path)

        result = _load_checkpoint_state_dict(str(tmp_path))
        assert result is not None
        assert "w" in result

    def test_corrupt_hub_download_degrades_to_none(self, tmp_path, monkeypatch):
        bad = tmp_path / "bad.safetensors"
        bad.write_bytes(b"corrupt-not-safetensors")
        monkeypatch.setattr("huggingface_hub.hf_hub_download", lambda *_a, **_k: str(bad))
        assert _load_checkpoint_state_dict(str(tmp_path)) is None


class TestSafetensorsUnavailable:
    def test_missing_safetensors_returns_none(self, tmp_path, monkeypatch):
        # Simulate safetensors not installed: the guarded import returns None.
        monkeypatch.setitem(__import__("sys").modules, "safetensors", None)
        monkeypatch.setitem(__import__("sys").modules, "safetensors.torch", None)
        assert _load_checkpoint_state_dict(str(tmp_path)) is None


def test_loader_is_referenced_by_recovery_path():
    """Guard that the recovery path still calls the loader unwrapped."""
    assert hasattr(processor_mod, "_load_checkpoint_state_dict")
