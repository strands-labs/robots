"""Declarative-path canonicalization of bare camera frames.

Regression for the declarative ``embodiment`` path of ``LerobotLocalPolicy``.
There, the camera rename (``front`` -> ``observation.images.front``) runs
INSIDE the preprocessor pipeline, so the observation handed to
``_canonicalize_obs_images`` still carries the bare source key. The helper used
to key purely off the ``"image"`` substring, so a bare ``front`` frame was
skipped and reached the normalizer as raw HWC uint8 -- which overflows
(``value cannot be converted to type uint8 without overflow``) or mismatches
shapes. A single-camera SO-101 ACT checkpoint driven via the declarative path
therefore could not run at all; only the legacy (no-embodiment) path worked.

These pin the fix: ``_embodiment_image_source_keys`` surfaces the embodiment's
image rename sources, and ``_canonicalize_obs_images`` canonicalizes those bare
keys (while leaving the legacy substring-only behavior intact when no source
keys are supplied).
"""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import torch

from strands_robots.policies.lerobot_local.embodiment import EmbodimentMap
from strands_robots.policies.lerobot_local.policy import LerobotLocalPolicy


class _VisualFeature:
    """Minimal stand-in for a declared VISUAL ``PolicyFeature``."""

    class _T:
        name = "VISUAL"

    type = _T()


def _policy():
    with patch.object(LerobotLocalPolicy, "_load_model"):
        return LerobotLocalPolicy(pretrained_name_or_path="test/model")


def test_embodiment_image_source_keys_lists_only_image_renames():
    """Only obs_rename sources targeting a declared image feature are returned."""
    p = _policy()
    p._input_features = {
        "observation.images.front": _VisualFeature(),
        "observation.state": object(),
    }
    p._embodiment = EmbodimentMap(
        name="so101_front_only",
        obs_rename={"front": "observation.images.front"},
        state_keys=["1", "2", "3", "4", "5", "6"],
    )
    assert p._embodiment_image_source_keys() == {"front"}


def test_embodiment_image_source_keys_empty_without_embodiment():
    """No embodiment configured -> no extra source keys (legacy behavior)."""
    p = _policy()
    assert p._embodiment_image_source_keys() == set()


def test_bare_camera_key_canonicalized_with_source_keys():
    """A bare ``front`` HWC uint8 frame is converted to CHW float32 [0,1]."""
    p = _policy()
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    out = p._canonicalize_obs_images({"front": img}, image_source_keys={"front"})["front"]
    assert tuple(out.shape) == (3, 480, 640)  # HWC -> CHW
    assert out.dtype == torch.float32
    assert float(out.max()) <= 1.0 and float(out.min()) >= 0.0  # uint8 -> [0,1]


def test_bare_camera_key_untouched_without_source_keys():
    """Legacy path (no source keys): a bare camera key is left raw, unchanged."""
    p = _policy()
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    out = p._canonicalize_obs_images({"front": img})["front"]
    # Still the original raw HWC uint8 array (substring-only gate, unchanged).
    assert isinstance(out, np.ndarray)
    assert out.dtype == np.uint8
    assert out.shape == (480, 640, 3)


def test_declarative_branch_uses_source_keys_for_bare_camera():
    """End-to-end on get_actions' declarative branch: a bare camera key is
    canonicalized to CHW float32 BEFORE preprocess, not left as raw HWC uint8."""
    p = _policy()
    p._loaded = True
    p._device = "cpu"
    p._rtc_enabled = False
    p.actions_per_step = 1
    p.inference_kwargs = {}
    p._input_features = {
        "observation.images.front": _VisualFeature(),
        "observation.state": object(),
    }
    p._embodiment = EmbodimentMap(
        name="so101_front_only",
        obs_rename={"front": "observation.images.front"},
        state_keys=["1", "2", "3", "4", "5", "6"],
    )

    captured: dict[str, object] = {}

    class _Bridge:
        has_preprocessor = True
        has_postprocessor = False

        def preprocess(self, observation, instruction=""):
            captured["obs"] = observation
            return {"observation.state": torch.zeros(1, 6)}

    p._processor_bridge = _Bridge()

    class _Inner:
        def eval(self):
            return None

        def select_action(self, batch, **kw):
            return torch.zeros(1, 6)

    p._policy = _Inner()

    def _no_chunk():
        return False

    p._requires_action_chunk = _no_chunk
    p._tensor_to_action_dicts = lambda t: [{"ok": True}]

    obs = {"front": np.ones((64, 64, 3), dtype=np.uint8) * 255}
    p.get_actions_sync(obs, "pick up the cube")

    frame = captured["obs"]["front"]
    assert isinstance(frame, torch.Tensor)
    assert frame.dtype == torch.float32
    assert tuple(frame.shape) == (3, 64, 64)
    assert float(frame.max()) <= 1.0
