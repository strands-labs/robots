"""Behavioral tests for Mesh._encode_and_publish_frames.

_encode_and_publish_frames is the shared JPEG/raw encode + publish path used by
both the hardware (_publish_cameras_once) and simulation (_publish_sim_cameras_once)
camera loops. These tests pin its output contract on the mesh wire: which frames
are published vs skipped, how tensors and non-uint8 dtypes are normalized, the
cv2-unavailable raw fallback, encode failures, and per-frame error isolation.

Each frame is published on strands/<peer_id>/camera/<cam_name> with a payload
carrying the original shape, a uint8 dtype tag, an encoding tag, and base64 data.
"""

from __future__ import annotations

import base64
import sys
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest


@pytest.fixture
def mesh():
    """A Mesh over a duck-typed connected robot with one wrist camera."""
    from strands_robots.mesh import Mesh

    inner = SimpleNamespace(
        is_connected=True,
        name="so101_test",
        config=SimpleNamespace(cameras={"wrist": {"index": 0}}),
    )
    robot = SimpleNamespace(tool_name_str="so101", robot=inner)
    return Mesh(robot, peer_id="test-encode-1")


def _frame(h=4, w=4, c=3, dtype=np.uint8):
    return np.arange(h * w * c, dtype=dtype).reshape(h, w, c)


def test_publishes_jpeg_for_uint8_frame(mesh):
    """A well-formed uint8 HxWxC frame is JPEG-encoded and published once."""
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": _frame()}, ["wrist"])

    assert mock_put.call_count == 1
    topic, payload = mock_put.call_args[0]
    assert topic == "strands/test-encode-1/camera/wrist"
    assert payload["cam"] == "wrist"
    assert payload["shape"] == [4, 4, 3]
    assert payload["dtype"] == "uint8"
    assert payload["encoding"] == "jpeg"
    # data must be valid, non-empty base64.
    assert len(base64.b64decode(payload["data"])) > 0


def test_skips_none_frame(mesh):
    """A camera whose frame is None is skipped without publishing."""
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": None}, ["wrist"])
    assert not mock_put.called


def test_skips_frame_without_2d_shape(mesh):
    """A frame with fewer than 2 dimensions is not a valid image and is skipped."""
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": np.arange(5, dtype=np.uint8)}, ["wrist"])
    assert not mock_put.called


def test_casts_non_uint8_frame_before_encoding(mesh):
    """A float frame is cast to uint8 before encoding, still publishing jpeg."""
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": _frame(dtype=np.float32)}, ["wrist"])

    assert mock_put.call_count == 1
    _, payload = mock_put.call_args[0]
    assert payload["dtype"] == "uint8"
    assert payload["encoding"] == "jpeg"


def test_detaches_tensor_frame(mesh):
    """A torch tensor frame is detached to numpy (via .detach().cpu().numpy())."""
    torch = pytest.importorskip("torch")
    frame = torch.zeros((4, 4, 3), dtype=torch.uint8)
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": frame}, ["wrist"])

    assert mock_put.call_count == 1
    _, payload = mock_put.call_args[0]
    assert payload["shape"] == [4, 4, 3]
    assert payload["encoding"] == "jpeg"


def test_raw_fallback_when_cv2_unavailable(mesh, monkeypatch):
    """With cv2 unimportable, frames fall back to a raw base64 encoding.

    Setting sys.modules['cv2'] = None makes ``import cv2`` raise ImportError,
    exercising the have_cv2=False branch and the raw byte-encoding fallback.
    """
    monkeypatch.setitem(sys.modules, "cv2", None)
    with patch("strands_robots.mesh.core.put") as mock_put:
        mesh._encode_and_publish_frames({"wrist": _frame()}, ["wrist"])

    assert mock_put.call_count == 1
    _, payload = mock_put.call_args[0]
    assert payload["encoding"] == "raw"
    assert len(base64.b64decode(payload["data"])) > 0


def test_skips_when_cv2_imencode_fails(mesh):
    """A cv2 encode failure (ok=False) drops that frame without publishing."""
    with patch("cv2.imencode", return_value=(False, None)):
        with patch("strands_robots.mesh.core.put") as mock_put:
            mesh._encode_and_publish_frames({"wrist": _frame()}, ["wrist"])
    assert not mock_put.called


def test_publish_error_is_isolated_per_frame(mesh):
    """A publish failure on one frame is swallowed and does not abort the batch.

    put() raises for every frame; the method must neither raise nor stop early,
    and must still attempt each requested camera.
    """
    obs = {"wrist": _frame(), "front": _frame()}
    with patch("strands_robots.mesh.core.put", side_effect=RuntimeError("link down")) as mock_put:
        # Must not raise despite every publish failing.
        mesh._encode_and_publish_frames(obs, ["wrist", "front"])
    assert mock_put.call_count == 2
