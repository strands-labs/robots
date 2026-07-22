# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Media utility contracts: encode_clip round-trip + mjpeg_frames framing."""

import numpy as np
import pytest

from strands_robots.rendering import encode_clip, mjpeg_frames


def _frames(n: int = 5, w: int = 32, h: int = 24) -> list:
    rng = np.random.default_rng(0)
    return [rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8) for _ in range(n)]


def test_encode_clip_gif_round_trip(tmp_path) -> None:
    imageio = pytest.importorskip("imageio.v2")
    out = tmp_path / "clip.gif"
    result = encode_clip(_frames(), out, fps=10)
    assert result == out
    assert out.exists() and out.stat().st_size > 0
    decoded = imageio.mimread(out)
    assert len(decoded) == 5


def test_encode_clip_mp4_round_trip(tmp_path) -> None:
    pytest.importorskip("imageio.v2")
    pytest.importorskip("imageio_ffmpeg")
    out = tmp_path / "clip.mp4"
    encode_clip(_frames(n=8), out, fps=10)
    assert out.exists() and out.stat().st_size > 0


def test_encode_clip_rejects_empty_frames(tmp_path) -> None:
    pytest.importorskip("imageio.v2")
    with pytest.raises(ValueError, match="no frames"):
        encode_clip([], tmp_path / "empty.mp4")


def test_encode_clip_creates_parent_dirs(tmp_path) -> None:
    pytest.importorskip("imageio.v2")
    out = tmp_path / "nested" / "dir" / "clip.gif"
    encode_clip(_frames(n=2), out, fps=5)
    assert out.exists()


def test_mjpeg_frames_yields_multipart_jpeg_chunks() -> None:
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    chunks = list(mjpeg_frames(lambda: frame, fps=1000.0, max_frames=3))
    assert len(chunks) == 3
    for chunk in chunks:
        assert chunk.startswith(b"--frame\r\nContent-Type: image/jpeg\r\n\r\n")
        assert chunk.endswith(b"\r\n")
        assert b"\xff\xd8" in chunk  # JPEG SOI marker


def test_mjpeg_frames_resizes_to_requested_size() -> None:
    import io

    from PIL import Image

    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    (chunk,) = list(mjpeg_frames(lambda: frame, fps=1000.0, size=(16, 12), max_frames=1))
    jpeg = chunk.split(b"\r\n\r\n", 1)[1].rstrip(b"\r\n")
    with Image.open(io.BytesIO(jpeg)) as img:
        size = img.size
    assert size == (16, 12)


def test_mjpeg_frames_strict_reraises_frame_errors() -> None:
    def boom() -> np.ndarray:
        raise RuntimeError("render exploded")

    gen = mjpeg_frames(boom, fps=1000.0, max_frames=1, strict=True)
    with pytest.raises(RuntimeError, match="render exploded"):
        next(gen)


def test_mjpeg_frames_non_strict_skips_bad_frames() -> None:
    calls = {"n": 0}

    def flaky() -> np.ndarray:
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("first frame failed")
        return np.zeros((4, 4, 3), dtype=np.uint8)

    chunks = list(mjpeg_frames(flaky, fps=1000.0, max_frames=1))
    assert len(chunks) == 1  # stream survived the bad frame
    assert calls["n"] >= 2
