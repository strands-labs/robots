# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared media utilities for render pipelines.

One MP4/GIF encoder (:func:`encode_clip`) and one MJPEG live-stream
generator (:func:`mjpeg_frames`), consolidating the several hand-rolled
``imageio`` writers that previously lived in the recording mixins and the
GS-demo examples (issue #1537).
"""

from __future__ import annotations

import io
import logging
import time
from collections.abc import Callable, Iterable, Iterator
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)


def encode_clip(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: int = 20,
    quality: int = 8,
    macro_block_size: int = 1,
) -> Path:
    """Encode RGB frames into a clip at ``path`` (``.mp4`` or ``.gif``).

    MP4 output streams frames through ``imageio``'s ffmpeg writer (libx264)
    so the whole clip never has to be materialised twice in memory; ``.gif``
    output uses Pillow's GIF writer (which takes per-frame duration rather
    than the libx264-only knobs).

    Args:
        frames: iterable of ``(H, W, 3) uint8`` RGB frames. All frames must
            share one shape.
        path: output file path; the extension selects the container
            (``.gif`` -> GIF, anything else -> MP4 via ffmpeg).
        fps: playback frame rate.
        quality: imageio/ffmpeg quality knob (0-10, higher is better).
            Ignored for GIF.
        macro_block_size: libx264 macro-block rounding; ``1`` preserves the
            exact frame size (the recorders' convention), ``8``/``16`` lets
            ffmpeg pad to codec-friendly sizes. Ignored for GIF.

    Returns:
        The output path.

    Raises:
        ImportError: if ``imageio`` (and, for MP4, ``imageio-ffmpeg``) is not
            installed.
        ValueError: if ``frames`` is empty -- an empty clip write would
            silently produce a corrupt/zero-length artifact.
    """
    require_optional(
        "imageio",
        pip_install="imageio imageio-ffmpeg",
        extra="sim-mujoco",
        purpose="video encoding (encode_clip)",
    )
    import imageio.v2 as imageio

    frame_list: list[Any] = [np.asarray(f) for f in frames]
    if not frame_list:
        raise ValueError(f"encode_clip: no frames to encode for {path}")
    out = Path(path)
    if out.parent and not out.parent.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".gif":
        # Pillow's GIF writer takes per-frame duration (ms), not fps.
        imageio.mimsave(str(out), frame_list, duration=1000.0 / max(1, int(fps)))
        return out
    writer = imageio.get_writer(str(out), fps=int(fps), quality=quality, macro_block_size=macro_block_size)
    try:
        for frame in frame_list:
            writer.append_data(frame)
    finally:
        writer.close()
    return out


def mjpeg_frames(
    frame_fn: Callable[[], np.ndarray | None],
    fps: float = 12.0,
    quality: int = 75,
    size: tuple[int, int] | None = None,
    max_frames: int | None = None,
    strict: bool = False,
) -> Iterator[bytes]:
    """Yield ``multipart/x-mixed-replace`` JPEG chunks for a live MJPEG stream.

    Wraps any "give me the current frame" callable into the HTTP byte
    framing that ``<img src=...>`` MJPEG streaming expects (the pattern the
    GS demo Gradio apps use for their live views).

    Args:
        frame_fn: returns the current ``(H, W, 3)`` RGB frame, or ``None``
            when no frame is available yet (the generator then idles briefly
            instead of emitting a stale chunk).
        fps: target frame rate; the generator sleeps to pace emission.
        quality: JPEG quality (1-95).
        size: optional ``(width, height)`` to resize frames to before
            encoding (keeps the stream resolution stable while the source
            camera changes).
        max_frames: stop after this many emitted frames (``None`` = stream
            forever until the consumer disconnects). Mostly for tests.
        strict: when ``True``, exceptions raised by ``frame_fn`` propagate
            and terminate the stream. When ``False`` (default, the live-demo
            posture) a failed frame is logged at DEBUG and skipped so one bad
            render doesn't kill a long-lived stream.
    """
    from PIL import Image

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    frame_dt = 1.0 / float(fps) if fps > 0 else 0.0
    emitted = 0
    try:
        while max_frames is None or emitted < max_frames:
            t0 = time.time()
            frame: np.ndarray | None
            try:
                frame = frame_fn()
            except Exception as exc:
                if strict:
                    raise
                logger.debug("mjpeg_frames: dropped frame (%s: %s)", type(exc).__name__, exc)
                frame = None
            if frame is not None:
                im = Image.fromarray(np.asarray(frame)[:, :, :3].astype(np.uint8))
                if size is not None and im.size != size:
                    im = im.resize(size)
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=quality)
                yield boundary + buf.getvalue() + b"\r\n"
                emitted += 1
            else:
                time.sleep(0.2)
            elapsed = time.time() - t0
            if frame_dt and elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except GeneratorExit:  # client disconnected
        return
