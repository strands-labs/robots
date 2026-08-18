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
from collections.abc import Callable, Iterable, Iterator, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from strands_robots.utils import (
    finite_number_error,
    positive_whole_number_error,
    require_optional,
    sequence_length,
)

logger = logging.getLogger(__name__)


# libx264 quality bounds. ``imageio-ffmpeg`` enforces them with a bare
# ``assert 1 <= quality <= 10`` in its writer, and ``python -O`` strips
# assertions: on an optimized interpreter an out-of-range quality is encoded at
# whatever the arithmetic produces rather than refused, a non-numeric one leaks
# a raw ``TypeError``/``ValueError`` out of the bitrate computation, and a value
# above the range surfaces only as "the encoder wrote no clip". The lower bound
# is 1, not the 0 this module and imageio's own plugin documentation both used
# to advertise as the lowest quality.
_MIN_CLIP_QUALITY = 1
_MAX_CLIP_QUALITY = 10


def _clip_quality_error(quality: Any) -> str | None:
    """Error text when ``quality`` is outside the range the clip encoder honors.

    Only a finite number in ``[_MIN_CLIP_QUALITY, _MAX_CLIP_QUALITY]`` reaches
    the encoder as the requested quality. ``bool`` is rejected explicitly - an
    ``int`` subclass whose ``True`` acted as a silent quality of 1, the lowest
    the encoder offers, which is the same substitution
    :func:`_jpeg_quality_error` rejects on the streaming side.

    The numeric half is delegated to
    :func:`~strands_robots.utils.finite_number_error` rather than hand-rolled,
    so an integer too large to convert to a float reports the shared
    64-bit-float reason instead of raising ``OverflowError`` out of the
    ``float()`` this guard would otherwise apply itself. Only the interval is
    decided here.

    A fractional quality such as ``2.7`` is honorable and accepted: the encoder
    maps this knob onto a bitrate by arithmetic rather than indexing a table of
    discrete levels. That is the one way this domain is wider than
    :func:`_jpeg_quality_error`'s, where Pillow's JPEG quality is a whole-number
    scale.

    Args:
        quality: The caller-supplied clip quality.

    Returns:
        An error message, or ``None`` when the quality is usable.
    """
    if text := finite_number_error(quality, "quality", "encode_clip"):
        return text
    if not _MIN_CLIP_QUALITY <= float(quality) <= _MAX_CLIP_QUALITY:
        return f"encode_clip: quality must be between {_MIN_CLIP_QUALITY} and {_MAX_CLIP_QUALITY}, got {quality!r}."
    return None


def encode_clip(
    frames: Iterable[np.ndarray],
    path: str | Path,
    fps: int = 20,
    quality: float = 8,
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
        fps: playback frame rate, a positive whole number of frames per
            second (the shared media domain - see
            :func:`~strands_robots.utils.positive_whole_number_error`).
        quality: encoder quality knob, a finite number in ``[1, 10]`` (higher
            is better - see :func:`_clip_quality_error`). Read only by the MP4
            writer, since the GIF writer takes a per-frame duration instead,
            but the domain applies to both containers so that one call does not
            become valid by changing the output extension. A NumPy real is
            accepted and converted for the writer.
        macro_block_size: libx264 macro-block rounding; ``1`` preserves the
            exact frame size (the recorders' convention), ``8``/``16`` lets
            ffmpeg pad to codec-friendly sizes. Ignored for GIF.

    Returns:
        The output path.

    Raises:
        ImportError: if ``imageio`` (and, for MP4, ``imageio-ffmpeg``) is not
            installed.
        ValueError: if ``frames`` is empty, if ``fps`` is not a positive whole
            number, or if ``quality`` is not a finite number in ``[1, 10]`` --
            each would silently produce a corrupt, wrongly-timed or
            wrongly-encoded artifact rather than the requested clip.
        RuntimeError: if the encoder wrote no clip despite accepting the
            frames (for example a ``macro_block_size`` that rounds the frame
            size to dimensions libx264 refuses), so the returned path always
            names a clip that exists.
    """
    # Guard the caller's parameters before probing the optional encoder, so
    # the same mistake reports identically whether or not imageio is installed.
    if text := positive_whole_number_error(fps, "fps", "encode_clip"):
        raise ValueError(text)
    if text := _clip_quality_error(quality):
        raise ValueError(text)
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
        imageio.mimsave(str(out), frame_list, duration=1000.0 / int(fps))
    else:
        # ``float(quality)`` is load-bearing rather than cosmetic: the ffmpeg
        # writer gates the knob on ``isinstance(quality, (float, int))``, which
        # a NumPy scalar such as ``np.int64(8)`` or ``np.float32(8.0)`` fails
        # even though it names a quality this function has already accepted.
        writer = imageio.get_writer(str(out), fps=int(fps), quality=float(quality), macro_block_size=macro_block_size)
        try:
            for frame in frame_list:
                writer.append_data(frame)
        finally:
            writer.close()
    # The ffmpeg writer reports a refused encode on its own stderr and closes
    # without writing anything, so without this check the caller is handed a
    # path to a clip that does not exist.
    if not out.exists() or out.stat().st_size == 0:
        raise RuntimeError(
            f"encode_clip: the encoder wrote no clip to {out} for {len(frame_list)} "
            f"frames of shape {frame_list[0].shape} at {int(fps)}fps (quality={quality}, "
            f"macro_block_size={macro_block_size}); see the encoder output above for the "
            "refusal reason."
        )
    return out


# JPEG quality bounds. Pillow silently substitutes for anything outside them
# (a quality above 100 encodes as 100, a quality below 1 encodes as 1), so an
# out-of-range request produces a stream at a quality nobody asked for.
_MIN_JPEG_QUALITY = 1
_MAX_JPEG_QUALITY = 95


def _stream_rate_error(fps: Any) -> str | None:
    """Error text when ``fps`` is not a rate the stream can pace itself to.

    The pacing loop sleeps ``1 / fps - elapsed`` between chunks, so only a
    positive finite rate can be honored. ``0`` and any negative value both fell
    into a ``frame_dt = 0.0`` fallback that disables pacing entirely; ``nan``
    took the same branch (``nan > 0`` is ``False``); and ``inf`` reached it
    through the front door, since ``1 / inf`` is also ``0.0``. In every case the
    generator emitted as fast as it could encode - the opposite of a rate limit.

    Deliberately wider than :func:`encode_clip`'s whole-number domain (see
    :func:`~strands_robots.utils.positive_whole_number_error`): a container
    header stores an integer frame rate, but pacing a live view is just a sleep
    interval, so a fractional rate such as ``12.5`` is honorable here.

    The "is this a number at all" half is delegated to
    :func:`~strands_robots.utils.finite_number_error`, which every rate and
    quality guard in this module shares, so an integer too large to convert to a
    float reports rather than raising ``OverflowError`` out of the ``float()``
    below - out of a function whose whole contract is to return the message.

    Args:
        fps: The caller-supplied frame rate.

    Returns:
        An error message, or ``None`` when the rate is usable.
    """
    message = f"mjpeg_frames: fps must be a positive finite number, got {fps!r}."
    if finite_number_error(fps, "fps", "mjpeg_frames"):
        return message
    if float(fps) <= 0:
        return message
    return None


def _jpeg_quality_error(quality: Any) -> str | None:
    """Error text when ``quality`` is outside the JPEG quality Pillow honors.

    Only a whole number in ``[_MIN_JPEG_QUALITY, _MAX_JPEG_QUALITY]`` reaches the
    encoder unchanged; Pillow clamps anything else, so ``quality=500`` encoded
    identically to ``100`` and ``quality=0`` or ``-5`` identically to ``1``.
    ``bool`` is rejected explicitly - an ``int`` subclass whose ``True`` would
    act as a silent quality of 1.

    Shares the numeric half with :func:`_stream_rate_error` and
    :func:`_clip_quality_error` via
    :func:`~strands_robots.utils.finite_number_error`, so only the whole-number
    scale and the bounds are decided here.

    Args:
        quality: The caller-supplied JPEG quality.

    Returns:
        An error message, or ``None`` when the quality is usable.
    """
    message = (
        f"mjpeg_frames: quality must be a whole number between {_MIN_JPEG_QUALITY} "
        f"and {_MAX_JPEG_QUALITY}, got {quality!r}."
    )
    if finite_number_error(quality, "quality", "mjpeg_frames"):
        return message
    numeric = float(quality)
    if numeric != int(numeric):
        return message
    if not _MIN_JPEG_QUALITY <= int(numeric) <= _MAX_JPEG_QUALITY:
        return message
    return None


def _frame_size_error(size: Any) -> str | None:
    """Error text when ``size`` is not a resizable ``(width, height)`` pair.

    ``Image.resize`` raises straight out of the suspended generator for a
    malformed pair (``(0, 0)`` and ``(-4, 10)`` as ``ValueError``, a 1-tuple or a
    bare ``int`` as ``TypeError``), which surfaces mid-response rather than at
    the call. Validate the pair up front and reuse the shared pixel-count domain
    per component so ``size`` and the recorders' ``width``/``height`` cannot
    diverge.

    Args:
        size: The caller-supplied ``(width, height)`` pair, or ``None``.

    Returns:
        An error message, or ``None`` when the pair is usable.
    """
    if size is None:
        return None
    message = f"mjpeg_frames: size must be a (width, height) pair of positive whole numbers, got {size!r}."
    if isinstance(size, str | bytes | Mapping) or not hasattr(size, "__getitem__"):
        return message
    if sequence_length(size) != 2:
        return message
    for index, label in ((0, "size width"), (1, "size height")):
        if text := positive_whole_number_error(size[index], label, "mjpeg_frames"):
            return text
    return None


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
        fps: target frame rate; the generator sleeps to pace emission. A
            positive finite number of frames per second (see
            :func:`_stream_rate_error`).
        quality: JPEG quality, a whole number in ``[1, 95]``.
        size: optional ``(width, height)`` of positive whole numbers to resize
            frames to before encoding (keeps the stream resolution stable while
            the source camera changes).
        max_frames: stop after this many emitted frames, a positive whole number
            (``None`` = stream forever until the consumer disconnects). Mostly
            for tests.
        strict: when ``True``, exceptions raised by ``frame_fn`` propagate
            and terminate the stream. When ``False`` (default, the live-demo
            posture) a failed frame is logged at DEBUG and skipped so one bad
            render doesn't kill a long-lived stream.

    Returns:
        An iterator of ``multipart/x-mixed-replace`` byte chunks.

    Raises:
        ValueError: if ``fps``, ``quality``, ``size`` or ``max_frames`` names a
            stream this generator cannot produce. Raised by this call, before
            the first chunk exists, so a caller that has already written the
            ``multipart`` response headers never has to abort mid-body.
    """
    # Validate here rather than in the generator body: a generator function
    # runs nothing until the consumer's first ``next()``, by which point an
    # HTTP handler has committed the response headers and can only truncate
    # the stream. The body lives in ``_mjpeg_stream`` so these raise at the
    # call site instead.
    for text in (
        _stream_rate_error(fps),
        _jpeg_quality_error(quality),
        _frame_size_error(size),
        None if max_frames is None else positive_whole_number_error(max_frames, "max_frames", "mjpeg_frames"),
    ):
        if text:
            raise ValueError(text)
    return _mjpeg_stream(
        frame_fn,
        float(fps),
        int(quality),
        None if size is None else (int(size[0]), int(size[1])),
        None if max_frames is None else int(max_frames),
        strict,
    )


def _mjpeg_stream(
    frame_fn: Callable[[], np.ndarray | None],
    fps: float,
    quality: int,
    size: tuple[int, int] | None,
    max_frames: int | None,
    strict: bool,
) -> Iterator[bytes]:
    """Emit the MJPEG chunks for an already-validated stream configuration.

    Separated from :func:`mjpeg_frames` only so that function can reject an
    unusable configuration at call time; every argument here is normalised and
    in range.

    Args:
        frame_fn: returns the current ``(H, W, 3)`` RGB frame, or ``None``.
        fps: validated positive finite pacing rate.
        quality: validated JPEG quality in ``[1, 95]``.
        size: validated ``(width, height)`` pair, or ``None``.
        max_frames: validated positive frame budget, or ``None`` for unbounded.
        strict: whether a ``frame_fn`` failure terminates the stream.

    Yields:
        ``multipart/x-mixed-replace`` byte chunks, one per emitted frame.
    """
    from PIL import Image

    boundary = b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
    frame_dt = 1.0 / fps
    emitted = 0
    try:
        while max_frames is None or emitted < max_frames:
            # ``time.monotonic()``: the sleep below is computed from this base,
            # so it decides how long the client waits for the next frame. On
            # ``time.time()`` a wall-clock step landing between the two
            # readings - an NTP correction, a ``date -s``, a resume from
            # suspend - changed that wait by the size of the step: forward the
            # pacing was skipped, backward the frame was held for
            # ``frame_dt + step``.
            frame_start_mono = time.monotonic()
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
            elapsed = time.monotonic() - frame_start_mono
            if elapsed < frame_dt:
                time.sleep(frame_dt - elapsed)
    except GeneratorExit:  # client disconnected
        return
