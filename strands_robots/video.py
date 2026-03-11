"""Video encoding utilities for strands-robots.

Bridges LeRobot's video encoding (H264/HEVC/SVT-AV1 via PyAV) into
strands-robots recording pipelines. Falls back to imageio if PyAV
is not available.

Usage:
    from strands_robots.video import encode_frames, VideoEncoder

    # One-shot: encode list of numpy frames to MP4
    encode_frames(frames, "output.mp4", fps=30, codec="h264")

    # Streaming: encode frames one at a time
    with VideoEncoder("output.mp4", fps=30, codec="h264") as enc:
        for frame in frames:
            enc.add_frame(frame)
"""

import logging
import os
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

# Supported codecs in priority order
SUPPORTED_CODECS = ["h264", "hevc", "libsvtav1"]
DEFAULT_CODEC = "h264"


def _check_pyav() -> bool:
    """Check if PyAV is available."""
    try:
        import av  # noqa: F401

        return True
    except ImportError:
        return False


def _check_imageio() -> bool:
    """Check if imageio is available."""
    try:
        import imageio  # noqa: F401

        return True
    except ImportError:
        return False


class VideoEncoder:
    """Streaming video encoder with codec selection.

    Supports H264 (universal), HEVC (better compression), and
    SVT-AV1 (best quality/size ratio). Falls back to imageio if
    PyAV is not available.

    Usage:
        with VideoEncoder("output.mp4", fps=30, codec="h264") as enc:
            for frame in frames:
                enc.add_frame(frame)  # numpy HWC uint8
    """

    def __init__(
        self,
        output_path: str,
        fps: int = 30,
        codec: str = DEFAULT_CODEC,
        crf: int = 23,
        pix_fmt: str = "yuv420p",
    ):
        self.output_path = str(output_path)
        self.fps = fps
        self.codec = codec
        self.crf = crf
        self.pix_fmt = pix_fmt
        self._frame_count = 0
        self._writer = None
        self._backend = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def add_frame(self, frame: np.ndarray):
        """Add a frame (HWC, uint8, RGB or BGR).

        Lazily initializes the encoder on first frame to detect resolution.
        """
        if self._writer is None:
            self._init_writer(frame.shape[1], frame.shape[0])

        if self._backend == "pyav":
            self._add_frame_pyav(frame)
        elif self._backend == "imageio":
            self._add_frame_imageio(frame)

        self._frame_count += 1

    def close(self):
        """Finalize and close the video file."""
        if self._writer is None:
            return

        try:
            if self._backend == "pyav":
                # Flush
                for pkt in self._stream.encode():
                    self._container.mux(pkt)
                self._container.close()
            elif self._backend == "imageio":
                self._writer.close()
        except Exception as e:
            logger.warning("Error closing video: %s", e)

        self._writer = None
        logger.info(
            f"Video saved: {self.output_path} "
            f"({self._frame_count} frames, {self.codec}, {self._backend})"
        )

    def _init_writer(self, width: int, height: int):
        """Initialize the video writer with the best available backend."""
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)

        # Try PyAV first (better codec support)
        if _check_pyav():
            try:
                self._init_pyav(width, height)
                self._backend = "pyav"
                logger.info("Video encoder: PyAV (%s, %sx%s)", self.codec, width, height)
                return
            except Exception as e:
                logger.warning("PyAV init failed: %s, falling back to imageio", e)

        # Fallback to imageio
        if _check_imageio():
            self._init_imageio(width, height)
            self._backend = "imageio"
            logger.info("Video encoder: imageio (%sx%s)", width, height)
            return

        raise RuntimeError(
            "No video encoder available. Install: pip install av  OR  pip install imageio[ffmpeg]"
        )

    def _init_pyav(self, width: int, height: int):
        """Initialize PyAV encoder."""
        import av

        self._container = av.open(self.output_path, mode="w")
        self._stream = self._container.add_stream(self.codec, rate=self.fps)
        self._stream.width = width
        self._stream.height = height
        self._stream.pix_fmt = self.pix_fmt

        # Set codec options
        if self.crf is not None:
            self._stream.options["crf"] = str(self.crf)

        # Keyframe interval
        self._stream.options["g"] = str(min(self.fps * 2, 250))

        # SVT-AV1 specific
        if self.codec == "libsvtav1":
            self._stream.options["preset"] = "12"  # Fast

        self._writer = True  # Flag that writer is initialized

    def _init_imageio(self, width: int, height: int):
        """Initialize imageio encoder."""
        import imageio

        self._writer = imageio.get_writer(
            self.output_path,
            fps=self.fps,
            quality=8,
            macro_block_size=1,
        )

    def _add_frame_pyav(self, frame: np.ndarray):
        """Add frame using PyAV."""
        import av

        # Ensure RGB
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)

        av_frame = av.VideoFrame.from_ndarray(frame, format="rgb24")
        for pkt in self._stream.encode(av_frame):
            self._container.mux(pkt)

    def _add_frame_imageio(self, frame: np.ndarray):
        """Add frame using imageio."""
        self._writer.append_data(frame)

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def get_info(self) -> dict:
        return {
            "output_path": self.output_path,
            "fps": self.fps,
            "codec": self.codec,
            "backend": self._backend,
            "frame_count": self._frame_count,
        }


def encode_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 30,
    codec: str = DEFAULT_CODEC,
    crf: int = 23,
) -> dict:
    """One-shot: encode a list of numpy frames to video.

    Args:
        frames: List of HWC uint8 numpy arrays (RGB)
        output_path: Output video file path
        fps: Frames per second
        codec: Video codec ('h264', 'hevc', 'libsvtav1')
        crf: Constant rate factor (quality, lower = better)

    Returns:
        Dict with encoding stats
    """
    if not frames:
        return {"status": "error", "message": "No frames to encode"}

    with VideoEncoder(output_path, fps=fps, codec=codec, crf=crf) as enc:
        for frame in frames:
            enc.add_frame(frame)

    file_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0

    return {
        "status": "success",
        "output_path": output_path,
        "frames": len(frames),
        "fps": fps,
        "codec": codec,
        "backend": enc._backend,
        "file_size_kb": round(file_size / 1024, 1),
    }


def get_video_info(video_path: str) -> dict:
    """Get video file metadata.

    Returns:
        Dict with duration, fps, width, height, codec, frames
    """
    try:
        import av

        container = av.open(str(video_path))
        stream = container.streams.video[0]
        info = {
            "path": str(video_path),
            "duration_s": (
                float(stream.duration * stream.time_base) if stream.duration else 0
            ),
            "fps": float(stream.average_rate) if stream.average_rate else 0,
            "width": stream.width,
            "height": stream.height,
            "codec": stream.codec_context.name,
            "frames": stream.frames,
            "file_size_kb": round(os.path.getsize(video_path) / 1024, 1),
        }
        container.close()
        return info
    except ImportError:
        # Fallback: basic file info only (no av installed)
        info = {
            "path": str(video_path),
            "note": "Install av for full video metadata",
        }
        if os.path.exists(video_path):
            info["file_size_kb"] = round(os.path.getsize(video_path) / 1024, 1)
        else:
            info["error"] = f"File not found: {video_path}"
        return info
    except Exception as e:
        return {"path": str(video_path), "error": str(e)}
