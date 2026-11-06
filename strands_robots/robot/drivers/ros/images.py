"""sensor_msgs/Image and CompressedImage -> HxWx3 uint8 RGB, the shape every strands camera returns."""

from __future__ import annotations

from typing import Any

import numpy as np

_CH = {"rgb8": 3, "bgr8": 3, "rgba8": 4, "bgra8": 4, "mono8": 1}


def to_rgb(msg: Any) -> np.ndarray | None:
    """None for an encoding we can't show (depth, bayer): a camera never crashes an observe."""
    if hasattr(msg, "format"):  # CompressedImage
        return _decode_compressed(bytes(msg.data))
    enc = str(getattr(msg, "encoding", "")).lower()
    ch = _CH.get(enc)
    if ch is None:
        return None
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if buf.size < h * step:
        return None
    img = buf[: h * step].reshape(h, step)[:, : w * ch].reshape(h, w, ch)
    if ch == 1:
        return np.repeat(img, 3, axis=2)
    if enc.startswith("bgr"):
        img = img[:, :, [2, 1, 0]] if ch == 3 else img[:, :, [2, 1, 0, 3]]
    return np.ascontiguousarray(img[:, :, :3])


def _decode_compressed(data: bytes) -> np.ndarray | None:
    try:
        import cv2

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        return None if img is None else img[:, :, ::-1].copy()
    except ImportError:
        pass
    try:
        import io

        from PIL import Image

        return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    except Exception:  # noqa: BLE001 - no decoder here: report no image, not an error
        return None
