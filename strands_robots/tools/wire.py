"""What crosses the wire: JSON-safe results (images become PNG blocks) and decoded arguments."""

from __future__ import annotations

import inspect
import io
import json
from collections.abc import Callable, Mapping
from dataclasses import fields, is_dataclass
from typing import Any

import numpy as np

from ..types import JointState, Observation


def is_image(x: Any) -> bool:
    return isinstance(x, np.ndarray) and x.ndim == 3 and x.dtype == np.uint8


def json_safe(x: Any) -> Any:
    """numpy, dataclasses, an Observation -> plain JSON; an image -> its shape (the pixels go as a block)."""
    if isinstance(x, Observation):
        out = {"t": x.t, "joints": x.joints.as_dict(), "pose": json_safe(x.pose), "extras": json_safe(x.extras)}
        if x.joints.velocities is not None:
            out["velocities"] = dict(zip(x.joints.names, json_safe(x.joints.velocities), strict=True))
        out["images"] = {k: list(v.shape) for k, v in x.images.items()}
        return out
    if isinstance(x, JointState):
        return x.as_dict()
    if is_image(x):
        return {"image_shape": list(x.shape)}
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if is_dataclass(x) and not isinstance(x, type):
        return {f.name: json_safe(getattr(x, f.name)) for f in fields(x)}
    if isinstance(x, Mapping):
        return {str(k): json_safe(v) for k, v in x.items()}
    if isinstance(x, list | tuple):
        return [json_safe(v) for v in x]
    if isinstance(x, str | int | float | bool) or x is None:
        return x
    return str(x)


def images_of(x: Any) -> dict[str, np.ndarray]:
    if isinstance(x, Observation):
        return dict(x.images)
    if is_image(x):
        return {"image": x}
    return {}


def png_block(img: np.ndarray) -> dict[str, Any]:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(img)).save(buf, format="PNG")
    return {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}}


def content(result: Any) -> list[dict[str, Any]]:
    """A tool result's content blocks: the JSON text first, then one PNG block per image."""
    blocks: list[dict[str, Any]] = [{"text": json.dumps(json_safe(result))}]
    return blocks + [png_block(img) for img in images_of(result).values()]


def decode(fn: Callable[..., Any], args: Mapping[str, Any]) -> dict[str, Any]:
    """Refuse arguments ``fn`` does not take (a policy spec string stays a string: ``run`` resolves it)."""
    params = inspect.signature(fn).parameters
    unknown = sorted(k for k in args if k not in params)
    if unknown:
        raise TypeError(f"{getattr(fn, '__name__', fn)} does not take {unknown}; it takes {sorted(params)}")
    return dict(args)
