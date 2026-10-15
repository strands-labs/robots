"""Notebook helpers: inline MP4 video from frames, the design drawings, small tables. Not part of the package."""

from __future__ import annotations

import base64
import io
from pathlib import Path

import numpy as np
from IPython.display import HTML, Markdown, display

ROOT = Path(__file__).resolve().parents[1]  # the package checkout this folder sits in
DRAWINGS = ROOT / "docs" / "assets" / "drawings"
MEDIA = Path(__file__).resolve().parent / "media"


def video(frames, fps: int = 30, width: int = 640) -> HTML:
    """Encode HxWx3 uint8 frames to an H.264 MP4 and show it inline (base64, plays in Jupyter and VS Code)."""
    import av

    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    h, w = frames[0].shape[:2]
    h, w = h - h % 2, w - w % 2
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("h264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        s.options = {"crf": "23", "preset": "veryfast"}
        for f in frames:
            for p in s.encode(av.VideoFrame.from_ndarray(f[:h, :w, :3], format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return HTML(
        f'<video width="{width}" controls autoplay loop muted playsinline>'
        f'<source src="data:video/mp4;base64,{b64}" type="video/mp4"></video>'
    )


def drawing(name: str, theme: str = "paper", width: str = "100%") -> HTML:
    """One of the design drawings (animated SVG, inline so the animation runs)."""
    svg = (DRAWINGS / f"{name}.{theme}.svg").read_text()
    return HTML(f'<div style="max-width:{width}">{svg}</div>')


def media(path: str, width: int = 640) -> HTML:
    """A GIF/PNG from notebooks/media/, inline."""
    p = MEDIA / path
    mime = {"gif": "image/gif", "png": "image/png"}[p.suffix[1:]]
    return HTML(f'<img width="{width}" src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}">')


def table(rows: list[dict]) -> Markdown:
    keys = list(rows[0])
    lines = ["| " + " | ".join(keys) + " |", "|" + "---|" * len(keys)]
    lines += ["| " + " | ".join(str(r[k]) for k in keys) + " |" for r in rows]
    return Markdown("\n".join(lines))


def show(x) -> None:
    display(x)


def image(*arrays, width: int | None = None) -> HTML:
    """One or more HxWx3 uint8 arrays side by side, inline PNG."""
    from PIL import Image

    tags = []
    for a in arrays:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(a, dtype=np.uint8)).save(buf, format="PNG")
        w = f' width="{width}"' if width else ""
        tags.append(f'<img{w} style="margin-right:6px" src="data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}">')
    return HTML("".join(tags))
