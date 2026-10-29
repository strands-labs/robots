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
    """display() in a notebook; in a plain script print the text form (Markdown source, or a one-line note)."""
    if in_notebook():
        display(x)
    elif isinstance(x, Markdown):
        print(x.data)
    else:
        print(f"[{type(x).__name__} shown in the notebook]")


def image(*arrays, width: int | None = None) -> HTML:
    """One or more HxWx3 uint8 arrays side by side, inline PNG."""
    from PIL import Image

    tags = []
    for a in arrays:
        buf = io.BytesIO()
        Image.fromarray(np.asarray(a, dtype=np.uint8)).save(buf, format="PNG")
        w = f' width="{width}"' if width else ""
        b64 = base64.b64encode(buf.getvalue()).decode()
        tags.append(f'<img{w} style="margin-right:6px" src="data:image/png;base64,{b64}">')
    return HTML("".join(tags))


def quiet_lerobot() -> None:
    """Silence LeRobot's dataset stack on stderr (AV1 encoder info, per-episode progress bars, ffmpeg colour notices).
    Errors still print. Call before importing lerobot / recording."""
    import os
    import warnings

    os.environ.setdefault("SVT_LOG", "1")
    os.environ.setdefault("HF_DATASETS_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings("ignore", "Cannot enable progress bars")
    try:
        import av

        av.logging.set_libav_level(av.logging.ERROR)
    except ImportError:
        pass  # PyAV absent: no libav logger to quiet


def gif(frames, name: str, *, width: int = 360, fps: int = 25, colors: int | None = None) -> Path:
    """HxWx3 frames -> notebooks/media/<name> (e.g. "05/wave.gif"), scaled to ``width`` px; prints and returns the path.
    ``colors=64`` quantises (a GIF over 3 MB usually needs it, or fewer frames)."""
    from PIL import Image

    frames = [np.asarray(f, dtype=np.uint8) for f in frames]
    h = round(frames[0].shape[0] * width / frames[0].shape[1])
    imgs = [Image.fromarray(f).resize((width, h)) for f in frames]
    if colors:
        imgs = [im.quantize(colors) for im in imgs]
    path = MEDIA / name
    path.parent.mkdir(parents=True, exist_ok=True)
    imgs[0].save(path, save_all=True, append_images=imgs[1:], duration=round(1000 / fps), loop=0)
    print(f"media/{name}: {len(imgs)} frames, {path.stat().st_size / 1e6:.2f} MB")
    return path


def quiet_import(module: str):
    """Import ``module`` with fd 2 closed. LeRobot's import chain prints known-harmless macOS lines to the C-level
    stderr (torch's "Redirects are not supported", an ObjC notice that opencv and av both bundle libavdevice)."""
    import contextlib
    import importlib
    import os
    import sys

    sys.stderr.flush()
    saved, null = os.dup(2), os.open(os.devnull, os.O_WRONLY)
    os.dup2(null, 2)
    try:
        with contextlib.redirect_stderr(io.StringIO()):
            return importlib.import_module(module)
    finally:
        os.dup2(saved, 2)
        os.close(null)
        os.close(saved)


def tool_calls(agent, since: int = 0, width: int = 120) -> list[str]:
    """The tool calls in ``agent.messages[since:]``, one per line: tool.rpc(args) -> ok | the error the model read."""
    import json

    results = {
        b["toolResult"]["toolUseId"]: b["toolResult"] for m in agent.messages for b in m["content"] if "toolResult" in b
    }
    lines = []
    for m in agent.messages[since:]:
        for b in m["content"]:
            if "toolUse" not in b:
                continue
            use = b["toolUse"]
            args = use["input"]
            rpc = args.get("rpc") if isinstance(args, dict) else None
            res = results.get(use["toolUseId"], {})
            note = "ok" if res.get("status") == "success" else (res.get("content") or [{}])[0].get("text", "")[:width]
            shown = json.dumps(args.get(rpc) or {}) if rpc else json.dumps(args)
            lines.append(f"{use['name']}{'.' + rpc if rpc else ''}({shown}) -> {note}")
    return lines


def in_notebook() -> bool:
    try:
        from IPython import get_ipython

        return get_ipython() is not None and "IPKernelApp" in get_ipython().config
    except ImportError:
        return False


# --- builder: notebooks/scripts/NN_x.py (``# %%`` cells) -> notebooks/NN_x.ipynb ---------------------------------
# A line ending in ``# nb: script-only`` is dropped from the notebook (e.g. the sys.path line a script needs to
# import this module). ``# %% [markdown]`` cells are ``# ``-prefixed lines.

SCRIPT_ONLY = "# nb: script-only"


def cells_of(script: str) -> list[tuple[str, str]]:
    cells: list[tuple[str, list[str]]] = []
    for line in script.splitlines():
        if line.rstrip().endswith(SCRIPT_ONLY):
            continue
        if line.startswith("# %%"):
            cells.append(("markdown" if "[markdown]" in line else "code", []))
            continue
        if not cells:
            continue  # shebang / module docstring before the first cell
        kind, body = cells[-1]
        if kind == "markdown":
            line = line[2:] if line.startswith("# ") else line[1:] if line.startswith("#") else line
        body.append(line)
    out = []
    for kind, body in cells:
        while body and not body[-1].strip():
            body.pop()
        while body and not body[0].strip():
            body.pop(0)
        if body:
            out.append((kind, "\n".join(body)))
    return out


def build(script_path: str | Path, notebook_path: str | Path | None = None) -> Path:
    """Write the .ipynb for a paired script (outputs empty; execute it with nbconvert afterwards)."""
    import nbformat

    script_path = Path(script_path)
    nb_path = Path(notebook_path) if notebook_path else script_path.parents[1] / (script_path.stem + ".ipynb")
    nb = nbformat.v4.new_notebook()
    nb.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    nb.cells = [
        nbformat.v4.new_markdown_cell(src) if kind == "markdown" else nbformat.v4.new_code_cell(src)
        for kind, src in cells_of(script_path.read_text())
    ]
    for i, c in enumerate(nb.cells):
        c["id"] = f"{script_path.stem[:2]}-{i:02d}"  # stable ids: rebuilding does not churn the diff
    nbformat.write(nb, nb_path)
    return nb_path


if __name__ == "__main__":  # python notebooks/_nb.py build notebooks/scripts/NN_x.py [...]
    import sys

    if len(sys.argv) < 3 or sys.argv[1] != "build":
        raise SystemExit("usage: python notebooks/_nb.py build notebooks/scripts/NN_x.py [...]")
    for s in sys.argv[2:]:
        print(build(s))
