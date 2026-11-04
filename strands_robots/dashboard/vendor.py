"""The browser engines the dashboard page loads, fetched on first use instead of shipped in the package.

``mujoco.wasm`` (the twin) and ``ort-wasm-simd-threaded.wasm`` (standalone ONNX skills) are 20 MB together, 95% of
the installed package. A robot that never opens the dashboard never downloads them. The first page load (or
``python -m strands_robots.dashboard --fetch``, for a LAN with no internet) downloads each file from the npm CDN at
the version ``static/vendor/NOTICE`` names, refuses it unless its sha256 matches, and keeps it in
``$STRANDS_ROBOTS_CACHE/dashboard`` (default ``~/.cache/strands_robots/dashboard``). The small JS glue stays in
``static/vendor/``.
"""

from __future__ import annotations

import hashlib
import os
import threading
import urllib.request
from pathlib import Path

from ..description.assets import cache_dir

#: name -> (url, sha256). The versions are twin_scene.BROWSER_MUJOCO and BROWSER_ORT; NOTICE carries the same hashes.
FILES: dict[str, tuple[str, str]] = {
    "mujoco.wasm": (
        "https://cdn.jsdelivr.net/npm/@mujoco/mujoco@3.14.0/mujoco.wasm",
        "2d380f1070a556c3003b4a7c7c10c98a5f61db4e91815de1075ec478e9487e76",
    ),
    "ort-wasm-simd-threaded.wasm": (
        "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort-wasm-simd-threaded.wasm",
        "207d02be4591c156b0a98f024f3d58005b5b04c92274d759fb390338c63559ea",
    ),
}
_LOCK = threading.Lock()


def directory() -> Path:
    """Where the fetched engines live."""
    return cache_dir() / "dashboard"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def path(name: str, *, timeout: float = 120.0) -> Path:
    """The cached file for ``name``, downloaded and checked first if it is missing or does not match its hash."""
    url, sha = FILES[name]
    dst = directory() / name
    with _LOCK:
        if dst.is_file() and _sha256(dst) == sha:
            return dst
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(f".{name}.{os.getpid()}.part")
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r, open(tmp, "wb") as f:  # noqa: S310 - fixed https
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            got = _sha256(tmp)
            if got != sha:
                raise OSError(f"{name} from {url} has sha256 {got}, expected {sha}: refusing it")
            tmp.replace(dst)
        except OSError as e:
            raise OSError(
                f"the dashboard could not fetch {name} ({e}). On a machine with internet run "
                f"`python -m strands_robots.dashboard --fetch`, then copy {directory()} to this one"
            ) from e
        finally:
            tmp.unlink(missing_ok=True)
        return dst


def fetch_all() -> list[Path]:
    """Download every engine now (for a robot LAN with no internet later)."""
    return [path(name) for name in FILES]
