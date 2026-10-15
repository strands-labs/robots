"""The 1.0 acceptance suite (one test per check, real backends, no doubles) and the helpers its checks share.

A check that needs something this machine does not have (a GPU, hardware, credentials, a part that has
not landed yet) is skipped with the reason named, so ``pytest tests/acceptance -rs`` is the checklist.
Check 9 (a physical SO-101) is ``tests_integ/acceptance/test_feetech_so101.py``: it needs the arm on the bus.
"""

from __future__ import annotations

import importlib.util
import os
import socket
from pathlib import Path

import pytest

HUB = Path(os.environ.get("HF_HUB_CACHE", Path.home() / ".cache" / "huggingface" / "hub"))


def needs_module(name: str, why: str) -> None:
    """Skip unless ``name`` imports; ``why`` names the part that ships it."""
    try:
        found = importlib.util.find_spec(name) is not None
    except ModuleNotFoundError:
        found = False
    if not found:
        pytest.skip(f"needs {name} ({why})")


def cached_checkpoint(repo: str) -> str:
    """The snapshot directory of a Hub checkpoint already in the local cache, else skip."""
    snaps = HUB / f"models--{repo.replace('/', '--')}" / "snapshots"
    for snap in sorted(snaps.glob("*")) if snaps.is_dir() else ():
        if (snap / "model.safetensors").exists() and (snap / "config.json").exists():
            return str(snap)
    pytest.skip(f"needs the {repo} checkpoint in the Hugging Face cache (offline run)")


def free_endpoint() -> str:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return f"tcp/127.0.0.1:{s.getsockname()[1]}"
