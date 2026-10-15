"""The first python example on every docs page (and in the README) runs, as written, in a fresh process.

A page can say why its first example cannot run here with an HTML comment right above the fence:
``<!-- test: skip, needs CUDA -->`` skips it, ``<!-- test: network -->`` marks it as a network test.
``my_policy`` is the one placeholder the pages use; the test defines it as a policy that holds still.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
FENCE = re.compile(r"^(?:<!-- test: (?P<mark>[^>]*?) -->\n)?```py(?:thon)?[^\n]*\n(?P<code>.*?)^```", re.M | re.S)
OPTIONAL = ("mujoco", "onnxruntime", "huggingface_hub", "zenoh", "strands", "lerobot", "fastapi", "uvicorn")
PRELUDE = "def my_policy(obs):\n    return {}\n\n"


def first_examples() -> list:
    params = []
    for page in [ROOT / "README.md", *sorted(DOCS.glob("*.md"))]:
        m = FENCE.search(page.read_text())
        if m is None:
            continue
        mark = (m["mark"] or "").strip()
        marks = []
        if mark.startswith("skip"):
            marks.append(pytest.mark.skip(reason=f"{page.name}: {mark.partition(',')[2].strip() or 'skipped'}"))
        elif mark == "network":
            marks.append(pytest.mark.network)
        params.append(pytest.param(m["code"], id=page.stem, marks=marks))
    return params


def missing_optional(stderr: str) -> str | None:
    m = re.search(r"No module named '([\w.]+)'", stderr)
    return m[1] if m and m[1].split(".")[0] in OPTIONAL else None


@pytest.mark.parametrize("code", first_examples())
def test_first_example_on_the_page_runs(code: str, tmp_path: Path) -> None:
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    out = subprocess.run(
        [sys.executable, "-c", PRELUDE + code], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=240
    )
    if out.returncode and (dep := missing_optional(out.stderr)):
        pytest.skip(f"needs the optional dependency {dep}")
    assert out.returncode == 0, out.stderr[-2000:]
