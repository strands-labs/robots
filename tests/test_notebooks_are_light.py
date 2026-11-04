"""The notebooks are committed without outputs and their GIFs/screenshots live in the `media` release, not in git."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = sorted((ROOT / "notebooks").glob("*.ipynb"))
RELEASE = "https://github.com/strands-labs/robots/releases/download/media/"


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
def test_a_notebook_is_committed_without_outputs(path: Path) -> None:
    for cell in json.loads(path.read_text(encoding="utf-8"))["cells"]:
        if cell["cell_type"] == "code":
            assert cell["outputs"] == [] and cell["execution_count"] is None, (
                f"{path.name}: clear it first, `jupyter nbconvert --clear-output --inplace {path.name}`"
            )


def test_notebook_images_point_at_the_media_release() -> None:
    raster = re.compile(r"\]\((?!https?://)[^)]*\.(gif|png|webp)\)")
    for path in [*NOTEBOOKS, *(ROOT / "notebooks" / "scripts").glob("*.py"), ROOT / "notebooks" / "README.md"]:
        found = raster.findall(path.read_text(encoding="utf-8"))
        assert not found, f"{path.name}: a local GIF/PNG link; use {RELEASE}<dir>-<name>"


def test_raster_media_is_gitignored() -> None:
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for ext in ("gif", "png", "webp"):
        assert f"notebooks/media/**/*.{ext}" in gitignore
