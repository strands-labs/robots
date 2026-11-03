"""Every docs page's front matter is a YAML mapping, so its description reaches the page head, not the body."""

from __future__ import annotations

from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")
DOCS = Path(__file__).resolve().parents[1] / "docs"


@pytest.mark.parametrize("page", sorted(DOCS.glob("*.md")), ids=lambda p: p.name)
def test_front_matter_parses(page: Path) -> None:
    text = page.read_text(encoding="utf-8")
    if not text.startswith("---\n"):
        return
    meta = yaml.safe_load(text[4 : text.index("\n---\n", 4)])
    assert isinstance(meta, dict), f"{page.name}: front matter is not a mapping"
    assert isinstance(meta.get("description", ""), str), f"{page.name}: description is not a string"
