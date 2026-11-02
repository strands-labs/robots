"""The docs describe the library as it is: no emoji, and no build-planning references, in any page or template.

Scanned: ``README.md``, ``mkdocs.yml``, every text file under ``docs/`` and ``overrides/``.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TEXT = {".md", ".yml", ".yaml", ".html", ".js", ".css", ".py", ".svg", ".txt", ".json"}
# Emoji and pictograph blocks, the dingbats and misc-symbols blocks, and the emoji presentation selector.
EMOJI = re.compile("[\U0001f000-\U0001faff\u2600-\u27bf\u2b00-\u2bff\u231a\u231b\u23e9-\u23fa\ufe0f]")
PLANNING = re.compile(r"#4749|#3818|issues/4749|issues/3818|Where v1 is|status\.md")


def scanned() -> list[Path]:
    files = [ROOT / "README.md", ROOT / "mkdocs.yml"]
    for top in ("docs", "overrides"):
        files += sorted(p for p in (ROOT / top).rglob("*") if p.is_file() and p.suffix in TEXT)
    return files


@pytest.mark.parametrize("path", scanned(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_emoji(path: Path) -> None:
    hits = [
        (n, m.group())
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        for m in EMOJI.finditer(line)
    ]
    assert not hits, f"{path.relative_to(ROOT)}: emoji at lines {hits[:10]}"


@pytest.mark.parametrize("path", scanned(), ids=lambda p: str(p.relative_to(ROOT)))
def test_no_planning_references(path: Path) -> None:
    hits = [n for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1) if PLANNING.search(line)]
    assert not hits, f"{path.relative_to(ROOT)}: design/schedule references at lines {hits}"
