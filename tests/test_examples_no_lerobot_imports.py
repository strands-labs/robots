"""Ensure examples/ never imports from lerobot directly.

Examples are the first thing a new user reads. They must demonstrate the
strands_robots abstraction, not bypass it. Any ``from lerobot`` or
``import lerobot`` at the top-level of an example file is a documentation
failure: it teaches users to skip the SDK and wire lerobot manually.

Internal helper directories (like examples/lerobot/) are excluded — only
top-level example scripts are checked.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_EXAMPLES_DIR = _REPO_ROOT / "examples"

# Regex matching bare lerobot imports (top-level or inside functions).
# Matches: "from lerobot", "import lerobot"
_LEROBOT_IMPORT_RE = re.compile(r"^\s*(from\s+lerobot|import\s+lerobot)", re.MULTILINE)


def _example_scripts() -> list[Path]:
    """Collect top-level .py files in examples/ (not subdirectories)."""
    if not _EXAMPLES_DIR.is_dir():
        return []
    return sorted(_EXAMPLES_DIR.glob("*.py"))


@pytest.mark.parametrize(
    "script",
    _example_scripts(),
    ids=[p.name for p in _example_scripts()],
)
def test_no_direct_lerobot_import(script: Path):
    """Top-level example scripts must not import lerobot directly."""
    content = script.read_text(encoding="utf-8")
    matches = _LEROBOT_IMPORT_RE.findall(content)
    assert not matches, (
        f"{script.name} imports lerobot directly: {matches}. "
        f"Examples should use strands_robots.Robot / strands_robots.policies.create_policy "
        f"instead. Move lerobot usage behind the strands_robots abstraction."
    )
