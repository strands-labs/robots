"""Regression: no ``strands_robots`` module may embed Unicode dash punctuation.

AGENTS.md mandates plain ASCII in user-facing strings and warns specifically
about orphan typographic glyphs left behind by automated authoring/sweeps. The
em dash (``U+2014``), en dash (``U+2013``), figure dash (``U+2012``), and
horizontal bar (``U+2015``) are an AI-authoring tell: they render
inconsistently across terminals and log pipelines, are tokenizer noise for the
agents that read these strings programmatically, and several appear inside
``logger`` / error / tool-result strings the ASCII rule covers. An ASCII hyphen
``-`` carries the same meaning everywhere.

This scan walks every Python module under the ``strands_robots`` package and
rejects any of those four dash codepoints. It is the dash analogue of
``test_source_strings_no_emoji`` and would have failed when 118 lines across 24
modules still carried ``U+2014``.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

# Figure dash, en dash, em dash, horizontal bar. ASCII hyphen-minus (U+002D) is
# the sanctioned replacement and is, of course, allowed.
_UNICODE_DASH = re.compile("[\u2012-\u2015]")

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_sources_discovered() -> None:
    """Guard: the scan actually walked the whole package, not one subtree."""
    sources = _python_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "drivers", "device_connect"} <= rel_dirs


def test_no_unicode_dashes_in_package_sources() -> None:
    """No ``strands_robots`` module may embed em/en/figure dashes or the horizontal bar."""
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _UNICODE_DASH.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_PACKAGE_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "Unicode dash punctuation found in strands_robots sources (use ASCII '-'):\n" + "\n".join(
        offenders
    )
