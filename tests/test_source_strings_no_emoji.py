"""Regression: no package module, test module, or changelog fragment may embed emoji.

AGENTS.md forbids emoji in code, logs, and error messages: agents read these
strings programmatically, emoji are tokenizer noise, and they render
inconsistently (or as mojibake) across terminals and log pipelines. The
prohibition is package-wide, not specific to one subpackage - a stray glyph in
a tool banner, an RPC ``print`` log line, a registry docstring, or an inline
code comment is just as harmful as one in the simulation engine.

This scan walks every Python module under the ``strands_robots`` package and
rejects any pictograph / dingbat / symbol-emoji codepoint (plus orphan
``U+FE00-FE0F`` variation selectors). It deliberately does NOT require pure
ASCII: modules legitimately use math typography (``+/-``, multiplication sign,
base arrows) in comments and numeric output. Only emoji codepoints are
rejected.
"""

from __future__ import annotations

import re
from pathlib import Path

import strands_robots

# Emoji / pictograph / dingbat / symbol ranges plus variation selectors.
# Intentionally excludes the Mathematical Operators arrows (U+2190-21FF base
# arrows are allowed in comments) but DOES include emoji-presentation arrows
# such as U+25B6 (play) and the U+FE00-FE0F variation selectors that turn a
# plain glyph into an emoji. The regional-indicator (flag) block
# U+1F1E6-1F1FF is not listed separately: it already falls inside the
# U+1F000-1FAFF range above, and CodeQL flags the duplicate as overlapping.
_EMOJI = re.compile(
    "["
    "\U0001f000-\U0001faff"  # supplemental symbols, pictographs, emoticons, flags
    "\U00002600-\U000027bf"  # misc symbols + dingbats (includes U+2713 check mark)
    "\U00002300-\U000023ff"  # technical (stopwatch, hourglass, stop/play, etc.)
    "\U00002b00-\U00002bff"  # arrows/stars emoji block
    "\U000025a0-\U000025ff"  # geometric shapes (play/stop emoji bases)
    "\U0000fe00-\U0000fe0f"  # variation selectors (orphan emoji markers)
    "]"
)

_PACKAGE_DIR = Path(strands_robots.__file__).resolve().parent


def _python_sources() -> list[Path]:
    return sorted(p for p in _PACKAGE_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_package_sources_discovered() -> None:
    """Guard: the scan actually walked the whole package, not one subtree."""
    sources = _python_sources()
    # The package spans many subpackages; a healthy scan sees dozens of modules
    # across simulation, tools, registry, drivers, device_connect, mesh, etc.
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_PACKAGE_DIR).parts[0] for p in sources if p.parent != _PACKAGE_DIR}
    assert {"simulation", "tools", "registry", "drivers", "device_connect"} <= rel_dirs


def test_no_emoji_in_package_sources() -> None:
    """No ``strands_robots`` module may embed emoji codepoints or variation selectors."""
    offenders: list[str] = []
    for path in _python_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMOJI.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_PACKAGE_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "emoji found in strands_robots sources:\n" + "\n".join(offenders)


# The test tree itself is held to the same bar. A half-applied emoji sweep is
# easy to spot in production code but slips through in test files: prod stops
# emitting a glyph, yet an assertion (or a debug ``print``) still pins it. The
# package-only scan above cannot catch that, so scan ``tests/`` too. This guard
# would have failed when ``test_policy_runner.py`` still asserted ``"X Video:"``
# against output the engine is forbidden to emit.
_TESTS_DIR = Path(__file__).resolve().parent


def _test_sources() -> list[Path]:
    return sorted(p for p in _TESTS_DIR.rglob("*.py") if "__pycache__" not in p.parts)


def test_test_sources_discovered() -> None:
    """Guard: the scan walked the whole test tree, not just the top level."""
    sources = _test_sources()
    assert len(sources) > 50
    rel_dirs = {p.relative_to(_TESTS_DIR).parts[0] for p in sources if p.parent != _TESTS_DIR}
    assert {"simulation", "policies", "drivers"} <= rel_dirs


def test_no_emoji_in_test_sources() -> None:
    """No test module may embed emoji in assertions, fixtures, or debug prints."""
    offenders: list[str] = []
    for path in _test_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMOJI.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_TESTS_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "emoji found in test sources:\n" + "\n".join(offenders)


# Changelog news fragments are the third graded surface. ``changelog.d/*.md`` is
# folded *verbatim* into ``CHANGELOG.md`` by ``scripts/assemble_changelog.py
# --apply``, so a glyph in a fragment is a glyph in the released notes - and
# neither scan above can see it, because both walk ``*.py`` only. That is not
# hypothetical: the ``U+1F6A8`` marker this directory's ``2982-`` fragment
# describes as *stripped* was still sitting in the fragment prose, and would
# have folded into the log on the next release, behind a fully green run.
#
# Every ``*.md`` in the directory is scanned, with no reserved-name exemption.
# ``README.md`` documents the fragment convention and is the file a contributor
# copies a skeleton out of, so holding it to the same bar is the point rather
# than an oversight.
_FRAGMENT_DIR = Path(__file__).resolve().parents[1] / "changelog.d"


def _fragment_sources() -> list[Path]:
    return sorted(_FRAGMENT_DIR.glob("*.md"))


def test_fragment_dir_still_resolves() -> None:
    """Guard: the scan points at the fragment directory rather than at nothing.

    Deliberately *not* a file-count assertion, unlike the two guards above.
    ``assemble_changelog.py --apply`` ``unlink``s every fragment it consumes, so
    an empty ``changelog.d/`` is the legitimate state of the tree immediately
    after a release, and a ``len(...) > N`` guard would turn ``main`` red on the
    release commit itself. The failure actually worth catching is the scan
    walking a path the fragments have moved out of; the convention doc lives
    alongside them, so its presence is what proves the path still resolves.
    """
    assert _FRAGMENT_DIR.is_dir(), f"fragment directory is missing: {_FRAGMENT_DIR}"
    assert (_FRAGMENT_DIR / "README.md").is_file(), (
        f"{_FRAGMENT_DIR / 'README.md'} is absent - the fragment directory was moved or renamed "
        "and this scan is now walking the wrong path"
    )


def test_no_emoji_in_changelog_fragments() -> None:
    """No news fragment may embed emoji: its text is folded verbatim into the log."""
    offenders: list[str] = []
    for path in _fragment_sources():
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            for match in _EMOJI.finditer(line):
                cp = match.group()
                offenders.append(
                    f"{path.relative_to(_FRAGMENT_DIR.parent)}:{lineno}: U+{ord(cp[0]):04X} {line.strip()[:80]!r}"
                )
    assert not offenders, "emoji found in changelog fragments (folded verbatim into CHANGELOG.md):\n" + "\n".join(
        offenders
    )
