"""The docs site's look is one stylesheet, and it covers what the site renders.

``docs/stylesheets/extra.css`` styles Material's own custom properties rather
than overriding its rules, so the whole look fits in one budgeted file. MkDocs
does not validate ``extra_css`` / ``extra_javascript`` paths and does not know
that generated markup (the robot cards) or a phone-only layout (tables that
stack into cards) depend on it, so nothing but a grader ties the pieces
together. This module pins the joins that a silent edit would break:

* every declared stylesheet and script resolves on disk, and the look lives in
  one file under a stated line budget;
* the two colour schemes declare the same variables, so dark mode cannot lose a
  colour light mode has, and the accent is declared once per scheme;
* every element the robot-pages hook emits carries a styled class (the catalog
  cards, chips and thumbnails are generated markup, and the hook does not know
  which stylesheet covers them);
* the sidebar does not render every page of every section at once, and
  ``navigation.tabs`` is only on while the nav is short enough for the tab strip
  (the six-tab nav fits; 24 top-level sections needed a 2,565px strip in a
  1,216px bar at 1,280px wide, which clipped 13 of them).
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
MKDOCS_YML = REPO_ROOT / "mkdocs.yml"
DOCS_DIR = REPO_ROOT / "docs"
STYLESHEET = DOCS_DIR / "stylesheets" / "extra.css"
ROBOT_PAGES_HOOK = REPO_ROOT / "docs" / "hooks" / "robot_pages.py"

# The stylesheet carries the palette with GitHub code colours, the one-row
# header, the page head and pager, the landing hero and footer, the 3D viewer
# chrome, the robot cards and the filter chips in 335 lines; the budget leaves
# room to grow without room to drift back into a per-rule override sheet.
LINE_BUDGET = 360

# Material names the light scheme in mkdocs.yml; the dark one is its ``slate``.
LIGHT_SCHEME = "paper"
DARK_SCHEME = "slate"

# Material lays the tab strip out in one row and clips the overflow, so tabs
# only work for a nav short enough to fit.
MAX_SECTIONS_FOR_TABS = 8


def _block(key: str) -> list[str]:
    """The ``- item`` entries of a top-level or nested mkdocs.yml block.

    The items may sit at the key's indent or one level deeper; the first item
    fixes the item indent, and deeper lines (a mapping item's other keys) are
    skipped. A mapping item (``- path: x``) is reduced to its first value.
    """
    text = MKDOCS_YML.read_text(encoding="utf-8")
    match = re.search(rf"^(\s*){re.escape(key)}:\s*$", text, re.M)
    assert match, f"mkdocs.yml has no {key!r} block"
    key_indent = len(match.group(1))
    item_indent: int | None = None
    items: list[str] = []
    for line in text[match.end() :].splitlines()[1:]:
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= key_indent and not (indent == key_indent and line.strip().startswith("- ")):
            break
        if item_indent is None:
            item_indent = indent
        if indent != item_indent or not line.strip().startswith("- "):
            continue
        item = line.strip()[2:].strip()
        if re.match(r"^[a-z_]+:\s", item):
            item = item.split(":", 1)[1].strip()
        items.append(item)
    return items


def _declared_assets() -> list[str]:
    """Local (non-URL) paths declared in extra_css and extra_javascript."""
    declared = _block("extra_css") + _block("extra_javascript")
    return [entry.split("?", 1)[0] for entry in declared if not entry.startswith("http")]


def _scheme_variables(scheme: str) -> set[str]:
    """The custom properties declared in one colour scheme's block."""
    css = STYLESHEET.read_text(encoding="utf-8")
    match = re.search(rf'\[data-md-color-scheme="{scheme}"\]\s*\{{(.*?)\}}', css, re.S)
    assert match, f"extra.css declares no {scheme!r} colour scheme"
    return set(re.findall(r"(--[a-z0-9-]+):", match.group(1)))


def _hook_class_lists() -> set[tuple[str, ...]]:
    """Every ``class="..."`` list the robot-pages hook puts in a page.

    Covers the catalog cards, every generated robot page and every family page,
    so a chip that only appears on a per-robot page is checked too.
    """
    spec = importlib.util.spec_from_file_location("docs_robot_pages_hook", ROBOT_PAGES_HOOK)
    assert spec is not None and spec.loader is not None
    hook = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hook)
    html = [hook.cards(None, "")]
    html += [hook.robot_page(name) for name in hook.registry()]
    html += [hook.family_page(category) for category in hook._families_in_order()]
    return {tuple(value.split()) for chunk in html for value in re.findall(r'class="([^"]+)"', chunk)}


def test_every_declared_stylesheet_and_script_exists() -> None:
    """A stylesheet or script mkdocs never finds is styling nobody sees."""
    missing = sorted(path for path in _declared_assets() if not (DOCS_DIR / path).is_file())
    assert not missing, (
        f"mkdocs.yml declares extra_css/extra_javascript paths that do not "
        f"exist under docs/: {missing}. MkDocs emits the link tag anyway, so "
        f"the page silently loads nothing."
    )


def test_the_look_is_one_stylesheet_within_budget() -> None:
    """One local stylesheet, under the line budget."""
    sheets = [path for path in _declared_assets() if path.endswith(".css")]
    assert sheets == ["stylesheets/extra.css"], f"expected one stylesheet, got {sheets}"
    lines = STYLESHEET.read_text(encoding="utf-8").splitlines()
    assert len(lines) <= LINE_BUDGET, (
        f"{STYLESHEET.name} is {len(lines)} lines, over the {LINE_BUDGET}-line "
        f"budget. Style Material's custom properties instead of adding "
        f"per-element overrides."
    )


def test_both_colour_schemes_declare_the_same_variables() -> None:
    """A variable set in one scheme only leaves the other reading a stale value."""
    light, dark = _scheme_variables(LIGHT_SCHEME), _scheme_variables(DARK_SCHEME)
    assert light == dark, (
        f"colour schemes declare different variables; light only: "
        f"{sorted(light - dark)}, dark only: {sorted(dark - light)}"
    )


def test_the_accent_is_declared_once_per_scheme() -> None:
    """One accent per scheme, read through the variable everywhere else."""
    css = STYLESHEET.read_text(encoding="utf-8")
    declarations = re.findall(r"--sr-accent:\s*(#[0-9a-fA-F]{3,8});", css)
    assert len(declarations) == 2, f"expected one --sr-accent per scheme, found {declarations}"
    for accent in declarations:
        assert css.count(accent) == 1, (
            f"{accent} is spelled {css.count(accent)} times; a rule that wants "
            f"the accent reads var(--sr-accent) so the palette is a two-line change."
        )


def test_every_generated_card_class_is_styled() -> None:
    """The card component covers the markup the robot-pages hook emits.

    An element is styled when at least one class on it has a rule: a modifier
    such as ``sr-chip-sim`` rides on ``sr-chip`` and needs no rule of its own.
    """
    css = STYLESHEET.read_text(encoding="utf-8")
    unstyled = sorted(
        " ".join(classes) for classes in _hook_class_lists() if not any(f".{name}" in css for name in classes)
    )
    assert not unstyled, (
        f"docs/hooks/robot_pages.py emits elements with no styled class in "
        f"{STYLESHEET.name}: {unstyled}. It is the site's only stylesheet, so "
        f"the generated catalog would render unstyled."
    )


def test_motion_is_optional() -> None:
    """The stylesheet honours prefers-reduced-motion."""
    assert "prefers-reduced-motion" in STYLESHEET.read_text(encoding="utf-8")


def test_the_sidebar_does_not_render_every_page_at_once() -> None:
    """Neither navigation.sections nor navigation.expand: both keep every child open."""
    features = _block("features")
    walls = sorted({"navigation.sections", "navigation.expand"} & set(features))
    assert not walls, (
        f"theme.features enables {walls}, which renders every page of every "
        f"section in the sidebar at once (2,308px of nav on the home page)."
    )


def test_tabs_stay_off_until_the_nav_fits_the_strip() -> None:
    """navigation.tabs needs a nav short enough that no section is clipped."""
    sections = len(_block("nav"))
    if "navigation.tabs" in _block("features"):
        assert sections <= MAX_SECTIONS_FOR_TABS, (
            f"navigation.tabs is on with {sections} top-level nav sections; "
            f"Material clips the strip, hiding the overflow tabs entirely."
        )


def test_code_blocks_keep_their_copy_button() -> None:
    """Every fence is meant to be copied, not retyped."""
    assert "content.code.copy" in _block("features")
