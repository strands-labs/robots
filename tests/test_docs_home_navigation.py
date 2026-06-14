"""Repo hygiene: keep the docs home page's left navigation enabled.

The MkDocs Material site enables the global navigation sidebar via
``theme.features`` (``navigation.sections``/``navigation.expand``) in
``mkdocs.yml``. A page can opt out of that sidebar with a YAML front matter
block::

    ---
    hide:
      - navigation
    ---

When the home page (``docs/index.md``) carries ``hide: navigation`` the landing
page renders with no left navigation, so a first-time visitor lands on the site
with no visible way to browse the rest of the docs. The sidebar is present in
the built HTML but flagged ``hidden``; every other page shows it.

This guard parses the home page front matter and fails fast if ``navigation``
is ever re-added to its ``hide`` list. Other pages may legitimately hide
navigation; this guard is scoped to the landing page only.
"""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HOME_PAGE = REPO_ROOT / "docs" / "index.md"


def _hidden_items(md_path: Path) -> list[str]:
    """Return the ``hide:`` list from a Markdown file's YAML front matter.

    Parses the leading ``---`` fenced block without a YAML dependency: it reads
    the ``hide:`` key and collects its ``- item`` list entries. Returns an empty
    list when the file has no front matter or no ``hide`` key.
    """
    text = md_path.read_text(encoding="utf-8")
    if not text.startswith("---"):
        return []
    end = text.find("\n---", 3)
    if end == -1:
        return []
    items: list[str] = []
    in_hide = False
    for raw in text[3:end].splitlines():
        line = raw.rstrip()
        if not line:
            continue
        if not line.startswith((" ", "\t", "-")):
            # A new top-level key ends any open ``hide:`` block.
            in_hide = line.split(":", 1)[0].strip() == "hide"
            continue
        if in_hide and line.lstrip().startswith("-"):
            items.append(line.lstrip()[1:].strip())
    return items


def test_home_page_does_not_hide_navigation() -> None:
    """docs/index.md must not hide the left navigation sidebar."""
    assert HOME_PAGE.exists(), f"missing docs home page: {HOME_PAGE}"
    assert "navigation" not in _hidden_items(HOME_PAGE), (
        "docs/index.md hides the left navigation via front matter "
        "(hide: [navigation]); the home page would render with no sidebar to "
        "browse the rest of the docs. Remove 'navigation' from its hide list."
    )
