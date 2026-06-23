"""Repo hygiene: the animated brand SVGs stay wired into the docs and README.

The site and README open with three hand-authored animated SVGs that carry the
project's visual identity (true-black glassmorphism, brand green ``#00FF77`` +
cyan ``#22D3EE``):

* ``hero_loop.svg`` - the perceive/reason/act/world control loop, on the docs
  home page and at the top of the README.
* ``architecture_flow.svg`` - the four-layer stack with action/observation
  signal flow, on the architecture page and the README "How it works" section.
* ``mesh_network.svg`` - peer coordination over the Zenoh mesh, on the mesh
  page and the README "Mesh networking" section.

Each is surfaced through the ``brand-figure``/``brand-svg`` CSS treatment in
``docs/stylesheets/extra.css``. This guard fails fast if an asset is deleted,
an embed is dropped, an SVG stops being valid (or static) XML, or the CSS hook
disappears - any of which would silently degrade the landing experience.
"""

from __future__ import annotations

import xml.dom.minidom
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = REPO_ROOT / "docs"
ASSETS = DOCS / "assets"
EXTRA_CSS = DOCS / "stylesheets" / "extra.css"
README = REPO_ROOT / "README.md"

BRAND_SVGS = ("hero_loop.svg", "architecture_flow.svg", "mesh_network.svg")


def test_brand_svg_assets_exist() -> None:
    """All three animated brand SVGs ship under docs/assets/."""
    for name in BRAND_SVGS:
        assert (ASSETS / name).is_file(), f"missing brand asset: docs/assets/{name}"


def test_brand_svgs_are_valid_animated_xml() -> None:
    """Each brand SVG parses as XML and carries SMIL animation (<animate*>)."""
    for name in BRAND_SVGS:
        text = (ASSETS / name).read_text(encoding="utf-8")
        # Raises on malformed XML - the docs build would otherwise serve a broken asset.
        xml.dom.minidom.parseString(text)
        assert "<animate" in text, f"{name} lost its SMIL animation"


def test_docs_pages_embed_their_brand_svg() -> None:
    """Home, architecture, and mesh pages each embed their SVG via the brand class."""
    pairs = {
        DOCS / "index.md": "hero_loop.svg",
        DOCS / "architecture.md": "architecture_flow.svg",
        DOCS / "mesh.md": "mesh_network.svg",
    }
    for page, asset in pairs.items():
        text = page.read_text(encoding="utf-8")
        assert asset in text, f"{page.name} no longer embeds {asset}"
        assert "brand-svg" in text, f"{page.name} dropped the brand-svg CSS hook"


def test_readme_embeds_all_brand_svgs() -> None:
    """The README surfaces all three animated brand SVGs.

    ``hero_loop.svg`` opens the README; ``architecture_flow.svg`` illustrates
    the "How it works" section and ``mesh_network.svg`` the "Mesh networking"
    section - matching the docs pages that embed the same assets. Each is
    referenced by its ``docs/assets/`` repo-relative path so GitHub renders it.
    """
    text = README.read_text(encoding="utf-8")
    for name in BRAND_SVGS:
        assert f"docs/assets/{name}" in text, f"README no longer embeds {name}"


def test_extra_css_defines_brand_figure() -> None:
    """The active stylesheet defines the brand-figure / brand-svg treatment."""
    css = EXTRA_CSS.read_text(encoding="utf-8")
    assert ".brand-figure" in css, "extra.css missing .brand-figure rule"
    assert ".brand-svg" in css or "img.brand-svg" in css, "extra.css missing .brand-svg rule"
