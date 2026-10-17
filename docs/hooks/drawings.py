"""mkdocs hook: ``{{drawing:<name>}}`` in a page becomes the paper and dark versions of that animated design drawing (the palette toggle picks one)."""

from __future__ import annotations

import re

_DRAWING = re.compile(r"\{\{drawing:([a-z0-9_]+)\}\}")


def on_page_markdown(markdown, page, config, files):  # noqa: ANN001 - mkdocs hook signature
    depth = page.file.dest_uri.count("/")
    up = "../" * depth

    def picture(m: re.Match) -> str:
        n = m.group(1)
        return (
            f'<p class="sr-drawing"><img class="sr-on-paper" alt="{n}" src="{up}assets/drawings/{n}.paper.svg">'
            f'<img class="sr-on-slate" alt="{n}" src="{up}assets/drawings/{n}.dark.svg"></p>'
        )

    return _DRAWING.sub(picture, markdown)
