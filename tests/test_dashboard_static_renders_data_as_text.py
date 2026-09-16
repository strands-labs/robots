"""The dashboard's pages build their nodes and hand each one its text as text.

A view renders strings a route served - a registry description, a mesh peer's
name, a refusal - and a peer-supplied string is attacker-controlled in this
repo's threat model. Interpolating one into markup would put script in the
dashboard's own origin, which is behind every guard `access.py` builds: it
carries the session cookie, `origin_is_self` passes by construction, and
`POST /api/settings` is then reachable (`runtime.trust_remote_code` is exported
to the process as `STRANDS_TRUST_REMOTE_CODE`, `security.auth_token` is
overwritable). So the rule is read off the shipped asset rather than left to a
reviewer noticing: nothing under `dashboard/static/` writes markup, and no
template literal in it mixes a tag with an interpolation.

The scan is per line, which is what these hand-written files are; a sink written
across a line break would be reported by neither cell, and by nothing else
either, so the rule belongs beside the files it grades rather than in a linter
that does not exist here.
"""

from __future__ import annotations

import pathlib
import re

STATIC = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "static"

# Vendored third-party code is not ours to rewrite; NOTICE names what it is.
SCRIPTS = sorted(p for p in STATIC.rglob("*.js") if "vendor" not in p.parts)

# Every DOM property and method that parses its argument as HTML.
MARKUP_SINKS = re.compile(r"\.(innerHTML|outerHTML)\s*=|\.insertAdjacentHTML\(|document\.write(ln)?\(")

# A backticked string that opens a tag and also interpolates: markup built from data.
TAG_IN_TEMPLATE = re.compile(r"`[^`]*<[a-zA-Z/][^`]*\$\{|`[^`]*\$\{[^`]*<[a-zA-Z/]")


def _hits(pattern: re.Pattern[str]) -> list[str]:
    return [
        f"{path.name}:{n}: {line.strip()}"
        for path in SCRIPTS
        for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]


class TestTheStaticPagesRenderDataAsText:
    def test_the_scan_has_something_to_read(self) -> None:
        """Both cells below pass vacuously on an empty roster."""
        assert [p.name for p in SCRIPTS] == ["app.js"]
        assert "createElement" in (STATIC / "app.js").read_text(encoding="utf-8")

    def test_no_script_writes_markup(self) -> None:
        assert _hits(MARKUP_SINKS) == []

    def test_no_template_literal_mixes_a_tag_with_an_interpolation(self) -> None:
        assert _hits(TAG_IN_TEMPLATE) == []
