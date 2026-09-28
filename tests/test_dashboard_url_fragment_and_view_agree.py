"""The address bar and the screen name the same tab, whichever way the tab was entered.

`show()` writes `location.hash` on every tab change, so the page fills the
browser's history with entries it owns. Only the initial load read that
fragment back: there was no `hashchange` listener, so a fragment that changed
on an open page - a bookmarked `#agent` pasted into the tab, a link, Back or
Forward through the entries `show()` itself wrote - moved the URL and left the
view where it was. Measured in a browser on the pre-fix files, Back from Fleet
to `#agent` showed Fleet under the address `#agent`; an operator reaching for
the Agent tab read the Fleet page and the highlighted tab agreed with the page,
not with the URL they asked for.

The dispatch that enters a view is one function, `route()`, and the three ways
in - a click, the loaded URL's fragment, a fragment change - all call it.

The UI is now the React SPA (`frontend/src/App.tsx`), where the same rule reads:
`route()` is the only writer of the panel state and of `location.hash`, the
mount effect routes the loaded fragment, the `hashchange` listener routes a
changed one (skipping the view already shown, so route()'s own write does not
re-enter it), and every click handler calls `route()` rather than the state
setter. These cells read `App.tsx` for that shape, so the rule holds without a
browser in the suite.
"""

from __future__ import annotations

import pathlib
import re

APP_TSX = pathlib.Path(__file__).parent.parent / "strands_robots" / "dashboard" / "frontend" / "src" / "App.tsx"


def _function_body(source: str, header: str) -> str:
    """The text from ``header`` to the end of its braced block, braces balanced."""
    start = source.index(header)
    depth = 0
    for i in range(source.index("{", start), len(source)):
        depth += {"{": 1, "}": -1}.get(source[i], 0)
        if depth == 0:
            return source[start : i + 1]
    raise AssertionError(f"unbalanced braces after {header!r}")


class TestTheFragmentAndTheViewAgree:
    def test_one_router_is_the_only_place_a_view_is_entered(self) -> None:
        """The panel state and the fragment are written in route() and nowhere else."""
        source = APP_TSX.read_text(encoding="utf-8")
        assert source.count("function route(") == 1, "no single router: each entry point carries its own dispatch"
        router = _function_body(source, "function route(")
        assert "setPanel(" in router and "location.hash = " in router, (
            "route() does not write both the view and the address"
        )
        outside = source.replace(router, "")
        assert not re.search(r"(?<!\[panel, )setPanel\(", outside), "the panel state is written outside route()"
        assert "location.hash = " not in outside, "the address is written outside route()"

    def test_a_fragment_change_routes_to_the_view_it_names(self) -> None:
        """A hashchange - bookmark, link, Back, Forward - enters the view the fragment names."""
        source = APP_TSX.read_text(encoding="utf-8")
        listener = _function_body(source, "const onHash = () =>")
        assert "panelFromHash(location.hash)" in listener, "the listener does not read the fragment"
        assert "route(" in listener, "the hashchange listener does not enter the view"
        assert "shownPanel.current !== next" in listener, (
            "the listener does not skip the view already shown, so route()'s own hash write re-enters it"
        )
        assert re.search(r"addEventListener\('hashchange', onHash\)", source), "no hashchange listener is installed"

    def test_the_loaded_fragment_and_every_click_share_that_router(self) -> None:
        """The mount routes the loaded URL; every handler that opens a panel calls route()."""
        source = APP_TSX.read_text(encoding="utf-8")
        assert "route(initialPanel())" in source, "the loaded URL's fragment does not go through the router"
        assert "panelFromHash(location.hash)" in _function_body(source, "function initialPanel()"), (
            "the loaded fragment is not read"
        )
        clicks = re.findall(r"on(?:Settings|Activity|Devices|Training|Record|Sim|Help)=\{[^}]*\}", source)
        assert len(clicks) >= 7, f"the tab handlers were not found ({len(clicks)})"
        for click in clicks:
            assert "route(" in click, f"a tab handler bypasses the router: {click}"
