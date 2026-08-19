"""Repo hygiene: the docs never call a symbol future when the package ships it.

A page written while a symbol was still a plan keeps that sentence after the
symbol lands. The reader who stops at the sentence walks away believing a
capability is unavailable, and nothing in the toolchain notices: ruff, mypy and
the xref guards all check symbols the docs *name*, never claims about whether a
named symbol exists yet. ``docs/policies/wbc.md`` said layering an upper body on
WBC locomotion "is the job of a future ``CompositePolicy``, out of scope for this
provider" for two months after ``CompositePolicy`` landed - and the same page
documented it, with a runnable example and a rollout artifact, 260 lines further
down.

This guard grades the claim against the code: a sentence that says a symbol is
future, planned or out of scope must not name a symbol the package defines. The
package index is built with :mod:`ast` rather than by importing, so a symbol
behind an optional extra is still resolved.

Two deliberate narrowings, each measured against the current tree:

* ``is planned`` is not a marker. "The full collision-free trajectory is planned
  and cached on the first call" (``docs/policies/curobo.md``) is the domain verb,
  not a roadmap claim.
* ``future`` must not open a hyphenated compound. "the lookahead offsets for the
  future-reference window" (``docs/policies/protomotions.md``) describes a
  window, not a plan.

A claim about a name the package does not define is left alone: it is either
forward-looking prose that is still true, or it is about a third party. "any new
type a future lerobot or a plugin adds", naming lerobot's own
``RewardModelConfig``, is the live example and is graded clean.
"""

from __future__ import annotations

import ast
import re
import unicodedata
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = REPO_ROOT / "docs"
PACKAGE_DIR = REPO_ROOT / "strands_robots"

#: A sentence saying the thing it names does not exist yet.
_NOT_YET = re.compile(
    r"\b(?:a|an|the)\s+future(?![-\w])"
    r"|\bnot\s+yet\s+(?:implemented|supported|available|shipped|wired|exists?)\b"
    r"|\bout\s+of\s+scope\b"
    r"|\bwill\s+(?:be\s+)?(?:added|land|ship)\b"
    r"|\bdoes\s+not\s+(?:currently\s+)?(?:ship|support|exist)\b",
    re.IGNORECASE,
)

#: A backticked class-like name. Restricting to CamelCase keeps the sweep on
#: claims about a *symbol*: "Per-episode velocity variation in eval is out of
#: scope for this provider" names only snake_case functions and is not a claim
#: that any of them is missing.
_BACKTICKED_SYMBOL = re.compile(r"`([A-Z][A-Za-z0-9]*)`")

_FENCE = re.compile(r"^\s*```")
_ABBREVIATION = re.compile(r"\b(?:e\.g|i\.e|etc|vs|cf|approx)\.$", re.IGNORECASE)
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_ON_PAGE_LINK = re.compile(r"\[[^\]]*\]\(#([a-z0-9-]+)\)")


def _defined_symbols() -> dict[str, str]:
    """Map every public module-level class/function name to the file defining it.

    Read with :mod:`ast` so a symbol whose module needs an uninstalled extra is
    still indexed.
    """
    found: dict[str, str] = {}
    for path in sorted(PACKAGE_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except SyntaxError:  # pragma: no cover - a syntax error is another test's
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
                if not node.name.startswith("_"):
                    found.setdefault(node.name, str(path.relative_to(REPO_ROOT)))
    return found


def _prose_blocks(text: str) -> list[tuple[int, str]]:
    """Whitespace-normalized prose paragraphs, as ``(first line number, text)``.

    Fenced code is skipped; a blockquote marker is stripped so a claim inside a
    ``> **Note:**`` block is graded like any other prose.
    """
    blocks: list[tuple[int, str]] = []
    buffer: list[str] = []
    start = 0
    in_fence = False

    def flush() -> None:
        nonlocal buffer
        if buffer:
            blocks.append((start, " ".join(" ".join(buffer).split())))
            buffer = []

    for lineno, raw in enumerate(text.splitlines(), start=1):
        if _FENCE.match(raw):
            in_fence = not in_fence
            flush()
            continue
        if in_fence:
            continue
        line = re.sub(r"^>\s?", "", raw).strip()
        if line:
            if not buffer:
                start = lineno
            buffer.append(line)
        else:
            flush()
    flush()
    return blocks


def _sentences(paragraph: str) -> list[str]:
    """Split on sentence ends, keeping ``e.g.``-style abbreviations intact."""
    sentences: list[str] = []
    pending: list[str] = []
    for chunk in re.split(r"(?<=[.!?])\s+", paragraph):
        pending.append(chunk)
        if not _ABBREVIATION.search(chunk.strip()):
            sentences.append(" ".join(pending))
            pending = []
    if pending:
        sentences.append(" ".join(pending))
    return sentences


def _not_yet_claims(pages: dict[Path, str]) -> list[tuple[Path, int, list[str], str]]:
    """Every ``(page, line, backticked symbols, sentence)`` claiming "not yet"."""
    claims: list[tuple[Path, int, list[str], str]] = []
    for path, text in pages.items():
        for lineno, paragraph in _prose_blocks(text):
            for sentence in _sentences(paragraph):
                if not _NOT_YET.search(sentence):
                    continue
                symbols = sorted(set(_BACKTICKED_SYMBOL.findall(sentence)))
                if symbols:
                    claims.append((path, lineno, symbols, sentence))
    return claims


def _stale_claims(pages: dict[Path, str]) -> list[str]:
    """Report every not-yet claim that names a symbol the package defines."""
    defined = _defined_symbols()
    offenders: list[str] = []
    for path, lineno, symbols, sentence in _not_yet_claims(pages):
        try:
            shown = path.relative_to(REPO_ROOT)
        except ValueError:  # a synthetic page outside the repo
            shown = path
        for symbol in (s for s in symbols if s in defined):
            documented_here = pages[path].count(f"`{symbol}`") > 1
            offenders.append(
                f"{shown}:{lineno} calls `{symbol}` future or out of scope, "
                f"but {defined[symbol]} defines it"
                + (" and this same page documents it further down" if documented_here else "")
                + f" -- {sentence}"
            )
    return offenders


def _docs_pages() -> dict[Path, str]:
    return {p: p.read_text(encoding="utf-8") for p in sorted(DOCS_DIR.rglob("*.md"))}


def _heading_anchor(heading: str) -> str:
    """The slug python-markdown's ``toc`` extension derives for a heading.

    Reimplemented here rather than imported: the docs toolchain is not a test
    dependency, and this rule reproduces ``toc``'s slug for all of the headings
    under ``docs/``.
    """
    text = re.sub(r"`|\*\*|\*", "", heading)
    text = unicodedata.normalize("NFKD", text)
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[-\s]+", "-", text)


def _anchors(text: str) -> set[str]:
    return {_heading_anchor(m.group(2)) for line in text.splitlines() if (m := _HEADING.match(line))}


class TestNoDocsPageCallsAShippedSymbolFuture:
    """The claim "this does not exist yet" is graded against the package."""

    def test_no_not_yet_claim_names_a_symbol_the_package_defines(self) -> None:
        offenders = _stale_claims(_docs_pages())
        assert not offenders, "docs claim a shipped symbol does not exist yet:\n" + "\n".join(offenders)

    def test_a_claim_about_a_name_the_package_does_not_define_is_left_alone(self, tmp_path: Path) -> None:
        """A sentence naming both kinds of symbol reports only the shipped one.

        Forward-looking prose about a third party's type is still true: the docs
        say "any new type a future lerobot or a plugin adds" of lerobot's own
        ``RewardModelConfig``. Fails if the sweep is widened to report every
        "not yet" sentence rather than the ones naming a symbol this package
        ships.
        """
        defined = _defined_symbols()
        shipped = "CompositePolicy"
        unshipped = "NoSuchSymbolThePackageDefines"
        assert shipped in defined, f"premise: the package no longer defines {shipped}"
        assert unshipped not in defined, f"premise: the package now defines {unshipped}"

        page = tmp_path / "mixed.md"
        page.write_text(
            f"# Mixed\n\nThat is the job of a future `{shipped}`, and of a future `{unshipped}`.\n",
            encoding="utf-8",
        )
        offenders = _stale_claims({page: page.read_text(encoding="utf-8")})
        assert len(offenders) == 1, f"expected only the shipped symbol reported, got {offenders}"
        assert f"calls `{shipped}`" in offenders[0]
        assert f"calls `{unshipped}`" not in offenders[0]

    def test_the_sweep_reaches_the_docs_tree(self) -> None:
        """A clean result must mean the docs are right, not that nothing was read."""
        pages = _docs_pages()
        assert len(pages) > 50, f"only {len(pages)} docs pages were read"
        assert DOCS_DIR / "policies" / "wbc.md" in pages
        assert len(_defined_symbols()) > 100, "the package symbol index is suspiciously small"
        assert _not_yet_claims(pages), "premise: no docs sentence was graded at all"

    def test_a_planted_claim_about_a_shipped_symbol_is_reported(self, tmp_path: Path) -> None:
        """The grader reports the shape it exists to catch."""
        shipped = "CompositePolicy"
        assert shipped in _defined_symbols(), f"premise: the package no longer defines {shipped}"
        page = tmp_path / "planted.md"
        page.write_text(
            f"# Planted\n\nDoing that is the job of a future `{shipped}`, out of scope here.\n",
            encoding="utf-8",
        )
        offenders = _stale_claims({page: page.read_text(encoding="utf-8")})
        assert len(offenders) == 1 and shipped in offenders[0], (
            f"a planted claim about `{shipped}` was not reported: {offenders}"
        )

    def test_a_hyphenated_compound_and_the_planning_verb_are_not_claims(self, tmp_path: Path) -> None:
        """The two measured narrowings, pinned so neither is widened back."""
        page = tmp_path / "narrow.md"
        page.write_text(
            "# Narrow\n\n"
            "`Alpha` reads the lookahead offsets for the future-reference window.\n\n"
            "The trajectory is planned and cached, then `Beta` streams it.\n",
            encoding="utf-8",
        )
        pages = {page: page.read_text(encoding="utf-8")}
        assert _not_yet_claims(pages) == []
        assert _stale_claims(pages) == []


class TestTheWBCPagePointsAtTheCompositeSection:
    """The intro sends a reader to the section that shows how to compose."""

    @pytest.fixture
    def page(self) -> str:
        return (DOCS_DIR / "policies" / "wbc.md").read_text(encoding="utf-8")

    def test_the_first_composite_mention_links_to_a_heading_on_the_page(self, page: str) -> None:
        intro = next(p for _, p in _prose_blocks(page) if "CompositePolicy" in p)
        targets = _ON_PAGE_LINK.findall(intro)
        assert targets, f"the intro names CompositePolicy without linking the section: {intro}"
        anchors = _anchors(page)
        dangling = [t for t in targets if t not in anchors]
        assert not dangling, f"intro links {dangling}, which is not a heading on the page"

    def test_the_composite_section_it_points_at_shows_the_call(self, page: str) -> None:
        """The linked section is the one carrying the runnable composite."""
        assert "## Composing an upper body" in page
        assert "CompositePolicy(" in page
