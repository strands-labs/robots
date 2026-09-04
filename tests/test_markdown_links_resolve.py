"""Repo hygiene: a relative Markdown link names something inside this repository.

A Markdown link written with the wrong number of ``../`` segments still renders
as a link. GitHub, the docs site and an editor all show it as clickable prose,
and the only way to learn it is broken is to click it, so the failure surfaces
to a reader rather than to the author.

Nothing in this repository graded that. The one link checker here is
``mkdocs build --strict``, run by the ``build`` job in ``.github/workflows/docs.yml``,
and it is narrower than it looks in two ways:

* It resolves links only for files under ``docs_dir`` (``docs/``). A Markdown
  file anywhere else -- ``README.md``, ``AGENTS.md``, a ``changelog.d``
  fragment, a package reference page such as
  ``strands_robots/policies/moveit2/server/README.md`` -- is never read, so a
  broken link there is reported by nothing at all. That is where this guard
  found its first offender: a five-segment ``../../../../../`` prefix on a page
  four directories deep, which resolves outside the checkout entirely.
* That job is not among the branch ruleset's required status checks, so even a
  broken link *under* ``docs/`` cannot block a merge -- and once merged, the
  same job runs on ``main`` and its failure holds back the Pages deploy the
  ``deploy`` job performs.

So this guard lives in the test suite, which the required check runs, and it
grades every Markdown file in the tree rather than the ``docs/`` subset.

The rule is one sentence: a relative link target must resolve to a path that
exists and that lies inside the repository. The second half matters as much as
the first -- a target escaping the checkout may happen to exist beside it on one
machine and not on another, so existence alone is a verdict that depends on what
sits next to the clone.

Deliberately out of scope, each for a reason:

* ``#fragment`` resolution. Whether a heading anchor exists depends on the
  slugifier the site renders with, and MkDocs already owns that question
  (``validation.links.anchors``). Re-deriving its slug rules here would be a
  second implementation of somebody else's contract.
* Site-absolute targets (``/path``). MkDocs reads those as site-root-relative
  and GitHub as repository-root-relative, so they need a policy rather than a
  resolution; the tree ships none today.
* ``http``/``mailto`` and other schemed targets, which name nothing in the tree.

Markdown is not the only syntax a link is written in. GitHub and MkDocs both
render raw HTML embedded in a Markdown file, and this tree uses it where
Markdown has no equivalent - a ``<figure>``/``<figcaption>`` pair, an
``<img>`` carrying a width. Those targets are links to a reader, and reading
only Markdown syntax left them graded by nothing: the sweep reported a clean
tree while ``docs/policies/wbc.md`` carried two ``<a href>`` MP4 links with one
``../`` too many, five lines below a Markdown image whose target was correct and
graded.

The two syntaxes are not resolved the same way, which is why the wrong prefix is
easy to write and hard to notice:

* MkDocs **rewrites** a Markdown target, so it is authored relative to the
  source file and comes out relative to the rendered page. ``wbc.md`` ships
  ``../assets/wbc/g1_walk.gif`` and the built site serves
  ``../../assets/wbc/g1_walk.gif``.
* MkDocs **passes raw HTML through unchanged**, so whatever prefix is written is
  what both surfaces get - and the rendered page sits one directory deeper than
  the source file, so a target cannot be correct on both. An author picks a
  surface, and nothing records which.

So the rule this module applies to raw HTML is the one surface that is not
optional: GitHub serves the source file, keeps ``<a>`` and ``<img>``, and
resolves their targets against the source directory. A relative target in one of
those elements must resolve there. A target that only ever needs to work on the
published site belongs in an element GitHub does not render, or in an absolute
URL - which is what the fix for those two links used, matching the three
``https://github.com/strands-labs/robots/blob/main/...`` links already in the
same page.

Deliberately out of scope for the same reason it is in scope elsewhere:

* ``<video>``, ``<source>``, ``<figure>`` and ``<figcaption>``. GitHub's
  sanitizer drops them, so a target inside one reaches no reader of the source
  file and the site is its only consumer - where the site-relative spelling this
  tree uses is correct. ``docs/device-connect.md`` ships one such
  ``<source src>``, and grading it against the source tree would report a
  working embed as broken.
"""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent

# Areas that must contribute Markdown to the sweep. Kept as a floor rather than
# the whole list: the discovery below derives the areas, so a new top-level
# directory is picked up on arrival, and this only fails if a known one drops out.
_REQUIRED_AREAS = frozenset({"docs", "examples", "strands_robots"})

_FENCE = re.compile(r"^\s*(```+|~~~+)")
_INLINE_CODE = re.compile(r"`[^`\n]*`")
# An inline link or image: [text](target) / ![alt](target), with an optional
# <angle-bracketed> target and an optional "title".
_INLINE_LINK = re.compile(r"!?\[[^\]]*\]\(\s*<?([^)>\s]+)>?(?:\s+[\"'][^\"']*[\"'])?\s*\)")
# A reference definition: [label]: target
_REFERENCE_DEFINITION = re.compile(r"^\s{0,3}\[[^\]]+\]:\s*<?([^>\s]+)>?", re.MULTILINE)
_SCHEME = re.compile(r"^[A-Za-z][A-Za-z0-9+.-]*:")
# A link written as raw HTML. Only the elements GitHub renders are read, and
# only the attribute that carries a target for each: ``<a href>`` and
# ``<img src>``. The attribute must be preceded by whitespace so ``data-src``
# is not read as ``src``.
_HTML_LINK = re.compile(
    r"<(?P<tag>a|img)\b[^>]*?\s(?P<attr>href|src)\s*=\s*[\"'](?P<target>[^\"']+)[\"']",
    re.IGNORECASE,
)
# The element/attribute pairs that carry a target. Written as a pair rather than
# two independent lists so ``<img href>`` and ``<a src>`` - neither of which
# names anything - are not read as links.
_HTML_TARGET_ATTRIBUTES = frozenset({("a", "href"), ("img", "src")})


def _prose(text: str) -> str:
    """Return ``text`` with fenced blocks and inline code spans blanked out.

    Both hold link-shaped text that is not a link. A fenced block quotes a
    command or a program; an inline code span quotes a message such as
    ``Contact the maintainer on [Discord](...)``, whose ``...`` is a placeholder
    a reader is not meant to follow. Lines are blanked rather than dropped so a
    caller reporting a line number still reports the right one.

    Args:
        text: The full contents of a Markdown file.

    Returns:
        The same number of lines, with code content replaced by blanks.
    """
    kept: list[str] = []
    in_fence = False
    fence = ""
    for line in text.splitlines():
        opener = _FENCE.match(line)
        if opener:
            if not in_fence:
                in_fence, fence = True, opener.group(1)[0] * 3
            elif line.strip().startswith(fence):
                in_fence = False
            kept.append("")
            continue
        kept.append("" if in_fence else _INLINE_CODE.sub("", line))
    return "\n".join(kept)


def _link_targets(text: str) -> list[str]:
    """Return every link target in ``text``, ignoring code.

    Both syntaxes a reader clicks are read: Markdown inline links and reference
    definitions, and the raw-HTML elements GitHub renders (``<a href>`` and
    ``<img src>``). Reading only the Markdown half is what let two broken
    ``<a href>`` targets sit in a page the sweep reported as clean.

    Args:
        text: The full contents of a Markdown file.

    Returns:
        Markdown inline-link, reference-definition and raw-HTML targets.
    """
    prose = _prose(text)
    targets = [match.group(1) for match in _INLINE_LINK.finditer(prose)]
    targets.extend(match.group(1) for match in _REFERENCE_DEFINITION.finditer(prose))
    targets.extend(
        match.group("target")
        for match in _HTML_LINK.finditer(prose)
        if (match.group("tag").lower(), match.group("attr").lower()) in _HTML_TARGET_ATTRIBUTES
    )
    return targets


def _unresolved_targets(text: str, source: Path, root: Path) -> list[tuple[str, str]]:
    """Return the relative link targets in ``text`` that name nothing usable.

    Args:
        text: The full contents of the Markdown file.
        source: The file's path, which relative targets resolve against.
        root: The tree the target must stay inside. The sweep passes the
            repository root; an exemplar passes a constructed tree, which is
            what lets the rule be graded without a real offender in the tree.

    Returns:
        ``(target, reason)`` pairs, in the order the targets appear. ``reason``
        is ``"escapes the repository"`` when the target resolves outside
        ``root`` and ``"does not exist"`` when it resolves inside but names no
        file or directory.
    """
    unresolved: list[tuple[str, str]] = []
    for target in _link_targets(text):
        if target.startswith("#") or target.startswith("/") or _SCHEME.match(target):
            continue
        head = unquote(target.split("#", 1)[0].split("?", 1)[0])
        if not head:
            continue
        resolved = (source.parent / head).resolve()
        if not resolved.is_relative_to(root.resolve()):
            unresolved.append((target, "escapes the repository"))
        elif not resolved.exists():
            unresolved.append((target, "does not exist"))
    return unresolved


def _markdown_files() -> list[Path]:
    """Return every Markdown file in the tree.

    Derived rather than listed so a new top-level directory that ships Markdown
    is graded on arrival. Dot-directories are skipped, which keeps ``.git`` and
    any local virtualenv out.

    Returns:
        Sorted repository-relative Markdown paths.
    """
    found = set(_REPO_ROOT.glob("*.md"))
    for entry in _REPO_ROOT.iterdir():
        if entry.is_dir() and not entry.name.startswith("."):
            found.update(entry.rglob("*.md"))
    return sorted(found)


class TestEveryRelativeLinkNamesSomethingInTheRepository:
    """The tree's own Markdown, graded by the check that gates a merge."""

    def test_no_markdown_file_links_to_a_path_it_cannot_reach(self) -> None:
        """Every relative link target resolves inside the repository."""
        offenders: list[str] = []
        for path in _markdown_files():
            text = path.read_text(encoding="utf-8")
            for target, reason in _unresolved_targets(text, path, _REPO_ROOT):
                offenders.append(f"{path.relative_to(_REPO_ROOT)} -> {target} ({reason})")
        assert not offenders, "Markdown links that name nothing reachable:\n  " + "\n  ".join(offenders)

    def test_the_sweep_reads_every_area_that_ships_markdown(self) -> None:
        """The derived discovery reaches each area known to carry Markdown."""
        reached = set()
        for path in _markdown_files():
            parts = path.relative_to(_REPO_ROOT).parts
            reached.add(parts[0] if len(parts) > 1 else "<root>")
        assert _REQUIRED_AREAS <= reached, f"sweep reached only {sorted(reached)}"
        assert "<root>" in reached, "the sweep did not reach the repository's own top-level Markdown"

    def test_the_sweep_reads_more_than_the_docs_directory(self) -> None:
        """The offenders this guard exists for live outside ``docs/``.

        MkDocs already resolves the links under ``docs/``. The reason this guard
        walks the whole tree is that a Markdown file anywhere else is read by no
        link checker at all, so a sweep narrowed to ``docs/`` would add nothing.
        """
        outside = [p for p in _markdown_files() if p.relative_to(_REPO_ROOT).parts[0] != "docs"]
        assert outside, "the sweep found no Markdown outside docs/, so it grades nothing mkdocs does not"

    def test_the_sweep_reads_link_targets_at_all(self) -> None:
        """The tree really carries relative links, so a clean sweep means something.

        Without this, a discovery that read the files but found no targets -- a
        broken pattern, say -- would report the same clean result as a tree whose
        links all resolve.
        """
        relative = [
            target
            for path in _markdown_files()
            for target in _link_targets(path.read_text(encoding="utf-8"))
            if not (target.startswith("#") or target.startswith("/") or _SCHEME.match(target))
        ]
        assert len(relative) > 100, f"only {len(relative)} relative targets found; the pattern reads too little"

    def test_the_sweep_reads_raw_html_targets_in_this_tree(self) -> None:
        """The tree really carries raw-HTML links, so a clean sweep means something.

        The Markdown half is exercised by hundreds of targets, so a raw-HTML
        pattern that matched nothing at all would leave the count above healthy
        and the sweep silent on the syntax it was extended for. ``README.md``
        ships three ``<img src="docs/assets/...svg">`` embeds, and they resolve -
        so this asserts they are read, not merely that nothing was reported.
        """
        read = [
            (path.relative_to(_REPO_ROOT), match.group("target"))
            for path in _markdown_files()
            for match in _HTML_LINK.finditer(_prose(path.read_text(encoding="utf-8")))
            if (match.group("tag").lower(), match.group("attr").lower()) in _HTML_TARGET_ATTRIBUTES
            and not (match.group("target").startswith(("#", "/")) or _SCHEME.match(match.group("target")))
        ]
        assert read, "the raw-HTML pattern read no relative target in a tree that ships several"
        assert all((_REPO_ROOT / source).parent.joinpath(target).resolve().exists() for source, target in read), (
            f"raw-HTML targets that do not resolve: {read}"
        )


class TestTheRuleSeesTheLinksItMustSee:
    """Constructed exemplars, because the tree carries no offender to grade.

    Once the tree is clean the sweep above cannot exercise its own failing
    branch, so a weakened rule would keep reporting a pass. These drive the same
    predicate over a tree built for the purpose: inputs that must be reported,
    and inputs that must not.
    """

    @staticmethod
    def _tree(tmp_path: Path) -> tuple[Path, Path]:
        """Build a small tree and return its root and the page to link from.

        Shaped like the offender this guard was written for: a page nested two
        directories deep, so a target can be given one ``../`` too many. A file
        is placed *beside* the root as well, so an escaping target names
        something that really exists -- which is the only arrangement in which
        an existence check and a containment check disagree, and the reason the
        rule asks for both.

        Args:
            tmp_path: The pytest-provided temporary directory, which holds both
                the tree and the decoy beside it.

        Returns:
            ``(root, page)``: the tree the targets must stay inside, and the
            Markdown page the exemplars resolve against.
        """
        root = tmp_path / "repo"
        (root / "pkg" / "inner").mkdir(parents=True)
        (root / "sibling.md").write_text("body\n", encoding="utf-8")
        (root / "assets").mkdir()
        (root / "assets" / "shot.png").write_bytes(b"")
        (root / "a b.md").write_text("body\n", encoding="utf-8")
        (tmp_path / "outside.md").write_text("beside the tree, not in it\n", encoding="utf-8")
        page = root / "pkg" / "inner" / "README.md"
        page.write_text("body\n", encoding="utf-8")
        return root, page

    def test_a_target_that_resolves_is_accepted(self, tmp_path: Path) -> None:
        """A link naming a file that is there is not reported."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("[x](../../sibling.md)", page, root)

    def test_a_missing_target_is_reported(self, tmp_path: Path) -> None:
        """A link inside the tree naming no such file is reported as missing."""
        root, page = self._tree(tmp_path)
        assert _unresolved_targets("[x](gone.md)", page, root) == [("gone.md", "does not exist")]

    def test_one_climb_too_many_is_reported_as_escaping(self, tmp_path: Path) -> None:
        """An extra ``../`` leaves the tree, and is reported as that.

        The target names a file that is really there -- just outside the tree.
        An existence check alone accepts it, so this is the cell that makes the
        containment half of the rule load-bearing, and it is the shape the sweep
        found: a verdict that depends on what happens to sit beside the checkout
        is not a verdict.
        """
        root, page = self._tree(tmp_path)
        assert (tmp_path / "outside.md").exists(), "the decoy beside the tree was not created"
        assert _unresolved_targets("[x](../../../outside.md)", page, root) == [
            ("../../../outside.md", "escapes the repository")
        ]

    def test_the_two_reasons_are_distinguishable(self, tmp_path: Path) -> None:
        """A missing target and an escaping one do not share one message."""
        root, page = self._tree(tmp_path)
        reasons = {
            reason
            for text in ("[a](gone.md)", "[b](../../../outside.md)")
            for _, reason in _unresolved_targets(text, page, root)
        }
        assert reasons == {"does not exist", "escapes the repository"}

    def test_a_directory_target_is_accepted(self, tmp_path: Path) -> None:
        """A link to a directory names something that exists."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("[x](../../assets)", page, root)

    def test_an_image_target_is_resolved(self, tmp_path: Path) -> None:
        """``![alt](target)`` is a link to an asset, graded the same way."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("![shot](../../assets/shot.png)", page, root)
        assert _unresolved_targets("![shot](../../assets/gone.png)", page, root) == [
            ("../../assets/gone.png", "does not exist")
        ]

    def test_a_reference_definition_is_resolved(self, tmp_path: Path) -> None:
        """``[label]: target`` carries a target as much as an inline link does."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("[label]: ../../sibling.md\n", page, root)
        assert _unresolved_targets("[label]: gone.md\n", page, root) == [("gone.md", "does not exist")]

    def test_a_percent_encoded_target_is_decoded_before_resolving(self, tmp_path: Path) -> None:
        """``%20`` names a space, so the encoded spelling must resolve too."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("[x](../../a%20b.md)", page, root)

    def test_a_fragment_after_the_path_does_not_change_the_verdict(self, tmp_path: Path) -> None:
        """The path half is resolved; the anchor half is MkDocs' question."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("[x](../../sibling.md#a-heading)", page, root)
        assert _unresolved_targets("[x](gone.md#a-heading)", page, root) == [("gone.md#a-heading", "does not exist")]

    @pytest.mark.parametrize(
        "target",
        ["https://example.com/x.md", "http://example.com", "mailto:someone@example.com", "#a-heading", "/x.md"],
    )
    def test_a_target_that_names_nothing_in_the_tree_is_not_resolved(self, tmp_path: Path, target: str) -> None:
        """Schemed, anchor-only and site-absolute targets are out of scope."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets(f"[x]({target})", page, root)

    def test_a_link_inside_a_fenced_block_is_not_a_link(self, tmp_path: Path) -> None:
        """A fenced block quotes a program, not prose a reader clicks."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets("```python\n[x](gone.md)\n```\n", page, root)

    def test_a_link_inside_an_inline_code_span_is_not_a_link(self, tmp_path: Path) -> None:
        """An inline span quotes a message; its placeholder is not a target.

        The tree really carries this shape: two pages quote the message
        ``Contact the maintainer on [Discord](...)``, whose ``...`` names no file
        and is not meant to. Reading it as a target reports two offenders that
        are not offenders.
        """
        root, page = self._tree(tmp_path)
        text = "It raises `NotImplementedError: Contact the maintainer on [Discord](...)` there.\n"
        assert not _unresolved_targets(text, page, root)

    def test_an_html_anchor_target_is_resolved(self, tmp_path: Path) -> None:
        """``<a href>`` is a link a reader clicks, graded like a Markdown one.

        This is the shape the sweep was blind to: the two MP4 links in
        ``docs/policies/wbc.md`` were written this way, carried one ``../`` too
        many, and were reported by nothing.
        """
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets('<a href="../../sibling.md">x</a>', page, root)
        assert _unresolved_targets('<a href="gone.md">x</a>', page, root) == [("gone.md", "does not exist")]

    def test_an_html_image_target_is_resolved(self, tmp_path: Path) -> None:
        """``<img src>`` names an asset, and a missing one is reported."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets('<img src="../../assets/shot.png" width="100%">', page, root)
        assert _unresolved_targets('<img src="../../assets/gone.png">', page, root) == [
            ("../../assets/gone.png", "does not exist")
        ]

    def test_an_html_target_that_climbs_out_is_reported_as_escaping(self, tmp_path: Path) -> None:
        """The containment half of the rule applies to raw HTML too."""
        root, page = self._tree(tmp_path)
        assert _unresolved_targets('<a href="../../../outside.md">x</a>', page, root) == [
            ("../../../outside.md", "escapes the repository")
        ]

    @pytest.mark.parametrize(
        "element",
        [
            '<source src="gone.mp4" type="video/mp4">',
            '<video src="gone.mp4" controls>',
            '<iframe src="gone.html">',
        ],
    )
    def test_a_target_github_does_not_render_is_not_read(self, tmp_path: Path, element: str) -> None:
        """An element GitHub's sanitizer drops is graded against the site, not here.

        ``<video>``, ``<source>`` and ``<iframe>`` never reach a reader of the
        source file, so the published site is their only consumer and the
        site-relative spelling this tree uses for them is correct.
        ``docs/device-connect.md`` ships one such ``<source src>``; reading it
        against the source tree would report a working embed as broken.
        """
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets(element, page, root)

    @pytest.mark.parametrize("element", ['<img href="gone.md">', '<a src="gone.md">x</a>'])
    def test_an_element_and_attribute_that_do_not_pair_are_not_read(self, tmp_path: Path, element: str) -> None:
        """``<img href>`` and ``<a src>`` carry no target, so neither is a link."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets(element, page, root)

    def test_a_single_quoted_html_attribute_is_read(self, tmp_path: Path) -> None:
        """Both quoting styles are HTML, so both are graded."""
        root, page = self._tree(tmp_path)
        assert _unresolved_targets("<a href='gone.md'>x</a>", page, root) == [("gone.md", "does not exist")]

    def test_an_attribute_merely_ending_in_src_is_not_read(self, tmp_path: Path) -> None:
        """``data-src`` is a different attribute, and names no target to follow."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets('<img data-src="gone.png" src="../../assets/shot.png">', page, root)

    def test_an_html_link_inside_a_fenced_block_is_not_a_link(self, tmp_path: Path) -> None:
        """A fenced block quoting HTML quotes a program, not prose to click."""
        root, page = self._tree(tmp_path)
        assert not _unresolved_targets('```html\n<a href="gone.md">x</a>\n```\n', page, root)

    def test_the_exemplars_reach_both_verdicts(self, tmp_path: Path) -> None:
        """Some exemplar is reported and some accepted, so neither side is vacuous."""
        root, page = self._tree(tmp_path)
        verdicts = {
            bool(_unresolved_targets(text, page, root)) for text in ("[ok](../../sibling.md)", "[bad](gone.md)")
        }
        assert verdicts == {True, False}
