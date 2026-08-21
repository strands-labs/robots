"""Contract pin: every workflow ``uses:`` reference pins a commit SHA.

``AGENTS.md`` > **Action Pinning** states the rule and names the incident it exists
to prevent::

    All `uses:` references in workflows pin to a full 40-character commit SHA,
    with the version tag preserved as a trailing comment: `uses: actions/checkout@<sha>  # v4.2.2`.
    ...
    Especially `pypa/gh-action-pypi-publish` - it uses a moving `release/v1` branch,
    which is exactly the supply-chain pattern that the `tj-actions/changed-files`
    incident exploited. This pin is non-negotiable.

Nothing graded it. ``tests/`` reads ``.github/workflows/`` in seven places -- apt
recommends, the overlap gate, the CodeQL config, the ruleset scope, the viewer
scope, the dependabot location, the lockfile gate -- and not one of them reads a
``uses:`` line. So ``docs.yml`` sat outside the rule with every signal green, and
all four of its references named a moving tag:

===================================  ======  ==========================================
file                                 line    reference
===================================  ======  ==========================================
``.github/workflows/docs.yml``          35   ``actions/checkout@v4``
``.github/workflows/docs.yml``          36   ``actions/setup-python@v5``
``.github/workflows/docs.yml``          47   ``actions/upload-pages-artifact@v3``
``.github/workflows/docs.yml``          65   ``actions/deploy-pages@v4``
===================================  ======  ==========================================

``docs.yml`` is also the workflow holding ``id-token: write`` and ``pages: write``
for the Pages deployment, so it was the most expensive place in the tree to leave
a mutable reference.

That the tags move is not hypothetical. ``actions/checkout@v4`` resolved to
``11bd7190`` (v4.2.2) when the 13 sibling workflows were pinned; resolving the
same tag while writing this file returned ``11d5960a`` (**v4.4.0**). The tag had
moved two minor versions under a reference that reads as though it were fixed.

Four properties are graded here, and the last one is what the above measurement
demands. Pinning ``docs.yml`` to whatever ``v4`` names *today* would have removed
the mutability and silently upgraded the docs build's checkout to v4.4.0 while
its 13 siblings stayed on v4.2.2 -- a behaviour change smuggled into a
supply-chain fix, and a second version for Dependabot to reconcile separately.
Requiring one SHA per action tree-wide makes the sibling SHA the only legal
answer, and keeps each Dependabot bump a single atomic edit.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"

# A ``uses:`` line, with the trailing ``# <tag>`` comment the rule requires when
# one is present. Both the list-item form (``- uses: x``) and the keyed form
# (``uses: x`` under a named step) appear in the tree.
_USES = re.compile(r"^\s*(?:-\s+)?uses:\s*(?P<ref>\S+?)\s*(?:#\s*(?P<comment>\S.*?))?\s*$")

# Every line that *mentions* a ``uses:`` key, whether or not the parser above
# could read it. The two counts must agree, or the parser has gone blind to a
# shape the tree uses and would report clean over references it never saw.
_USES_MENTION = re.compile(r"^\s*(?:-\s+)?uses:")

_SHA = re.compile(r"^[0-9a-f]{40}$")

# Floors, so a walk that stops finding the tree fails instead of passing
# vacuously. The tree holds 14 workflows and 33 non-local references; these sit
# below that with room for a workflow to be retired.
_WORKFLOW_FLOOR = 12
_REMOTE_REF_FLOOR = 28


class ActionRef(NamedTuple):
    """One ``uses:`` reference, with where it was found."""

    workflow: str
    line: int
    ref: str
    comment: str | None

    @property
    def is_local(self) -> bool:
        """A ``./``-relative reusable-workflow call, which names no SHA by design."""
        return self.ref.startswith("./")

    @property
    def action(self) -> str:
        """The part before ``@`` -- ``actions/checkout`` for ``actions/checkout@<sha>``."""
        return self.ref.split("@", 1)[0]

    @property
    def version(self) -> str | None:
        """The part after ``@``, or ``None`` when the reference names none."""
        _, _, version = self.ref.partition("@")
        return version or None

    def __str__(self) -> str:
        return f"{self.workflow}:{self.line} {self.ref}"


def _parse(path: Path) -> tuple[tuple[ActionRef, ...], int]:
    """Return the references in ``path`` and the number of ``uses:`` lines it holds."""
    refs: list[ActionRef] = []
    mentions = 0
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not _USES_MENTION.match(line):
            continue
        mentions += 1
        match = _USES.match(line)
        if match is None:
            continue
        refs.append(
            ActionRef(
                workflow=path.name,
                line=number,
                ref=match.group("ref"),
                comment=match.group("comment"),
            )
        )
    return tuple(refs), mentions


_WORKFLOW_FILES = sorted(_WORKFLOWS.glob("*.yml")) if _WORKFLOWS.is_dir() else []
_PARSED = {path.name: _parse(path) for path in _WORKFLOW_FILES}
_ALL_REFS = tuple(ref for refs, _ in _PARSED.values() for ref in refs)
_REMOTE_REFS = tuple(ref for ref in _ALL_REFS if not ref.is_local)
_LOCAL_REFS = tuple(ref for ref in _ALL_REFS if ref.is_local)


def _ids(refs: tuple[ActionRef, ...]) -> list[str]:
    return [f"{ref.workflow}:{ref.line}" for ref in refs]


class TestTheCensusSeesTheTree:
    """Floors. Without these, a broken walk reports a clean pinning record."""

    def test_the_workflow_directory_is_found(self) -> None:
        assert _WORKFLOWS.is_dir(), f"{_WORKFLOWS} is missing"

    def test_the_workflow_count_clears_its_floor(self) -> None:
        assert len(_WORKFLOW_FILES) >= _WORKFLOW_FLOOR, (
            f"found {len(_WORKFLOW_FILES)} workflows, floor is {_WORKFLOW_FLOOR}; "
            f"either the tree shrank or the glob stopped matching"
        )

    def test_the_remote_reference_count_clears_its_floor(self) -> None:
        assert len(_REMOTE_REFS) >= _REMOTE_REF_FLOOR, (
            f"found {len(_REMOTE_REFS)} non-local uses: references, floor is "
            f"{_REMOTE_REF_FLOOR}; a parser that matches nothing would otherwise pass"
        )

    def test_the_local_reusable_workflow_calls_are_still_seen(self) -> None:
        assert _LOCAL_REFS, (
            "no ./-relative reusable-workflow call was found; the tree has always had "
            "at least one (pr-and-push.yml, pypi-publish-on-release.yml), so the "
            "local-reference branch below is no longer being exercised"
        )

    @pytest.mark.parametrize("workflow", sorted(_PARSED))
    def test_every_uses_line_is_read(self, workflow: str) -> None:
        """A ``uses:`` line the parser cannot read is a reference nothing grades."""
        refs, mentions = _PARSED[workflow]
        assert len(refs) == mentions, (
            f"{workflow} holds {mentions} uses: line(s) but the parser read {len(refs)}; "
            f"an unread line is ungraded, so widen the pattern rather than ignoring it"
        )


class TestEveryActionReferencePinsACommitSha:
    """The rule itself: a remote reference names an immutable commit."""

    @pytest.mark.parametrize("ref", _REMOTE_REFS, ids=_ids(_REMOTE_REFS))
    def test_a_forty_character_sha_is_named(self, ref: ActionRef) -> None:
        version = ref.version
        assert version is not None, f"{ref} names no version at all, so it resolves to the action's default branch"
        assert _SHA.match(version), (
            f"{ref} pins '{version}', which is not a 40-character commit SHA. A tag or "
            f"branch is mutable and resolves to whatever it points at when the job starts "
            f"- the pattern the tj-actions/changed-files incident exploited. See AGENTS.md "
            f"> Action Pinning."
        )


class TestEveryPinNamesTheVersionItPins:
    """A bare SHA is unreadable, so the rule requires the tag alongside it."""

    @pytest.mark.parametrize("ref", _REMOTE_REFS, ids=_ids(_REMOTE_REFS))
    def test_a_trailing_comment_names_the_version(self, ref: ActionRef) -> None:
        assert ref.comment, (
            f"{ref} carries no trailing '# <tag>' comment. A 40-character SHA names no "
            f"version a reader or reviewer can recognise, which is why AGENTS.md requires "
            f"the tag be preserved beside it."
        )


class TestTheOnlyUnpinnedReferenceShapeIsLocal:
    """The exemption is structural, not discretionary."""

    @pytest.mark.parametrize("ref", _LOCAL_REFS, ids=_ids(_LOCAL_REFS))
    def test_an_unpinned_reference_is_a_workflow_in_this_repository(self, ref: ActionRef) -> None:
        """A ``./`` call runs this commit's own file, so there is no third party to pin."""
        assert ref.version is None, f"{ref} is a local call and should name no version"
        target = _REPO_ROOT / ref.ref[len("./") :]
        assert target.is_file(), (
            f"{ref} points at {target}, which does not exist; a local call that resolves "
            f"to nothing is not an exempt reference, it is a broken one"
        )


class TestAnActionResolvesToOneShaTreeWide:
    """One version per action, so a bump stays a single atomic edit.

    ``actions/checkout@v4`` moved from v4.2.2 to v4.4.0 between the sibling
    workflows being pinned and ``docs.yml`` being brought in line. Pinning to the
    tag's current target would have left the docs build on a different checkout
    than the other 13 workflows -- an upgrade nobody asked for, arriving inside a
    change whose stated purpose was to remove mutability. Dependabot rewrites
    every occurrence of an action together, so a split is drift rather than a
    state it produces.
    """

    def test_no_action_is_pinned_to_two_different_shas(self) -> None:
        by_action: dict[str, dict[str, list[ActionRef]]] = defaultdict(lambda: defaultdict(list))
        for ref in _REMOTE_REFS:
            version = ref.version
            if version is not None and _SHA.match(version):
                by_action[ref.action][version].append(ref)

        assert by_action, "no SHA-pinned reference was found to group"

        split = {action: shas for action, shas in by_action.items() if len(shas) > 1}
        assert not split, "an action is pinned to more than one SHA:\n" + "\n".join(
            f"  {action}:\n"
            + "\n".join(
                f"    {sha[:8]} ({refs[0].comment or 'no tag comment'}) "
                f"- {', '.join(f'{r.workflow}:{r.line}' for r in refs)}"
                for sha, refs in sorted(shas.items())
            )
            for action, shas in sorted(split.items())
        )
