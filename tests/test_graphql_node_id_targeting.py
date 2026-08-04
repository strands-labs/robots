"""Contract pin for what names a GraphQL mutation's subject, and what checks it.

``AGENTS.md`` > PR Workflow > step 8 tells a contributor to verify a pull
request's state by reading it back, and covers the mutation *reporting* the
wrong thing - a ``mergePullRequest`` that says "not mergeable" on a merge that
landed. It did not cover the mutation *addressing* the wrong thing, which is the
more expensive direction because it has no undo.

A mutation names its subject by node ID and by nothing else: ``createIssue``
takes a ``repositoryId``, not an owner and a name. So a well-formed ID that is
wrong does not fail - it succeeds against whatever object it does name. Filing
an issue for this repository with a ``repositoryId`` carried over from an
earlier response rather than queried created issue #1 in an unrelated
third-party repository and returned success. ``deleteIssue`` needs admin on the
*target*, so it could not be undone; it was closed as ``NOT_PLANNED`` with an
apology. See #1916.

What makes the rule worth pinning rather than merely regretting is that the
premise the incident was written up under is measurably false. The ID is *not*
opaque. It is ``<TypePrefix>_<urlsafe-base64(msgpack array)>``, where a
repository is ``[0, databaseId]`` and anything a repository owns is
``[0, repository databaseId, own databaseId]``, so both the type and the target
repository are readable offline before the write:

=============================  ==========================  =======================
node ID                        decodes to                  target
=============================  ==========================  =======================
``R_kgDORUMiZg``               ``[0, 1162027622]``          this repository
``R_kgDOD1WOFw``               ``[0, 257265175]``           the stray repository
``PR_kwDOD1WOF87DdSjQ``        ``[0, 257265175, ...]``      the same stray one
=============================  ==========================  =======================

That third row is the finding the incident write-up did not have: all three
guessed IDs in that run carried **one** wrong repository, so a single stale
value contaminated every mutation. The two that failed did so only because
their own databaseId happened not to exist under that repository
(``Could not resolve to a node``). Failing closed was luck about the guess, not
a property of the API - and the one that got lucky the other way is the one
that wrote.

So two classes are asserted here, and the first is what keeps the second
honest:

``TestTheNodeIdEnvelopeIsCheckableOffline`` *executes* the claim. It decodes
this repository's own node ID and the node IDs of an issue and a pull request in
it, and asserts each recovers the ``databaseId`` the API publishes alongside it
- values obtained from one ``repository(owner: "strands-labs", name: "robots")``
query, which is exactly the literal-owner-and-name query the guidance asks for.
A pin that merely asserted ``AGENTS.md`` *says* the ID is checkable would pass
against a future ID format that had stopped being checkable, leaving the
guidance reading plausibly while advising something impossible. This fails
instead.

``TestTheGuidanceNamesTheDecodableEnvelope`` pins the prose, because the prose
is the deliverable: an agent reads ``AGENTS.md``, not this module. What is
asserted is *adjacency* rather than vocabulary - the fail-open property, the
decodable envelope and the absence of an undo have to stay in the same breath as
the instruction, since each one alone is unactionable. A future edit tightening
the passage back to "resolve IDs with a query" is exactly the regression, it
looks like an improvement, and nothing else in the tree would notice. That is
the same structural reason ``tests/test_merge_gate_viewer_scope.py`` and
``tests/test_codeql_query_filters.py`` exist, and these text assertions follow
the shape those modules established.

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, all 5 tests in
``TestTheGuidanceNamesTheDecodableEnvelope`` fail - the four qualifiers and the
context guard that locates the passage - while ``TestTheNodeIdEnvelopeIsCheckableOffline``
passes unchanged. The envelope is a property of GitHub's IDs rather than of this
change; only the guidance is new.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

# Ground truth, read from one `repository(owner: "strands-labs", name: "robots")`
# query. Each pair is a node ID and the `databaseId` the API returns beside it,
# so the decoder below is checked against GitHub's own answer rather than
# against a reimplementation of itself.
_REPOSITORY_NODE_ID = "R_kgDORUMiZg"
_REPOSITORY_DATABASE_ID = 1162027622

#: ``(node ID, own databaseId)`` for objects this repository owns.
_OWNED_OBJECTS = [
    pytest.param("I_kwDORUMiZs8AAAABLT9z4g", 5054100450, id="issue-1916"),
    pytest.param("PR_kwDORUMiZs76PCIu", 4198244910, id="pull-request-1920"),
]

# The node IDs from the #1916 incident. Both name a repository that is not this
# one, and both name the *same* one.
_STRAY_REPOSITORY_NODE_ID = "R_kgDOD1WOFw"
_STRAY_PULL_REQUEST_NODE_ID = "PR_kwDOD1WOF87DdSjQ"
_STRAY_REPOSITORY_DATABASE_ID = 257265175

# msgpack tags the envelope uses. A GitHub node ID is a short array of unsigned
# integers, so this is the whole grammar needed - anything else is a shape this
# decoder must refuse rather than guess at.
_FIXARRAY_MASK = 0xF0
_FIXARRAY_TAG = 0x90
_UINT32_TAG = 0xCE
_UINT64_TAG = 0xCF
_POSITIVE_FIXINT_MAX = 0x80


def _decode_node_id(node_id: str) -> tuple[str, list[int]]:
    """Split a GitHub node ID into its type prefix and its integer payload.

    Args:
        node_id: A next-generation node ID, ``<TypePrefix>_<base64 payload>``.

    Returns:
        The type prefix (``"R"``, ``"I"``, ``"PR"``, ...) and the decoded
        integers. A repository yields ``[0, databaseId]``; an object a
        repository owns yields ``[0, repository databaseId, own databaseId]``.

    Raises:
        ValueError: If ``node_id`` is not a decodable envelope of that shape.
            Raised rather than returning a partial answer, because a decoder
            that guesses is worse than no decoder here: the value it would
            return is the one used to decide whether a write is safe.
    """
    prefix, _, payload = node_id.partition("_")
    if not payload:
        raise ValueError(f"node ID {node_id!r} has no '_' separator")
    try:
        raw = base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
    except (ValueError, TypeError) as exc:
        raise ValueError(f"node ID {node_id!r} payload is not base64: {exc}") from exc
    # base64.urlsafe_b64decode is permissive - it discards bytes outside the
    # alphabet and everything after the padding - so a corrupted payload decodes
    # to the same integers as a valid one. Re-encoding is what proves the whole
    # value was read: "kgDORUMiZg==extra" otherwise yields this repository's id.
    if base64.urlsafe_b64encode(raw).rstrip(b"=").decode() != payload.rstrip("="):
        raise ValueError(f"node ID {node_id!r} payload does not round-trip through base64")
    if not raw or raw[0] & _FIXARRAY_MASK != _FIXARRAY_TAG:
        raise ValueError(f"node ID {node_id!r} payload is not a msgpack array")

    values: list[int] = []
    index = 1
    while index < len(raw):
        tag = raw[index]
        if tag == _UINT32_TAG:
            (value,) = struct.unpack_from(">I", raw, index + 1)
            index += 5
        elif tag == _UINT64_TAG:
            (value,) = struct.unpack_from(">Q", raw, index + 1)
            index += 9
        elif tag < _POSITIVE_FIXINT_MAX:
            value = tag
            index += 1
        else:
            raise ValueError(f"node ID {node_id!r} holds unsupported msgpack tag {tag:#x}")
        values.append(value)
    expected = raw[0] & ~_FIXARRAY_MASK
    if len(values) != expected:
        raise ValueError(f"node ID {node_id!r} declares {expected} values, decoded {len(values)}")
    return prefix, values


def _target_repository(node_id: str) -> int:
    """The ``databaseId`` of the repository ``node_id`` writes to.

    A repository's own ID names it directly; anything it owns carries it as the
    second element. Either way the answer is available without a network call,
    which is the property the guidance rests on.
    """
    prefix, values = _decode_node_id(node_id)
    if prefix == "R":
        return values[1]
    if len(values) < 3:
        raise ValueError(f"node ID {node_id!r} names no repository")
    return values[1]


class TestTheNodeIdEnvelopeIsCheckableOffline:
    """The type and the target repository are readable before the write."""

    def test_the_repository_id_decodes_to_its_published_database_id(self) -> None:
        prefix, values = _decode_node_id(_REPOSITORY_NODE_ID)
        assert prefix == "R"
        assert values == [0, _REPOSITORY_DATABASE_ID], (
            "This repository's node ID no longer decodes to the databaseId the API "
            "publishes beside it. Either the envelope format changed - in which case "
            "the AGENTS.md guidance that a repositoryId can be checked offline is now "
            "wrong and must be corrected rather than this test relaxed - or the "
            "constants drifted. See #1916."
        )

    @pytest.mark.parametrize(("node_id", "database_id"), _OWNED_OBJECTS)
    def test_an_owned_object_carries_the_repository_it_belongs_to(self, node_id: str, database_id: int) -> None:
        _, values = _decode_node_id(node_id)
        assert values == [0, _REPOSITORY_DATABASE_ID, database_id], (
            f"{node_id!r} should decode to [0, this repository, its own databaseId]. "
            "That middle element is what lets a mutation on an issue or a pull "
            "request be checked against the repository it was meant for."
        )

    def test_the_type_prefix_separates_a_repository_from_what_it_owns(self) -> None:
        # A `PR_...` handed to a parameter wanting a `repositoryId` is wrong by
        # type alone, with nothing else to consult - the cheapest of the checks.
        assert _decode_node_id(_REPOSITORY_NODE_ID)[0] == "R"
        assert _decode_node_id("PR_kwDORUMiZs76PCIu")[0] == "PR"

    def test_the_stray_id_is_distinguishable_from_the_intended_one(self) -> None:
        # The check that would have caught #1916, in the form it was available:
        # the two spellings are visually close and decode to different targets.
        assert _target_repository(_REPOSITORY_NODE_ID) == _REPOSITORY_DATABASE_ID
        assert _target_repository(_STRAY_REPOSITORY_NODE_ID) == _STRAY_REPOSITORY_DATABASE_ID
        assert _target_repository(_STRAY_REPOSITORY_NODE_ID) != _target_repository(_REPOSITORY_NODE_ID)

    def test_every_stray_id_from_the_incident_names_one_wrong_repository(self) -> None:
        strays = {_STRAY_REPOSITORY_NODE_ID, _STRAY_PULL_REQUEST_NODE_ID}
        targets = {_target_repository(node_id) for node_id in strays}
        assert targets == {_STRAY_REPOSITORY_DATABASE_ID}, (
            "The repository ID and the pull-request ID guessed in that run should both "
            "decode to the same wrong repository. That is why one stale value was able "
            "to contaminate three mutations, and why the two that failed failed only by "
            "luck about their own databaseId rather than by any check. See #1916."
        )

    @pytest.mark.parametrize(
        "malformed",
        ["RkgDORUMiZg", "R_!!!!", "R_AAAA", "R_kgDORUMiZg==extra"],
        ids=["no-separator", "not-base64", "not-an-array", "trailing-garbage"],
    )
    def test_a_shape_it_cannot_read_is_refused(self, malformed: str) -> None:
        # Non-vacuity: the decoder must not answer for an envelope it cannot
        # read. A plausible-looking integer here would be used to decide that a
        # write is safe, so guessing is worse than refusing.
        with pytest.raises(ValueError):
            _decode_node_id(malformed)


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


#: The sentence the correction introduces. Every other assertion is positioned
#: from it, so its absence fails outright rather than making the rest vacuous.
_SUBJECT_CLAIM = "names its subject by node ID"

#: How far a qualifier may sit from the claim while still reading as one
#: instruction. Generous enough to survive rewording, tight enough that moving a
#: qualifier out of step 8 fails.
_ADJACENCY_WINDOW = 2600


def _window_after(text: str, anchor: str) -> str | None:
    """The ``_ADJACENCY_WINDOW`` characters following ``anchor``, or ``None``."""
    position = text.find(anchor)
    if position < 0:
        return None
    return text[position : position + _ADJACENCY_WINDOW]


class TestTheGuidanceNamesTheDecodableEnvelope:
    """The rule is only actionable with all three qualifiers beside it."""

    def test_the_subject_claim_is_present(self) -> None:
        # Context guard: the assertions below are positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _SUBJECT_CLAIM in _agents_text(), (
            f"AGENTS.md no longer contains {_SUBJECT_CLAIM!r}, which this class uses to "
            "locate the node-ID rule. If the claim was deliberately reworded, update "
            "_SUBJECT_CLAIM to match rather than deleting these tests - the point is "
            "that the rule and its qualifiers stay together."
        )

    def test_the_guidance_states_that_a_wrong_id_fails_open(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "does not fail" in window, (
            "AGENTS.md must say that a well-formed but wrong node ID succeeds against "
            "whatever object it does name. Without that, the rule reads as tidiness "
            "rather than as the reason the write is unsafe. See #1916."
        )

    def test_the_guidance_names_the_decodable_envelope(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "databaseId" in window, (
            "AGENTS.md must say that a node ID decodes to a type and a target "
            "repository databaseId offline. 'Always query the ID' is advice that can be "
            "forgotten under a stale value, which is exactly what happened; a check that "
            "can be run on the value in hand is not. See #1916."
        )

    def test_the_guidance_states_that_there_is_no_undo(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "deleteIssue" in window, (
            "AGENTS.md must say that a write to the wrong repository cannot be undone - "
            "deleteIssue needs admin on the target. That is what makes this a "
            "check-before rather than a verify-after. See #1916."
        )

    def test_the_guidance_tells_the_reader_to_check_the_response(self) -> None:
        window = _window_after(_agents_text(), _SUBJECT_CLAIM)
        assert window is not None and "url" in window, (
            "AGENTS.md must keep the response-url check beside the rule: it is the only "
            "signal for the cases the envelope cannot cover, and in #1916 it was the "
            "single clue that anything had gone wrong. See #1916."
        )
