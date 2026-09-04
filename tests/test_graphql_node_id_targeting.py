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
``[0, repository databaseId, own databaseId]``, so a decode costs no network
call:

=========================  ===============================  ==========================
node ID                    decodes to                       resolves to
=========================  ===============================  ==========================
``R_kgDORUMiZg``           ``[0, 1162027622]``              this repository
``R_kgDOD1WOFw``           ``[0, 257265175]``               the #1916 stray
``PR_kwDOD1WOF87DdSjQ``    ``[0, 257265175, 3279235280]``   the same stray
``PR_kwDORUMiZs7Kw3fA``    ``[0, 1162027622, 3401807808]``  ``uutils/coreutils#11342``
``PR_kwDORUMiZs7DHqZ3``    ``[0, 1162027622, 3273565815]``  ``gip-inclusion/autometa#8``
=========================  ===============================  ==========================

That third row is the finding the incident write-up did not have: all three
guessed IDs in that run carried **one** wrong repository, so a single stale
value contaminated every mutation. The two that failed did so only because
their own databaseId happened not to exist under that repository
(``Could not resolve to a node``). Failing closed was luck about the guess, not
a property of the API - and the one that got lucky the other way is the one
that wrote.

The fourth row is why the decode can only ever *reject*. That ID carries this
repository's databaseId in its middle field and resolves to a merged pull
request in ``uutils/coreutils``, whose own repository databaseId is
``11847500``: GitHub routes on the third field, the object's own id, and
neither validates nor uses the middle one. So the middle field agreeing is not
evidence the ID names anything here, and a ``mergePullRequest`` aimed at that ID
while merging #2006 was stopped by permissions rather than by any check. The two
IDs also share 14 of their 19 characters, so comparing one by eye against a
known-good ID for the same repository is the same unsound test done less
precisely. See #2007.

The fifth row is the same failure again, and it is here for what the *refusal*
turned out to mean. That ID was recalled rather than queried while merging #3175,
and ``mergePullRequest`` answered ``cagataycali does not have the correct
permissions to execute MergePullRequest`` - a sentence about the account. Four
minutes later the same account, token and mutation squashed that same pull
request as ``c418f74``, using the ID from
``repository(owner:, name:) { pullRequest(number: 3175) { id } }``. So the
permission being reported on was a stranger's, and the wording is the one an
agent has no reason to doubt: taken at face value it says merging is a
maintainer's job, which is self-consistent, terminal, and leaves an approved
``CLEAN`` pull request open with every required context ``SUCCESS`` - a third
cause of the presentation ``tests/test_merge_gate_viewer_scope.py`` was written
for (#1917, #1905), and the first on the write side, so re-reading the gate with
``PAT_TOKEN`` does not reach it. Nothing in the response distinguishes the two
cases either, because permission is evaluated before state: the stray target was
already ``MERGED``, and a merge that could not have succeeded for anyone was
still refused for permissions rather than for mergeability.

The exposure is also not symmetric, and that is what the reject direction is
worth keeping for. A refused ``mergePullRequest`` leaves nothing behind; a
``createIssue`` against a wrong ``repositoryId`` succeeds and cannot be undone
by the account that made it. It has now happened twice - the second was
``Ali111q/todo#1``, twenty minutes after #2007 was filed - and that second ID
named a repository that is not this one, which is precisely the shape a decode
does catch. Narrowing the check is therefore not the same as dropping it.

So three classes are asserted here, and the first two are what keep the third
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

``TestTheDecodeIsARejectAndNeverAPass`` *executes* the limit, on the recorded
cross-repository ID: its middle field equals this repository's databaseId while
the object it names belongs to another repository, so a matching middle field
cannot establish the target. The reject direction is asserted alongside it, so
the fix is not satisfiable by deleting the decode.

``TestARefusalIsNotAVerdictOnThePermission`` *executes* the fifth row. It holds
the two IDs against the outcomes they earned, and asserts that the reading the
refusal invites - this account cannot merge here - contradicts the merge the same
account performed minutes later, while a reading keyed on the ID does not. The
recorded ``MERGED`` state of the stray target carries the before-state ordering,
which is why the response cannot be triaged by its wording.

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

Negative control: with ``origin/main``'s ``AGENTS.md`` restored, the qualifiers
each correction introduces fail while the ones the write-up before it already
carried keep passing, and the executable classes pass unchanged. The envelope,
the routing field and the recorded outcomes are properties of GitHub's IDs and of
what already happened rather than of any change here; only the guidance is new.
"""

from __future__ import annotations

import base64
import struct
from pathlib import Path
from typing import NamedTuple

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

# The node ID a `mergePullRequest` was aimed at while merging #2006. Its middle
# field is *this* repository's databaseId, so the decode clears it; the object it
# names is a merged pull request in another repository entirely. Both values read
# back from one `node(id:)` query afterwards. See #2007.
_CROSS_REPOSITORY_NODE_ID = "PR_kwDORUMiZs7Kw3fA"
_CROSS_REPOSITORY_OWN_DATABASE_ID = 3401807808
_CROSS_REPOSITORY_RESOLVES_TO = "uutils/coreutils#11342"
_CROSS_REPOSITORY_ACTUAL_DATABASE_ID = 11847500

# The repository a second stray write landed in, twenty minutes after #2007 was
# filed: `createIssue` with a `repositoryId` that was never resolved in that run.
# Unlike the cross-repository ID above, its repository field is *not* this
# repository, so the surviving reject direction covers it. Read back from
# `repository(owner: "Ali111q", name: "todo")`. See #2007.
_SECOND_STRAY_REPOSITORY_NODE_ID = "R_kgDOPzXPeg"
_SECOND_STRAY_REPOSITORY_DATABASE_ID = 1060491130
_SECOND_STRAY_RESOLVES_TO = "Ali111q/todo#1"

# The ID a `mergePullRequest` was aimed at while merging #3175 - recalled from an
# earlier response rather than resolved in that run. Like the row above it carries
# this repository's databaseId, so the decode clears it; it names a merged pull
# request in a third repository. Every value read back from one `node(id:)` query
# and one `repository(owner: "gip-inclusion", name: "autometa")` query. See #3180.
_RECALLED_NODE_ID = "PR_kwDORUMiZs7DHqZ3"
_RECALLED_OWN_DATABASE_ID = 3273565815
_RECALLED_RESOLVES_TO = "gip-inclusion/autometa#8"
_RECALLED_ACTUAL_DATABASE_ID = 1127363528

#: The state that target was already in. It matters because it rules out the
#: hope that an impossible merge announces itself as one: permission is evaluated
#: first, so this pull request was refused for permissions and not for
#: mergeability.
_RECALLED_TARGET_STATE = "MERGED"

#: What the mutation answered, verbatim. Pinned as a string because the reading it
#: invites - an account without the permission - is the whole defect.
_REFUSAL = "does not have the correct permissions to execute MergePullRequest"

# The ID that mutation should have carried, resolved from
# `repository(owner: "strands-labs", name: "robots") { pullRequest(number: 3175) }`
# four minutes later, and the squash the same account, token and mutation then
# performed - which is what proves the refusal above was not about its permission.
_QUERIED_NODE_ID = "PR_kwDORUMiZs8AAAABCBKOHA"
_QUERIED_OWN_DATABASE_ID = 4430401052
_QUERIED_MERGE_COMMIT = "c418f74"

#: The ID that mutation should have carried, from
#: ``repository(owner:, name:) { pullRequest(number: 2006) { id } }``. Kept to
#: measure how little an eyeball comparison against a known-good ID buys.
_INTENDED_PULL_REQUEST_NODE_ID = "PR_kwDORUMiZs78G3VE"

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


def _encoded_repository(node_id: str) -> int:
    """The repository ``databaseId`` *encoded in* ``node_id``.

    A repository's own ID names it directly; anything it owns carries one as its
    second element. Named for what it reads rather than for what the mutation
    will address, because those differ: GitHub routes an owned object by its own
    id - the third element - and neither validates nor uses this one. So a value
    that disagrees with the intended repository is proof of a wrong ID, and a
    value that agrees proves nothing. ``TestTheDecodeIsARejectAndNeverAPass``
    holds that asymmetry.
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
            "That middle element is what makes a *disagreeing* value a fast reject; "
            "an agreeing one establishes nothing, per #2007."
        )

    def test_the_type_prefix_separates_a_repository_from_what_it_owns(self) -> None:
        # A `PR_...` handed to a parameter wanting a `repositoryId` is wrong by
        # type alone, with nothing else to consult - the cheapest of the checks.
        assert _decode_node_id(_REPOSITORY_NODE_ID)[0] == "R"
        assert _decode_node_id("PR_kwDORUMiZs76PCIu")[0] == "PR"

    def test_the_stray_id_is_distinguishable_from_the_intended_one(self) -> None:
        # The check that would have caught #1916, in the form it was available:
        # the two spellings are visually close and decode to different targets.
        assert _encoded_repository(_REPOSITORY_NODE_ID) == _REPOSITORY_DATABASE_ID
        assert _encoded_repository(_STRAY_REPOSITORY_NODE_ID) == _STRAY_REPOSITORY_DATABASE_ID
        assert _encoded_repository(_STRAY_REPOSITORY_NODE_ID) != _encoded_repository(_REPOSITORY_NODE_ID)

    def test_every_stray_id_from_the_incident_names_one_wrong_repository(self) -> None:
        strays = {_STRAY_REPOSITORY_NODE_ID, _STRAY_PULL_REQUEST_NODE_ID}
        targets = {_encoded_repository(node_id) for node_id in strays}
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


class TestTheDecodeIsARejectAndNeverAPass:
    """A middle field that agrees is not evidence about what the ID names."""

    def test_the_cross_repository_id_decodes_to_this_repository(self) -> None:
        # The check as AGENTS.md first stated it, run on the stray: it passes.
        _, values = _decode_node_id(_CROSS_REPOSITORY_NODE_ID)
        assert values == [
            0,
            _REPOSITORY_DATABASE_ID,
            _CROSS_REPOSITORY_OWN_DATABASE_ID,
        ], (
            f"{_CROSS_REPOSITORY_NODE_ID!r} should carry this repository's databaseId "
            "in its middle field. That it does is the whole point: the offline check "
            "clears it. See #2007."
        )
        assert _encoded_repository(_CROSS_REPOSITORY_NODE_ID) == _REPOSITORY_DATABASE_ID

    def test_the_object_it_names_belongs_to_another_repository(self) -> None:
        # ...and the object is somewhere else, so the pass was false.
        assert _CROSS_REPOSITORY_ACTUAL_DATABASE_ID != _REPOSITORY_DATABASE_ID, (
            f"{_CROSS_REPOSITORY_RESOLVES_TO} must be in a different repository from "
            "this one for the measurement to mean anything. If these two databaseIds "
            "ever agree the constants have drifted; re-read them from a node(id:) "
            "query rather than relaxing this. See #2007."
        )
        assert _encoded_repository(_CROSS_REPOSITORY_NODE_ID) != _CROSS_REPOSITORY_ACTUAL_DATABASE_ID, (
            "The repository encoded in the ID must differ from the repository the ID "
            "resolves to. That inconsistency is what proves GitHub neither validates "
            "nor uses the middle field, and therefore that a decode cannot confirm a "
            "target. See #2007."
        )

    def test_a_differing_middle_field_is_still_a_sound_reject(self) -> None:
        # The direction that survives, so "delete the decode" does not pass this
        # class: #1916's IDs are refusable offline and must stay refusable.
        for stray in (_STRAY_REPOSITORY_NODE_ID, _STRAY_PULL_REQUEST_NODE_ID):
            assert _encoded_repository(stray) != _REPOSITORY_DATABASE_ID, (
                f"{stray!r} named the wrong repository in #1916, which is the one shape "
                "a decode does catch. Keeping this reject is why the correction narrows "
                "the check rather than removing it."
            )

    def test_the_second_incident_is_refusable_by_the_surviving_direction(self) -> None:
        # Why the correction narrows the check instead of deleting it: the write
        # that landed after #2007 was filed is one a decode would have stopped.
        encoded = _encoded_repository(_SECOND_STRAY_REPOSITORY_NODE_ID)
        assert encoded == _SECOND_STRAY_REPOSITORY_DATABASE_ID, (
            f"{_SECOND_STRAY_REPOSITORY_NODE_ID!r} should decode to the databaseId of "
            f"the repository {_SECOND_STRAY_RESOLVES_TO} landed in. If it does not, the "
            "constant drifted; re-read it from a repository(owner:, name:) query."
        )
        assert encoded != _REPOSITORY_DATABASE_ID, (
            f"{_SECOND_STRAY_RESOLVES_TO} was filed from an ID naming another "
            "repository, which is the one shape a decode catches. This is why "
            "AGENTS.md keeps the reject rather than dropping the decode outright."
        )

    def test_the_two_incidents_name_different_wrong_repositories(self) -> None:
        # #1916's write-up read as one stale value contaminating one run. Two
        # distinct wrong repositories make it a recurring class instead, which is
        # what justifies weighting the rule by reversibility at the call site.
        assert _STRAY_REPOSITORY_DATABASE_ID != _SECOND_STRAY_REPOSITORY_DATABASE_ID, (
            "the two stray writes must name different repositories for the recurrence "
            "to be the finding. If these agree, one of the constants is wrong."
        )

    def test_the_prefix_check_is_unaffected_by_the_correction(self) -> None:
        # Orthogonal and still sound: the type is a property of the envelope, not
        # a claim about which object the payload addresses.
        assert _decode_node_id(_CROSS_REPOSITORY_NODE_ID)[0] == "PR"
        assert _decode_node_id(_REPOSITORY_NODE_ID)[0] == "R"

    def test_an_eyeball_comparison_is_the_same_unsound_test(self) -> None:
        # Why "it looked right" is not a defence: the stray and the intended ID
        # agree on their whole prefix and differ only in the routing field.
        shared = 0
        for left, right in zip(_CROSS_REPOSITORY_NODE_ID, _INTENDED_PULL_REQUEST_NODE_ID, strict=False):
            if left != right:
                break
            shared += 1
        assert shared >= 14, (
            f"{_CROSS_REPOSITORY_NODE_ID!r} and {_INTENDED_PULL_REQUEST_NODE_ID!r} share "
            f"only {shared} leading characters. The recorded measurement is 14 of 19, "
            "which is why AGENTS.md says an eyeball comparison against a known-good ID "
            "is no better than the decode. See #2007."
        )
        assert _decode_node_id(_CROSS_REPOSITORY_NODE_ID)[1][2] != _decode_node_id(_INTENDED_PULL_REQUEST_NODE_ID)[1][2]


class _Call(NamedTuple):
    """One recorded ``mergePullRequest`` call, as issued and as answered.

    Every field except ``node_id`` is identical across the two recorded calls, so
    the pair isolates the ID as the only input that differed. ``refusal`` and
    ``merge_commit`` are mutually exclusive by construction: a call either was
    answered with an error or landed a squash.
    """

    node_id: str
    obtained_by: str
    account: str
    mutation: str
    intended_target: int
    resolves_to: str
    refusal: str | None
    merge_commit: str | None


#: The pair measured while merging #3175, in the order they were issued. See #3180.
_RECORDED_CALLS = (
    _Call(
        node_id=_RECALLED_NODE_ID,
        obtained_by="recalled from an earlier response",
        account="cagataycali",
        mutation="mergePullRequest",
        intended_target=3175,
        resolves_to=_RECALLED_RESOLVES_TO,
        refusal=f"cagataycali {_REFUSAL}",
        merge_commit=None,
    ),
    _Call(
        node_id=_QUERIED_NODE_ID,
        obtained_by="repository(owner:, name:) { pullRequest(number: 3175) { id } }",
        account="cagataycali",
        mutation="mergePullRequest",
        intended_target=3175,
        resolves_to="strands-labs/robots#3175",
        refusal=None,
        merge_commit=_QUERIED_MERGE_COMMIT,
    ),
)


def _viewer_lacks_the_permission(call: _Call) -> bool:
    """The verdict the refusal's wording invites: read it as about the account."""
    return call.refusal is not None


def _the_id_named_something_else(call: _Call) -> bool:
    """The verdict that survives the pair: read it as about the ID."""
    return call.resolves_to != f"strands-labs/robots#{call.intended_target}"


class TestARefusalIsNotAVerdictOnThePermission:
    """A permissions error reports on the object the ID resolved to."""

    def test_the_recalled_id_clears_the_decode(self) -> None:
        # The reject direction cannot fire here either: the middle field is this
        # repository's, exactly as in the `uutils/coreutils` row above.
        _, values = _decode_node_id(_RECALLED_NODE_ID)
        assert values == [0, _REPOSITORY_DATABASE_ID, _RECALLED_OWN_DATABASE_ID], (
            f"{_RECALLED_NODE_ID!r} should decode to [0, this repository, its own "
            "databaseId]. If it does not, re-read the constants from a node(id:) query "
            "rather than relaxing this - the decode clearing a wrong ID is the premise "
            "of the whole class. See #3180."
        )
        assert _encoded_repository(_RECALLED_NODE_ID) == _REPOSITORY_DATABASE_ID

    def test_the_object_it_named_belongs_to_another_repository(self) -> None:
        assert _RECALLED_ACTUAL_DATABASE_ID != _REPOSITORY_DATABASE_ID, (
            f"{_RECALLED_RESOLVES_TO} must be in a different repository from this one "
            "for the measurement to mean anything. If these databaseIds ever agree the "
            "constants have drifted; re-read them from a node(id:) query. See #3180."
        )
        assert _encoded_repository(_RECALLED_NODE_ID) != _RECALLED_ACTUAL_DATABASE_ID

    def test_the_two_ids_differ_in_the_routing_field_alone(self) -> None:
        # Why nothing offline separated them: both name this repository, and the
        # field GitHub actually routes on is the only one that disagrees.
        recalled = _decode_node_id(_RECALLED_NODE_ID)[1]
        queried = _decode_node_id(_QUERIED_NODE_ID)[1]
        assert queried == [0, _REPOSITORY_DATABASE_ID, _QUERIED_OWN_DATABASE_ID], (
            f"{_QUERIED_NODE_ID!r} should decode to [0, this repository, pull request "
            "3175's own databaseId]. Pinned to the recorded value rather than compared "
            "field-by-field for the same reason as the two rows above: the inequality "
            "below would still hold if this ID had drifted to name a third object, and "
            "then the row it stands for would no longer be the one that merged. See #3180."
        )
        assert recalled[:2] == queried[:2] == [0, _REPOSITORY_DATABASE_ID]
        assert recalled[2] != queried[2], (
            "the recalled and the queried ID must differ in their own-databaseId field. "
            "That third element is what GitHub routes on, and it is the only difference "
            "between a refusal and a merge here. See #3180."
        )

    def test_the_verdict_the_refusal_invites_is_false_of_the_account(self) -> None:
        # The arithmetic behind the guidance: one verdict keyed on the response,
        # one keyed on the ID, over the same two observations.
        refused = [call for call in _RECORDED_CALLS if _viewer_lacks_the_permission(call)]
        merged = [call for call in _RECORDED_CALLS if call.merge_commit is not None]
        assert refused and merged, (
            "the recorded pair must hold one refused call and one that landed a squash; "
            "without both there is nothing to contradict. See #3180."
        )
        assert refused[0].refusal is not None and _REFUSAL in refused[0].refusal
        assert {call.account for call in _RECORDED_CALLS} == {"cagataycali"}
        assert {call.mutation for call in _RECORDED_CALLS} == {"mergePullRequest"}
        assert {call.intended_target for call in _RECORDED_CALLS} == {3175}
        assert merged[0].merge_commit == _QUERIED_MERGE_COMMIT, (
            "the second call must record the squash it produced. It is the only thing "
            "that proves the account held every permission the merge needed, and so "
            "that the refusal was not about that permission. See #3180."
        )

    def test_the_verdict_keyed_on_the_id_holds_on_both_rows(self) -> None:
        # And the reading that survives: it separates the rows the refusal cannot.
        assert [_the_id_named_something_else(call) for call in _RECORDED_CALLS] == [
            True,
            False,
        ], (
            "asking which object the ID resolved to must separate the refused call from "
            "the one that merged. That is the substitute for a verdict the response "
            "cannot support, and it is what AGENTS.md now tells the reader to reach for."
        )

    def test_permission_is_evaluated_before_the_pull_requests_state(self) -> None:
        # Why the response cannot be triaged by its wording: the stray target was
        # already merged, so no permission on earth could have merged it again -
        # and the error still spoke of permissions rather than of mergeability.
        assert _RECALLED_TARGET_STATE == "MERGED"
        assert "mergeable" not in _REFUSAL.lower(), (
            "the recorded refusal must not mention mergeability. An already-merged "
            "target being refused for permissions is what shows the check order, and "
            "therefore that an impossible merge does not announce itself. See #3180."
        )


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


#: The sentence the correction introduces. Every other assertion is positioned
#: from it, so its absence fails outright rather than making the rest vacuous.
_SUBJECT_CLAIM = "names its subject by node ID"

#: The step-8 bullet boundary. Bounding the slice here makes "in the same
#: bullet" the literal assertion, so a qualifier reworded down into a
#: neighbouring bullet fails while one reworded within this bullet keeps passing
#: however far the bullet grows. The fixed 2600-character window this replaced
#: was measured against a 2320-character bullet, so it reached 280 characters
#: into the next one - and it would have excluded the qualifiers this correction
#: adds, since the bullet is now 3813 characters.
_BULLET_DELIMITER = "\n   - *"


def _bullet_after(text: str, anchor: str) -> str | None:
    """The step-8 bullet containing ``anchor``, whitespace collapsed.

    Collapsed because the assertion is about a qualifier being in the same
    breath as the instruction, not about where the line happens to wrap: a
    reflow must not fail a pin, and a phrase moved to another bullet must.
    """
    position = text.find(anchor)
    if position < 0:
        return None
    end = text.find(_BULLET_DELIMITER, position)
    bullet = text[position:] if end < 0 else text[position:end]
    return " ".join(bullet.split())


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
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "does not fail" in bullet, (
            "AGENTS.md must say that a well-formed but wrong node ID succeeds against "
            "whatever object it does name. Without that, the rule reads as tidiness "
            "rather than as the reason the write is unsafe. See #1916."
        )

    def test_the_guidance_names_the_decodable_envelope(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "databaseId" in bullet, (
            "AGENTS.md must say that a node ID decodes to a type and a target "
            "repository databaseId offline. 'Always query the ID' is advice that can be "
            "forgotten under a stale value, which is exactly what happened; a check that "
            "can be run on the value in hand is not. See #1916."
        )

    def test_the_guidance_states_that_there_is_no_undo(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "deleteIssue" in bullet, (
            "AGENTS.md must say that a write to the wrong repository cannot be undone - "
            "deleteIssue needs admin on the target. That is what makes this a "
            "check-before rather than a verify-after. See #1916."
        )

    def test_the_guidance_tells_the_reader_to_check_the_response(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "url" in bullet, (
            "AGENTS.md must keep the response-url check beside the rule: it is the only "
            "signal for the cases the envelope cannot cover, and in #1916 it was the "
            "single clue that anything had gone wrong. See #1916."
        )


class TestTheGuidanceLimitsTheDecodeToAReject:
    """The correction is only actionable with its seven qualifiers beside it."""

    def test_the_slice_is_the_bullet_and_not_its_neighbour(self) -> None:
        # Non-vacuity for `_bullet_after`: an empty or runaway slice would make
        # every assertion below meaningless in opposite directions.
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None
        assert len(bullet) > 1500, (
            f"the node-ID bullet collapsed to {len(bullet)} characters, so the pins "
            "below are reading a fragment rather than the passage."
        )
        assert "still accepts writes at all" not in bullet, (
            "the slice ran past the node-ID bullet into the archived-repository one, so "
            "a qualifier moved out of this bullet would still pass. Check "
            "_BULLET_DELIMITER against AGENTS.md's list indentation."
        )

    def test_the_guidance_states_the_decode_is_a_reject_not_a_pass(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "reject and never a pass" in bullet, (
            "AGENTS.md must say the decode is a reject and never a pass. Stated as a "
            "check that can be run before the write, it reads as a guard and clears an "
            "ID naming an object in another repository. See #2007."
        )

    def test_the_guidance_says_a_matching_repository_proves_nothing(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "proves nothing at all" in bullet, (
            "AGENTS.md must state the false-safe direction explicitly. 'Reject only' "
            "without it invites the reader to keep treating a clean decode as "
            "confirmation, which is the whole failure. See #2007."
        )

    def test_the_guidance_carries_the_cross_repository_measurement(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "uutils/coreutils" in bullet, (
            "AGENTS.md must keep the measured ID that clears the decode and resolves "
            "elsewhere. Without it the narrowing is an assertion about GitHub's routing "
            "that a reader has no reason to accept. See #2007."
        )

    def test_the_guidance_weights_the_rule_by_reversibility(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "reversibility" in bullet, (
            "AGENTS.md must say the two directions are not symmetric: a refused merge "
            "leaves nothing behind, a createIssue against a wrong ID cannot be undone. "
            "That is the sentence that changes behaviour at the call site. See #2007."
        )

    def test_the_guidance_scopes_the_rule_to_mutations(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "Only a mutation takes a bare ID" in bullet, (
            "AGENTS.md must say only a mutation takes a bare ID. Without it the rule "
            "reads as applying to every call, which makes it look expensive enough to "
            "skip - and a query cannot address the wrong repository at all. See #2007."
        )

    def test_the_guidance_records_the_second_incident(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and _SECOND_STRAY_RESOLVES_TO in bullet, (
            f"AGENTS.md must record that this recurred: {_SECOND_STRAY_RESOLVES_TO} "
            "landed twenty minutes after #2007 was filed, from an ID a decode would "
            "have rejected. One incident reads as a mishap; two is the reason the "
            "reject direction is kept and the read-back is not optional. See #2007."
        )

    def test_the_guidance_names_the_remedy_for_a_write_that_landed(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "opened in error" in bullet, (
            "AGENTS.md must name the remedy for a stray write that succeeded, because "
            "the instinct - delete it - is the one thing that does not work: deleteIssue "
            "needs admin on the target. See #2007."
        )


class TestTheGuidanceStatesWhatARefusalDecides:
    """The correction is only actionable with its four qualifiers beside it."""

    def test_the_guidance_says_a_refusal_is_not_a_verdict_on_the_permission(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "not a verdict on your own permission" in bullet, (
            "AGENTS.md must say the refusal is not a verdict on the viewer's own "
            "permission. Recorded only as 'was stopped by permissions' it reads as a "
            "safety net, which is the half that is true and the half that costs "
            "nothing. See #3180."
        )

    def test_the_guidance_carries_the_pair_that_measures_it(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None
        assert _RECALLED_RESOLVES_TO in bullet and _QUERIED_MERGE_COMMIT in bullet, (
            f"AGENTS.md must keep both halves of the measurement - {_RECALLED_RESOLVES_TO} "
            f"and the squash {_QUERIED_MERGE_COMMIT} the same account performed minutes "
            "later. Either alone is an anecdote; together they are what rules out the "
            "account genuinely lacking the permission. See #3180."
        )

    def test_the_guidance_states_that_permission_is_evaluated_before_state(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "evaluated before" in bullet, (
            "AGENTS.md must say permission is evaluated before state. Without it a "
            "reader is left with the hope that an impossible merge would have announced "
            "itself as unmergeable, and it does not. See #3180."
        )

    def test_the_guidance_names_the_misreading_it_prevents(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "merging is a maintainer" in bullet, (
            "AGENTS.md must name the conclusion the refusal invites. The failure is not "
            "confusion but a confident wrong belief that ends the work with an approved "
            "pull request left open, which is the presentation #1905 and #1917 record "
            "for their own causes. See #3180."
        )

    def test_the_reversibility_clause_no_longer_says_a_refusal_costs_nothing(self) -> None:
        bullet = _bullet_after(_agents_text(), _SUBJECT_CLAIM)
        assert bullet is not None and "behind in the repository" in bullet, (
            "AGENTS.md's reversibility weighting must scope 'leaves nothing behind' to "
            "the repository. Unqualified it is the sentence that invites skipping the "
            "read-back on exactly the mutation measured here. See #3180."
        )
