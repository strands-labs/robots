"""Contract pin for the reader attached to the documented merge gate.

``AGENTS.md`` > PR Workflow > step 8 tells a contributor to gate a merge on the
required contexts' own conclusions and ``mergeStateStatus == CLEAN``, and calls
``mergeStateStatus`` "the field that already accounts for the required set,
which is why it is the one to trust". Both halves are true and the sentence is
still not usable on its own, because the field answers *can the viewer merge
this pull request* - so the answer depends on which token asked.

Measured on a control pair, both approved, both with no unresolved thread and
``call-test-lint / Test and Lint`` = ``SUCCESS``, read minutes apart:

===============  ==============================  ================  =============
pull request     edits ``.github/workflows/**``  ``GITHUB_TOKEN``  ``PAT_TOKEN``
===============  ==============================  ================  =============
#1915            yes (``pr-and-push.yml``)       ``BLOCKED``       ``CLEAN``
#1902            no                              ``CLEAN``         ``CLEAN``
===============  ==============================  ================  =============

Both merged clean minutes later (``f4dfde6``, ``6cf0470``), so ``BLOCKED`` was
the wrong answer about the pull request and the right answer to the question the
field asks: an installation token is refused writes under
``.github/workflows/``, so it genuinely cannot perform that merge. The same
token, on the same branch at the same instant, created ``zz_probe.txt`` and was
refused ``.github/workflows/zz_probe.yml`` with ``Resource not accessible by
integration``.

The pin exists because the failure is silent in both directions. On a genuinely
blocked pull request both tokens read ``blocked``, so their agreement proves
nothing; and an agent polling the gate exactly as documented reads ``BLOCKED``,
correctly declines to merge, and reports a ready pull request as waiting on a
reviewer. That presentation stood in eight consecutive scheduled scan summaries
as "reviewer bandwidth is the sole constraint" - the same symptom #1905 records
for a different cause. See #1917.

So what is asserted here is *adjacency*, not vocabulary: the instruction to
trust the field, the scoping that makes it readable, and the class of pull
request it misreads have to stay in the same breath. A future edit tightening
that section back down to "poll ``mergeStateStatus == CLEAN``" is exactly the
regression, it looks like an improvement, and nothing else in the tree would
notice - the same structural reason #1810's correction needed the pin in
``tests/test_codeql_query_filters.py`` rather than trusting prose to stay fixed.

Anchoring deserves a note, because the obvious way to write this pin does not
hold. Asserting "``PAT_TOKEN`` appears within N characters of the trust claim"
passes on the *uncorrected* file: ``PAT_TOKEN`` already occurs 1150 characters
later, in the code-scanning paragraph about ``PAT_TOKEN``-only endpoints, so any
window loose enough to be robust is also loose enough to be vacuous, and the
margin to the corrected distance (625) is too thin to tune against. The assertions
below therefore hang off the scoping sentence, which the correction introduces and
whose absence is the regression itself.

These are text assertions rather than a parsed document because that is the
shape the existing ``AGENTS.md`` pins use (``tests/test_codeql_query_filters.py``
reads the file the same way), and because the claim being pinned is prose: there
is no structured artifact to read instead.
"""

from __future__ import annotations

from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: The phrase naming ``mergeStateStatus`` as the field to gate a merge on.
_TRUST_CLAIM = "which is why it is the one to trust"

#: The scoping sentence the correction adds. Everything else is asserted relative
#: to this, so its absence fails rather than making the other checks vacuous.
_SCOPING = "can the viewer merge"

#: How far the scoping sentence may sit from the trust claim, and the qualifiers
#: from the scoping sentence, while still reading as one instruction. Generous
#: enough to survive rewording, tight enough that moving either out of the gate
#: discussion fails.
_ADJACENCY_WINDOW = 1200


def _agents_text() -> str:
    return _AGENTS_PATH.read_text(encoding="utf-8")


def _window_after(text: str, anchor: str) -> str | None:
    """The ``_ADJACENCY_WINDOW`` characters following ``anchor``, or ``None``."""
    position = text.find(anchor)
    if position < 0:
        return None
    return text[position : position + _ADJACENCY_WINDOW]


class TestTheGateNamesItsReader:
    """``mergeStateStatus == CLEAN`` is only actionable with the reader named."""

    def test_the_trust_claim_is_still_present(self) -> None:
        # Context guard: the assertions below are positioned from this phrase, so
        # a silent rewording would move the pin rather than break it.
        assert _TRUST_CLAIM in _agents_text(), (
            f"AGENTS.md no longer contains {_TRUST_CLAIM!r}, which this class uses to "
            "locate the merge-gate instruction. If the claim was deliberately reworded, "
            "update _TRUST_CLAIM to match rather than deleting these tests - the point "
            "is that the instruction and its qualifiers stay together."
        )

    def test_the_gate_states_that_it_is_viewer_scoped(self) -> None:
        window = _window_after(_agents_text(), _TRUST_CLAIM)
        assert window is not None and _SCOPING in window, (
            "AGENTS.md tells a contributor to trust mergeStateStatus without saying that "
            "the field answers 'can the viewer merge this pull request'. Unscoped, the "
            "instruction is unusable by an agent: the Actions GITHUB_TOKEN reads BLOCKED "
            "on any PR editing .github/workflows/** however ready it is, because an "
            "installation token cannot write workflow files. See #1917."
        )

    def test_the_token_that_can_read_the_gate_is_named(self) -> None:
        window = _window_after(_agents_text(), _SCOPING)
        assert window is not None and "PAT_TOKEN" in window, (
            "AGENTS.md scopes the merge gate to the viewer but does not name the token "
            "that can read it, which leaves the reader knowing the field is unreliable "
            "and not what to do instead. See #1917."
        )

    def test_the_workflow_file_exception_is_stated(self) -> None:
        window = _window_after(_agents_text(), _SCOPING)
        assert window is not None and ".github/workflows" in window, (
            "AGENTS.md must say which pull requests the gate misreads, not only which "
            "token to use. BLOCKED is correct for the Actions token on a PR that edits "
            ".github/workflows/** and equally correct on a PR that is really blocked, so "
            "the two are indistinguishable from the field alone - naming the class is "
            "what makes a BLOCKED read diagnosable. See #1917."
        )
