"""Contract pin for the reply gate attached to the documented review-comment step.

``AGENTS.md`` > PR Workflow > step 5 tells a contributor to "address all review
comments". That instruction is complete for a human, who remembers replying, and
incomplete for an agent that rebuilds its context from the GitHub payload each
scheduled run: the reviewer's question is still sitting there verbatim, and the
agent's own prior reply is indistinguishable from context it has not yet acted
on. So the instruction is *satisfied* by replying again, and again.

Measured on #1899, one review thread, replies by the pull request's own author:

===========  ==============  =========================================
reply        posted (UTC)    thread state when posted
===========  ==============  =========================================
1st          21:46           open, question unanswered
2nd          21:52           resolved *by this reply*
3rd - 12th   22:28 - 01:30   ``isResolved: true``, ``isOutdated: true``
===========  ==============  =========================================

Twelve consecutive replies, every one announcing the same commit
(``35ee25d2``), against a branch whose last push was 22:37 - so from the 3rd on
they described work already complete and already announced twice, and were still
arriving hourly after the pull request was green and waiting on nothing but a
reviewer. The loop is self-feeding rather than decaying, which is why it needs a
written gate: replying makes the thread the most recently active thing on the
pull request, so it is the first thing the next cycle reads.

What is pinned here is *adjacency*, for the same reason as
``tests/test_merge_gate_viewer_scope.py``: both halves of the uncorrected step 5
are true, and a future edit tightening it back to a bare "address all review
comments" is exactly the regression, looks like a simplification, and nothing
else in the tree would notice. The gate has to stay in the same breath as the
instruction it qualifies.

Two anchoring notes, because the obvious ways to write this pin do not hold.

First, the assertions run against **whitespace-normalised** text. ``AGENTS.md``
is hard-wrapped near 76 columns, so any multi-word phrase can straddle a line
break: ``the push is the message`` is present in the corrected file and a
line-oriented ``grep`` for it returns nothing, because it wraps after "is". A
line-based pin would therefore fail on a pure reflow that changes no meaning,
and - worse - could pass vacuously if a phrase it searched for happened not to
wrap. Normalising first makes the pin insensitive to wrapping and sensitive only
to wording.

Second, the directives are bounded by the **step 5 / step 6 slice** rather than a
tuned character window. A window wide enough to be robust is also wide enough to
be vacuous here: ``AGENTS.md`` discusses review threads in the CI Security
Baseline section too ("resolve the thread with a reply carrying the reasoning"),
so a loose window around step 5 can be satisfied by prose about CodeQL
dismissals that says nothing about replying twice. Slicing to the step it belongs
to means a directive moved out of step 5 fails, which is the intent - the gate is
useless where a contributor addressing review comments will not read it.
"""

from __future__ import annotations

import re
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_AGENTS_PATH = _REPO_ROOT / "AGENTS.md"

#: PR Workflow step 5 - the instruction the gate qualifies. Unique in the file.
_STEP_5 = "address all review comments"

#: PR Workflow step 6 - the end of step 5's prose, used as the slice bound.
_STEP_6 = "Track follow-up items as issues on the [project board]"

#: The gate sentence. Its absence is the regression, so every other assertion
#: hangs off it rather than off a character offset from step 5.
_GATE = "Check whether you are already the thread's last author before replying."

#: The reason the bare instruction is insufficient. Dropping this is how the
#: rule gets "simplified" back into the loop it exists to stop.
_PER_CONCERN = "per-*concern* obligation, not a per-*cycle* one"

#: The two dispositions, each paired with its directive. The pairing is the
#: point: naming the fields without "do not reply" documents a payload, not a
#: rule.
_OWN_REPLY_DIRECTIVE = "is **yours** -> do not reply"
_TERMINAL_STATE_DIRECTIVE = "`isResolved` or `isOutdated`** -> do not reply"

#: The single-reply case, so the gate cannot be read as "never reply".
_REPLY_ONCE = "reply once, then resolve"


def _normalised() -> str:
    """``AGENTS.md`` with every whitespace run collapsed to one space.

    See the module docstring: the file is hard-wrapped, so phrases straddle line
    breaks and a line-oriented read is both fragile and potentially vacuous.
    """
    return re.sub(r"\s+", " ", _AGENTS_PATH.read_text(encoding="utf-8"))


def _step_5_prose() -> str:
    """The text of PR Workflow step 5, bounded by step 6."""
    text = _normalised()
    start = text.index(_STEP_5)
    end = text.index(_STEP_6, start)
    return text[start:end]


def test_step_5_anchor_is_unique() -> None:
    """The slice bounds must be unambiguous or every assertion below drifts."""
    text = _normalised()
    assert text.count(_STEP_5) == 1, (
        f"{_STEP_5!r} is no longer a unique anchor in AGENTS.md; the step 5 slice "
        "this module asserts against cannot be located unambiguously."
    )
    assert text.count(_STEP_6) == 1, (
        f"{_STEP_6!r} is no longer a unique anchor in AGENTS.md; the step 5 slice has no reliable end bound."
    )
    assert text.index(_STEP_5) < text.index(_STEP_6), (
        "PR Workflow step 6 now precedes step 5; the slice bounds are inverted."
    )


def test_reply_gate_is_in_the_same_breath_as_the_instruction() -> None:
    """The gate qualifies step 5, so it has to live inside step 5."""
    prose = _step_5_prose()
    assert _GATE in prose, (
        "AGENTS.md > PR Workflow step 5 no longer carries the reply gate "
        f"({_GATE!r}). Without it, 'address all review comments' is satisfied by "
        "replying to a thread already answered - the #1899 loop, where one "
        "thread took 12 consecutive replies all announcing the same commit."
    )
    # The gate must open step 5's prose, not trail it: a contributor who stops
    # reading after the numbered line should still hit it.
    assert prose.index(_GATE) < 200, (
        "The reply gate has drifted away from the head of PR Workflow step 5 "
        f"(now {prose.index(_GATE)} characters in). It qualifies the instruction "
        "on that line and is only read if it stays next to it."
    )


def test_gate_keeps_the_reason_it_exists() -> None:
    """Per-concern vs per-cycle is what makes the gate non-obvious."""
    prose = _step_5_prose()
    assert _PER_CONCERN in prose, (
        "AGENTS.md > PR Workflow step 5 no longer distinguishes a per-concern "
        "obligation from a per-cycle one. That distinction is the whole reason "
        "the gate is not self-evident: an agent rebuilding context each run "
        "reads an answered question as an open one."
    )


def test_both_do_not_reply_dispositions_survive_with_their_directive() -> None:
    """Naming the fields is not the rule; pairing them with a directive is."""
    prose = _step_5_prose()
    assert _OWN_REPLY_DIRECTIVE in prose, (
        "AGENTS.md > PR Workflow step 5 no longer says that a thread whose last "
        "comment is your own gets no further reply. This is the cheap, purely "
        "mechanical half of the gate - it needs no semantic comparison, just the "
        "author of the last comment - and on its own it would have prevented ten "
        "of the twelve replies on #1899."
    )
    assert _TERMINAL_STATE_DIRECTIVE in prose, (
        "AGENTS.md > PR Workflow step 5 no longer treats isResolved / isOutdated "
        "as terminal. Both fields are already in the context payload; on #1899 "
        "they were fetched and not read, and ten replies landed on a thread "
        "carrying both."
    )


def test_gate_does_not_read_as_never_reply() -> None:
    """The gate suppresses duplicates, not correspondence."""
    prose = _step_5_prose()
    assert _REPLY_ONCE in prose, (
        "AGENTS.md > PR Workflow step 5 no longer states the case where a reply "
        f"IS owed ({_REPLY_ONCE!r}). Without it the gate over-reads as 'do not "
        "reply to review threads', and a direct question from a reviewer goes "
        "unanswered - the opposite failure, and the more expensive one."
    )
