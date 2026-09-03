"""Prerequisites an operator must satisfy to recover a fleet from an e-stop.

``Mesh.emergency_stop`` latches a lockout on every peer that receives it, and
nothing clears it on a timer - an e-stop that expired by itself would not be an
e-stop. The only way back is an explicit ``resume``, so every precondition that
resume depends on is a precondition for recovering the fleet at all.

Two of those preconditions are invisible until the fleet is already locked out:

1. ``STRANDS_MESH_OVERRIDE_CODE`` must be set on every peer. The mesh already
   logs a WARNING at startup when it is unset.
2. Fleet clocks must agree. A resume envelope carries the issuer's wall clock and
   a receiver refuses one that is stale (older than
   ``STRANDS_MESH_RESUME_FRESHNESS_S``) or future-dated (more than
   ``STRANDS_MESH_RESUME_FORWARD_SKEW_S`` ahead). The forward bound is the tight
   one and the asymmetry is the trap: a receiver whose clock is a few seconds
   *behind* the operator reads a correct, correctly-signed resume as future-dated
   and refuses it, and every retry fails identically. Nothing in the process
   recovers from that - the fleet stays stopped until the clock is corrected or
   the bound is widened on every peer.

The clock precondition has no startup warning, so documentation is the only place
an operator can learn it before it matters. These tests pin the behaviour and pin
that the knobs it depends on are documented, so the numbers in the docs cannot
drift away from the numbers the receiver enforces.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

import strands_robots
from strands_robots.mesh import core

_REPO_ROOT = Path(strands_robots.__file__).resolve().parent.parent

#: Long enough to be a realistic operator secret rather than a crackable PIN.
_CODE = "operator-code-1234567890abcdef"


def _mesh(peer_id: str) -> Any:
    """A Mesh wired for the safety handlers without joining a real session."""
    m = core.Mesh.__new__(core.Mesh)
    core.Mesh.__init__(m, MagicMock(), peer_id)
    m.publish_safety_event = lambda **kw: None  # type: ignore[method-assign]
    return m


def _mint_resume(*, issuer_clock_offset_s: float = 0.0) -> dict[str, Any]:
    """Return the resume envelope the real issuer publishes.

    ``issuer_clock_offset_s`` models an operator whose wall clock runs ahead of
    the receiver's, which is the same relative skew as a receiver running behind.
    The offset is applied only while the issuer mints the envelope, so the
    receiver under test always runs on the real clock.
    """
    operator = _mesh("operator-1")
    captured: dict[str, Any] = {}
    operator._publish_safety_envelope = lambda key, env: captured.update(env)
    operator._estop_lockout.set()
    operator._last_estop_ts = time.time() - 3.0
    operator._last_estop_mono = time.monotonic() - 3.0
    real_time = time.time
    with patch.object(core.time, "time", lambda: real_time() + issuer_clock_offset_s):
        result = operator._resume_lockout(_CODE)
    assert result["status"] == "ok", result
    assert captured, "the issuer published no resume envelope"
    return captured


def _deliver(envelope: dict[str, Any]) -> bool:
    """Deliver *envelope* to a locked-out receiver; True if it recovered."""
    robot = _mesh("robot-1")
    robot._estop_lockout.set()
    sample = MagicMock()
    sample.payload.to_bytes.return_value = json.dumps(envelope).encode()
    robot._on_safety_resume(sample)
    return not robot._estop_lockout.is_set()


class TestClockSkewBlocksEstopRecovery:
    """A receiver behind the operator refuses a resume it should honour."""

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    def test_a_receiver_within_the_forward_bound_recovers(self) -> None:
        """Control: with clocks in sync the correct code clears the lockout."""
        assert _deliver(_mint_resume()) is True

    def test_a_receiver_seconds_behind_the_operator_stays_locked_out(self, caplog: pytest.LogCaptureFixture) -> None:
        """One second past the forward bound is enough to refuse recovery."""
        skew = core._resume_forward_skew_s() + 1.0
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=skew))
        assert recovered is False, (
            f"a receiver {skew:.0f}s behind the operator accepted the resume; "
            "the forward-skew bound is what makes this refusal happen"
        )
        assert any("in future" in r.message for r in caplog.records), (
            "the refusal must say the envelope looked future-dated so an "
            f"operator can tell a clock problem from a bad code: {caplog.text}"
        )

    def test_retrying_does_not_help_because_every_envelope_is_refused(self) -> None:
        """The failure is not transient - a retry loop cannot recover the fleet."""
        skew = core._resume_forward_skew_s() + 1.0
        assert [_deliver(_mint_resume(issuer_clock_offset_s=skew)) for _ in range(3)] == [
            False,
            False,
            False,
        ]

    def test_widening_the_documented_bound_restores_recovery(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The documented remedy works: raise the bound and the same skew passes.

        This is what makes the knob worth documenting - it is the only in-process
        way out of a skew-induced lockout.
        """
        skew = core._resume_forward_skew_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FORWARD_SKEW_S", str(skew + 10.0))
        assert _deliver(_mint_resume(issuer_clock_offset_s=skew)) is True


class TestTheFreshnessBoundGovernsAReceiverAheadOfTheOperator:
    """The two bounds are not interchangeable: each governs one clock direction.

    ``_on_safety_resume`` refuses on ``envelope_t > now + forward_skew_s``
    (the envelope reads future-dated, which happens when the receiver's clock
    trails the operator's) and separately on ``now - envelope_t >
    freshness_window_s`` (the envelope reads stale, which happens when the
    receiver's clock *leads* the operator's). Documenting either bound against
    the wrong direction hands a locked-out operator a knob that cannot clear
    the refusal they are looking at, so the direction is pinned here.
    """

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    def test_a_receiver_ahead_of_the_operator_is_refused_as_stale(self, caplog: pytest.LogCaptureFixture) -> None:
        """A negative issuer offset models a receiver whose clock leads."""
        lead = core._resume_freshness_window_s() + 1.0
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=-lead))
        assert recovered is False, (
            f"a receiver {lead:.0f}s ahead of the operator accepted the resume; "
            "the freshness window is what makes this refusal happen"
        )
        assert any("too old" in r.message for r in caplog.records), (
            "the refusal must say the envelope looked stale, which is the "
            f"direction STRANDS_MESH_RESUME_FRESHNESS_S governs: {caplog.text}"
        )

    def test_widening_the_freshness_window_recovers_a_receiver_that_leads(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The documented remedy clears the refusal for the direction it names."""
        lead = core._resume_freshness_window_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", str(lead + 10.0))
        assert _deliver(_mint_resume(issuer_clock_offset_s=-lead)) is True

    def test_widening_the_freshness_window_does_not_recover_a_receiver_that_trails(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Why the direction matters: the wrong knob leaves the fleet stopped.

        A receiver *behind* the operator is refused by the forward-skew bound,
        so raising the freshness window -- however far -- changes nothing. This
        is the concrete cost of describing the freshness bound as the one that
        governs a trailing receiver.
        """
        trail = core._resume_forward_skew_s() + 1.0
        monkeypatch.setenv("STRANDS_MESH_RESUME_FRESHNESS_S", "300")
        with caplog.at_level("WARNING"):
            recovered = _deliver(_mint_resume(issuer_clock_offset_s=trail))
        assert recovered is False, (
            "widening the freshness window recovered a trailing receiver; if "
            "that ever becomes true the README remedy for each direction changes"
        )
        assert any("in future" in r.message for r in caplog.records), (
            "a trailing receiver must still be refused by the forward-skew "
            f"bound, not the freshness window: {caplog.text}"
        )


def _env_table_rows() -> list[tuple[str, str]]:
    """Return ``(name_cell, description_cell)`` for every env-var README row."""
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    rows = []
    for name_cell, desc_cell, _default in re.findall(r"^\|\s*(.+?)\s*\|(.*)\|(.*)\|\s*$", readme, re.M):
        if re.search(r"`(?:STRANDS|ZENOH)_[A-Z0-9_]+`", name_cell):
            rows.append((name_cell, desc_cell))
    return rows


class TestTheRecoveryKnobsAreDocumented:
    """The knobs a locked-out operator needs must be findable."""

    def test_every_knob_the_resume_freshness_gate_consults_is_documented(self) -> None:
        """Each bound the receiver enforces needs a row of its own.

        A knob that only exists in the source cannot be reached for by an
        operator whose fleet is already refusing every resume.
        """
        documented = {
            name for name_cell, _desc in _env_table_rows() for name in re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)
        }
        assert documented, "found no env-var rows in README.md; the scan is broken"
        required = (
            "STRANDS_MESH_OVERRIDE_CODE",
            "STRANDS_MESH_RESUME_FRESHNESS_S",
            "STRANDS_MESH_RESUME_FORWARD_SKEW_S",
        )
        missing = [name for name in required if name not in documented]
        assert not missing, (
            f"these govern whether a resume is accepted but have no README row: {missing}. "
            "An operator can only discover them once the fleet is already locked out."
        )

    def test_the_env_table_names_no_variable_it_never_lists(self) -> None:
        """A row that cites another variable implies the reader can look it up.

        Citing a name the table never lists sends the reader looking for a row
        that does not exist, which is worst on a safety knob a locked-out
        operator is trying to reach.
        """
        rows = _env_table_rows()
        assert rows, "found no env-var rows in README.md; the scan is broken"
        listed = {name for name_cell, _desc in rows for name in re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)}
        dangling = sorted(
            {
                (re.findall(r"`([A-Z_][A-Z0-9_]*)`", name_cell)[0], cited)
                for name_cell, desc in rows
                for cited in re.findall(r"`((?:STRANDS|ZENOH)_[A-Z0-9_]+)`", desc)
                if cited not in listed
            }
        )
        assert not dangling, (
            f"these env vars are cited in a description but have no row of their own (cited_by, missing): {dangling}"
        )

    def test_each_timestamp_bound_is_documented_against_the_clock_it_governs(self) -> None:
        """The rows must not swap the two directions.

        Behaviour tests above pin which bound refuses which skew; nothing
        otherwise ties that to the prose an operator actually reads, so a row
        can invert while the suite stays green.
        """
        rows = dict(_env_table_rows())
        assert rows, "found no env-var rows in README.md; the scan is broken"

        def _row_for(name: str) -> str:
            matches = [desc for name_cell, desc in rows.items() if f"`{name}`" in name_cell]
            assert matches, f"premise: README has a row for {name}"
            return matches[0]

        freshness = _row_for("STRANDS_MESH_RESUME_FRESHNESS_S")
        lockout_claim = freshness.split("stays locked out")[0]
        assert "ahead of* the operator" in lockout_claim, (
            "the freshness row must attribute the stale-envelope lockout to a "
            "receiver whose clock is AHEAD of the operator -- that is the "
            f"direction `now - envelope_t > freshness_window_s` trips on: {freshness!r}"
        )

        skew = _row_for("STRANDS_MESH_RESUME_FORWARD_SKEW_S")
        assert "*behind* the operator" in skew, (
            "the forward-skew row must attribute its lockout to a receiver "
            f"whose clock is BEHIND the operator: {skew!r}"
        )

    def test_the_recovery_procedure_is_documented_beside_the_estop_call(self) -> None:
        """``docs/mesh.md`` shows ``emergency_stop()``; it must show the way back."""
        mesh_doc = (_REPO_ROOT / "docs" / "mesh.md").read_text(encoding="utf-8")
        assert "emergency_stop()" in mesh_doc, "premise: mesh.md documents emergency_stop"
        assert '"action": "resume"' in mesh_doc, "mesh.md documents how to stop a fleet but not how to resume it"
        for knob in ("STRANDS_MESH_OVERRIDE_CODE", "STRANDS_MESH_RESUME_FORWARD_SKEW_S"):
            assert knob in mesh_doc, f"mesh.md's recovery guidance omits {knob}"


#: Skew directions a receiver's clock can carry relative to the operator's, and
#: the ``issuer_clock_offset_s`` sign :func:`_mint_resume` needs for each. A
#: receiver *behind* the operator sees an envelope stamped in its own future, so
#: the issuer's clock runs ahead; a receiver *ahead* sees one already past.
_SKEW_SIGN = {"ahead": -1.0, "behind": +1.0}

#: The two bounds a skewed resume envelope can trip.
_BOUNDS = (
    "STRANDS_MESH_RESUME_FRESHNESS_S",
    "STRANDS_MESH_RESUME_FORWARD_SKEW_S",
)

#: Skew (seconds) past both defaults, so a receiver at this offset is refused
#: until the bound governing its own direction is widened.
_SKEW_S = 61.0


def _recovers_when_widened(knob: str, direction: str, monkeypatch: pytest.MonkeyPatch) -> bool:
    """Widen *knob*, deliver a resume to a receiver skewed *direction*; recovered?"""
    monkeypatch.setenv(knob, str(_SKEW_S + 30.0))
    try:
        return _deliver(_mint_resume(issuer_clock_offset_s=_SKEW_SIGN[direction] * _SKEW_S))
    finally:
        monkeypatch.delenv(knob, raising=False)


def _bound_governing(direction: str, monkeypatch: pytest.MonkeyPatch) -> str:
    """The one bound in :data:`_BOUNDS` whose widening recovers *direction*.

    Measured against the shipped receiver rather than recorded here, so if which
    bound catches which direction ever changes, the rows are re-graded against
    the new behaviour instead of staying pinned to a claim that has gone stale.
    """
    clearing = [knob for knob in _BOUNDS if _recovers_when_widened(knob, direction, monkeypatch)]
    assert len(clearing) == 1, (
        f"expected exactly one bound to clear a receiver {direction} the operator, got {clearing}. "
        "The rows describe a one-bound-per-direction split; if that is no longer how the receiver "
        "behaves the rows need rewriting, not this assertion relaxing."
    )
    return clearing[0]


def _own_direction_claim(knob: str) -> str:
    """The skew direction *knob*'s own README row says *it* governs.

    Anchored on ``more than this <direction>``, where ``this`` is the row's own
    bound. Each row also names its sibling for the opposite direction; reading
    only the ``this`` clause keeps that cross-reference from being mistaken for a
    claim about this bound.
    """
    rows = [desc for name_cell, desc in _env_table_rows() if f"`{knob}`" in name_cell]
    assert len(rows) == 1, f"premise: expected exactly one README row for {knob}, got {len(rows)}"
    match = re.search(r"more than this \*(ahead of|behind)\*", rows[0])
    assert match, (
        f"premise: {knob}'s README row no longer says what a clock 'more than this "
        f"<ahead of|behind> the operator' does, so the direction it claims can no "
        f"longer be read: {rows[0].strip()!r}"
    )
    return match.group(1).removesuffix(" of")


class TestTheDocumentedDirectionIsGradedAgainstTheReceiver:
    """Grade both rows against the receiver rather than against recorded text.

    ``test_each_timestamp_bound_is_documented_against_the_clock_it_governs``
    pins each row's direction word to the direction that bound catches today.
    That catches a row being edited to the wrong direction, but not the mirror
    case: swap which bound the receiver applies to which sign of skew and the
    pinned words silently become wrong again, in the other direction, with the
    suite green. Deriving both sides means neither can drift alone.
    """

    @pytest.fixture(autouse=True)
    def _code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", _CODE)

    @pytest.mark.parametrize("knob", _BOUNDS)
    def test_each_row_names_the_direction_its_own_bound_governs(
        self, knob: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The direction a row claims must be the one its bound really catches."""
        documented = _own_direction_claim(knob)
        enforced = _bound_governing(documented, monkeypatch)
        assert enforced == knob, (
            f"{knob}'s README row says it governs a receiver {documented} the operator, but a "
            f"receiver {documented} the operator is refused until {enforced} is widened - widening "
            f"{knob} leaves it locked out. Name the direction this bound catches."
        )

    @pytest.mark.parametrize("knob", _BOUNDS)
    def test_widening_the_bound_a_row_names_clears_the_skew_it_names(
        self, knob: str, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Follow each row literally: widen its bound, in the case it describes.

        Both rows now tell a locked-out operator to widen the bound for the skew
        they are looking at. Executing that instruction grades the remedy rather
        than the wording, so a row can be refused for prescribing a knob that
        does not clear the refusal it names even if its direction word is right.
        """
        documented = _own_direction_claim(knob)
        assert _recovers_when_widened(knob, documented, monkeypatch) is True, (
            f"{knob}'s README row prescribes widening it for a receiver {documented} the "
            f"operator, but doing exactly that left the receiver locked out. A remedy that "
            "does not clear the refusal it names is worse than none - the fleet is already stopped."
        )
