# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
"""A WebAuthn challenge's five-minute life is measured on a clock that cannot step.

``_stash_challenge`` stamps a challenge record and ``_pop_challenge`` reads that
stamp back to decide whether the ceremony is still open. Both readings are this
process's own, so the age between them is a *duration* -- local bookkeeping, and
the tree's thrice-settled boundary puts it on ``time.monotonic()``. It was
measured against ``time.time()``, which is not a clock but the current opinion
about the date, and one NTP correction, ``date -s`` or resume from suspend moved
all three decisions the stamp feeds. Driving the real store through a clock
double that takes one step, with the default 300s TTL::

    pop a challenge                              before        after
    10s old, no step (control)                  accepted    accepted
    301s old, no step (control)                  refused     refused
    301s old, wall clock steps -1h              accepted     refused
    1s old, wall clock steps +1h                 refused    accepted

    a 1s-old ceremony, next stash after a +1h step   swept out by the GC    kept
    the cap drops, entries either side of a -1h step       a newer entry  the oldest

Backwards, a challenge whose window has closed is accepted for the size of the
step: the TTL is the only thing bounding how long that nonce stays replayable.
Forwards, an operator mid-login is refused with "challenge expired", and the next
caller's ``_stash_challenge`` sweeps the record out of the table on the way in --
so one client's arriving request ends another's ceremony. The third row is the
per-client cap: ``_evict_oldest`` orders the pool by that stamp, and across a
backward step the entry stamped *after* the correction sorts oldest, so "drop the
oldest" drops a newer challenge and keeps a staler one -- inverting the fairness
property the cap exists for.

Neither direction was observable from the existing suite, because both cells that
covered expiry simulated age by rewriting the stored stamp
(``_challenges[cid]["t"] = time.time() - _CHAL_TTL - 1``) rather than by moving
the clock. A stamp rewritten in the code's own clock domain agrees with the code
whichever clock that is, so those cells were green either way; the cells here
move the clock instead and leave the store alone.

The other half of the boundary is pinned too: a session token's ``iat``/``exp``
are absolute stamps that a browser and ``renewal_verdict`` correlate, so they stay
on the wall clock and this module must not be swept clock-blind.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import jwt
import pytest
from fastapi import HTTPException

import strands_robots.dashboard.auth as auth


class SteppableClock:
    """A wall clock that can be corrected, beside a monotonic one that cannot.

    ``advance`` is real time passing: both clocks move together. ``step_wall`` is
    the correction -- an NTP slew, a ``date -s``, a resume from suspend -- and
    moves only :meth:`time`. Correct code reads :meth:`monotonic` and is
    therefore immune to any number of steps.
    """

    def __init__(self, *, wall_base: float = 1_700_000_000.0, mono_base: float = 1000.0) -> None:
        self._elapsed = 0.0
        self._offset = 0.0
        self._wall_base = wall_base
        self._mono_base = mono_base

    def advance(self, seconds: float) -> None:
        self._elapsed += seconds

    def step_wall(self, seconds: float) -> None:
        self._offset += seconds

    def monotonic(self) -> float:
        return self._mono_base + self._elapsed

    def time(self) -> float:
        return self._wall_base + self._elapsed + self._offset


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Iterator[SteppableClock]:
    """The real challenge store, reading the clock double, empty before and after."""
    double = SteppableClock()
    # Annotated Any: the real attribute is the `time` module, and the double
    # answers the two readings this module makes of it.
    stand_in: Any = double
    monkeypatch.setattr(auth, "time", stand_in)
    auth._challenges.clear()
    try:
        yield double
    finally:
        auth._challenges.clear()


def _pop(cid: str) -> str:
    """``"accepted"``, or the refusal the caller is given."""
    try:
        auth._pop_challenge(cid, "auth")
    except HTTPException as exc:
        return f"refused: {exc.detail}"
    return "accepted"


# ---------------------------------------------------------------------------
# The double itself, so that a clean run below means something
# ---------------------------------------------------------------------------
def test_the_double_advance_moves_both_clocks(clock: SteppableClock) -> None:
    wall, mono = clock.time(), clock.monotonic()
    clock.advance(7.0)
    assert clock.time() == pytest.approx(wall + 7.0)
    assert clock.monotonic() == pytest.approx(mono + 7.0)


@pytest.mark.parametrize("step", [30.0, 3600.0, -30.0, -3600.0])
def test_the_double_step_moves_only_the_wall_clock(clock: SteppableClock, step: float) -> None:
    wall, mono = clock.time(), clock.monotonic()
    clock.step_wall(step)
    assert clock.time() == pytest.approx(wall + step), "the double must really hand out the step"
    assert clock.monotonic() == pytest.approx(mono), "monotonic must not move on a wall-clock step"


# ---------------------------------------------------------------------------
# The TTL gate at pop
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("aged", "expected"),
    [(10.0, "accepted"), (auth._CHAL_TTL + 1.0, "refused: challenge expired")],
)
def test_the_ttl_is_enforced_when_the_clock_does_not_step(clock: SteppableClock, aged: float, expected: str) -> None:
    """Control: absent a correction, a challenge lives exactly its TTL."""
    cid = auth._stash_challenge("auth", b"operator", {}, ip="192.0.2.5")
    clock.advance(aged)
    assert _pop(cid) == expected


def test_a_live_challenge_still_pops_after_the_clock_jumps_forward(clock: SteppableClock) -> None:
    """A correction must not refuse the ceremony the operator is in the middle of."""
    cid = auth._stash_challenge("auth", b"operator", {}, ip="192.0.2.5")
    clock.advance(1.0)
    clock.step_wall(+3600.0)
    assert _pop(cid) == "accepted", (
        "a +1h wall-clock step expired a challenge one second old, so an operator "
        "mid-login is told the ceremony timed out"
    )


def test_an_expired_challenge_is_refused_after_the_clock_jumps_back(clock: SteppableClock) -> None:
    """A correction must not extend the window a nonce stays replayable in."""
    cid = auth._stash_challenge("auth", b"operator", {}, ip="192.0.2.5")
    clock.advance(auth._CHAL_TTL + 1.0)
    clock.step_wall(-3600.0)
    assert _pop(cid) == "refused: challenge expired", (
        "a -1h wall-clock step accepted a challenge past its TTL, extending the replay window by the size of the step"
    )


# ---------------------------------------------------------------------------
# The sweep the next stash performs on the way in
# ---------------------------------------------------------------------------
def test_the_stash_sweep_keeps_a_live_challenge_after_a_forward_step(clock: SteppableClock) -> None:
    """One caller's arriving request must not end another's ceremony."""
    mine = auth._stash_challenge("auth", b"operator", {}, ip="192.0.2.5")
    clock.advance(1.0)
    clock.step_wall(+3600.0)
    auth._stash_challenge("auth", b"other", {}, ip="198.51.100.7")
    assert mine in auth._challenges, (
        "after a +1h step the next stash swept out a challenge one second old, so "
        "an unrelated caller's request ended the operator's in-flight login"
    )


def test_the_stash_sweep_still_prunes_a_genuinely_expired_challenge(clock: SteppableClock) -> None:
    """Control: the sweep is still a sweep. Age simulated by moving the clock."""
    stale = auth._stash_challenge("auth", b"old", {}, ip="192.0.2.5")
    clock.advance(auth._CHAL_TTL + 1.0)
    auth._stash_challenge("auth", b"new", {}, ip="192.0.2.6")
    assert stale not in auth._challenges, "an expired challenge must not survive the next stash"


# ---------------------------------------------------------------------------
# Which entry the cap evicts
# ---------------------------------------------------------------------------
def test_the_cap_evicts_the_challenge_stashed_first_across_a_backward_step(
    clock: SteppableClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ "Drop the oldest" must mean the one stashed first, however the date moves."""
    monkeypatch.setattr(auth, "_CHAL_MAX", 3)
    monkeypatch.setattr(auth, "_CHAL_MAX_PER_IP", 100)  # isolate the global cap
    first = auth._stash_challenge("auth", b"first", {}, ip="192.0.2.5")
    clock.advance(1.0)
    clock.step_wall(-3600.0)
    second = auth._stash_challenge("auth", b"second", {}, ip="192.0.2.6")
    clock.advance(1.0)
    third = auth._stash_challenge("auth", b"third", {}, ip="192.0.2.7")
    clock.advance(1.0)
    auth._stash_challenge("auth", b"trigger", {}, ip="192.0.2.8")  # crosses the cap
    dropped = [
        name for name, cid in (("first", first), ("second", second), ("third", third)) if cid not in auth._challenges
    ]
    assert dropped == ["first"], (
        f"the cap dropped {dropped or ['nothing']}: across a -1h step an entry stashed "
        "after the correction sorts oldest, so a newer challenge is evicted and a "
        "staler one survives"
    )


# ---------------------------------------------------------------------------
# The other half of the boundary: an absolute stamp is not a duration
# ---------------------------------------------------------------------------
def test_a_session_tokens_stamps_stay_on_the_wall_clock(clock: SteppableClock, monkeypatch: pytest.MonkeyPatch) -> None:
    """``iat``/``exp`` name a date a browser reads, so they must not move to monotonic.

    This fails if the module is swept clock-blind rather than per value: seconds of
    process uptime in ``exp`` is a token that expired 55 years ago.
    """
    secret = "a-secret-long-enough-for-HS256-in-this-test"
    monkeypatch.setattr(auth, "_jwt_secret", lambda: secret)
    token = auth.issue_token("operator")
    # verify_exp off: the assertions below are about which clock stamped the
    # token, and the double's date is deliberately not today's.
    claims = jwt.decode(token, secret, algorithms=["HS256"], options={"verify_exp": False})
    assert claims["iat"] == int(clock.time()), (
        f"a token's iat is {claims['iat']}, not the wall clock's {int(clock.time())}"
    )
    assert claims["iat"] != int(clock.monotonic()), "iat must not be seconds of process uptime"
    assert claims["exp"] > claims["iat"], "a freshly minted token must not already be expired"
