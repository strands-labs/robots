"""A locked-out arm must never render as a healthy one (Q43).

Measured 2026-08-20: both SO-101 arms had been e-stop locked for ten hours while
/api/fleet showed so101-arm-2 with six live joints, stale=false and no safety field of
any kind. The only representation of a lockout in the product was a five-second flash,
so a page reload erased it.
"""

from __future__ import annotations

from strands_robots.dashboard.safety_state import (
    Lockout,
    apply_event,
    note_command_accepted,
    proves_clear,
    resolve_peer,
)


class TestSeenEvents:
    def test_before_anything_is_seen_the_answer_is_unknown_not_clear(self) -> None:
        # The mesh does not advertise lockout state, so silence is not safety.
        assert Lockout().state == "unknown"
        assert "since this dashboard started" in Lockout().reason

    def test_an_estop_locks_and_names_who_sent_it(self) -> None:
        # the real incident: COORDINATOR_ID of examples/fleet/04_emergency_evacuation.py
        lock = apply_event(Lockout(), kind="estop", data={"source": "evac-coordinator", "t": 100.0}, now=200.0)
        assert lock.state == "locked"
        assert lock.by == "evac-coordinator"
        assert lock.since == 100.0
        assert "evac-coordinator" in lock.reason

    def test_an_estop_without_a_named_source_still_locks(self) -> None:
        lock = apply_event(Lockout(), kind="estop", data={}, now=50.0)
        assert lock.state == "locked" and lock.since == 50.0 and lock.by is None

    def test_a_resume_is_a_broadcast_not_a_receipt(self) -> None:
        # Each peer re-verifies the override code and MAY REFUSE. Painting the fleet
        # green here would be a claim about hardware that nobody checked.
        locked = apply_event(Lockout(), kind="estop", data={"source": "x"}, now=10.0)
        after = apply_event(locked, kind="resume", data={"source": "operator"}, now=20.0)
        assert after.state == "unknown", "a published resume is not proof any peer cleared"
        assert "verifies the override code itself" in after.reason

    def test_an_unrelated_event_kind_changes_nothing(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=1.0)
        assert apply_event(locked, kind="weather", data={}, now=2.0) == locked


class TestProof:
    def test_only_an_action_a_lockout_would_refuse_proves_anything(self) -> None:
        assert proves_clear("task") is True
        assert proves_clear("teleop_publish") is True
        assert proves_clear("status") is False, "a locked peer answers status - it proves nothing"
        assert proves_clear("resume") is False
        assert proves_clear("") is False

    def test_an_accepted_command_clears_the_verdict(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=10.0)
        proved = note_command_accepted(locked, now=30.0)
        assert proved.state == "clear"
        assert "accepted" in proved.reason

    def test_proof_older_than_the_estop_is_ignored(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=100.0)
        # a command accepted an hour before the e-stop says nothing about now
        assert resolve_peer(locked, first_seen=1.0, proof_at=50.0).state == "locked"

    def test_proof_after_the_estop_wins(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=100.0)
        assert resolve_peer(locked, first_seen=1.0, proof_at=150.0).state == "clear"

    def test_proof_after_a_resume_is_how_a_fleet_goes_green_again(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=10.0)
        resumed = apply_event(locked, kind="resume", data={}, now=20.0)
        assert resolve_peer(resumed, first_seen=1.0, proof_at=25.0).state == "clear"


class TestPerPeer:
    def test_a_peer_that_appeared_after_the_estop_is_unknown_not_locked(self) -> None:
        # A freshly spawned child has its own unset lockout flag; inheriting the fleet's
        # verdict would mark it red on no evidence.
        locked = apply_event(Lockout(), kind="estop", data={}, now=100.0)
        v = resolve_peer(locked, first_seen=150.0)
        assert v.state == "unknown"
        assert "appeared after the fleet e-stop" in v.reason

    def test_a_peer_present_before_the_estop_is_locked(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=100.0)
        assert resolve_peer(locked, first_seen=10.0).state == "locked"

    def test_an_unknown_first_seen_does_not_soften_a_lockout(self) -> None:
        locked = apply_event(Lockout(), kind="estop", data={}, now=100.0)
        assert resolve_peer(locked, first_seen=None).state == "locked"

    def test_the_verdict_always_carries_a_reason(self) -> None:
        for v in (
            Lockout(),
            apply_event(Lockout(), kind="estop", data={}, now=1.0),
            resolve_peer(apply_event(Lockout(), kind="estop", data={}, now=1.0), first_seen=2.0),
            note_command_accepted(Lockout(), now=1.0),
        ):
            fields = v.as_fields()
            assert fields["reason"], "a badge with no explanation sends the operator hunting"
            assert fields["state"] in {"locked", "clear", "unknown"}
