"""Smoke test for examples/fleet/03_failover_and_degraded_ops.py (issue #2182).

Part 1 (robot failure mid-task) drives the deterministic failover core -
presence-timeout detection, rollout bookkeeping closure, capability-matched
reassignment, and the no-survivor / robot-alive-but-failing verdicts -
through stub presence and transport seams: no simulator, no Zenoh session,
no network.

Part 2 (dispatcher failure) asserts the estop-during-dispatcher-outage claim
rather than narrating it: two real ``Mesh`` peers exchange the real
``strands/safety/estop`` envelope (captured from the issuer's own publish
path and delivered to the receiver's real subscriber handler) with no
dispatcher object in existence, and the receiver's dispatch refuses
``execute`` while still answering ``status``. The restarted orchestrator's
re-sync is then asserted against the audit records that same drill genuinely
wrote.

The audit log is redirected to ``tmp_path`` (never the developer's real
``~/.strands_robots/mesh_audit.jsonl``) and signed with a test PSK so
``verify_audit_integrity`` attests the trail end to end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import security as mesh_security

_FLEET_DIR = Path(__file__).resolve().parent.parent / "examples" / "fleet"
_EXAMPLE_PATH = _FLEET_DIR / "03_failover_and_degraded_ops.py"

# Loaded under a distinctive module name: "capabilities" (its sibling import)
# is generic enough to collide, so both are evicted between loads.
_MODULE_NAME = "fleet_failover_and_degraded_ops_example"


@pytest.fixture
def example(monkeypatch, tmp_path):
    """Load the example with the audit log confined to tmp_path and signed."""
    from strands_robots.mesh import audit

    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "smoke-test-psk")
    # Reset the process-global audit state (same isolation as
    # tests/mesh/test_audit_integrity.py): the PSK fingerprint and sequence
    # counters are one-shot per process, so records written by earlier tests
    # in the suite would otherwise poison this test's fresh, signed log.
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None
    monkeypatch.syspath_prepend(str(_FLEET_DIR))
    for name in (_MODULE_NAME, "capabilities"):
        sys.modules.pop(name, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _EXAMPLE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    yield mod
    for name in (_MODULE_NAME, "capabilities"):
        sys.modules.pop(name, None)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


class _StubFleet:
    """Presence + transport stubs with a per-robot kill switch.

    Unlike the example's ``ScriptedFleet`` (which the dry-run tests below use
    as shipped), this stub can keep a dead robot *fresh in presence* so the
    reply-failure-then-confirm path is exercised: ``kill(robot,
    linger_in_presence=True)`` models the window between the last heartbeat
    and the presence timeout.
    """

    def __init__(self, robots):
        self._alive = set(robots)
        self._lingering = set()
        self.sends = []

    def kill(self, robot, *, linger_in_presence=False):
        self._alive.discard(robot)
        if linger_in_presence:
            self._lingering.add(robot)

    def expire(self, robot):
        self._lingering.discard(robot)

    def get_peer(self, robot):
        if robot in self._alive or robot in self._lingering:
            return {"peer_id": robot, "type": "sim", "age": 0.5}
        return None

    def send(self, robot, instruction, n_steps):
        self.sends.append((robot, instruction))
        if robot not in self._alive:
            return {"status": "timeout"}
        return {"type": "response", "result": {"status": "success"}}


def _audit_records(example):
    from strands_robots.mesh.audit import read_audit_log

    return [r for r in read_audit_log() if isinstance(r.get("payload"), dict)]


def _orchestrator_chain(example, task_id):
    return [
        r["event"]
        for r in _audit_records(example)
        if r.get("peer_id") == example.ORCHESTRATOR_ID and r["payload"].get("task_id") == task_id
    ]


def test_failover_reassigns_remaining_legs_without_operator_code_changes(example):
    """A mid-task heartbeat death re-dispatches the remainder to a capable peer.

    The whole drill goes through ``run_task_with_failover`` exactly as the
    live path calls it - the reassignment is the function's own doing, not
    operator code reacting to the loss.
    """
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = _StubFleet(list(example.ROBOT_EMBODIMENT))

    summary = example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        sleep=lambda _s: None,
        on_leg_done=lambda robot, leg: fleet.kill(robot) if leg == 1 else None,
    )

    assert summary["status"] == "done"
    assert summary["reassignments"] == [{"from": "go2-a1", "to": "lekiwi-a1", "at_leg": 2}]
    # Rollout bookkeeping: the lost robot's entry is closed as interrupted
    # with only leg 1 done; the failover entry carries the remainder.
    interrupted, completed = summary["book"]
    assert interrupted == {"robot": "go2-a1", "legs_done": [1], "status": "interrupted"}
    assert completed == {"robot": "lekiwi-a1", "legs_done": [2, 3, 4], "status": "done"}
    # The audit chain tells the same story, task_id-threaded.
    assert _orchestrator_chain(example, "T-42") == [
        "task_dispatch",
        "task_leg_done",
        "peer_lost",
        "task_reassigned",
        "task_leg_done",
        "task_leg_done",
        "task_leg_done",
        "task_completed",
    ]


def test_reassignment_is_a_capability_match_not_a_name_match(example):
    """The surviving arm is never chosen: it fails the skill constraint.

    With both transport robots dead, so101-a1 is still alive and fresh in
    presence - a name-match (or an any-alive-robot match) would hand it the
    task. The capability filter refuses it with a machine-readable reason
    instead, and the task fails structurally rather than guessing.
    """
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = _StubFleet(list(example.ROBOT_EMBODIMENT))

    def kill_transporters(robot, leg):
        if leg == 1:
            fleet.kill("go2-a1")
        elif leg == 2:
            fleet.kill("lekiwi-a1")

    summary = example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        sleep=lambda _s: None,
        on_leg_done=kill_transporters,
    )

    assert summary["status"] == "failed"
    assert summary["failure"]["code"] == "no_feasible_robot_after_loss"
    rejections = {r["robot"]: r for r in summary["failure"]["rejections"]}
    assert rejections["so101-a1"]["constraint"] == "skill"
    # so101-a1 was alive throughout and never received a single leg.
    assert not any(robot == "so101-a1" for robot, _ in fleet.sends)


def test_reply_failure_from_an_alive_peer_is_a_task_failure_not_a_failover(example):
    """Presence is the failover trigger, not a failed reply on its own.

    The executor stays fresh in presence while its reply times out;
    ``confirm_peer_loss`` polls out the full window, finds the heartbeat
    still there, and the verdict is a structured dispatch failure with no
    reassignment - a misbehaving robot must not shed its task to a peer.
    """
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = _StubFleet(list(example.ROBOT_EMBODIMENT))
    fleet.kill("go2-a1", linger_in_presence=True)  # deaf on the wire, fresh in presence

    clock = {"now": 0.0}

    def fake_clock():
        return clock["now"]

    def fake_sleep(s):
        clock["now"] += s

    summary = example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        clock=fake_clock,
        sleep=fake_sleep,
    )

    assert summary["status"] == "failed"
    assert summary["failure"]["code"] == "dispatch_failed"
    assert summary["reassignments"] == []
    assert "task_reassigned" not in _orchestrator_chain(example, "T-42")


def test_death_between_heartbeat_and_timeout_is_confirmed_then_failed_over(example):
    """A robot dead mid-leg but not yet pruned still fails over, once confirmed.

    Presence stays fresh when leg 2's reply times out (the timeout window has
    not elapsed); the confirm loop keeps polling and the failover happens the
    moment the presence timeout is observed - the leg is then re-dispatched,
    so delivery is at-least-once.
    """
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = _StubFleet(list(example.ROBOT_EMBODIMENT))

    clock = {"now": 0.0}

    def fake_sleep(s):
        clock["now"] += s
        # Two polls in, the heartbeat crosses the presence timeout.
        if clock["now"] >= 2.0:
            fleet.expire("go2-a1")

    summary = example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        clock=lambda: clock["now"],
        sleep=fake_sleep,
        on_leg_done=lambda robot, leg: fleet.kill(robot, linger_in_presence=True) if leg == 1 else None,
    )

    assert summary["status"] == "done"
    assert summary["reassignments"] == [{"from": "go2-a1", "to": "lekiwi-a1", "at_leg": 2}]
    # Leg 2 was dispatched twice: once to the dying robot (timeout), once to
    # the failover robot after confirmation.
    leg2_targets = [robot for robot, instruction in fleet.sends if "leg 2/4" in instruction]
    assert leg2_targets == ["go2-a1", "lekiwi-a1"]


def test_dry_run_scripted_fleet_completes_the_failover_as_shipped(example):
    """The shipped --dry-run seams produce the same reassignment verdict."""
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = example.ScriptedFleet(list(example.ROBOT_EMBODIMENT))

    summary = example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        sleep=lambda _s: None,
        on_leg_done=lambda robot, leg: fleet.kill(robot) if leg == 1 else None,
    )

    assert summary["status"] == "done"
    assert summary["reassignments"] == [{"from": "go2-a1", "to": "lekiwi-a1", "at_leg": 2}]


# Part 2 -- the estop-during-dispatcher-outage claim, asserted.


class _StoppableRobot:
    """Mesh owner whose stop_task records the stop (so the broadcast is honest)."""

    def __init__(self):
        self.stopped = False

    def stop_task(self):
        self.stopped = True
        return {"ok": True, "status": "stopped"}


def _capturing_bus(monkeypatch):
    """Capture every payload the mesh publishes, keyed by topic."""
    published = []
    monkeypatch.setattr(mesh_core, "put", lambda key, payload: published.append((key, payload)))
    return published


def _as_sample(payload: dict) -> SimpleNamespace:
    """Wrap a published body the way the Zenoh subscriber hands it over."""
    raw = json.dumps(payload).encode()
    return SimpleNamespace(payload=SimpleNamespace(to_bytes=lambda r=raw: r))


def _live_unstarted_mesh(peer_id: str) -> mesh_core.Mesh:
    """A real Mesh peer minus the Zenoh transport.

    ``_running`` is flipped on so the safety paths behave exactly as on a
    started peer (publishing and audit-logging instead of early-returning);
    ``_stop_event`` is pre-set so the broadcast collection window returns
    immediately instead of sleeping out its full timeout. No session is ever
    opened - every publish lands in the captured bus.
    """
    mesh = mesh_core.Mesh(_StoppableRobot(), peer_id=peer_id, peer_type="robot")
    mesh._running = True
    mesh._stop_event.set()
    return mesh


def test_estop_propagates_peer_to_peer_with_no_dispatcher_anywhere(example, monkeypatch):
    """One robot's emergency_stop locks out another with no dispatcher extant.

    The issuer's real ``emergency_stop`` publish is captured off the bus and
    delivered to the receiver's real ``strands/safety/estop`` subscriber
    handler - the exact peer-to-peer path a live mesh uses. No orchestrator
    or dispatcher object exists in this test at all, so nothing about the
    propagation can depend on one.
    """
    published = _capturing_bus(monkeypatch)
    issuer = _live_unstarted_mesh("lekiwi-a1")
    receiver = _live_unstarted_mesh("so101-a1")

    # Before the estop the receiver accepts work.
    assert receiver._dispatch({"action": "status"}) is not None

    issuer.emergency_stop()

    estop_envelopes = [payload for key, payload in published if key == "strands/safety/estop"]
    assert len(estop_envelopes) == 1
    assert estop_envelopes[0]["peer_id"] == "lekiwi-a1"

    receiver._on_safety_estop(_as_sample(estop_envelopes[0]))

    # The receiver still answers status (local loop alive) and refuses
    # execute (lockout engaged) - the wire-observable degraded-ops contract.
    assert isinstance(receiver._dispatch({"action": "status"}), dict)
    with pytest.raises(mesh_security.LockoutError):
        receiver._dispatch({"action": "execute", "instruction": "probe: must be refused"})


def test_restarted_orchestrator_resyncs_the_outage_from_presence_and_audit(example, monkeypatch):
    """Re-sync reconstructs the dispatcher-down estop from what survived it.

    The audit records fed to ``resync_after_restart`` are the ones the real
    safety handlers wrote during a dispatcher-free estop exchange - not
    hand-written fixtures - plus the orchestrator's own pre-outage task
    trail. The reconstruction names the issuer, the engaged peers, and the
    task chain, and the missing robot is visible as absent from presence.
    """
    from strands_robots.mesh.audit import read_audit_log

    # Pre-outage: the failover drill writes the orchestrator's task trail.
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    fleet = _StubFleet(list(example.ROBOT_EMBODIMENT))
    run_start = time.time()
    example.run_task_with_failover(
        example.TRANSPORT_TASK,
        manifests,
        fleet.get_peer,
        fleet.send,
        sleep=lambda _s: None,
        on_leg_done=lambda robot, leg: fleet.kill(robot) if leg == 1 else None,
    )

    # Outage: dispatcher dead (no orchestrator peer exists here), estop
    # propagates peer to peer through the real handlers, which audit it.
    published = _capturing_bus(monkeypatch)
    issuer = _live_unstarted_mesh("lekiwi-a1")
    receiver = _live_unstarted_mesh("so101-a1")
    issuer.emergency_stop()
    envelope = next(payload for key, payload in published if key == "strands/safety/estop")
    receiver._on_safety_estop(_as_sample(envelope))

    # Restart: presence shows the survivors; the audit log tells the story.
    resync = example.resync_after_restart(
        [fleet.get_peer(robot) for robot in ("lekiwi-a1", "so101-a1")],
        read_audit_log(since=run_start - 1.0),
    )

    assert resync["alive_peers"] == ["lekiwi-a1", "so101-a1"]  # go2-a1 is gone
    assert resync["estop_issuer"] == "lekiwi-a1"
    assert resync["lockouts"] == {"lekiwi-a1": True, "so101-a1": True}
    assert resync["tasks"]["T-42"][0] == "task_dispatch"
    assert "task_reassigned" in resync["tasks"]["T-42"]
    assert resync["tasks"]["T-42"][-1] == "task_completed"


def test_degraded_safety_assertion_refuses_a_peer_that_accepts_execute(example):
    """assert_degraded_safety raises when a robot's lockout did not engage."""

    def locked_out_probe(robot, cmd):
        if cmd["action"] == "status":
            return {"type": "response", "result": {"status": "idle"}}
        return {"type": "error", "error": "command rejected"}

    verdicts = example.assert_degraded_safety(locked_out_probe, ["lekiwi-a1", "so101-a1"])
    assert verdicts == {
        "lekiwi-a1": {"status_answered": True, "execute_refused": True},
        "so101-a1": {"status_answered": True, "execute_refused": True},
    }

    def broken_lockout_probe(robot, cmd):
        if cmd["action"] == "status":
            return {"type": "response", "result": {"status": "idle"}}
        return {"type": "response", "result": {"status": "success"}}  # accepted!

    with pytest.raises(RuntimeError, match="accepted execute during lockout"):
        example.assert_degraded_safety(broken_lockout_probe, ["lekiwi-a1"])

    def dead_robot_probe(robot, cmd):
        return {"status": "timeout"}

    with pytest.raises(RuntimeError, match="did not answer status"):
        example.assert_degraded_safety(dead_robot_probe, ["so101-a1"])


# --- Part 2 re-sync: the audit evidence is awaited, not sampled ----------
#
# A peer's own ``remote_estop_engaged`` row is written by that peer's safety
# handler, on its callback thread, after the estop broadcast reaches it. The
# issuer's ``emergency_stop()`` returns before that write lands and cannot
# make it synchronous, so a single point-in-time read of the audit log can
# reconstruct the issuer's lockout and miss the survivor's. These pin that
# the drill awaits the evidence on a bounded monotonic clock instead.

_ISSUER_ROWS = [
    {"peer_id": "fleet-orchestrator", "event": "task_dispatch", "payload": {"task_id": "T-42"}},
    {"peer_id": "lekiwi-a1", "event": "emergency_stop", "payload": {}},
]
_SURVIVOR_ROW = {"peer_id": "so101-a1", "event": "remote_estop_engaged", "payload": {}}
_PEERS = [{"peer_id": "lekiwi-a1"}, {"peer_id": "so101-a1"}]


def _reader(*reads):
    """A read_records seam returning each supplied trail in turn, then the last."""
    calls = []

    def read():
        trail = reads[min(len(calls), len(reads) - 1)]
        calls.append(len(trail))
        return list(trail)

    return read, calls


def test_a_survivor_row_that_lands_after_the_read_is_awaited_not_missed(example):
    """The lockout is recovered even when its row lands after the first read.

    Both halves run against the same trail: reduced from the first read alone
    the survivor has no lockout, which is exactly the drill's failure mode;
    awaited, the reconstruction recovers it.
    """
    read, calls = _reader(_ISSUER_ROWS, _ISSUER_ROWS + [_SURVIVOR_ROW])
    slept = []

    # A single point-in-time read reduces to the issuer's lockout only.
    sampled = example.resync_after_restart(_PEERS, _ISSUER_ROWS)
    assert sampled["lockouts"] == {"lekiwi-a1": True}
    assert not any(sampled["lockouts"].get(robot) for robot in ["so101-a1"])

    resync, records = example.resync_until_lockout_recovered(
        read, _PEERS, ["so101-a1"], poll_s=0.01, sleep=slept.append
    )

    assert resync["lockouts"] == {"lekiwi-a1": True, "so101-a1": True}
    assert resync["estop_issuer"] == "lekiwi-a1"
    assert _SURVIVOR_ROW in records
    assert calls == [2, 3], "must re-read rather than reuse the first trail"
    assert slept == [0.01], "one bounded wait between the two reads"


def test_a_complete_trail_is_reconstructed_on_the_first_read(example):
    """An already-complete trail costs no delay: one read, no waiting."""
    read, calls = _reader(_ISSUER_ROWS + [_SURVIVOR_ROW])
    slept = []

    resync, records = example.resync_until_lockout_recovered(read, _PEERS, ["so101-a1"], sleep=slept.append)

    assert resync["lockouts"]["so101-a1"] is True
    assert len(records) == 3
    assert calls == [3], "a complete trail must not be polled again"
    assert slept == [], "no wait when the evidence is already there"


def test_an_unrecoverable_lockout_still_names_the_reconstruction(example):
    """On timeout the drill reports the same verdict, with diagnostics.

    The message is the one the drill has always raised, so an operator
    reading it still gets the reconstruction that fell short.
    """
    read, calls = _reader(_ISSUER_ROWS)

    with pytest.raises(RuntimeError, match="re-sync did not recover the estop lockout") as excinfo:
        example.resync_until_lockout_recovered(
            read, _PEERS, ["so101-a1"], timeout_s=0.05, poll_s=0.01, sleep=lambda _s: None
        )

    assert "'lekiwi-a1': True" in str(excinfo.value), "names the reconstruction it fell short of"
    assert len(calls) > 1, "polls before giving up"


def test_the_wait_is_bounded_on_a_monotonic_clock_and_reads_before_waiting(example):
    """The bound is measured on the injected clock, after at least one read.

    A wall-clock step must not be able to extend or truncate the bound, so
    the deadline is taken from a monotonic seam - and an expired bound still
    reads once, so a caller never skips the evidence entirely.
    """
    read, calls = _reader(_ISSUER_ROWS)
    ticks = iter([100.0, 1000.0])  # deadline taken, then already elapsed
    slept = []

    with pytest.raises(RuntimeError, match="re-sync did not recover the estop lockout"):
        example.resync_until_lockout_recovered(
            read,
            _PEERS,
            ["so101-a1"],
            timeout_s=15.0,
            clock=lambda: next(ticks),
            sleep=slept.append,
        )

    assert calls == [2], "reads once even when the bound has already elapsed"
    assert slept == [], "does not wait past the bound"


def test_the_live_resync_awaits_the_audit_evidence(example):
    """The live drill routes its re-sync through the awaiting core.

    A regression to a single ``read_audit_log`` + ``resync_after_restart``
    pair inside ``_run_live`` would restore the race, so pin the seam the
    drill reconstructs through.
    """
    import inspect

    source = inspect.getsource(example._run_live)
    assert "resync_until_lockout_recovered(" in source
    assert "resync_after_restart(" not in source, "must reconstruct through the awaiting core"
