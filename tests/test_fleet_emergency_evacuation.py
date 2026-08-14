"""Smoke test for examples/fleet/04_emergency_evacuation.py (issue #2183).

Phases 1-2 (abort + deterministic retreat) and the benchmark scoring drive
the shipped protocol core through the scripted evacuation world: no
simulator, no Zenoh session, no network. The corridor-distance ordering, the
clear-path-before-lockout gate, and the benchmark's three claims (clearance
held for all scored ticks, proxy reached the exit unimpeded, abort inside
deadline) are asserted on outputs, not narrated.

Phase 3 (lockout + HITL resume) is asserted through the REAL mesh safety
handlers, the same pattern the failover example's test uses: a real ``Mesh``
peer minus the Zenoh transport receives the real estop envelope captured from
the issuer's own publish path, refuses everything but ``status``/``resume``
while locked, refuses a wrong-code resume WITHOUT clearing the lockout - the
acceptance criterion this issue names - and resumes only on the real HMAC
override compare.

The audit log is redirected to ``tmp_path`` (never the developer's real
``~/.strands_robots/mesh_audit.jsonl``) and signed with a test PSK so
``verify_audit_integrity`` attests the trail end to end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from strands_robots.mesh import core as mesh_core
from strands_robots.mesh import security as mesh_security

_FLEET_DIR = Path(__file__).resolve().parent.parent / "examples" / "fleet"
_EXAMPLE_PATH = _FLEET_DIR / "04_emergency_evacuation.py"
_MODULE_NAME = "fleet_emergency_evacuation_example"


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
    sys.modules.pop(_MODULE_NAME, None)
    spec = importlib.util.spec_from_file_location(_MODULE_NAME, _EXAMPLE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[_MODULE_NAME] = mod
    spec.loader.exec_module(mod)
    yield mod
    sys.modules.pop(_MODULE_NAME, None)
    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


def _coordinator_events(example):
    from strands_robots.mesh.audit import read_audit_log

    return [r["event"] for r in read_audit_log() if r.get("peer_id") == example.COORDINATOR_ID]


def _mustered_world(example, fleet=None):
    """A scripted world driven to muster through the shipped retreat ticks."""
    world = example.ScriptedEvacuationWorld(fleet)
    world.stop_all_tasks()
    for name in world.robot_names:
        budget = 400
        while not world.retreat_tick(name):
            budget -= 1
            assert budget > 0, f"{name}: scripted retreat did not converge"
    return world


def test_corridor_distance_is_zero_inside_and_euclidean_outside(example):
    assert example.corridor_distance(0.0, 0.0) == 0.0
    assert example.corridor_distance(example.CORRIDOR["x_max"], example.CORRIDOR["y_max"]) == 0.0
    assert example.corridor_distance(0.0, example.CORRIDOR["y_max"] + 1.5) == pytest.approx(1.5)
    # Off a corner the distance is Euclidean, not per-axis.
    assert example.corridor_distance(example.CORRIDOR["x_max"] + 3.0, example.CORRIDOR["y_max"] + 4.0) == pytest.approx(
        5.0
    )


def test_retreat_order_is_corridor_distance_ascending_with_name_tiebreak(example):
    positions = {"far": (0.0, 2.0), "near": (0.0, 0.1), "mid": (0.0, -1.0)}
    assert example.retreat_order(positions) == ["near", "mid", "far"]
    # Equidistant robots order by name - deterministic, never arrival order.
    tied = {"b-robot": (1.0, 0.5), "a-robot": (-1.0, -0.5)}
    assert example.retreat_order(tied) == ["a-robot", "b-robot"]


def test_alarm_gate_rate_limits_a_flood_and_audits_the_suppression(example):
    clock = {"now": 0.0}
    gate = example.AlarmGate(max_alarms=3, window_s=60.0, clock=lambda: clock["now"])
    assert [gate.admit(f"a{i}") for i in range(3)] == [True, True, True]
    # The fourth alarm inside the window is suppressed - and audited, never
    # silently dropped.
    assert gate.admit("a3") is False
    assert "evacuation_alarm_suppressed" in _coordinator_events(example)
    # Outside the rolling window the gate admits again.
    clock["now"] = 61.0
    assert gate.admit("a4") is True


def test_dry_run_protocol_orders_phases_and_signs_the_trail(example):
    """Abort completes, the retreat runs closest-first, the path-clear claim
    precedes everything phase 3 would do, and the benchmark passes - all read
    from outputs and the signed audit chain."""
    from strands_robots.mesh.audit import verify_audit_integrity

    world = example.ScriptedEvacuationWorld()
    summary = example.run_evacuation(world, sleep=lambda _s: None)

    assert summary["order"] == ["lekiwi-1", "go2-1", "arm-1"]
    assert all(c > example.CLEARANCE_M for c in summary["clearances"].values())
    assert world.sim.evacuation_abort_elapsed_s == summary["abort_elapsed_s"]

    example.register_evacuation_predicates()
    tracked = {name: world.tracked_body(name) for name in world.robot_names}
    benchmark = example.DeclarativeBenchmark.from_dict(example.build_benchmark_spec(tracked))
    verdict = example.score_evacuation(world, benchmark)
    assert verdict["passed"] is True
    assert verdict["reason"] == "proxy reached the exit unimpeded"

    events = _coordinator_events(example)
    assert events.index("evacuation_abort_complete") < events.index("evacuation_retreat_order")
    assert events.index("evacuation_retreat_order") < events.index("evacuation_path_clear")
    assert events.index("evacuation_path_clear") < events.index("evacuation_scored")
    integrity = verify_audit_integrity()
    assert integrity["ok"] is True
    assert integrity["signed"] == integrity["total"]


def test_a_robot_that_cannot_clear_the_path_blocks_lockout_engagement(example):
    """Epic D7: phase 2 must complete BEFORE lockout. A muster pose still
    inside the clearance margin is a structured refusal, never a lockout on
    top of a blocked corridor."""
    fleet = [dict(entry) for entry in example.FLEET]
    for entry in fleet:
        if entry["robot"] == "go2-1":
            entry["muster"] = [-1.2, -1.0]  # clearance 0.3 m < the 0.8 m margin
    world = example.ScriptedEvacuationWorld(fleet)

    with pytest.raises(RuntimeError, match="refusing to engage lockout"):
        example.run_evacuation(world, sleep=lambda _s: None)

    events = _coordinator_events(example)
    assert "evacuation_clear_failed" in events
    assert "evacuation_path_clear" not in events


def test_benchmark_fails_the_run_when_a_robot_reenters_the_corridor(example):
    """The clearance claim holds for ALL scored ticks: a robot drifting back
    into the inflated corridor mid-traversal fails the benchmark at that tick."""

    class _DriftingWorld(example.ScriptedEvacuationWorld):
        def __init__(self):
            super().__init__()
            self._ticks = 0

        def settle(self, n):
            self._ticks += n
            if self._ticks == 5:
                self._xy["go2-1"] = [-1.2, -1.0]  # back inside the margin

    world = _mustered_world(example)
    drifting = _DriftingWorld()
    drifting._xy = dict(world._xy)
    drifting.sim.evacuation_abort_elapsed_s = 0.1

    example.register_evacuation_predicates()
    tracked = {name: drifting.tracked_body(name) for name in drifting.robot_names}
    benchmark = example.DeclarativeBenchmark.from_dict(example.build_benchmark_spec(tracked))
    verdict = example.score_evacuation(drifting, benchmark)
    assert verdict["passed"] is False
    assert verdict["reason"] == "corridor clearance breached"
    assert verdict["steps"] == 5


def test_a_blocked_proxy_waits_and_the_run_fails_on_the_step_budget(example):
    """Unimpeded is asserted, not narrated: the proxy never advances through
    an obstruction on the path, so a blocker parks the run into a failure."""
    fleet = [dict(entry) for entry in example.FLEET] + [
        {
            "robot": "blocker-1",
            "data_config": "lekiwi",
            "kind": "mobile",
            "spawn": [2.0, 0.2, 0.0],
            "muster": [2.0, 0.2],
        }
    ]
    world = _mustered_world(example, fleet)
    world.sim.evacuation_abort_elapsed_s = 0.1

    example.register_evacuation_predicates()
    # The blocker is deliberately untracked: the failure being asserted is the
    # proxy's, not a clearance breach.
    tracked = {name: world.tracked_body(name) for name in world.robot_names if name != "blocker-1"}
    benchmark = example.DeclarativeBenchmark.from_dict(example.build_benchmark_spec(tracked))
    verdict = example.score_evacuation(world, benchmark)
    assert verdict["passed"] is False
    assert verdict["reason"] == "proxy never reached the exit"
    # The proxy stopped short of the blocker rather than teleporting past it.
    assert world.proxy_xy()[0] < 2.0 - example.PROXY_STANDOFF_M + example.PROXY_STEP_M


def test_benchmark_requires_the_abort_deadline_not_just_the_traversal(example):
    """An abort that missed its deadline fails the benchmark even though the
    corridor is clear and the proxy walks out."""
    world = _mustered_world(example)
    world.sim.evacuation_abort_elapsed_s = example.ABORT_DEADLINE_S + 5.0

    example.register_evacuation_predicates()
    tracked = {name: world.tracked_body(name) for name in world.robot_names}
    benchmark = example.DeclarativeBenchmark.from_dict(example.build_benchmark_spec(tracked))
    verdict = example.score_evacuation(world, benchmark)
    assert verdict["passed"] is False

    # The predicate itself: missing, malformed and boolean stamps all read as
    # NOT within deadline - a deadline that cannot be read is never met.
    check = example._evacuation_abort_within(5.0)
    assert check(SimpleNamespace()) is False
    assert check(SimpleNamespace(evacuation_abort_elapsed_s=True)) is False
    assert check(SimpleNamespace(evacuation_abort_elapsed_s=7.0)) is False
    assert check(SimpleNamespace(evacuation_abort_elapsed_s=3.0)) is True


def test_incident_report_is_a_deterministic_timeline_with_integrity(example):
    records = [
        {"ts": 100.0, "event": "evacuation_alarm", "peer_id": "evac-coordinator", "payload": {"alarm_id": "a1"}},
        {"ts": 101.5, "event": "evacuation_path_clear", "peer_id": "evac-coordinator", "payload": {"required_m": 0.8}},
        {"ts": "bogus", "event": "dropped", "peer_id": "x", "payload": {}},
    ]
    report = example.build_incident_report(records, {"ok": True, "signed": 2, "total": 2})
    lines = report.splitlines()
    assert "Audit integrity: ok=True (signed=2/2)" in report
    assert any("+0.00s" in line and "evacuation_alarm" in line for line in lines)
    assert any("+1.50s" in line and "evacuation_path_clear" in line for line in lines)
    assert "dropped" not in report  # an unreadable timestamp row is skipped, not guessed


def test_the_report_attests_the_records_it_shows_and_not_the_whole_log(example):
    """The header and the timeline must describe one record set.

    ``main`` scopes its read to this run (``read_audit_log(since=...)``) and
    renders those records as the timeline. The integrity verdict has to be
    about the same records, or the report states two things at once: a header
    counting the developer's entire history above a table showing one run.

    The prior-run records planted here are unsigned, which is what a log
    written before ``STRANDS_MESH_AUDIT_PSK`` was configured looks like. With
    a PSK set at verification time an unsigned record is a forgery by
    definition, so an unscoped verdict does not merely inflate ``total`` - it
    reports ``ok=False`` on a run in which nothing went wrong.
    """
    import time

    from strands_robots.mesh.audit import audit_log_path, log_safety_event, read_audit_log, verify_audit_integrity

    # A machine that has used the mesh before, from runs this report is not about.
    log_file = Path(audit_log_path())
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with log_file.open("a") as handle:
        for i in range(120):
            handle.write(
                json.dumps(
                    {"ts": 1000.0 + i, "event": "some_earlier_run", "peer_id": "prior-run", "seq": i + 1, "payload": {}}
                )
                + "\n"
            )

    run_start = time.time()
    for event in ("evacuation_alarm", "evacuation_path_clear", "evacuation_scored"):
        log_safety_event(event, example.COORDINATOR_ID, {"phase": event})

    records = read_audit_log(since=run_start - 1.0)
    report = example.build_incident_report(records)

    # The header counts exactly the rows below it.
    rows = [line for line in report.splitlines() if line.startswith("| +")]
    assert len(rows) == len(records)
    assert f"(signed={len(records)}/{len(records)})" in report
    assert "Audit integrity: ok=True" in report

    # Non-vacuity: the planted history is really in the log, and an unscoped
    # verdict really would have reported on it. Without this the assertions
    # above would also hold on a log that happened to contain only this run.
    unscoped = verify_audit_integrity()
    assert unscoped["total"] > len(records)
    assert unscoped["ok"] is False


# Phase 3 -- lockout + HMAC resume, asserted through the real safety handlers.


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
    """A real Mesh peer minus the Zenoh transport (see the failover test)."""
    mesh = mesh_core.Mesh(_StoppableRobot(), peer_id=peer_id, peer_type="robot")
    mesh._running = True
    mesh._stop_event.set()
    return mesh


def test_declined_resume_does_not_clear_the_lockout(example, monkeypatch):
    """The acceptance criterion, end to end through the real handlers: the
    estop envelope engages a receiving peer's lockout; a wrong-code resume is
    refused and every non-status/resume action stays refused; only the real
    HMAC override compare clears it."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "drill-override-code")
    published = _capturing_bus(monkeypatch)
    issuer = _live_unstarted_mesh(example.COORDINATOR_ID)
    receiver = _live_unstarted_mesh(example.FLEET_PEER_ID)

    issuer.emergency_stop()
    envelope = next(payload for key, payload in published if key == "strands/safety/estop")
    receiver._on_safety_estop(_as_sample(envelope))

    # Locked out: status still answers, everything else is refused.
    assert isinstance(receiver._dispatch({"action": "status"}), dict)
    with pytest.raises(mesh_security.LockoutError):
        receiver._dispatch({"action": "stop"})

    # The declined resume: refused, and the lockout is INTACT afterwards.
    denied = receiver._dispatch({"action": "resume", "override_code": "not-the-code"})
    assert denied == {"status": "error", "error": "resume rejected"}
    with pytest.raises(mesh_security.LockoutError):
        receiver._dispatch({"action": "stop"})

    # The audit trail names the denial with a structured local reason.
    from strands_robots.mesh.audit import read_audit_log

    denials = [r for r in read_audit_log() if r.get("event") == "resume_denied"]
    assert denials, "a refused resume must leave a resume_denied audit record"

    # Only the correct override code clears it - and then the fleet stop that
    # was refused a moment ago executes.
    resumed = receiver._dispatch({"action": "resume", "override_code": "drill-override-code"})
    assert resumed == {"status": "ok"}
    assert receiver._dispatch({"action": "stop"}) == {"ok": True, "status": "stopped"}


def test_resume_success_publishes_a_proof_other_peers_verify(example, monkeypatch):
    """Fleet-wide resume is second-factor gated: the resume envelope carries
    an HMAC override proof, and a second locked peer clears only on it."""
    monkeypatch.setenv("STRANDS_MESH_OVERRIDE_CODE", "drill-override-code")
    published = _capturing_bus(monkeypatch)
    issuer = _live_unstarted_mesh(example.COORDINATOR_ID)
    peer_a = _live_unstarted_mesh("evac-peer-a")
    peer_b = _live_unstarted_mesh("evac-peer-b")

    issuer.emergency_stop()
    estop = next(payload for key, payload in published if key == "strands/safety/estop")
    peer_a._on_safety_estop(_as_sample(estop))
    peer_b._on_safety_estop(_as_sample(estop))

    assert peer_a._dispatch({"action": "resume", "override_code": "drill-override-code"}) == {"status": "ok"}
    resume = next(payload for key, payload in published if key == "strands/safety/resume")
    assert "override_proof" in resume
    peer_b._on_safety_resume(_as_sample(resume))
    # peer_b never saw the code itself - only the verified proof - and is out
    # of lockout.
    assert peer_b._dispatch({"action": "stop"}) == {"ok": True, "status": "stopped"}


def test_benchmark_spec_tracks_every_robot_and_compiles_in_the_dsl(example):
    example.register_evacuation_predicates()
    tracked = {"r1": "r1/base", "r2": "r2/base"}
    spec = example.build_benchmark_spec(tracked)
    failure_bodies = {clause["body"] for clause in spec["failure"]["any"]}
    assert failure_bodies == {"r1/base", "r2/base"}
    success_predicates = {clause["predicate"] for clause in spec["success"]["all"]}
    assert success_predicates == {"inside_region", "evacuation_abort_within"}
    benchmark = example.DeclarativeBenchmark.from_dict(spec)
    assert benchmark.name == "emergency-evacuation-corridor-clear"


def test_abort_that_misses_its_deadline_is_a_structured_failure(example):
    """A fleet that cannot stop is a refusal with an audit record, never a
    retreat over still-running rollouts."""

    class _StuckWorld(example.ScriptedEvacuationWorld):
        def stop_all_tasks(self):
            pass  # nothing stops; the deadline must trip

    world = _StuckWorld()
    clock = {"now": 0.0}

    def fake_sleep(s):
        clock["now"] += s

    with pytest.raises(RuntimeError, match="missed its"):
        example.run_evacuation(world, clock=lambda: clock["now"], sleep=fake_sleep, abort_deadline_s=2.0)
    assert "evacuation_abort_timeout" in _coordinator_events(example)


def test_registering_the_custom_predicate_twice_is_idempotent(example):
    example.register_evacuation_predicates()
    example.register_evacuation_predicates()  # a reload must not raise
    from strands_robots.simulation.predicates import make_predicate

    check = make_predicate("evacuation_abort_within", deadline_s=5.0)
    assert check(SimpleNamespace(evacuation_abort_elapsed_s=1.0)) is True
