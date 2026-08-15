"""Smoke tests for examples/fleet/02_cross_zone_transport.py and dashboard.py (issue #2181).

The transport example's deterministic core - decomposition into per-zone legs
joined at a dock, zone-side robot selection through the shared skill
requirement, custody-ordered dispatch behind the HITL gate, and the
machine-readable refusals - is driven through stub approval and transport
seams: no simulator, no Zenoh session, no network.

The dashboard is asserted on its two contracts rather than its rendering:
the subscribe-only surface (every command-capable method refuses, and raw
``publish`` is confined to the peer's own namespace) and the snapshot
pipeline (presence/health/safety folding plus the audit tail, with no record
folded twice).

The audit log is redirected to ``tmp_path`` (never the developer's real
``~/.strands_robots/mesh_audit.jsonl``) and signed with a test PSK so
``verify_audit_integrity`` attests the trail end to end.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_FLEET_DIR = Path(__file__).resolve().parent.parent / "examples" / "fleet"

# Loaded under distinctive module names: "capabilities" (example 02's sibling
# import) is generic enough to collide, so it is evicted between loads.
_EXAMPLE_MODULE = "fleet_cross_zone_transport_example"
_DASHBOARD_MODULE = "fleet_dashboard_example"


def _reset_audit_state() -> None:
    from strands_robots.mesh import audit

    audit._SEQ_COUNTERS.clear()
    audit._AUDIT_STATE.seq_loaded = False
    audit._AUDIT_STATE.audit_log_seeded = False
    audit._AUDIT_STATE.psk_fingerprint = None


def _load(module_name: str, filename: str):
    spec = importlib.util.spec_from_file_location(module_name, _FLEET_DIR / filename)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def example(monkeypatch, tmp_path):
    """Load example 02 with the audit log confined to tmp_path and signed."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "smoke-test-psk")
    # Reset the process-global audit state (same isolation as
    # tests/mesh/test_audit_integrity.py): the PSK fingerprint and sequence
    # counters are one-shot per process, so records written by earlier tests
    # in the suite would otherwise poison this test's fresh, signed log.
    _reset_audit_state()
    monkeypatch.syspath_prepend(str(_FLEET_DIR))
    for name in (_EXAMPLE_MODULE, "capabilities"):
        sys.modules.pop(name, None)
    mod = _load(_EXAMPLE_MODULE, "02_cross_zone_transport.py")
    yield mod
    for name in (_EXAMPLE_MODULE, "capabilities"):
        sys.modules.pop(name, None)
    _reset_audit_state()


@pytest.fixture
def dash(monkeypatch, tmp_path):
    """Load dashboard.py with the audit log confined to tmp_path."""
    monkeypatch.setenv("STRANDS_MESH_AUDIT_DIR", str(tmp_path / "audit"))
    monkeypatch.setenv("STRANDS_MESH_AUDIT_PSK", "smoke-test-psk")
    _reset_audit_state()
    sys.modules.pop(_DASHBOARD_MODULE, None)
    mod = _load(_DASHBOARD_MODULE, "dashboard.py")
    yield mod
    sys.modules.pop(_DASHBOARD_MODULE, None)
    _reset_audit_state()


class _RecordingSend:
    """Transport stub: records dispatch order, fails or times out on cue."""

    def __init__(self, fail_on_leg: int | None = None):
        self.sent: list[tuple[str, str]] = []
        self._fail_on_leg = fail_on_leg

    def __call__(self, zone: str, cmd: dict) -> dict:
        self.sent.append((zone, cmd["robot_name"]))
        if self._fail_on_leg is not None and len(self.sent) == self._fail_on_leg:
            return {"status": "timeout"}
        return {"type": "response", "result": {"status": "success"}}


def _approve_all(_action: str, _zone: str, _instruction: str) -> bool:
    return True


# -- decomposition ----------------------------------------------------------


def test_cross_zone_request_decomposes_into_two_legs_joined_at_the_dock(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    result = example.decompose_cross_zone(request, example.ZONES, example.DOCKS)
    assert [leg["zone"] for leg in result["legs"]] == ["zone-a", "zone-b"]
    assert result["legs"][0]["dst"] == "dock-ab"
    assert result["legs"][1]["src"] == "dock-ab"
    assert result["handoff"] == {"location": "dock-ab", "from_zone": "zone-a", "to_zone": "zone-b"}


def test_same_zone_request_is_one_leg_and_no_handoff(example):
    request = {"request_id": "X-02", "payload_kg": 3.0, "src": "dock-ab", "dst": "etch"}
    result = example.decompose_cross_zone(request, example.ZONES, example.DOCKS)
    assert len(result["legs"]) == 1
    assert result["handoff"] is None


def test_unroutable_location_is_refused_naming_the_location(example):
    request = {"request_id": "X-03", "payload_kg": 3.0, "src": "stock", "dst": "cleanroom"}
    result = example.decompose_cross_zone(request, example.ZONES, example.DOCKS)
    assert result["refused"] == {"code": "unroutable_location", "location": "cleanroom"}


def test_zone_pair_without_a_dock_is_refused_naming_the_zones(example):
    zones = {
        "zone-a": dict(example.ZONES["zone-a"]),
        "zone-c": {"locations": ("far",), "manifests": example.ZONES["zone-b"]["manifests"]},
    }
    request = {"request_id": "X-04", "payload_kg": 3.0, "src": "stock", "dst": "far"}
    result = example.decompose_cross_zone(request, zones, {})
    assert result["refused"] == {"code": "no_dock_between_zones", "zones": ["zone-a", "zone-c"]}


# -- zone-side selection through the one shared skill ------------------------


def test_each_leg_is_served_by_its_own_zones_robot_via_the_shared_skill(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    assert [(leg["zone"], leg["robot"]) for leg in plan["legs"]] == [("zone-a", "lekiwi-a1"), ("zone-b", "go2-b1")]


def test_leg_requirement_is_derived_from_the_single_shared_skill_artifact(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    req_a = example.leg_requirement(request, {"zone": "zone-a", "src": "stock", "dst": "dock-ab"})
    req_b = example.leg_requirement(request, {"zone": "zone-b", "src": "dock-ab", "dst": "etch"})
    # Same skill, same fixture, same payload for both zones: the only
    # difference between the legs is where they run.
    assert (req_a.skill, req_a.fixture, req_a.payload_kg) == (req_b.skill, req_b.fixture, req_b.payload_kg)
    assert req_a.skill == example.TRANSPORT_SKILL["name"]


def test_a_payload_no_zone_robot_can_carry_refuses_the_plan_with_rejections(example):
    request = {"request_id": "X-05", "payload_kg": 40.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    assert plan["refused"]["code"] == "no_capable_robot_in_zone"
    assert plan["refused"]["rejections"][0]["constraint"] == "payload_kg"


# -- custody-ordered execution behind the HITL gate ---------------------------


def test_leg_two_is_dispatched_only_after_leg_one_succeeds(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    send = _RecordingSend()
    summary = example.execute_handoff(plan, _approve_all, send)
    assert summary["status"] == "done"
    assert send.sent == [("zone-a", "lekiwi-a1"), ("zone-b", "go2-b1")]
    assert summary["at"] == "etch"
    assert summary["legs_done"] == [1, 2]


def test_a_failed_first_leg_aborts_before_the_second_zone_is_ever_asked(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    send = _RecordingSend(fail_on_leg=1)
    summary = example.execute_handoff(plan, _approve_all, send)
    assert summary["status"] == "failed"
    assert summary["failed_leg"] == 1
    assert send.sent == [("zone-a", "lekiwi-a1")]
    # Custody is honest: the tote never left its source location.
    assert summary["at"] == "stock"


def test_a_declined_second_leg_strands_the_payload_at_the_dock_explicitly(example):
    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    send = _RecordingSend()

    def decline_second(_action: str, zone: str, _instruction: str) -> bool:
        return zone != "zone-b"

    summary = example.execute_handoff(plan, decline_second, send)
    assert summary["status"] == "declined"
    assert summary["declined_leg"] == 2
    assert send.sent == [("zone-a", "lekiwi-a1")]
    assert summary["at"] == "dock-ab"


def test_a_refused_plan_reaches_neither_the_gate_nor_the_wire(example):
    request = {"request_id": "X-03", "payload_kg": 3.0, "src": "stock", "dst": "cleanroom"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    send = _RecordingSend()
    approvals: list[str] = []

    def approve(_action: str, zone: str, _instruction: str) -> bool:
        approvals.append(zone)
        return True

    summary = example.execute_handoff(plan, approve, send)
    assert summary["status"] == "refused"
    assert summary["refusal"]["code"] == "unroutable_location"
    assert send.sent == []
    assert approvals == []


def test_hitl_gate_auto_approves_only_under_the_documented_env_opt_out(example, monkeypatch, capsys):
    monkeypatch.setenv("STRANDS_MESH_HITL_ACTIONS", "none")
    gate = example.make_hitl_gate()
    assert gate("dispatch", "zone-a", "leg 1") is True
    assert "auto-approved" in capsys.readouterr().out


@pytest.mark.parametrize("n_steps", ["0", "-5"])
def test_a_non_positive_n_steps_is_refused_before_anything_starts(example, n_steps):
    with pytest.raises(SystemExit, match="--n-steps must be a positive number of policy steps"):
        example.main(["--dry-run", "--n-steps", n_steps])


def test_handoff_writes_a_signed_custody_chain_to_the_audit_log(example):
    from strands_robots.mesh.audit import read_audit_log, verify_audit_integrity

    request = {"request_id": "X-01", "payload_kg": 3.0, "src": "stock", "dst": "etch"}
    plan = example.plan_request(request, example.ZONES, example.DOCKS)
    example.execute_handoff(plan, _approve_all, _RecordingSend())
    events = [r["event"] for r in read_audit_log() if r.get("payload", {}).get("request_id") == "X-01"]
    assert events == [
        "handoff_dispatch",
        "handoff_leg_done",
        "handoff_custody",
        "handoff_leg_done",
        "handoff_complete",
    ]
    integrity = verify_audit_integrity()
    assert integrity["ok"] is True
    assert integrity["signed"] == integrity["total"] > 0


# -- dashboard: subscribe-only surface ----------------------------------------


class _FakeMesh:
    """Just enough Mesh: publish/subscribe recorders + command methods."""

    def __init__(self, peer_id: str = "fleet-dashboard"):
        self.peer_id = peer_id
        self.peers: list[dict] = []
        self.published: list[tuple[str, dict]] = []
        self.subscribed: list[str] = []
        self.commands: list[str] = []

    def publish(self, key: str, payload: dict) -> None:
        self.published.append((key, payload))

    def subscribe(self, topic: str, callback=None, name=None):
        self.subscribed.append(topic)
        return name or topic

    def send(self, *args, **kwargs):
        self.commands.append("send")
        return {}

    def tell(self, *args, **kwargs):
        self.commands.append("tell")
        return {}

    def broadcast(self, *args, **kwargs):
        self.commands.append("broadcast")
        return []

    def emergency_stop(self):
        self.commands.append("emergency_stop")
        return []

    def publish_step(self, *args, **kwargs):
        self.commands.append("publish_step")


@pytest.mark.parametrize("method", ["send", "tell", "broadcast", "emergency_stop", "publish_step"])
def test_every_command_method_refuses_on_the_restricted_peer(dash, method):
    mesh = dash.restrict_to_subscribe_only(_FakeMesh())
    with pytest.raises(RuntimeError, match="read-only dashboard"):
        getattr(mesh, method)()
    assert mesh.commands == []  # the underlying command surface was never reached


def test_restricted_publish_refuses_every_key_outside_the_peers_own_namespace(dash):
    mesh = dash.restrict_to_subscribe_only(_FakeMesh())
    for key in ("strands/zone-a/cmd", "strands/broadcast", "strands/safety/estop", "strands/safety/resume"):
        with pytest.raises(RuntimeError, match="read-only dashboard"):
            mesh.publish(key, {})
    assert mesh.published == []


def test_restricted_publish_still_allows_the_peers_own_liveness_topics(dash):
    mesh = dash.restrict_to_subscribe_only(_FakeMesh())
    mesh.publish("strands/fleet-dashboard/presence", {"robot_id": "fleet-dashboard"})
    assert mesh.published == [("strands/fleet-dashboard/presence", {"robot_id": "fleet-dashboard"})]


# -- dashboard: snapshot pipeline ---------------------------------------------


class _RecordingRenderer:
    def __init__(self):
        self.snapshots: list[dict] = []

    def render(self, snapshot: dict) -> None:
        self.snapshots.append(snapshot)


def test_attach_subscribes_the_whole_read_surface_and_raises_on_refusal(dash):
    mesh = _FakeMesh()
    board = dash.FleetDashboard(mesh, _RecordingRenderer(), tail_audit=False)
    board.attach()
    assert mesh.subscribed == list(dash.SUBSCRIBE_TOPICS)

    class _RefusingMesh(_FakeMesh):
        def subscribe(self, topic, callback=None, name=None):
            return None

    board = dash.FleetDashboard(_RefusingMesh(), _RecordingRenderer(), tail_audit=False)
    with pytest.raises(RuntimeError, match="could not subscribe"):
        board.attach()


def test_snapshot_folds_presence_health_and_safety_into_stable_rows(dash):
    mesh = _FakeMesh()
    mesh.peers = [
        {"peer_id": "zone-b", "type": "sim", "age": 0.5, "task_status": "running"},
        {"peer_id": "zone-a", "type": "sim", "age": 1.5},
    ]
    board = dash.FleetDashboard(mesh, _RecordingRenderer(), tail_audit=False)
    board._on_sample("strands/zone-a/health", {"peer_id": "zone-a", "battery": 88})
    board._on_sample("strands/zone-a/safety/event", {"peer_id": "zone-a", "type": "remote_estop_engaged"})
    snapshot = board.snapshot()
    assert [row["peer"] for row in snapshot["peers"]] == ["zone-a", "zone-b"]  # sorted, stable
    assert snapshot["peers"][0]["battery"] == 88
    assert snapshot["peers"][0]["safety"] == "remote_estop_engaged"
    assert snapshot["peers"][1]["task"] == "running"


def test_fleet_estop_and_resume_flip_the_fleet_safety_state(dash):
    board = dash.FleetDashboard(_FakeMesh(), _RecordingRenderer(), tail_audit=False)
    board._on_sample("strands/safety/estop", {"peer_id": "zone-a", "t": 1.0})
    assert board.snapshot()["fleet_safety"] == "LOCKOUT (estop by zone-a)"
    board._on_sample("strands/safety/resume", {"t": 2.0})
    assert board.snapshot()["fleet_safety"] == "ok (resumed)"
    assert [e["event"] for e in board.snapshot()["events"]] == ["estop", "resume"]


def test_audit_tail_reaches_the_timeline_once_and_only_once(dash):
    from strands_robots.mesh.audit import log_safety_event

    board = dash.FleetDashboard(_FakeMesh(), renderer := _RecordingRenderer(), tail_audit=True)
    board._audit_since = 0.0  # everything this test writes is in the window
    log_safety_event("handoff_dispatch", "fleet-coordinator", {"request_id": "X-01"})
    board.tick()
    board.tick()  # a second poll must not fold the same record twice
    events = [e for e in renderer.snapshots[-1]["events"] if e["source"] == "audit"]
    assert [e["event"] for e in events] == ["handoff_dispatch"]


def test_renderer_degrades_to_the_terminal_when_rerun_is_absent(dash, monkeypatch, capsys):
    def _absent(*_args, **_kwargs):
        raise ImportError("rerun is not installed")

    monkeypatch.setattr(dash, "require_optional", _absent)
    renderer = dash.make_renderer(prefer_rerun=True)
    assert isinstance(renderer, dash.TerminalRenderer)
    assert "terminal" in capsys.readouterr().out
    # And the fallback actually renders a snapshot without raising.
    renderer.render(
        dash.build_snapshot(
            [{"peer_id": "zone-a", "type": "sim", "age": 0.5}],
            {},
            {"fleet": "ok"},
            [{"ts": 1.0, "source": "wire", "event": "estop", "peer": "zone-a"}],
        )
    )
    assert "zone-a" in capsys.readouterr().out
