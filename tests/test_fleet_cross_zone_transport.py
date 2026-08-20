"""Smoke tests for examples/fleet/02_cross_zone_transport.py and dashboard.py (issue #2181).

The transport example's deterministic core - decomposition into per-zone legs
joined at a dock, zone-side robot selection through the shared skill
requirement, custody-ordered dispatch behind the HITL gate, and the
machine-readable refusals - is driven through stub approval and transport
seams: no simulator, no Zenoh session, no network.

The dashboard is asserted on its contracts rather than its rendering: the
subscribe-only surface (every method the module declares command-capable
refuses, and raw ``publish`` is confined to the peer's own namespace), the
snapshot pipeline (presence/health/safety folding plus the audit tail, with
no record folded twice), and the classification of the mesh surface itself -
every public ``Mesh`` method must be refused, confined, or documented as an
exempt write path, so a method added to ``Mesh`` cannot land on a peer
advertised as read-only without a decision.

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
        self.safety_events: list[str] = []

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

    def publish_safety_event(self, event_type: str, severity: str = "warning", payload: dict | None = None) -> None:
        # Mirrors the real method: publishes the peer's OWN safety topic (so
        # the confinement governs it) and appends an audit record. Present so
        # this fake can express the exempt write path at all - without it the
        # surface tests below would pass by the method simply being absent.
        self.publish(f"strands/{self.peer_id}/safety/event", {"peer_id": self.peer_id, "type": event_type})
        self.safety_events.append(event_type)


def test_every_command_method_refuses_on_the_restricted_peer(dash):
    # Driven from the module's own list rather than a copy of it, so a method
    # added to _COMMAND_METHODS is exercised here without editing this test.
    assert dash._COMMAND_METHODS, "premise: the module declares a refusal list"
    for method in dash._COMMAND_METHODS:
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


# -- dashboard: the mesh surface is classified, not assumed --------------------

# Public ``Mesh`` methods that read, or manage this peer's own lifecycle: safe
# to hold on a subscribe-only peer, so they need no refusal. Recorded here as
# an explicit decision rather than derived, so a method added to ``Mesh`` falls
# into neither this set nor the module's refusal list and fails the
# classification test below instead of silently landing on the read-only peer.
_READ_ONLY_MESH_METHODS = frozenset({"get_peer", "on_stream", "start", "stop", "subscribe", "unsubscribe"})

# The write path that is neither refused nor confinable: the mesh's own safety
# handlers call it (see the test below), so it must be documented instead.
_EXEMPT_WRITE_METHOD = "publish_safety_event"

_SAFETY_HANDLERS = ("_on_safety_estop", "_on_safety_resume")


def _public_mesh_methods() -> frozenset[str]:
    from strands_robots.mesh.core import Mesh

    return frozenset(name for name in dir(Mesh) if not name.startswith("_") and callable(getattr(Mesh, name, None)))


def _safety_event_calls(handler: str) -> list[tuple[bool, set[str]]]:
    """Every ``self.publish_safety_event`` call in *handler*: (guarded, caught).

    ``guarded`` is True when the call sits inside a ``try``; ``caught`` is the
    set of exception names that ``try``'s handlers name.
    """
    import ast
    import inspect
    import textwrap

    from strands_robots.mesh.core import Mesh

    tree = ast.parse(textwrap.dedent(inspect.getsource(getattr(Mesh, handler))))
    guards: list[tuple[set[int], set[str]]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Try):
            caught: set[str] = set()
            for clause in node.handlers:
                for name in ast.walk(clause.type) if clause.type is not None else []:
                    if isinstance(name, ast.Name):
                        caught.add(name.id)
            guards.append(({id(child) for child in ast.walk(node)}, caught))

    calls: list[tuple[bool, set[str]]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not (isinstance(node.func, ast.Attribute) and node.func.attr == "publish_safety_event"):
            continue
        enclosing = [caught for members, caught in guards if id(node) in members]
        calls.append((bool(enclosing), set().union(*enclosing) if enclosing else set()))
    return calls


def test_every_public_mesh_method_is_refused_confined_or_documented(dash):
    """No method may write to the mesh from this peer without a recorded decision.

    A peer advertised as read-only is only as read-only as the surface it
    holds, and that surface is ``Mesh``, which grows independently of this
    example. So every public ``Mesh`` method must be accounted for: refused
    (``_COMMAND_METHODS``), confined (``publish``), read-only/lifecycle, or
    named in ``restrict_to_subscribe_only``'s docstring as a write path that
    deliberately stays. An unaccounted method is one a reader auditing "can
    this peer write?" is told nothing about.
    """
    surface = _public_mesh_methods()
    documented = dash.restrict_to_subscribe_only.__doc__ or ""
    unclassified = sorted(
        name
        for name in surface
        if name not in _READ_ONLY_MESH_METHODS
        and name not in dash._COMMAND_METHODS
        and name != "publish"  # confined; pinned by the namespace tests above
        and f"``{name}``" not in documented
    )
    assert not unclassified, (
        f"{unclassified} can write to the mesh from a peer advertised as read-only, and are neither refused "
        f"(_COMMAND_METHODS), confined (publish), classified read-only, nor named in "
        f"restrict_to_subscribe_only's docstring as a write path that stays. Refuse it, confine it, or "
        f"document why it stays and what it can reach."
    )


def test_the_read_only_classification_names_only_real_mesh_methods():
    """Non-vacuity: a rename on ``Mesh`` must not leave a stale exemption behind."""
    surface = _public_mesh_methods()
    assert len(surface) >= 10, f"premise: Mesh exposes a public surface to classify, found {sorted(surface)}"
    assert _READ_ONLY_MESH_METHODS <= surface, sorted(_READ_ONLY_MESH_METHODS - surface)
    assert _EXEMPT_WRITE_METHOD in surface


def test_the_module_docstring_names_every_write_path_the_peer_keeps(dash):
    """The read-only claim must name what it keeps, not only what it removes.

    The module docstring is where an operator learns what this peer can do.
    Listing only the refusals reads as "this peer writes nothing", which is
    not true of a peer whose own presence loop publishes and whose safety
    handlers append to the audit trail.
    """
    doc = dash.__doc__ or ""
    for name in ("publish", _EXEMPT_WRITE_METHOD):
        assert f"``{name}``" in doc, (
            f"the module docstring does not name ``{name}``, a write path the restricted peer keeps; "
            f"an operator reading it cannot tell what this peer can still put on the mesh"
        )


def test_the_safety_event_exemption_is_load_bearing_on_the_estop_path():
    """Why ``publish_safety_event`` cannot simply be added to the refusal list.

    The mesh's own remote-estop and remote-resume handlers call it to record
    this peer's lockout transitions. At least one call is outside a ``try``,
    and the guarded ones narrow to payload/disk errors - none of them catches
    ``RuntimeError``. So refusing it would raise inside a Zenoh subscription
    callback on the safety path: a break, not a guard. This test fails if that
    stops being true, which is the point at which refusing it becomes an
    option worth reconsidering.
    """
    from strands_robots.mesh.core import Mesh

    for handler in _SAFETY_HANDLERS:
        assert hasattr(Mesh, handler), f"premise: Mesh.{handler} is the receiver-side safety handler"
        calls = _safety_event_calls(handler)
        assert calls, f"premise: Mesh.{handler} records transitions through publish_safety_event"
        assert any(not guarded for guarded, _ in calls), (
            f"every publish_safety_event call in Mesh.{handler} is now guarded; if the guards also absorb "
            f"RuntimeError, the refusal list can take publish_safety_event"
        )
        for guarded, caught in calls:
            if guarded:
                assert "RuntimeError" not in caught, (
                    f"a guarded publish_safety_event call in Mesh.{handler} now absorbs RuntimeError; "
                    f"re-check whether the read-only peer can refuse the method"
                )


def test_the_exempt_safety_write_is_scoped_to_the_peers_own_id(dash):
    """The exemption is bounded: it can only ever name this peer.

    ``publish_safety_event`` stays callable on the restricted peer, so the
    bound that matters is what it can reach - its own ``safety/event`` topic
    and audit records carrying its own ``peer_id``. It cannot speak for
    another peer, which is what keeps the exemption narrow enough to document
    rather than close.
    """
    import inspect

    from strands_robots.mesh.core import Mesh

    source = inspect.getsource(getattr(Mesh, _EXEMPT_WRITE_METHOD))
    assert 'f"strands/{self.peer_id}/safety/event"' in source, "the wire topic is no longer peer-scoped"
    assert "peer_id=self.peer_id" in source, "the audit record is no longer peer-scoped"

    mesh = dash.restrict_to_subscribe_only(_FakeMesh())
    mesh.publish_safety_event("remote_estop_engaged", severity="critical")
    assert mesh.published == [
        ("strands/fleet-dashboard/safety/event", {"peer_id": "fleet-dashboard", "type": "remote_estop_engaged"}),
    ]
    assert mesh.safety_events == ["remote_estop_engaged"]
