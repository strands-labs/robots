"""Smoke test for examples/fleet/05_work_order_dispatch.py (issue #2185).

Drives the deterministic core - schema validation, hard-constraint filtering,
NACK with a machine-readable reason, multi-step sequencing, HITL decline, and
the order_id-threaded signed audit trail - through a stub transport. No
simulator, no Zenoh session, no network: the transport seam
(``send(robot_name, instruction, n_steps)``) is exactly where the live path
plugs in ``mesh.send``, so everything upstream of the wire is exercised as
shipped.

The audit log is redirected to ``tmp_path`` (never the developer's real
``~/.strands_robots/mesh_audit.jsonl``) and signed with a test PSK so
``verify_audit_integrity`` attests the trail end to end.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_FLEET_DIR = Path(__file__).resolve().parent.parent / "examples" / "fleet"
_EXAMPLE_PATH = _FLEET_DIR / "05_work_order_dispatch.py"
_ORDERS_PATH = _FLEET_DIR / "work_orders.jsonl"

# Loaded under a distinctive module name: "capabilities" (its sibling import)
# is generic enough to collide, so both are evicted between loads.
_MODULE_NAME = "fleet_work_order_dispatch_example"


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


def _approve_all(action, target, instruction):
    return True


def _sending_stub(log):
    def send(robot_name, instruction, n_steps):
        log.append((robot_name, instruction))
        return {"result": {"status": "success", "detail": "stub"}}

    return send


def _run_shipped_queue(example, tmp_path, approve=_approve_all, send=None, sends=None):
    sends = sends if sends is not None else []
    events_path = tmp_path / "events.jsonl"
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    summary = example.process_queue(
        orders_path=_ORDERS_PATH,
        events_path=events_path,
        manifests=manifests,
        send=send or _sending_stub(sends),
        approve=approve,
    )
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    return summary, events, sends


def _audit_records(example):
    from strands_robots.mesh.audit import read_audit_log

    return [
        r for r in read_audit_log() if r.get("peer_id") == example.DISPATCHER_ID and isinstance(r.get("payload"), dict)
    ]


def test_shipped_queue_dispatches_per_site_to_different_robot_types(example, tmp_path):
    """One order per site lands on a different robot type; the heavy order is NACKed."""
    summary, events, sends = _run_shipped_queue(example, tmp_path)

    assert summary["completed"] == ["WO-1001", "WO-1002", "WO-1003"]
    assert summary["nacked"] == ["WO-1004"]
    assert summary["failed"] == []

    completed = {e["order_id"]: e for e in events if e["event"] == "work_order_completed"}
    site_a_robot = completed["WO-1001"]["steps"][0]["robot"]
    site_b_robot = completed["WO-1002"]["steps"][0]["robot"]
    assert completed["WO-1001"]["steps"][0]["site"] == "site-a"
    assert completed["WO-1002"]["steps"][0]["site"] == "site-b"
    # Different robot TYPES, not just different robots (arm vs quadruped).
    assert example.ROBOT_EMBODIMENT[site_a_robot] != example.ROBOT_EMBODIMENT[site_b_robot]


def test_multi_step_order_is_sequenced_across_capable_robots(example, tmp_path):
    """The 'process' order dispatches handle then transport, in order, order_id threaded."""
    _, events, sends = _run_shipped_queue(example, tmp_path)

    wo_1003 = [(robot, instruction) for robot, instruction in sends if "WO-1003" in instruction]
    assert len(wo_1003) == 2
    assert "step 1/2" in wo_1003[0][1] and "handle" in wo_1003[0][1]
    assert "step 2/2" in wo_1003[1][1] and "transport" in wo_1003[1][1]
    # Sequenced onto the robots capable of each step - the arm cannot
    # transport and the base cannot handle, so the steps must split.
    assert wo_1003[0][0] != wo_1003[1][0]

    completed = {e["order_id"]: e for e in events if e["event"] == "work_order_completed"}
    assert [s["skill"] for s in completed["WO-1003"]["steps"]] == ["handle", "transport"]


def test_infeasible_order_is_nacked_with_machine_readable_reason(example, tmp_path):
    """No feasible robot -> NACK naming the constraint per robot; nothing dispatched."""
    _, events, sends = _run_shipped_queue(example, tmp_path)

    nacks = [e for e in events if e["event"] == "work_order_nacked"]
    assert [e["order_id"] for e in nacks] == ["WO-1004"]
    reason = nacks[0]["reason"]
    assert reason["code"] == "no_feasible_robot"
    assert reason["step"]["skill"] == "transport"
    by_robot = {r["robot"]: r for r in reason["rejections"]}
    # Machine-readable: each rejection names the failing constraint and both values.
    assert by_robot["lekiwi-a1"]["constraint"] == "payload_kg"
    assert by_robot["lekiwi-a1"]["required"] > by_robot["lekiwi-a1"]["actual"]
    assert by_robot["go2-b1"]["constraint"] == "site"
    assert by_robot["so101-a1"]["constraint"] == "skill"
    # A NACK is a rejection, not a dispatch.
    assert not any("WO-1004" in instruction for _, instruction in sends)


def test_audit_log_reconstructs_every_order_end_to_end(example, tmp_path):
    """order -> dispatch -> action -> completion for every order, signed and gap-free."""
    from strands_robots.mesh.audit import verify_audit_integrity

    _run_shipped_queue(example, tmp_path)

    chains: dict[str, list[str]] = {}
    for record in _audit_records(example):
        order_ref = record["payload"].get("order_id")
        if order_ref:
            chains.setdefault(order_ref, []).append(record["event"])

    assert chains["WO-1001"] == [
        "work_order_received",
        "work_order_dispatch",
        "work_order_action",
        "work_order_completed",
    ]
    assert chains["WO-1002"] == chains["WO-1001"]
    assert chains["WO-1003"] == [
        "work_order_received",
        "work_order_dispatch",
        "work_order_action",
        "work_order_dispatch",
        "work_order_action",
        "work_order_completed",
    ]
    assert chains["WO-1004"] == ["work_order_received", "work_order_nacked"]

    # Dispatch and action rows carry the robot, so the trail names who acted.
    dispatches = [r for r in _audit_records(example) if r["event"] == "work_order_dispatch"]
    assert all(r["payload"]["robot"] for r in dispatches)

    integrity = verify_audit_integrity()
    assert integrity["ok"] is True
    assert integrity["signed"] == integrity["total"] > 0


def test_invalid_orders_are_nacked_never_dropped_or_raised(example, tmp_path):
    """Malformed JSON, unknown fields, and cross-site orders all NACK with a code."""
    orders = tmp_path / "orders.jsonl"
    orders.write_text(
        "this is not json\n"
        + json.dumps(
            {
                "order_id": "WO-2001",
                "material": "wafer_lot",
                "operation": "inspect",
                "qty": 1,
                "from": "site-a/litho",
                "to": "site-a/litho",
                "due": "2026-09-01T00:00:00Z",
                "priority": "high",
            }
        )
        + "\n"
        + json.dumps(
            {
                "order_id": "WO-2002",
                "material": "tote",
                "operation": "transfer",
                "qty": 1,
                "from": "site-a/stock",
                "to": "site-b/stock",
                "due": "2026-09-01T00:00:00Z",
            }
        )
        + "\n"
    )
    events_path = tmp_path / "events.jsonl"
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    sends = []
    summary = example.process_queue(
        orders_path=orders,
        events_path=events_path,
        manifests=manifests,
        send=_sending_stub(sends),
        approve=_approve_all,
    )
    assert sends == []
    assert summary["completed"] == []
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    assert [e["event"] for e in events] == ["work_order_nacked"] * 3
    assert events[0]["reason"]["code"] == "invalid_json"
    assert events[0]["order_id"] is None
    assert events[1]["reason"]["code"] == "invalid_order"
    assert events[1]["reason"]["field"] == "priority"
    assert events[2]["reason"]["code"] == "invalid_order"
    assert events[2]["reason"]["field"] == "to"


def test_hitl_decline_fails_the_order_and_dispatches_nothing(example, tmp_path):
    """A declined approval is a structured failure event, not a silent skip."""

    def decline_all(action, target, instruction):
        return False

    summary, events, sends = _run_shipped_queue(example, tmp_path, approve=decline_all)

    assert sends == []
    assert summary["completed"] == []
    assert summary["failed"] == ["WO-1001", "WO-1002", "WO-1003"]
    failed = [e for e in events if e["event"] == "work_order_failed"]
    assert all(e["reason"]["code"] == "hitl_declined" for e in failed)
    audited = [r["event"] for r in _audit_records(example) if r["payload"].get("order_id") == "WO-1001"]
    assert audited == ["work_order_received", "work_order_failed"]


def test_transport_failure_is_a_structured_failure_event(example, tmp_path):
    """A timeout reply fails the order with the reply quoted; no false completion."""

    def timeout_send(robot_name, instruction, n_steps):
        return {"status": "timeout"}

    summary, events, _ = _run_shipped_queue(example, tmp_path, send=timeout_send)

    assert summary["completed"] == []
    assert summary["failed"] == ["WO-1001", "WO-1002", "WO-1003"]
    failed = [e for e in events if e["event"] == "work_order_failed"]
    assert all(e["reason"]["code"] == "dispatch_failed" for e in failed)
    assert all("timeout" in e["reason"]["reply"] for e in failed)


def test_guard_refuses_a_choice_outside_the_feasible_set(example, capsys):
    """The agent seam cannot invent a capability: out-of-set picks fall back."""
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    feasible = [m for m in manifests if m.site == "site-a"]
    fallback = feasible[0]

    chosen = example.guard_choice(feasible[1].robot, feasible, fallback)
    assert chosen is feasible[1]

    # go2-b1 is a real robot but NOT in this feasible set - still refused.
    refused = example.guard_choice("go2-b1", feasible, fallback)
    assert refused is fallback
    assert "refused" in capsys.readouterr().out

    invented = example.guard_choice("robot-that-does-not-exist", feasible, fallback)
    assert invented is fallback


def test_chooser_out_of_set_pick_still_dispatches_deterministically(example, tmp_path):
    """An LLM chooser hallucinating a robot never blocks the queue or misassigns."""

    def hallucinating_chooser(order, step, feasible):
        return "invented-robot-9000"

    events_path = tmp_path / "events.jsonl"
    manifests = [example.manifest_from_dict(m) for m in example.FLEET_MANIFESTS]
    sends = []
    summary = example.process_queue(
        orders_path=_ORDERS_PATH,
        events_path=events_path,
        manifests=manifests,
        send=_sending_stub(sends),
        approve=_approve_all,
        chooser=hallucinating_chooser,
    )
    assert summary["completed"] == ["WO-1001", "WO-1002", "WO-1003"]
    dispatched_robots = {robot for robot, _ in sends}
    assert dispatched_robots <= set(example.ROBOT_EMBODIMENT)
