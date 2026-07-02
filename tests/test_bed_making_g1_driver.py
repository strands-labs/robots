"""Smoke tests for the bed-making Device Connect swarm driver and sidecar runner."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

try:
    from examples.mujoco_bed_making.bed_making_swarm.unitree_g1_bed_making_g1_driver import (
        BedMakingG1Driver,
        SwarmAgent,
        SwarmState,
    )
except ImportError:  # pragma: no cover - optional Device Connect edge runtime
    pytest.skip("device_connect_edge not installed", allow_module_level=True)


def test_driver_identity_reflects_role_and_device_id():
    control = BedMakingG1Driver(role="control", device_id="dev-control")
    worker = BedMakingG1Driver(role="worker", device_id="dev-worker")

    assert control.identity.device_type == "unitree_g1_bed_making"
    assert control.identity.manufacturer == "Unitree"
    assert "control" in control.identity.description
    assert "worker" in worker.identity.description
    assert control.device_id == "dev-control"
    assert worker.device_id == "dev-worker"


def test_status_defaults_to_available_for_green_pulse_pill():
    driver = BedMakingG1Driver(role="peer", device_id="dev-x")
    # The Device Connect dashboard renders ``availability="available"`` as
    # the green pulsing live pill (same convention as the other registered
    # devices). Using "online" or "idle" would render grey. So the driver
    # must default to "available" while connected and not running an RPC.
    assert driver.status.availability == "available"
    assert driver.status.busy_score == 0.0
    assert driver.status.online is True


def test_get_status_returns_swarm_snapshot():
    driver = BedMakingG1Driver(role="control", device_id="dev-control")
    status = asyncio.run(driver.getStatus())

    assert status["device_id"] == "dev-control"
    assert status["role"] == "control"
    assert status["status"] == "online"
    assert status["held_corner"] is None
    assert status["placed_corners"] == []
    assert status["peer_claims"] == {}
    assert "rpc_counts" in status
    assert status["rpc_counts"]["getStatus"] == 1


def test_peer_event_handlers_update_swarm_state():
    """The driver subscribes to peer DC events and mutates SwarmState."""

    driver = BedMakingG1Driver(role="peer", device_id="dev-a")
    agent = driver.agent

    asyncio.run(driver.onPeerClaim("dev-b", "claimCorner", {"corner": "sheet_foot_left"}))
    assert agent.state.peer_claims["dev-b"] == "sheet_foot_left"

    asyncio.run(
        driver.onPeerHeld(
            "dev-b",
            "cornerHeld",
            {"corner": "sheet_foot_left", "x": 0.5, "y": -0.2, "z": 0.3},
        )
    )
    view = agent.state.peers["dev-b"]
    assert view.holding_corner == "sheet_foot_left"
    assert view.holding_position == (0.5, -0.2, 0.3)
    assert view.status == "holding"

    asyncio.run(
        driver.onPeerPlaced(
            "dev-b",
            "cornerPlaced",
            {"corner": "sheet_foot_left", "bed_corner": "bed_head_right"},
        )
    )
    assert "sheet_foot_left" in agent.state.placed_corners
    assert agent.state.peers["dev-b"].holding_corner is None

    asyncio.run(driver.onPeerReleased("dev-b", "cornerReleased", {"corner": "sheet_foot_left"}))
    assert agent.state.peers["dev-b"].status == "online"


def test_peer_event_for_self_is_ignored():
    driver = BedMakingG1Driver(role="peer", device_id="dev-self")
    asyncio.run(driver.onPeerClaim("dev-self", "claimCorner", {"corner": "sheet_foot_left"}))
    assert driver.agent.state.peer_claims == {}


def test_sidecar_credentials_file_layout():
    """The sidecar relies on creds files having device_id and tenant."""

    repo_root = Path(__file__).resolve().parents[1]
    for filename in (
        "beta-unitree-g1-humanoid-0.creds.json",
        "beta-unitree-g1-humanoid-1.creds.json",
    ):
        path = repo_root / ".credentials" / filename
        if not path.exists():
            pytest.skip(f"creds file {path} not present in this checkout")
        payload = json.loads(path.read_text())
        assert payload.get("device_id"), f"missing device_id in {path}"
        assert payload.get("tenant"), f"missing tenant in {path}"
        assert payload.get("auth_type") == "jwt"
        nats = payload.get("nats") or {}
        assert nats.get("jwt"), f"missing nats.jwt in {path}"
        assert nats.get("nkey_seed"), f"missing nats.nkey_seed in {path}"


def test_sidecar_runner_argument_resolution(tmp_path):
    """Verify the sidecar's argument parsing without launching the runtime."""

    from examples.mujoco_bed_making.bed_making_swarm import unitree_g1_bed_making_device_connect_sidecar as runner

    fake_creds = tmp_path / "fake.creds.json"
    fake_creds.write_text(json.dumps({"device_id": "fake-dev-1", "tenant": "demo"}))

    args = type(
        "A",
        (),
        {
            "credentials": [f"peer-a={fake_creds}"],
            "nats_url": "nats://example:4222",
            "tenant": None,
            "log_level": "INFO",
        },
    )
    specs = runner._resolve_specs(args)
    assert len(specs) == 1
    spec = specs[0]
    assert spec.role == "peer-a"
    assert spec.device_id == "fake-dev-1"
    assert spec.credentials_file == fake_creds


def test_swarm_state_defaults():
    state = SwarmState(device_id="dev-x")
    assert state.role == "peer"
    assert state.status == "online"
    assert state.held_corner is None
    assert state.placed_corners == set()
    assert state.peer_claims == {}


def test_swarm_agent_snapshot_is_lockable():
    agent = SwarmAgent(device_id="dev-x")
    agent.set_status("holding")
    agent.set_held_corner("sheet_foot_left")
    agent.note_peer_claim("dev-y", "sheet_head_right")
    snap = agent.snapshot()
    assert snap["status"] == "holding"
    assert snap["held_corner"] == "sheet_foot_left"
    assert snap["peer_claims"]["dev-y"] == "sheet_head_right"
