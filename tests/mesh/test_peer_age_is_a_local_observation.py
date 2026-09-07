"""A peer's presence payload does not decide what this process observed about it.

``PeerInfo.caps`` is the peer's own presence payload, merged into
:meth:`~strands_robots.mesh.session.PeerInfo.to_dict` so a consumer reads a
peer's capabilities (``tool_name``, ``connected``, ``cameras``, ...) beside its
identity. Five of the keys in that dict are not capabilities: ``peer_id`` is the
key the registry files the peer under, ``type`` and ``hostname`` are what
:func:`~strands_robots.mesh.session.update_peer` derived from the wire,
``reachable`` is the verdict derived from the local heartbeat reading, and
``age`` is this process's own reading of when it last heard a heartbeat --
described on ``last_seen_mono`` as "a local observation, never a stamp the peer
sent", and named in ``docs/mesh.md`` among the things the mesh decides from a
duration on ``time.monotonic()`` that no clock correction can move.

Spread last, ``caps`` overrode all five. This pins that the local reading wins a
name collision, and that every capability key still merges.
"""

from __future__ import annotations

import ast
import inspect
import json
import textwrap
import time
import types
from typing import Any

import pytest

from strands_robots.mesh import session as mesh_session

# The five names ``to_dict`` decides locally rather than reading off the wire.
LOCAL_FIELDS = ("peer_id", "type", "hostname", "age", "reachable")


def _peer(**caps: Any) -> mesh_session.PeerInfo:
    """A peer last heard from 5s ago, carrying *caps* as its presence payload."""
    return mesh_session.PeerInfo(
        peer_id="arm-1",
        peer_type="robot",
        hostname="jetson-01",
        last_seen_mono=time.monotonic() - 5.0,
        caps=dict(caps),
    )


class _Sample:
    """A Zenoh sample carrying *payload* as JSON, as ``_on_presence`` reads it."""

    def __init__(self, payload: dict[str, Any]) -> None:
        self._raw = json.dumps(payload).encode()

    @property
    def payload(self) -> Any:
        return types.SimpleNamespace(to_bytes=lambda: self._raw)


@pytest.fixture
def registry() -> Any:
    """The process-wide peer registry, emptied before and after."""
    mesh_session._PEERS.clear()
    yield mesh_session
    mesh_session._PEERS.clear()


class TestTheLocalObservationWinsAName:
    """A payload key that collides with a locally decided field is not honoured."""

    def test_a_payload_carrying_age_does_not_replace_the_local_reading(self) -> None:
        peer = _peer(robot_id="arm-1", age=0.0)

        reported = peer.to_dict()["age"]

        assert reported == pytest.approx(5.0, abs=0.5), (
            f"the peer last heartbeat 5s ago and its own presence payload said age=0.0; "
            f"to_dict reported {reported!r}. An age the sender chooses is not an "
            f"observation, and every staleness verdict is read from it."
        )

    def test_a_payload_carrying_peer_id_does_not_rekey_the_peer(self) -> None:
        peer = _peer(robot_id="arm-1", peer_id="some-other-robot")

        assert peer.to_dict()["peer_id"] == "arm-1", (
            "peer_id is the key the registry files the peer under and the key "
            "Mesh.peers_by_id / Mesh.get_peer look it up by."
        )

    def test_a_payload_carrying_type_does_not_relabel_the_peer(self) -> None:
        peer = _peer(robot_id="arm-1", robot_type="robot", type="agent")

        assert peer.to_dict()["type"] == "robot", (
            "type is what update_peer derived from the payload's robot_type; a "
            "second key spelled 'type' is not a second opinion about it."
        )

    @pytest.mark.parametrize("field", LOCAL_FIELDS)
    def test_no_locally_decided_field_is_settable_from_the_payload(self, field: str) -> None:
        """Whatever the payload says, the five fields keep the local answer."""
        sentinel = "PAYLOAD-WINS" if field != "age" else -999.0
        peer = _peer(robot_id="arm-1", **{field: sentinel})

        assert peer.to_dict()[field] != sentinel, f"the payload set {field!r} to its own value"


class TestTheRegistryAndTheSerialisedViewAgree:
    """Every peer the registry holds is reachable under the id it is filed under."""

    def test_every_registered_peer_id_survives_serialisation(self, registry: Any) -> None:
        registry.update_peer(peer_id="arm-1", peer_type="robot", hostname="h", caps={"robot_id": "arm-1"})
        registry.update_peer(
            peer_id="arm-2",
            peer_type="robot",
            hostname="h",
            caps={"robot_id": "arm-2", "peer_id": "arm-1"},
        )

        filed = sorted(registry._PEERS)
        served = sorted(p["peer_id"] for p in registry.get_peers())

        assert served == filed, (
            f"the registry holds {filed} but get_peers reported {served}: a payload key "
            f"renamed a peer, so two entries claim one id and the other peer is unreachable."
        )

    def test_a_masquerading_peer_is_still_reachable_by_its_own_id(self, registry: Any) -> None:
        registry.update_peer(peer_id="arm-1", peer_type="robot", hostname="h", caps={"robot_id": "arm-1"})
        registry.update_peer(
            peer_id="arm-2",
            peer_type="robot",
            hostname="h",
            caps={"robot_id": "arm-2", "peer_id": "arm-1", "type": "agent"},
        )

        by_id = {p["peer_id"]: p for p in registry.get_peers()}

        assert set(by_id) == {"arm-1", "arm-2"}, f"only {sorted(by_id)} could be looked up"
        assert by_id["arm-1"]["type"] == "robot", "arm-2's payload answered a lookup for arm-1"


class TestWhatAMeshConsumerSees:
    """The registry reaches ``Mesh.peers`` / ``peers_by_id`` / ``get_peer`` intact."""

    @staticmethod
    def _mesh(peer_id: str = "me") -> Any:
        from strands_robots.mesh import core as mesh_core

        mesh = mesh_core.Mesh.__new__(mesh_core.Mesh)
        mesh.peer_id = peer_id
        mesh.peer_type = "robot"
        return mesh

    def test_get_peer_resolves_a_peer_whose_payload_claims_another_id(self, registry: Any) -> None:
        mesh = self._mesh()
        now = time.time()
        mesh._on_presence(_Sample({"robot_id": "arm-1", "robot_type": "robot", "hostname": "h", "timestamp": now}))
        mesh._on_presence(
            _Sample(
                {
                    "robot_id": "arm-2",
                    "robot_type": "robot",
                    "hostname": "h",
                    "timestamp": now,
                    "peer_id": "arm-1",
                }
            )
        )
        assert set(registry._PEERS) == {"arm-1", "arm-2"}, "premise: both peers were admitted"

        assert mesh.get_peer("arm-2") is not None, (
            "arm-2 is in the registry but get_peer could not find it: its own payload filed it under arm-1's id."
        )
        assert sorted(mesh.peers_by_id) == ["arm-1", "arm-2"]

    def test_a_peer_claiming_our_own_id_is_still_reported(self, registry: Any) -> None:
        """``Mesh.peers`` excludes *us* by peer_id, so a payload could hide a peer."""
        mesh = self._mesh(peer_id="me")
        mesh._on_presence(
            _Sample(
                {
                    "robot_id": "ghost",
                    "robot_type": "robot",
                    "hostname": "h",
                    "timestamp": time.time(),
                    "peer_id": "me",
                }
            )
        )
        assert "ghost" in registry._PEERS, "premise: the peer was admitted to the registry"

        assert [p["peer_id"] for p in mesh.peers] == ["ghost"], (
            "a peer that names our own peer_id in its payload was filtered out of our "
            "own fleet view while staying in the registry."
        )


class TestNothingElseChanges:
    """Capability keys still merge, and an honest payload is untouched."""

    def test_capability_keys_still_merge(self) -> None:
        peer = _peer(tool_name="unitree_g1", connected=True, cameras=["front"])
        served = peer.to_dict()

        assert served["tool_name"] == "unitree_g1"
        assert served["connected"] is True
        assert served["cameras"] == ["front"]

    def test_an_honest_presence_payload_reports_every_key(self) -> None:
        """The vocabulary ``_build_presence`` emits, none of which decides a local field."""
        honest = {
            "robot_id": "arm-1",
            "robot_type": "robot",
            "hostname": "jetson-01",
            "timestamp": time.time(),
            "tool_name": "so101",
            "task_status": "idle",
            "instruction": "",
            "connected": True,
            "hw": "so101",
            "cameras": ["front"],
            "inputs": ["leader"],
            "topics": ["state"],
        }
        served = _peer(**honest).to_dict()

        for key, value in honest.items():
            assert served[key] == value, f"{key} did not survive"
        assert served["peer_id"] == "arm-1"
        assert served["type"] == "robot"

    def test_the_only_colliding_key_an_honest_payload_carries_is_hostname(self) -> None:
        """``hostname`` is wire-derived by ``update_peer``, so the collision is a no-op."""
        emitted = {
            "robot_id",
            "robot_type",
            "hostname",
            "timestamp",
            "tool_name",
            "task_status",
            "instruction",
            "connected",
            "hw",
            "cameras",
            "inputs",
            "topics",
        }

        assert emitted & set(LOCAL_FIELDS) == {"hostname"}

    def test_a_wire_hostname_still_reaches_the_serialised_view(self, registry: Any) -> None:
        registry.update_peer(
            peer_id="arm-1",
            peer_type="robot",
            hostname="jetson-01",
            caps={"robot_id": "arm-1", "hostname": "jetson-01"},
        )

        (served,) = registry.get_peers()

        assert served["hostname"] == "jetson-01"

    def test_age_is_still_a_rounded_non_negative_duration(self) -> None:
        served = _peer(robot_id="arm-1").to_dict()

        assert isinstance(served["age"], float)
        assert served["age"] == round(served["age"], 1)
        assert served["age"] >= 0.0


class TestTheOrderIsTheContract:
    """A future edit cannot reintroduce the override by moving the spread."""

    def test_caps_merges_before_every_locally_decided_field(self) -> None:
        source = textwrap.dedent(inspect.getsource(mesh_session.PeerInfo.to_dict))
        returned = next(
            node.value
            for node in ast.walk(ast.parse(source))
            if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict)
        )

        # ``{**caps, ...}`` parses as a Dict whose first key is None.
        spread_at = [i for i, key in enumerate(returned.keys) if key is None]
        local_at = [
            i for i, key in enumerate(returned.keys) if isinstance(key, ast.Constant) and key.value in LOCAL_FIELDS
        ]

        assert spread_at, "to_dict no longer merges caps at all"
        assert local_at, "to_dict reports none of the locally decided fields"
        assert set(LOCAL_FIELDS) <= {key.value for key in returned.keys if isinstance(key, ast.Constant)}, (
            "to_dict stopped reporting one of the locally decided fields"
        )
        assert max(spread_at) < min(local_at), (
            "caps is spread after a locally decided field, so a presence payload "
            "carrying that name replaces the local reading again."
        )
