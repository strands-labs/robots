"""Regression tests for the ``lerobot_async`` policy provider.

``LerobotAsyncPolicy`` is a gRPC client to a lerobot ``async_inference``
``PolicyServer``. These tests pin two layers:

* Construction / validation and registry resolution (no gRPC needed) - so the
  provider is discoverable via ``create_policy`` and rejects a bad config early.
* A full wire round-trip against a real in-process gRPC ``AsyncInference``
  server (a stand-in that returns a canned action chunk). This proves the
  client speaks lerobot's actual protocol: the ``SendPolicyInstructions``
  handshake carries a well-formed ``RemotePolicyConfig`` (state + camera
  features), the observation is streamed as a ``must_go`` ``TimedObservation``,
  and the returned ``TimedAction`` chunk is decoded into per-joint action dicts.

Before this provider existed, ``create_policy("lerobot_async", ...)`` raised
``Unknown policy provider`` (the name was advertised but phantom), so every
assertion here fails on pre-fix code.
"""

from __future__ import annotations

import pickle
import sys
import time
from concurrent import futures

import numpy as np
import pytest

from strands_robots.policies import create_policy, list_providers
from strands_robots.policies.lerobot_async import LerobotAsyncPolicy

STATE_KEYS = [f"joint_{i}" for i in range(6)]


# -- Construction / validation (no gRPC) --------------------------------------


def test_provider_is_registered() -> None:
    assert "lerobot_async" in list_providers()


def test_create_via_name() -> None:
    policy = create_policy(
        "lerobot_async",
        server_address="gpu-box:8080",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.provider_name == "lerobot_async"
    assert policy.server_address == "gpu-box:8080"


def test_create_via_grpc_smart_string_parses_address() -> None:
    policy = create_policy(
        "grpc://gpu-box:9000",
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.server_address == "gpu-box:9000"


def test_actions_per_step_defaults_to_chunk_size() -> None:
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        actions_per_chunk=32,
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.actions_per_step == 32
    assert policy.execution_horizon == 32


def test_rename_map_defaults_to_empty() -> None:
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.rename_map == {}


def test_rename_map_is_stored() -> None:
    """A supplied rename_map is honored (not swallowed by the ignored-kwargs bag)."""
    mapping = {"observation.images.front": "observation.images.laptop"}
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        rename_map=mapping,
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.rename_map == mapping
    # A copy, not the caller's object.
    assert policy.rename_map is not mapping


def test_rename_map_invalid_type_raises() -> None:
    with pytest.raises(ValueError, match="rename_map must be a dict"):
        create_policy(
            "lerobot_async",
            server_address="h:1",
            policy_type="act",
            pretrained_name_or_path="x/y",
            rename_map=["observation.images.front"],
        )


def test_missing_policy_type_raises() -> None:
    with pytest.raises(ValueError, match="policy_type"):
        create_policy("lerobot_async", server_address="h:1", pretrained_name_or_path="x/y")


def test_unsupported_policy_type_raises() -> None:
    with pytest.raises(ValueError, match="not served"):
        create_policy("lerobot_async", server_address="h:1", policy_type="bogus", pretrained_name_or_path="x/y")


def test_missing_checkpoint_raises() -> None:
    with pytest.raises(ValueError, match="pretrained_name_or_path"):
        create_policy("lerobot_async", server_address="h:1", policy_type="act")


def test_server_address_accepts_url_with_scheme() -> None:
    """A ``scheme://host:port`` server_address has its scheme stripped.

    The gRPC channel target is a bare ``host:port``; passing a full URL (e.g.
    copied from a dashboard) must not leak the ``grpc://`` scheme into the
    channel address.
    """
    policy = create_policy(
        "lerobot_async",
        server_address="grpc://gpu-box:9000",
        policy_type="act",
        pretrained_name_or_path="x/y",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.server_address == "gpu-box:9000"


def test_unexpected_constructor_kwargs_warn_not_raise(caplog: pytest.LogCaptureFixture) -> None:
    """Unknown constructor kwargs are tolerated with a warning, never fatal.

    ``create_policy`` forwards a shared kwargs bag to every provider; the async
    client must ignore server-side config it does not own (and say so) rather
    than crash the caller.
    """
    import logging

    with caplog.at_level(logging.WARNING):
        policy = create_policy(
            "lerobot_async",
            server_address="h:1",
            policy_type="act",
            pretrained_name_or_path="x/y",
            temperature=0.7,  # server-side policy config, not a client kwarg
        )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert any("ignoring unexpected constructor kwarg" in r.message for r in caplog.records)
    assert "temperature" in caplog.text


def test_chunk_decode_rejects_empty_state_keys_and_empty_chunk() -> None:
    """``_chunk_to_action_dicts`` guards both undeclared keys and an empty chunk.

    These are defensive contracts on the decode path: without declared state
    keys the index->joint mapping is undefined, and an empty decoded chunk must
    raise rather than silently return no actions.
    """
    policy = create_policy(
        "lerobot_async",
        server_address="h:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
    )
    assert isinstance(policy, LerobotAsyncPolicy)

    # No state keys declared -> undefined index mapping.
    with pytest.raises(RuntimeError, match="robot_state_keys is empty"):
        policy._chunk_to_action_dicts([])

    # Keys declared but the server chunk decoded to nothing -> never fabricate.
    policy.set_robot_state_keys(STATE_KEYS)
    with pytest.raises(RuntimeError, match="empty action chunk"):
        policy._chunk_to_action_dicts([])


# -- Full gRPC round-trip against a real in-process AsyncInference server ------

grpc = pytest.importorskip("grpc")
pytest.importorskip("lerobot.transport")
pytest.importorskip("lerobot.async_inference.helpers")

from lerobot.async_inference.helpers import (  # noqa: E402
    RemotePolicyConfig,
    TimedAction,
    TimedObservation,
)
from lerobot.transport import services_pb2, services_pb2_grpc  # noqa: E402
from lerobot.transport.utils import receive_bytes_in_chunks  # noqa: E402

CHUNK_LEN = 4
ACTION_DIM = 6


class _RecordingServicer(services_pb2_grpc.AsyncInferenceServicer):
    """Minimal real gRPC AsyncInference server that returns a canned chunk.

    Records the ``RemotePolicyConfig`` and ``TimedObservation`` the client
    sends so the test can assert the client built the wire messages correctly,
    then returns a deterministic ``list[TimedAction]`` on ``GetActions``.
    """

    def __init__(self) -> None:
        self.policy_config: RemotePolicyConfig | None = None
        self.observation: TimedObservation | None = None
        self.ready_calls = 0
        self.return_empty_actions = False

    def Ready(self, request, context):  # noqa: N802
        self.ready_calls += 1
        return services_pb2.Empty()

    def SendPolicyInstructions(self, request, context):  # noqa: N802
        self.policy_config = pickle.loads(request.data)  # nosec B301
        return services_pb2.Empty()

    def SendObservations(self, request_iterator, context):  # noqa: N802
        import threading

        payload = receive_bytes_in_chunks(request_iterator, None, threading.Event(), "test")
        self.observation = pickle.loads(payload)  # nosec B301
        return services_pb2.Empty()

    def GetActions(self, request, context):  # noqa: N802
        import torch

        if self.return_empty_actions:
            return services_pb2.Empty()

        chunk = [
            TimedAction(
                timestamp=time.time(),
                timestep=i,
                action=torch.arange(ACTION_DIM, dtype=torch.float32) + float(i),
            )
            for i in range(CHUNK_LEN)
        ]
        return services_pb2.Actions(data=pickle.dumps(chunk))  # nosec B301


@pytest.fixture()
def running_server():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = _RecordingServicer()
    services_pb2_grpc.add_AsyncInferenceServicer_to_server(servicer, server)
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    try:
        yield servicer, f"127.0.0.1:{port}"
    finally:
        server.stop(grace=None)


def _client(address: str, **kwargs) -> LerobotAsyncPolicy:
    """Create the provider via the registry and narrow the type for the checker."""
    policy = create_policy("lerobot_async", server_address=address, **kwargs)
    assert isinstance(policy, LerobotAsyncPolicy)
    return policy


def _observation() -> dict:
    obs: dict = {key: 0.1 * i for i, key in enumerate(STATE_KEYS)}
    obs["top"] = np.zeros((8, 8, 3), dtype=np.uint8)
    return obs


def test_roundtrip_returns_decoded_action_chunk(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)

    actions = policy.get_actions_sync(_observation(), "pick up the cube")

    # The canned chunk had CHUNK_LEN actions of [0..5]+i; decoded to dicts.
    assert len(actions) == CHUNK_LEN
    for i, action in enumerate(actions):
        assert set(action) == set(STATE_KEYS)
        assert action["joint_0"] == pytest.approx(float(i))
        assert action["joint_5"] == pytest.approx(5.0 + i)
    policy.close()


def test_roundtrip_sends_wellformed_instructions_and_observation(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    policy.get_actions_sync(_observation(), "pick up the cube")

    cfg = servicer.policy_config
    assert isinstance(cfg, RemotePolicyConfig)
    assert cfg.policy_type == "act"
    assert cfg.pretrained_name_or_path == "lerobot/act_so101"
    assert cfg.actions_per_chunk == CHUNK_LEN
    assert cfg.device == "cpu"
    # State scalars concatenated into observation.state (order preserved) +
    # the camera declared as an image feature.
    assert cfg.lerobot_features["observation.state"]["names"] == STATE_KEYS
    assert "observation.images.top" in cfg.lerobot_features

    obs = servicer.observation
    assert isinstance(obs, TimedObservation)
    assert obs.must_go is True
    assert obs.observation["task"] == "pick up the cube"
    assert obs.observation["joint_3"] == pytest.approx(0.3)
    assert obs.observation["top"].shape == (8, 8, 3)
    policy.close()


def test_roundtrip_forwards_rename_map_to_server(running_server) -> None:
    """rename_map reaches the server's RemotePolicyConfig so it can remap obs keys.

    The async transport applies rename_map server-side (as a
    RenameObservationsProcessorStep), so a checkpoint that expects different
    camera/state keys than the robot exposes is only reachable if the client
    forwards the map. Without forwarding, the server receives the empty default
    and cannot decode a mismatched observation.
    """
    servicer, address = running_server
    mapping = {"top": "observation.images.laptop"}
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
        rename_map=mapping,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    policy.get_actions_sync(_observation(), "pick up the cube")

    cfg = servicer.policy_config
    assert isinstance(cfg, RemotePolicyConfig)
    assert cfg.rename_map == mapping
    policy.close()


def test_roundtrip_default_rename_map_is_empty(running_server) -> None:
    """With no rename_map the server receives the empty default (no accidental remap)."""
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    policy.get_actions_sync(_observation(), "task")
    assert servicer.policy_config.rename_map == {}
    policy.close()


def test_server_empty_response_raises(running_server) -> None:
    """If the server yields no actions, the client must raise, never fabricate zeros."""
    servicer, address = running_server
    servicer.return_empty_actions = True

    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
    )
    policy.set_robot_state_keys(STATE_KEYS)
    with pytest.raises(RuntimeError, match="no actions"):
        policy.get_actions_sync(_observation(), "task")
    policy.close()


def test_missing_state_key_in_observation_raises(running_server) -> None:
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
    )
    policy.set_robot_state_keys(STATE_KEYS)
    incomplete = {key: 0.0 for key in STATE_KEYS[:-1]}  # drop last joint
    with pytest.raises(RuntimeError, match="missing declared state key"):
        policy.get_actions_sync(incomplete, "task")
    policy.close()


def test_unreachable_server_raises_connectionerror() -> None:
    # Reserved TEST-NET address / closed port -> Ready handshake fails fast.
    policy = _client(
        "127.0.0.1:1",
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
        connect_timeout=2.0,
    )
    policy.set_robot_state_keys(STATE_KEYS)
    with pytest.raises(ConnectionError, match="could not reach a lerobot PolicyServer"):
        policy.get_actions_sync(_observation(), "task")
    policy.close()


def test_reset_recalls_ready_and_reconnect_is_idempotent(running_server) -> None:
    """reset() re-runs the server Ready handshake; a second call reuses the channel.

    The client keeps the loaded policy resident across episodes: reset() flushes
    server-side per-episode state via ``Ready`` (without a reconnect), and the
    already-open channel is reused on subsequent inference (no duplicate dial).
    """
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="lerobot/act_so101",
        device="cpu",
        actions_per_chunk=CHUNK_LEN,
    )
    policy.set_robot_state_keys(STATE_KEYS)

    # First inference connects (one Ready during the connect handshake).
    policy.get_actions_sync(_observation(), "task")
    assert servicer.ready_calls == 1

    # reset() re-runs Ready on the live stub to clear the server episode state.
    policy.reset()
    assert servicer.ready_calls == 2

    # A second inference reuses the open channel (no third Ready handshake).
    policy.get_actions_sync(_observation(), "task")
    assert servicer.ready_calls == 2
    policy.close()


def test_inference_without_state_keys_raises(running_server) -> None:
    """Connecting then inferring without declared state keys raises, never guesses.

    The feature spec sent to the server is built from the joint state keys; with
    none declared the client must refuse rather than send an empty spec.
    """
    servicer, address = running_server
    policy = _client(
        address,
        policy_type="act",
        pretrained_name_or_path="x/y",
        device="cpu",
    )
    # set_robot_state_keys intentionally not called.
    with pytest.raises(RuntimeError, match="robot_state_keys is empty"):
        policy.get_actions_sync(_observation(), "task")
    policy.close()


# -- lerobot drift guard: SUPPORTED_POLICY_TYPES is live-sourced -----------------
#
# The set of policy types a lerobot ``PolicyServer`` can serve lives in lerobot's
# own ``async_inference.constants.SUPPORTED_POLICIES``. The client validation
# must track that constant, not a hand-copied list: a stale copy either rejects a
# newly-servable policy client-side or accepts a dropped one only to fail after
# the gRPC handshake. These pin that the copy cannot silently drift.

from strands_robots.policies.lerobot_async import policy as async_policy  # noqa: E402


def test_supported_policy_types_sourced_live_from_lerobot() -> None:
    """The client's supported set is the live lerobot constant, not a fixed copy."""
    constants = pytest.importorskip("lerobot.async_inference.constants")
    live = getattr(constants, "SUPPORTED_POLICIES", None)
    if live is None:
        pytest.skip("lerobot async SUPPORTED_POLICIES not available")
    assert set(async_policy._supported_policy_types()) == set(live)
    assert set(async_policy.SUPPORTED_POLICY_TYPES) == set(live)


def test_fallback_is_a_faithful_snapshot_of_lerobot() -> None:
    """The offline fallback stays in sync with lerobot; drift trips this test."""
    constants = pytest.importorskip("lerobot.async_inference.constants")
    live = getattr(constants, "SUPPORTED_POLICIES", None)
    if live is None:
        pytest.skip("lerobot async SUPPORTED_POLICIES not available")
    assert set(async_policy._SUPPORTED_POLICY_TYPES_FALLBACK) == set(live), (
        "_SUPPORTED_POLICY_TYPES_FALLBACK drifted from lerobot's SUPPORTED_POLICIES; "
        "update the fallback tuple to match the current lerobot constant."
    )


def test_supported_policy_types_falls_back_when_lerobot_constants_unimportable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail-soft: an unimportable lerobot constants module yields the snapshot.

    ``_supported_policy_types`` sources the servable set live from
    ``lerobot.async_inference.constants.SUPPORTED_POLICIES``. If lerobot moves or
    renames that module (a real risk when tracking lerobot from source), the
    client must degrade to :data:`_SUPPORTED_POLICY_TYPES_FALLBACK` rather than
    raise at import time - otherwise merely importing the provider would crash.
    The ``None`` sentinel in ``sys.modules`` forces the ``from ... import`` to
    raise ``ImportError`` even though lerobot is installed here.
    """
    monkeypatch.setitem(sys.modules, "lerobot.async_inference.constants", None)
    assert async_policy._supported_policy_types() == async_policy._SUPPORTED_POLICY_TYPES_FALLBACK


def test_validation_tracks_a_newly_added_lerobot_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    """A policy type lerobot adds to SUPPORTED_POLICIES is accepted immediately.

    Proves the client-side check reads lerobot live rather than a frozen copy: a
    type that only exists in lerobot's (here monkeypatched) constant passes
    validation instead of being wrongly rejected as "not served".
    """
    constants = pytest.importorskip("lerobot.async_inference.constants")
    live = getattr(constants, "SUPPORTED_POLICIES", None)
    if live is None:
        pytest.skip("lerobot async SUPPORTED_POLICIES not available")
    fake = "fake_async_policy_xyz"
    monkeypatch.setattr(constants, "SUPPORTED_POLICIES", tuple(sorted({*live, fake})))

    assert fake in async_policy._supported_policy_types()
    # create_policy must NOT reject the freshly-supported type (lazy connect, so
    # no server is contacted here).
    policy = create_policy(
        "lerobot_async",
        server_address="gpu-box:8080",
        policy_type=fake,
        pretrained_name_or_path="x/y",
    )
    assert isinstance(policy, LerobotAsyncPolicy)
    assert policy.policy_type == fake
