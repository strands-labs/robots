"""End-to-end client/server tests for remote policy inference.

A :class:`PolicyServer` wrapping a recording policy runs in a background
thread; a :class:`RemotePolicy` client drives it over a real loopback
WebSocket. These assert the observable contract - metadata mirroring, the
observation->chunk round-trip, config/reset/RTC forwarding, and loud error
propagation - not internal state.
"""

import logging
import socket
from typing import Any

import numpy as np
import pytest

from strands_robots.inference import PolicyServer, RemotePolicy
from strands_robots.policies import create_policy
from strands_robots.policies.base import Policy


class RecordingPolicy(Policy):
    """Chunk-emitting RTC policy that records everything the server forwards it."""

    def __init__(self, *, fail: bool = False) -> None:
        self.actions_per_step = 8
        self.supports_rtc = True
        self.robot_state_keys: list[str] = ["j0", "j1", "j2"]
        self.reset_seeds: list[int | None] = []
        self.seen_rtc_delays: list[int | None] = []
        self.seen_instructions: list[str] = []
        self.seen_kwargs: list[dict[str, Any]] = []
        self._fail = fail

    @property
    def provider_name(self) -> str:
        return "recording"

    @property
    def requires_images(self) -> bool:
        return False

    @property
    def execution_horizon(self) -> int:
        # RTC re-query interval, deliberately smaller than the trained chunk.
        return 4

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        self.robot_state_keys = list(robot_state_keys)

    def reset(self, seed: int | None = None) -> None:
        self.reset_seeds.append(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        if self._fail:
            raise RuntimeError("boom: deliberate inference failure")
        # Record what the server handed us (RTC delay is set on the instance
        # by the server immediately before this call).
        self.seen_rtc_delays.append(self.rtc_observed_delay_steps)
        self.seen_instructions.append(instruction)
        self.seen_kwargs.append(dict(kwargs))
        state = observation_dict.get("observation.state", [0.0] * len(self.robot_state_keys))
        base = float(np.asarray(state).sum())
        return [{key: base + i * 0.01 for key in self.robot_state_keys} for i in range(self.actions_per_step)]


class WritabilityAssertingPolicy(Policy):
    """Records whether observation arrays received over the wire are writable.

    A locally-run policy always receives writable observation arrays (a freshly
    rendered camera frame / state vector); the remote path must not silently
    deliver a read-only array, which breaks in-place normalization.
    """

    def __init__(self) -> None:
        self.image_writable: bool | None = None

    @property
    def provider_name(self) -> str:
        return "writability-asserting"

    @property
    def requires_images(self) -> bool:
        return True

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        pass

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        image = observation_dict["observation.images.cam"]
        self.image_writable = bool(image.flags.writeable)
        # In-place normalization is the common VLA preprocessing step; it raises
        # "output array is read-only" if the wire delivered a read-only buffer.
        image += 1
        return [{"j0": 0.0}]


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.fixture
def served_recording():
    """Yield ``(server, policy, RemotePolicy)`` over a live loopback WebSocket."""
    policy = RecordingPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{server.port}")
    # Trigger the lazy connect + handshake so the config-forwarding tests below
    # exercise the live path (pre-connect buffering is covered separately by
    # test_config_set_before_connect_is_replayed_on_connect).
    assert client.requires_images is False
    try:
        yield server, policy, client
    finally:
        client.close()
        server.stop()


def test_client_mirrors_server_metadata(served_recording):
    _server, _policy, client = served_recording
    # Accessing an introspection property triggers the lazy connect + handshake.
    assert client.requires_images is False
    assert client.execution_horizon == 4
    assert client.actions_per_step == 8
    assert client.supports_rtc is True
    assert client.provider_name == "remote"
    assert client.remote_provider_name == "recording"


def test_observation_to_chunk_roundtrip(served_recording):
    _server, policy, client = served_recording
    client.set_robot_state_keys(["j0", "j1", "j2"])
    obs = {"observation.state": np.array([1.0, 2.0, 3.0], dtype=np.float32)}
    chunk = client.get_actions_sync(obs, "pick the red cube")

    assert len(chunk) == policy.actions_per_step
    assert set(chunk[0].keys()) == {"j0", "j1", "j2"}
    # base == sum(state) == 6.0, first action offset 0.0
    assert chunk[0]["j0"] == pytest.approx(6.0)
    assert policy.seen_instructions[-1] == "pick the red cube"


def test_set_robot_state_keys_forwarded(served_recording):
    _server, policy, client = served_recording
    client.set_robot_state_keys(["a", "b"])
    assert policy.robot_state_keys == ["a", "b"]


def test_set_control_frequency_forwarded_and_validated(served_recording):
    _server, policy, client = served_recording
    client.set_control_frequency(50.0)
    assert policy.control_frequency == 50.0
    with pytest.raises(ValueError, match="must be positive"):
        client.set_control_frequency(0.0)


def test_reset_forwarded_to_server_policy(served_recording):
    _server, policy, client = served_recording
    client.reset(seed=123)
    assert policy.reset_seeds[-1] == 123


def test_rtc_observed_delay_forwarded_per_request(served_recording):
    """The runner-counted RTC delay must reach the server policy each request."""
    _server, policy, client = served_recording
    client.set_rtc_observed_delay(3)
    client.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, "")
    assert policy.seen_rtc_delays[-1] == 3

    client.set_rtc_observed_delay(None)
    client.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, "")
    assert policy.seen_rtc_delays[-1] is None


def test_extra_kwargs_forwarded(served_recording):
    _server, policy, client = served_recording
    client.get_actions_sync({"observation.state": [0.0, 0.0, 0.0]}, "", target_joints={"j0": 0.5})
    assert policy.seen_kwargs[-1] == {"target_joints": {"j0": 0.5}}


def test_server_error_propagates_to_client():
    policy = RecordingPolicy(fail=True)
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{server.port}")
    try:
        with pytest.raises(RuntimeError, match="deliberate inference failure"):
            client.get_actions_sync({"observation.state": [0.0]}, "")
    finally:
        client.close()
        server.stop()


def test_config_set_before_connect_is_replayed_on_connect():
    """State keys / control freq / reset set pre-connection must reach the server."""
    policy = RecordingPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{server.port}")
    try:
        # No network yet: these are buffered client-side.
        client.set_robot_state_keys(["x", "y"])
        client.set_control_frequency(30.0)
        client.reset(seed=7)
        # First inference triggers connect, which must flush the buffered config.
        client.get_actions_sync({"observation.state": [1.0, 1.0]}, "")
        assert policy.robot_state_keys == ["x", "y"]
        assert policy.control_frequency == 30.0
        assert policy.reset_seeds[-1] == 7
    finally:
        client.close()
        server.stop()


def test_unreachable_server_raises_actionable_connection_error():
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{_free_port()}", connect_timeout=1.0)
    with pytest.raises(ConnectionError, match="could not reach a PolicyServer"):
        client.get_actions_sync({"observation.state": [0.0]}, "")


def test_create_policy_smart_string_builds_remote_policy():
    policy = create_policy("ws://gpu-box:8765")
    assert isinstance(policy, RemotePolicy)
    assert policy.uri == "ws://gpu-box:8765"


def test_create_policy_named_remote_provider_with_endpoint():
    policy = create_policy("remote", endpoint="wss://secure-box:9000")
    assert isinstance(policy, RemotePolicy)
    assert policy.uri == "wss://secure-box:9000"


def test_wrong_named_endpoint_kwarg_warns_instead_of_silently_defaulting(caplog):
    """A connection endpoint passed under the wrong kwarg name is tolerated (the
    cross-provider ignore-unknown-constructor-kwargs contract) but MUST NOT be
    dropped silently: the client would otherwise connect to the default
    ws://127.0.0.1:8765 with no hint that the intended endpoint never took
    effect, surfacing only as a confusing "connection refused" to a port the
    user never chose (cf. the #317 no-silent-localhost-default fix). ``uri`` is
    this object's own attribute name -> the single most likely such slip.
    """
    with caplog.at_level(logging.WARNING, logger="strands_robots.inference.client"):
        policy = RemotePolicy(uri="ws://gpu-box:9000")
    # The mistyped endpoint was ignored -> the client fell back to the default.
    assert policy.uri == "ws://127.0.0.1:8765"
    # ...but the drop is now surfaced, naming the offending kwarg + the endpoint.
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("uri" in m and "ws://127.0.0.1:8765" in m for m in warnings), warnings


def test_known_constructor_kwargs_do_not_warn(caplog):
    """The warning fires only for genuinely-unrecognized kwargs; the documented
    endpoint/host/port/timeout kwargs stay quiet so the diagnostic is high-signal
    (the normal ``create_policy("ws://...")`` path passes only known kwargs)."""
    with caplog.at_level(logging.WARNING, logger="strands_robots.inference.client"):
        policy = RemotePolicy(endpoint="ws://gpu-box:9000", connect_timeout=2.0, request_timeout=5.0)
    assert policy.uri == "ws://gpu-box:9000"
    assert not any("ignoring unexpected constructor" in r.getMessage() for r in caplog.records), [
        r.getMessage() for r in caplog.records
    ]


def test_policy_server_requires_exactly_one_policy_source():
    with pytest.raises(ValueError, match="exactly one"):
        PolicyServer()
    with pytest.raises(ValueError, match="exactly one"):
        PolicyServer(policy=RecordingPolicy(), policy_provider="mock")


def test_observation_delivered_to_server_policy_is_writable():
    """The observation the server hands the wrapped policy must be writable.

    Regression: the wire codec decoded arrays via ``np.frombuffer`` on immutable
    ``bytes``, yielding read-only observations. In-place normalization inside the
    policy then raised on the server and propagated as a RuntimeError to the
    client - a divergence from the transparent local-inference path.
    """
    policy = WritabilityAssertingPolicy()
    server = PolicyServer(policy=policy, host="127.0.0.1", port=0).start()
    client = RemotePolicy(endpoint=f"ws://127.0.0.1:{server.port}")
    try:
        obs = {"observation.images.cam": np.zeros((8, 8, 3), dtype=np.uint8)}
        client.get_actions_sync(obs, "")
        assert policy.image_writable is True
    finally:
        client.close()
        server.stop()
