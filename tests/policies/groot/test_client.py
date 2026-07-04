"""Tests for strands_robots.policies.groot.client - ZMQ serialization and client.

Covers: MsgSerializer roundtrips, Gr00tInferenceClient construction, api_token
handling, and error paths.

Requires: msgpack, pyzmq (groot-service extras). Tests are skipped when not installed.
"""

import logging
from unittest.mock import MagicMock

import numpy as np
import pytest

msgpack = pytest.importorskip("msgpack", reason="msgpack not installed - pip install 'strands-robots[groot-service]'")
zmq = pytest.importorskip("zmq", reason="zmq not installed - pip install 'strands-robots[groot-service]'")

# E402: importorskip must execute before these imports to skip the module cleanly.
from strands_robots.policies.groot.client import Gr00tInferenceClient, MsgSerializer  # noqa: E402
from strands_robots.policies.groot.data_config import ModalityConfig  # noqa: E402

# (section)
# MsgSerializer
# (section)


class TestMsgSerializer:
    def test_roundtrip_plain_dict(self):
        data = {"endpoint": "ping", "data": {"foo": 42}}
        assert MsgSerializer.from_bytes(MsgSerializer.to_bytes(data)) == data

    def test_roundtrip_numpy_array(self):
        array = np.random.rand(3, 4).astype(np.float32)
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"obs": array}))
        np.testing.assert_array_almost_equal(result["obs"], array)

    def test_roundtrip_numpy_uint8(self):
        array = np.random.randint(0, 255, (2, 3, 3), dtype=np.uint8)
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"img": array}))
        np.testing.assert_array_equal(result["img"], array)

    def test_roundtrip_modality_config(self):
        config = ModalityConfig(delta_indices=[0, 1], modality_keys=["state.arm"])
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes({"config": config}))
        decoded = result["config"]
        assert isinstance(decoded, ModalityConfig)
        assert decoded.delta_indices == [0, 1]
        assert decoded.modality_keys == ["state.arm"]

    def test_decode_modality_config_n17_dict_form(self):
        """N1.7 server sends `as_json` as a dict (not a JSON string).

        This is the wire-format change between N1.6 and N1.7. The client must
        accept both so a single client binary can talk to either server.
        """
        # Hand-craft the exact bytes an N1.7 server would emit.
        wire = msgpack.packb(
            {
                "config": {
                    "__ModalityConfig_class__": True,
                    "as_json": {"delta_indices": [0, 1], "modality_keys": ["state.arm"]},
                }
            }
        )
        decoded = MsgSerializer.from_bytes(wire)["config"]
        assert isinstance(decoded, ModalityConfig)
        assert decoded.delta_indices == [0, 1]
        assert decoded.modality_keys == ["state.arm"]

    def test_decode_modality_config_n17_with_extra_fields(self):
        """N1.7's server-side ModalityConfig has extra optional fields.

        Our lightweight client-side dataclass only tracks ``delta_indices`` and
        ``modality_keys``.  Unknown fields in the wire payload must be silently
        dropped so clients don't break when NVIDIA adds new metadata in future
        N1.x releases.  This was discovered live against GR00T-N1.7-3B - the
        server sends ``sin_cos_embedding_keys``, ``mean_std_embedding_keys``,
        and ``action_configs`` on every response.
        """
        wire = msgpack.packb(
            {
                "config": {
                    "__ModalityConfig_class__": True,
                    "as_json": {
                        "delta_indices": [-20, 0],
                        "modality_keys": ["video.ego_view"],
                        "sin_cos_embedding_keys": None,
                        "mean_std_embedding_keys": None,
                        "action_configs": None,
                    },
                }
            }
        )
        decoded = MsgSerializer.from_bytes(wire)["config"]
        assert isinstance(decoded, ModalityConfig)
        assert decoded.delta_indices == [-20, 0]
        assert decoded.modality_keys == ["video.ego_view"]

    def test_decode_modality_config_n16_string_form(self):
        """N1.6 server sends `as_json` as a JSON string (Pydantic `model_dump_json`)."""
        import json as _json

        wire = msgpack.packb(
            {
                "config": {
                    "__ModalityConfig_class__": True,
                    "as_json": _json.dumps({"delta_indices": [-20, 0], "modality_keys": ["video.ego_view"]}),
                }
            }
        )
        decoded = MsgSerializer.from_bytes(wire)["config"]
        assert isinstance(decoded, ModalityConfig)
        assert decoded.delta_indices == [-20, 0]
        assert decoded.modality_keys == ["video.ego_view"]

    def test_roundtrip_nested_arrays(self):
        data = {
            "video": np.zeros((1, 1, 64, 64, 3), dtype=np.uint8),
            "state": np.ones((1, 1, 6), dtype=np.float32),
        }
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(data))
        assert result["video"].shape == (1, 1, 64, 64, 3)
        assert result["state"].dtype == np.float32

    def test_encode_non_custom_returns_as_is(self):
        data = {"key": "value", "num": 42, "list": [1, 2, 3]}
        result = MsgSerializer.from_bytes(MsgSerializer.to_bytes(data))
        assert result["key"] == "value"
        assert result["num"] == 42


# (section)
# Gr00tInferenceClient - construction & api_token
# (section)


class TestGr00tInferenceClient:
    def test_construction_defaults(self):
        client = Gr00tInferenceClient(host="localhost", port=5555)
        assert client.host == "localhost"
        assert client.port == 5555
        assert client.timeout_ms == 15000
        assert client.api_token is None

    def test_construction_with_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=5555, api_token="secret")
        assert client.api_token == "secret"

    def test_api_token_warning_on_remote_host(self, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.groot.client"):
            Gr00tInferenceClient(host="10.0.0.1", port=5555, api_token="tok")
        assert any("plaintext" in record.message for record in caplog.records)

    def test_no_warning_for_localhost_token(self, caplog):
        with caplog.at_level(logging.WARNING, logger="strands_robots.policies.groot.client"):
            Gr00tInferenceClient(host="localhost", port=5555, api_token="tok")
        assert not any("plaintext" in record.message for record in caplog.records)

    def test_custom_timeout(self):
        client = Gr00tInferenceClient(host="localhost", port=5555, timeout_ms=5000)
        assert client.timeout_ms == 5000

    def test_call_endpoint_includes_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=9999, api_token="mytoken")
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert len(sent_data) == 1
        assert sent_data[0]["api_token"] == "mytoken"

    def test_call_endpoint_without_api_token(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert "api_token" not in sent_data[0]

    def test_call_endpoint_raises_on_server_error(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"error": "bad input"}))
        with pytest.raises(RuntimeError, match="Server error: bad input"):
            client.call_endpoint("get_action", {"obs": {}})

    def test_ping_returns_false_on_failure(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.send = MagicMock(side_effect=Exception("timeout"))
        assert client.ping() is False

    def test_ping_does_not_reconnect(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        original_socket = client.socket
        client.socket.send = MagicMock(side_effect=Exception("timeout"))
        client.ping()
        assert client.socket is original_socket

    def test_get_action_calls_endpoint(self):
        """Accept both bare dict (legacy) and (action, info) tuple (N1.6/N1.7) responses."""
        client = Gr00tInferenceClient(host="localhost", port=9999)
        action = {"single_arm": np.zeros((1, 16, 5), dtype=np.float32)}
        # N1.6/N1.7 servers send (action, info).
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes((action, {})))
        result = client.get_action({"some": "obs"})
        assert "single_arm" in result

    def test_get_action_legacy_bare_dict(self):
        """Some older / custom servers return just the action dict."""
        client = Gr00tInferenceClient(host="localhost", port=9999)
        action = {"single_arm": np.zeros((1, 16, 5), dtype=np.float32)}
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes(action))
        result = client.get_action({"some": "obs"})
        assert "single_arm" in result

    def test_get_action_wraps_observation_envelope(self):
        """Verify that get_action wraps observations in {observation, options}.

        The N1.6/N1.7 PolicyServer spreads request['data'] as kwargs into
        ``policy.get_action(observation, options)``, so we must send the
        envelope or the server throws
        "got an unexpected keyword argument 'video'".
        """
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent = []
        client.socket.send = lambda data: sent.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes(({"joint": np.zeros(3)}, {})))
        client.get_action({"video": {"cam": np.zeros((1, 1, 4, 4, 3), dtype=np.uint8)}})
        data = sent[0]["data"]
        assert set(data.keys()) == {"observation", "options"}
        assert data["options"] is None
        assert "video" in data["observation"]

    def test_call_endpoint_data_none_omits_data_key(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("ping")
        assert "data" not in sent_data[0]

    def test_call_endpoint_data_present_includes_data_key(self):
        client = Gr00tInferenceClient(host="localhost", port=9999)
        sent_data = []
        client.socket.send = lambda data: sent_data.append(MsgSerializer.from_bytes(data))
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        client.call_endpoint("get_action", {"obs": "test"})
        assert "data" in sent_data[0]
        assert sent_data[0]["data"] == {"obs": "test"}


# (section)
# Dependency check
# (section)


class TestZmqDeps:
    def test_require_optional_loads_zmq(self):
        from strands_robots.utils import require_optional

        zmq = require_optional("zmq", pip_install="pyzmq")
        assert hasattr(zmq, "Context")

    def test_require_optional_loads_msgpack(self):
        from strands_robots.utils import require_optional

        msgpack = require_optional("msgpack")
        assert hasattr(msgpack, "packb")


# (section)
# Gr00tInferenceClient - reconnect & ping success
# (section)


class TestReconnect:
    def test_reconnect_replaces_socket_and_reconnects(self):
        """reconnect() closes the old socket and connects a fresh one.

        The new socket is a distinct object bound to the same host/port, so a
        client can recover from a broken connection without being reconstructed
        (host/port/timeout are preserved).
        """
        client = Gr00tInferenceClient(host="localhost", port=9999)
        old_socket = client.socket

        client.reconnect()

        assert old_socket.closed is True
        assert client.socket is not old_socket
        assert client.socket.closed is False
        assert client.host == "localhost"
        assert client.port == 9999
        client.socket.close(linger=0)

    def test_reconnect_tolerates_close_failure(self):
        """A failing close() on the stale socket must not abort reconnect.

        The socket may already be dead/closed; reconnect swallows the close
        error and still creates a working replacement socket.
        """
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.close(linger=0)  # dispose the real socket before swapping in a stub
        stale = MagicMock()
        stale.close.side_effect = RuntimeError("already closed")
        client.socket = stale

        client.reconnect()

        stale.close.assert_called_once()
        assert client.socket is not stale
        assert client.socket.closed is False
        client.socket.close(linger=0)


class TestPingSuccess:
    def test_ping_returns_true_when_server_responds(self):
        """ping() returns True when call_endpoint completes without error."""
        client = Gr00tInferenceClient(host="localhost", port=9999)
        client.socket.send = MagicMock()
        client.socket.recv = MagicMock(return_value=MsgSerializer.to_bytes({"status": "ok"}))
        assert client.ping() is True
        client.socket.close(linger=0)


# (section)
# MsgSerializer - hook passthrough contract
# (section)


class TestMsgSerializerPassthrough:
    def test_encode_unknown_type_is_left_for_msgpack_to_reject(self):
        """_encode returns unknown objects unchanged so msgpack raises TypeError.

        The ``default=`` hook only handles ModalityConfig and ndarray; any other
        non-native type must pass straight through, letting msgpack surface an
        explicit TypeError rather than silently corrupting the payload.
        """
        with pytest.raises(TypeError):
            MsgSerializer.to_bytes({"bad": {1, 2, 3}})  # a set is not msgpack-native

    def test_decode_passthrough_non_dict(self):
        """_decode is a no-op for non-dict values (the object_hook contract).

        msgpack only invokes the hook for map values, but the guard keeps the
        serializer safe if fed a scalar or list directly.
        """
        assert MsgSerializer._decode(42) == 42
        assert MsgSerializer._decode([1, 2, 3]) == [1, 2, 3]
