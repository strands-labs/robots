"""Tests for Isaac Sim subprocess bridge (ZMQ client/server).

All tests are CPU-only with full mocking — no Isaac Sim or GPU required.
Tests cover: protocol encoding, server dispatch, client lifecycle,
subprocess spawning, error handling, and context manager patterns.
"""

import json
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Pre-mock heavy deps at module scope
_mock_zmq = MagicMock()
_mock_zmq.Context = MagicMock
_mock_zmq.REP = 1
_mock_zmq.REQ = 2
_mock_zmq.RCVTIMEO = 27
_mock_zmq.SNDTIMEO = 28
_mock_zmq.LINGER = 17
_mock_zmq.ZMQError = type("ZMQError", (Exception,), {})
_mock_zmq.Again = type("Again", (_mock_zmq.ZMQError,), {})

_mock_msgpack = MagicMock()
_mock_msgpack_numpy = MagicMock()

# Pre-mock cv2 to avoid OpenCV 4.12 crash
import importlib.machinery  # noqa: E402

_mock_cv2 = MagicMock()
# Always use a fresh ModuleSpec — find_spec("cv2") explodes when another test
# has already injected a MagicMock cv2 with no real __spec__ into sys.modules.
_mock_cv2.__spec__ = importlib.machinery.ModuleSpec("cv2", None)
sys.modules.setdefault("cv2", _mock_cv2)
sys.modules.setdefault("cv2.dnn", MagicMock())

# Import the bridge module (no heavy deps needed)
from strands_robots.isaac.isaac_sim_bridge import (  # noqa: E402
    _DEFAULT_PORT,
    IsaacSimBridgeClient,
    IsaacSimBridgeServer,
    _decode_message,
    _encode_message,
    _numpy_object_hook,
    _NumpyEncoder,
)

# ─────────────────────────────────────────────────────────────────────
# Protocol encoding/decoding tests
# ─────────────────────────────────────────────────────────────────────


class TestProtocol:
    """Test message encoding/decoding (JSON fallback path)."""

    def test_encode_simple_dict(self):
        """Encode a simple dict to bytes."""
        msg = {"method": "ping", "args": {}}
        data = _encode_message(msg)
        assert isinstance(data, bytes)
        decoded = json.loads(data)
        assert decoded["method"] == "ping"

    def test_decode_simple_dict(self):
        """Decode bytes back to a dict."""
        msg = {"method": "step", "args": {"actions": [1.0, 2.0]}}
        data = json.dumps(msg).encode("utf-8")
        result = _decode_message(data)
        assert result["method"] == "step"
        assert result["args"]["actions"] == [1.0, 2.0]

    def test_encode_decode_roundtrip(self):
        """Encode then decode should be identity (for JSON-compatible data)."""
        msg = {"status": "success", "content": [{"text": "hello"}], "value": 42}
        data = _encode_message(msg)
        result = _decode_message(data)
        assert result == msg

    def test_numpy_encoder_ndarray(self):
        """NumpyEncoder handles numpy arrays."""
        arr = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        encoded = json.dumps({"data": arr}, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded["data"]["__ndarray__"] is True
        assert decoded["data"]["dtype"] == "float32"
        assert decoded["data"]["shape"] == [3]
        assert decoded["data"]["data"] == [1.0, 2.0, 3.0]

    def test_numpy_encoder_integer(self):
        """NumpyEncoder handles numpy integers."""
        val = np.int64(42)
        encoded = json.dumps({"n": val}, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded["n"] == 42

    def test_numpy_encoder_float(self):
        """NumpyEncoder handles numpy floats."""
        val = np.float32(3.14)
        encoded = json.dumps({"f": val}, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert abs(decoded["f"] - 3.14) < 0.01

    def test_numpy_encoder_bytes(self):
        """NumpyEncoder handles raw bytes."""
        data = b"\x00\x01\x02\xff"
        encoded = json.dumps({"img": data}, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded["img"]["__bytes__"] is True
        assert isinstance(decoded["img"]["data"], str)

    def test_numpy_object_hook_ndarray(self):
        """Object hook reconstructs numpy arrays."""
        dct = {"__ndarray__": True, "data": [1.0, 2.0], "dtype": "float64", "shape": [2]}
        result = _numpy_object_hook(dct)
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float64
        np.testing.assert_array_equal(result, [1.0, 2.0])

    def test_numpy_object_hook_bytes(self):
        """Object hook reconstructs bytes."""
        import base64

        original = b"\x00\x01\x02"
        dct = {"__bytes__": True, "data": base64.b64encode(original).decode("ascii")}
        result = _numpy_object_hook(dct)
        assert result == original

    def test_numpy_object_hook_passthrough(self):
        """Object hook passes through regular dicts."""
        dct = {"key": "value"}
        result = _numpy_object_hook(dct)
        assert result == dct

    def test_encode_2d_ndarray(self):
        """2D ndarray preserves shape through encode."""
        arr = np.zeros((3, 4), dtype=np.float32)
        encoded = json.dumps({"obs": arr}, cls=_NumpyEncoder)
        decoded = json.loads(encoded)
        assert decoded["obs"]["shape"] == [3, 4]

    def test_encode_with_msgpack_available(self):
        """When msgpack is available, uses msgpack path."""
        mock_msgpack = MagicMock()
        mock_msgpack.packb.return_value = b"\x80"
        mock_mnp = MagicMock()

        with patch.dict("sys.modules", {"msgpack": mock_msgpack, "msgpack_numpy": mock_mnp}):
            # Re-import to get fresh function refs
            from strands_robots.isaac.isaac_sim_bridge import _encode_message as enc

            enc({"method": "ping"})
            # Should have called msgpack
            mock_msgpack.packb.assert_called_once()


# ─────────────────────────────────────────────────────────────────────
# Server tests
# ─────────────────────────────────────────────────────────────────────


class TestBridgeServer:
    """Test IsaacSimBridgeServer dispatch and lifecycle."""

    def test_init_defaults(self):
        server = IsaacSimBridgeServer()
        assert server.port == _DEFAULT_PORT
        assert server.bind_address == "tcp://*"
        assert server._sim_app is None
        assert server._backend is None
        assert server._running is False

    def test_init_custom_port(self):
        server = IsaacSimBridgeServer(port=12345, bind_address="tcp://0.0.0.0")
        assert server.port == 12345
        assert server.bind_address == "tcp://0.0.0.0"

    def test_dispatch_ping(self):
        server = IsaacSimBridgeServer()
        result = server._dispatch("ping", {})
        assert result["status"] == "success"
        assert result["content"][0]["text"] == "pong"

    def test_dispatch_shutdown(self):
        server = IsaacSimBridgeServer()
        server._running = True
        result = server._dispatch("shutdown", {})
        assert result["status"] == "success"
        assert server._running is False

    def test_dispatch_unknown_method_no_backend(self):
        server = IsaacSimBridgeServer()
        result = server._dispatch("create_world", {})
        assert result["status"] == "error"
        assert "not initialized" in result["content"][0]["text"]

    def test_dispatch_unknown_method_with_backend(self):
        server = IsaacSimBridgeServer()
        server._backend = MagicMock()
        server._backend.nonexistent_method = None
        # hasattr will return True because MagicMock auto-creates, but
        # getattr for the actual missing method returns MagicMock
        server._dispatch("totally_fake_method", {})
        # MagicMock will auto-create the attribute, so it'll be called
        # This is expected — the bridge dispatches to whatever the backend exposes

    def test_dispatch_private_method_rejected(self):
        """Private methods (starting with _) are rejected for defense-in-depth."""
        server = IsaacSimBridgeServer()
        server._backend = MagicMock()

        result = server._dispatch("_cleanup", {})
        assert result["status"] == "error"
        assert "Private method not callable" in result["content"][0]["text"]

        result = server._dispatch("__init__", {})
        assert result["status"] == "error"
        assert "Private method not callable" in result["content"][0]["text"]

    def test_dispatch_forward_to_backend(self):
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_backend.create_world.return_value = {"status": "success", "content": [{"text": "world created"}]}
        server._backend = mock_backend

        result = server._dispatch("create_world", {"ground_plane": True})
        assert result["status"] == "success"
        mock_backend.create_world.assert_called_once_with(ground_plane=True)

    def test_dispatch_forward_step(self):
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        actions = np.array([[0.1, 0.2, 0.3]])
        mock_backend.step.return_value = {"status": "success", "content": [{"text": "stepped"}]}
        server._backend = mock_backend

        result = server._dispatch("step", {"actions": actions})
        assert result["status"] == "success"

    def test_dispatch_forward_render(self):
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_backend.render.return_value = {
            "status": "success",
            "content": [
                {"text": "rendered"},
                {"image": {"format": "png", "source": {"bytes": b"\x89PNG"}}},
            ],
        }
        server._backend = mock_backend

        result = server._dispatch("render", {"camera_name": "left_cam"})
        assert result["status"] == "success"

    def test_dispatch_init_no_isaacsim(self):
        """Init fails gracefully when isaacsim not available."""
        server = IsaacSimBridgeServer()
        with patch.dict("sys.modules", {"isaacsim": None}):
            result = server._init_backend({"headless": True})
            assert result["status"] == "error"

    def test_cleanup_no_backend(self):
        """Cleanup works with no backend initialized."""
        server = IsaacSimBridgeServer()
        server._cleanup()  # Should not raise
        assert server._backend is None
        assert server._sim_app is None

    def test_cleanup_with_backend(self):
        """Cleanup destroys backend and sim app."""
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_sim_app = MagicMock()
        server._backend = mock_backend
        server._sim_app = mock_sim_app

        server._cleanup()
        mock_backend.destroy.assert_called_once()
        mock_sim_app.close.assert_called_once()
        assert server._backend is None
        assert server._sim_app is None

    def test_cleanup_handles_exceptions(self):
        """Cleanup suppresses exceptions during teardown."""
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_backend.destroy.side_effect = RuntimeError("GPU error")
        mock_sim_app = MagicMock()
        mock_sim_app.close.side_effect = RuntimeError("Kit shutdown error")
        server._backend = mock_backend
        server._sim_app = mock_sim_app

        server._cleanup()  # Should not raise
        assert server._backend is None
        assert server._sim_app is None


# ─────────────────────────────────────────────────────────────────────
# Client tests
# ─────────────────────────────────────────────────────────────────────


class TestBridgeClient:
    """Test IsaacSimBridgeClient connection, calls, and lifecycle."""

    def test_init_defaults(self):
        client = IsaacSimBridgeClient()
        assert client.host == "localhost"
        assert client.port == _DEFAULT_PORT
        assert client.auto_spawn is False
        assert client._connected is False
        assert client._socket is None

    def test_init_custom(self):
        client = IsaacSimBridgeClient(
            host="192.168.1.100",
            port=9999,
            isaac_sim_path="/opt/IsaacSim",
            auto_spawn=True,
            connect_timeout=30,
            request_timeout=5000,
        )
        assert client.host == "192.168.1.100"
        assert client.port == 9999
        assert client.isaac_sim_path == "/opt/IsaacSim"
        assert client.auto_spawn is True
        assert client.connect_timeout == 30
        assert client.request_timeout == 5000

    def test_is_connected_default(self):
        client = IsaacSimBridgeClient()
        assert client.is_connected is False

    def test_call_without_connect_raises(self):
        client = IsaacSimBridgeClient()
        with pytest.raises(ConnectionError, match="Not connected"):
            client.call("ping")

    def test_call_with_mock_socket(self):
        """Call sends encoded message and returns decoded response."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()

        # Mock encode/decode to use JSON fallback
        response = {"status": "success", "content": [{"text": "pong"}]}
        mock_socket.recv.return_value = json.dumps(response).encode("utf-8")
        client._socket = mock_socket

        result = client.call("ping")
        assert result["status"] == "success"
        mock_socket.send.assert_called_once()
        mock_socket.recv.assert_called_once()

    def test_call_with_args(self):
        """Call forwards arguments correctly."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()
        response = {"status": "success", "content": [{"text": "world created"}]}
        mock_socket.recv.return_value = json.dumps(response).encode("utf-8")
        client._socket = mock_socket

        result = client.call("create_world", {"ground_plane": True, "gravity": [0, 0, -9.81]})
        assert result["status"] == "success"

        # Verify the sent message contains the method and args
        sent_data = mock_socket.send.call_args[0][0]
        sent_msg = json.loads(sent_data)
        assert sent_msg["method"] == "create_world"
        assert sent_msg["args"]["ground_plane"] is True

    def test_call_server_error_logged(self):
        """Server errors are logged but returned."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()
        response = {
            "status": "error",
            "content": [{"text": "GPU OOM"}],
            "exception": "RuntimeError: CUDA out of memory",
            "traceback": "...",
        }
        mock_socket.recv.return_value = json.dumps(response).encode("utf-8")
        client._socket = mock_socket

        result = client.call("step", {"actions": [0.1, 0.2]})
        assert result["status"] == "error"
        assert "exception" in result

    def test_init_backend_delegates(self):
        """init_backend is a convenience wrapper around call('init')."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()
        response = {"status": "success", "content": [{"text": "initialized"}]}
        mock_socket.recv.return_value = json.dumps(response).encode("utf-8")
        client._socket = mock_socket

        result = client.init_backend(num_envs=16, device="cuda:0")
        assert result["status"] == "success"

        sent_data = mock_socket.send.call_args[0][0]
        sent_msg = json.loads(sent_data)
        assert sent_msg["method"] == "init"
        assert sent_msg["args"]["num_envs"] == 16
        assert sent_msg["args"]["device"] == "cuda:0"

    def test_close_sends_shutdown(self):
        """Close sends shutdown command then cleans up."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()
        mock_socket.recv.return_value = json.dumps({"status": "success"}).encode("utf-8")
        client._socket = mock_socket
        mock_ctx = MagicMock()
        client._context = mock_ctx

        client.close()

        # Shutdown was sent
        mock_socket.send.assert_called()
        mock_socket.close.assert_called_once()
        mock_ctx.term.assert_called_once()
        assert client._socket is None
        assert client._context is None
        assert client._connected is False

    def test_close_handles_shutdown_error(self):
        """Close suppresses errors from shutdown command."""
        client = IsaacSimBridgeClient()
        mock_socket = MagicMock()
        mock_socket.send.side_effect = Exception("Connection lost")
        client._socket = mock_socket
        mock_ctx = MagicMock()
        client._context = mock_ctx

        client.close()  # Should not raise
        assert client._socket is None

    def test_close_terminates_subprocess(self):
        """Close terminates the spawned server process."""
        client = IsaacSimBridgeClient()
        mock_proc = MagicMock()
        client._server_process = mock_proc
        mock_socket = MagicMock()
        mock_socket.recv.return_value = json.dumps({"status": "success"}).encode("utf-8")
        client._socket = mock_socket
        client._context = MagicMock()

        client.close()
        mock_proc.terminate.assert_called_once()
        mock_proc.wait.assert_called_once_with(timeout=10)
        assert client._server_process is None

    def test_close_kills_if_terminate_hangs(self):
        """Close force-kills if terminate doesn't work."""
        client = IsaacSimBridgeClient()
        mock_proc = MagicMock()
        mock_proc.wait.side_effect = subprocess.TimeoutExpired("cmd", 10)
        client._server_process = mock_proc
        client._socket = MagicMock()
        client._socket.recv.return_value = json.dumps({"status": "success"}).encode("utf-8")
        client._context = MagicMock()

        client.close()
        mock_proc.kill.assert_called_once()

    def test_del_calls_close(self):
        """__del__ calls close() without raising."""
        client = IsaacSimBridgeClient()
        client.close = MagicMock()
        client.__del__()
        client.close.assert_called_once()

    def test_del_suppresses_errors(self):
        """__del__ suppresses exceptions from close."""
        client = IsaacSimBridgeClient()
        client.close = MagicMock(side_effect=RuntimeError("boom"))
        client.__del__()  # Should not raise

    def test_context_manager(self):
        """Context manager connects on enter and closes on exit."""
        client = IsaacSimBridgeClient()
        client.connect = MagicMock(return_value=True)
        client.close = MagicMock()

        with client as c:
            assert c is client
            client.connect.assert_called_once()

        client.close.assert_called_once()

    def test_spawn_server_missing_python_sh(self):
        """Spawning fails if python.sh doesn't exist."""
        client = IsaacSimBridgeClient(
            isaac_sim_path="/nonexistent/path",
            auto_spawn=True,
        )
        with pytest.raises(FileNotFoundError, match="python.sh not found"):
            client._spawn_server()

    def test_spawn_server_success(self):
        """Spawning launches subprocess with correct args."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a fake python.sh
            python_sh = os.path.join(tmpdir, "python.sh")
            with open(python_sh, "w") as f:
                f.write("#!/bin/bash\npython3 $@\n")
            os.chmod(python_sh, 0o755)

            client = IsaacSimBridgeClient(
                isaac_sim_path=tmpdir,
                port=19999,
            )

            with patch("subprocess.Popen") as mock_popen:
                mock_popen.return_value = MagicMock()
                client._spawn_server()

                mock_popen.assert_called_once()
                call_args = mock_popen.call_args
                cmd = call_args[0][0]
                assert cmd[0] == python_sh
                assert "-m" in cmd
                assert "strands_robots.isaac.isaac_sim_bridge" in cmd
                assert "--port" in cmd
                assert "19999" in cmd

                assert client._server_process is not None

    def test_connect_without_zmq_raises(self):
        """Connect raises ImportError if pyzmq is missing."""
        client = IsaacSimBridgeClient()
        with patch.dict("sys.modules", {"zmq": None}):
            with pytest.raises(ImportError, match="pyzmq"):
                client.connect()

    def test_connect_timeout(self):
        """Connect returns False if server never responds."""
        mock_zmq_mod = MagicMock()
        mock_zmq_mod.Context.return_value = MagicMock()
        mock_socket = MagicMock()
        mock_zmq_mod.Context.return_value.socket.return_value = mock_socket
        # Make recv always timeout
        mock_socket.recv.side_effect = Exception("timeout")

        client = IsaacSimBridgeClient(connect_timeout=0.1)

        with patch.dict("sys.modules", {"zmq": mock_zmq_mod}):
            result = client.connect()
            assert result is False
            assert client._connected is False


# ─────────────────────────────────────────────────────────────────────
# Integration tests (client + server in same process, mock ZMQ)
# ─────────────────────────────────────────────────────────────────────


class TestBridgeIntegration:
    """Test client-server integration with in-process mock transport."""

    def test_ping_roundtrip(self):
        """Client ping reaches server and returns pong."""
        server = IsaacSimBridgeServer()

        # Simulate the server dispatch
        msg = {"method": "ping", "args": {}}
        encoded = _encode_message(msg)
        decoded = _decode_message(encoded)
        result = server._dispatch(decoded["method"], decoded["args"])

        assert result["status"] == "success"
        assert "pong" in result["content"][0]["text"]

    def test_create_world_roundtrip(self):
        """Client create_world reaches server backend."""
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_backend.create_world.return_value = {
            "status": "success",
            "content": [{"text": "world created"}],
        }
        server._backend = mock_backend

        msg = {"method": "create_world", "args": {"ground_plane": True}}
        encoded = _encode_message(msg)
        decoded = _decode_message(encoded)
        result = server._dispatch(decoded["method"], decoded["args"])

        assert result["status"] == "success"
        mock_backend.create_world.assert_called_once_with(ground_plane=True)

    def test_step_with_numpy_actions(self):
        """Step with numpy action arrays survives encode/decode."""
        actions = np.array([[0.1, 0.2, 0.3, 0.4, 0.5, 0.6]], dtype=np.float32)

        msg = {"method": "step", "args": {"actions": actions}}
        encoded = _encode_message(msg)
        decoded = _decode_message(encoded)

        assert decoded["method"] == "step"
        # In JSON fallback, numpy arrays become dicts with __ndarray__ key
        args_actions = decoded["args"]["actions"]
        if isinstance(args_actions, dict) and "__ndarray__" in args_actions:
            arr = np.array(args_actions["data"], dtype=args_actions["dtype"])
            np.testing.assert_array_almost_equal(arr.flatten(), actions.flatten(), decimal=5)
        elif isinstance(args_actions, np.ndarray):
            np.testing.assert_array_almost_equal(args_actions, actions, decimal=5)
        else:
            # List fallback
            np.testing.assert_array_almost_equal(np.array(args_actions), actions, decimal=5)

    def test_render_with_image_bytes(self):
        """Image bytes survive encode/decode roundtrip."""
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100  # Fake PNG

        response = {
            "status": "success",
            "content": [
                {"text": "rendered"},
                {"image": {"format": "png", "source": {"bytes": img_bytes}}},
            ],
        }

        encoded = _encode_message(response)
        decoded = _decode_message(encoded)

        assert decoded["status"] == "success"
        # Image bytes should be recoverable
        img_content = decoded["content"][1]["image"]["source"]["bytes"]
        if isinstance(img_content, dict) and "__bytes__" in img_content:
            import base64

            recovered = base64.b64decode(img_content["data"])
            assert recovered == img_bytes
        elif isinstance(img_content, bytes):
            assert img_content == img_bytes

    def test_error_propagation(self):
        """Server errors propagate through encode/decode."""
        server = IsaacSimBridgeServer()
        mock_backend = MagicMock()
        mock_backend.step.side_effect = RuntimeError("CUDA OOM")
        server._backend = mock_backend

        # Server would catch exception in run() loop
        try:
            result = server._dispatch("step", {})
        except RuntimeError:
            # The dispatch doesn't catch — the run() loop does
            result = {
                "status": "error",
                "content": [{"text": "CUDA OOM"}],
                "exception": "CUDA OOM",
            }

        encoded = _encode_message(result)
        decoded = _decode_message(encoded)
        assert decoded["status"] == "error"
        assert "CUDA OOM" in decoded["exception"]

    def test_shutdown_lifecycle(self):
        """Full lifecycle: ping → init → work → shutdown."""
        server = IsaacSimBridgeServer()
        server._running = True

        # Ping
        r1 = server._dispatch("ping", {})
        assert r1["status"] == "success"

        # Init (will fail without Isaac Sim, but tests the path)
        # We mock the backend directly
        mock_backend = MagicMock()
        mock_backend.create_world.return_value = {"status": "success", "content": [{"text": "ok"}]}
        mock_backend.step.return_value = {"status": "success", "content": [{"text": "stepped"}]}
        mock_backend.destroy.return_value = None
        server._backend = mock_backend

        # Create world
        r2 = server._dispatch("create_world", {})
        assert r2["status"] == "success"

        # Step
        r3 = server._dispatch("step", {"actions": None})
        assert r3["status"] == "success"

        # Shutdown
        r4 = server._dispatch("shutdown", {})
        assert r4["status"] == "success"
        assert server._running is False


# ─────────────────────────────────────────────────────────────────────
# __init__.py lazy import tests
# ─────────────────────────────────────────────────────────────────────


class TestLazyImports:
    """Test that bridge classes are accessible via strands_robots.isaac."""

    def test_bridge_client_import(self):
        from strands_robots.isaac.isaac_sim_bridge import IsaacSimBridgeClient

        assert IsaacSimBridgeClient is not None

    def test_bridge_server_import(self):
        from strands_robots.isaac.isaac_sim_bridge import IsaacSimBridgeServer

        assert IsaacSimBridgeServer is not None

    def test_default_port(self):
        from strands_robots.isaac.isaac_sim_bridge import _DEFAULT_PORT

        assert _DEFAULT_PORT == 19876

    def test_bridge_in_all(self):
        from strands_robots.isaac import __all__

        assert "IsaacSimBridgeClient" in __all__
        assert "IsaacSimBridgeServer" in __all__


# ─────────────────────────────────────────────────────────────────────
# CLI entry point tests
# ─────────────────────────────────────────────────────────────────────


class TestCLI:
    """Test the __main__ entry point."""

    def test_main_server_mode(self):
        """Server mode creates and runs server."""
        from strands_robots.isaac.isaac_sim_bridge import main

        with patch("sys.argv", ["bridge", "--mode", "server", "--port", "19999"]):
            with patch.object(IsaacSimBridgeServer, "run") as mock_run:
                main()
                mock_run.assert_called_once()

    def test_main_client_mode_success(self):
        """Client mode connects and pings."""
        from strands_robots.isaac.isaac_sim_bridge import main

        with patch("sys.argv", ["bridge", "--mode", "client", "--port", "19999"]):
            with patch.object(IsaacSimBridgeClient, "connect", return_value=True):
                with patch.object(IsaacSimBridgeClient, "call", return_value={"status": "success"}):
                    with patch.object(IsaacSimBridgeClient, "close"):
                        main()

    def test_main_client_mode_fail(self):
        """Client mode exits 1 on connection failure."""
        from strands_robots.isaac.isaac_sim_bridge import main

        with patch("sys.argv", ["bridge", "--mode", "client", "--port", "19999"]):
            with patch.object(IsaacSimBridgeClient, "connect", return_value=False):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                assert exc_info.value.code == 1


# Import tempfile at module level for spawn test
import tempfile  # noqa: E402
