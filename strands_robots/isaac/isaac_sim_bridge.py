"""
Isaac Sim Subprocess Bridge — ZMQ-based communication with Isaac Sim runtime.

Solves the fundamental runtime boundary: Isaac Sim's ``SimulationApp``
requires Omniverse Kit's own Python runtime (``python.sh``), which cannot
be loaded from a standard pip Python environment.

This module provides:

1. ``IsaacSimBridgeServer`` — runs inside Isaac Sim's Python, listens on ZMQ
2. ``IsaacSimBridgeClient`` — runs in pip Python, spawns server as subprocess

The bridge is transparent: ``IsaacSimBackend`` detects when ``SimulationApp``
is unavailable and automatically creates a bridge client. All method calls
are serialized over ZMQ REQ/REP sockets using msgpack.

Architecture::

    ┌──────────────────────┐     ZMQ IPC/TCP      ┌───────────────────────┐
    │  pip Python (venv)   │ ◄──────────────────► │  Isaac Sim python.sh   │
    │                      │                      │                        │
    │  IsaacSimBackend     │  REQ: method + args  │  IsaacSimBridgeServer  │
    │   └─ BridgeClient    │ ────────────────────►│   └─ SimulationApp     │
    │                      │  REP: result dict    │   └─ GPU Physics       │
    │                      │ ◄────────────────────│   └─ RTX Rendering     │
    └──────────────────────┘                      └───────────────────────┘

Performance:
    ZMQ IPC overhead: ~0.1-0.3ms per round-trip (negligible vs 5ms physics step)
    Image transfer:   Raw bytes + shape metadata (no PNG encode/decode in loop)
    Batch tensors:    msgpack-numpy for zero-copy-like ndarray serialization

Requires:
    - pyzmq (``pip install pyzmq``)
    - msgpack (``pip install msgpack``)
    - Isaac Sim installed on filesystem (detected via ``get_isaac_sim_path()``)
"""

import json
import logging
import os
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Default port range for ZMQ TCP sockets
_DEFAULT_PORT = 19876
_CONNECT_TIMEOUT_S = 60
_REQUEST_TIMEOUT_MS = 30_000  # 30s per request


# ─────────────────────────────────────────────────────────────────────
# Protocol — shared between client and server
# ─────────────────────────────────────────────────────────────────────


def _encode_message(msg: Dict[str, Any]) -> bytes:
    """Encode a message dict to bytes using msgpack."""
    try:
        import msgpack
        import msgpack_numpy as m

        m.patch()
        return msgpack.packb(msg, default=m.encode)
    except ImportError:
        # Fallback: JSON with numpy array serialization
        return json.dumps(msg, cls=_NumpyEncoder).encode("utf-8")


def _decode_message(data: bytes) -> Dict[str, Any]:
    """Decode bytes to a message dict."""
    try:
        import msgpack
        import msgpack_numpy as m

        m.patch()
        return msgpack.unpackb(data, object_hook=m.decode, raw=False)
    except ImportError:
        return json.loads(data.decode("utf-8"))


class _NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types (fallback when msgpack unavailable)."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return {"__ndarray__": True, "data": obj.tolist(), "dtype": str(obj.dtype), "shape": list(obj.shape)}
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, bytes):
            import base64

            return {"__bytes__": True, "data": base64.b64encode(obj).decode("ascii")}
        return super().default(obj)


def _numpy_object_hook(dct):
    """Decode numpy arrays from JSON (fallback)."""
    if "__ndarray__" in dct:
        return np.array(dct["data"], dtype=dct["dtype"]).reshape(dct["shape"])
    if "__bytes__" in dct:
        import base64

        return base64.b64decode(dct["data"])
    return dct


# ─────────────────────────────────────────────────────────────────────
# Bridge Server — runs inside Isaac Sim's python.sh
# ─────────────────────────────────────────────────────────────────────


class IsaacSimBridgeServer:
    """ZMQ server that wraps Isaac Sim's SimulationApp and physics pipeline.

    This class runs inside Isaac Sim's own Python runtime (``python.sh``)
    and exposes the simulation API over a ZMQ REP socket.

    Usage (via python.sh)::

        /path/to/IsaacSim/python.sh -m strands_robots.isaac.isaac_sim_bridge \\
            --mode server --port 19876

    Or programmatically::

        server = IsaacSimBridgeServer(port=19876)
        server.run()  # Blocks, handles requests until shutdown
    """

    def __init__(
        self,
        port: int = _DEFAULT_PORT,
        bind_address: str = "tcp://*",
    ):
        self.port = port
        self.bind_address = bind_address
        self._sim_app = None
        self._backend = None
        self._running = False

    def run(self):
        """Start the server loop. Blocks until shutdown."""
        import zmq

        ctx = zmq.Context()
        socket = ctx.socket(zmq.REP)
        endpoint = f"{self.bind_address}:{self.port}"
        socket.bind(endpoint)
        logger.info(f"Isaac Sim bridge server listening on {endpoint}")

        self._running = True

        # Handle SIGTERM/SIGINT gracefully
        def _signal_handler(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self._running = False

        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

        # Send ready signal
        try:
            while self._running:
                if not socket.poll(timeout=1000):  # 1s poll
                    continue

                raw = socket.recv()
                msg = _decode_message(raw)

                method = msg.get("method", "")
                args = msg.get("args", {})

                try:
                    result = self._dispatch(method, args)
                except Exception as e:
                    import traceback

                    result = {
                        "status": "error",
                        "content": [{"text": f"Server error in {method}: {e}"}],
                        "exception": str(e),
                        "traceback": traceback.format_exc(),
                    }

                socket.send(_encode_message(result))

        except zmq.ZMQError as e:
            if self._running:
                logger.error(f"ZMQ error: {e}")
        finally:
            socket.close()
            ctx.term()
            self._cleanup()
            logger.info("Isaac Sim bridge server stopped")

    def _dispatch(self, method: str, args: Dict[str, Any]) -> Dict[str, Any]:
        """Route a method call to the appropriate handler."""
        if method == "ping":
            return {"status": "success", "content": [{"text": "pong"}]}

        if method == "init":
            return self._init_backend(args)

        if method == "shutdown":
            self._running = False
            return {"status": "success", "content": [{"text": "Shutting down"}]}

        # Forward to backend
        if self._backend is None:
            return {"status": "error", "content": [{"text": "Backend not initialized. Call init first."}]}

        # Only forward public backend methods — defense-in-depth
        if method.startswith("_"):
            return {"status": "error", "content": [{"text": f"Private method not callable: {method}"}]}

        handler = getattr(self._backend, method, None)
        if handler is None:
            return {"status": "error", "content": [{"text": f"Unknown method: {method}"}]}

        return handler(**args)

    def _init_backend(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize SimulationApp and IsaacSimBackend."""
        try:
            from isaacsim import SimulationApp

            headless = args.get("headless", True)
            self._sim_app = SimulationApp({"headless": headless})
            logger.info("SimulationApp created")

            # Now import the backend (Isaac Lab is available under python.sh)
            from strands_robots.isaac.isaac_sim_backend import IsaacSimBackend, IsaacSimConfig

            config = IsaacSimConfig(
                num_envs=args.get("num_envs", 1),
                device=args.get("device", "cuda:0"),
                physics_dt=args.get("physics_dt", 1.0 / 200.0),
                rendering_dt=args.get("rendering_dt", 1.0 / 60.0),
                headless=headless,
                enable_cameras=args.get("enable_cameras", True),
                camera_width=args.get("camera_width", 640),
                camera_height=args.get("camera_height", 480),
            )

            self._backend = IsaacSimBackend(config=config)
            return {
                "status": "success",
                "content": [{"text": f"Isaac Sim backend initialized (num_envs={config.num_envs})"}],
            }

        except Exception as e:
            import traceback

            return {
                "status": "error",
                "content": [{"text": f"Failed to initialize: {e}"}],
                "traceback": traceback.format_exc(),
            }

    def _cleanup(self):
        """Clean up Isaac Sim resources."""
        if self._backend is not None:
            try:
                self._backend.destroy()
            except Exception:
                logger.debug("Failed to destroy backend during cleanup", exc_info=True)
            self._backend = None

        if self._sim_app is not None:
            try:
                self._sim_app.close()
            except Exception:
                logger.debug("Failed to close SimulationApp during cleanup", exc_info=True)
            self._sim_app = None


# ─────────────────────────────────────────────────────────────────────
# Bridge Client — runs in pip Python, communicates with server
# ─────────────────────────────────────────────────────────────────────


class IsaacSimBridgeClient:
    """ZMQ client that communicates with an Isaac Sim bridge server.

    Can either connect to an existing server or spawn one as a subprocess
    using Isaac Sim's ``python.sh``.

    Usage::

        # Auto-spawn server
        client = IsaacSimBridgeClient(
            isaac_sim_path="/home/ubuntu/IsaacSim",
            auto_spawn=True,
        )
        client.connect()

        # Or connect to existing server
        client = IsaacSimBridgeClient(host="localhost", port=19876)
        client.connect()

        # Use like IsaacSimBackend
        result = client.call("create_world", {"ground_plane": True})
        result = client.call("add_robot", {"name": "so100", "usd_path": "/data/so100.usd"})
        result = client.call("step", {"actions": numpy_array})
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = _DEFAULT_PORT,
        isaac_sim_path: Optional[str] = None,
        auto_spawn: bool = False,
        connect_timeout: float = _CONNECT_TIMEOUT_S,
        request_timeout: int = _REQUEST_TIMEOUT_MS,
    ):
        self.host = host
        self.port = port
        self.isaac_sim_path = isaac_sim_path
        self.auto_spawn = auto_spawn
        self.connect_timeout = connect_timeout
        self.request_timeout = request_timeout

        self._socket = None
        self._context = None
        self._server_process = None
        self._connected = False

    def connect(self) -> bool:
        """Connect to the bridge server, optionally spawning it first.

        Returns:
            True if connected successfully, False otherwise.
        """
        if self.auto_spawn and self.isaac_sim_path:
            self._spawn_server()

        try:
            import zmq
        except ImportError:
            raise ImportError("pyzmq is required for the Isaac Sim bridge.\n" "Install with: pip install pyzmq")

        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, self.request_timeout)
        self._socket.setsockopt(zmq.SNDTIMEO, self.request_timeout)
        self._socket.setsockopt(zmq.LINGER, 0)

        endpoint = f"tcp://{self.host}:{self.port}"
        self._socket.connect(endpoint)
        logger.info(f"Connecting to Isaac Sim bridge at {endpoint}...")

        # Wait for server to be ready
        deadline = time.monotonic() + self.connect_timeout
        while time.monotonic() < deadline:
            try:
                result = self.call("ping")
                if result.get("status") == "success":
                    self._connected = True
                    logger.info("Connected to Isaac Sim bridge server")
                    return True
            except Exception:
                logger.debug("Connection attempt failed, retrying...", exc_info=True)
                time.sleep(1.0)

        logger.error(f"Failed to connect within {self.connect_timeout}s")
        return False

    def call(self, method: str, args: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Call a method on the remote Isaac Sim backend.

        Args:
            method: Method name (e.g., "create_world", "step", "render")
            args: Keyword arguments for the method

        Returns:
            Result dict from the server

        Raises:
            ConnectionError: If not connected
            TimeoutError: If the server doesn't respond in time
        """
        if self._socket is None:
            raise ConnectionError("Not connected. Call connect() first.")

        msg = {"method": method, "args": args or {}}

        try:
            self._socket.send(_encode_message(msg))
            raw = self._socket.recv()
            result = _decode_message(raw)

            # Re-raise server exceptions
            if result.get("status") == "error" and "exception" in result:
                logger.error(f"Server error in {method}: {result['exception']}\n" f"{result.get('traceback', '')}")

            return result

        except Exception as e:
            import zmq

            if isinstance(e, zmq.Again):
                raise TimeoutError(f"Server timeout on {method} (>{self.request_timeout}ms)") from e
            raise

    def init_backend(self, **config_kwargs) -> Dict[str, Any]:
        """Initialize the remote Isaac Sim backend.

        Args:
            **config_kwargs: IsaacSimConfig fields (num_envs, device, etc.)

        Returns:
            Result dict
        """
        return self.call("init", config_kwargs)

    def close(self):
        """Close the connection and optionally stop the server."""
        if self._socket is not None:
            try:
                # Send shutdown command
                self.call("shutdown")
            except Exception:
                logger.debug("Failed to send shutdown during cleanup", exc_info=True)

            try:
                self._socket.close()
            except Exception:
                logger.debug("Failed to close ZMQ socket during cleanup", exc_info=True)
            self._socket = None

        if self._context is not None:
            try:
                self._context.term()
            except Exception:
                logger.debug("Failed to terminate ZMQ context during cleanup", exc_info=True)
            self._context = None

        if self._server_process is not None:
            try:
                self._server_process.terminate()
                self._server_process.wait(timeout=10)
            except Exception:
                logger.debug("Failed to terminate server process, sending SIGKILL", exc_info=True)
                try:
                    self._server_process.kill()
                except Exception:
                    logger.debug("Failed to kill server process during cleanup", exc_info=True)
            self._server_process = None

        self._connected = False

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.debug("Failed to close client in __del__", exc_info=True)

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *args):
        self.close()

    @property
    def is_connected(self) -> bool:
        """Return True if connected to the server."""
        return self._connected

    def _spawn_server(self):
        """Spawn the bridge server as a subprocess using Isaac Sim's python.sh."""
        python_sh = os.path.join(self.isaac_sim_path, "python.sh")
        if not os.path.isfile(python_sh):
            raise FileNotFoundError(
                f"Isaac Sim python.sh not found at {python_sh}.\n"
                f"Verify Isaac Sim is installed at {self.isaac_sim_path}"
            )

        # Launch the server module
        cmd = [
            python_sh,
            "-m",
            "strands_robots.isaac.isaac_sim_bridge",
            "--mode",
            "server",
            "--port",
            str(self.port),
        ]

        logger.info(f"Spawning Isaac Sim bridge server: {' '.join(cmd)}")

        self._server_process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "ISAAC_SIM_BRIDGE_SERVER": "1"},
        )

        # Give the server time to start SimulationApp (~10-30s on first run)
        logger.info("Waiting for Isaac Sim to initialize (may take 30-60s on first run)...")


# ─────────────────────────────────────────────────────────────────────
# CLI entry point — ``python.sh -m strands_robots.isaac.isaac_sim_bridge``
# ─────────────────────────────────────────────────────────────────────


def main():
    """CLI entry point for the Isaac Sim bridge server."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Isaac Sim Subprocess Bridge",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server (run via Isaac Sim's Python)
  /path/to/IsaacSim/python.sh -m strands_robots.isaac.isaac_sim_bridge --mode server --port 19876

  # Test connection from pip Python
  python -m strands_robots.isaac.isaac_sim_bridge --mode client --port 19876
""",
    )
    parser.add_argument("--mode", choices=["server", "client"], required=True)
    parser.add_argument("--port", type=int, default=_DEFAULT_PORT)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--bind", default="tcp://*", help="Bind address for server")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.mode == "server":
        server = IsaacSimBridgeServer(port=args.port, bind_address=args.bind)
        server.run()

    elif args.mode == "client":
        # Simple connectivity test
        client = IsaacSimBridgeClient(host=args.host, port=args.port)
        if client.connect():
            print("✅ Connected to Isaac Sim bridge server")
            result = client.call("ping")
            print(f"Ping result: {result}")
            client.close()
        else:
            print("❌ Failed to connect")
            sys.exit(1)


if __name__ == "__main__":
    main()
