#!/usr/bin/env python3
"""
Zenoh Mesh — Native peer-to-peer layer for every Robot and Simulation.

Every Robot() is a peer by default. No join_mesh(), no opt-in.
Constructed → on mesh. Destroyed → off mesh. That simple.

Namespace design:
    strands/{peer_id}/presence   — who I am, what I can do (2Hz)
    strands/{peer_id}/state      — joint positions, sim time (10Hz)
    strands/{peer_id}/cmd        — receive commands
    strands/{peer_id}/response/* — send responses
    strands/broadcast            — to everyone

Cross-compatible with:
    {prefix}/**     — Reachy Mini (Pollen Robotics, same eclipse-zenoh)

Disable with: Robot("so100", mesh=False) or STRANDS_MESH=false

Security model:
    The mesh is designed for **trusted local networks** (localhost by default).
    Commands from peers are dispatched to robot methods, including policy execution.

    When ZENOH_CONNECT is set to a non-localhost address, all peers on that
    network can send commands. Set STRANDS_MESH_SECRET to require a shared
    token for command authentication. Commands without a valid token are rejected.

    Environment variables:
        STRANDS_MESH_SECRET  — shared secret token for command authentication.
                               If set, all command messages must include a matching
                               HMAC token. Presence/state messages are not
                               authenticated (they are read-only telemetry).
        STRANDS_MESH         — set to "false" to disable mesh entirely.
        ZENOH_CONNECT        — remote Zenoh endpoints (comma-separated).
                               A warning is logged when connecting to non-localhost
                               addresses without STRANDS_MESH_SECRET set.
"""

import hashlib
import hmac
import json
import logging
import os
import re
import socket
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Shared session — ONE Zenoh session per process, refcounted
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_SESSION = None
_SESSION_LOCK = threading.Lock()
_SESSION_REFS = 0

# All peers we know about (shared across all mesh-enabled instances)
_PEERS: Dict[str, "PeerInfo"] = {}
_PEERS_VERSION = 0
_PEERS_LOCK = threading.Lock()

# All local robots on mesh in this process
_LOCAL_ROBOTS: Dict[str, "Mesh"] = {}

HEARTBEAT_HZ = 2.0
STATE_HZ = 10.0
PEER_TIMEOUT = 10.0

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mesh authentication — shared secret for command validation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

_MESH_SECRET: Optional[str] = os.getenv("STRANDS_MESH_SECRET")

# Allowlisted actions that are safe without authentication (read-only / emergency)
_SAFE_ACTIONS = frozenset({"status", "features", "state", "stop"})

# Allowlist of valid policy providers (prevents arbitrary module loading)
_ALLOWED_POLICY_PROVIDERS = frozenset(
    {
        "mock",
        "groot",
        "lerobot_async",
        "lerobot_local",
        "dreamgen",
        "act",
        "diffusion",
        "smolvla",
        "pi0",
        "cosmos_predict",
        "openvla",
        "internvla",
        "cogact",
        "robobrain",
        "magma",
        "alpamayo",
        "unifolm",
        "omnivla",
        "rdt",
        "dreamzero",
    }
)


def _compute_mesh_token(secret: str, turn_id: str) -> str:
    """Compute HMAC-SHA256 token for command authentication.

    The token is computed over the turn_id using the shared secret,
    binding authentication to each specific command message.
    """
    return hmac.new(
        secret.encode("utf-8"),
        turn_id.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _verify_mesh_token(data: dict) -> bool:
    """Verify that a command message has a valid mesh_secret token.

    Returns True if:
    - No STRANDS_MESH_SECRET is configured (open mode), OR
    - The message contains a valid HMAC token matching the shared secret

    Returns False if:
    - STRANDS_MESH_SECRET is configured but the message has no/invalid token
    """
    if _MESH_SECRET is None:
        return True

    token = data.get("mesh_token", "")
    turn_id = data.get("turn_id", "")
    if not token or not turn_id:
        return False

    expected = _compute_mesh_token(_MESH_SECRET, turn_id)
    return hmac.compare_digest(token, expected)


def _is_localhost_endpoint(endpoint: str) -> bool:
    """Check if a Zenoh endpoint string points to localhost."""
    localhost_patterns = ("127.0.0.1", "localhost", "::1", "[::1]")
    return any(pat in endpoint for pat in localhost_patterns)


def _validate_policy_host(host: str) -> str:
    """Validate and sanitize the policy_host parameter.

    Only allows hostnames/IPs that match safe patterns.
    Prevents command injection via crafted hostnames.
    """
    if not host:
        return "localhost"
    # Allow only alphanumeric, dots, hyphens, colons (for IPv6), and brackets
    if not re.match(r"^[a-zA-Z0-9.\-:\[\]]+$", host):
        logger.warning(f"Mesh: rejected invalid policy_host: {host!r}")
        return "localhost"
    return host


def _validate_model_path(path: str) -> Optional[str]:
    """Validate model_path to prevent path traversal or command injection.

    Allows:
    - HuggingFace model IDs (org/model)
    - Absolute paths without traversal
    - Relative paths without traversal

    Rejects:
    - Paths with .. components
    - Paths with shell metacharacters
    """
    if not path:
        return None
    # Reject path traversal
    if ".." in path:
        logger.warning(f"Mesh: rejected model_path with traversal: {path!r}")
        return None
    # Reject shell metacharacters
    if any(c in path for c in (";", "|", "&", "`", "$", "(", ")", "{", "}", "<", ">", "\n", "\r")):
        logger.warning(f"Mesh: rejected model_path with shell metacharacters: {path!r}")
        return None
    return path


@dataclass
class PeerInfo:
    """A discovered peer on the Zenoh mesh."""

    peer_id: str
    peer_type: str  # "robot", "sim", "agent"
    hostname: str = ""
    last_seen: float = 0.0
    caps: Dict[str, Any] = field(default_factory=dict)

    @property
    def age(self) -> float:
        return time.time() - self.last_seen

    def to_dict(self) -> dict:
        return {
            "peer_id": self.peer_id,
            "type": self.peer_type,
            "hostname": self.hostname,
            "age": round(self.age, 1),
            **self.caps,
        }


def _get_session():
    """Acquire the shared Zenoh session (lazy, refcounted)."""
    global _SESSION, _SESSION_REFS

    with _SESSION_LOCK:
        if _SESSION is not None:
            _SESSION_REFS += 1
            return _SESSION

        try:
            import importlib

            zenoh = importlib.import_module("zenoh")
        except ImportError:
            logger.debug("eclipse-zenoh not installed — mesh disabled")
            return None

        try:
            config = zenoh.Config()
        except Exception:
            config = zenoh.Config()

        # Remote endpoints from env
        connect = os.getenv("ZENOH_CONNECT")
        listen = os.getenv("ZENOH_LISTEN")
        if connect:
            endpoints = [e.strip() for e in connect.split(",")]
            # Security warning for non-localhost endpoints without authentication
            non_local = [e for e in endpoints if not _is_localhost_endpoint(e)]
            if non_local and _MESH_SECRET is None:
                logger.warning(
                    "⚠️  SECURITY: ZENOH_CONNECT includes non-localhost endpoints %s "
                    "but STRANDS_MESH_SECRET is not set. Any peer on the network can "
                    "send commands to this robot. Set STRANDS_MESH_SECRET to enable "
                    "command authentication.",
                    non_local,
                )
            try:
                config.insert_json5("connect/endpoints", json.dumps(endpoints))
            except Exception as e:
                logger.debug(f"Zenoh connect config error: {e}")
        if listen:
            try:
                config.insert_json5("listen/endpoints", json.dumps([e.strip() for e in listen.split(",")]))
            except Exception as e:
                logger.debug(f"Zenoh listen config error: {e}")

        # Auto-mesh on localhost: first process listens, others connect
        # Port 7447 is the default Zenoh router port
        MESH_PORT = int(os.getenv("STRANDS_MESH_PORT", "7447"))
        MESH_EP = f"tcp/127.0.0.1:{MESH_PORT}"

        if not connect and not listen:
            # Try listen+connect first (works if we're first)
            try:
                cfg_try = zenoh.Config()
                cfg_try.insert_json5("listen/endpoints", json.dumps([MESH_EP]))
                cfg_try.insert_json5("connect/endpoints", json.dumps([MESH_EP]))
                _SESSION = zenoh.open(cfg_try)
                _SESSION_REFS = 1
                logger.info(f"Zenoh mesh session opened (listener on {MESH_EP})")
                return _SESSION
            except Exception:
                pass  # Port taken — someone else is listening

            # Just connect as CLIENT (another process is the router)
            try:
                config.insert_json5("mode", '"client"')
                config.insert_json5("connect/endpoints", json.dumps([MESH_EP]))
            except Exception as e:
                logger.debug(f"Zenoh client config error: {e}")

        _SESSION = zenoh.open(config)
        _SESSION_REFS = 1
        logger.info("Zenoh mesh session opened")
        return _SESSION


def _release_session():
    """Release one reference to the shared session."""
    global _SESSION, _SESSION_REFS

    with _SESSION_LOCK:
        _SESSION_REFS -= 1
        if _SESSION_REFS <= 0 and _SESSION is not None:
            try:
                _SESSION.close()
            except Exception:
                pass
            _SESSION = None
            _SESSION_REFS = 0
            logger.info("Zenoh mesh session closed")


def _put(key: str, data: dict):
    """Publish JSON to Zenoh. No-op if session is None."""
    if _SESSION:
        try:
            _SESSION.put(key, json.dumps(data).encode())
        except Exception as e:
            logger.debug(f"Zenoh put error: {e}")


def _update_peer(peer_id: str, peer_type: str, hostname: str, caps: dict):
    """Upsert a peer. Returns True if it's new."""
    global _PEERS_VERSION
    with _PEERS_LOCK:
        is_new = peer_id not in _PEERS
        _PEERS[peer_id] = PeerInfo(
            peer_id=peer_id,
            peer_type=peer_type,
            hostname=hostname,
            last_seen=time.time(),
            caps=caps,
        )
        if is_new:
            _PEERS_VERSION += 1
        return is_new


def _prune_peers():
    """Remove stale peers."""
    global _PEERS_VERSION
    now = time.time()
    with _PEERS_LOCK:
        stale = [pid for pid, p in _PEERS.items() if now - p.last_seen > PEER_TIMEOUT]
        for pid in stale:
            del _PEERS[pid]
            _PEERS_VERSION += 1
            logger.info(f"Mesh: peer {pid} timed out")


def get_peers() -> List[dict]:
    """Get all known peers as dicts."""
    with _PEERS_LOCK:
        return [p.to_dict() for p in _PEERS.values()]


def get_peer(peer_id: str) -> Optional[dict]:
    """Get a single peer by id."""
    with _PEERS_LOCK:
        p = _PEERS.get(peer_id)
        return p.to_dict() if p else None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Mesh — the mixin that gets embedded into Robot/Simulation
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


class Mesh:
    """Peer-to-peer mesh for a single Robot or Simulation instance.

    Created automatically inside Robot.__init__() / Simulation.__init__().
    Handles presence, state publishing, and command dispatch.

    This is NOT a wrapper — it's a component, like _task_state or _executor.
    The robot owns a Mesh, the Mesh holds a back-reference to the robot.

    Security:
        The mesh operates on a trusted-network model. By default it binds to
        localhost only. When STRANDS_MESH_SECRET is set, commands require a
        valid HMAC token. Unauthenticated commands to privileged actions
        (execute, start) are rejected with a warning.
    """

    def __init__(self, robot, peer_id: str, peer_type: str = "robot"):
        self.robot = robot  # back-reference (Robot or Simulation)
        self.peer_id = peer_id
        self.peer_type = peer_type
        self._running = False
        self._subs: list = []
        self._pending: Dict[str, threading.Event] = {}
        self._responses: Dict[str, list] = {}

    # ── lifecycle ──────────────────────────────────────────────

    def start(self):
        """Start mesh (called from Robot.__init__)."""
        if self._running:
            return

        session = _get_session()
        if session is None:
            logger.debug(f"{self.peer_id}: zenoh unavailable, mesh off")
            return

        self._running = True
        _LOCAL_ROBOTS[self.peer_id] = self

        # Subscribe
        self._subs.append(session.declare_subscriber("strands/*/presence", self._on_presence))
        self._subs.append(session.declare_subscriber("strands/broadcast", self._on_cmd))
        self._subs.append(session.declare_subscriber(f"strands/{self.peer_id}/cmd", self._on_cmd))
        self._subs.append(session.declare_subscriber(f"strands/{self.peer_id}/response/*", self._on_response))

        # Threads
        threading.Thread(target=self._heartbeat_loop, daemon=True).start()
        threading.Thread(target=self._state_loop, daemon=True).start()

        logger.info(f"🔗 {self.peer_id} on mesh ({self.peer_type})")

    def stop(self):
        """Stop mesh (called from Robot.cleanup)."""
        if not self._running:
            return
        self._running = False
        _LOCAL_ROBOTS.pop(self.peer_id, None)
        for sub in self._subs:
            try:
                sub.undeclare()
            except Exception:
                pass
        self._subs.clear()
        _release_session()
        logger.info(f"🔌 {self.peer_id} off mesh")

    @property
    def alive(self) -> bool:
        return self._running

    # ── presence ───────────────────────────────────────────────

    def _build_presence(self) -> dict:
        """Build presence payload from the robot."""
        r = self.robot
        p = {
            "robot_id": self.peer_id,
            "robot_type": self.peer_type,
            "hostname": socket.gethostname(),
            "timestamp": time.time(),
        }
        try:
            if hasattr(r, "tool_name_str"):
                p["tool_name"] = r.tool_name_str
            if hasattr(r, "_task_state"):
                p["task_status"] = r._task_state.status.value
                p["instruction"] = r._task_state.instruction
            if hasattr(r, "robot") and hasattr(r.robot, "is_connected"):
                p["connected"] = r.robot.is_connected
            if hasattr(r, "robot") and hasattr(r.robot, "name"):
                p["hw"] = r.robot.name
            if hasattr(r, "_action_features"):
                p["action_keys"] = list(r._action_features.keys())
            # Simulation
            if hasattr(r, "_robots"):
                p["sim_robots"] = list(r._robots.keys())
            if hasattr(r, "_world") and r._world is not None:
                p["world"] = True
        except Exception:
            pass
        return p

    def _heartbeat_loop(self):
        while self._running:
            _put(f"strands/{self.peer_id}/presence", self._build_presence())
            _prune_peers()
            time.sleep(1.0 / HEARTBEAT_HZ)

    def _on_presence(self, sample):
        try:
            d = json.loads(sample.payload.to_bytes().decode())
            pid = d.get("robot_id")
            if not pid or pid == self.peer_id:
                return
            is_new = _update_peer(pid, d.get("robot_type", "robot"), d.get("hostname", ""), d)
            if is_new:
                logger.info(f"🤖 New peer: {pid} ({d.get('robot_type', '?')})")
        except Exception:
            pass

    # ── state publishing (10Hz) ────────────────────────────────

    def _state_loop(self):
        while self._running:
            state = self._read_state()
            if state:
                _put(f"strands/{self.peer_id}/state", state)
            time.sleep(1.0 / STATE_HZ)

    def _read_state(self) -> Optional[dict]:
        """Read current robot/sim state for publishing."""
        r = self.robot
        s: dict = {"peer_id": self.peer_id, "t": time.time()}
        try:
            # Hardware robot
            if hasattr(r, "robot") and hasattr(r.robot, "get_observation"):
                if getattr(r.robot, "is_connected", False):
                    obs = r.robot.get_observation()
                    cam_keys = list(getattr(getattr(r.robot, "config", None), "cameras", {}).keys())
                    s["joints"] = {
                        k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in obs.items() if k not in cam_keys
                    }

            # Task state
            if hasattr(r, "_task_state"):
                ts = r._task_state
                s["task"] = {
                    "status": ts.status.value,
                    "instruction": ts.instruction,
                    "steps": ts.step_count,
                    "duration": ts.duration,
                }

            # Simulation
            if hasattr(r, "_world") and r._world is not None:
                w = r._world
                if hasattr(w, "_data") and w._data is not None:
                    s["sim_time"] = float(w._data.time)
                elif hasattr(r, "_data") and r._data is not None:
                    s["sim_time"] = float(r._data.time)
                if hasattr(r, "_robots"):
                    for name in r._robots:
                        s.setdefault("robots", {})[name] = {"active": True}

        except Exception:
            pass

        return s if len(s) > 2 else None

    # ── command handling ───────────────────────────────────────

    def _on_cmd(self, sample):
        try:
            d = json.loads(sample.payload.to_bytes().decode())
            if d.get("sender_id") == self.peer_id:
                return

            # Authentication: verify mesh token if STRANDS_MESH_SECRET is set
            if not _verify_mesh_token(d):
                sender = d.get("sender_id", "unknown")
                action = "unknown"
                cmd = d.get("command", d)
                if isinstance(cmd, dict):
                    action = cmd.get("action", "unknown")
                elif isinstance(cmd, str):
                    action = "execute"

                # Allow safe read-only actions without auth
                if action not in _SAFE_ACTIONS:
                    logger.warning(
                        "🔒 Mesh: rejected unauthenticated command from peer %s "
                        "(action=%s). Set STRANDS_MESH_SECRET on all peers to "
                        "enable authenticated mesh commands.",
                        sender,
                        action,
                    )
                    # Send rejection response
                    turn = d.get("turn_id", "")
                    if sender and turn:
                        _put(
                            f"strands/{sender}/response/{turn}",
                            {
                                "type": "error",
                                "responder_id": self.peer_id,
                                "turn_id": turn,
                                "error": "authentication_required",
                                "message": "Command rejected: STRANDS_MESH_SECRET authentication required",
                                "timestamp": time.time(),
                            },
                        )
                    return

            threading.Thread(target=self._exec_cmd, args=(d,), daemon=True).start()
        except json.JSONDecodeError as e:
            logger.debug(f"Mesh: invalid JSON in command: {e}")
        except Exception as e:
            logger.debug(f"Mesh: command handler error: {e}")

    def _exec_cmd(self, data: dict):
        sender = data.get("sender_id", "")
        turn = data.get("turn_id", uuid.uuid4().hex[:8])
        cmd = data.get("command", data)
        if isinstance(cmd, str):
            cmd = {"action": "execute", "instruction": cmd}

        rkey = f"strands/{sender}/response/{turn}" if sender else None

        try:
            result = self._dispatch(cmd)
            if rkey:
                _put(
                    rkey,
                    {
                        "type": "response",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "result": result,
                        "timestamp": time.time(),
                    },
                )
        except Exception as e:
            logger.debug(f"Mesh: command execution error: {e}")
            if rkey:
                _put(
                    rkey,
                    {
                        "type": "error",
                        "responder_id": self.peer_id,
                        "turn_id": turn,
                        "error": str(e),
                        "timestamp": time.time(),
                    },
                )

    def _dispatch(self, cmd: dict) -> dict:
        """Route command to robot methods.

        Security: Validates and sanitizes policy_host, model_path, and
        policy_provider fields before passing them to robot methods.
        """
        action = cmd.get("action", "status")
        r = self.robot

        if action == "status":
            if hasattr(r, "get_task_status"):
                return r.get_task_status()
            return {"status": getattr(getattr(r, "_task_state", None), "status", "unknown")}

        if action == "stop":
            return r.stop_task() if hasattr(r, "stop_task") else {"ok": True}

        if action == "features":
            return r.get_features() if hasattr(r, "get_features") else {}

        if action == "state":
            return self._read_state() or {}

        if action in ("execute", "start"):
            instr = cmd.get("instruction", "")
            if not instr:
                return {"error": "instruction required"}

            # Validate policy_provider against allowlist
            pp = cmd.get("policy_provider", "mock")
            if pp not in _ALLOWED_POLICY_PROVIDERS:
                logger.warning(
                    "Mesh: rejected unknown policy_provider %r from peer command. " "Allowed providers: %s",
                    pp,
                    ", ".join(sorted(_ALLOWED_POLICY_PROVIDERS)),
                )
                return {"error": f"policy_provider '{pp}' not in allowlist"}

            port = cmd.get("policy_port")
            # Validate and sanitize policy_host
            host = _validate_policy_host(cmd.get("policy_host", "localhost"))
            dur = cmd.get("duration", 30.0)
            # Cap duration to prevent abuse (max 5 minutes from mesh commands)
            dur = min(float(dur), 300.0)

            kw = {}
            for k in ("model_path", "server_address", "policy_type", "pretrained_name_or_path"):
                if k in cmd:
                    val = cmd[k]
                    # Validate path-like parameters
                    if k in ("model_path", "pretrained_name_or_path"):
                        val = _validate_model_path(val)
                        if val is None:
                            return {"error": f"invalid {k}: rejected by security validation"}
                    # Validate server_address (same rules as policy_host)
                    if k == "server_address" and isinstance(val, str):
                        if not re.match(r"^[a-zA-Z0-9.\-:\[\]/]+$", val):
                            logger.warning(f"Mesh: rejected invalid server_address: {val!r}")
                            return {"error": "invalid server_address"}
                    kw[k] = val

            if action == "execute" and hasattr(r, "_execute_task_sync"):
                return r._execute_task_sync(instr, pp, port, host, dur, **kw)
            if action == "start" and hasattr(r, "start_task"):
                return r.start_task(instr, pp, port, host, dur, **kw)

        # Sim actions
        if action == "step" and hasattr(r, "step"):
            return r.step(cmd.get("steps", 1))
        if action == "reset" and hasattr(r, "reset"):
            return r.reset()

        return {"error": f"unknown action: {action}"}

    def _on_response(self, sample):
        try:
            d = json.loads(sample.payload.to_bytes().decode())
            turn = d.get("turn_id")
            if turn in self._pending:
                self._responses.setdefault(turn, []).append(d)
                self._pending[turn].set()
        except Exception:
            pass

    # ── subscribe to ANY topic ─────────────────────────────────

    def subscribe(self, topic: str, callback=None, name: str = None):
        """Subscribe to ANY Zenoh topic. Returns parsed JSON dicts.

        This is how you listen to Reachy Mini, other robots, sensors — anything.

        Args:
            topic: Zenoh key expression (supports wildcards).
                Examples:
                    "reachy_mini/joint_positions"
                    "reachy_mini/*"              — all reachy topics
                    "*/joint_positions"          — joint positions from ANY robot
                    "strands/*/state"            — state from all strands robots
                    "my_sensor/imu"              — custom sensor
            callback: Called with (topic: str, data: dict) on each message.
                If None, messages are buffered in self.inbox[name].
            name: Subscription name for inbox access. Default: topic.

        Returns:
            Subscription name (use for unsubscribe or inbox access)

        Examples:
            # With callback
            def on_joints(topic, data):
                print(f"joints: {data}")
            mesh.subscribe("reachy_mini/joint_positions", on_joints)

            # Buffer mode (poll later)
            mesh.subscribe("reachy_mini/*", name="reachy")
            msgs = mesh.inbox["reachy"]  # list of (topic, data) tuples

            # Wildcard: all joint positions on the network
            mesh.subscribe("*/joint_positions")
        """
        if not self._running or not _SESSION:
            return None

        sub_name = name or topic
        if not hasattr(self, "inbox"):
            self.inbox = {}
        if not hasattr(self, "_user_subs"):
            self._user_subs = {}

        self.inbox.setdefault(sub_name, [])

        def handler(sample):
            try:
                key = str(sample.key_expr)
                raw = sample.payload.to_bytes().decode()
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    data = {"raw": raw}

                if callback:
                    callback(key, data)
                else:
                    self.inbox[sub_name].append((key, data))
                    # Cap buffer at 1000
                    if len(self.inbox[sub_name]) > 1000:
                        self.inbox[sub_name] = self.inbox[sub_name][-500:]
            except Exception as e:
                logger.debug(f"Subscribe handler error on {topic}: {e}")

        sub = _SESSION.declare_subscriber(topic, handler)
        self._subs.append(sub)
        self._user_subs[sub_name] = sub
        logger.info(f"📡 {self.peer_id} subscribed to: {topic}")
        return sub_name

    def unsubscribe(self, name: str):
        """Unsubscribe from a topic by name."""
        if not hasattr(self, "_user_subs"):
            return
        sub = self._user_subs.pop(name, None)
        if sub:
            try:
                sub.undeclare()
                self._subs.remove(sub)
            except Exception:
                pass
            self.inbox.pop(name, None)

    # ── VLA execution stream ───────────────────────────────────

    def publish_step(self, step: int, observation: dict, action: dict, instruction: str = "", policy: str = ""):
        """Publish a single policy execution step to the mesh.

        Called from the Robot control loop during VLA inference.
        Other peers can subscribe to strands/{peer_id}/stream to watch.

        Only publishes numeric data — camera frames are excluded.

        Args:
            step: Step number
            observation: Raw observation dict from robot
            action: Action dict sent to robot
            instruction: Current task instruction
            policy: Policy provider name
        """
        # Filter out camera frames (numpy arrays / tensors)
        obs_numeric = {}
        for k, v in observation.items():
            if hasattr(v, "shape") and len(getattr(v, "shape", ())) > 1:
                continue  # Skip images/tensors with >1 dim
            if hasattr(v, "tolist"):
                obs_numeric[k] = v.tolist()
            elif isinstance(v, (int, float, bool, str)):
                obs_numeric[k] = v
            elif isinstance(v, (list, tuple)) and len(v) < 100:
                obs_numeric[k] = list(v)

        act_numeric = {}
        for k, v in action.items():
            if hasattr(v, "tolist"):
                act_numeric[k] = v.tolist()
            elif isinstance(v, (int, float, bool, str, list, tuple)):
                act_numeric[k] = v

        _put(
            f"strands/{self.peer_id}/stream",
            {
                "peer_id": self.peer_id,
                "step": step,
                "t": time.time(),
                "instruction": instruction,
                "policy": policy,
                "observation": obs_numeric,
                "action": act_numeric,
            },
        )

    def on_stream(self, peer_id: str, callback=None):
        """Subscribe to another robot's VLA execution stream.

        Args:
            peer_id: The robot to watch
            callback: Called with (topic, data) for each step.
                If None, buffered in self.inbox[f"stream:{peer_id}"]

        Example:
            def on_step(topic, data):
                print(f"step {data['step']}: obs={data['observation']}")

            mesh.on_stream("so100-a1b2", on_step)
        """
        return self.subscribe(f"strands/{peer_id}/stream", callback, name=f"stream:{peer_id}")

    # ── outgoing messages ──────────────────────────────────────

    def send(self, target: str, cmd: dict, timeout: float = 30.0) -> dict:
        """Send command to a specific peer. Returns response.

        If STRANDS_MESH_SECRET is set, the command is signed with an HMAC
        token so the receiving peer can verify authenticity.
        """
        turn = uuid.uuid4().hex[:8]
        ev = threading.Event()
        self._pending[turn] = ev
        self._responses[turn] = []

        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}

        # Sign with mesh secret if configured
        if _MESH_SECRET is not None:
            msg["mesh_token"] = _compute_mesh_token(_MESH_SECRET, turn)

        _put(f"strands/{target}/cmd", msg)

        ev.wait(timeout=timeout)
        resps = self._responses.pop(turn, [])
        self._pending.pop(turn, None)
        return resps[0] if resps else {"status": "timeout"}

    def broadcast(self, cmd: dict, timeout: float = 5.0) -> List[dict]:
        """Broadcast command to all peers.

        If STRANDS_MESH_SECRET is set, the command is signed with an HMAC
        token so receiving peers can verify authenticity.
        """
        turn = uuid.uuid4().hex[:8]
        ev = threading.Event()
        self._pending[turn] = ev
        self._responses[turn] = []

        msg = {"sender_id": self.peer_id, "turn_id": turn, "command": cmd, "timestamp": time.time()}

        # Sign with mesh secret if configured
        if _MESH_SECRET is not None:
            msg["mesh_token"] = _compute_mesh_token(_MESH_SECRET, turn)

        _put("strands/broadcast", msg)

        ev.wait(timeout=timeout)
        time.sleep(0.3)
        resps = self._responses.pop(turn, [])
        self._pending.pop(turn, None)
        return resps

    def tell(self, target: str, instruction: str, **kw) -> dict:
        """Shorthand: tell a peer to execute an instruction."""
        cmd = {"action": "execute", "instruction": instruction, **kw}
        return self.send(target, cmd)

    def emergency_stop(self) -> List[dict]:
        """Broadcast stop to every peer."""
        return self.broadcast({"action": "stop"}, timeout=3.0)

    @property
    def peers(self) -> List[dict]:
        """All known peers."""
        return get_peers()


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# init_mesh() — called from Robot.__init__ and Simulation.__init__
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━


def init_mesh(robot, peer_id: str = None, peer_type: str = "robot", mesh: bool = True) -> Optional[Mesh]:
    """Initialize mesh for a Robot/Simulation.

    Called inside __init__(). Returns the Mesh instance (or None if disabled).
    The caller stores it as self.mesh.

    Args:
        robot: The Robot or Simulation instance (self)
        peer_id: Explicit peer ID. Default: tool_name_str
        peer_type: "robot" or "sim"
        mesh: False to disable entirely
    """
    # Global kill switch
    if os.getenv("STRANDS_MESH", "true").lower() == "false":
        mesh = False

    if not mesh:
        return None

    if peer_id is None:
        base = getattr(robot, "tool_name_str", "robot")
        peer_id = f"{base}-{uuid.uuid4().hex[:4]}"

    m = Mesh(robot, peer_id=peer_id, peer_type=peer_type)
    m.start()
    return m
