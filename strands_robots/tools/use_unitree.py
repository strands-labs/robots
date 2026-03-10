"""
Universal Unitree Robotics SDK tool — like use_aws wraps boto3, this wraps unitree_sdk2py.

Provides a unified `__getattr__`-style dispatch interface to control ALL Unitree robots
(Go2, B2, G1, H1, H1_2) through their DDS-based Python SDK. The agent discovers
available services and methods dynamically.

Architecture:
    use_unitree(action="go2.sport.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})
    → ChannelFactoryInitialize(domain_id, interface)
    → SportClient().Init()
    → SportClient().Move(vx=0.3, vy=0, vyaw=0)
    → {"status": "success", "return_code": 0}

Supported robots & services:
    Go2:   sport (40+ actions), video, robot_state, obstacles_avoid, vui
    B2:    sport, front_video, back_video, robot_state, vui
    G1:    loco (12+ actions), arm (16 gestures), audio (TTS/LED)
    H1:    loco (12+ actions)
    H1_2:  (low-level only)
    Common: motion_switcher

Usage:
    # High-level locomotion
    use_unitree(action="go2.sport.StandUp")
    use_unitree(action="go2.sport.Move", parameters={"vx": 0.5, "vy": 0, "vyaw": 0.2})

    # G1 humanoid arm gestures
    use_unitree(action="g1.arm.hug")
    use_unitree(action="g1.arm.ExecuteAction", parameters={"action_id": 19})

    # G1 locomotion
    use_unitree(action="g1.loco.Move", parameters={"vx": 0.3, "vy": 0, "vyaw": 0})

    # Discovery
    use_unitree(action="list_robots")
    use_unitree(action="list_services", parameters={"robot": "go2"})
    use_unitree(action="list_methods", parameters={"robot": "go2", "service": "sport"})

    # Diagnostics
    use_unitree(action="diagnose")

Environment variables:
    UNITREE_NETWORK_INTERFACE: Network interface name (e.g., "eth0", "enp2s0")
    UNITREE_DOMAIN_ID: DDS domain ID (default: 0)
    UNITREE_MOCK: Set to "true" for mock mode (no hardware/SDK required)
"""

import importlib
import inspect
import logging
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────
# Service Registry — maps (robot, service) → (module_path, client_class)
# ─────────────────────────────────────────────────────────────────────

_SERVICE_REGISTRY: Dict[Tuple[str, str], Tuple[str, str]] = {
    # Go2 — quadruped dog
    ("go2", "sport"): ("unitree_sdk2py.go2.sport.sport_client", "SportClient"),
    ("go2", "video"): ("unitree_sdk2py.go2.video.video_client", "VideoClient"),
    ("go2", "robot_state"): ("unitree_sdk2py.go2.robot_state.robot_state_client", "RobotStateClient"),
    ("go2", "obstacles_avoid"): ("unitree_sdk2py.go2.obstacles_avoid.obstacles_avoid_client", "ObstaclesAvoidClient"),
    ("go2", "vui"): ("unitree_sdk2py.go2.vui.vui_client", "VuiClient"),
    # B2 — industrial quadruped
    ("b2", "sport"): ("unitree_sdk2py.b2.sport.sport_client", "SportClient"),
    ("b2", "front_video"): ("unitree_sdk2py.b2.front_video.video_client", "VideoClient"),
    ("b2", "back_video"): ("unitree_sdk2py.b2.back_video.video_client", "VideoClient"),
    ("b2", "robot_state"): ("unitree_sdk2py.b2.robot_state.robot_state_client", "RobotStateClient"),
    ("b2", "vui"): ("unitree_sdk2py.b2.vui.vui_client", "VuiClient"),
    # G1 — humanoid
    ("g1", "loco"): ("unitree_sdk2py.g1.loco.g1_loco_client", "LocoClient"),
    ("g1", "arm"): ("unitree_sdk2py.g1.arm.g1_arm_action_client", "G1ArmActionClient"),
    ("g1", "audio"): ("unitree_sdk2py.g1.audio.g1_audio_client", "AudioClient"),
    # H1 — humanoid
    ("h1", "loco"): ("unitree_sdk2py.h1.loco.h1_loco_client", "LocoClient"),
    # Common services
    ("common", "motion_switcher"): (
        "unitree_sdk2py.comm.motion_switcher.motion_switcher_client",
        "MotionSwitcherClient",
    ),
}

# Robot metadata
_ROBOT_INFO: Dict[str, Dict[str, Any]] = {
    "go2": {
        "type": "quadruped",
        "description": "Unitree Go2 — consumer quadruped dog",
        "services": ["sport", "video", "robot_state", "obstacles_avoid", "vui"],
        "joints": 12,
    },
    "b2": {
        "type": "quadruped",
        "description": "Unitree B2 — industrial quadruped",
        "services": ["sport", "front_video", "back_video", "robot_state", "vui"],
        "joints": 12,
    },
    "g1": {
        "type": "humanoid",
        "description": "Unitree G1 — humanoid robot (29-DOF + dexterous hands)",
        "services": ["loco", "arm", "audio"],
        "joints": 29,
    },
    "h1": {
        "type": "humanoid",
        "description": "Unitree H1 — humanoid robot (19-DOF)",
        "services": ["loco"],
        "joints": 19,
    },
    "common": {
        "type": "shared",
        "description": "Shared services across all Unitree robots",
        "services": ["motion_switcher"],
        "joints": 0,
    },
}

# G1 arm gesture map (friendly name → action_id)
_G1_ARM_GESTURES: Dict[str, int] = {
    "release_arm": 99,
    "two_hand_kiss": 11,
    "left_kiss": 12,
    "right_kiss": 13,
    "hands_up": 15,
    "clap": 17,
    "high_five": 18,
    "hug": 19,
    "heart": 20,
    "right_heart": 21,
    "reject": 22,
    "right_hand_up": 23,
    "x_ray": 24,
    "face_wave": 25,
    "high_wave": 26,
    "shake_hand": 27,
}

# Dangerous actions that require explicit confirmation
_DANGEROUS_ACTIONS: set = {
    "BackFlip",
    "FrontFlip",
    "LeftFlip",
    "HandStand",
    "FrontJump",
    "FrontPounce",
    "ZeroTorque",
    "Damp",
}

# Velocity-clamped actions (auto-clamp vx/vy/vyaw to safe limits)
_VELOCITY_LIMITS: Dict[str, Dict[str, float]] = {
    "go2": {"vx": 1.5, "vy": 0.5, "vyaw": 1.0},
    "g1": {"vx": 0.6, "vy": 0.3, "vyaw": 0.5},
    "h1": {"vx": 0.8, "vy": 0.4, "vyaw": 0.6},
    "b2": {"vx": 1.0, "vy": 0.5, "vyaw": 0.8},
}


# ─────────────────────────────────────────────────────────────────────
# Connection Manager — singleton, thread-safe DDS client cache
# ─────────────────────────────────────────────────────────────────────


class _ConnectionManager:
    """Thread-safe DDS connection manager with client caching."""

    def __init__(self):
        self._lock = threading.Lock()
        self._channel_initialized = False
        self._clients: Dict[str, Any] = {}

    def _ensure_channel(self, interface: Optional[str], domain_id: int) -> None:
        """Initialize CycloneDDS channel factory (singleton, once per process)."""
        if self._channel_initialized:
            return

        with self._lock:
            if self._channel_initialized:
                return

            from unitree_sdk2py.core.channel import ChannelFactoryInitialize

            ChannelFactoryInitialize(domain_id, interface)
            self._channel_initialized = True
            logger.info("DDS channel initialized (domain=%d, interface=%s)", domain_id, interface)

    def get_client(
        self,
        robot: str,
        service: str,
        interface: Optional[str] = None,
        domain_id: int = 0,
    ) -> Any:
        """Get or create a cached SDK client for robot.service."""
        cache_key = f"{robot}.{service}:{interface}:{domain_id}"

        if cache_key in self._clients:
            return self._clients[cache_key]

        # Ensure DDS channel BEFORE acquiring lock (_ensure_channel has its
        # own internal locking; calling it inside self._lock would deadlock
        # because threading.Lock is not reentrant).
        self._ensure_channel(interface, domain_id)

        with self._lock:
            if cache_key in self._clients:
                return self._clients[cache_key]

            # Look up registry
            key = (robot, service)
            if key not in _SERVICE_REGISTRY:
                raise ValueError(
                    f"Unknown service: {robot}.{service}. " f"Available: {[f'{r}.{s}' for r, s in _SERVICE_REGISTRY]}"
                )

            module_path, class_name = _SERVICE_REGISTRY[key]
            mod = importlib.import_module(module_path)
            client_cls = getattr(mod, class_name)

            client = client_cls()
            client.Init()

            self._clients[cache_key] = client
            logger.info("Created client: %s.%s (%s)", robot, service, class_name)
            return client

    def close_all(self) -> None:
        """Close all cached clients."""
        with self._lock:
            self._clients.clear()
            logger.info("All clients closed")


_conn = _ConnectionManager()


# ─────────────────────────────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────────────────────────────


def _clamp_velocity(robot: str, params: Dict[str, Any]) -> Dict[str, Any]:
    """Auto-clamp velocity parameters to safe limits for the given robot."""
    limits = _VELOCITY_LIMITS.get(robot)
    if not limits:
        return params

    clamped = dict(params)
    for key, limit in limits.items():
        if key in clamped:
            val = float(clamped[key])
            if abs(val) > limit:
                clamped[key] = limit if val > 0 else -limit
                logger.warning("Clamped %s from %.2f to %.2f (limit: ±%.2f)", key, val, clamped[key], limit)
    return clamped


def _parse_action(action: str) -> Tuple[str, str, str]:
    """Parse dot-separated action string into (robot, service, method).

    Formats:
        "go2.sport.Move"           → ("go2", "sport", "Move")
        "g1.arm.hug"               → ("g1", "arm", "hug")
        "list_robots"              → ("__discovery__", "", "list_robots")
        "diagnose"                 → ("__discovery__", "", "diagnose")
    """
    parts = action.split(".")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    elif len(parts) == 2:
        # Could be "robot.service" (list methods) or discovery
        return parts[0], parts[1], ""
    elif len(parts) == 1:
        # Discovery action
        return "__discovery__", "", parts[0]
    else:
        raise ValueError(f"Invalid action format: {action!r}. Expected 'robot.service.method'")


def _get_client_methods(robot: str, service: str) -> List[Dict[str, Any]]:
    """Introspect a client class to list available methods and their signatures."""
    key = (robot, service)
    if key not in _SERVICE_REGISTRY:
        return []

    module_path, class_name = _SERVICE_REGISTRY[key]

    try:
        mod = importlib.import_module(module_path)
        client_cls = getattr(mod, class_name)
    except (ImportError, AttributeError):
        return []

    methods = []
    for name, func in inspect.getmembers(client_cls, predicate=inspect.isfunction):
        if name.startswith("_"):
            continue
        if name == "Init":
            continue

        sig = inspect.signature(func)
        params = []
        for pname, param in sig.parameters.items():
            if pname == "self":
                continue
            info = {
                "name": pname,
                "type": str(param.annotation.__name__) if param.annotation != inspect.Parameter.empty else "Any",
            }
            if param.default != inspect.Parameter.empty:
                info["default"] = param.default
            params.append(info)

        is_dangerous = name in _DANGEROUS_ACTIONS
        methods.append(
            {
                "name": name,
                "parameters": params,
                "dangerous": is_dangerous,
            }
        )

    return methods


def _resolve_gesture(method: str) -> Optional[int]:
    """Resolve a friendly gesture name to an action_id for G1 arm."""
    # Normalize: "hug" → "hug", "high five" → "high_five", "high-five" → "high_five"
    normalized = method.lower().replace(" ", "_").replace("-", "_")
    return _G1_ARM_GESTURES.get(normalized)


# ─────────────────────────────────────────────────────────────────────
# Mock mode for testing without hardware/SDK
# ─────────────────────────────────────────────────────────────────────


class _MockClient:
    """Mock client that records calls for testing."""

    def __init__(self, robot: str, service: str):
        self._robot = robot
        self._service = service
        self._calls: List[Dict] = []

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)

        def mock_method(**kwargs):
            self._calls.append({"method": name, "parameters": kwargs})
            logger.info("[MOCK] %s.%s.%s(%s)", self._robot, self._service, name, kwargs)
            return 0  # Success code

        return mock_method


def _is_mock_mode() -> bool:
    return os.getenv("UNITREE_MOCK", "").lower() in ("true", "1", "yes")


# ─────────────────────────────────────────────────────────────────────
# Main tool function
# ─────────────────────────────────────────────────────────────────────

try:
    from strands import tool as _tool_decorator
except ImportError:

    def _tool_decorator(f):
        return f


@_tool_decorator
def use_unitree(
    action: str,
    parameters: Optional[Dict[str, Any]] = None,
    interface: Optional[str] = None,
    domain_id: Optional[int] = None,
    confirm: bool = False,
) -> Dict[str, Any]:
    """Control Unitree robots via their DDS-based Python SDK.

    A universal dispatch interface for Go2, B2, G1, H1 quadrupeds and humanoids.
    Wraps unitree_sdk2_python with dynamic client creation, safety guards,
    and gesture shortcuts.

    Args:
        action: Dot-separated action string. Format: "robot.service.method"
            Examples:
                "go2.sport.Move"           — Move the Go2
                "go2.sport.StandUp"        — Stand the Go2 up
                "g1.loco.Move"             — Move the G1 humanoid
                "g1.arm.hug"               — G1 arm hug gesture
                "g1.arm.ExecuteAction"     — Execute arm action by ID
                "list_robots"              — List supported robots
                "list_services"            — List services for a robot
                "list_methods"             — List methods for a service
                "diagnose"                 — Check SDK installation status
        parameters: Method parameters as a dict.
            Examples:
                {"vx": 0.3, "vy": 0, "vyaw": 0}   — for Move
                {"action_id": 19}                    — for ExecuteAction
                {"level": 1}                         — for SpeedLevel
                {"robot": "go2"}                     — for list_services
                {"robot": "go2", "service": "sport"} — for list_methods
        interface: Network interface (default: UNITREE_NETWORK_INTERFACE env var)
        domain_id: DDS domain ID (default: UNITREE_DOMAIN_ID env var or 0)
        confirm: Required for dangerous actions (BackFlip, FrontFlip, etc.)

    Returns:
        Dict with status, return_code, and details.
    """
    if parameters is None:
        parameters = {}

    # Resolve network config from env if not provided
    if interface is None:
        interface = os.getenv("UNITREE_NETWORK_INTERFACE")
    if domain_id is None:
        domain_id = int(os.getenv("UNITREE_DOMAIN_ID", "0"))

    try:
        robot, service, method = _parse_action(action)
    except ValueError as e:
        return {"status": "error", "content": [{"text": str(e)}]}

    # ── Discovery actions ──
    if robot == "__discovery__" or method in ("list_robots", "list_services", "list_methods", "diagnose"):
        return _handle_discovery(method or action, parameters)

    # ── Gesture shortcut for G1 arm ──
    if robot == "g1" and service == "arm":
        gesture_id = _resolve_gesture(method)
        if gesture_id is not None:
            method = "ExecuteAction"
            parameters = {**parameters, "action_id": gesture_id}

    # ── Safety check for dangerous actions ──
    if method in _DANGEROUS_ACTIONS and not confirm:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"⚠️ '{method}' is a dangerous action. Pass confirm=True to execute. "
                    f"This action may cause the robot to perform a physically risky movement."
                }
            ],
        }

    # ── Velocity clamping for Move-like actions ──
    if method in ("Move", "SetVelocity"):
        parameters = _clamp_velocity(robot, parameters)

    # ── Execute ──
    try:
        if _is_mock_mode():
            client = _MockClient(robot, service)
        else:
            client = _conn.get_client(robot, service, interface, domain_id)

        # Get the method
        func = getattr(client, method, None)
        if func is None:
            available = [m for m in dir(client) if not m.startswith("_") and m != "Init"]
            return {
                "status": "error",
                "content": [
                    {"text": f"Method '{method}' not found on {robot}.{service}. " f"Available methods: {available}"}
                ],
            }

        # Call the method
        result = func(**parameters)

        # Format result
        if isinstance(result, tuple):
            code, data = result
            return {
                "status": "success" if code == 0 else "error",
                "content": [
                    {
                        "text": f"{robot}.{service}.{method} → code={code}",
                        "json": {"return_code": code, "data": data},
                    }
                ],
            }
        else:
            code = result if isinstance(result, int) else 0
            return {
                "status": "success" if code == 0 else "error",
                "content": [
                    {
                        "text": f"{robot}.{service}.{method} → code={code}",
                        "json": {"return_code": code},
                    }
                ],
            }

    except ImportError as e:
        return {
            "status": "error",
            "content": [
                {
                    "text": f"unitree_sdk2py not installed. Install with: "
                    f"pip install unitree_sdk2py\n\n"
                    f"Requires cyclonedds==0.10.2. See: "
                    f"https://github.com/unitreerobotics/unitree_sdk2_python\n\n"
                    f"Error: {e}"
                }
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"{robot}.{service}.{method} failed: {e}"}],
        }


def _handle_discovery(action: str, parameters: Dict[str, Any]) -> Dict[str, Any]:
    """Handle discovery/introspection actions."""

    if action == "list_robots":
        robots = []
        for name, info in _ROBOT_INFO.items():
            robots.append(
                {
                    "name": name,
                    "type": info["type"],
                    "description": info["description"],
                    "services": info["services"],
                    "joints": info["joints"],
                }
            )
        return {
            "status": "success",
            "content": [
                {
                    "text": f"Supported Unitree robots ({len(robots)}):\n"
                    + "\n".join(f"  • {r['name']} ({r['type']}) — {r['description']}" for r in robots),
                    "json": {"robots": robots},
                }
            ],
        }

    elif action == "list_services":
        robot = parameters.get("robot", "")
        if not robot:
            return {
                "status": "error",
                "content": [
                    {"text": 'Missing \'robot\' parameter. Example: list_services with parameters={"robot": "go2"}'}
                ],
            }

        info = _ROBOT_INFO.get(robot)
        if not info:
            return {
                "status": "error",
                "content": [{"text": f"Unknown robot: {robot!r}. Available: {list(_ROBOT_INFO.keys())}"}],
            }

        services = []
        for svc in info["services"]:
            key = (robot, svc)
            reg = _SERVICE_REGISTRY.get(key)
            services.append(
                {
                    "name": svc,
                    "module": reg[0] if reg else "unknown",
                    "client_class": reg[1] if reg else "unknown",
                }
            )

        return {
            "status": "success",
            "content": [
                {
                    "text": f"{robot} services ({len(services)}):\n"
                    + "\n".join(f"  • {s['name']} → {s['client_class']}" for s in services),
                    "json": {"robot": robot, "services": services},
                }
            ],
        }

    elif action == "list_methods":
        robot = parameters.get("robot", "")
        service = parameters.get("service", "")
        if not robot or not service:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "Missing 'robot' and/or 'service' parameters. "
                            'Example: list_methods with parameters={"robot": "go2", "service": "sport"}'
                        )
                    }
                ],
            }

        methods = _get_client_methods(robot, service)
        if not methods:
            return {
                "status": "error",
                "content": [{"text": f"No methods found for {robot}.{service}. Check service name."}],
            }

        # Add gesture shortcuts for G1 arm
        extra = ""
        if robot == "g1" and service == "arm":
            gesture_list = ", ".join(_G1_ARM_GESTURES.keys())
            extra = f"\n\nGesture shortcuts (use as method name): {gesture_list}"

        return {
            "status": "success",
            "content": [
                {
                    "text": f"{robot}.{service} methods ({len(methods)}):\n"
                    + "\n".join(
                        f"  • {m['name']}({', '.join(p['name'] + ': ' + p['type'] for p in m['parameters'])})"
                        + (" ⚠️ DANGEROUS" if m["dangerous"] else "")
                        for m in methods
                    )
                    + extra,
                    "json": {"robot": robot, "service": service, "methods": methods},
                }
            ],
        }

    elif action == "diagnose":
        diag = {
            "sdk_installed": False,
            "cyclonedds_installed": False,
            "mock_mode": _is_mock_mode(),
            "network_interface": os.getenv("UNITREE_NETWORK_INTERFACE", "(not set)"),
            "domain_id": os.getenv("UNITREE_DOMAIN_ID", "0"),
            "registered_services": len(_SERVICE_REGISTRY),
            "supported_robots": list(_ROBOT_INFO.keys()),
        }

        try:
            import unitree_sdk2py

            diag["sdk_installed"] = True
            diag["sdk_version"] = getattr(unitree_sdk2py, "__version__", "unknown")
        except ImportError:
            pass

        try:
            import cyclonedds

            diag["cyclonedds_installed"] = True
            diag["cyclonedds_version"] = getattr(cyclonedds, "__version__", "unknown")
        except ImportError:
            pass

        status_emoji = "✅" if diag["sdk_installed"] and diag["cyclonedds_installed"] else "⚠️"
        return {
            "status": "success",
            "content": [
                {
                    "text": f"{status_emoji} Unitree SDK Diagnostics:\n"
                    f"  SDK installed: {diag['sdk_installed']}\n"
                    f"  CycloneDDS installed: {diag['cyclonedds_installed']}\n"
                    f"  Mock mode: {diag['mock_mode']}\n"
                    f"  Network interface: {diag['network_interface']}\n"
                    f"  Domain ID: {diag['domain_id']}\n"
                    f"  Registered services: {diag['registered_services']}\n"
                    f"  Supported robots: {', '.join(diag['supported_robots'])}",
                    "json": diag,
                }
            ],
        }

    return {
        "status": "error",
        "content": [
            {"text": f"Unknown discovery action: {action!r}. Try: list_robots, list_services, list_methods, diagnose"}
        ],
    }
