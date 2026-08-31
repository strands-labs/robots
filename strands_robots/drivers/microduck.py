"""Native daemon driver for the Pollen Robotics Microduck.

``Robot("microduck", mode="real", port="/run/robotd.sock")`` builds one of
these. The instance satisfies
:class:`~strands_robots.drivers.base.HardwareDriver` structurally, so
:func:`~strands_robots.robot.Robot` returns it and the mesh, teleop rail and
agent tool surface consume it exactly like any other driver.

Why a native driver: the Microduck is driven by ``robotd``, a daemon that owns
the 50 Hz control loop *and runs the walking/skill policy on-device*. Its IPC
surface (``duck-ipc-proto``) is JSON-RPC 2.0, one object per line (NDJSON), over
a unix socket. There is no lerobot robot type for it and no serial servo bus to
address; the honest client speaks ``robotd``'s protocol.

THE DECISIVE FACT - robotd exposes **no per-joint write**. The whole ``robot.*``
surface is intent-level: ``robot.move`` (twist), ``robot.head``, ``robot.pose``,
``robot.do`` (skills), ``robot.enable``/``robot.relax`` (torque), ``robot.init``,
``robot.stop``, and ``robot.state`` (read). So ``mode="real"`` is *delegate-only*
by the robot's own design: this driver sends INTENTS and robotd's on-device
policy produces the joint targets. ``run_policy``/``start_task`` therefore do not
pretend to stream a MicroduckPolicy's 14 joint targets to hardware - the wire has
no such method - they refuse and name the intent path instead. Sim-to-real parity
is preserved regardless: the on-robot policy is the *same* ``alpha_walking.onnx``
run in sim (byte-compat 0.0), so a sim rollout predicts the hardware for equal obs.

Continuous intents (``robot.move``/``robot.head``/``robot.pose``/``robot.mouth``)
are sent as JSON-RPC *notifications* (no ``id``, no reply); discrete ones
(``robot.do``/``robot.enable``/``robot.stop``/``robot.relax``) as
*requests* whose id-correlated reply is awaited.

The 15-vs-14 papercut: robotd's ``JOINT_NAMES`` is 15 wide with ``"mouth"``
spliced at index 9; the policy/sim contract is the 14 locomotion joints (no
mouth). :meth:`MicroduckDriver.read_state` drops index 9 so the joints it
publishes are the 14 the policy speaks, and mouth travels via ``robot.mouth``.

Nothing here imports a transport at module load: every socket touch is inside a
method body, so the module imports on CI and in every unit test.
"""

from __future__ import annotations

import json
import logging
import socket
import threading
from collections.abc import AsyncGenerator
from typing import TYPE_CHECKING, Any, cast

from strands_robots.policies.microduck import MICRODUCK_JOINT_NAMES
from strands_robots.utils import (
    boolean_flag_error,
    finite_number_error,
    positive_count_error,
    positive_finite_number_error,
)

if TYPE_CHECKING:
    from strands.types.tools import ToolSpec, ToolUse

    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

#: The socket robotd serves (``duck-ipc-proto`` ``socket::ROBOT``). ``port=``
#: overrides it; a ``host:path`` form names a remote whose socket has been
#: forwarded (how ``duckctl`` reaches a robot over SSH).
DEFAULT_SOCKET: str = "/run/robotd.sock"

#: The API version this driver speaks, pinned to ``duck-ipc-proto`` ``API_VERSION``.
#: The Hello handshake refuses a robotd whose version differs rather than
#: mis-parsing its frames later.
MICRODUCK_API_VERSION: int = 16

#: JSON-RPC version string every frame carries.
JSONRPC_VERSION: str = "2.0"

#: robotd's HARDWARE joint order - 15 wide, ``"mouth"`` spliced at index 9.
#: ``robot.state`` ``joints``/``targets`` are indexed by this. Kept as the wire
#: truth; :data:`MOUTH_INDEX` is dropped to reach the 14 locomotion joints.
HARDWARE_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "mouth",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
)

#: Index of ``"mouth"`` in :data:`HARDWARE_JOINT_NAMES`; dropped to map 15->14.
MOUTH_INDEX: int = 9

#: The 14 locomotion joints the policy/sim contract speaks, in contract order.
#: Equal to :data:`HARDWARE_JOINT_NAMES` with index 9 removed - asserted in the
#: tests so a divergence between the wire map and the policy contract is caught.
LOCOMOTION_JOINT_NAMES: tuple[str, ...] = MICRODUCK_JOINT_NAMES

# robotd method names (duck-ipc-proto ``method`` module).
_M_HELLO = "hello"
_M_MOVE = "robot.move"
_M_HEAD = "robot.head"
_M_POSE = "robot.pose"
_M_MOUTH = "robot.mouth"
_M_DO = "robot.do"
_M_ENABLE = "robot.enable"
_M_RELAX = "robot.relax"
_M_STOP = "robot.stop"
_M_STATE = "robot.state"
_M_HEALTH = "robot.health"
_M_SUBSCRIBE = "robot.subscribe"

#: Skills robotd's ``robot.do`` accepts. The wire enum is ``snake_case``
#: (``Skill`` ``#[serde(rename_all="snake_case")]``), so a ``skill`` action value
#: is normalised to these before it is sent - a typo is refused here, not
#: silently no-op'd on the robot.
SKILLS: tuple[str, ...] = ("ground_pick", "kick_left", "kick_right", "sit_toggle", "roulade")

#: Action keys this driver knows how to turn into an intent, for the refusal
#: message when an action names none of them.
_ACTION_KEYS: tuple[str, ...] = (
    "vx",
    "vy",
    "vyaw",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "z",
    "roll",
    "pitch",
    "active",
    "open",
    "skill",
)


# --------------------------------------------------------------------------- #
# Pure wire encoding - no socket, so a test asserts the exact bytes.          #
# --------------------------------------------------------------------------- #


def _encode(obj: dict[str, Any]) -> bytes:
    """Serialise one JSON-RPC frame to a single NDJSON line.

    Compact separators (no spaces) and a trailing ``\\n`` match what
    ``serde_json`` emits on the robotd side, so the bytes this driver writes are
    the bytes a real robotd would accept.
    """
    return (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")


def _request(request_id: int, method: str, params: dict[str, Any]) -> bytes:
    """A JSON-RPC request (carries ``id``) as NDJSON bytes.

    Field order - ``jsonrpc``, ``id``, ``method``, ``params`` - matches the Rust
    ``Request`` struct so the serialised line is byte-identical. Methods that
    take no parameters send ``params: {}`` (an empty object), the same as
    robotd's ``Call::params`` for its unit variants.
    """
    return _encode({"jsonrpc": JSONRPC_VERSION, "id": request_id, "method": method, "params": params})


def _notification(method: str, params: dict[str, Any]) -> bytes:
    """A JSON-RPC notification (no ``id``, no reply) as NDJSON bytes."""
    return _encode({"jsonrpc": JSONRPC_VERSION, "method": method, "params": params})


def action_to_wire(action: dict[str, Any]) -> list[tuple[str, dict[str, Any], bool]] | str:
    """Translate a validated action dict into robotd intents.

    Returns a list of ``(method, params, is_notification)`` in a fixed order -
    twist, head, pose, mouth, skill - so two identical actions always produce
    the same wire sequence. Continuous intents are notifications
    (``is_notification=True``); ``robot.do`` is a request (``False``). Returns a
    reason string when a ``skill`` value is not a known skill, so the driver
    refuses at the door rather than sending a frame robotd will reject.

    The param structs mirror the Rust field order exactly:
    ``MoveParams{vx,vy,vyaw}``, ``HeadParams{neck_pitch,head_pitch,head_yaw,
    head_roll}``, ``PoseParams{z,roll,pitch,active}``, ``MouthParams{open}``,
    ``DoParams{skill}``.
    """
    if "active" in action and (reason := boolean_flag_error(action["active"], "active", "send_action")) is not None:
        # A posture flag selects a standing pose on hardware; reading it by
        # truthiness would send active=true for the string "false". Refuse.
        return reason

    commands: list[tuple[str, dict[str, Any], bool]] = []

    if any(k in action for k in ("vx", "vy", "vyaw")):
        commands.append(
            (
                _M_MOVE,
                {
                    "vx": float(action.get("vx", 0.0)),
                    "vy": float(action.get("vy", 0.0)),
                    "vyaw": float(action.get("vyaw", 0.0)),
                },
                True,
            )
        )

    if any(k in action for k in ("neck_pitch", "head_pitch", "head_yaw", "head_roll")):
        commands.append(
            (
                _M_HEAD,
                {
                    "neck_pitch": float(action.get("neck_pitch", 0.0)),
                    "head_pitch": float(action.get("head_pitch", 0.0)),
                    "head_yaw": float(action.get("head_yaw", 0.0)),
                    "head_roll": float(action.get("head_roll", 0.0)),
                },
                True,
            )
        )

    if any(k in action for k in ("z", "roll", "pitch", "active")):
        commands.append(
            (
                _M_POSE,
                {
                    "z": float(action.get("z", 0.0)),
                    "roll": float(action.get("roll", 0.0)),
                    "pitch": float(action.get("pitch", 0.0)),
                    "active": bool(action.get("active", True)),
                },
                True,
            )
        )

    if "open" in action:
        commands.append((_M_MOUTH, {"open": float(action["open"])}, True))

    if "skill" in action:
        skill = str(action["skill"]).strip().lower()
        if skill not in SKILLS:
            return f"unknown skill {action['skill']!r}; expected one of {list(SKILLS)}"
        commands.append((_M_DO, {"skill": skill}, False))

    return commands


def map_hardware_joints(values: list[float]) -> dict[str, float]:
    """Map robotd's 15-wide ``joints``/``targets`` to the 14 locomotion joints.

    Drops index 9 (``mouth``) and names the rest by
    :data:`LOCOMOTION_JOINT_NAMES`.

    A vector that is not 15 wide is mapped positionally for whatever it does
    carry rather than raising, so a robotd whose vector changed width degrades
    instead of crashing the reader thread. What "degrades" means differs by
    direction, and the two are not symmetric: a *shorter* vector yields a
    partial read (only the joints it reaches are named), while a *longer* one
    yields a full 14-joint read in which index 9 is no longer dropped, so every
    joint after the mouth is named one position early. Which of those a 14-wide
    vector is - robotd having dropped the mouth itself, or having dropped some
    other joint - is not knowable from the width, so this maps by position and
    leaves the reading to the caller rather than guessing.
    """
    if len(values) == len(HARDWARE_JOINT_NAMES):
        locomotion = [v for i, v in enumerate(values) if i != MOUTH_INDEX]
        return dict(zip(LOCOMOTION_JOINT_NAMES, locomotion, strict=True))
    return {name: float(v) for name, v in zip(LOCOMOTION_JOINT_NAMES, values, strict=False)}


def parse_robot_state(params: dict[str, Any]) -> dict[str, Any]:
    """Normalise a ``robot.state`` payload into the driver's cached shape.

    Pure, so a test parses a real RobotState fixture without a socket. The
    ``move``/``loop`` wire keys (renamed from the Rust ``movement``/
    ``control_loop`` fields) are read as they arrive; ``joints`` and ``targets``
    are mapped 15->14 via :func:`map_hardware_joints`.

    Args:
        params: The ``params`` object of a ``robot.state`` notification.

    Returns:
        A dict with ``t``, ``policy``, ``move``, ``head``, ``safety``, ``loop``,
        ``odom`` verbatim, plus ``joints``/``targets`` as 14-joint name->angle
        maps.
    """
    return {
        "t": params.get("t"),
        "policy": params.get("policy"),
        "move": params.get("move", {}),
        "head": params.get("head", []),
        "safety": params.get("safety", {}),
        "loop": params.get("loop", {}),
        "odom": params.get("odom", {}),
        "joints": map_hardware_joints(list(params.get("joints", []))),
        "targets": map_hardware_joints(list(params.get("targets", []))),
    }


def _refuse(reason: str) -> dict[str, Any]:
    """The driver's error envelope, one shape for every refusal path."""
    return {"status": "error", "content": [{"text": reason}]}


# --------------------------------------------------------------------------- #
# Transport - one NDJSON JSON-RPC conversation over a unix socket.            #
# --------------------------------------------------------------------------- #


class _RobotdClient:
    """A single robotd connection: one reader thread, id-correlated replies.

    All inbound bytes are read by one thread. A discrete request registers an
    event keyed by its id which the reader fulfils; a ``robot.state``
    notification is handed to ``on_state``. Writes are serialised by a lock so a
    notification cannot interleave a request mid-line. The Hello handshake runs
    synchronously before the reader starts, since nothing else reads yet.
    """

    def __init__(self, socket_path: str, *, timeout: float = 5.0) -> None:
        self._path = socket_path
        self._timeout = timeout
        self._sock: socket.socket | None = None
        self._rfile: Any = None
        self._wlock = threading.Lock()
        self._id_lock = threading.Lock()
        self._next_id = 0
        self._pending: dict[int, tuple[threading.Event, dict[str, Any]]] = {}
        self._pending_lock = threading.Lock()
        self._reader: threading.Thread | None = None
        self._on_state: Any = None
        self._stop = threading.Event()
        self.alive = False

    def _alloc_id(self) -> int:
        with self._id_lock:
            self._next_id += 1
            return self._next_id

    def _write(self, data: bytes) -> None:
        sock = self._sock
        if sock is None:
            raise ConnectionError("robotd socket is not connected")
        with self._wlock:
            sock.sendall(data)

    def connect(self) -> None:
        """Open the unix socket. Raises on failure so the caller names it."""
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        sock.connect(self._path)
        self._sock = sock
        self._rfile = sock.makefile("rb")
        self.alive = True

    def hello(self, api_version: int) -> dict[str, Any]:
        """Send the Hello handshake and return its ``result`` (synchronous)."""
        self._write(_request(self._alloc_id(), _M_HELLO, {"api_version": api_version}))
        line = self._rfile.readline()
        if not line:
            raise ConnectionError("robotd closed the connection during the Hello handshake")
        obj = json.loads(line)
        if obj.get("error"):
            raise ConnectionError(f"robotd refused Hello: {obj['error']}")
        return obj.get("result") or {}

    def start_reader(self, on_state: Any) -> None:
        """Begin the background reader that dispatches replies and state."""
        self._on_state = on_state
        self._reader = threading.Thread(target=self._read_loop, name="robotd-reader", daemon=True)
        self._reader.start()

    def _read_loop(self) -> None:
        try:
            for line in self._rfile:
                if self._stop.is_set():
                    break
                if not line.strip():
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    logger.debug("robotd sent an unparseable line: %r", line[:200])
                    continue
                if "method" not in obj and obj.get("id") is not None:
                    self._resolve(obj)
                elif obj.get("method") == _M_STATE and self._on_state is not None:
                    self._on_state(obj.get("params") or {})
        except OSError as exc:
            logger.debug("robotd reader stopped: %s", exc)
        finally:
            self.alive = False

    def _resolve(self, obj: dict[str, Any]) -> None:
        with self._pending_lock:
            slot = self._pending.pop(obj["id"], None)
        if slot is not None:
            event, box = slot
            box["result"] = obj.get("result")
            box["error"] = obj.get("error")
            event.set()

    def call(self, method: str, params: dict[str, Any], timeout: float | None = None) -> dict[str, Any]:
        """Send a request and block for its id-correlated reply.

        Returns:
            The reply's ``result`` object (``{}`` when absent).

        Raises:
            TimeoutError: If no reply with the matching id arrives in time.
            ConnectionError: If robotd returned a JSON-RPC error.
        """
        rid = self._alloc_id()
        event = threading.Event()
        box: dict[str, Any] = {}
        with self._pending_lock:
            self._pending[rid] = (event, box)
        self._write(_request(rid, method, params))
        if not event.wait(timeout or self._timeout):
            with self._pending_lock:
                self._pending.pop(rid, None)
            raise TimeoutError(f"robotd did not answer {method!r} within {timeout or self._timeout}s")
        if box.get("error"):
            raise ConnectionError(f"robotd error on {method!r}: {box['error']}")
        return box.get("result") or {}

    def notify(self, method: str, params: dict[str, Any]) -> None:
        """Send a notification (no id, no reply)."""
        self._write(_notification(method, params))

    def close(self) -> None:
        """Stop the reader and close the socket. Idempotent."""
        self._stop.set()
        self.alive = False
        if self._sock is not None:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # already half-closed; a shutdown race here is not actionable
            try:
                self._sock.close()
            except OSError:
                pass  # closing a socket that is already gone is fine
        self._sock = None
        self._rfile = None


# --------------------------------------------------------------------------- #
# The driver.                                                                 #
# --------------------------------------------------------------------------- #


class MicroduckDriver:
    """Native robotd delegate driver for the Pollen Microduck.

    Satisfies :class:`~strands_robots.drivers.base.HardwareDriver` structurally,
    so no import from that module is needed; the surface check
    :func:`~strands_robots.drivers.register_native_driver` runs at registration
    pins the contract.
    """

    def __init__(
        self,
        tool_name: str = "microduck",
        cameras: dict[str, dict[str, Any]] | None = None,
        data_config: str | None = None,
        *,
        port: str | None = None,
        api_version: int = MICRODUCK_API_VERSION,
        timeout: float = 5.0,
        subscribe_hz: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Record configuration; :meth:`connect_eagerly` talks to robotd.

        Args:
            tool_name: Name the agent invokes the driver by, and the mesh peer id.
            cameras: Accepted for parity with other drivers; unused here (the
                Microduck's cameras are not addressed by this driver).
            data_config: Accepted for parity; unused.
            port: The robotd unix socket path. Defaults to
                :data:`DEFAULT_SOCKET`. For a remote robot, forward its socket to
                a local path (as ``duckctl`` does over SSH) and pass that path.
            api_version: API version to send in the Hello handshake, pinned to
                :data:`MICRODUCK_API_VERSION`. A robotd answering a different
                version is refused, not mis-parsed.
            timeout: Socket and request timeout in seconds. A positive, finite
                number: it is handed to ``socket.settimeout`` and to the reply
                wait, neither of which can report what it was given.
            subscribe_hz: State-stream decimation. ``None`` = every control tick;
                otherwise a positive integer, because it is sent to robotd as one.
            **kwargs: Ignored; accepted so the factory can forward extras.

        Raises:
            ValueError: If ``timeout`` is not a positive finite number, or
                ``subscribe_hz`` is neither ``None`` nor a positive integer.
                Raised here rather than returned from
                :meth:`connect_eagerly`, which is declared ``-> str | None``:
                a value the transport cannot use is not a connection this
                driver can degrade to reporting.
        """
        del cameras, data_config
        if kwargs:
            logger.debug("MicroduckDriver ignoring extra kwargs: %s", sorted(kwargs))

        # The two transport knobs are held to the shared numeric domains for
        # the same reason the actuation flags are held to ``boolean_flag_error``:
        # each reaches a consumer that cannot report what it was handed.
        # ``timeout`` goes to ``socket.settimeout`` and to the reply wait, so a
        # ``nan``, an ``inf``, a negative or a numeric string raised out of
        # :meth:`connect_eagerly` from inside the socket call - naming neither
        # this driver nor the parameter, out of a method declared
        # ``-> str | None`` - while ``True`` acted as a silent one second and
        # ``None`` left both the socket and the reply wait unbounded.
        # ``subscribe_hz`` is interpolated straight into the ``robot.subscribe``
        # params: ``nan``/``inf`` serialise to the bare ``NaN``/``Infinity``
        # tokens, which are not JSON (RFC 8259), so a strict daemon parser
        # refuses the frame while ``connect_eagerly`` reports success; a
        # ``numpy`` integer is not JSON-serialisable at all.  The strict-int
        # domain is the right member because robotd is sent an integer.
        if reason := positive_finite_number_error(timeout, "timeout", "MicroduckDriver"):
            raise ValueError(reason)
        if subscribe_hz is not None and (
            reason := positive_count_error(subscribe_hz, "subscribe_hz", "MicroduckDriver")
        ):
            raise ValueError(reason)

        self._tool_name = tool_name
        self._socket_path = str(port) if port else DEFAULT_SOCKET
        self._api_version = api_version
        self._timeout = timeout
        self._subscribe_hz = subscribe_hz

        self._cache_lock = threading.Lock()
        self._joints: dict[str, float] = {}
        self._pose: dict[str, Any] | None = None
        self._imu: dict[str, Any] | None = None
        self._battery: dict[str, Any] | None = None
        self._last_state: dict[str, Any] | None = None

        self._client: _RobotdClient | None = None
        self._connected: bool = False
        self._connect_error: str | None = None
        self._hello: dict[str, Any] | None = None
        self._stopped: bool = False

    # ------------------------------------------------------------------ #
    # Agent tool surface.                                                #
    # ------------------------------------------------------------------ #

    @property
    def tool_name(self) -> str:
        """The name the Strands agent invokes this driver by."""
        return self._tool_name

    @property
    def tool_type(self) -> str:
        """Always ``\"robot\"`` - mirrors the other drivers."""
        return "robot"

    @property
    def is_connected(self) -> bool:
        """Whether the robotd connection is live, for the mesh joint read."""
        return self._connected and self._client is not None and self._client.alive

    @property
    def tool_spec(self) -> ToolSpec:
        """A minimal agent-facing spec: read sensors, report status, stop."""
        return cast(
            "ToolSpec",
            {
                "name": self._tool_name,
                "description": (
                    "Pollen Microduck native driver: delegates to the on-robot robotd daemon over "
                    "its JSON-RPC unix socket. Reads joints/pose/battery, reports status, and stops "
                    "the robot. Motion intents (walk twist, head, pose, skills) go through send_action."
                ),
                "inputSchema": {
                    "json": {
                        "type": "object",
                        "properties": {
                            "action": {
                                "type": "string",
                                "description": (
                                    "sensors: latest cached joints/pose/battery; "
                                    "status: robotd reachability and version; "
                                    "stop: ask robotd to stop the robot"
                                ),
                                "enum": ["sensors", "status", "stop"],
                                "default": "sensors",
                            },
                        },
                        "required": ["action"],
                    }
                },
            },
        )

    async def stream(
        self,
        tool_use: ToolUse,
        invocation_state: dict[str, Any],
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        """Handle one agent invocation and yield exactly one tool result."""
        del kwargs, invocation_state
        tool_use_id = tool_use.get("toolUseId", "")
        action = (tool_use.get("input") or {}).get("action", "sensors")
        if action == "sensors":
            envelope: dict[str, Any] = {
                "status": "success",
                "content": [
                    {
                        "json": {
                            "joints": self._snapshot("_joints"),
                            "pose": self._snapshot("_pose"),
                            "imu": self._snapshot("_imu"),
                            "battery": self._snapshot("_battery"),
                        }
                    }
                ],
            }
        elif action == "status":
            envelope = {"status": "success", "content": [{"json": await self.get_status()}]}
        else:  # "stop"
            # Report the halt outcome rather than assert one.  ``stop`` is the
            # protocol's shutdown hook and returns ``None``: it returns early
            # for a client that is gone and swallows an ``OSError`` from
            # robotd, so an envelope built beside it can only restate the
            # intent - and its text named a socket nothing had been written to.
            # ``stop_task`` performs the same ``robot.stop`` call and already
            # decides the verdict, so the verb returns that envelope rather
            # than re-deriving one.
            envelope = self.stop_task()
        yield {"toolUseId": tool_use_id, **envelope}

    # ------------------------------------------------------------------ #
    # Lifecycle.                                                         #
    # ------------------------------------------------------------------ #

    def connect_eagerly(self) -> str | None:
        """Connect to robotd, Hello-handshake, then subscribe to state.

        Returns ``None`` on success. Off hardware - no socket at
        :data:`_socket_path` - returns a reason naming the socket that did not
        answer and leaves the driver usable (every read returns its empty cache,
        every write refuses "not connected"). Idempotent: a second call on a live
        connection is a no-op success.

        A version mismatch is a *refusal*, not a silent downgrade: the reason
        names both versions, because a driver that mis-parsed a newer robotd's
        frames would be worse than one that declined to talk to it.
        """
        if self.is_connected:
            return None

        client = _RobotdClient(self._socket_path, timeout=self._timeout)
        try:
            client.connect()
        except OSError as exc:
            self._connect_error = f"robotd socket {self._socket_path!r} did not answer: {exc}"
            return self._connect_error

        try:
            hello = client.hello(self._api_version)
        except (OSError, ValueError) as exc:
            client.close()
            self._connect_error = f"robotd Hello failed: {exc}"
            return self._connect_error

        their_version = hello.get("api_version")
        if their_version != self._api_version:
            client.close()
            self._connect_error = (
                f"robotd speaks api_version {their_version}, this driver speaks {self._api_version}; "
                "refusing rather than mis-parsing its frames"
            )
            return self._connect_error

        client.start_reader(self._on_state)
        try:
            client.call(_M_SUBSCRIBE, {} if self._subscribe_hz is None else {"hz": self._subscribe_hz})
        except OSError as exc:
            client.close()
            self._connect_error = f"robot.subscribe failed: {exc}"
            return self._connect_error

        # Battery is not in robot.state - it rides on robot.health. Best-effort:
        # a robotd that cannot read the bus omits it, and the driver stays up.
        try:
            self._absorb_health(client.call(_M_HEALTH, {}))
        except OSError as exc:
            logger.debug("%s: robot.health unavailable at connect: %s", self._tool_name, exc)

        self._client = client
        self._hello = hello
        self._connected = True
        self._connect_error = None
        return None

    async def get_status(self) -> dict[str, Any]:
        """Report reachability, robotd version and the latest battery read."""
        return {
            "status": "success",
            "content": [
                {
                    "json": {
                        "tool_name": self._tool_name,
                        "connected": self.is_connected,
                        "connect_error": self._connect_error,
                        "socket": self._socket_path,
                        "api_version": (self._hello or {}).get("api_version"),
                        "daemon_version": (self._hello or {}).get("daemon_version"),
                        "motion_stopped": self._stopped,
                        "battery_pct": (self._battery or {}).get("pct"),
                    }
                }
            ],
        }

    async def stop(self) -> None:
        """Ask robotd to stop the robot (``robot.stop``)."""
        if self._client is None or not self._client.alive:
            return
        try:
            self._client.call(_M_STOP, {})
            self._stopped = True
        except OSError as exc:
            logger.warning("%s.stop(): robotd refused the stop: %s", self._tool_name, exc)

    def cleanup(self) -> None:
        """Close the robotd connection. Idempotent."""
        if self._client is not None:
            self._client.close()
        self._client = None
        self._connected = False

    # ------------------------------------------------------------------ #
    # Command path.                                                      #
    # ------------------------------------------------------------------ #

    def send_action(
        self,
        action: dict[str, Any],
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Map an action dict onto robotd intents and send them.

        Gates, in order: this driver fronts this robot; it is connected; every
        numeric value is finite; the action names at least one intent this driver
        can send. Continuous intents (twist/head/pose/mouth) are notifications;
        a ``skill`` is a ``robot.do`` request whose IntentResult is returned.

        Accepted keys: ``vx``/``vy``/``vyaw`` (twist, m/s and rad/s),
        ``neck_pitch``/``head_pitch``/``head_yaw``/``head_roll`` (rad),
        ``z``/``roll``/``pitch``/``active`` (standing pose), ``open`` (mouth
        0..1), ``skill`` (one of :data:`SKILLS`).
        """
        if robot_name is not None and robot_name != self._tool_name:
            return _refuse(f"send_action: this driver fronts {self._tool_name!r} only, not {robot_name!r}")
        if not self.is_connected or self._client is None:
            return _refuse("not connected - call connect_eagerly() first")

        for name, value in action.items():
            if name in ("skill", "active"):
                continue
            if (reason := finite_number_error(value, name, "send_action")) is not None:
                return _refuse(reason)

        commands = action_to_wire(action)
        if isinstance(commands, str):
            return _refuse(f"send_action: {commands}")
        if not commands:
            return _refuse(
                f"send_action: nothing to send - none of {sorted(action)} names a Microduck intent; "
                f"expected any of {sorted(_ACTION_KEYS)}"
            )

        sent: list[dict[str, Any]] = []
        for method, params, is_notification in commands:
            try:
                if is_notification:
                    self._client.notify(method, params)
                    sent.append({"method": method, "params": params})
                else:
                    result = self._client.call(method, params)
                    sent.append({"method": method, "params": params, "result": result})
            except OSError as exc:
                return _refuse(f"send_action: {method} failed: {exc}")
        self._stopped = False
        return {"status": "success", "content": [{"json": {"sent": sent, "robot": self._tool_name}}]}

    # ------------------------------------------------------------------ #
    # Task and policy paths - delegate-only, so these refuse.            #
    # ------------------------------------------------------------------ #

    def start_task(
        self,
        instruction: str,
        policy_port: int | None = None,
        policy_host: str = "localhost",
        policy_provider: str = "groot",
        duration: float = 30.0,
        **policy_kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: robotd runs the policy on-device, there is no joint-stream path."""
        del instruction, policy_port, policy_host, policy_provider, duration, policy_kwargs
        return _refuse(
            "start_task: robotd runs the walking/skill policy on-device and exposes no per-joint write; "
            "send an intent through send_action (twist vx/vy/vyaw, or skill=...) instead"
        )

    def run_policy(
        self,
        policy_object: Policy,
        instruction: str = "",
        duration: float = 30.0,
        n_steps: int | None = None,
    ) -> dict[str, Any]:
        """Refuse a host-driven rollout, for the same reason as :meth:`start_task`.

        The sim path (``Robot("microduck", mode="sim").run_policy``) runs a
        MicroduckPolicy against MuJoCo; on hardware the *same* ONNX runs inside
        robotd, so there is nothing for a host rollout to stream.
        """
        del policy_object, instruction, duration, n_steps
        return _refuse(
            "run_policy: on hardware the policy runs inside robotd (the same ONNX proven in sim); "
            "robotd exposes no per-joint write to stream targets to. Use send_action intents, or "
            'mode="sim" for a host-driven MicroduckPolicy rollout'
        )

    def get_task_status(self) -> dict[str, Any]:
        """Report the policy robotd is running, from the last state frame."""
        with self._cache_lock:
            policy = (self._last_state or {}).get("policy")
        return {
            "status": "success",
            "content": [{"json": {"running": policy not in (None, "held"), "policy": policy}}],
        }

    def stop_task(self) -> dict[str, Any]:
        """Stop the robot, the closest thing to halting an on-device policy."""
        if self._client is None or not self._client.alive:
            return _refuse("stop_task: not connected")
        try:
            self._client.call(_M_STOP, {})
        except OSError as exc:
            return _refuse(f"stop_task: robotd refused the stop: {exc}")
        self._stopped = True
        return {"status": "success", "content": [{"text": "asked robotd to stop the robot (robot.stop)"}]}

    # ------------------------------------------------------------------ #
    # Torque + emergency stop - discrete intents.                        #
    # ------------------------------------------------------------------ #

    def enable_torque(self, on: bool = True) -> dict[str, Any]:
        """Enable or disable the policy/torque via ``robot.enable``."""
        if (reason := boolean_flag_error(on, "on", "enable_torque")) is not None:
            # on=false energising the servos is the failure mode this refuses.
            return _refuse(reason)
        return self._discrete(_M_ENABLE, {"on": on, "toggle": False}, "enable_torque")

    def relax(self) -> dict[str, Any]:
        """Cut joint power via ``robot.relax`` (the robot collapses if unheld)."""
        return self._discrete(_M_RELAX, {}, "relax")

    def emergency_stop(self) -> dict[str, Any]:
        """Stop the robot immediately via ``robot.stop``."""
        return self._discrete(_M_STOP, {}, "emergency_stop")

    def _discrete(self, method: str, params: dict[str, Any], label: str) -> dict[str, Any]:
        if self._client is None or not self._client.alive:
            return _refuse(f"{label}: not connected")
        try:
            result = self._client.call(method, params)
        except OSError as exc:
            return _refuse(f"{label}: {method} failed: {exc}")
        return {"status": "success", "content": [{"json": {"method": method, "result": result}}]}

    # ------------------------------------------------------------------ #
    # Mesh telemetry.                                                    #
    # ------------------------------------------------------------------ #

    def get_observation(self) -> dict[str, float]:
        """The 14 locomotion joints (name -> radians) for the mesh joint read."""
        with self._cache_lock:
            return dict(self._joints)

    def read_state(self) -> dict[str, Any]:
        """The last robot.state frame the subscribe stream delivered, or a refusal."""
        with self._cache_lock:
            state = self._last_state
        if state is None:
            return _refuse("read_state: no robot.state received yet (not connected, or no frames)")
        return {"status": "success", "content": [{"json": state}]}

    # ------------------------------------------------------------------ #
    # Reader callback + snapshots. Run on the reader thread; keep fast.  #
    # ------------------------------------------------------------------ #

    def _on_state(self, params: dict[str, Any]) -> None:
        state = parse_robot_state(params)
        safety = state.get("safety") or {}
        odom = state.get("odom") or {}
        with self._cache_lock:
            self._last_state = state
            self._joints = dict(state.get("joints") or {})
            self._imu = {"projected_gravity": safety.get("gravity")}
            self._pose = {"position": odom.get("position"), "yaw": odom.get("yaw")}

    def _absorb_health(self, health: dict[str, Any]) -> None:
        battery = (health or {}).get("battery")
        if isinstance(battery, dict):
            with self._cache_lock:
                self._battery = {"pct": battery.get("percent"), "volts": battery.get("volts")}

    def _snapshot(self, attr: str) -> Any:
        with self._cache_lock:
            value = getattr(self, attr)
            if isinstance(value, dict):
                return dict(value)
            return value
