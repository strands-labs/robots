"""A fake rclpy + rosidl_runtime_py + message packages, good enough to test strands_robots.ros without ROS.

Messages are strict (unknown fields raise, like real rosidl classes). Delivery is filtered by topic AND type
(rmw_zenoh matches by type) and by QoS: a BEST_EFFORT writer never reaches a RELIABLE reader, and a
TRANSIENT_LOCAL writer replays its last message to a late TRANSIENT_LOCAL reader. Callbacks run on the
executor thread, as in rclpy. Vendor message definitions here are field NAMES only, written from the
public field lists; no vendor source.
"""

from __future__ import annotations

import enum
import queue
import sys
import threading
import time
import types
from typing import Any

# field spec: name -> type string. "[]" suffix = sequence, "[N]" = fixed array, "pkg/Name" = nested
SPECS: dict[str, dict[str, str]] = {
    "builtin_interfaces/Time": {"sec": "int32", "nanosec": "uint32"},
    "builtin_interfaces/Duration": {"sec": "int32", "nanosec": "uint32"},
    "std_msgs/Header": {"stamp": "builtin_interfaces/Time", "frame_id": "string"},
    "std_msgs/String": {"data": "string"},
    "std_msgs/Bool": {"data": "boolean"},
    "std_msgs/Float64MultiArray": {"data": "double[]"},
    "geometry_msgs/Vector3": {"x": "double", "y": "double", "z": "double"},
    "geometry_msgs/Point": {"x": "double", "y": "double", "z": "double"},
    "geometry_msgs/Quaternion": {"x": "double", "y": "double", "z": "double", "w": "double"},
    "geometry_msgs/Pose": {"position": "geometry_msgs/Point", "orientation": "geometry_msgs/Quaternion"},
    "geometry_msgs/PoseWithCovariance": {"pose": "geometry_msgs/Pose", "covariance": "double[36]"},
    "geometry_msgs/Twist": {"linear": "geometry_msgs/Vector3", "angular": "geometry_msgs/Vector3"},
    "geometry_msgs/TwistWithCovariance": {"twist": "geometry_msgs/Twist", "covariance": "double[36]"},
    "geometry_msgs/Transform": {"translation": "geometry_msgs/Vector3", "rotation": "geometry_msgs/Quaternion"},
    "geometry_msgs/TransformStamped": {
        "header": "std_msgs/Header",
        "child_frame_id": "string",
        "transform": "geometry_msgs/Transform",
    },
    "tf2_msgs/TFMessage": {"transforms": "geometry_msgs/TransformStamped[]"},
    "nav_msgs/Odometry": {
        "header": "std_msgs/Header",
        "child_frame_id": "string",
        "pose": "geometry_msgs/PoseWithCovariance",
        "twist": "geometry_msgs/TwistWithCovariance",
    },
    "sensor_msgs/JointState": {
        "header": "std_msgs/Header",
        "name": "string[]",
        "position": "double[]",
        "velocity": "double[]",
        "effort": "double[]",
    },
    "sensor_msgs/Imu": {
        "header": "std_msgs/Header",
        "orientation": "geometry_msgs/Quaternion",
        "angular_velocity": "geometry_msgs/Vector3",
        "linear_acceleration": "geometry_msgs/Vector3",
    },
    "sensor_msgs/Image": {
        "header": "std_msgs/Header",
        "height": "uint32",
        "width": "uint32",
        "encoding": "string",
        "is_bigendian": "uint8",
        "step": "uint32",
        "data": "uint8[]",
    },
    "trajectory_msgs/JointTrajectoryPoint": {
        "positions": "double[]",
        "velocities": "double[]",
        "time_from_start": "builtin_interfaces/Duration",
    },
    "trajectory_msgs/JointTrajectory": {
        "header": "std_msgs/Header",
        "joint_names": "string[]",
        "points": "trajectory_msgs/JointTrajectoryPoint[]",
    },
    # Unitree hg/go/api: public IDL field names
    "unitree_hg/IMUState": {
        "quaternion": "float[4]",
        "gyroscope": "float[3]",
        "accelerometer": "float[3]",
        "rpy": "float[3]",
    },
    "unitree_hg/MotorState": {
        "mode": "uint8",
        "q": "float",
        "dq": "float",
        "tau_est": "float",
        "temperature": "int16[2]",
    },
    "unitree_hg/LowState": {
        "mode_pr": "uint8",
        "mode_machine": "uint8",
        "tick": "uint32",
        "imu_state": "unitree_hg/IMUState",
        "motor_state": "unitree_hg/MotorState[35]",
    },
    "unitree_hg/MotorCmd": {
        "mode": "uint8",
        "q": "float",
        "dq": "float",
        "tau": "float",
        "kp": "float",
        "kd": "float",
        "reserve": "uint32",
    },
    "unitree_hg/LowCmd": {
        "mode_pr": "uint8",
        "mode_machine": "uint8",
        "motor_cmd": "unitree_hg/MotorCmd[35]",
        "reserve": "uint32[4]",
        "crc": "uint32",
    },
    "unitree_go/SportModeState": {"position": "float[3]", "velocity": "float[3]"},
    "unitree_api/RequestIdentity": {"id": "int64", "api_id": "int64"},
    "unitree_api/RequestHeader": {"identity": "unitree_api/RequestIdentity"},
    "unitree_api/Request": {"header": "unitree_api/RequestHeader", "parameter": "string", "binary": "uint8[]"},
    "rcl_interfaces/ParameterValue": {
        "type": "uint8",
        "bool_value": "boolean",
        "integer_value": "int64",
        "double_value": "double",
        "string_value": "string",
        "string_array_value": "string[]",
    },
    "rcl_interfaces/Parameter": {"name": "string", "value": "rcl_interfaces/ParameterValue"},
    "rcl_interfaces/SetParametersResult": {"successful": "boolean", "reason": "string"},
    "rcl_interfaces/ListParametersResult": {"names": "string[]", "prefixes": "string[]"},
}
SRV_SPECS = {
    "rcl_interfaces/GetParameters": ({"names": "string[]"}, {"values": "rcl_interfaces/ParameterValue[]"}),
    "rcl_interfaces/SetParameters": (
        {"parameters": "rcl_interfaces/Parameter[]"},
        {"results": "rcl_interfaces/SetParametersResult[]"},
    ),
    "rcl_interfaces/ListParameters": (
        {"prefixes": "string[]", "depth": "uint64"},
        {"result": "rcl_interfaces/ListParametersResult"},
    ),
    "example_interfaces/AddTwoInts": ({"a": "int64", "b": "int64"}, {"sum": "int64"}),
}
ACTION_SPECS = {"example_interfaces/Fibonacci": ({"order": "int32"}, {"sequence": "int32[]"}, {"sequence": "int32[]"})}
_NUM = {"int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
_CLASSES: dict[str, type] = {}


def _default(t: str) -> Any:
    if t.endswith("]"):
        base, n = t[: t.index("[")], t[t.index("[") + 1 : -1]
        return [_default(base) for _ in range(int(n))] if n else []
    if "/" in t:
        return msg_class(t)()
    return {"string": "", "boolean": False}.get(t, 0 if t in _NUM else 0.0)


def msg_class(full: str, spec: dict[str, str] | None = None) -> type:
    pkg, name = full.split("/")[0], full.split("/")[-1]
    key = f"{pkg}/{name}"
    if key in _CLASSES:
        return _CLASSES[key]
    fields = spec if spec is not None else SPECS[key]

    def __init__(self: Any, **kw: Any) -> None:
        for f, t in fields.items():
            object.__setattr__(self, f, _default(t))
        for k, v in kw.items():
            setattr(self, k, v)

    def __setattr__(self: Any, k: str, v: Any) -> None:
        if k not in fields:
            raise AttributeError(f"{key} has no field {k!r}")
        object.__setattr__(self, k, v)

    cls = type(
        name,
        (),
        {
            "__init__": __init__,
            "__setattr__": __setattr__,
            "__slots_spec__": fields,
            "get_fields_and_field_types": classmethod(lambda c: dict(fields)),
            "__repr__": lambda s: f"{key}({ {f: getattr(s, f) for f in fields} })",
        },
    )
    cls.__module__ = f"{pkg}.msg._fake"
    _CLASSES[key] = cls
    return cls


def set_message_fields(msg: Any, values: dict[str, Any]) -> None:
    fields = type(msg).get_fields_and_field_types()
    for k, v in values.items():
        if k not in fields:
            raise AttributeError(f"{type(msg).__name__} has no field {k!r}")
        t = fields[k]
        base = t[: t.index("[")] if t.endswith("]") else t
        if "/" in base:
            if t.endswith("]"):
                cls = msg_class(base)
                items = []
                for item in v:
                    m = cls()
                    set_message_fields(m, item)
                    items.append(m)
                setattr(msg, k, items)
            else:
                sub = getattr(msg, k)
                set_message_fields(sub, v)
        else:
            setattr(msg, k, list(v) if t.endswith("]") and not isinstance(v, bytes | bytearray) else v)


class Reliability(enum.IntEnum):
    RELIABLE = 1
    BEST_EFFORT = 2


class Durability(enum.IntEnum):
    TRANSIENT_LOCAL = 1
    VOLATILE = 2


class History(enum.IntEnum):
    KEEP_LAST = 1
    KEEP_ALL = 2


class QoSProfile:
    def __init__(
        self,
        depth: int = 10,
        reliability: Any = Reliability.RELIABLE,
        durability: Any = Durability.VOLATILE,
        history: Any = History.KEEP_LAST,
    ) -> None:
        self.depth, self.reliability, self.durability, self.history = depth, reliability, durability, history


def _qos(q: Any) -> QoSProfile:
    return QoSProfile(depth=q) if isinstance(q, int) else q


class Graph:
    """The one in-process 'network'. reset() between tests."""

    def __init__(self) -> None:
        self.lock = threading.RLock()
        self.pubs: list[Publisher] = []
        self.subs: list[Subscription] = []
        self.services: dict[str, tuple[str, Any, Any]] = {}  # name -> (type, handler, node)
        self.actions: dict[str, tuple[str, Any]] = {}
        self.nodes: list[Node] = []
        self.published: list[tuple[str, Any]] = []

    def deliver(self, pub: Publisher, msg: Any) -> None:
        with self.lock:
            self.published.append((pub.topic, msg))
            if pub.qos.durability == Durability.TRANSIENT_LOCAL:
                pub.last = msg
            targets = [s for s in self.subs if _matches(pub, s)]
        for s in targets:
            s.node.queue.put((s.callback, msg))


def _matches(pub: Publisher, sub: Subscription) -> bool:
    if pub.topic != sub.topic or pub.cls is not sub.cls:
        return False
    return not (pub.qos.reliability == Reliability.BEST_EFFORT and sub.qos.reliability == Reliability.RELIABLE)


GRAPH = Graph()


class Publisher:
    def __init__(self, node: Node, cls: type, topic: str, qos: Any) -> None:
        self.node, self.cls, self.topic, self.qos, self.last = node, cls, topic, _qos(qos), None

    def publish(self, msg: Any) -> None:
        if type(msg) is not self.cls:
            raise TypeError(f"publisher for {self.cls.__name__} got {type(msg).__name__}")
        GRAPH.deliver(self, msg)


class Subscription:
    def __init__(self, node: Node, cls: type, topic: str, callback: Any, qos: Any) -> None:
        self.node, self.cls, self.topic, self.callback, self.qos = node, cls, topic, callback, _qos(qos)


class Future:
    def __init__(self) -> None:
        self._done, self._result, self._cbs = False, None, []

    def set_result(self, r: Any) -> None:
        self._result, self._done = r, True
        for cb in self._cbs:
            cb(self)

    def done(self) -> bool:
        return self._done

    def result(self) -> Any:
        return self._result

    def add_done_callback(self, cb: Any) -> None:
        if self._done:
            cb(self)
        else:
            self._cbs.append(cb)


class Client:
    def __init__(self, node: Node, cls: Any, name: str) -> None:
        self.node, self.cls, self.name = node, cls, name

    def wait_for_service(self, timeout_sec: float = 1.0) -> bool:
        return self.name in GRAPH.services

    def call_async(self, req: Any) -> Future:
        fut = Future()
        _t, handler, _n = GRAPH.services[self.name]
        self.node.queue.put((lambda r: fut.set_result(handler(r, self.cls.Response())), req))
        return fut


class PubInfo:
    def __init__(self, node_name: str, qos: QoSProfile) -> None:
        self.node_name, self.qos_profile = node_name, qos


class Node:
    def __init__(self, name: str, namespace: str = "/") -> None:
        self.name, self.namespace, self.queue = name, namespace, queue.Queue()
        with GRAPH.lock:
            GRAPH.nodes.append(self)

    def get_name(self) -> str:
        return self.name

    def create_publisher(self, cls: type, topic: str, qos: Any) -> Publisher:
        p = Publisher(self, cls, topic, qos)
        with GRAPH.lock:
            GRAPH.pubs.append(p)
        return p

    def create_subscription(self, cls: type, topic: str, callback: Any, qos: Any) -> Subscription:
        s = Subscription(self, cls, topic, callback, qos)
        with GRAPH.lock:
            GRAPH.subs.append(s)
            late = [
                p
                for p in GRAPH.pubs
                if p.last is not None and _matches(p, s) and s.qos.durability == Durability.TRANSIENT_LOCAL
            ]
        for p in late:
            self.queue.put((callback, p.last))
        return s

    def destroy_subscription(self, s: Subscription) -> None:
        with GRAPH.lock:
            if s in GRAPH.subs:
                GRAPH.subs.remove(s)

    def create_client(self, cls: Any, name: str) -> Client:
        return Client(self, cls, name)

    def destroy_client(self, c: Client) -> None:
        pass

    def create_service(self, cls: Any, name: str, handler: Any) -> None:
        GRAPH.services[name] = (cls._type, handler, self)

    def get_topic_names_and_types(self) -> list[tuple[str, list[str]]]:
        with GRAPH.lock:
            seen: dict[str, set[str]] = {}
            for e in [*GRAPH.pubs, *GRAPH.subs]:
                seen.setdefault(e.topic, set()).add(_tname(e.cls))
        return [(t, sorted(v)) for t, v in seen.items()]

    def get_service_names_and_types(self) -> list[tuple[str, list[str]]]:
        out = [(n, [t]) for n, (t, _h, _n) in GRAPH.services.items()]
        for n, (t, _h) in GRAPH.actions.items():
            out.append((f"{n}/_action/send_goal", [t + "_SendGoal"]))
        return out

    def get_node_names_and_namespaces(self) -> list[tuple[str, str]]:
        return [(n.name, n.namespace) for n in GRAPH.nodes]

    def get_publishers_info_by_topic(self, topic: str) -> list[PubInfo]:
        return [PubInfo(p.node.name, p.qos) for p in GRAPH.pubs if p.topic == topic]

    def get_subscriptions_info_by_topic(self, topic: str) -> list[PubInfo]:
        return [PubInfo(s.node.name, s.qos) for s in GRAPH.subs if s.topic == topic]

    def destroy_node(self) -> None:
        with GRAPH.lock:
            GRAPH.subs = [s for s in GRAPH.subs if s.node is not self]
            GRAPH.pubs = [p for p in GRAPH.pubs if p.node is not self]
            if self in GRAPH.nodes:
                GRAPH.nodes.remove(self)


def _tname(cls: type) -> str:
    return f"{cls.__module__.split('.')[0]}/msg/{cls.__name__}"


class Executor:
    def __init__(self, num_threads: int = 1) -> None:
        self.nodes: list[Node] = []

    def add_node(self, n: Node) -> None:
        self.nodes.append(n)

    def spin_once(self, timeout_sec: float = 0.1) -> None:
        end = time.monotonic() + timeout_sec
        while time.monotonic() < end:
            for n in self.nodes:
                try:
                    cb, msg = n.queue.get_nowait()
                except queue.Empty:
                    continue
                cb(msg)
                return
            time.sleep(0.001)


class ActionClient:
    def __init__(self, node: Node, cls: Any, name: str) -> None:
        self.node, self.cls, self.name = node, cls, name

    def wait_for_server(self, timeout_sec: float = 1.0) -> bool:
        return self.name in GRAPH.actions

    def send_goal_async(self, goal: Any, feedback_callback: Any = None) -> Future:
        _t, server = GRAPH.actions[self.name]
        fut = Future()
        fut.set_result(server(goal, feedback_callback, self.cls))
        return fut

    def destroy(self) -> None:
        pass


_STATE = {"ok": False}


def install() -> types.SimpleNamespace:
    """Put the fakes in sys.modules. Returns handles for tests."""
    rclpy = types.ModuleType("rclpy")
    rclpy.ok = lambda: _STATE["ok"]
    rclpy.init = lambda **kw: _STATE.update(ok=True)
    rclpy.try_shutdown = lambda: _STATE.update(ok=False)
    rclpy.shutdown = rclpy.try_shutdown
    rclpy.create_node = lambda name, **kw: Node(name)
    ex = types.ModuleType("rclpy.executors")
    ex.MultiThreadedExecutor = ex.SingleThreadedExecutor = Executor
    qosm = types.ModuleType("rclpy.qos")
    qosm.QoSProfile, qosm.ReliabilityPolicy, qosm.DurabilityPolicy, qosm.HistoryPolicy = (
        QoSProfile,
        Reliability,
        Durability,
        History,
    )
    act = types.ModuleType("rclpy.action")
    act.ActionClient = ActionClient
    nodem = types.ModuleType("rclpy.node")
    nodem.Node = Node
    rclpy.executors, rclpy.qos, rclpy.action, rclpy.node = ex, qosm, act, nodem
    rr = types.ModuleType("rosidl_runtime_py")
    util = types.ModuleType("rosidl_runtime_py.utilities")
    util.get_message = lambda t: msg_class(t)
    util.get_service = _srv
    util.get_action = _action
    sm = types.ModuleType("rosidl_runtime_py.set_message")
    sm.set_message_fields = set_message_fields
    rr.utilities, rr.set_message = util, sm
    sys.modules.update(
        {
            "rclpy": rclpy,
            "rclpy.executors": ex,
            "rclpy.qos": qosm,
            "rclpy.action": act,
            "rclpy.node": nodem,
            "rosidl_runtime_py": rr,
            "rosidl_runtime_py.utilities": util,
            "rosidl_runtime_py.set_message": sm,
        }
    )
    return types.SimpleNamespace(graph=GRAPH, msg=msg_class, node=Node)


def reset() -> None:
    global GRAPH
    GRAPH.__init__()
    _STATE["ok"] = False


def _srv(t: str) -> Any:
    pkg, _, name = t.split("/")
    req, res = SRV_SPECS[f"{pkg}/{name}"]
    cls = types.SimpleNamespace(
        Request=msg_class(f"{pkg}/{name}_Request", req), Response=msg_class(f"{pkg}/{name}_Response", res), _type=t
    )
    return cls


def _action(t: str) -> Any:
    pkg, _, name = t.split("/")
    goal, result, fb = ACTION_SPECS[f"{pkg}/{name}"]
    return types.SimpleNamespace(
        Goal=msg_class(f"{pkg}/{name}_Goal", goal),
        Result=msg_class(f"{pkg}/{name}_Result", result),
        Feedback=msg_class(f"{pkg}/{name}_Feedback", fb),
        _type=t,
    )
