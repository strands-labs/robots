"""How one message family maps to strands-robots: state -> JointState/Pose, targets -> a command message.

A codec is pure: it never talks to the bus. It gets the newest messages and returns data or a message,
so every family is unit-tested with plain objects. Built in: ``joint_state`` (any ros2_control robot) and
``unitree_hg`` (Unitree G1/H1-2 LowState/LowCmd). Any other robot brings its own codec from outside the
package: ``"codec": "my_pkg.codec:MyCodec"`` in its profile, or a ``strands_robots.ros_codecs`` entry point
named after the profile's ``family``. Subclass :class:`Codec` and override ``read`` and ``build``; arming,
id echo and the shadow comparison are data in the profile.
"""

from __future__ import annotations

import importlib
from importlib.metadata import entry_points
from typing import Any

import numpy as np

from ....ros.convert import from_dict, message_class
from ....ros.profile import Profile
from ....types import JointState, Pose


class Codec:
    """Base: joints keyed by wire id/name, never by wire order."""

    needs_gains = False  # True when the command message carries kp/kd per joint
    order: list[int] | None = None  # an unnamed array command: slot i carries joint ids[order[i]]

    def __init__(self, profile: Profile, names: tuple[str, ...]) -> None:
        self.profile, self.names = profile, names
        missing = []
        self.ids: list[Any] = []
        for n in names:
            try:
                self.ids.append(profile.wire_id(n))
            except KeyError:
                if profile.joints:
                    missing.append(n)
                self.ids.append(n)  # an empty joints map means: the wire uses the model's names
        if missing:
            raise ValueError(f"ROS profile {profile.name!r} has no wire id for {missing}; refusing a partial body")

    def set_array_order(self, joints: Any) -> None:
        """Slot order of an unnamed array command: the controller's own ``joints``, which must be exactly ours."""
        want, ids = [str(j) for j in joints or ()], [str(w) for w in self.ids]
        if len(set(want)) != len(want) or sorted(want) != sorted(ids):
            raise ValueError(f"the controller's joints {want} are not exactly this robot's {ids}; refusing to stream")
        self.order = [ids.index(j) for j in want]

    def read(self, state: Any) -> JointState:
        raise NotImplementedError

    def build(self, q: np.ndarray, kp: np.ndarray | None, kd: np.ndarray | None, state: Any, sender: str) -> Any:
        raise NotImplementedError

    def armed(self, status: Any, state: Any) -> tuple[bool, str]:
        """The profile's ``status.armed`` fields must all hold; no status stream means always armed."""
        st = self.profile.stream("status")
        if not st:
            return True, "no arming state on this robot"
        if status is None:
            return False, f"not armed: no {st['topic']} yet"
        labels = st.get("labels") or {}
        bad = []
        for f, want in (st.get("armed") or {}).items():
            have = getattr(status, f, None)
            if have != want:
                lab = labels.get(f) or {}
                bad.append(f"{f}={lab.get(str(have), have)} (needs {lab.get(str(want), want)})")
        return (not bad, "armed" if not bad else "not armed: " + ", ".join(bad))

    def echo(self, msg: Any, frame: Any) -> Any:
        """Copy ``command.echo`` fields from the sensor frame into the command (``{"dst": "src.path"}``)."""
        for dst, src in ((self.profile.stream("command") or {}).get("echo") or {}).items():
            v = _get(frame, src)
            if v is not None:
                _set(msg, dst, v)
        return msg

    def positions(self, command: Any) -> np.ndarray | None:
        """Joint positions in a command message, in ``names`` order (shadow mode's comparison)."""
        if hasattr(command, "data"):
            data = [float(x) for x in command.data]
            if self.order is None or len(data) != len(self.order):
                return np.array(data)
            out = np.full(len(self.order), np.nan)
            out[self.order] = data
            return out
        if hasattr(command, "points") and command.points:
            return np.array([float(x) for x in command.points[0].positions])
        if hasattr(command, "name") and hasattr(command, "position"):
            got = dict(zip(command.name, command.position, strict=False))
            return np.array([float(got.get(w, np.nan)) for w in self.ids])
        return None

    def imu(self, state: Any) -> tuple[np.ndarray, np.ndarray] | None:
        return None

    def pose(self, state: Any, odom: Any) -> Pose | None:
        return pose_from(self.imu(state) if state is not None else None, odom)

    def extras(self, state: Any, status: Any) -> dict[str, Any]:
        """The status fields arming depends on, by label when the profile gives one."""
        st = self.profile.stream("status") or {}
        labels = st.get("labels") or {}
        out: dict[str, Any] = {}
        for f in st.get("armed") or {}:
            if status is not None and hasattr(status, f):
                v = getattr(status, f)
                out[f] = (labels.get(f) or {}).get(str(v), v)
        return out

    def default_gains(self, info: Any) -> tuple[np.ndarray, np.ndarray] | None:
        """kp/kd to hold with: the profile's ``gains`` (subclasses may read ``info`` from ``gains_from``)."""
        if not (self.profile.data.get("gains") or {}).get("kp"):
            return None
        return np.array(self.profile.gains("kp", self.names)), np.array(self.profile.gains("kd", self.names))


def _get(obj: Any, path: str) -> Any:
    for part in path.split("."):
        obj = getattr(obj, part, None)
        if obj is None:
            return None
    return obj


def _set(obj: Any, path: str, value: Any) -> None:
    *head, last = path.split(".")
    for part in head:
        obj = getattr(obj, part)
    setattr(obj, last, value)


class JointStateCodec(Codec):
    """sensor_msgs/JointState in; Float64MultiArray, JointState or a one-point JointTrajectory out."""

    def read(self, state: Any) -> JointState:
        idx = {n: i for i, n in enumerate(state.name)}
        missing = [w for w in self.ids if w not in idx]
        if missing:
            raise ValueError(f"/joint_states has no {missing} (it has {list(state.name)})")

        def col(field: str) -> np.ndarray:
            arr = list(getattr(state, field) or ())
            return np.array([float(arr[idx[w]]) if idx[w] < len(arr) else 0.0 for w in self.ids])

        return JointState(self.names, col("position"), col("velocity"), col("effort"))

    def build(self, q: np.ndarray, kp: np.ndarray | None, kd: np.ndarray | None, state: Any, sender: str) -> Any:
        t = self.profile.need("command")["type"]
        if t.endswith("/Float64MultiArray"):  # no names on the wire: slots follow the controller's joints
            if self.order is None:
                raise ValueError("no controller joint order for a Float64MultiArray command (set_array_order)")
            return from_dict(message_class(t), {"data": [float(q[i]) for i in self.order]})
        if t.endswith("/JointState"):
            return from_dict(message_class(t), {"name": [str(w) for w in self.ids], "position": [float(x) for x in q]})
        if t.endswith("/JointTrajectory"):
            dt = float(self.profile.need("command").get("point_time_s", 0.05))
            return from_dict(
                message_class(t),
                {
                    "joint_names": [str(w) for w in self.ids],
                    "points": [
                        {
                            "positions": [float(x) for x in q],
                            "time_from_start": {"sec": int(dt), "nanosec": int(dt % 1 * 1e9)},
                        }
                    ],
                },
            )
        raise ValueError(f"joint_state family cannot write {t}")


def quat_xyzw_to_wxyz(o: Any) -> np.ndarray:
    return np.array([float(o.w), float(o.x), float(o.y), float(o.z)])


def pose_from(imu: tuple[np.ndarray, np.ndarray] | None, odom: Any) -> Pose | None:
    """Pose from an IMU (orientation, gyro) and, if fresh, a nav_msgs/Odometry (position, world velocity)."""
    if imu is None and odom is None:
        return None
    pos, lin = np.zeros(3), None
    quat, gyro = imu if imu is not None else (np.array([1.0, 0, 0, 0]), None)
    if odom is not None and hasattr(odom, "pose"):
        p = odom.pose.pose.position
        pos = np.array([p.x, p.y, p.z], dtype=float)
        if imu is None:
            quat = quat_xyzw_to_wxyz(odom.pose.pose.orientation)
        v = odom.twist.twist.linear  # nav_msgs twist is in child_frame (body); rotate to world
        lin = _rotate(quat, np.array([v.x, v.y, v.z], dtype=float))
    return Pose(pos, quat, lin, gyro)


def _rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    r = np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )
    return r @ v


def gains_vector(v: Any, n: int, what: str) -> np.ndarray | None:
    """A per-joint kp or kd: a scalar or ``n`` values, finite and >= 0 (None stays None)."""
    if v is None:
        return None
    arr = np.broadcast_to(np.asarray(v, dtype=float), (n,)).copy()
    if not np.isfinite(arr).all() or (arr < 0).any():
        raise ValueError(f"{what} must be finite and >= 0 for every joint")
    return arr


BUILTIN: dict[str, str] = {
    "joint_state": "strands_robots.robot.drivers.ros.codec:JointStateCodec",
    "unitree_hg": "strands_robots.robot.drivers.ros.unitree_hg:UnitreeHgCodec",
}
ENTRY_POINTS = "strands_robots.ros_codecs"


def codec_class(profile: Profile) -> type[Codec]:
    """The profile's ``codec`` dotted path, else its family: a ``strands_robots.ros_codecs`` entry point or built in."""
    spec = str(profile.data.get("codec") or "")
    eps = {} if spec else {ep.name: ep for ep in entry_points(group=ENTRY_POINTS)}
    if spec:
        mod, _, attr = spec.partition(":")
        cls = getattr(importlib.import_module(mod), attr)
    elif profile.family in eps:
        cls = eps[profile.family].load()
    elif profile.family in BUILTIN:
        mod, _, attr = BUILTIN[profile.family].partition(":")
        cls = getattr(importlib.import_module(mod), attr)
    else:
        raise ValueError(
            f"ROS profile {profile.name!r}: no codec for family {profile.family!r} (built in: {sorted(BUILTIN)}; "
            f'add "codec": "pkg.module:Class" to the profile or a {ENTRY_POINTS} entry point)'
        )
    if not (isinstance(cls, type) and issubclass(cls, Codec)):
        raise TypeError(f"ROS profile {profile.name!r}: codec {cls!r} is not a strands_robots Codec subclass")
    return cls
