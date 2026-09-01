"""Curated ROS 2 message IDL bundle for the pure-RTPS backend.

To publish a message over DDS you must own its type definition locally. These
``cyclonedds.idl`` dataclasses mirror the standard ROS 2 messages an agent most
often needs to act as a mobile base or arm. Each is registered under its ROS 2
type string (``geometry_msgs/msg/Twist``) so the ``use_rtps`` tool can resolve a
type by name with no rclpy and no sourced ROS 2 distro.

Why a static bundle and not dynamic types: cyclonedds-python's XTypes dynamic
type support is not yet complete enough to synthesise a publishable type from
remote discovery alone, so the publish path needs a local definition. This
bundle is the RTPS-backend equivalent of the ``[ros2]`` extra's documented-and-
minimal scope: it covers the common messages, not every custom interface.

Adding a message: define a ``@dataclass`` subclassing ``idl.IdlStruct`` with the
ROS field layout, give it the mangled DDS typename via ``@annotate.typename``,
and register it in ``REGISTRY`` under its ROS 2 type string. The wire layout
(field order + types) must match the upstream ``.msg`` exactly or real ROS 2
nodes will reject the sample.
"""

from typing import TYPE_CHECKING, Any

# cyclonedds is an optional dep (the [ros2] extra). Importing this module
# without it installed raises a clear, actionable error rather than a bare
# ModuleNotFoundError, matching the require_optional() convention used elsewhere.
try:
    from dataclasses import dataclass, field

    from cyclonedds.idl import IdlStruct
    from cyclonedds.idl.annotations import keylist  # noqa: F401  (re-exported for msg defs)
    from cyclonedds.idl.types import float64, int32, sequence, uint8, uint32  # noqa: F401

    _HAVE_CYCLONEDDS = True
except ImportError:  # pragma: no cover - exercised only without the extra
    _HAVE_CYCLONEDDS = False

    if not TYPE_CHECKING:

        def dataclass(cls=None, **kwargs):  # type: ignore[no-redef]
            raise ImportError(_INSTALL_HINT, name="cyclonedds")

        class IdlStruct:  # type: ignore[no-redef]
            pass


_INSTALL_HINT = (
    "cyclonedds is required for the pure-RTPS ROS 2 backend. Install the extra:\n"
    "  pip install 'strands-robots[ros2]'\n"
    "Unlike use_ros this needs NO sourced ROS 2 distro - cyclonedds is a "
    "self-contained pip wheel."
)


def have_cyclonedds() -> bool:
    """Return True if cyclonedds is importable (the RTPS backend is usable)."""
    return _HAVE_CYCLONEDDS


# --- Message definitions ---------------------------------------------------
# Defined only when cyclonedds is present; otherwise REGISTRY is empty and the
# tool surfaces _INSTALL_HINT. Field layouts mirror the upstream .msg files.

REGISTRY: dict[str, Any] = {}

if _HAVE_CYCLONEDDS:

    @dataclass
    class Vector3(IdlStruct, typename="geometry_msgs::msg::dds_::Vector3_"):
        """``geometry_msgs/msg/Vector3``: a free 3-D vector (x, y, z)."""

        x: float64 = 0.0
        y: float64 = 0.0
        z: float64 = 0.0

    @dataclass
    class Twist(IdlStruct, typename="geometry_msgs::msg::dds_::Twist_"):
        """``geometry_msgs/msg/Twist``: linear + angular velocity, the standard mobile-base command."""

        linear: Vector3 = field(default_factory=Vector3)
        angular: Vector3 = field(default_factory=Vector3)

    @dataclass
    class Point(IdlStruct, typename="geometry_msgs::msg::dds_::Point_"):
        """``geometry_msgs/msg/Point``: a 3-D position (x, y, z) in metres."""

        x: float64 = 0.0
        y: float64 = 0.0
        z: float64 = 0.0

    @dataclass
    class Quaternion(IdlStruct, typename="geometry_msgs::msg::dds_::Quaternion_"):
        """``geometry_msgs/msg/Quaternion``: an orientation (x, y, z, w); identity is ``w=1``."""

        x: float64 = 0.0
        y: float64 = 0.0
        z: float64 = 0.0
        w: float64 = 1.0

    @dataclass
    class Pose(IdlStruct, typename="geometry_msgs::msg::dds_::Pose_"):
        """``geometry_msgs/msg/Pose``: a :class:`Point` position plus a :class:`Quaternion` orientation."""

        position: Point = field(default_factory=Point)
        orientation: Quaternion = field(default_factory=Quaternion)

    REGISTRY.update(
        {
            "geometry_msgs/msg/Vector3": Vector3,
            "geometry_msgs/msg/Twist": Twist,
            "geometry_msgs/msg/Point": Point,
            "geometry_msgs/msg/Quaternion": Quaternion,
            "geometry_msgs/msg/Pose": Pose,
        }
    )

    # --- std_msgs / builtin_interfaces (Header chain) ----------------------
    # sensor_msgs messages are stamped, so the bundle needs the Header chain.
    # Layouts mirror the upstream .msg exactly:
    #   builtin_interfaces/msg/Time:  int32 sec; uint32 nanosec
    #   std_msgs/msg/Header:          builtin_interfaces/Time stamp; string frame_id

    @dataclass
    class Time(IdlStruct, typename="builtin_interfaces::msg::dds_::Time_"):
        """``builtin_interfaces/msg/Time``: a timestamp split into whole ``sec`` and ``nanosec`` remainder."""

        sec: int32 = 0
        nanosec: uint32 = 0

    @dataclass
    class Header(IdlStruct, typename="std_msgs::msg::dds_::Header_"):
        """``std_msgs/msg/Header``: a frame-tagged stamp (:class:`Time` ``stamp`` + ``frame_id``) prepended to stamped messages."""

        stamp: Time = field(default_factory=Time)
        frame_id: str = ""

    # --- sensor_msgs -------------------------------------------------------
    # sensor_msgs/msg/JointState:
    #   std_msgs/Header header
    #   string[] name
    #   float64[] position
    #   float64[] velocity
    #   float64[] effort

    @dataclass
    class JointState(IdlStruct, typename="sensor_msgs::msg::dds_::JointState_"):
        """``sensor_msgs/msg/JointState``: per-joint ``name``/``position``/``velocity``/``effort`` arrays with a :class:`Header`."""

        header: Header = field(default_factory=Header)
        name: sequence[str] = field(default_factory=list)  # type: ignore[type-arg, assignment]
        position: sequence[float64] = field(default_factory=list)  # type: ignore[type-arg, assignment]
        velocity: sequence[float64] = field(default_factory=list)  # type: ignore[type-arg, assignment]
        effort: sequence[float64] = field(default_factory=list)  # type: ignore[type-arg, assignment]

    # sensor_msgs/msg/Image:
    #   std_msgs/Header header
    #   uint32 height
    #   uint32 width
    #   string encoding
    #   uint8 is_bigendian
    #   uint32 step
    #   uint8[] data

    @dataclass
    class Image(IdlStruct, typename="sensor_msgs::msg::dds_::Image_"):
        """``sensor_msgs/msg/Image``: a raw camera frame (``encoding``, dimensions, row ``step``, ``data`` bytes) with a :class:`Header`."""

        header: Header = field(default_factory=Header)
        height: uint32 = 0
        width: uint32 = 0
        encoding: str = ""
        is_bigendian: uint8 = 0
        step: uint32 = 0
        data: sequence[uint8] = field(default_factory=list)  # type: ignore[type-arg, assignment]

    REGISTRY.update(
        {
            "builtin_interfaces/msg/Time": Time,
            "std_msgs/msg/Header": Header,
            "sensor_msgs/msg/JointState": JointState,
            "sensor_msgs/msg/Image": Image,
        }
    )


def get_type(ros_type: str) -> Any:
    """Resolve a ROS 2 type string to its IDL dataclass.

    Args:
        ros_type: e.g. ``geometry_msgs/msg/Twist``.

    Raises:
        ImportError: If cyclonedds is not installed.
        KeyError: If the type is not in the bundle (custom messages are out of
            scope for v1 - see the module docstring).
    """
    if not _HAVE_CYCLONEDDS:
        raise ImportError(_INSTALL_HINT, name="cyclonedds")
    if ros_type not in REGISTRY:
        known = ", ".join(sorted(REGISTRY)) or "(none)"
        raise KeyError(
            f"{ros_type!r} is not in the RTPS IDL bundle. Known types: {known}. "
            "Custom messages require rclpy (use the use_ros tool) until dynamic "
            "DDS types are supported."
        )
    return REGISTRY[ros_type]
