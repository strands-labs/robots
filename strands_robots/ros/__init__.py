"""ROS 2 for strands-robots: a bus, profiles, a driver behind Robot(...), presence, and the use_ros tool.

rclpy comes from a sourced ROS 2 distro; nothing here imports it until a bus starts.
"""

from .bus import RosBus
from .connect import connect
from .discovery import Discovery
from .gate import CommandGate, CommandRefused
from .lowlevel import wire_order
from .names import RosUnavailable
from .presence import RosPresence
from .profile import Profile, available, load
from .tool import use_ros

__all__ = [
    "RosBus",
    "connect",
    "Discovery",
    "CommandGate",
    "CommandRefused",
    "RosPresence",
    "Profile",
    "available",
    "load",
    "use_ros",
    "RosUnavailable",
    "wire_order",
]
