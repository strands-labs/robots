"""ROS 2 <-> DDS name mangling.

ROS 2 maps its graph names onto DDS topic and type names with a fixed,
documented scheme (see the ROS 2 design doc "Topic and Service name mapping to
DDS"). Getting these exactly right is what makes a bare DDS participant
interoperable with real ROS 2 nodes.

Topic names
-----------
A ROS 2 topic ``/turtle1/cmd_vel`` becomes the DDS topic ``rt/turtle1/cmd_vel``:

* the ROS namespace separator ``/`` is preserved,
* a domain prefix is prepended: ``rt`` for topics (services use ``rq``/``rr``,
  out of scope here),
* the leading ``/`` is dropped after the prefix join (``rt`` + ``/turtle1/...``).

Type names
----------
A ROS 2 type ``geometry_msgs/msg/Twist`` becomes the DDS type
``geometry_msgs::msg::dds_::Twist_``:

* ``/`` separators become ``::``,
* the final segment gains a trailing underscore (``Twist`` -> ``Twist_``),
* a ``dds_`` segment is inserted before the final segment.

Message interfaces only. ROS 2 has no single DDS type for a service or an
action: ``rosidl`` renders the message template once per constituent message, so
``pkg/srv/Name`` becomes ``pkg::srv::dds_::Name_Request_`` and
``pkg::srv::dds_::Name_Response_`` (an action adds goal/result/feedback and two
nested services), exchanged over the ``rq``/``rr`` prefixes. There is no name
this mangling could return for one, so it refuses them and says which types
ROS 2 does generate. Returning ``pkg::srv::dds_::Name_`` instead would make the
participant advertise, in DDS discovery, a type name ROS 2 generates no struct
for, and nothing reports that: matching is by topic name, so a reader created
with the wrong name can still receive, leaving a participant that misrepresents
itself to every discovery tool and to any stack that does enforce type
consistency.

Both directions are pure string transforms with no ROS or DDS import, so they
are trivially unit-testable with no middleware present.
"""

from __future__ import annotations

import re

# ROS 2 graph names: leading slash plus alnum/_ segments (tilde/braces are
# substitution syntax that must already be resolved before mangling).
_ROS_TOPIC_RE = re.compile(r"^/[A-Za-z0-9_/]*[A-Za-z0-9_]\Z")

# Message interfaces only - see the module docstring for why ``srv``/``action``
# cannot appear here.
_ROS_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/msg/[A-Za-z0-9_]+\Z")

# A well-formed service or action interface. Matched separately so it is refused
# with the mapping explained instead of being reported as a malformed name.
_ROS_SERVICE_TYPE_RE = re.compile(r"^[A-Za-z0-9_]+/(srv|action)/[A-Za-z0-9_]+\Z")

# DDS topic prefixes per the ROS 2 mapping. Only ``rt`` (topics) is used in v1.
_TOPIC_PREFIX = "rt"


def _require_ros_topic(ros_topic: str, *, source: str = "") -> None:
    """Refuse a name this module's topic mangling cannot map.

    Both directions apply this one rule, so a name :func:`dds_topic_name` would
    refuse is never a name :func:`ros_topic_name` hands back either.

    Args:
        ros_topic: Candidate absolute ROS 2 topic name.
        source: DDS topic the name was recovered from, quoted in the refusal so
            a caller of :func:`ros_topic_name` sees which name it came from.

    Raises:
        ValueError: If *ros_topic* is not a valid absolute ROS 2 topic name.
    """
    if _ROS_TOPIC_RE.match(ros_topic):
        return
    recovered = f" recovered from {source!r}" if source else ""
    raise ValueError(f"invalid ROS 2 topic {ros_topic!r}{recovered}: expected an absolute name like /turtle1/cmd_vel")


def dds_topic_name(ros_topic: str, *, prefix: str = _TOPIC_PREFIX) -> str:
    """Map a ROS 2 topic name to its DDS topic name.

    ``/turtle1/cmd_vel`` -> ``rt/turtle1/cmd_vel``

    Args:
        ros_topic: A fully-qualified ROS 2 topic (must start with ``/``).
        prefix: DDS domain prefix; ``rt`` for topics (the default).

    Raises:
        ValueError: If *ros_topic* is not a valid absolute ROS 2 topic name.
    """
    _require_ros_topic(ros_topic)
    return prefix + ros_topic


def ros_topic_name(dds_topic: str, *, prefix: str = _TOPIC_PREFIX) -> str:
    """Inverse of :func:`dds_topic_name`.

    ``rt/turtle1/cmd_vel`` -> ``/turtle1/cmd_vel``

    A DDS graph carries topics no ROS 2 node published, so stripping the prefix
    is not enough to have recovered a ROS 2 name: the result is checked against
    the same rule :func:`dds_topic_name` applies, which is what makes this its
    inverse rather than a transform that can hand back a name the module itself
    refuses.

    Args:
        dds_topic: A DDS topic name carrying a ROS 2 domain prefix.
        prefix: The DDS domain prefix to strip (default ``rt``).

    Raises:
        ValueError: If *dds_topic* does not begin with ``<prefix>/``, or if what
            remains after the prefix is not a valid absolute ROS 2 topic name.
    """
    head = prefix + "/"
    if not dds_topic.startswith(head):
        raise ValueError(f"DDS topic {dds_topic!r} does not carry the {prefix!r} ROS 2 prefix")
    ros_topic = dds_topic[len(prefix) :]
    _require_ros_topic(ros_topic, source=dds_topic)
    return ros_topic


def dds_type_name(ros_type: str) -> str:
    """Map a ROS 2 interface type to its DDS type name.

    ``geometry_msgs/msg/Twist`` -> ``geometry_msgs::msg::dds_::Twist_``

    Message interfaces only. A service or action is refused with the per-message
    types ROS 2 generates for it quoted, because there is no single DDS type this
    could return for one (see the module docstring for what returning one anyway
    would advertise on the graph).

    Args:
        ros_type: A ``pkg/msg/Name`` interface type.

    Raises:
        ValueError: If *ros_type* is not a valid ``pkg/msg/Name`` triple, or
            names a service or action interface.
    """
    if _ROS_SERVICE_TYPE_RE.match(ros_type):
        pkg, kind, name = ros_type.split("/")
        if kind == "srv":
            generated = f"{pkg}::srv::dds_::{name}_Request_ and {pkg}::srv::dds_::{name}_Response_"
        else:
            generated = (
                f"{pkg}::action::dds_::{name}_Goal_, ..._Result_ and ..._Feedback_, "
                "plus the types of its nested send_goal/get_result services"
            )
        raise ValueError(
            f"ROS 2 {kind} interface {ros_type!r} has no single DDS type: ROS 2 generates one type "
            f"per constituent message ({generated}), exchanged over the rq/rr topic prefixes rather "
            f"than rt. This package maps topics only, so pass a pkg/msg/Name type."
        )
    if not _ROS_TYPE_RE.match(ros_type):
        raise ValueError(f"invalid ROS 2 type {ros_type!r}: expected pkg/msg/Name")
    pkg, kind, name = ros_type.split("/")
    return f"{pkg}::{kind}::dds_::{name}_"
