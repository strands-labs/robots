"""Unit tests for ROS 2 <-> DDS name mangling (pure string transforms).

Beyond the happy-path transforms, these pin one property: every name this module
returns is a name it would itself accept, and a ROS 2 name it cannot map is
refused rather than mangled into something plausible.

Two edges settle that. ``dds_type_name`` maps *message* interfaces only - ROS 2
generates one DDS type per constituent message of a service or an action, so
there is no single type to return for one, and an invented
``pkg::srv::dds_::Name_`` is what the participant would then advertise in DDS
discovery for a struct ROS 2 never generates. ``ros_topic_name`` checks the name
it recovers against the same rule ``dds_topic_name`` applies, because a DDS graph
carries topics no ROS 2 node published, so stripping the prefix alone does not
yield a ROS 2 name.
"""

from __future__ import annotations

import pytest

from strands_robots.rtps.idl import REGISTRY, have_cyclonedds
from strands_robots.rtps.mangling import dds_topic_name, dds_type_name, ros_topic_name

# Names the module's topic rule refuses. Shared by both directions so the two
# cannot come to disagree about what a valid ROS 2 topic name is.
_MALFORMED_ROS_TOPICS = ("cmd_vel", "", "/", "/bad name", "/a/", "/x;y")


@pytest.mark.parametrize(
    ("ros", "dds"),
    [
        ("/turtle1/cmd_vel", "rt/turtle1/cmd_vel"),
        ("/cmd_vel", "rt/cmd_vel"),
        ("/a/b/c", "rt/a/b/c"),
    ],
)
def test_topic_roundtrip(ros: str, dds: str) -> None:
    assert dds_topic_name(ros) == dds
    assert ros_topic_name(dds) == ros


@pytest.mark.parametrize("bad", _MALFORMED_ROS_TOPICS)
def test_topic_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 topic"):
        dds_topic_name(bad)


@pytest.mark.parametrize("bad", [t for t in _MALFORMED_ROS_TOPICS if t.startswith("/")])
def test_ros_topic_name_never_hands_back_a_name_dds_topic_name_refuses(bad: str) -> None:
    """The documented inverse has to survive a DDS topic that is not a ROS 2 one.

    A subscriber enumerating a DDS graph feeds whatever it discovered here. If
    stripping the prefix is the whole transform, a caller gets a "ROS 2 topic
    name" this module refuses, and the failure surfaces at whatever it hands the
    name to next rather than at the name it came from.
    """
    dds_topic = "rt" + bad
    try:
        recovered = ros_topic_name(dds_topic)
    except ValueError:
        return
    try:
        dds_topic_name(recovered)
    except ValueError as exc:
        pytest.fail(
            f"ros_topic_name({dds_topic!r}) handed back {recovered!r}, a name dds_topic_name itself "
            f"refuses ({exc}), so the documented inverse does not hold"
        )
    pytest.fail(f"premise: dds_topic_name({recovered!r}) was expected to be refused")


@pytest.mark.parametrize(
    ("ros", "dds"),
    [("/turtle1/cmd_vel", "rt/turtle1/cmd_vel"), ("/j/state", "rt/j/state")],
)
def test_a_valid_topic_survives_the_recovered_name_check(ros: str, dds: str) -> None:
    """Checking the recovered name must not narrow what a valid graph can carry."""
    assert ros_topic_name(dds) == ros
    assert dds_topic_name(ros_topic_name(dds)) == dds


def test_ros_topic_name_requires_prefix() -> None:
    with pytest.raises(ValueError, match="does not carry"):
        ros_topic_name("/turtle1/cmd_vel")  # missing the rt prefix


@pytest.mark.parametrize(
    ("ros", "dds"),
    [
        ("geometry_msgs/msg/Twist", "geometry_msgs::msg::dds_::Twist_"),
        ("sensor_msgs/msg/LaserScan", "sensor_msgs::msg::dds_::LaserScan_"),
        ("turtlesim/msg/Pose", "turtlesim::msg::dds_::Pose_"),
    ],
)
def test_type_mangling(ros: str, dds: str) -> None:
    assert dds_type_name(ros) == dds


@pytest.mark.parametrize("bad", ["Twist", "geometry_msgs/Twist", "a/b/c/d", "pkg/badkind/Name"])
def test_type_rejects_malformed(bad: str) -> None:
    with pytest.raises(ValueError, match="invalid ROS 2 type"):
        dds_type_name(bad)


@pytest.mark.parametrize(
    ("ros_type", "generated"),
    [
        (
            "example_interfaces/srv/AddTwoInts",
            ("example_interfaces::srv::dds_::AddTwoInts_Request_", "AddTwoInts_Response_"),
        ),
        ("std_srvs/srv/Trigger", ("std_srvs::srv::dds_::Trigger_Request_", "Trigger_Response_")),
        (
            "control_msgs/action/FollowJointTrajectory",
            ("control_msgs::action::dds_::FollowJointTrajectory_Goal_", "_Result_"),
        ),
    ],
)
def test_a_service_interface_is_refused_rather_than_given_a_type_ros2_never_generates(
    ros_type: str, generated: tuple[str, ...]
) -> None:
    """A service has no single DDS type, so there is nothing to return for one.

    ``rosidl`` renders its message template once per constituent message, so a
    service yields ``Name_Request_`` and ``Name_Response_`` structs (an action
    yields goal/result/feedback plus two nested services) and never a ``Name_``.
    Returning ``pkg::srv::dds_::Name_`` is what the participant then advertises in
    DDS discovery, for a struct that exists nowhere in the ROS 2 type system, and
    nothing reports it: matching is by topic name, so the wrong name does not
    even surface as a failure to connect. The refusal has to name the types ROS 2
    does generate, or it just moves the dead end one call further out.
    """
    try:
        minted = dds_type_name(ros_type)
    except ValueError as exc:
        message = str(exc)
        for expected in generated:
            assert expected in message, f"the refusal {message!r} does not name {expected!r}"
        return
    pytest.fail(
        f"dds_type_name({ros_type!r}) returned {minted!r}, but ROS 2 generates no such struct: the "
        f"wire types are {' / '.join(generated)}. That name is what the participant would advertise "
        "in DDS discovery, and nothing reports the disagreement."
    )


def test_the_malformed_type_refusal_does_not_offer_an_interface_kind_it_will_not_map() -> None:
    """A refusal must not send the caller after a spelling that is also refused."""
    with pytest.raises(ValueError) as excinfo:
        dds_type_name("Twist")
    message = str(excinfo.value)
    assert "pkg/msg/Name" in message
    for unmappable in ("pkg/srv/Name", "pkg/action/Name"):
        assert unmappable not in message, f"the refusal {message!r} offers {unmappable!r}, which it also refuses"


def test_an_unknown_interface_kind_is_still_reported_as_malformed() -> None:
    """Only ``srv``/``action`` get the mapping explanation; anything else is a typo."""
    with pytest.raises(ValueError, match="invalid ROS 2 type"):
        dds_type_name("pkg/badkind/Name")


@pytest.mark.skipif(not have_cyclonedds(), reason="cyclonedds not installed ([ros2] extra)")
def test_every_bundled_message_type_mangles_to_the_name_cyclonedds_puts_on_the_wire() -> None:
    """The message path is graded against the binding that carries it, not a copy.

    ``cyclonedds`` derives a topic's wire type name from the IDL dataclass, so
    the bundle's ``typename=`` annotations are what a real ROS 2 node matches
    against. Mangling has to agree with them for every bundled type.
    """
    assert len(REGISTRY) >= 9, f"premise: the IDL bundle looks empty ({sorted(REGISTRY)})"
    for ros_type, idl_cls in sorted(REGISTRY.items()):
        on_the_wire = idl_cls.__idl_typename__
        assert dds_type_name(ros_type) == on_the_wire, ros_type
