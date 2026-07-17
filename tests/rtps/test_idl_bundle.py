"""Tests for the RTPS IDL bundle.

Skips the cyclonedds-dependent assertions when the [ros2] extra is not
installed, but always pins the no-backend error contract.
"""

from __future__ import annotations

import pytest

import strands_robots.rtps.idl as idl


def test_get_type_without_cyclonedds_raises_actionable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idl, "_HAVE_CYCLONEDDS", False)
    with pytest.raises(ImportError, match=r"strands-robots\[ros2\]"):
        idl.get_type("geometry_msgs/msg/Twist")


def test_have_cyclonedds_matches_registry_population() -> None:
    # When cyclonedds is present the registry is non-empty; otherwise empty.
    if idl.have_cyclonedds():
        assert idl.REGISTRY, "cyclonedds present but IDL REGISTRY is empty"
    else:
        assert idl.REGISTRY == {}


@pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")
def test_bundle_typenames_match_ros_dds_mapping() -> None:
    from strands_robots.rtps.mangling import dds_type_name

    for ros_type, cls in idl.REGISTRY.items():
        # The IDL dataclass typename must equal the ROS-on-DDS mangled name so a
        # real ROS 2 node accepts the sample.
        assert cls.__idl_typename__ == dds_type_name(ros_type), ros_type


@pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")
def test_unknown_type_lists_known() -> None:
    with pytest.raises(KeyError, match="not in the RTPS IDL bundle"):
        idl.get_type("custom_msgs/msg/Nope")


@pytest.mark.skipif(not idl.have_cyclonedds(), reason="requires the [ros2] extra (cyclonedds)")
def test_sensor_msgs_chain_present() -> None:
    """The hardware RTPS bridge needs the JointState/Image + Header chain.

    These were added so the hardware bridge can run rclpy-free; the wire
    layouts are validated against a live ROS 2 node separately. Here we only
    pin that the types resolve and carry the correct DDS typenames.
    """
    from strands_robots.rtps.mangling import dds_type_name

    for ros_type in (
        "builtin_interfaces/msg/Time",
        "std_msgs/msg/Header",
        "sensor_msgs/msg/JointState",
        "sensor_msgs/msg/Image",
    ):
        cls = idl.get_type(ros_type)
        assert cls.__idl_typename__ == dds_type_name(ros_type), ros_type


# --- get_type resolver contract, decoupled from the optional [ros2] extra ----
# The lines below exercise get_type's resolution + unknown-type error formatting
# WITHOUT requiring cyclonedds: the resolver reads the module-level REGISTRY and
# _HAVE_CYCLONEDDS flag, so monkeypatching a stand-in registry pins the
# user-facing contract on every CI matrix leg, not only the [ros2] one.


def test_get_type_resolves_registered_type(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    monkeypatch.setattr(idl, "_HAVE_CYCLONEDDS", True)
    monkeypatch.setattr(idl, "REGISTRY", {"pkg_msgs/msg/Foo": sentinel})
    assert idl.get_type("pkg_msgs/msg/Foo") is sentinel


def test_get_type_unknown_lists_known_types_sorted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(idl, "_HAVE_CYCLONEDDS", True)
    monkeypatch.setattr(
        idl,
        "REGISTRY",
        {"pkg_msgs/msg/Zed": object(), "pkg_msgs/msg/Alpha": object()},
    )
    with pytest.raises(KeyError) as excinfo:
        idl.get_type("pkg_msgs/msg/Missing")
    message = str(excinfo.value)
    # Known types are listed sorted so the hint is deterministic, and the
    # message points at use_ros for out-of-bundle (custom) messages.
    assert message.index("Alpha") < message.index("Zed")
    assert "use_ros" in message


def test_get_type_unknown_with_empty_registry_reports_none(monkeypatch: pytest.MonkeyPatch) -> None:
    # Degenerate but real: backend present, bundle empty -> the hint must say
    # "(none)" rather than dangle after "Known types: ".
    monkeypatch.setattr(idl, "_HAVE_CYCLONEDDS", True)
    monkeypatch.setattr(idl, "REGISTRY", {})
    with pytest.raises(KeyError, match=r"Known types: \(none\)"):
        idl.get_type("pkg_msgs/msg/Missing")
