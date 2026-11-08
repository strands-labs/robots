"""The URDF walk reads only the model's own joints: <ros2_control> and <transmission> name joints too."""

from __future__ import annotations

import pytest

from strands_robots.description.introspect.xml import introspect_xml

ARM = (
    '<robot name="arm"><link name="base"/><link name="l0"/><link name="l1"/>'
    '<joint name="shoulder" type="revolute"><parent link="base"/><child link="l0"/>'
    '<limit lower="-1" upper="1" effort="1" velocity="1"/></joint>'
    '<joint name="elbow" type="revolute"><parent link="l0"/><child link="l1"/>'
    '<limit lower="-2" upper="2" effort="1" velocity="1"/></joint>'
    "{extra}</robot>"
)


def test_ros2_control_and_transmission_joints_are_not_model_joints(tmp_path):
    extra = (
        '<ros2_control name="arm" type="system"><joint name="shoulder"/><joint name="elbow"/></ros2_control>'
        '<transmission name="t"><joint name="elbow"/></transmission><gazebo><joint name="elbow"/></gazebo>'
    )
    path = tmp_path / "arm.urdf"
    path.write_text(ARM.format(extra=extra))
    facts = introspect_xml(path, "urdf")
    assert facts["action_joints"] == ["shoulder", "elbow"]
    assert facts["limits"] == {"shoulder": (-1.0, 1.0), "elbow": (-2.0, 2.0)}


def test_a_joint_named_twice_is_refused(tmp_path):
    dup = '<joint name="elbow" type="revolute"><parent link="base"/><child link="l1"/></joint>'
    path = tmp_path / "arm.urdf"
    path.write_text(ARM.format(extra=dup))
    with pytest.raises(ValueError, match="names joint 'elbow' twice"):
        introspect_xml(path, "urdf")
