"""Franka Control Interface native driver package.

:class:`~strands_robots.drivers.franka.driver.FrankaDriver` drives the Panda and
both Research 3 generations over FCI, so
``Robot("panda", mode="real", driver="strands", port="172.16.0.2")`` builds a
real arm instead of raising ``ValueError`` - lerobot registers no Franka robot
type, which left all three simulation-only.

Nothing imported from this package touches the vendor binding: ``panda_py`` is
resolved inside :meth:`~strands_robots.drivers.franka.driver.FrankaDriver.connect_eagerly`,
so the module imports on any machine and every test runs against a fake FCI.

The pure helpers are exported beside the driver because they are the reusable
half: :func:`~strands_robots.drivers.franka.driver.decode_robot_state` turns a
libfranka state into this package's joint vocabulary and
:func:`~strands_robots.drivers.franka.driver.action_to_targets` reads an action
dict, both without an arm - which is what a caller replaying a recorded episode
or grading a dataset needs. :func:`~strands_robots.drivers.franka.driver.joint_names_for`
answers which vocabulary a given Franka speaks, which is not the same for all
three: see :data:`~strands_robots.drivers.franka.driver.JOINT_PREFIXES`.
"""

from __future__ import annotations

from strands_robots.drivers.franka.driver import (
    DEFAULT_SPEED_FACTOR,
    DEFAULT_STREAM_RATE_HZ,
    DOF,
    FCI_RATE_HZ,
    GRIPPER_KEY,
    JOINT_PREFIXES,
    SUPPORTED_ROBOTS,
    FrankaDriver,
    action_keys_for,
    action_to_targets,
    decode_robot_state,
    downsample_stride,
    joint_names_for,
)

__all__ = [
    "DEFAULT_SPEED_FACTOR",
    "DEFAULT_STREAM_RATE_HZ",
    "DOF",
    "FCI_RATE_HZ",
    "GRIPPER_KEY",
    "JOINT_PREFIXES",
    "SUPPORTED_ROBOTS",
    "FrankaDriver",
    "action_keys_for",
    "action_to_targets",
    "decode_robot_state",
    "downsample_stride",
    "joint_names_for",
]
