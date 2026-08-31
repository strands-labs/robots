"""Pollen Microduck locomotion policies - native provider.

The Microduck is Pollen Robotics' open 14-DOF biped. Its skills ship as a family
of ONNX policies (``alpha_walking``, ``alpha_stand``, ``roulade``,
``ball_kick_*``, ``roller*``, ``alpha_ground_pick``), each an actor with the
input normaliser fused into the exported graph.

:class:`MicroduckPolicy` adapts one such export to the
:class:`~strands_robots.policies.base.Policy` interface: it self-configures from
the ONNX metadata (``joint_names`` / ``default_joint_pos`` / ``action_scale`` /
``command_names``), feeds the observation RAW (never re-normalising), and decodes
``motor_target = DEFAULT_POSE + action * action_scale``.
:class:`MicroduckPolicyBundle` holds several skills warm and hot-swaps between
them mid-rollout.
"""

from __future__ import annotations

from .composite import MicroduckPolicyBundle
from .observation import (
    GRAVITY_SOURCE_PROJECTED,
    GRAVITY_SOURCE_RAW_ACCEL,
    build_observation,
    decode_action,
    projected_gravity,
    quat_rotate_inverse,
    raw_accel_gravity,
)
from .policy import (
    MICRODUCK_DEFAULT_POSE,
    MICRODUCK_JOINT_NAMES,
    MicroduckPolicy,
)

__all__ = [
    "MicroduckPolicy",
    "MicroduckPolicyBundle",
    "MICRODUCK_JOINT_NAMES",
    "MICRODUCK_DEFAULT_POSE",
    "GRAVITY_SOURCE_PROJECTED",
    "GRAVITY_SOURCE_RAW_ACCEL",
    "build_observation",
    "decode_action",
    "projected_gravity",
    "raw_accel_gravity",
    "quat_rotate_inverse",
]
