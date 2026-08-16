"""ProtoMotions Generalist Tracking Policy - native provider.

The ONNX Generalist Tracking Policy (GTP) from NVIDIA GEAR's ProtoMotions
framework, adapted as a first-class :class:`~strands_robots.policies.base.Policy`
provider. Consumes a per-tick observation (root/anchor rotation + joint pos/vel)
plus a future-reference window from a :class:`MotionPlayer`, and emits PD joint
targets for the Unitree G1's 29 leg+waist+arm actuators.

Two entry points:

* Direct construction with a ``.pt`` / ``.npz`` motion cache:
  ``ProtoMotionsPolicy(onnx_path=..., yaml_path=..., motion_path=...)``.
* Chained after :class:`~strands_robots.policies.kimodo.KimodoPolicy` - Kimodo
  samples a qpos trajectory,
  :func:`~strands_robots.policies.protomotions.bridge.qpos_to_motion_data`
  converts it to a MotionPlayer cache, this policy tracks it under physics.
  See :issue:`279` for why this is the correct pairing for a whole-body
  kinematic generator (WBC would overwrite Kimodo's leg+waist reference).

The pretrained artifact lives on HuggingFace at
``cagataydev/protomotions-gtp-unitree-g1`` (``unified_pipeline.onnx`` +
``unified_pipeline.yaml``). BeyondMimic-trained (arXiv:2408.07295) on
NVIDIA GEAR's ProtoMotions.
"""

from __future__ import annotations

from strands_robots.policies.protomotions.bridge import qpos_to_motion_data
from strands_robots.policies.protomotions.config import (
    GTP_G1_ANCHOR_BODY_INDEX,
    GTP_G1_BODY_NAMES,
    GTP_G1_CONTROL_DT,
    GTP_G1_DEFAULT_LOOKAHEAD_STEPS,
    GTP_G1_JOINT_NAMES,
    GTP_G1_ROOT_BODY_INDEX,
    ProtoMotionsConfig,
    load_config_from_yaml,
)
from strands_robots.policies.protomotions.motion_utils import MotionPlayer, lerp, slerp
from strands_robots.policies.protomotions.policy import (
    ProtoMotionsPolicy,
    ProtoMotionsSession,
)
from strands_robots.policies.protomotions.state_utils import (
    apply_heading_offset,
    compute_anchor_rot,
    compute_root_local_ang_vel,
    compute_yaw_offset,
    extract_yaw_quat,
    mujoco_wxyz_to_xyzw,
    quat_conjugate,
    quat_mul,
    quat_rotate_inverse,
)

__all__ = [
    "ProtoMotionsPolicy",
    "ProtoMotionsSession",
    "ProtoMotionsConfig",
    "MotionPlayer",
    "load_config_from_yaml",
    "qpos_to_motion_data",
    "GTP_G1_JOINT_NAMES",
    "GTP_G1_BODY_NAMES",
    "GTP_G1_ANCHOR_BODY_INDEX",
    "GTP_G1_ROOT_BODY_INDEX",
    "GTP_G1_DEFAULT_LOOKAHEAD_STEPS",
    "GTP_G1_CONTROL_DT",
    "lerp",
    "slerp",
    "mujoco_wxyz_to_xyzw",
    "quat_rotate_inverse",
    "quat_mul",
    "quat_conjugate",
    "compute_anchor_rot",
    "compute_root_local_ang_vel",
    "extract_yaw_quat",
    "compute_yaw_offset",
    "apply_heading_offset",
]
