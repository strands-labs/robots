"""Configuration + YAML loader for the ProtoMotions Generalist Tracking Policy.

The upstream ONNX artifact
(``cagataydev/protomotions-gtp-unitree-g1/unified_pipeline.onnx``) ships with a
YAML sidecar (``unified_pipeline.yaml``) that pins:

* The 29 joint names in the ONNX action order.
* The 33 body names + anchor index (``torso_link`` = 16) + root index
  (``pelvis`` = 0).
* Per-joint stiffness + damping.
* Timing: ``control_dt = 0.02s`` (50Hz), ``physics_dt = 0.001s`` (1kHz),
  ``decimation = 20``.
* The future-reference lookahead schedule ``[1, 2, 4, 8]`` control steps.

:class:`ProtoMotionsConfig` is the typed dataclass representation of that
sidecar. Loading it once at construction (rather than reading the YAML on every
tick) means the hot path never touches disk, and a dimension or joint-count
error surfaces at policy build time with a clean message.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

__all__ = [
    "ProtoMotionsConfig",
    "load_config_from_yaml",
    "GTP_G1_JOINT_NAMES",
    "GTP_G1_BODY_NAMES",
    "GTP_G1_ANCHOR_BODY_INDEX",
    "GTP_G1_ROOT_BODY_INDEX",
    "GTP_G1_DEFAULT_LOOKAHEAD_STEPS",
    "GTP_G1_CONTROL_DT",
]

# ---------------------------------------------------------------------------
# Canonical constants - pinned from unified_pipeline.yaml (2026-08-14 upload)
# ---------------------------------------------------------------------------

# The 29 joint names in the order the ONNX policy emits them, matching the
# yaml `joint_names` field (id001). Order matters - ONNX output index i drives
# GTP_G1_JOINT_NAMES[i]. Kept identical to :data:`strands_robots.policies.
# kimodo.KIMODO_G1_JOINTS` so a Kimodo qpos plugs straight into a ProtoMotions
# tracker without any per-joint reordering.
GTP_G1_JOINT_NAMES: tuple[str, ...] = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# The 33 body names in the model's body order (yaml `body_names` id002).
# Index 0 = pelvis (root), 16 = torso_link (anchor). ``rubber_hand`` bodies are
# placeholders on the ``g1_29dof_rev_1_0`` URDF that ships fingerless - a real
# manipulation URDF would substitute finger links here without changing the
# tracker's input contract.
GTP_G1_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "head",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "left_rubber_hand",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
    "right_rubber_hand",
)

GTP_G1_ANCHOR_BODY_INDEX = 16  # torso_link
GTP_G1_ROOT_BODY_INDEX = 0  # pelvis

# The four future-step offsets the ONNX consumes at each tick (yaml
# ``motion.future_step_indices``). Kept as a tuple so it hashes and cannot be
# mutated across policy instances.
GTP_G1_DEFAULT_LOOKAHEAD_STEPS: tuple[int, ...] = (1, 2, 4, 8)

GTP_G1_CONTROL_DT: float = 0.02  # seconds - 50Hz outer control loop

# Upstream per-joint SONIC PD gains from the yaml sidecar. Order matches
# :data:`GTP_G1_JOINT_NAMES` (id003 stiffness, id004 damping).
_G1_STIFFNESS: tuple[float, ...] = (
    40.17923847137318,
    99.09842777666113,
    40.17923847137318,
    99.09842777666113,
    28.50124619574858,
    28.50124619574858,
    40.17923847137318,
    99.09842777666113,
    40.17923847137318,
    99.09842777666113,
    28.50124619574858,
    28.50124619574858,
    40.17923847137318,
    28.50124619574858,
    28.50124619574858,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    14.25062309787429,
    16.77832748089279,
    16.77832748089279,
)
_G1_DAMPING: tuple[float, ...] = (
    2.5578897650279457,
    6.3088018534966395,
    2.5578897650279457,
    6.3088018534966395,
    1.814445686584846,
    1.814445686584846,
    2.5578897650279457,
    6.3088018534966395,
    2.5578897650279457,
    6.3088018534966395,
    1.814445686584846,
    1.814445686584846,
    2.5578897650279457,
    1.814445686584846,
    1.814445686584846,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    0.907222843292423,
    1.06814150219,
    1.06814150219,
)


@dataclass(frozen=True)
class ProtoMotionsConfig:
    """Frozen typed view of the ONNX ``unified_pipeline.yaml`` sidecar.

    All fields have upstream-verified defaults matching the shipped
    ``cagataydev/protomotions-gtp-unitree-g1`` weights. A caller that points at
    a different checkpoint should always pair it with a matching config.

    Attributes:
        joint_names: 29 joint names in ONNX action order.
        body_names: 33 body names in the model's rigid-body order.
        anchor_body_index: Row index of the anchor body inside ``body_names``.
        root_body_index: Row index of the root body inside ``body_names``.
        stiffness: Per-joint kp used by the deployed PD loop.
        damping: Per-joint kd used by the deployed PD loop.
        control_dt: Seconds per outer control tick.
        physics_dt: Seconds per inner physics substep.
        decimation: Physics substeps per control tick (``control_dt /
            physics_dt``).
        future_step_indices: Future-reference lookahead offsets in control
            steps.
        action_ema_alpha: Optional exponential-moving-average smoothing on the
            joint target output (``1.0`` = passthrough, upstream default).
    """

    joint_names: tuple[str, ...] = GTP_G1_JOINT_NAMES
    body_names: tuple[str, ...] = GTP_G1_BODY_NAMES
    anchor_body_index: int = GTP_G1_ANCHOR_BODY_INDEX
    root_body_index: int = GTP_G1_ROOT_BODY_INDEX
    stiffness: tuple[float, ...] = _G1_STIFFNESS
    damping: tuple[float, ...] = _G1_DAMPING
    control_dt: float = GTP_G1_CONTROL_DT
    physics_dt: float = 0.001
    decimation: int = 20
    future_step_indices: tuple[int, ...] = GTP_G1_DEFAULT_LOOKAHEAD_STEPS
    action_ema_alpha: float = 1.0
    # ONNX I/O names - kept as tuples so the ordering is stable and the ONNX
    # session's ``get_inputs()`` order can be validated against it at load.
    onnx_in_names: tuple[str, ...] = (
        "current_anchor_rot",
        "current_dof_pos",
        "current_dof_vel",
        "current_root_local_ang_vel",
        "historical_processed_actions",
        "mimic_future_anchor_rot",
        "mimic_future_dof_pos",
        "mimic_future_dof_vel",
    )
    onnx_out_names: tuple[str, ...] = (
        "actions",
        "joint_pos_targets",
        "stiffness_targets",
        "damping_targets",
    )
    metadata: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------------
    # Derived properties - computed on read, never stored, so a frozen
    # dataclass with a single source of truth for each field stays that way.
    # ------------------------------------------------------------------

    @property
    def num_dofs(self) -> int:
        """Length of :attr:`joint_names` (also the ONNX action width)."""
        return len(self.joint_names)

    @property
    def num_bodies(self) -> int:
        """Length of :attr:`body_names`."""
        return len(self.body_names)

    @property
    def num_future_steps(self) -> int:
        """Length of :attr:`future_step_indices`."""
        return len(self.future_step_indices)

    @property
    def anchor_body_name(self) -> str:
        """Name of the anchor body (``torso_link`` on the shipped G1 config).

        The tracker consumes this body's WORLD orientation, not the floating
        base's. Resolving the name here keeps
        :attr:`~strands_robots.policies.base.Policy.required_bodies` and the
        observation lookup reading one source of truth.

        Raises:
            IndexError: If :attr:`anchor_body_index` is out of range for
                :attr:`body_names`.
        """
        return self.body_names[self.anchor_body_index]

    @property
    def root_body_name(self) -> str:
        """Name of the root (floating-base) body - ``pelvis`` on the G1.

        Raises:
            IndexError: If :attr:`root_body_index` is out of range for
                :attr:`body_names`.
        """
        return self.body_names[self.root_body_index]

    @property
    def anchor_is_root(self) -> bool:
        """Whether the anchor body IS the floating base.

        Only when this holds is the observation's ``base_quat`` the anchor
        orientation. On the G1 it is ``False``: the torso differs from the
        pelvis by the three waist joints, so substituting ``base_quat`` would
        feed the tracker a silently wrong frame.
        """
        return self.anchor_body_index == self.root_body_index


def load_config_from_yaml(path: str | Path) -> ProtoMotionsConfig:
    """Parse a ``unified_pipeline.yaml`` sidecar into a typed config.

    The yaml is the artifact's source of truth. Fields absent from the yaml
    fall back to the dataclass defaults (which are themselves pinned to the
    shipped weights, so a missing block is not an error).

    Args:
        path: Path to the yaml file.

    Returns:
        A :class:`ProtoMotionsConfig` - validated for consistent dimensions.

    Raises:
        FileNotFoundError: If ``path`` does not exist.
        ImportError: If ``pyyaml`` is not installed.
        ValueError: If the yaml contains an inconsistent dimension (e.g.
            ``stiffness`` length != number of joints).
    """
    yaml = require_optional(
        "yaml",
        pip_install="pyyaml",
        extra="protomotions",
        purpose="reading the unified_pipeline.yaml checkpoint sidecar",
    )

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"ProtoMotions yaml not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)  # type: ignore[attr-defined]

    joint_names = tuple(data.get("joint_names", GTP_G1_JOINT_NAMES))
    body_names = tuple(data.get("body_names", GTP_G1_BODY_NAMES))

    robot = data.get("robot", {})
    anchor_idx = int(robot.get("anchor_body_index", GTP_G1_ANCHOR_BODY_INDEX))
    root_idx = int(robot.get("root_body_index", GTP_G1_ROOT_BODY_INDEX))

    control = data.get("control", {})
    stiffness = tuple(control.get("stiffness", data.get("default_joint_stiffness", _G1_STIFFNESS)))
    damping = tuple(control.get("damping", data.get("default_joint_damping", _G1_DAMPING)))
    ema = float(control.get("action_ema_alpha", 1.0))

    timing = data.get("timing", {})
    control_dt = float(timing.get("control_dt", GTP_G1_CONTROL_DT))
    physics_dt = float(timing.get("physics_dt", 0.001))
    decimation = int(timing.get("decimation", 20))

    motion = data.get("motion", {})
    future_steps = tuple(int(x) for x in motion.get("future_step_indices", GTP_G1_DEFAULT_LOOKAHEAD_STEPS))

    runtime = data.get("_runtime", {})
    onnx_in_names = tuple(
        runtime.get(
            "onnx_in_names",
            ProtoMotionsConfig.__dataclass_fields__["onnx_in_names"].default,
        )
    )
    onnx_out_names = tuple(
        runtime.get(
            "onnx_out_names",
            ProtoMotionsConfig.__dataclass_fields__["onnx_out_names"].default,
        )
    )

    if len(stiffness) != len(joint_names):
        raise ValueError(f"stiffness length ({len(stiffness)}) != joint count ({len(joint_names)}) in {path}.")
    if len(damping) != len(joint_names):
        raise ValueError(f"damping length ({len(damping)}) != joint count ({len(joint_names)}) in {path}.")

    cfg = ProtoMotionsConfig(
        joint_names=joint_names,
        body_names=body_names,
        anchor_body_index=anchor_idx,
        root_body_index=root_idx,
        stiffness=stiffness,
        damping=damping,
        control_dt=control_dt,
        physics_dt=physics_dt,
        decimation=decimation,
        future_step_indices=future_steps,
        action_ema_alpha=ema,
        onnx_in_names=onnx_in_names,
        onnx_out_names=onnx_out_names,
        metadata=data.get("metadata", {}),
    )
    logger.info(
        "ProtoMotionsConfig loaded from %s: %d joints, %d bodies, %d future steps @ %.0f Hz",
        path.name,
        cfg.num_dofs,
        cfg.num_bodies,
        cfg.num_future_steps,
        1.0 / cfg.control_dt,
    )
    return cfg
