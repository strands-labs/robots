#!/usr/bin/env python3
"""
GR00T Data Configuration — Complete coverage of all N1.5 + N1.6 embodiments.

Every config here is verified against Isaac-GR00T's native DATA_CONFIG_MAP
so that modality keys, state/action shapes, and language keys match exactly.
"""

import logging
from abc import ABC
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for a modality (cameras, state, actions, language)."""

    delta_indices: List[int]
    modality_keys: List[str]

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"delta_indices": self.delta_indices, "modality_keys": self.modality_keys})


@dataclass
class BaseDataConfig(ABC):
    """Base class for GR00T data configurations."""

    video_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]
    language_keys: List[str]
    observation_indices: List[int]
    action_indices: List[int]

    def modality_config(self) -> Dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }


# ═══════════════════════════════════════════════════════════════════════
#  SO-100 / SO-101  (LeRobot-compatible low-cost arms)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class So100DataConfig(BaseDataConfig):
    """SO-100 single camera."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.webcam"]
        self.state_keys = self.state_keys or ["state.single_arm", "state.gripper"]
        self.action_keys = self.action_keys or ["action.single_arm", "action.gripper"]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class So100DualCamDataConfig(So100DataConfig):
    """SO-100 dual camera (front + wrist)."""

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.front", "video.wrist"]


@dataclass
class So100QuadCamDataConfig(So100DataConfig):
    """SO-100 quad camera."""

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.front", "video.wrist", "video.top", "video.side"]


@dataclass
class So101DataConfig(BaseDataConfig):
    """SO-101 single camera (upgraded SO-100, 6-DOF).

    Same joint layout as SO-100 but with improved actuators. Single webcam
    view suitable for basic policy evaluation and lightweight inference.
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.webcam"]
        self.state_keys = self.state_keys or ["state.single_arm", "state.gripper"]
        self.action_keys = self.action_keys or ["action.single_arm", "action.gripper"]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class So101DualCamDataConfig(So101DataConfig):
    """SO-101 dual camera (front + wrist)."""

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.front", "video.wrist"]


@dataclass
class So101TriCamDataConfig(So101DataConfig):
    """SO-101 tri-camera multiview for Cosmos Transfer 2.5.

    Three synchronized cameras providing the multiview input required
    by Cosmos Transfer 2.5 for sim-to-real visual augmentation:
      - video.front: Front workspace view (primary manipulation view)
      - video.wrist: Wrist-mounted camera (close-up gripper view)
      - video.side:  Side view (depth/geometry reference for transfer)

    This config is specifically designed for the Cosmos Transfer 2.5
    pipeline which requires 3 camera viewpoints for controllable
    generation (depth, edge, segmentation control signals).
    """

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.front", "video.wrist", "video.side"]


# ═══════════════════════════════════════════════════════════════════════
#  Fourier GR-1  (humanoid — arms, waist, full upper body)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class FourierGr1ArmsOnlyDataConfig(BaseDataConfig):
    """Fourier GR-1 arms only."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.ego_view"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class FourierGr1ArmsWaistDataConfig(FourierGr1ArmsOnlyDataConfig):
    """Fourier GR-1 arms + waist."""

    def __post_init__(self):
        super().__post_init__()
        self.state_keys = ["state.left_arm", "state.right_arm", "state.left_hand", "state.right_hand", "state.waist"]
        self.action_keys = [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
            "action.waist",
        ]
        self.language_keys = ["annotation.human.coarse_action"]


@dataclass
class FourierGr1FullUpperBodyDataConfig(FourierGr1ArmsOnlyDataConfig):
    """Fourier GR-1 full upper body (arms + hands + waist + neck)."""

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.front_view"]
        self.state_keys = [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
            "state.waist",
            "state.neck",
        ]
        self.action_keys = [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
            "action.waist",
            "action.neck",
        ]


# ═══════════════════════════════════════════════════════════════════════
#  Unitree G1  (humanoid — arms, full body with locomotion)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class UnitreeG1DataConfig(BaseDataConfig):
    """Unitree G1 arms only (teleop/manipulation tasks).

    NOTE: This is NOT the upstream Isaac-GR00T unitree_g1 config (which is full-body).
    Use UnitreeG1LocoManipDataConfig for the upstream-faithful 1:1 config.
    This is a convenience config for arms-only manipulation tasks.
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.ego_view"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class UnitreeG1FullBodyDataConfig(UnitreeG1DataConfig):
    """Unitree G1 full body (legs + waist + arms + hands).

    N1.6 inference returns actions for ALL body groups:
      left_leg(6) + right_leg(6) + waist(3) + left_arm(7) + right_arm(7)
      + left_hand(7) + right_hand(7) = 43 DOF per timestep × 16 action horizon.
    """

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.ego_view"]
        self.state_keys = [
            "state.left_leg",
            "state.right_leg",
            "state.waist",
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        # N1.6: actions include ALL body groups (verified from Thor inference)
        self.action_keys = [
            "action.left_leg",
            "action.right_leg",
            "action.waist",
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]


@dataclass
class UnitreeG1LocoManipDataConfig(UnitreeG1DataConfig):
    """Unitree G1 locomanipulation — FAITHFUL 1:1 mirror of upstream Isaac-GR00T N1.6.

    Upstream embodiment_configs.py "unitree_g1" definition:
      video: ["ego_view"]
      state: ["left_leg","right_leg","waist","left_arm","right_arm","left_hand","right_hand"]
      action: ["left_arm","right_arm","left_hand","right_hand","waist","base_height_command","navigate_command"]
             delta_indices=list(range(30))
      language: ["annotation.human.task_description"]

    Note: Our internal convention prefixes with "video.", "state.", "action."
    The _local_inference method in groot/__init__.py strips these when building
    the nested dict for N1.6's Gr00tPolicy.get_action().
    """

    def __post_init__(self):
        super().__post_init__()
        self.video_keys = ["video.ego_view"]
        self.state_keys = [
            "state.left_leg",
            "state.right_leg",
            "state.waist",
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
            "action.waist",
            "action.base_height_command",
            "action.navigate_command",
        ]
        self.language_keys = ["annotation.human.task_description"]
        self.action_indices = list(range(30))


# ═══════════════════════════════════════════════════════════════════════
#  Franka Panda  (bimanual gripper, bimanual hand, single arm)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class BimanualPandaGripperDataConfig(BaseDataConfig):
    """Bimanual Panda with grippers."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.right_wrist_view", "video.left_wrist_view", "video.front_view"]
        self.state_keys = self.state_keys or [
            "state.right_arm_eef_pos",
            "state.right_arm_eef_quat",
            "state.right_gripper_qpos",
            "state.left_arm_eef_pos",
            "state.left_arm_eef_quat",
            "state.left_gripper_qpos",
        ]
        self.action_keys = self.action_keys or [
            "action.right_arm_eef_pos",
            "action.right_arm_eef_rot",
            "action.right_gripper_close",
            "action.left_arm_eef_pos",
            "action.left_arm_eef_rot",
            "action.left_gripper_close",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class BimanualPandaHandDataConfig(BaseDataConfig):
    """Bimanual Panda with dexterous hands."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.right_wrist_view", "video.left_wrist_view", "video.ego_view"]
        self.state_keys = self.state_keys or [
            "state.right_arm_eef_pos",
            "state.right_arm_eef_quat",
            "state.right_hand",
            "state.left_arm_eef_pos",
            "state.left_arm_eef_quat",
            "state.left_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.right_arm_eef_pos",
            "action.right_arm_eef_rot",
            "action.right_hand",
            "action.left_arm_eef_pos",
            "action.left_arm_eef_rot",
            "action.left_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class SinglePandaGripperDataConfig(BaseDataConfig):
    """Single Panda arm with gripper (RoboCasa / mobile manipulation)."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.left_view", "video.right_view", "video.wrist_view"]
        self.state_keys = self.state_keys or [
            "state.end_effector_position_relative",
            "state.end_effector_rotation_relative",
            "state.gripper_qpos",
            "state.base_position",
            "state.base_rotation",
        ]
        self.action_keys = self.action_keys or [
            "action.end_effector_position",
            "action.end_effector_rotation",
            "action.gripper_close",
            "action.base_motion",
            "action.control_mode",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


# ═══════════════════════════════════════════════════════════════════════
#  OXE  (Open X-Embodiment — DROID, Google RT, WidowX)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class OxeDroidDataConfig(BaseDataConfig):
    """OXE DROID — FAITHFUL 1:1 mirror of upstream Isaac-GR00T N1.6.

    Upstream: video=["exterior_image_1_left","wrist_image_left"],
              state=["joint_position","gripper_position"],
              action=["joint_position","gripper_position"], delta_indices=range(32)
              language=["annotation.language.language_instruction"]
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.exterior_image_1_left", "video.wrist_image_left"]
        self.state_keys = self.state_keys or ["state.joint_position", "state.gripper_position"]
        self.action_keys = self.action_keys or ["action.joint_position", "action.gripper_position"]
        self.language_keys = self.language_keys or ["annotation.language.language_instruction"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(32))


@dataclass
class OxeGoogleDataConfig(BaseDataConfig):
    """OXE Google RT — FAITHFUL 1:1 mirror of upstream Isaac-GR00T N1.6.

    Upstream: video=["image"], state=["x","y","z","rx","ry","rz","rw","gripper"],
              action=["x","y","z","roll","pitch","yaw","gripper"], delta_indices=range(8)
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.image"]
        self.state_keys = self.state_keys or [
            "state.x",
            "state.y",
            "state.z",
            "state.rx",
            "state.ry",
            "state.rz",
            "state.rw",
            "state.gripper",
        ]
        self.action_keys = self.action_keys or [
            "action.x",
            "action.y",
            "action.z",
            "action.roll",
            "action.pitch",
            "action.yaw",
            "action.gripper",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(8))


@dataclass
class OxeWidowXDataConfig(BaseDataConfig):
    """OXE WidowX — FAITHFUL 1:1 mirror of upstream Isaac-GR00T N1.6.

    Upstream: video=["image_0"], state=["x","y","z","roll","pitch","yaw","pad","gripper"],
              action=["x","y","z","roll","pitch","yaw","gripper"], delta_indices=range(8)
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.image_0"]
        self.state_keys = self.state_keys or [
            "state.x",
            "state.y",
            "state.z",
            "state.roll",
            "state.pitch",
            "state.yaw",
            "state.pad",
            "state.gripper",
        ]
        self.action_keys = self.action_keys or [
            "action.x",
            "action.y",
            "action.z",
            "action.roll",
            "action.pitch",
            "action.yaw",
            "action.gripper",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(8))


# ═══════════════════════════════════════════════════════════════════════
#  Simulation  (Libero Panda, RoboCasa)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class LiberoPandaDataConfig(BaseDataConfig):
    """Libero Panda simulation (N1.6)."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.image", "video.wrist_image"]
        self.state_keys = self.state_keys or [
            "state.x",
            "state.y",
            "state.z",
            "state.roll",
            "state.pitch",
            "state.yaw",
            "state.gripper",
        ]
        self.action_keys = self.action_keys or [
            "action.x",
            "action.y",
            "action.z",
            "action.roll",
            "action.pitch",
            "action.yaw",
            "action.gripper",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


# ═══════════════════════════════════════════════════════════════════════
#  Other Robots  (Agibot Genie1, Galaxea R1 Pro)
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class AgibotGenie1DataConfig(BaseDataConfig):
    """Agibot Genie1 (bimanual humanoid with mobile base)."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.top_head", "video.hand_left", "video.hand_right"]
        self.state_keys = self.state_keys or [
            "state.left_arm_joint_position",
            "state.right_arm_joint_position",
            "state.left_effector_position",
            "state.right_effector_position",
            "state.head_position",
            "state.waist_position",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm_joint_position",
            "action.right_arm_joint_position",
            "action.left_effector_position",
            "action.right_effector_position",
            "action.head_position",
            "action.waist_position",
            "action.robot_velocity",
        ]
        self.language_keys = self.language_keys or ["annotation.language.action_text"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


@dataclass
class AgibotDualArmGripperDataConfig(BaseDataConfig):
    """AgiBot World dual-arm with parallel grippers (14-DOF arms + 2-DOF grippers).

    Matches AgiBot World Beta HDF5 proprioceptive layout:
      /state/joint/position [N, 14]  → left_arm[0:7] + right_arm[7:14]
      /state/effector/position [N, 2]  → gripper open distance (mm)
      /action/joint/position [N, 14]  → commanded positions
      /action/effector/position [N, 2]  → commanded gripper

    For converting AgiBot World data to GR00T N1.6 training format via LeRobot.
    GO-1 model uses this 16-dim layout: 7 left arm + 1 left gripper + 7 right arm + 1 right gripper.
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.head", "video.left_hand", "video.right_hand"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_gripper",
            "state.right_gripper",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
        ]
        self.language_keys = self.language_keys or ["annotation.language.action_text"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(30))  # GO-1 uses 30-step chunks


@dataclass
class AgibotDualArmDexHandDataConfig(BaseDataConfig):
    """AgiBot World dual-arm with 6-DOF dexterous hands (14-DOF arms + 12-DOF hands).

    Matches AgiBot World Beta HDF5 layout for dexterous hand variants:
      /state/joint/position [N, 14]  → left_arm[0:7] + right_arm[7:14]
      /state/effector/position [N, 12]  → 6 joints per hand
      /action/joint/position [N, 14]  → commanded arm positions
      /action/effector/position [N, 12]  → commanded hand joint positions

    Total 26-DOF per timestep (14 arm + 12 hand).
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.head", "video.left_hand", "video.right_hand"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.language.action_text"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(30))


@dataclass
class AgibotDualArmFullDataConfig(BaseDataConfig):
    """AgiBot World dual-arm with full body (arms + grippers + head + waist + base).

    Complete AgiBot World Beta layout including mobile base navigation:
      /state/joint/position [N, 14]
      /state/effector/position [N, 2]
      /state/head/position [N, 2]  → yaw + pitch
      /state/waist/position [N, 2]  → pitch + lift
      /action/robot/velocity [N, 2]  → base linear + angular velocity

    Total 22-DOF per timestep (14 arm + 2 gripper + 2 head + 2 waist + 2 base).
    """

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.head", "video.left_hand", "video.right_hand"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_gripper",
            "state.right_gripper",
            "state.head",
            "state.waist",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_gripper",
            "action.right_gripper",
            "action.head",
            "action.waist",
            "action.base_velocity",
        ]
        self.language_keys = self.language_keys or ["annotation.language.action_text"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(30))


@dataclass
class GalaxeaR1ProDataConfig(BaseDataConfig):
    """Galaxea R1 Pro / BEHAVIOR suite bimanual (N1.6)."""

    video_keys: List[str] = None
    state_keys: List[str] = None
    action_keys: List[str] = None
    language_keys: List[str] = None
    observation_indices: List[int] = None
    action_indices: List[int] = None

    def __post_init__(self):
        self.video_keys = self.video_keys or ["video.head_camera_rgb"]
        self.state_keys = self.state_keys or [
            "state.left_arm",
            "state.right_arm",
            "state.left_hand",
            "state.right_hand",
        ]
        self.action_keys = self.action_keys or [
            "action.left_arm",
            "action.right_arm",
            "action.left_hand",
            "action.right_hand",
        ]
        self.language_keys = self.language_keys or ["annotation.human.action.task_description"]
        self.observation_indices = self.observation_indices or [0]
        self.action_indices = self.action_indices or list(range(16))


# ═══════════════════════════════════════════════════════════════════════
#  Registry  — all configs in one map
# ═══════════════════════════════════════════════════════════════════════

DATA_CONFIG_MAP: Dict[str, BaseDataConfig] = {
    # SO-100
    "so100": So100DataConfig(),
    "so100_dualcam": So100DualCamDataConfig(),
    "so100_4cam": So100QuadCamDataConfig(),
    # SO-101 (upgraded SO-100)
    "so101": So101DataConfig(),
    "so101_dualcam": So101DualCamDataConfig(),
    "so101_tricam": So101TriCamDataConfig(),
    # Fourier GR-1
    "fourier_gr1_arms_only": FourierGr1ArmsOnlyDataConfig(),
    "fourier_gr1_arms_waist": FourierGr1ArmsWaistDataConfig(),
    "fourier_gr1_full_upper_body": FourierGr1FullUpperBodyDataConfig(),
    # Unitree G1
    "unitree_g1": UnitreeG1DataConfig(),
    "unitree_g1_full_body": UnitreeG1FullBodyDataConfig(),
    "unitree_g1_locomanip": UnitreeG1LocoManipDataConfig(),
    # Franka Panda
    "bimanual_panda_gripper": BimanualPandaGripperDataConfig(),
    "bimanual_panda_hand": BimanualPandaHandDataConfig(),
    "single_panda_gripper": SinglePandaGripperDataConfig(),
    # OXE
    "oxe_droid": OxeDroidDataConfig(),
    "oxe_google": OxeGoogleDataConfig(),
    "oxe_widowx": OxeWidowXDataConfig(),
    # Simulation
    "libero_panda": LiberoPandaDataConfig(),
    # Other
    "agibot_genie1": AgibotGenie1DataConfig(),
    "agibot_dual_arm": AgibotDualArmGripperDataConfig(),
    "agibot_dual_arm_gripper": AgibotDualArmGripperDataConfig(),
    "agibot_dual_arm_dexhand": AgibotDualArmDexHandDataConfig(),
    "agibot_dual_arm_full": AgibotDualArmFullDataConfig(),
    "galaxea_r1_pro": GalaxeaR1ProDataConfig(),
}


def load_data_config(data_config: Union[str, BaseDataConfig]) -> BaseDataConfig:
    """Load a data configuration from string name or return the object directly."""
    if isinstance(data_config, BaseDataConfig):
        return data_config
    if isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            return DATA_CONFIG_MAP[data_config]
        raise ValueError(f"Unknown data_config '{data_config}'. Available: {list(DATA_CONFIG_MAP.keys())}")
    raise ValueError(f"data_config must be str or BaseDataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    video_keys: List[str],
    state_keys: List[str],
    action_keys: List[str],
    language_keys: Optional[List[str]] = None,
    observation_indices: Optional[List[int]] = None,
    action_indices: Optional[List[int]] = None,
) -> BaseDataConfig:
    """Create and register a custom data config at runtime."""

    class CustomDataConfig(BaseDataConfig):
        def __init__(self):
            self.video_keys = video_keys
            self.state_keys = state_keys
            self.action_keys = action_keys
            self.language_keys = language_keys or ["annotation.human.task_description"]
            self.observation_indices = observation_indices or [0]
            self.action_indices = action_indices or list(range(16))

    config = CustomDataConfig()
    DATA_CONFIG_MAP[name] = config
    logger.info(f"Registered custom config '{name}': cameras={video_keys} state={state_keys}")
    return config


__all__ = [
    "ModalityConfig",
    "BaseDataConfig",
    "DATA_CONFIG_MAP",
    "load_data_config",
    "create_custom_data_config",
    "So100DataConfig",
    "So100DualCamDataConfig",
    "So100QuadCamDataConfig",
    "So101DataConfig",
    "So101DualCamDataConfig",
    "So101TriCamDataConfig",
    "FourierGr1ArmsOnlyDataConfig",
    "FourierGr1ArmsWaistDataConfig",
    "FourierGr1FullUpperBodyDataConfig",
    "UnitreeG1DataConfig",
    "UnitreeG1FullBodyDataConfig",
    "UnitreeG1LocoManipDataConfig",
    "BimanualPandaGripperDataConfig",
    "BimanualPandaHandDataConfig",
    "SinglePandaGripperDataConfig",
    "OxeDroidDataConfig",
    "OxeGoogleDataConfig",
    "OxeWidowXDataConfig",
    "LiberoPandaDataConfig",
    "AgibotGenie1DataConfig",
    "AgibotDualArmGripperDataConfig",
    "AgibotDualArmDexHandDataConfig",
    "AgibotDualArmFullDataConfig",
    "GalaxeaR1ProDataConfig",
]
