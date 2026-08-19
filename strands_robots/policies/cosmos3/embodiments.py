"""Cosmos 3 embodiment specs - data-driven action/observation layouts.

Each embodiment maps a Cosmos 3 ``domain_name`` (the world-model conditioning
domain) to its raw action dimensionality, default action-chunk size, and the
named layout of the action vector columns so the policy can emit per-actuator
dicts instead of opaque float rows.

Dimensions verified against ``cosmos_framework.data.vfm.action.domain_utils``
(``EMBODIMENT_TO_RAW_ACTION_DIM``) and the released RoboLab DROID policy server
defaults (``action_policy_server_robolab.py``).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Cosmos3Embodiment:
    """Static description of a Cosmos 3 action embodiment.

    Attributes:
        name: Short embodiment key (``"droid"``).
        domain_name: Cosmos 3 conditioning domain (``"droid_lerobot"``).
        raw_action_dim: Raw model action dim for the domain (DROID = 10).
        action_chunk_size: Default number of predicted action steps.
        fps: Conditioning FPS the policy was trained at.
        camera_keys: Server observation image keys (OpenPI ``/`` namespace).
        action_layouts: ``{action_space: [column_name, ...]}`` naming each
            output action column so :class:`Cosmos3Policy` can build
            per-actuator step dicts. Used by the ``service`` backend. The
            RoboLab server post-processes only ``joint_pos``, converting the
            unified action's effector pose into joint targets (DROID: 8D =
            7 joints + gripper); ``midtrain`` is the model's own unified
            action served through, so a ``midtrain`` entry names the same
            columns as :attr:`raw_action_layout`. A narrower ``midtrain``
            entry does not shorten the served row - it leaves the surplus
            columns unnamed and shifts every name it does carry onto the
            wrong column.
        raw_action_layout: Column names for the **raw unified action** that the
            in-process ``diffusers`` :class:`Cosmos3OmniPipeline` emits directly
            (width = :attr:`raw_action_dim`). This is the model's native action
            (e.g. DROID = 9D end-effector pose + 1D gripper = 10D), *before* the
            RoboLab server's joint_pos conversion. The ``diffusers`` backend
            names its columns from this layout (no fabricated IK to joints).
        default_action_space: The action space the server serves by default.
        normalization: Action normalization method the model emits in
            (``"quantile"`` for all current Cosmos 3 domains). The ``diffusers``
            backend's raw action is in this normalized space; the de-normalize +
            IK sim bridge (:mod:`action_decode` / :mod:`sim_ik`) inverts it with
            the bundled per-domain ``q01``/``q99`` stats before solving joint
            targets.
    """

    name: str
    domain_name: str
    raw_action_dim: int
    action_chunk_size: int
    fps: int
    camera_keys: list[str] = field(default_factory=list)
    action_layouts: dict[str, list[str]] = field(default_factory=dict)
    raw_action_layout: list[str] = field(default_factory=list)
    default_action_space: str = "joint_pos"
    normalization: str = "quantile"


# Canonical 7-DOF Franka joint names (DROID/RoboMIND-Franka), matching the
# ordered joint convention used by the released Cosmos3-Nano-Policy-DROID.
_FRANKA_JOINTS = [f"joint_{i}" for i in range(7)]

# DROID joint_pos action = [7 joint deltas/targets, 1 gripper]. This is the one
# layout the RoboLab server converts, so it is the only one that legitimately
# differs from the model's unified action width.
_DROID_JOINT_POS = _FRANKA_JOINTS + ["gripper"]

# Raw unified-action layouts: the native action the Cosmos3OmniPipeline emits
# (diffusers backend), BEFORE the RoboLab server's joint_pos conversion. The
# unified action composes a 9D effector pose (3D translation tx/ty/tz + 6D
# rotation r0..r5, the over-parameterized rotation of Zhou et al. 2019) and a
# 1D grasp state (Cosmos 3 paper Fig. 3). Width = raw_action_dim.
_POSE9 = ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5"]
_RAW_POSE9_GRASP = _POSE9 + ["grasp"]  # 10D: one arm = 9D pose + 1D gripper


EMBODIMENTS: dict[str, Cosmos3Embodiment] = {
    "droid": Cosmos3Embodiment(
        name="droid",
        domain_name="droid_lerobot",
        raw_action_dim=10,
        action_chunk_size=32,
        fps=15,
        camera_keys=[
            "observation/wrist_image_left",
            "observation/exterior_image_1_left",
            "observation/exterior_image_2_left",
        ],
        action_layouts={
            "joint_pos": _DROID_JOINT_POS,
            # ``midtrain`` is the unified action itself, un-converted.
            "midtrain": _RAW_POSE9_GRASP,
        },
        raw_action_layout=_RAW_POSE9_GRASP,
        default_action_space="joint_pos",
    ),
    "umi": Cosmos3Embodiment(
        name="umi",
        domain_name="umi",
        raw_action_dim=10,
        action_chunk_size=16,
        fps=20,
        camera_keys=["observation/image"],
        action_layouts={
            # EE 9D pose delta (3D translation + 6D rotation) + 1D grasp.
            "midtrain": [
                "tx",
                "ty",
                "tz",
                "r0",
                "r1",
                "r2",
                "r3",
                "r4",
                "r5",
                "grasp",
            ],
        },
        raw_action_layout=_RAW_POSE9_GRASP,
        default_action_space="midtrain",
    ),
    "av": Cosmos3Embodiment(
        name="av",
        domain_name="av",
        raw_action_dim=9,
        action_chunk_size=60,
        fps=10,
        camera_keys=["observation/image"],
        action_layouts={
            # Ego pose 9D (3D translation + 6D rotation), no gripper.
            "midtrain": ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5"],
        },
        raw_action_layout=_POSE9,
        default_action_space="midtrain",
    ),
    "bridge": Cosmos3Embodiment(
        name="bridge",
        domain_name="bridge_orig_lerobot",
        raw_action_dim=10,
        action_chunk_size=16,
        fps=5,
        camera_keys=["observation/image"],
        action_layouts={
            "midtrain": ["tx", "ty", "tz", "r0", "r1", "r2", "r3", "r4", "r5", "grasp"],
        },
        raw_action_layout=_RAW_POSE9_GRASP,
        default_action_space="midtrain",
    ),
}

# Aliases → canonical embodiment key.
_EMBODIMENT_ALIASES = {
    "droid_lerobot": "droid",
    "robomind-franka": "droid",
    "franka": "droid",
    "bridge_orig_lerobot": "bridge",
    "autonomous_vehicle": "av",
}


# Built-in action mappings: DROID action-layout column -> robot actuator name.
# The released Cosmos3-Nano-Policy-DROID joint_pos layout is
# [joint_0..joint_6, gripper]; a MuJoCo Franka/Panda exposes joint1..joint7 +
# finger_joint1. Pass ``action_mapping=ROBOT_ACTION_MAPPINGS["panda"]`` (or just
# ``robot="panda"`` sugar in create_policy) so per-step dicts use real actuator
# names and don't silently miss in ``send_action``.
ROBOT_ACTION_MAPPINGS: dict[str, dict[str, str]] = {
    "panda": {
        **{f"joint_{i}": f"joint{i + 1}" for i in range(7)},  # joint_0->joint1 ... joint_6->joint7
        "gripper": "finger_joint1",
    },
    "franka": {
        **{f"joint_{i}": f"joint{i + 1}" for i in range(7)},
        "gripper": "finger_joint1",
    },
}


def get_robot_action_mapping(robot: str) -> dict[str, str] | None:
    """Return a built-in DROID-layout -> robot-actuator action_mapping, if known."""
    return ROBOT_ACTION_MAPPINGS.get(robot.lower().strip())


def get_embodiment(name: str) -> Cosmos3Embodiment:
    """Resolve an embodiment by name or alias.

    Args:
        name: Embodiment key or alias (``"droid"``, ``"droid_lerobot"``, ...).

    Returns:
        The matching :class:`Cosmos3Embodiment`.

    Raises:
        ValueError: If the embodiment is unknown (consistent with the other
            invalid-argument validations in Cosmos3Policy).
    """
    key = name.lower().strip()
    key = _EMBODIMENT_ALIASES.get(key, key)
    if key not in EMBODIMENTS:
        raise ValueError(
            f"Unknown Cosmos 3 embodiment {name!r}. "
            f"Available: {sorted(EMBODIMENTS)} (+ aliases {sorted(_EMBODIMENT_ALIASES)})"
        )
    return EMBODIMENTS[key]


def list_robot_action_mappings() -> list[str]:
    """List robots with a built-in DROID action mapping."""
    return sorted(ROBOT_ACTION_MAPPINGS)


def list_embodiments() -> list[str]:
    """List canonical embodiment names."""
    return sorted(EMBODIMENTS)
