"""Action-key bridge between :class:`KimodoPolicy` and lerobot's ``unitree_g1`` driver.

Kimodo emits per-frame joint targets keyed by URDF joint names
(``left_hip_pitch_joint``, ...). lerobot's ``UnitreeG1`` driver accepts action
keys named after its ``G1_29_JointIndex`` enum (``kLeftHipPitch.q``, ...). The
two vocabularies name the SAME 29 joints, so the bridge is a pure key rename:
no reordering, no scaling, no unit conversion.

The rename table is derived BY NAME, never by position. Each driver enum member
canonicalises to the URDF spelling of the same joint (``kLeftHipPitch`` ->
``left_hip_pitch_joint``) and is paired with it. Position is not usable as the
pairing key here because the driver silently drops an action key it does not
recognise (``UnitreeG1.publish_lowcmd`` writes only the keys it finds, leaving
every other motor on its previous command), so a mis-pairing raises nothing at
all -- the robot simply moves wrong. Two consequences follow from pairing by
name:

* A driver-side reorder cannot pair a hip target with a waist actuator.
* A driver-side rename or DOF change is refused, naming the unmatched joints on
  both sides, instead of being mapped onto whatever now sits at that index.

The map is between the canonical G1 joint names and the driver's action keys, so
any provider emitting those names (Kimodo, ``wbc``, ``motionbricks``) can use
it. The rename is one-way (policy -> driver): the driver's ``get_observation``
already reports ``<motor>.q`` keys, so the read path needs no inverse map.

Nothing in the rollout loop applies this implicitly. A hardware caller renames
the policy's action dict before handing it to the driver::

    action = build_lerobot_g1_action_dict(policy_action)
    robot.send_action(action)
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from typing import Protocol

from strands_robots.policies.kimodo.policy import KIMODO_G1_JOINTS
from strands_robots.utils import require_optional

logger = logging.getLogger(__name__)

#: Dotted path to the module that owns the driver's joint enum.
_DRIVER_MODULE = "lerobot.robots.unitree_g1.g1_utils"

#: Name of the driver's 29-DOF joint enum inside :data:`_DRIVER_MODULE`.
_DRIVER_ENUM = "G1_29_JointIndex"

#: The driver names a joint-position action key ``<enum member>.q``.
_DRIVER_POSITION_SUFFIX = ".q"

#: The driver's enum members are prefixed camel case (``kLeftHipPitch``).
_DRIVER_NAME_PREFIX = "k"

#: URDF suffix every canonical G1 joint name carries (``left_hip_pitch_joint``).
_URDF_NAME_SUFFIX = "_joint"

_CAMEL_BOUNDARY = re.compile(r"(?<!^)(?=[A-Z])")


class _DriverJoint(Protocol):
    """The one attribute the bridge reads off a driver enum member."""

    name: str


def _canonical_joint_name(driver_name: str) -> str:
    """Translate a driver enum member name to its canonical G1 joint name.

    ``kLeftHipPitch`` -> ``left_hip_pitch_joint``.

    Args:
        driver_name: Enum member name as the driver spells it.

    Returns:
        The URDF spelling of the same joint.
    """
    core = driver_name.removeprefix(_DRIVER_NAME_PREFIX)
    return f"{_CAMEL_BOUNDARY.sub('_', core).lower()}{_URDF_NAME_SUFFIX}"


def _build_joint_map(driver_joints: Iterable[_DriverJoint]) -> dict[str, str]:
    """Pair every canonical G1 joint with the driver action key of the same joint.

    Args:
        driver_joints: Iterable of driver enum members (each exposing ``name``),
            normally ``G1_29_JointIndex``.

    Returns:
        ``{canonical joint name: driver action key}`` in
        :data:`~strands_robots.policies.kimodo.policy.KIMODO_G1_JOINTS` order.

    Raises:
        RuntimeError: If two driver joints canonicalise to one name, or if any
            joint is named on only one side. Either way at least one target
            would be dropped by the driver without an error, so the mismatch is
            reported instead of mapped.
    """
    by_canonical: dict[str, str] = {}
    for joint in driver_joints:
        canonical = _canonical_joint_name(joint.name)
        if canonical in by_canonical:
            raise RuntimeError(
                f"Unitree G1 driver joints {by_canonical[canonical]!r} and "
                f"{joint.name + _DRIVER_POSITION_SUFFIX!r} both name the joint "
                f"{canonical!r}. The bridge cannot tell which actuator a "
                f"{canonical!r} target belongs to; audit the driver's "
                f"{_DRIVER_ENUM} before driving the robot."
            )
        by_canonical[canonical] = f"{joint.name}{_DRIVER_POSITION_SUFFIX}"

    unmapped_policy = [name for name in KIMODO_G1_JOINTS if name not in by_canonical]
    unmapped_driver = sorted(set(by_canonical) - set(KIMODO_G1_JOINTS))
    if unmapped_policy or unmapped_driver:
        raise RuntimeError(
            "Unitree G1 joint sets disagree between the policy and lerobot's "
            f"driver. Joints the policy commands that the driver does not name: "
            f"{unmapped_policy}. Joints the driver names that the policy does "
            f"not command: {[by_canonical[name] for name in unmapped_driver]}. "
            "The driver applies only the action keys it recognises and leaves "
            "every other motor on its previous command, so a partial map would "
            "move the robot without reporting anything. Audit the joint sets "
            "before driving the robot."
        )
    return {name: by_canonical[name] for name in KIMODO_G1_JOINTS}


_joint_map: dict[str, str] | None = None


def get_joint_map() -> dict[str, str]:
    """Return the canonical-to-driver joint rename table, building it on first call.

    Built lazily so a pure-sim import never looks for lerobot, then cached: the
    table is a property of the two joint vocabularies, not of a rollout.

    Returns:
        ``{"left_hip_pitch_joint": "kLeftHipPitch.q", ...}``, one entry per
        joint in :data:`~strands_robots.policies.kimodo.policy.KIMODO_G1_JOINTS`.

    Raises:
        ImportError: If lerobot is not installed.
        RuntimeError: If the two joint vocabularies do not name the same 29
            joints, naming the unmatched joints on both sides.
    """
    global _joint_map
    if _joint_map is None:
        module = require_optional(
            _DRIVER_MODULE,
            extra="lerobot",
            purpose="the Kimodo action-key bridge for lerobot's Unitree G1 driver",
        )
        _joint_map = _build_joint_map(getattr(module, _DRIVER_ENUM))
        logger.debug("Built Unitree G1 action-key map with %d joints", len(_joint_map))
    return _joint_map


def kimodo_action_to_lerobot_g1(kimodo_action: dict[str, float]) -> dict[str, float]:
    """Rename a Kimodo joint dict to the lerobot ``UnitreeG1`` action dict.

    Args:
        kimodo_action: One per-tick dict from :meth:`KimodoPolicy.get_actions`
            (keys are :data:`~strands_robots.policies.kimodo.policy.KIMODO_G1_JOINTS`,
            values are radians).

    Returns:
        A new dict keyed for ``UnitreeG1.send_action`` (``kLeftHipPitch.q``,
        ...). Keys outside the canonical joint set are dropped rather than
        forwarded: the driver ignores keys it does not recognise, so forwarding
        them only hides a naming mistake.

    Raises:
        KeyError: If any canonical G1 joint is absent from ``kimodo_action``.
            The driver would hold that motor on its previous command, so a
            short action dict is refused rather than partially applied.
    """
    joint_map = get_joint_map()
    missing = [name for name in joint_map if name not in kimodo_action]
    if missing:
        raise KeyError(
            f"Kimodo action dict is missing G1 joints: {missing}. "
            f"KimodoPolicy.get_actions() returns all {len(joint_map)} joints; "
            f"got keys={sorted(kimodo_action)}."
        )
    return {joint_map[name]: float(kimodo_action[name]) for name in joint_map}


def build_lerobot_g1_action_dict(
    kimodo_action: dict[str, float],
    *,
    extra_action_keys: dict[str, float] | None = None,
) -> dict[str, float]:
    """Build the driver-side action dict for one control tick.

    Args:
        kimodo_action: One per-tick dict from :meth:`KimodoPolicy.get_actions`.
        extra_action_keys: Optional driver keys merged AFTER the rename, so they
            win over a renamed joint target (locomotion remote inputs such as
            ``remote.lx``, and per-joint overrides). Most callers pass ``None``.

    Returns:
        An action dict ready for ``UnitreeG1.send_action``.

    Raises:
        KeyError: If ``kimodo_action`` is missing a canonical G1 joint.
    """
    action = kimodo_action_to_lerobot_g1(kimodo_action)
    if extra_action_keys:
        action.update(extra_action_keys)
    return action


__all__ = [
    "build_lerobot_g1_action_dict",
    "get_joint_map",
    "kimodo_action_to_lerobot_g1",
]
