"""CompositePolicy - stack two policies that drive disjoint joints of one robot.

The canonical use is whole-body humanoid control: a locomotion controller
(e.g. :class:`~strands_robots.policies.wbc.WBCPolicy`) drives the legs+waist
while a manipulation policy (GR00T / pi0 / MolmoAct, or any
:class:`~strands_robots.policies.base.Policy`) drives the arms - on the SAME
robot, every control tick. Upstream GR00T Whole-Body-Control layers a teleop /
manipulation upper body on top of the balance controller this way; this class
is the in-process equivalent.

Both children are queried each tick and their per-tick action dicts are merged
by JOINT NAME. The lower policy owns its joint group; the upper policy fills the
rest. Ownership is explicit (``lower_joints`` / ``upper_joints``) or, when left
default, the lower policy takes precedence on any name it emits and the upper
policy contributes the remaining names. A genuine ownership conflict (both
children emit a value for a name each claims) is raised, never silently
resolved - a dropped command on a humanoid is a fall, not a warning. By the same
rule, a child whose ENTIRE action dict is discarded by routing is refused: a
composite in which one child reaches no actuator is the other child alone.

This class composes joint groups in PARALLEL; it is not a cascade. A whole-body
kinematic generator (Kimodo / MotionBricks) and a controller that tracks its
reference drive the SAME joints and run in SERIES - the generator's targets are
the controller's input, not a disjoint half of one action dict. No composite
expresses that: a reference tracker has to consume the reference through its own
interface.

Example::

    from strands_robots.policies import create_policy
    from strands_robots.policies.composite import CompositePolicy
    from strands_robots.policies.wbc import WBC_G1_LEG_WAIST_JOINTS

    lower = create_policy("wbc", checkpoint="/path/to/grootwbc-g1")
    upper = create_policy("groot", port=5555)
    policy = CompositePolicy(
        lower=lower,
        upper=upper,
        lower_joints=WBC_G1_LEG_WAIST_JOINTS,   # legs + waist
        upper_joints=ARM_JOINTS,                 # both arms
    )
    sim.run_policy(robot_name="unitree_g1", policy_object=policy,
                   policy_kwargs={"target_velocity": [0.5, 0.0, 0.0]},
                   control_frequency=50.0, n_steps=500)
"""

import asyncio
import logging
from collections.abc import Sequence
from typing import Any

from strands_robots.policies.base import Policy

logger = logging.getLogger(__name__)


class CompositePolicy(Policy):
    """Compose a ``lower`` and an ``upper`` policy on one robot's joint set.

    Each :meth:`get_actions` queries both children with the (optionally
    per-child filtered) observation and the same instruction + kwargs, then
    merges their per-tick action dicts by joint name:

    * ``lower`` contributes the names in ``lower_joints`` (or all names it emits
      when ``lower_joints`` is ``None``).
    * ``upper`` contributes the names in ``upper_joints`` (or, when ``None``,
      every name it emits that the lower policy did not already claim - lower
      precedence).

    Routing that discards a child's ENTIRE action dict raises: the composite
    would otherwise silently be the surviving child alone. Two children that
    drive the same joint set cannot be composed, only cascaded (module docstring).

    The merged chunk length is the shorter of the two children's chunks, so the
    more frequently re-querying child sets the re-query cadence
    (:attr:`execution_horizon`). This keeps a per-tick controller (WBC,
    ``execution_horizon == 1``) closed-loop even when paired with a chunk-emitting
    manipulation policy.

    Args:
        lower: Policy driving the lower joint group (e.g. legs+waist locomotion).
        upper: Policy driving the upper joint group (e.g. arms manipulation).
        lower_joints: Joint/actuator names the lower policy is authoritative for.
            ``None`` (default) accepts every name the lower policy emits.
        upper_joints: Joint/actuator names the upper policy is authoritative for.
            ``None`` (default) accepts every upper name not already owned by the
            lower policy.
        lower_obs_keys: Observation keys to forward to the lower policy. ``None``
            (default) forwards the full observation (children read by name).
        upper_obs_keys: Observation keys to forward to the upper policy. ``None``
            (default) forwards the full observation.

    Raises:
        ValueError: If ``lower`` or ``upper`` is ``None``, or if ``lower_joints``
            and ``upper_joints`` are both given and share a name (ambiguous
            ownership).
    """

    def __init__(
        self,
        lower: Policy,
        upper: Policy,
        *,
        lower_joints: Sequence[str] | None = None,
        upper_joints: Sequence[str] | None = None,
        lower_obs_keys: Sequence[str] | None = None,
        upper_obs_keys: Sequence[str] | None = None,
    ) -> None:
        if lower is None or upper is None:
            raise ValueError("CompositePolicy requires both a 'lower' and an 'upper' policy.")
        self._lower = lower
        self._upper = upper
        self._lower_joints: set[str] | None = set(lower_joints) if lower_joints is not None else None
        self._upper_joints: set[str] | None = set(upper_joints) if upper_joints is not None else None
        if self._lower_joints is not None and self._upper_joints is not None:
            overlap = self._lower_joints & self._upper_joints
            if overlap:
                raise ValueError(
                    "CompositePolicy lower_joints and upper_joints must be disjoint; "
                    f"both claim {sorted(overlap)}. Assign each joint to exactly one policy."
                )
        self._lower_obs_keys: set[str] | None = set(lower_obs_keys) if lower_obs_keys is not None else None
        self._upper_obs_keys: set[str] | None = set(upper_obs_keys) if upper_obs_keys is not None else None
        logger.info("CompositePolicy: lower=%s upper=%s", self._lower.provider_name, self._upper.provider_name)

    @property
    def lower(self) -> Policy:
        """The lower-body (e.g. locomotion) child policy."""
        return self._lower

    @property
    def upper(self) -> Policy:
        """The upper-body (e.g. manipulation) child policy."""
        return self._upper

    @property
    def children(self) -> tuple[Policy, ...]:
        """Both child policies, ``lower`` first - the order they are merged in.

        Lets a runtime capability probe reach the concrete policies inside the
        composite; see :attr:`Policy.children`.
        """
        return (self._lower, self._upper)

    @property
    def provider_name(self) -> str:
        """Provider name for identification (always ``"composite"``)."""
        return "composite"

    @property
    def requires_images(self) -> bool:
        """True if EITHER child consumes camera frames.

        The composite cannot skip rendering unless both children opt out; a
        manipulation upper body typically needs images even when the locomotion
        lower body does not.
        """
        return self._lower.requires_images or self._upper.requires_images

    @property
    def execution_horizon(self) -> int:
        """Re-query interval: the shorter of the two children's horizons.

        The merged chunk is truncated to the shorter child's length, so the
        consumer must re-query at the faster child's cadence to keep that child
        closed-loop (a per-tick locomotion controller must not be starved by a
        slower chunk-emitting manipulation policy).
        """
        return max(1, min(self._lower.execution_horizon, self._upper.execution_horizon))

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Forward the robot's state-key list to both children."""
        self._lower.set_robot_state_keys(robot_state_keys)
        self._upper.set_robot_state_keys(robot_state_keys)

    def set_control_frequency(self, hz: float) -> None:
        """Set the control rate on the composite and forward it to both children."""
        super().set_control_frequency(hz)
        self._lower.set_control_frequency(hz)
        self._upper.set_control_frequency(hz)

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        """Forward the RTC observed-delay step count to the composite and both children."""
        super().set_rtc_observed_delay(steps)
        self._lower.set_rtc_observed_delay(steps)
        self._upper.set_rtc_observed_delay(steps)

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode state on both children."""
        self._lower.reset(seed)
        self._upper.reset(seed)

    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Query both children and merge their per-tick action dicts by joint name.

        Both children receive the same ``instruction`` and ``kwargs`` (each
        ignores keys it does not use, per the :class:`Policy` contract), and the
        observation filtered to its configured key subset. The two action chunks
        are merged element-wise up to the shorter length.

        Returns:
            The merged action chunk (length == the shorter child's chunk). Each
            dict maps a joint/actuator name to its target value, routed from the
            child that owns that name.

        Raises:
            ValueError: If either child returns an empty chunk, if routing
                discards a child's entire action dict (the composite would be
                the other child alone), or if the routed lower and upper action
                dicts both assign the same joint name.
        """
        lower_obs = self._filter_obs(observation_dict, self._lower_obs_keys)
        upper_obs = self._filter_obs(observation_dict, self._upper_obs_keys)
        lower_chunk, upper_chunk = await asyncio.gather(
            self._lower.get_actions(lower_obs, instruction, **kwargs),
            self._upper.get_actions(upper_obs, instruction, **kwargs),
        )
        if not lower_chunk:
            raise ValueError(f"CompositePolicy lower policy '{self._lower.provider_name}' returned no actions.")
        if not upper_chunk:
            raise ValueError(f"CompositePolicy upper policy '{self._upper.provider_name}' returned no actions.")

        n = min(len(lower_chunk), len(upper_chunk))
        if len(lower_chunk) != len(upper_chunk):
            logger.debug(
                "CompositePolicy: chunk lengths differ (lower=%d, upper=%d); merging the first %d tick(s).",
                len(lower_chunk),
                len(upper_chunk),
                n,
            )
        return [self._merge_tick(lower_chunk[i], upper_chunk[i]) for i in range(n)]

    def _merge_tick(self, lower_action: dict[str, Any], upper_action: dict[str, Any]) -> dict[str, Any]:
        """Merge one tick's lower + upper action dicts with joint-name routing."""
        lo = self._route(lower_action, self._lower_joints)
        # Upper default: every name not already claimed by the routed lower dict.
        up = self._route(upper_action, self._upper_joints, exclude=None if self._upper_joints else set(lo))
        self._reject_discarded_child("lower", self._lower, lower_action, lo, self._lower_joints, set())
        self._reject_discarded_child("upper", self._upper, upper_action, up, self._upper_joints, set(lo))
        collision = set(lo) & set(up)
        if collision:
            raise ValueError(
                "CompositePolicy: lower and upper policies both produced joint(s) "
                f"{sorted(collision)}. Set lower_joints / upper_joints so each joint "
                "is driven by exactly one policy."
            )
        lo.update(up)
        return lo

    def _reject_discarded_child(
        self,
        role: str,
        child: Policy,
        emitted: dict[str, Any],
        routed: dict[str, Any],
        group: set[str] | None,
        claimed_by_lower: set[str],
    ) -> None:
        """Refuse a tick in which routing discarded everything ``child`` commanded.

        A child whose whole action dict is dropped reaches no actuator, so the
        composite commands exactly what the other child alone would - a no-op
        wearing the shape of a composition. That is the same dropped command this
        class raises on elsewhere, so name the child, the joints it commanded and
        the reason instead of returning a tick the caller will read as composed.

        Args:
            role: The seat ``child`` occupies - ``"lower"`` or ``"upper"``.
            child: The child policy whose routed output is being checked.
            emitted: The action dict ``child`` returned for this tick.
            routed: What survived routing for ``child``.
            group: ``child``'s explicit joint group, or ``None`` when defaulted.
            claimed_by_lower: Names the routed lower dict already claims. Empty
                for the lower seat, which is routed first and claims freely.

        Raises:
            ValueError: If ``child`` commanded at least one joint and none of them
                survived routing. A child that commanded nothing is left alone -
                there is no command to drop.
        """
        if routed or not emitted:
            return
        names = sorted(emitted)
        head = (
            f"CompositePolicy: the {role} policy '{child.provider_name}' commanded "
            f"{len(names)} joint(s) and none of them reached the merged action"
        )
        if group is not None:
            raise ValueError(
                f"{head} - the {role}_joints group {sorted(group)} shares no name with the "
                f"joints it emits {names}. Set {role}_joints to names this policy emits."
            )
        lower_name = self._lower.provider_name
        raise ValueError(
            f"{head} - the lower policy '{lower_name}' already drives every one of "
            f"{sorted(set(names) & claimed_by_lower)}, so this composite commands exactly what "
            f"'{lower_name}' alone would. CompositePolicy merges DISJOINT joint groups (e.g. "
            "locomotion legs+waist plus manipulation arms); it does not feed one policy's targets "
            "into another policy. Give each joint exactly one owner via lower_joints / "
            "upper_joints, or drive the whole body with a single policy."
        )

    @staticmethod
    def _route(action: dict[str, Any], joints: set[str] | None, *, exclude: set[str] | None = None) -> dict[str, Any]:
        """Select the names this child contributes from its action dict.

        With ``joints`` set, keep only those names. Otherwise keep every name,
        minus any in ``exclude`` (used to give the lower policy precedence on
        shared names when the upper group is left default).
        """
        if joints is not None:
            return {k: v for k, v in action.items() if k in joints}
        if exclude:
            return {k: v for k, v in action.items() if k not in exclude}
        return dict(action)

    @staticmethod
    def _filter_obs(observation_dict: dict[str, Any], keys: set[str] | None) -> dict[str, Any]:
        """Forward the full observation, or only ``keys`` when a subset is configured."""
        if keys is None:
            return observation_dict
        return {k: v for k, v in observation_dict.items() if k in keys}


__all__ = ["CompositePolicy"]
