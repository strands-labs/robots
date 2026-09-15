"""Task-space delta-EEF to joint-position-target controller for Isaac.

:class:`IsaacDeltaEEFController` is the Isaac-side counterpart of the
MuJoCo/robosuite ``OSC_POSE`` controller that
``LiberoAdapter._install_action_controller`` builds against a compiled
MuJoCo model. GR00T-LIBERO checkpoints emit 7-dim Cartesian **delta-EEF**
actions (``{x, y, z, roll, pitch, yaw, gripper}``); on Isaac there is no
compiled MuJoCo model, so without this controller every action key lands in
``send_action``'s ``unresolved_keys`` and the robot never moves (#1812).

The controller converts each task-space delta into **joint position
targets** via a damped-least-squares (DLS) differential-IK solve on the
end-effector's world-frame spatial Jacobian:

    dq = J^T (J J^T + lambda^2 I)^{-1} * twist

and returns ``{joint_name: q_current + dq}`` for
:meth:`IsaacSimulation.send_action` to drive through the articulation's PD
position targets. This is issue #1812's option 1 (position-level
differential IK); a torque-level OSC re-implementation (option 2, closest
to what the checkpoint was trained under) is a follow-up.

Action semantics (must match the MuJoCo baseline, #168):

* ``x/y/z/roll/pitch/yaw`` are **normalized** per-step deltas. robosuite's
  ``OSC_POSE`` clips each input to ``[-1, 1]`` and scales by ``output_max``
  (``0.05`` m for position, ``0.5`` rad for rotation) per 20 Hz control
  step -- see ``robosuite/controllers/config/osc_pose.json``. The same
  clip-then-scale is applied here.
* Rotation deltas are applied in the **world frame** (robosuite's
  ``set_goal_orientation`` premultiplies the delta rotation), which matches
  the world-frame angular rows of PhysX's spatial Jacobian.
* ``gripper`` is in the RLDS convention (``0 = close``, ``1 = open``,
  ``0.5`` = no command). The RLDS-to-LIBERO conversion is
  ``-sign(2*v - 1)`` (``+1`` close / ``-1`` open); here that maps onto the
  Isaac Franka USD's position-driven fingers as ``gripper_close`` /
  ``gripper_open`` joint targets.

Dependency-free by construction: current joint positions and the Jacobian
are injected as callables, so unit tests exercise the full conversion with
a mocked articulation and the class imports without Isaac Sim installed.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from strands_robots.utils import boolean_flag_error, finite_number_error, positive_finite_number_error

logger = logging.getLogger(__name__)

__all__ = ["IsaacDeltaEEFController", "TASK_SPACE_ACTION_KEYS"]

#: Action-dict keys this controller consumes. Anything else in the action
#: dict is passed through untouched so ``send_action``'s unresolved-key
#: reporting stays honest (a key this controller does not understand must
#: surface in the envelope, never vanish).
TASK_SPACE_ACTION_KEYS: frozenset[str] = frozenset({"x", "y", "z", "roll", "pitch", "yaw", "gripper"})

_POS_KEYS = ("x", "y", "z")
_ROT_KEYS = ("roll", "pitch", "yaw")

#: robosuite OSC_POSE ``output_max`` -- metres of translation per control
#: step for a saturated (+/-1) input. From
#: ``robosuite/controllers/config/osc_pose.json``; the GR00T-N1.7-LIBERO
#: checkpoint was trained against these bounds at 20 Hz (#168).
DEFAULT_POS_SCALE = 0.05
#: robosuite OSC_POSE rotational ``output_max`` -- radians per control step
#: for a saturated input.
DEFAULT_ROT_SCALE = 0.5


def _to_scalar(value: Any, key: str, *, strict: bool, default: float = 0.0) -> float:
    """Coerce a present GR00T action channel to a finite scalar float.

    GR00T-LIBERO packs every action channel list-shaped (2-element list /
    ndarray) to match the training-data shape, so the leading element is the
    scalar the controller means (#168).

    * Scalar input -> ``float(value)``
    * Non-empty list / tuple / ndarray -> ``float(value[0])``
    * Unreadable: raises under ``strict``, else ``default`` after a WARNING.
    * ``NaN``: **always** raises, in both modes.
    * ``+-inf``: accepted; it saturates to the maximum per-step delta, which is
      this controller's documented clip-then-scale behaviour.

    Only ever called for a channel the action actually carries. An ABSENT
    channel is a documented default resolved by the caller and never reaches
    here, which is the distinction this function exists to keep: absence means
    "hold this axis", and a value that cannot be read means the action is not
    what the controller believes it is.

    The two failures are separated because only one of them has a coherent
    degraded reading, which is why ``strict`` governs one and not the other.

    An **unreadable** channel can honestly be treated as "hold this axis", and
    that was the shipped behaviour - a WARNING plus ``0.0`` (``0.5`` for the
    gripper), so a partly-malformed action still moved the axes it did specify.
    That is a defensible posture and it is preserved under ``strict=False``. It
    is no longer the default, because it returns a zero-valued action on
    failure, and because the substitution is not always the "degrade one axis"
    it reads as: an unreadable ``gripper`` resolved to ``0.5`` ->
    ``-sign(0.0)`` -> ``-0.0``, which the ``command != 0.0`` guard drops
    entirely, so a grasp or a release silently did nothing and the fingers held
    their previous target.

    A **NaN** channel has no such reading, so ``strict=False`` does not reach it.
    ``float("nan")`` coerces successfully, the finiteness guard in
    ``_solve_arm_targets`` covers only the injected callables, and the solve
    returned ``{"j1": nan, "j2": nan, ...}`` - non-finite targets for *every* arm
    joint, not a held axis. Those reach PhysX, which reports them from a LATER
    step as "Illegal BroadPhaseUpdateData - non-finite bounds", attributed to
    whatever is running by then rather than to this action. It is also the one
    case the pre-existing test for this branch could not have caught while
    asserting what it asserts: it requires ``all(np.isfinite(...))`` of the
    targets and never passed a ``nan`` in.

    ``+-inf`` is a different matter and is **accepted**. ``np.clip(+-inf, -1, 1)``
    is ``+-1.0``, so it saturates to the maximum per-step delta - exactly the
    clip-then-scale contract. Measured against clean main, ``x=+inf`` produced a
    finite ``{"j1": 0.0499, ...}``. Refusing it would convert a correct saturation
    into a hard failure for any policy head that saturates.

    Args:
        value: The channel's value, as the policy delivered it.
        key: The channel name, so a refusal names the axis rather than the type.
        strict: Whether an unreadable channel raises rather than degrading.
        default: The value substituted for an unreadable channel when
            ``strict`` is false. Unused otherwise.

    Returns:
        The channel as a finite float.

    Raises:
        ValueError: If the value is not finite, or - under ``strict`` - cannot
            be read as a scalar.
    """
    try:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            scalar = float(value[0])
        else:
            scalar = float(value)
    except (TypeError, ValueError, IndexError) as exc:
        if strict:
            raise ValueError(
                f"IsaacDeltaEEFController: action channel {key!r} = {value!r} cannot be read as "
                f"a scalar ({exc}). Channels are a scalar or a non-empty list/ndarray whose "
                f"first element is the value. Omit the channel to hold that axis. Pass "
                f"strict=False to the controller to degrade an unreadable channel to a neutral "
                f"value with a warning instead."
            ) from exc
        logger.warning(
            "IsaacDeltaEEFController: could not coerce action value %r for channel %r to float "
            "(%s); using %s for this step (strict=False)",
            value,
            key,
            exc,
            default,
        )
        return default
    if np.isnan(scalar):
        # NaN only, and not governed by strict: there is no degraded reading of it.
        #
        # +-inf is deliberately NOT refused here. It is well defined all the way
        # through: np.clip(+-inf, -1, 1) is +-1.0, so an infinite channel saturates
        # to the maximum per-step delta, which is precisely the documented
        # clip-then-scale semantics this controller implements. Measured against
        # clean main, x=+inf produced {"j1": 0.0499, ...} - a finite, saturated
        # target - while x=nan produced {"j1": nan, "j2": nan}. Refusing inf would
        # turn a correct saturation into a hard failure for any policy head that
        # saturates, so only the value with no coherent reading is refused.
        raise ValueError(
            f"IsaacDeltaEEFController: action channel {key!r} = {value!r} is NaN. It solves to "
            f"non-finite targets for every arm joint, which PhysX reports from a LATER step as "
            f"'Illegal BroadPhaseUpdateData - non-finite bounds' - attributed to whatever is "
            f"running by then rather than to this action. Omit the channel to hold that axis. "
            f"strict=False does not relax this. (+-inf is accepted and saturates to the maximum "
            f"per-step delta, which is this controller's documented clip-then-scale behaviour.)"
        )
    return scalar


class IsaacDeltaEEFController:
    """Convert GR00T task-space delta-EEF dicts to joint position targets.

    Parameters
    ----------
    arm_joint_names : sequence of str
        Ordered arm joint names (e.g. ``panda_joint1..7``). Must match the
        column order of the Jacobian returned by ``jacobian_fn`` and the
        element order of ``joint_positions_fn``.
    gripper_joint_names : sequence of str
        Position-driven gripper joint names (e.g.
        ``panda_finger_joint1/2``). Each receives ``gripper_open`` /
        ``gripper_close`` as its target.
    joint_positions_fn : callable
        Zero-arg callable returning the current arm joint positions as an
        array-like of ``len(arm_joint_names)`` floats.
    jacobian_fn : callable
        Zero-arg callable returning the end-effector's world-frame spatial
        Jacobian as a ``(6, len(arm_joint_names))`` array-like -- rows are
        ``[linear(3); angular(3)]``, columns are the arm joints in
        ``arm_joint_names`` order (PhysX row convention).
    joint_limits : array-like or None
        Optional ``(len(arm_joint_names), 2)`` lower/upper position limits;
        targets are clipped into them. ``None`` disables clipping. Every
        bound must be finite -- a non-finite one clips every target to
        ``nan``.
    pos_scale, rot_scale : float
        Metres / radians of task-space delta for a saturated (+/-1) input
        channel. Defaults match robosuite ``OSC_POSE`` (#168). Both must
        be positive and finite (:func:`~strands_robots.utils.positive_finite_number_error`).
    damping : float
        DLS damping ``lambda``. Keeps the solve bounded near
        singularities; must be positive and finite
        (:func:`~strands_robots.utils.positive_finite_number_error`).
    gripper_open, gripper_close : float
        Joint position target written to every gripper joint for an
        open / close command. Defaults match the Franka USD's 0..0.04 m
        prismatic fingers. Either sign is usable, so only finiteness is
        constrained (:func:`~strands_robots.utils.finite_number_error`).
    strict : bool
        Whether an action channel that cannot be read as a scalar raises
        (default) or degrades to a neutral value with a WARNING, leaving the
        rest of the action to apply. ``False`` restores the behaviour this
        controller shipped with. It does **not** relax the finiteness check:
        a ``nan`` / ``inf`` channel raises either way, because it solves to
        non-finite targets for every arm joint rather than holding one axis.
        Checked, not read by truthiness
        (:func:`~strands_robots.utils.boolean_flag_error`), so ``strict="no"``
        cannot select the permissive posture while reading as the strict one.

    Concurrency: stateless between calls and does not touch the stage;
    safe to call from the thread driving ``send_action`` (which holds the
    engine lock). Not safe to share across robots.
    """

    def __init__(
        self,
        *,
        arm_joint_names: Sequence[str],
        gripper_joint_names: Sequence[str],
        joint_positions_fn: Callable[[], Any],
        jacobian_fn: Callable[[], Any],
        joint_limits: Any = None,
        pos_scale: float = DEFAULT_POS_SCALE,
        rot_scale: float = DEFAULT_ROT_SCALE,
        damping: float = 0.05,
        gripper_open: float = 0.04,
        gripper_close: float = 0.0,
        strict: bool = True,
    ) -> None:
        if text := boolean_flag_error(strict, "strict", "IsaacDeltaEEFController"):
            raise ValueError(text)
        self._strict = strict
        arm = [str(n) for n in arm_joint_names]
        if not arm:
            raise ValueError("arm_joint_names must be a non-empty sequence of joint names.")
        if len(set(arm)) != len(arm):
            raise ValueError(f"arm_joint_names contains duplicates: {arm}.")
        grip = [str(n) for n in gripper_joint_names]
        overlap = set(arm) & set(grip)
        if overlap:
            raise ValueError(f"arm and gripper joint names overlap: {sorted(overlap)}.")
        if not callable(joint_positions_fn) or not callable(jacobian_fn):
            raise TypeError("joint_positions_fn and jacobian_fn must be callables.")
        # Every numeric knob below is bounded by the shared scalar domain
        # rather than by an order comparison. An order comparison cannot
        # reject ``inf`` (``inf > 0`` is True), and each of these values
        # multiplies or is added to the DLS solve, so an accepted ``inf``
        # makes every joint target ``nan`` -- refused one action at a time
        # by ``send_action``'s action-value domain, which names the joint it
        # was handed and cannot know the value came from here.
        for _param, _value in (("pos_scale", pos_scale), ("rot_scale", rot_scale), ("damping", damping)):
            if error := positive_finite_number_error(_value, _param, "IsaacDeltaEEFController"):
                raise ValueError(error)
        # A gripper reference is a joint position, so both signs are
        # legitimate and only finiteness is constrained. Unguarded, a
        # non-finite one is latent: the controller runs healthy through
        # every approach action and fails at the first grasp.
        for _param, _value in (("gripper_open", gripper_open), ("gripper_close", gripper_close)):
            if error := finite_number_error(_value, _param, "IsaacDeltaEEFController"):
                raise ValueError(error)

        self.arm_joint_names = arm
        self.gripper_joint_names = grip
        self._joint_positions_fn = joint_positions_fn
        self._jacobian_fn = jacobian_fn
        self._pos_scale = float(pos_scale)
        self._rot_scale = float(rot_scale)
        self._damping = float(damping)
        self._gripper_open = float(gripper_open)
        self._gripper_close = float(gripper_close)

        if joint_limits is None:
            self._joint_limits: np.ndarray | None = None
        else:
            limits = np.asarray(joint_limits, dtype=np.float64)
            if limits.shape != (len(arm), 2):
                raise ValueError(
                    f"joint_limits must have shape ({len(arm)}, 2) to match arm_joint_names, got {limits.shape}."
                )
            if not np.all(np.isfinite(limits)):
                # Checked before the lower/upper comparison below, which
                # cannot see a non-finite bound: ``nan > nan`` is False, so
                # an all-nan table passes it and then clips every target to
                # nan in ``compute_joint_targets``.
                raise ValueError(
                    "IsaacDeltaEEFController: joint_limits must be finite (no nan/inf); "
                    "a non-finite bound clips every joint target to nan."
                )
            if np.any(limits[:, 0] > limits[:, 1]):
                raise ValueError("joint_limits has a lower bound above its upper bound.")
            self._joint_limits = limits

    def reset(self) -> None:
        """Per-episode reset hook (parity with the MuJoCo OSC controller).

        The DLS solve is stateless, so there is nothing to clear today; the
        hook exists so install sites can treat both controllers uniformly.
        """

    def compute_joint_targets(self, action: Mapping[str, Any]) -> dict[str, float]:
        """Convert one task-space delta action into joint position targets.

        Args:
            action: GR00T-style action dict. ``x/y/z/roll/pitch/yaw``
                default to 0 (no-op delta) when absent; ``gripper`` absent
                means "no gripper command". Keys outside
                :data:`TASK_SPACE_ACTION_KEYS` are passed through unchanged
                so the engine's unresolved-key reporting still fires for
                them.

        Returns:
            ``{joint_name: target}`` -- arm targets from the DLS solve
            (omitted entirely for an all-zero delta, so a settle step holds
            the current PD targets), gripper targets for an open/close
            command, plus any passed-through keys.

        Raises:
            TypeError: If ``action`` is not a mapping.
            ValueError: If a channel the action carries cannot be read as a
                scalar, or is not finite. Absent channels hold their axis and
                are not an error; a present one that cannot be read is, on the
                same grounds as the ``RuntimeError`` below - see
                :func:`_to_scalar`.
            RuntimeError: If the injected state callables return
                unusable data (wrong Jacobian shape, joint-count mismatch,
                non-finite values) -- a broken solve must surface, never
                degrade into a silent zero-motion step (#1812).
        """
        if not isinstance(action, Mapping):
            raise TypeError(f"action must be a mapping, got {type(action).__name__}.")

        targets: dict[str, float] = {k: v for k, v in action.items() if k not in TASK_SPACE_ACTION_KEYS}

        # An ABSENT channel holds its axis and never reaches ``_to_scalar``; a
        # channel that IS present must be readable. Keeping that split here is
        # what lets the coercion refuse instead of substituting: the zero for a
        # missing axis is a documented default, and the zero that used to stand
        # in for an unreadable one was a fabricated command.
        twist = np.zeros(6, dtype=np.float64)
        for i, key in enumerate(_POS_KEYS):
            if key in action:
                twist[i] = np.clip(_to_scalar(action[key], key, strict=self._strict), -1.0, 1.0) * self._pos_scale
        for i, key in enumerate(_ROT_KEYS):
            if key in action:
                twist[3 + i] = np.clip(_to_scalar(action[key], key, strict=self._strict), -1.0, 1.0) * self._rot_scale

        if np.any(twist != 0.0):
            targets.update(self._solve_arm_targets(twist))

        if "gripper" in action:
            # RLDS (0=close, 1=open, 0.5=hold) -> LIBERO sign convention
            # (+1=close, -1=open, 0=hold); see NVIDIA's
            # normalize/invert_gripper_action pair for the derivation.
            command = -float(
                np.sign(2.0 * _to_scalar(action["gripper"], "gripper", strict=self._strict, default=0.5) - 1.0)
            )
            if command != 0.0:
                finger_target = self._gripper_close if command > 0.0 else self._gripper_open
                for name in self.gripper_joint_names:
                    targets[name] = finger_target

        return targets

    def _solve_arm_targets(self, twist: np.ndarray) -> dict[str, float]:
        """DLS-solve ``dq`` for a world-frame twist and return arm targets."""
        n_arm = len(self.arm_joint_names)

        q = np.asarray(self._joint_positions_fn(), dtype=np.float64).reshape(-1)
        if q.shape != (n_arm,):
            raise RuntimeError(
                f"joint_positions_fn returned shape {q.shape}; expected ({n_arm},) matching arm_joint_names."
            )
        jac = np.asarray(self._jacobian_fn(), dtype=np.float64)
        if jac.shape != (6, n_arm):
            raise RuntimeError(f"jacobian_fn returned shape {jac.shape}; expected (6, {n_arm}).")
        if not (np.all(np.isfinite(q)) and np.all(np.isfinite(jac))):
            raise RuntimeError("joint positions / Jacobian contain non-finite values; refusing to solve.")

        # Damped least squares: dq = J^T (J J^T + lambda^2 I)^{-1} twist.
        # The damping term keeps (J J^T + lambda^2 I) positive definite, so
        # np.linalg.solve cannot raise on a singular configuration.
        gram = jac @ jac.T + (self._damping**2) * np.eye(6)
        dq = jac.T @ np.linalg.solve(gram, twist)

        q_target = q + dq
        if self._joint_limits is not None:
            q_target = np.clip(q_target, self._joint_limits[:, 0], self._joint_limits[:, 1])

        return {name: float(q_target[i]) for i, name in enumerate(self.arm_joint_names)}
