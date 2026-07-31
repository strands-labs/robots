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


def _to_scalar(value: Any, default: float = 0.0) -> float:
    """Coerce a GR00T action channel to a scalar float.

    GR00T-LIBERO packs every action channel list-shaped (2-element list /
    ndarray) to match the training-data shape -- the same convention
    ``_LiberoOSCController._to_scalar`` handles on the MuJoCo path (#168).

    * Scalar input -> ``float(value)``
    * Non-empty list / tuple / ndarray -> ``float(value[0])``
    * Everything else (None, empty list, dict, ...) -> ``default`` after a
      WARNING log.
    """
    try:
        if isinstance(value, (list, tuple, np.ndarray)) and len(value) > 0:
            return float(value[0])
        return float(value)
    except (TypeError, ValueError, IndexError) as e:
        logger.warning(
            "IsaacDeltaEEFController: could not coerce action value %r to float (%s); using %s for this step",
            value,
            e,
            default,
        )
        return default


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
        targets are clipped into them. ``None`` disables clipping.
    pos_scale, rot_scale : float
        Metres / radians of task-space delta for a saturated (+/-1) input
        channel. Defaults match robosuite ``OSC_POSE`` (#168).
    damping : float
        DLS damping ``lambda`` (> 0). Keeps the solve bounded near
        singularities.
    gripper_open, gripper_close : float
        Joint position target written to every gripper joint for an
        open / close command. Defaults match the Franka USD's 0..0.04 m
        prismatic fingers.

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
    ) -> None:
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
        if not (float(pos_scale) > 0.0 and float(rot_scale) > 0.0):
            raise ValueError(f"pos_scale/rot_scale must be > 0, got {pos_scale!r}/{rot_scale!r}.")
        if not float(damping) > 0.0:
            raise ValueError(f"damping must be > 0, got {damping!r}.")

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
            RuntimeError: If the injected state callables return
                unusable data (wrong Jacobian shape, joint-count mismatch,
                non-finite values) -- a broken solve must surface, never
                degrade into a silent zero-motion step (#1812).
        """
        if not isinstance(action, Mapping):
            raise TypeError(f"action must be a mapping, got {type(action).__name__}.")

        targets: dict[str, float] = {k: v for k, v in action.items() if k not in TASK_SPACE_ACTION_KEYS}

        twist = np.zeros(6, dtype=np.float64)
        for i, key in enumerate(_POS_KEYS):
            twist[i] = np.clip(_to_scalar(action.get(key, 0.0)), -1.0, 1.0) * self._pos_scale
        for i, key in enumerate(_ROT_KEYS):
            twist[3 + i] = np.clip(_to_scalar(action.get(key, 0.0)), -1.0, 1.0) * self._rot_scale

        if np.any(twist != 0.0):
            targets.update(self._solve_arm_targets(twist))

        if "gripper" in action:
            # RLDS (0=close, 1=open, 0.5=hold) -> LIBERO sign convention
            # (+1=close, -1=open, 0=hold); see the MuJoCo controller's
            # derivation from NVIDIA's normalize/invert_gripper_action pair.
            command = -float(np.sign(2.0 * _to_scalar(action.get("gripper"), 0.5) - 1.0))
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
