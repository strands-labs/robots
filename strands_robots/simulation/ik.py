"""Shared inverse-kinematics utilities: mink IK bridge + EE-frame discovery.

The single home for the generic differential-IK solver wrapper
(:class:`MinkIKBridge`) and the end-effector frame auto-discovery heuristic
(:func:`discover_ee_frame`) that were previously duplicated per policy provider
(:mod:`strands_robots.policies.cosmos3.sim_ik` carried its own copy of the
bridge). That module now re-exports from here, keeping its provider-specific
decode glue (action-chunk semantics) in place - a change to one model's action
semantics cannot reach another provider, because only the model-agnostic solver
wrapper is shared.

:class:`MinkIKBridge` wraps `mink <https://github.com/kevinzakka/mink>`_, a
differential-IK library that works directly on the same ``mujoco.MjModel`` (no
URDF or second kinematics engine). Per target pose it runs a damped
least-squares ``solve_ik`` with a Cartesian :class:`mink.FrameTask` on the
end-effector frame plus a :class:`mink.PostureTask` regularizer, integrating
the joint velocity over the control timestep.

``mink`` + ``mujoco`` + ``qpsolvers`` are imported lazily so importing this
module in the light base env (no sim extras) stays free; a missing stack
raises an actionable install error rather than a silent default (AGENTS.md key
convention: no silent defaults on error). Provider subclasses customize the
install hint / no-backend message via the ``_INSTALL_HINT`` / ``_NO_BACKEND_MSG``
class attributes so the error a user sees names the extra they actually need.
"""

from __future__ import annotations

import logging
import math
import re
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

from ..utils import (
    finite_number_error,
    finite_vector_error,
    pose_vector_error,
    positive_count_error,
    positive_finite_number_error,
    refusal_repr,
)

if TYPE_CHECKING:
    import mujoco

logger = logging.getLogger(__name__)

_PREFERRED_QP_SOLVERS = ("daqp", "quadprog", "osqp", "proxqp", "cvxopt", "scs")

_DEFAULT_INSTALL_HINT = (
    "The mink IK bridge needs 'mink' + 'mujoco' + a qpsolvers backend, which "
    "were not importable. Install the sim extra:\n"
    "  uv pip install 'strands-robots[sim-mujoco]'\n"
    "This pulls mink (differential IK on the MuJoCo model), mujoco and a QP "
    "backend, turning Cartesian end-effector targets into joint configurations "
    "the arm can track."
)

_DEFAULT_NO_BACKEND_MSG = (
    "No qpsolvers backend is installed; the mink IK bridge needs one "
    "(e.g. 'daqp' or 'quadprog'). Install the sim extra: "
    "uv pip install 'strands-robots[sim-mujoco]'."
)


def _homogeneous_pose_error(method: str, param_name: str, pose: np.ndarray) -> str | None:
    """Return an error message if ``pose`` is not a ``(4, 4)`` matrix.

    Local because the shared pose domains are flat-vector shaped:
    :func:`~strands_robots.utils.pose_vector_error` counts ``expected_len``
    components, and a ``(2, 8)`` array flattens to the sixteen a ``(4, 4)`` has,
    so a length check accepts a matrix no homogeneous transform could be. The
    shape is what the consumer needs - ``mink.SE3.from_matrix`` slices
    ``matrix[:3, :3]`` and ``matrix[:3, 3]`` - so it is checked here and the
    component domain still owns the values.

    Args:
        method: The public method being called, for the message.
        param_name: The parameter name, so the caller is told which argument.
        pose: The already-``asarray``-ed candidate pose.

    Returns:
        ``None`` when ``pose`` is ``(4, 4)``; otherwise a message naming the
        method, the parameter, the required shape and the shape supplied.
    """
    if pose.shape != (4, 4):
        return f"{method}: {param_name!r} must be a (4, 4) homogeneous pose matrix, got shape {pose.shape}"
    return None


def _damping_error(value: Any, context: str) -> str | None:
    """Error text when ``value`` is not a usable Levenberg-Marquardt damping.

    ``damping`` is added to the diagonal of the QP's cost matrix by
    ``mink.solve_ik``. It must be a finite number ``>= 0``: ``0.0`` is the
    undamped least-squares solve and is legal, a negative value makes the
    matrix indefinite (``qpsolvers`` refuses it mid-solve with "matrix P is not
    positive definite"), and a non-finite value poisons the matrix so the solve
    returns an all-NaN configuration rather than raising.

    The finiteness half is the shared
    :func:`~strands_robots.utils.finite_number_error` domain, so a bool, a
    string and a non-finite number are refused with the library's wording; only
    the floor is stated here, because it is the QP's rule rather than a
    property of the number.

    Args:
        value: The caller-supplied damping.
        context: The class name to name in the message.

    Returns:
        The error text, or ``None`` when ``value`` can be honored.
    """
    if text := finite_number_error(value, "damping", context):
        return text
    if float(value) < 0.0:
        return (
            f"{context}: damping must be >= 0 (0.0 is the undamped solve); a negative value makes the "
            f"QP cost matrix indefinite, got {refusal_repr(value)}."
        )
    return None


def resolve_qp_solver(
    requested: str | None,
    *,
    install_hint: str = _DEFAULT_INSTALL_HINT,
    no_backend_msg: str = _DEFAULT_NO_BACKEND_MSG,
) -> str:
    """Pick an installed ``qpsolvers`` backend for ``mink.solve_ik``.

    ``mink`` defaults to (and pins) ``daqp``, but environments commonly ship
    only ``quadprog``. Auto-selecting from ``qpsolvers.available_solvers``
    (preferring daqp, then quadprog) keeps the IK bridge working everywhere
    without forcing an extra QP dependency. An explicit ``requested`` name is
    honoured when installed; if it is not, we fail with an actionable error
    that lists what *is* available (no silent fallback to a solver the caller
    did not ask for, but also no opaque KeyError deep in qpsolvers).

    Args:
        requested: Explicit backend name to force, or ``None`` to auto-select.
        install_hint: Message raised when ``qpsolvers`` itself is missing.
        no_backend_msg: Message raised when ``qpsolvers`` reports zero backends.

    Returns:
        The resolved backend name.

    Raises:
        ImportError: ``qpsolvers`` is not importable (with ``install_hint``).
        RuntimeError: No QP backend is installed (with ``no_backend_msg``).
        ValueError: ``requested`` names a backend that is not installed.
    """
    try:
        from qpsolvers import available_solvers
    except ImportError as e:
        raise ImportError(install_hint) from e
    available = list(available_solvers)
    if not available:
        raise RuntimeError(no_backend_msg)
    if requested is not None:
        if requested not in available:
            raise ValueError(
                f"Requested qpsolvers backend {requested!r} is not installed. "
                f"Available: {available}. Install it (e.g. pip install "
                f"'qpsolvers[{requested}]') or pass an available solver / None."
            )
        return requested
    for name in _PREFERRED_QP_SOLVERS:
        if name in available:
            return name
    return available[0]


class MinkIKBridge:
    """Differential-IK bridge from EE poses to MuJoCo joint configurations.

    Args:
        model: The ``mujoco.MjModel`` for the arm being controlled.
        ee_frame_name: Name of the end-effector frame (a body or site) the
            Cartesian task tracks (e.g. ``"hand"`` for a Franka/Panda).
        ee_frame_type: ``"body"`` (default), ``"site"``, or ``"geom"`` - the
            ``mink.FrameTask`` frame type for ``ee_frame_name``.
        position_cost: Cartesian position task weight. A finite number;
            ``mink`` refuses a negative one by name at task construction, and a
            non-finite one poisoned the QP so the solve returned an all-NaN
            configuration.
        orientation_cost: Cartesian orientation task weight (``0.0`` yields a
            position-only solve - important for arms with fewer than 6 DOF).
            A finite number, on the same terms as ``position_cost``.
        posture_cost: Posture (joint-regularizer) task weight - keeps the solve
            near the current configuration so it stays smooth and avoids
            flipping between IK branches. A finite number, on the same terms as
            ``position_cost``.
        solver: ``qpsolvers`` backend name passed to ``mink.solve_ik``.
            ``None`` (default) auto-selects an installed backend - preferring
            ``"daqp"`` (what ``mink`` pins), then ``"quadprog"``, then whatever
            ``qpsolvers.available_solvers`` reports. Pass an explicit name to
            force one.
        damping: Levenberg-Marquardt damping for ``solve_ik``. A finite number
            ``>= 0`` (``0.0`` is the undamped solve).
        max_iters: Max differential-IK iterations per target pose. A positive
            whole number: it bounds the ``range`` :meth:`solve` iterates over,
            so ``0`` ran the solver zero times and handed back ``q_init``
            unchanged as though it had solved.
        dt: Integration timestep for each IK iteration (s). A positive finite
            number: ``0.0`` and a non-finite value both produced an all-NaN
            configuration.
        pos_threshold: Convergence threshold on position error (m). A finite
            number: an infinite threshold is met by every residual, so it made
            the *first* iteration count as converged. A threshold no residual
            can reach (zero or negative) is left accepted - it means "never
            break early", which runs the full ``max_iters`` budget.
        ori_threshold: Convergence threshold on orientation error (rad), on the
            same terms as ``pos_threshold``.
        commanded_dofs: Velocity-space (``nv``) indices of the ONLY degrees of
            freedom the caller can command, or ``None`` (default) to leave the
            whole model free. ``mink`` optimizes over every DOF in ``model``,
            so an unconstrained solve is free to satisfy the Cartesian task by
            moving a DOF the caller will never send - a floating base, a second
            robot sharing the world model, a gripper the caller holds - and
            :meth:`solve` then returns, and :meth:`ee_pose` then scores, a
            configuration that is never realized. Restricting the solve keeps
            the returned configuration (and therefore any residual measured on
            it) inside what the caller can actually reach. ``None`` is correct
            only when the caller drives every DOF the frame depends on.

    Raises:
        ImportError: If ``mink``/``mujoco`` are not importable (with an
            actionable install hint).
        ValueError: If any numeric knob cannot be honored - a ``max_iters``
            that is not a positive whole number, a ``dt`` that is not a
            positive finite number, a non-finite convergence threshold, a
            ``damping`` that is not a finite number ``>= 0``, or a non-finite
            task cost. Each is applied rather than forwarded, so an
            unusable value produced a plausible-looking configuration instead
            of an error; the refusal happens before the QP backend is resolved
            and before any task is built.
        ValueError: If ``commanded_dofs`` is empty or names an index outside
            ``range(model.nv)`` - a solve that may move nothing, or a mask
            built against a different model, is a caller bug rather than a
            configuration to silently widen.
    """

    # Provider subclasses override these so failures name the extra the user
    # actually needs (e.g. cosmos3-sim vs sim-mujoco).
    _INSTALL_HINT: ClassVar[str] = _DEFAULT_INSTALL_HINT
    _NO_BACKEND_MSG: ClassVar[str] = _DEFAULT_NO_BACKEND_MSG
    _LOG_LABEL: ClassVar[str] = "MinkIKBridge"

    def __init__(
        self,
        model: mujoco.MjModel,
        ee_frame_name: str,
        ee_frame_type: str = "body",
        *,
        position_cost: float = 1.0,
        orientation_cost: float = 1.0,
        posture_cost: float = 1e-2,
        solver: str | None = None,
        damping: float = 1e-3,
        max_iters: int = 20,
        dt: float = 1e-2,
        pos_threshold: float = 1e-3,
        ori_threshold: float = 1e-3,
        commanded_dofs: Sequence[int] | None = None,
    ) -> None:
        try:
            import mink
        except ImportError as e:
            raise ImportError(self._INSTALL_HINT) from e

        # Every numeric knob is checked here, before the QP backend is resolved,
        # before the DOF mask reads the model, before the Configuration and the
        # two tasks are constructed and before the ready log line. Each one is
        # *applied* rather than forwarded - the iteration count bounds a
        # ``range``, the timestep integrates a velocity, the damping and the
        # costs weight a QP - so an unusable value produces a plausible-looking
        # configuration instead of an error. See the per-parameter notes in the
        # class docstring for what each unchecked value did.
        label = type(self).__name__
        for _cost_param, _cost in (
            ("position_cost", position_cost),
            ("orientation_cost", orientation_cost),
            ("posture_cost", posture_cost),
        ):
            # ``mink`` already refuses a negative cost by name at task
            # construction, so only the kind is checked here; ``0.0`` stays
            # legal because an ``orientation_cost`` of zero is the documented
            # position-only solve.
            if text := finite_number_error(_cost, _cost_param, label):
                raise ValueError(text)
        if text := _damping_error(damping, label):
            raise ValueError(text)
        if text := positive_count_error(max_iters, "max_iters", label):
            raise ValueError(text)
        if text := positive_finite_number_error(dt, "dt", label):
            raise ValueError(text)
        for _threshold_param, _threshold in (("pos_threshold", pos_threshold), ("ori_threshold", ori_threshold)):
            # Finiteness only, not a positive floor: a threshold no residual can
            # meet (zero, or a negative distance) means "never break early", so
            # the loop runs its whole budget - the conservative direction, and
            # the idiom the solve-loop suites use to exercise it. An *infinite*
            # threshold is the damaging one, because every residual satisfies it
            # and the first iteration counts as converged.
            if text := finite_number_error(_threshold, _threshold_param, label):
                raise ValueError(text)

        self._mink = mink
        self.model = model
        self.ee_frame_name = ee_frame_name
        self.ee_frame_type = ee_frame_type
        self.solver = resolve_qp_solver(solver, install_hint=self._INSTALL_HINT, no_backend_msg=self._NO_BACKEND_MSG)
        self.damping = damping
        self.max_iters = max_iters
        self.dt = dt
        self.pos_threshold = pos_threshold
        self.ori_threshold = ori_threshold

        # Read model.nv only when a mask is asked for: the unrestricted path must
        # touch nothing new, so a caller solving on a minimal model object is
        # unaffected by this parameter existing.
        self._dof_mask = None if commanded_dofs is None else self._build_dof_mask(int(model.nv), commanded_dofs)

        self._configuration = mink.Configuration(model)
        self._frame_task = mink.FrameTask(
            frame_name=ee_frame_name,
            frame_type=ee_frame_type,
            position_cost=position_cost,
            orientation_cost=orientation_cost,
            lm_damping=1.0,
        )
        self._posture_task = mink.PostureTask(model=model, cost=posture_cost)
        self._tasks = [self._frame_task, self._posture_task]
        logger.info(
            "%s ready [ee=%s/%s solver=%s nq=%d]",
            self._LOG_LABEL,
            ee_frame_type,
            ee_frame_name,
            self.solver,
            model.nq,
        )

    @staticmethod
    def _build_dof_mask(nv: int, commanded_dofs: Sequence[int] | None) -> np.ndarray | None:
        """Boolean ``nv`` mask of the commandable DOFs, or ``None`` for all.

        Args:
            nv: The model's velocity-space dimension.
            commanded_dofs: Indices to allow, or ``None`` to allow everything.

        Returns:
            A length-``nv`` boolean mask, or ``None`` when the whole model is
            free (which keeps the unrestricted path allocation-free).

        Raises:
            ValueError: ``commanded_dofs`` is empty, holds a non-integer (a
                ``bool`` included - it is an ``int`` subclass that would act as
                index 0 or 1), or names an index outside ``range(nv)``.
        """
        if commanded_dofs is None:
            return None
        indices = list(commanded_dofs)
        if not indices:
            raise ValueError(
                "commanded_dofs is empty, so the solve could not move any degree of freedom. "
                "Pass the indices the caller commands, or None to leave the whole model free."
            )
        mask = np.zeros(nv, dtype=bool)
        for index in indices:
            if isinstance(index, bool) or not isinstance(index, (int, np.integer)):
                raise ValueError(f"commanded_dofs must hold integer velocity-space indices; got {index!r}.")
            if not 0 <= int(index) < nv:
                raise ValueError(
                    f"commanded_dofs index {int(index)} is outside range(model.nv) = range({nv}). "
                    "The mask must be built against the same model this bridge solves on."
                )
            mask[int(index)] = True
        return mask

    def ee_pose(self, qpos: np.ndarray) -> np.ndarray:
        """Forward kinematics: ``(4, 4)`` EE pose at a joint configuration.

        Args:
            qpos: Joint configuration of length ``model.nq``.

        Returns:
            The end-effector frame's absolute ``(4, 4)`` homogeneous pose
            (``float64``).

        Raises:
            ValueError: If ``qpos`` is not ``model.nq`` finite numbers. The
                length is the half this method documents and
                ``mink.Configuration.update`` writes into: a wrong-width array
                reached the numpy assignment and raised ``could not broadcast
                input array from shape (6,) into shape (9,)``, naming neither
                the parameter nor this class. It arrives from a caller through
                :meth:`tracking_error`'s ``qpos_traj``, whose every row is read
                back through here. The configuration
                is applied rather than forwarded - it is the state forward
                kinematics is evaluated at - so a single ``nan`` or ``inf``
                reaches the returned pose, which comes back with a non-finite
                translation under a successful return shaped exactly like a
                reachable pose. Every consumer then inherits it: a
                ``norm(ee[:3, 3] - target)`` residual is ``nan``, and
                ``nan <= threshold`` is ``False``, so a convergence test never
                fires; a closed loop that composes a delta onto this pose and
                solves for it is refused by :meth:`solve` naming
                ``target_pose``, an argument the caller never supplied. Checked
                before the configuration is updated, so a refused call leaves
                the bridge as it was.
        """
        q = np.asarray(qpos, dtype=np.float64)
        # Checked before the configuration is updated, so a refused call mutates
        # nothing. ``solve`` holds its own seed to this same domain; this is the
        # third public method reading a joint configuration and was the only one
        # that did not, which is why a bad value surfaced at the *next* solve
        # under another argument's name.
        if text := pose_vector_error("ee_pose", "qpos", q, self.model.nq):
            raise ValueError(text)
        self._configuration.update(q)
        transform = self._configuration.get_transform_frame_to_world(self.ee_frame_name, self.ee_frame_type)
        return np.asarray(transform.as_matrix(), dtype=np.float64)

    def solve(self, target_pose: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Solve IK for a single Cartesian target from a seed configuration.

        Args:
            target_pose: Desired EE ``(4, 4)`` homogeneous pose.
            q_init: Seed joint configuration (length ``model.nq``); the solve is
                warm-started here and the posture task regularizes toward it.

        Returns:
            The solved joint configuration (length ``model.nq``, ``float64``).
            When ``commanded_dofs`` was given, every DOF outside it holds its
            ``q_init`` value exactly, so the caller can realize the answer.

        Raises:
            ValueError: If ``target_pose`` is not a ``(4, 4)`` matrix, or
                ``q_init`` is not ``model.nq`` long. Both shapes are documented
                above and neither was checked: a wrong-shaped pose reached
                ``mink.SE3.from_matrix``'s bare ``assert``, so the caller got an
                ``AssertionError`` carrying *no message at all* - and under
                ``python -O``, where that assertion is stripped, the same call
                raised ``IndexError`` or ``TypeError`` instead, so the exception
                type depended on an interpreter flag. A wrong-length seed
                reached the numpy assignment and raised an anonymous broadcast
                message. None of the three is the ``ValueError`` channel this
                method documents. :meth:`solve_trajectory` already refused a
                wrong-shaped ``poses`` batch by name and then delegated here per
                waypoint, so one wrong pose was reported two different ways
                depending on the entry point; checking here settles both,
                because that method solves through this one.
            ValueError: If ``target_pose`` or ``q_init`` holds a non-finite
                value. Both are read straight into the solver - the pose becomes
                the frame task's target and the seed becomes the configuration
                the QP warm-starts from - so a single ``nan`` or ``inf``
                propagates through every iteration and *each* joint of the
                returned configuration comes back non-finite, under a successful
                return that is shaped exactly like a converged solve. Checked
                before the configuration is updated, so a refused solve leaves
                the bridge as it was.
        """
        mink = self._mink
        pose = np.asarray(target_pose, dtype=np.float64)
        q = np.asarray(q_init, dtype=np.float64).copy()
        # Both arrays are checked before the configuration is updated and before
        # the frame target is set, so a refused solve mutates nothing. The pose
        # is flattened for the check because ``finite_vector_error`` reads a
        # 2-D argument's *rows* as its elements and would refuse a clean
        # ``(4, 4)``; the seed is already the 1-D vector that domain expects.
        if text := _homogeneous_pose_error("solve", "target_pose", pose):
            raise ValueError(text)
        if text := finite_vector_error("solve", "target_pose", pose.ravel()):
            raise ValueError(text)
        if text := pose_vector_error("solve", "q_init", q, self.model.nq):
            raise ValueError(text)
        self._configuration.update(q)
        self._posture_task.set_target(q)

        target = mink.SE3.from_matrix(pose)
        self._frame_task.set_target(target)

        for _ in range(self.max_iters):
            velocity = mink.solve_ik(self._configuration, self._tasks, self.dt, self.solver, self.damping)
            if self._dof_mask is not None:
                # Project the step onto the commandable subspace before
                # integrating. Zeroing here rather than post-filtering the
                # solution keeps every later iteration honest: the next
                # solve_ik sees the error that actually remains, so the loop
                # converges to the best configuration the caller can command
                # instead of one it can only report.
                velocity = np.asarray(velocity, dtype=np.float64).copy()
                velocity[~self._dof_mask] = 0.0
            self._configuration.integrate_inplace(velocity, self.dt)
            err = self._frame_task.compute_error(self._configuration)
            if np.linalg.norm(err[:3]) <= self.pos_threshold and np.linalg.norm(err[3:]) <= self.ori_threshold:
                break
        return np.asarray(self._configuration.q, dtype=np.float64).copy()

    def solve_trajectory(self, poses: np.ndarray, q_init: np.ndarray) -> np.ndarray:
        """Solve IK for an EE-pose trajectory, warm-starting each step.

        Args:
            poses: Absolute EE poses of shape ``[N, 4, 4]``.
            q_init: Seed joint configuration for the first pose; each subsequent
                solve warm-starts from the previous solution so the joint
                trajectory stays continuous.

        Returns:
            Joint configurations of shape ``[N, model.nq]`` (``float64``).
        """
        poses = np.asarray(poses, dtype=np.float64)
        if poses.ndim != 3 or poses.shape[1:] != (4, 4):
            raise ValueError(f"poses must be [N, 4, 4]; got {poses.shape}")
        q = np.asarray(q_init, dtype=np.float64).copy()
        out = []
        for pose in poses:
            q = self.solve(pose, q)
            out.append(q.copy())
        return np.stack(out) if out else np.empty((0, self.model.nq), dtype=np.float64)

    def tracking_error(self, poses: np.ndarray, qpos_traj: np.ndarray) -> dict[str, float]:
        """Cartesian position tracking error between targets and solved poses.

        Args:
            poses: Target EE poses ``[N, 4, 4]``.
            qpos_traj: Solved joint configs ``[N, nq]`` (from
                :meth:`solve_trajectory`).

        Returns:
            ``{"mean_mm": float, "max_mm": float}`` - mean / max Euclidean
            position error in millimetres across the trajectory.

        Raises:
            ValueError: If ``poses`` and ``qpos_traj`` differ in length (the
                ``strict=True`` zip), or a row of ``qpos_traj`` is not
                ``model.nq`` finite numbers - :meth:`ee_pose` owns that refusal,
                and every row is read back through it.
        """
        poses = np.asarray(poses, dtype=np.float32)
        errs = []
        for target, q in zip(poses, np.asarray(qpos_traj), strict=True):
            achieved = self.ee_pose(q)
            errs.append(float(np.linalg.norm(achieved[:3, 3] - target[:3, 3])))
        errs_arr = np.asarray(errs, dtype=np.float32)
        if errs_arr.size == 0:
            return {"mean_mm": 0.0, "max_mm": 0.0}
        return {"mean_mm": float(errs_arr.mean() * 1000.0), "max_mm": float(errs_arr.max() * 1000.0)}


# --------------------------------------------------------------------------
# End-effector frame auto-discovery
#
# Driving a MuJoCo arm in Cartesian space needs an IK target frame (the
# body/site the Cartesian task tracks). The robot registry does NOT record an
# ee-frame, so we discover it from the compiled ``mujoco.MjModel`` with a
# robust, namespace-aware heuristic - making Cartesian control zero-config.
#
# Hints match name *components*, not bare substrings (see
# :func:`hint_matches_name`),
# so the short hints cannot fire inside an unrelated word - a ``knee`` or a
# ``wheel`` is not an end-effector just because ``ee`` occurs in its name.
#
# Heuristic (first match wins), scoped to the robot's ``namespace``:
#   1. A **site** whose name hints at the tool point (``attachment_site`` /
#      ``grasp`` / ``tcp`` / ...) - the conventional MuJoCo IK targets (e.g.
#      menagerie Panda ships ``attachment_site``). Sites are preferred: they
#      are the intended TCP.
#   2. A **body** whose name hints at the hand/tool (``hand`` / ``gripper`` /
#      ``wrist`` / ...).
#   3. The **leaf body** of the robot's kinematic chain (the descendant of the
#      robot's joints with no child body) - the last link, where a tool mounts.
#
# Rung 1 searches the end-effector vocabulary of rung 2 as well, after its own
# TCP-specific names. The two rungs describe the same physical part of the robot
# in different words, so a token that identifies the end effector as a body
# identifies it as a site too - and a site is the more precise frame, because it
# is placed at the tool point while the body origin sits at the link's mount.
# Searching sites for TCP names only made a model that publishes its tool point
# as a site lose to the link of the same name: ``so101`` names both ``gripper``
# and resolved to the body, 98 mm behind its own fingertips.
# --------------------------------------------------------------------------

_SITE_HINTS = ("attachment_site", "attachment", "grasp", "pinch", "tcp", "ee_site", "ee", "flange")
_BODY_HINTS = ("hand", "gripper", "tool", "tcp", "ee", "wrist", "flange", "end_effector", "eef")

# Hint words for the gripper/EEF mount every backend's ``list_bodies``
# advertises in its ``gripper_body`` field. Shared rather than per-backend:
# ``gripper_body`` is one question about one robot, so a body that is the mount
# on one backend has to be the mount on every other. ``jaw`` is here because
# the SO-100 family names its gripper bodies ``Fixed_Jaw`` / ``Moving_Jaw`` -
# see :data:`~strands_robots.simulation.motion_primitives_base._GRIPPER_HINTS`,
# which reads the same vocabulary to resolve the gripper *joint*.
#
# Distinct from :data:`_BODY_HINTS` above, which answers a different question:
# which frame IK should solve *to*. That one legitimately prefers a wrist or a
# flange over a jaw, so the two vocabularies are not interchangeable.
GRIPPER_BODY_HINTS = ("gripper", "hand", "jaw", "ee", "tool")
# Rung 1's search order: TCP-specific site names first, then the end-effector
# names rung 2 matches on bodies. Order-preserving dedupe, because the two
# tuples share tokens and the first occurrence is the one that decides.
_SITE_SEARCH_HINTS = tuple(dict.fromkeys((*_SITE_HINTS, *_BODY_HINTS)))

# Hints name *components* of an element name, so they are matched on word
# boundaries rather than as bare substrings. A bare-substring match makes the
# short hints ("ee", "eef", "tcp") fire inside unrelated words - "ee" occurs in
# "knee", "wheel" and "unitree" - which resolved a leg or a drive wheel as a
# robot's end-effector. Names are split into lowercase tokens on separators
# ("_", "-", "/", "."), on camelCase boundaries ("wristYawLeft") and between
# letters and digits ("tool0"), and a hint matches when its own tokens appear
# as a consecutive run. Multi-token hints ("attachment_site", "end_effector")
# therefore still match, and so does a hint that is one token of a longer name
# ("ee" in "ee_link").
_TOKEN_BOUNDARY = re.compile(r"[^a-z0-9]+|(?<=[a-z])(?=[0-9])|(?<=[0-9])(?=[a-z])")
_CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")


def _name_tokens(name: str) -> list[str]:
    """Split a MuJoCo element name into its lowercase word tokens.

    Args:
        name: A body/site name, or a hint to match against one.

    Returns:
        The name's tokens, lowercased, with empty tokens dropped -
        ``"left_knee_link"`` -> ``["left", "knee", "link"]``,
        ``"wristYawLeft"`` -> ``["wrist", "yaw", "left"]``,
        ``"tool0"`` -> ``["tool", "0"]``.
    """
    return [tok for tok in _TOKEN_BOUNDARY.split(_CAMEL_BOUNDARY.sub("_", name).lower()) if tok]


def hint_matches_name(hint: str, name: str) -> bool:
    """True when ``hint``'s tokens occur as a consecutive token run in ``name``.

    Matching a hint on word boundaries keeps a short hint from firing inside an
    unrelated word: ``"ee"`` matches ``"ee_link"`` and ``"gripper_ee"`` but not
    ``"left_knee_link"`` or ``"wheel_hub_back_link"``.

    This is the one matcher for every surface that answers "which element names
    an end-effector" - :func:`discover_ee_frame` here, and the ``gripper_body``
    each backend's ``list_bodies`` advertises.

    Sharing the matcher is not on its own enough for two surfaces to agree: they
    agree only where they also read the same vocabulary. So the two backends'
    ``list_bodies`` read one shared :data:`GRIPPER_BODY_HINTS`, because
    ``gripper_body`` is one question about one robot and a name cannot be that
    robot's mount on one backend and not on another. :func:`discover_ee_frame`
    keeps :data:`_BODY_HINTS` / :data:`_SITE_HINTS` of its own because it
    answers a different question - which frame to solve IK *to* - and may
    legitimately prefer a wrist to a jaw on the very same robot.

    Args:
        hint: An end-effector hint word, e.g. one of :data:`_SITE_HINTS` /
            :data:`_BODY_HINTS`. A multi-token hint (``"end_effector"``)
            matches as a phrase.
        name: The candidate element name, with any robot namespace already
            stripped - a namespace must not supply a match.

    Returns:
        Whether the hint names a component of ``name``.
    """
    hint_tokens = _name_tokens(hint)
    if not hint_tokens:
        return False
    name_tokens = _name_tokens(name)
    span = len(hint_tokens)
    return any(name_tokens[i : i + span] == hint_tokens for i in range(len(name_tokens) - span + 1))


def _names_of(model: Any, obj_type: Any) -> list[tuple[int, str]]:
    """Return ``[(id, name), ...]`` for all objects of ``obj_type`` in the model."""
    import mujoco as mj

    out: list[tuple[int, str]] = []
    n = {
        mj.mjtObj.mjOBJ_SITE: model.nsite,
        mj.mjtObj.mjOBJ_BODY: model.nbody,
    }[obj_type]
    for i in range(n):
        nm = mj.mj_id2name(model, obj_type, i)
        if nm:
            out.append((i, nm))
    return out


def _scoped(name: str, namespace: str | None) -> bool:
    """True when ``name`` belongs to the robot's namespace (or no namespace set)."""
    if not namespace:
        return True
    return name.startswith(namespace)


def _basename(name: str, namespace: str | None) -> str:
    """Strip the robot namespace prefix for hint matching."""
    if namespace and name.startswith(namespace):
        return name[len(namespace) :]
    return name


REACH_AXIS_MIN_M = 0.02
"""Horizontal EE offset below which the arm is reported as over its base, not along an axis."""


def reach_axis(offset: Sequence[float]) -> str | None:
    """The horizontal world axis an end-effector offset from the base lies along.

    ``"-Y"`` for an offset whose largest horizontal component is negative Y,
    and so on; ``None`` when the horizontal offset is under
    :data:`REACH_AXIS_MIN_M`, so a folded or vertical pose is not assigned an
    axis it does not have. This reads the CURRENT pose - it is where the arm
    is, not a fixed property of the model - which is exactly the fact an
    agent needs to turn "in front of the robot" into a world coordinate on
    the right side of the base.

    Args:
        offset: ``ee_pos - base_pos`` in world meters, ``(dx, dy, dz)``.
    """
    dx, dy = float(offset[0]), float(offset[1])
    if math.hypot(dx, dy) < REACH_AXIS_MIN_M:
        return None
    if abs(dx) >= abs(dy):
        return "+X" if dx > 0 else "-X"
    return "+Y" if dy > 0 else "-Y"


def reach_axis_label(axis: str | None) -> str:
    """One clause for :func:`reach_axis`'s answer, ready to sit in parentheses."""
    if axis is None:
        return "the arm is currently over its base"
    return f"the arm currently extends along {axis}"


def discover_ee_frame(model: Any, namespace: str | None = None) -> tuple[str, str] | None:
    """Discover an IK end-effector frame ``(name, type)`` for a robot.

    Resolution is first-match-wins over three rungs: a site whose name denotes
    the tool point or the end effector, else a body whose name denotes the end
    effector, else the leaf body of the namespace's kinematic chain. A site
    outranks a body even when both carry the same name, because a site is placed
    at the tool point while the body origin sits at the link's mount.

    Args:
        model: The compiled ``mujoco.MjModel`` (the shared world model).
        namespace: The robot's body/site namespace prefix (e.g. ``"panda/"``).
            Discovery is scoped to this so multi-robot worlds resolve correctly.

    Returns:
        ``(frame_name, frame_type)`` where ``frame_type`` is ``"site"`` or
        ``"body"``, names keep the namespace; or ``None`` if nothing resolves.
    """
    try:
        import mujoco  # noqa: F401  (lazy availability check)
    except ImportError:
        logger.debug("mujoco not importable; cannot auto-discover ee-frame")
        return None

    # 1) Prefer a SITE: a TCP-like name first, then an end-effector name.
    sites = [(i, n) for i, n in _names_of(model, _site_obj()) if _scoped(n, namespace)]
    for hint in _SITE_SEARCH_HINTS:
        for _i, name in sites:
            if hint_matches_name(hint, _basename(name, namespace)):
                logger.info("ee-frame: site %r (hint %r)", name, hint)
                return name, "site"

    # 2) A hand/tool BODY.
    bodies = [(i, n) for i, n in _names_of(model, _body_obj()) if _scoped(n, namespace)]
    for hint in _BODY_HINTS:
        for _i, name in bodies:
            if hint_matches_name(hint, _basename(name, namespace)):
                logger.info("ee-frame: body %r (hint %r)", name, hint)
                return name, "body"

    # 3) Leaf body of the namespace's kinematic chain.
    leaf = _leaf_body(model, namespace, bodies)
    if leaf is not None:
        logger.info("ee-frame: leaf body %r (kinematic chain tail)", leaf)
        return leaf, "body"

    logger.warning(
        "ee-frame: could not auto-discover an end-effector frame for namespace %r; pass an explicit frame.",
        namespace,
    )
    return None


def _site_obj() -> Any:
    import mujoco as mj

    return mj.mjtObj.mjOBJ_SITE


def _body_obj() -> Any:
    import mujoco as mj

    return mj.mjtObj.mjOBJ_BODY


def _leaf_body(model: Any, namespace: str | None, bodies: list[tuple[int, str]]) -> str | None:
    """The deepest body in the namespace's chain (a body with no in-namespace child).

    MuJoCo stores ``body_parentid``; the leaf (no children within the namespace)
    that sits furthest from the world is the tool-mount link. Among multiple
    leaves we pick the one with the greatest depth from the world body.
    """
    if not bodies:
        return None
    ids = {i for i, _ in bodies}
    id_to_name = {i: n for i, n in bodies}
    # Children count within the namespace.
    has_child = set()
    for i in ids:
        parent = int(model.body_parentid[i])
        if parent in ids:
            has_child.add(parent)
    leaves = [i for i in ids if i not in has_child]
    if not leaves:
        return None

    # Depth from world for tie-break (more joints between world and body = tip).
    def depth(bi: int) -> int:
        d, cur = 0, bi
        seen = set()
        while cur not in seen:
            seen.add(cur)
            p = int(model.body_parentid[cur])
            if p == cur or p == 0:
                break
            cur = p
            d += 1
        return d

    leaves.sort(key=depth, reverse=True)
    return id_to_name[leaves[0]]
