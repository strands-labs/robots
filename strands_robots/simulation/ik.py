"""Shared inverse-kinematics utilities: mink IK bridge + EE-frame discovery.

The single home for the generic differential-IK solver wrapper
(:class:`MinkIKBridge`) and the end-effector frame auto-discovery heuristic
(:func:`discover_ee_frame`) that were previously duplicated per policy provider
(:mod:`strands_robots.policies.cosmos3.sim_ik` and
:mod:`strands_robots.policies.vera.sim_ik` each carried a copy of the bridge;
the discovery heuristic lived in :mod:`strands_robots.policies.vera.ee_frame`).
Those modules now re-export from here, keeping their provider-specific decode
glue (action-chunk semantics) in place - a change to one model's action
semantics still cannot break the other, because only the model-agnostic solver
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
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np

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
        position_cost: Cartesian position task weight.
        orientation_cost: Cartesian orientation task weight (``0.0`` yields a
            position-only solve - important for arms with fewer than 6 DOF).
        posture_cost: Posture (joint-regularizer) task weight - keeps the solve
            near the current configuration so it stays smooth and avoids
            flipping between IK branches.
        solver: ``qpsolvers`` backend name passed to ``mink.solve_ik``.
            ``None`` (default) auto-selects an installed backend - preferring
            ``"daqp"`` (what ``mink`` pins), then ``"quadprog"``, then whatever
            ``qpsolvers.available_solvers`` reports. Pass an explicit name to
            force one.
        damping: Levenberg-Marquardt damping for ``solve_ik``.
        max_iters: Max differential-IK iterations per target pose.
        dt: Integration timestep for each IK iteration (s).
        pos_threshold: Convergence threshold on position error (m).
        ori_threshold: Convergence threshold on orientation error (rad).
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
        """
        self._configuration.update(np.asarray(qpos, dtype=np.float64))
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
        """
        mink = self._mink
        q = np.asarray(q_init, dtype=np.float64).copy()
        self._configuration.update(q)
        self._posture_task.set_target(q)

        target = mink.SE3.from_matrix(np.asarray(target_pose, dtype=np.float64))
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
# Heuristic (first match wins), scoped to the robot's ``namespace``:
#   1. A **site** whose name hints at the tool point (``attachment_site`` /
#      ``grasp`` / ``tcp`` / ...) - the conventional MuJoCo IK targets (e.g.
#      menagerie Panda ships ``attachment_site``). Sites are preferred: they
#      are the intended TCP.
#   2. A **body** whose name hints at the hand/tool (``hand`` / ``gripper`` /
#      ``wrist`` / ...).
#   3. The **leaf body** of the robot's kinematic chain (the descendant of the
#      robot's joints with no child body) - the last link, where a tool mounts.
# --------------------------------------------------------------------------

_SITE_HINTS = ("attachment_site", "attachment", "grasp", "pinch", "tcp", "ee_site", "ee", "flange")
_BODY_HINTS = ("hand", "gripper", "tool", "tcp", "ee", "wrist", "flange", "end_effector", "eef")


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


def discover_ee_frame(model: Any, namespace: str | None = None) -> tuple[str, str] | None:
    """Discover an IK end-effector frame ``(name, type)`` for a robot.

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

    # 1) Prefer a TCP-like SITE.
    sites = [(i, n) for i, n in _names_of(model, _site_obj()) if _scoped(n, namespace)]
    for hint in _SITE_HINTS:
        for _i, name in sites:
            if hint in _basename(name, namespace).lower():
                logger.info("ee-frame: site %r (hint %r)", name, hint)
                return name, "site"

    # 2) A hand/tool BODY.
    bodies = [(i, n) for i, n in _names_of(model, _body_obj()) if _scoped(n, namespace)]
    for hint in _BODY_HINTS:
        for _i, name in bodies:
            if hint in _basename(name, namespace).lower():
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
