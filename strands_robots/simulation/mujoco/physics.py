"""Physics mixin - advanced MuJoCo physics introspection and manipulation.

Exposes the deep MuJoCo C API through clean Python methods:
- Raycasting (mj_ray)
- Jacobians (mj_jacBody, mj_jacSite, mj_jacGeom)
- Energy computation (mj_energyPos, mj_energyVel)
- External forces (mj_applyFT, xfrc_applied)
- Mass matrix (mj_fullM)
- State checkpointing (mj_getState, mj_setState)
- Inverse dynamics (mj_inverse)
- Body/joint introspection (poses, velocities, accelerations)
- Direct joint position/velocity control (qpos, qvel)
- Runtime model modification (mass, friction, color, size)
- Sensor readout (sensordata)
- Contact force analysis (mj_contactForce)
"""

import logging
import math
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.mujoco.backend import (
    _NO_WORLD_MSG,
    _ensure_mujoco,
    filter_mujoco_attach_noise,
)

logger = logging.getLogger(__name__)


def _full_mass_matrix(mj: Any, model: Any, data: Any) -> np.ndarray:
    """Return the dense ``nv x nv`` mass matrix M(q), robust to MuJoCo drift.

    ``mj_fullM`` changed its binding signature across MuJoCo releases:

    - MuJoCo >= 3.10: ``mj_fullM(model, data, dst)`` - the sparse buffer is
      read from ``data`` internally; ``dst`` must be writeable + C-contiguous.
    - Older builds: ``mj_fullM(model, dst, qM)`` where ``qM`` is the sparse
      inertia buffer, accepted either as a 1D array or a 2D ``[m, 1]`` column.

    Probe the modern signature first, then fall back to the legacy orders so
    the call works regardless of the installed MuJoCo version. ``dst`` is
    always allocated C-contiguous to satisfy the binding's buffer contract.

    Args:
        mj: The imported ``mujoco`` module.
        model: The ``MjModel`` whose DoF count defines the matrix size.
        data: The ``MjData`` holding the sparse inertia (after a forward pass).

    Returns:
        A C-contiguous ``(nv, nv)`` float64 array. Empty (``(0, 0)``) when the
        model has no DoFs.

    Raises:
        TypeError: If no known ``mj_fullM`` signature accepts the arguments.
    """
    nv = model.nv
    dst = np.zeros((nv, nv), dtype=np.float64, order="C")
    if nv == 0:
        return dst
    try:
        # MuJoCo >= 3.10: dst is the third positional argument.
        mj.mj_fullM(model, data, dst)
        return dst
    except TypeError:
        pass
    # Legacy signature: mj_fullM(model, dst, qM). Some builds require the
    # sparse buffer as a 2D [m, 1] column; others accept the raw 1D buffer.
    qm = np.ascontiguousarray(data.qM, dtype=np.float64)
    try:
        mj.mj_fullM(model, dst, qm.reshape(-1, 1))
    except TypeError:
        mj.mj_fullM(model, dst, qm)
    return dst


class PhysicsMixin:
    """Advanced MuJoCo physics capabilities mixed into ``Simulation``.

    Lives at roughly ``self._world._data`` + ``self._world._model`` level:
    reads/writes MuJoCo arrays directly for checkpointing, raycasts,
    jacobians, joint control, sensor readout, etc.

    **Coupling** (see simulation.py top-level docstring): mixin reaches
    into ``self._world``, ``self._lock``, and the host's
    ``_require_no_running_policy`` / ``_require_world`` / ``_prune_done_futures``
    helpers. ``TYPE_CHECKING`` stubs below exist so mypy accepts those
    lookups; they are a documentary contract, not an enforceable protocol.

    Naming: methods match action names in tool_spec.json for direct dispatch.
    """

    if TYPE_CHECKING:
        import threading

        from strands_robots.simulation.models import SimWorld

        _lock: "threading.RLock"
        _world: "SimWorld | None"

        def _require_no_running_policy(
            self, action_name: str, robot_name: str | None = None
        ) -> dict[str, Any] | None: ...
        def _require_world(self) -> dict[str, Any] | None: ...

    # State Checkpointing

    def save_state(self, name: str = "default") -> dict[str, Any]:
        """Save the full physics state (qpos, qvel, act, time) to a named checkpoint.

        Uses mj_getState with mjSTATE_FULLPHYSICS for complete state capture
        including ctrl and qfrc_applied buffers.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            state_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_FULLPHYSICS)
            state = np.zeros(state_size)
            mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_FULLPHYSICS)

        if not hasattr(self._world, "_checkpoints"):
            self._world._checkpoints = {}

        self._world._checkpoints[name] = {
            "state": state.copy(),
            "sim_time": self._world.sim_time,
            "step_count": self._world.step_count,
        }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"State '{name}' saved\n"
                        f"  t={self._world.sim_time:.4f}s, step={self._world.step_count}\n"
                        f"State vector: {state_size} floats\n"
                        f"Checkpoints: {list(self._world._checkpoints.keys())}"
                    )
                }
            ],
        }

    def load_state(self, name: str = "default") -> dict[str, Any]:
        """Restore physics state from a named checkpoint."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # load_state during a running policy races worker thread
        if err := self._require_no_running_policy("load_state"):
            return err

        checkpoints = getattr(self._world, "_checkpoints", {})
        if name not in checkpoints:
            available = list(checkpoints.keys()) if checkpoints else ["none"]
            return {
                "status": "error",
                "content": [{"text": f"Checkpoint '{name}' not found. Available: {available}"}],
            }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        checkpoint = checkpoints[name]

        with self._lock:
            mj.mj_setState(model, data, checkpoint["state"], mj.mjtState.mjSTATE_FULLPHYSICS)
            mj.mj_forward(model, data)

            self._world.sim_time = checkpoint["sim_time"]
            self._world.step_count = checkpoint["step_count"]

        return {
            "status": "success",
            "content": [
                {"text": f"State '{name}' restored (t={self._world.sim_time:.4f}s, step={self._world.step_count})"}
            ],
        }

    # External Forces

    def apply_force(
        self,
        body_name: str,
        force: list[float] | None = None,
        torque: list[float] | None = None,
        point: list[float] | None = None,
    ) -> dict[str, Any]:
        """Apply an external force and/or torque to a body (latched).

        Uses mj_applyFT for precise force application at a world-frame point.
        The force is latched in ``qfrc_applied`` and applied on every
        subsequent ``mj_step`` until overwritten by the next ``apply_force``
        call. Each call zeroes the buffer first (replacing, not accumulating).

        To stop the force: ``apply_force(body, force=[0, 0, 0])``.

        Args:
            body_name: Target body name.
            force: [fx, fy, fz] in world frame (Newtons).
            torque: [tx, ty, tz] in world frame (N·m).
            point: [px, py, pz] world-frame point of force application.
                   Defaults to body CoM if not specified.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # apply_force during a running policy races worker thread
        if err := self._require_no_running_policy("apply_force"):
            return err

        # must supply at least one non-zero force or torque
        if force is None and torque is None:
            return {
                "status": "error",
                "content": [{"text": "apply_force: specify at least one of 'force' or 'torque' (non-zero vector)."}],
            }

        # Validate vector lengths before hitting numpy
        for _name, _vec in (("force", force), ("torque", torque), ("point", point)):
            if _vec is not None:
                try:
                    if len(_vec) != 3:
                        return {
                            "status": "error",
                            "content": [
                                {"text": f"apply_force: '{_name}' must be a 3-element vector [x,y,z], got {len(_vec)}"}
                            ],
                        }
                except TypeError:
                    return {
                        "status": "error",
                        "content": [{"text": f"apply_force: '{_name}' must be a list/tuple of 3 numbers"}],
                    }
                # Every element must be a finite real number. Without this,
                # non-numeric elements (e.g. ["a", "b", "c"]) raise ValueError
                # inside np.array(dtype=float64) - escaping the structured-error
                # contract - and nan/inf or nested lists slip silently into
                # mj_applyFT, injecting bad state into the physics buffer.
                # bool is intentionally accepted (subclass of int -> finite);
                # rejecting it is out of scope for numeric-element validation.
                for _elem in _vec:
                    try:
                        _f = float(_elem)
                    except (TypeError, ValueError):
                        return {
                            "status": "error",
                            "content": [{"text": f"apply_force: '{_name}' elements must be numbers, got {_vec!r}"}],
                        }
                    if not math.isfinite(_f):
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": f"apply_force: '{_name}' must contain finite numbers (no nan/inf), got {_vec!r}"
                                }
                            ],
                        }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}

        f = np.array(force or [0, 0, 0], dtype=np.float64)
        t = np.array(torque or [0, 0, 0], dtype=np.float64)
        # Note: explicit [0,0,0] is a valid "clear the latched force" command; we only
        # reject the case where the caller forgot both args (handled above).
        p = np.array(point, dtype=np.float64) if point else data.xipos[body_id].copy()

        # Zero the buffer first so calls are idempotent (replace, not accumulate).
        # NOTE: MuJoCo does NOT reset qfrc_applied in mj_step - the force
        # persists on every subsequent step until the next apply_force call.
        with self._lock:
            data.qfrc_applied[:] = 0.0
            mj.mj_applyFT(model, data, f, t, p, body_id, data.qfrc_applied)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Force applied to '{body_name}' (body {body_id})\n"
                        f"Force: {f.tolist()} N\n"
                        f"Torque: {t.tolist()} N·m\n"
                        f"Point: {p.tolist()}"
                    )
                }
            ],
        }

    # Raycasting

    def _resolve_mj_name(self, obj_type: int, name: str) -> int:
        """Look up a MuJoCo name, tolerating robot namespacing.

        For physics/introspection methods that accept raw body/joint/site
        names (``get_body_state("gripper")`` etc.), we try the name
        verbatim first, then fall back to trying it prefixed with every
        robot's namespace. This preserves the pre-namespacing UX for
        single-robot scenes while still working in multi-robot scenes
        when the name is unambiguous.

        In multi-robot scenes where multiple robots contain a body with
        the same short name (e.g. two so101s each having ``gripper``),
        the caller MUST pass the namespaced form (``arm0/gripper``) to
        disambiguate. The fallback returns the first match it finds,
        which is non-deterministic - this is a deliberate
        "unambiguous or explicit" contract.
        """
        import mujoco as _mj

        assert self._world is not None and self._world._model is not None
        model = self._world._model
        mid = _mj.mj_name2id(model, obj_type, name)
        if mid >= 0:
            return int(mid)
        if "/" in name:  # already namespaced, no point retrying
            return -1
        for robot in self._world.robots.values():
            if robot.namespace:
                mid = _mj.mj_name2id(model, obj_type, robot.namespace + name)
                if mid >= 0:
                    return int(mid)
        return -1

    def raycast(
        self,
        origin: list[float],
        direction: list[float],
        exclude_body: int = -1,
        include_static: bool = True,
    ) -> dict[str, Any]:
        """Cast a ray and find the first geom intersection.

        Uses mj_ray for precise distance sensing / obstacle detection.

        Args:
            origin: [x, y, z] ray start point in world frame.
            direction: [dx, dy, dz] ray direction (auto-normalized).
            exclude_body: Body ID to exclude from intersection (-1 = none).
            include_static: Whether to include static geoms.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        # validate vector shapes and reject zero-direction (mj_ray aborts the process on len=0)
        try:
            if len(origin) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"raycast: 'origin' must be 3 elements [x,y,z], got {len(origin)}"}],
                }
            if len(direction) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"raycast: 'direction' must be 3 elements [dx,dy,dz], got {len(direction)}"}],
                }
        except TypeError:
            return {
                "status": "error",
                "content": [{"text": "raycast: 'origin' and 'direction' must be lists of 3 numbers"}],
            }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        pnt = np.array(origin, dtype=np.float64)
        vec = np.array(direction, dtype=np.float64)
        # Normalize direction
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return {
                "status": "error",
                "content": [{"text": "raycast: 'direction' vector is zero-length - supply a non-zero direction."}],
            }
        vec = vec / norm

        geomid = np.array([-1], dtype=np.int32)
        dist = mj.mj_ray(
            model,
            data,
            pnt,
            vec,
            None,  # geom group filter (None = all)
            1 if include_static else 0,
            exclude_body,
            geomid,
        )

        hit = dist >= 0
        geom_name = None
        if hit and geomid[0] >= 0:
            geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomid[0])

        result = {
            "hit": hit,
            "distance": float(dist) if hit else None,
            "geom_id": int(geomid[0]) if hit else None,
            "geom_name": geom_name,
            "hit_point": (pnt + vec * dist).tolist() if hit else None,
        }

        if hit:
            text = f"Ray hit '{geom_name or geomid[0]}' at dist={dist:.4f}m, point={result['hit_point']}"
        else:
            text = "Ray: no intersection"

        return {"status": "success", "content": [{"text": text}, {"json": result}]}

    # Jacobians

    def get_jacobian(
        self,
        body_name: str | None = None,
        site_name: str | None = None,
        geom_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute the Jacobian (position + rotation) for a body, site, or geom.

        The Jacobian maps joint velocities to Cartesian velocities:
            v = J @ dq

        Returns both positional (3×nv) and rotational (3×nv) Jacobians,
        computed at the current ``qpos`` (the position pipeline is recomputed
        first so the result is never a stale earlier configuration).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        with self._lock:
            # Reflect the CURRENT configuration. mj_jac* read data.xpos/site_xpos/
            # geom_xpos, data.subtree_com and data.cdof, all of which are stale if
            # qpos changed since the last forward (e.g. a direct data.qpos write or
            # set_joint_velocities). Recompute the position pipeline first so the
            # Jacobian is not silently that of an earlier pose. Matches
            # forward_kinematics; cheaper than a full mj_forward.
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            if body_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}
                mj.mj_jacBody(model, data, jacp, jacr, obj_id)
                label = f"body '{body_name}'"
            elif site_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_SITE, site_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": f"Site '{site_name}' not found."}]}
                mj.mj_jacSite(model, data, jacp, jacr, obj_id)
                label = f"site '{site_name}'"
            elif geom_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, geom_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": f"Geom '{geom_name}' not found."}]}
                mj.mj_jacGeom(model, data, jacp, jacr, obj_id)
                label = f"geom '{geom_name}'"
            else:
                return {"status": "error", "content": [{"text": "Specify body_name, site_name, or geom_name."}]}

        return {
            "status": "success",
            "content": [
                {"text": f"Jacobian for {label}: pos={jacp.shape}, rot={jacr.shape}, nv={model.nv}"},
                {"json": {"jacp": jacp.tolist(), "jacr": jacr.tolist(), "nv": model.nv}},
            ],
        }

    # Energy

    def get_energy(self) -> dict[str, Any]:
        """Compute potential and kinetic energy of the system."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            mj.mj_energyPos(model, data)
            mj.mj_energyVel(model, data)
            potential = float(data.energy[0])
            kinetic = float(data.energy[1])
        total = potential + kinetic

        return {
            "status": "success",
            "content": [
                {"text": f"Energy: potential={potential:.4f}J, kinetic={kinetic:.4f}J, total={total:.4f}J"},
                {"json": {"potential": potential, "kinetic": kinetic, "total": total}},
            ],
        }

    # Mass Matrix

    def get_mass_matrix(self) -> dict[str, Any]:
        """Compute the full mass (inertia) matrix M(q).

        M is nv×nv where nv is the number of DoFs.
        Useful for dynamics analysis, impedance control, etc.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        # data.qM is only valid after a forward pass. Serialize the
        # forward+fullM read against concurrent policy threads (GH: concurrency
        # audit) so a sibling robot's mj_step can't mutate data mid-read.
        with self._lock:
            mj.mj_forward(model, data)
            nv = model.nv
            M = _full_mass_matrix(mj, model, data)
            if nv > 0:
                rank = int(np.linalg.matrix_rank(M))
                cond = float(np.linalg.cond(M)) if rank > 0 else float("inf")
            else:
                # Empty scene (no DOFs yet) - return a well-typed zero payload
                # instead of crashing in numpy on the empty matrix.
                rank = 0
                cond = float("inf")

        return {
            "status": "success",
            "content": [
                {"text": f"Mass matrix: {nv}×{nv}, rank={rank}, cond={cond:.2e}"},
                {
                    "json": {
                        "shape": [nv, nv],
                        "rank": rank,
                        "condition_number": cond,
                        "diagonal": np.diag(M).tolist(),
                        "total_mass": float(np.sum(model.body_mass)),
                    }
                },
            ],
        }

    # Inverse Dynamics

    def inverse_dynamics(self) -> dict[str, Any]:
        """Compute the generalized forces required to hold the current state.

        Runs ``mj_inverse`` for a target acceleration of zero, so the result
        is the gravity- and velocity-bias (Coriolis/centrifugal) compensation
        torques that keep the robot at its current ``qpos``/``qvel`` with zero
        acceleration - the standard inverse-dynamics query for a manipulator
        (at rest, pure gravity compensation).

        ``mj_inverse`` reads ``data.qacc`` as the *desired* acceleration, so
        this method runs ``mj_forward`` first (so the position/velocity
        kinematics match the current ``qpos``/``qvel``, matching the defensive
        forward in ``get_mass_matrix``) and zeroes ``qacc`` for the solve,
        restoring the buffer afterwards. Without this it would use whatever
        stale forward-dynamics acceleration was left in ``data.qacc`` and ask
        ``mj_inverse`` to reproduce free-fall - returning ~0 forces regardless
        of pose, never the compensation torques the query is for.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            # Establish consistent position/velocity kinematics for the current
            # state, then solve inverse dynamics for zero desired acceleration.
            mj.mj_forward(model, data)
            saved_qacc = data.qacc.copy()
            data.qacc[:] = 0.0
            try:
                mj.mj_inverse(model, data)
                # Build named force mapping
                forces = {}
                for i in range(model.njnt):
                    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                    if name:
                        dof_adr = model.jnt_dofadr[i]
                        forces[name] = float(data.qfrc_inverse[dof_adr])
            finally:
                data.qacc[:] = saved_qacc

        return {
            "status": "success",
            "content": [
                {"text": f"Inverse dynamics: {len(forces)} joint forces computed"},
                {"json": {"qfrc_inverse": forces}},
            ],
        }

    # Body Introspection

    def get_body_state(
        self,
        body_name: str,
    ) -> dict[str, Any]:
        """Get the full state of a body: position, orientation, velocity, acceleration.

        Returns Cartesian pose + 6D spatial velocity (linear + angular),
        computed at the current ``qpos``/``qvel`` (the forward pipeline is run
        first so pose and velocity are never a stale earlier state).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}

        with self._lock:
            # Reflect the CURRENT qpos/qvel. This reads data.xpos/xquat/xmat/xipos
            # (position pipeline) and data.cvel via mj_objectVelocity (velocity
            # pipeline); both are stale if state changed since the last forward
            # (e.g. set_joint_velocities writes qvel without forwarding, or a
            # direct data.qpos write). Run the full pipeline so pose AND 6D
            # velocity are consistent with the current state. Matches
            # get_mass_matrix / inverse_dynamics / get_sensor_data.
            mj.mj_forward(model, data)
            # Position and orientation
            pos = data.xpos[body_id].tolist()
            quat = data.xquat[body_id].tolist()
            rotmat = data.xmat[body_id].reshape(3, 3).tolist()

            # Velocity (6D: angular then linear in world frame)
            vel = np.zeros(6)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body_id, vel, 0)
            linvel = vel[3:].tolist()
            angvel = vel[:3].tolist()

            # Mass and inertia
            mass = float(model.body_mass[body_id])
            com = data.xipos[body_id].tolist()

        state = {
            "position": pos,
            "quaternion": quat,
            "rotation_matrix": rotmat,
            "linear_velocity": linvel,
            "angular_velocity": angvel,
            "mass": mass,
            "center_of_mass": com,
        }

        text = (
            f"Body '{body_name}' (id={body_id}):\n"
            f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n"
            f"  linvel: [{linvel[0]:.4f}, {linvel[1]:.4f}, {linvel[2]:.4f}]\n"
            f"  angvel: [{angvel[0]:.4f}, {angvel[1]:.4f}, {angvel[2]:.4f}]\n"
            f"  mass: {mass:.4f}kg, com: {com}"
        )

        return {"status": "success", "content": [{"text": text}, {"json": state}]}

    # Direct Joint Control

    def set_joint_positions(
        self,
        positions: dict[str, float] | list[float] | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Set joint positions directly (bypassing actuators).

        Writes to qpos and runs mj_forward to update kinematics.
        Useful for teleportation, IK solutions, or keyframe setting.

        Accepts EITHER form:

        * dict: {joint_name: value, ...} - explicit per-joint, safest in multi-robot scenes.
        * list/tuple: [v0, v1, ...] - ordered positional. Must match a single robot's
          joint count (when ``robot_name`` is given, that robot's joints; otherwise the
          world must contain exactly one robot, or the call errors).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # mutating qpos under a running policy races mj_step
        if err := self._require_no_running_policy("set_joint_positions"):
            return err

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if positions is None:
            return {
                "status": "error",
                "content": [{"text": "set_joint_positions: 'positions' is required (list or dict of joint values)."}],
            }

        # normalize list input to dict using a deterministic joint ordering
        ignored: list[str] = []
        if isinstance(positions, (list, tuple)):
            robots = list(self._world.robots.values())
            if robot_name is not None:
                robots = [r for r in robots if r.name == robot_name]
                if not robots:
                    return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
            if len(robots) == 0:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "set_joint_positions: list form requires a robot in the world; pass a dict instead, or add a robot first."
                        }
                    ],
                }
            if len(robots) > 1 and robot_name is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"set_joint_positions: list form is ambiguous with {len(robots)} robots; pass 'robot_name=' or use a dict."
                        }
                    ],
                }
            robot = robots[0]
            joint_names = list(getattr(robot, "joint_names", []) or [])
            if not joint_names:
                # Fall back: enumerate joints that belong to this robot via namespace
                ns = getattr(robot, "namespace", "") or ""
                joint_names = []
                for jid in range(model.njnt):
                    jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                    if jn and (not ns or jn.startswith(ns)):
                        joint_names.append(jn)
            if len(positions) != len(joint_names):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_joint_positions: list length {len(positions)} does not match robot "
                                f"'{robot.name}' joint count {len(joint_names)}. Use a dict for partial updates."
                            )
                        }
                    ],
                }
            positions = dict(zip(joint_names, positions, strict=True))
        elif not isinstance(positions, dict):
            return {
                "status": "error",
                "content": [
                    {"text": f"set_joint_positions: 'positions' must be a dict or list, got {type(positions).__name__}"}
                ],
            }

        set_count = 0
        with self._lock:
            for jnt_name, value in positions.items():
                jnt_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_JOINT, jnt_name)
                if jnt_id >= 0:
                    qpos_adr = model.jnt_qposadr[jnt_id]
                    data.qpos[qpos_adr] = float(value)
                    set_count += 1
                else:
                    ignored.append(jnt_name)
                    logger.warning("Joint '%s' not found, skipping", jnt_name)

            mj.mj_forward(model, data)

        msg = f"Set {set_count}/{len(positions)} joint positions, FK updated"
        if ignored:
            msg += f" (ignored: {ignored})"
        return {
            "status": "success",
            "content": [{"text": msg}],
        }

    def set_joint_velocities(
        self,
        velocities: dict[str, float] | list[float] | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Set joint velocities directly.

        Writes to qvel. Useful for initializing dynamics. Accepts dict or list
        (see set_joint_positions for list semantics).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_joint_velocities"):
            return err

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if velocities is None:
            return {
                "status": "error",
                "content": [{"text": "set_joint_velocities: 'velocities' is required (list or dict)."}],
            }

        ignored: list[str] = []
        if isinstance(velocities, (list, tuple)):
            robots = list(self._world.robots.values())
            if robot_name is not None:
                robots = [r for r in robots if r.name == robot_name]
                if not robots:
                    return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
            if len(robots) == 0:
                return {
                    "status": "error",
                    "content": [{"text": "set_joint_velocities: list form requires a robot in the world."}],
                }
            if len(robots) > 1 and robot_name is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"set_joint_velocities: list form is ambiguous with {len(robots)} robots; pass 'robot_name=' or use a dict."
                        }
                    ],
                }
            robot = robots[0]
            joint_names = list(getattr(robot, "joint_names", []) or [])
            if not joint_names:
                ns = getattr(robot, "namespace", "") or ""
                joint_names = []
                for jid in range(model.njnt):
                    jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                    if jn and (not ns or jn.startswith(ns)):
                        joint_names.append(jn)
            if len(velocities) != len(joint_names):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_joint_velocities: list length {len(velocities)} does not match robot "
                                f"'{robot.name}' joint count {len(joint_names)}. Use a dict for partial updates."
                            )
                        }
                    ],
                }
            velocities = dict(zip(joint_names, velocities, strict=True))
        elif not isinstance(velocities, dict):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"set_joint_velocities: 'velocities' must be a dict or list, got {type(velocities).__name__}"
                    }
                ],
            }

        set_count = 0
        with self._lock:
            for jnt_name, value in velocities.items():
                jnt_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_JOINT, jnt_name)
                if jnt_id >= 0:
                    dof_adr = model.jnt_dofadr[jnt_id]
                    data.qvel[dof_adr] = float(value)
                    set_count += 1
                else:
                    ignored.append(jnt_name)

        msg = f"Set {set_count}/{len(velocities)} joint velocities"
        if ignored:
            msg += f" (ignored: {ignored})"
        return {
            "status": "success",
            "content": [{"text": msg}],
        }

    # Sensor Readout

    def get_sensor_data(self, sensor_name: str | None = None) -> dict[str, Any]:
        """Read sensor values from the simulation.

        MuJoCo supports: jointpos, jointvel, accelerometer, gyro, force,
        torque, touch, rangefinder, framequat, subtreecom, clock, etc.

        Args:
            sensor_name: Specific sensor name, or None for all sensors.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if model.nsensor == 0:
            # distinguish "no sensors at all" from "that specific sensor not found"
            if sensor_name:
                return {
                    "status": "error",
                    "content": [{"text": f"Sensor '{sensor_name}' not found. Model has no sensors."}],
                }
            return {"status": "success", "content": [{"text": "No sensors in model."}]}

        # Lock while running mj_forward + reading sensordata so a policy
        # thread's mj_step can't mutate data between our forward pass and
        # the slice read. Also snapshot sensor metadata under the lock
        # because the model could theoretically change during this call
        # (it's not gated with _require_no_running_policy).
        with self._lock:
            mj.mj_forward(model, data)
            sensordata_snapshot = np.asarray(data.sensordata).copy()

        sensors = {}
        for i in range(model.nsensor):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            if not name:
                name = f"sensor_{i}"

            adr = model.sensor_adr[i]
            dim = model.sensor_dim[i]
            values = sensordata_snapshot[adr : adr + dim].tolist()

            if sensor_name and name != sensor_name:
                continue

            sensors[name] = {
                "values": values if dim > 1 else values[0],
                "dim": int(dim),
                "type": int(model.sensor_type[i]),
            }

        if sensor_name and sensor_name not in sensors:
            return {"status": "error", "content": [{"text": f"Sensor '{sensor_name}' not found."}]}

        lines = [f"Sensors ({len(sensors)}/{model.nsensor}):"]
        for name, info in sensors.items():
            lines.append(f"{name}: {info['values']} (dim={info['dim']})")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"sensors": sensors}}],
        }

    # Runtime Model Modification

    def set_body_properties(
        self,
        body_name: str,
        mass: float | None = None,
    ) -> dict[str, Any]:
        """Modify body properties at runtime (no recompile needed).

        Currently supports setting a body's ``mass``. Because a rigid body's
        inertia tensor tracks its mass at fixed geometry (a uniform density
        change), the body's ``body_inertia`` is scaled by the same ratio so the
        translational and rotational dynamics stay physically consistent.

        Changes take effect on the next ``mj_step``.

        Args:
            body_name: Name of the body to modify.
            mass: New absolute mass (kg); must be ``> 0``. When set, the body's
                inertia is scaled by ``mass / old_mass`` to preserve consistency.

        Returns:
            A tool-result dict; ``status="error"`` if the world is missing, a
            policy is running, ``mass`` is not a positive number, or the body is
            not found, otherwise ``status="success"`` summarizing the change.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_body_properties"):
            return err

        # mass must be > 0 (physics invariant)
        if mass is not None:
            try:
                mass = float(mass)
            except (TypeError, ValueError):
                return {
                    "status": "error",
                    "content": [{"text": f"set_body_properties: 'mass' must be a positive number, got {mass!r}"}],
                }
            if mass <= 0:
                return {
                    "status": "error",
                    "content": [{"text": f"set_body_properties: 'mass' must be > 0, got {mass}"}],
                }

        mj = _ensure_mujoco()
        model = self._world._model
        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}

        changes = []
        with self._lock:
            if mass is not None:
                old_mass = float(model.body_mass[body_id])
                model.body_mass[body_id] = mass
                changes.append(f"mass: {old_mass:.3f} → {mass:.3f}")
                # Inertia tracks mass for fixed geometry: setting a rigid body's
                # mass to a new value at constant shape is a uniform density
                # change, which scales its inertia tensor by the same factor
                # (I = integral of r^2 dm). Updating body_mass alone leaves a
                # physically inconsistent body - heavy in translation but with
                # the old rotational resistance - which silently corrupts the
                # rotational dynamics (and cannot be corrected by the caller,
                # since mass is the only settable property). Scale body_inertia
                # by the same ratio (matches randomize(randomize_physics=True)
                # and the Newton backend, which scale both together).
                if old_mass > 0:
                    model.body_inertia[body_id] *= mass / old_mass

        return {
            "status": "success",
            "content": [{"text": f"Body '{body_name}': {', '.join(changes)}"}],
        }

    def set_geom_properties(
        self,
        geom_name: str | None = None,
        geom_id: int | None = None,
        color: list[float] | None = None,
        friction: list[float] | None = None,
        size: list[float] | None = None,
    ) -> dict[str, Any]:
        """Modify geom properties at runtime (no recompile needed).

        Changes take effect immediately for rendering (color) or next step (friction, size).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_geom_properties"):
            return err

        mj = _ensure_mujoco()
        model = self._world._model

        gid = geom_id
        if geom_name:
            gid = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, geom_name)
            # our add_object pipeline names geoms as ``{object_name}_geom``.
            # Accept the plain object name as a convenience alias.
            if (gid is None or gid < 0) and not geom_name.endswith("_geom"):
                gid = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, f"{geom_name}_geom")
        if gid is None or gid < 0 or gid >= model.ngeom:
            return {"status": "error", "content": [{"text": f"Geom '{geom_name or geom_id}' not found."}]}

        label = geom_name or f"geom_{gid}"
        changes = []

        with self._lock:
            if color is not None:
                model.geom_rgba[gid] = color[:4] if len(color) >= 4 else color[:3] + [1.0]
                changes.append(f"color → {model.geom_rgba[gid].tolist()}")

            if friction is not None:
                fric = friction[:3] if len(friction) >= 3 else friction + [0.0] * (3 - len(friction))
                model.geom_friction[gid] = fric
                changes.append(f"friction → {fric}")

            if size is not None:
                n = min(len(size), 3)
                model.geom_size[gid, :n] = size[:n]
                changes.append(f"size → {model.geom_size[gid].tolist()}")

        return {
            "status": "success",
            "content": [{"text": f"Geom '{label}': {', '.join(changes)}"}],
        }

    # Contact Force Analysis

    def get_contact_forces(self) -> dict[str, Any]:
        """Get detailed contact forces for all active contacts.

        Uses mj_contactForce for each active contact pair.
        Returns normal and friction forces.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        contacts = []
        with self._lock:
            for i in range(data.ncon):
                c = data.contact[i]
                g1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
                g2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"

                # Get contact force (normal + friction in contact frame)
                force = np.zeros(6)
                mj.mj_contactForce(model, data, i, force)

                contacts.append(
                    {
                        "geom1": g1,
                        "geom2": g2,
                        "distance": float(c.dist),
                        "position": c.pos.tolist(),
                        "normal_force": float(force[0]),
                        "friction_force": force[1:3].tolist(),
                        "full_wrench": force.tolist(),
                    }
                )

        if not contacts:
            return {"status": "success", "content": [{"text": "No active contacts."}]}

        lines = [f"{len(contacts)} contacts:"]
        for c in contacts[:15]:
            lines.append(f"{c['geom1']} ↔ {c['geom2']}: normal={c['normal_force']:.3f}N, dist={c['distance']:.4f}m")
        if len(contacts) > 15:
            lines.append(f"  ... and {len(contacts) - 15} more")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"contacts": contacts}}],
        }

    # Multi-Ray (batch raycasting)

    def multi_raycast(
        self,
        origin: list[float],
        directions: list[list[float]],
        exclude_body: int = -1,
    ) -> dict[str, Any]:
        """Cast multiple rays from a single origin (e.g., for LIDAR simulation).

        Efficiently casts N rays using individual mj_ray calls.
        Returns array of distances and hit geoms.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        # validate origin shape; per-ray zero-direction guard (avoid mj_ray abort)
        try:
            if len(origin) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"multi_raycast: 'origin' must be 3 elements [x,y,z], got {len(origin)}"}],
                }
        except TypeError:
            return {"status": "error", "content": [{"text": "multi_raycast: 'origin' must be a list of 3 numbers"}]}

        pnt = np.array(origin, dtype=np.float64)
        results: list[dict[str, Any]] = []

        for idx, d in enumerate(directions):
            try:
                if len(d) != 3:
                    results.append(
                        {
                            "distance": None,
                            "geom_id": None,
                            "error": f"ray[{idx}]: direction must have 3 elements, got {len(d)}",
                        }
                    )
                    continue
            except TypeError:
                results.append(
                    {"distance": None, "geom_id": None, "error": f"ray[{idx}]: direction must be a list of 3 numbers"}
                )
                continue
            vec = np.array(d, dtype=np.float64)
            norm = np.linalg.norm(vec)
            if norm < 1e-10:
                results.append({"distance": None, "geom_id": None, "error": f"ray[{idx}]: zero-length direction"})
                continue
            vec /= norm
            geomid = np.array([-1], dtype=np.int32)
            dist = mj.mj_ray(model, data, pnt, vec, None, 1, exclude_body, geomid)
            results.append(
                {
                    "distance": float(dist) if dist >= 0 else None,
                    "geom_id": int(geomid[0]) if dist >= 0 else None,
                }
            )

        hit_count = sum(1 for r in results if r["distance"] is not None)
        return {
            "status": "success",
            "content": [
                {"text": f"Multi-ray: {hit_count}/{len(directions)} hits from {origin}"},
                {"json": {"rays": results}},
            ],
        }

    # Forward Kinematics (explicit)

    def forward_kinematics(self, body_name: str | None = None) -> dict[str, Any]:
        """Run forward kinematics to update all body positions/orientations.

        Usually called implicitly by mj_step, but useful after manually
        setting qpos to see updated Cartesian positions.

        If ``body_name`` is given, the response is filtered to that
        single body (and errors cleanly if the body doesn't exist).
        Otherwise returns every body as before.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_camlight(model, data)

            if body_name is not None:
                bid = mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    return {"status": "error", "content": [{"text": f"Body '{body_name}' not found."}]}
                body_payload = {
                    "position": data.xpos[bid].tolist(),
                    "quaternion": data.xquat[bid].tolist(),
                }
                return {
                    "status": "success",
                    "content": [
                        {"text": f"FK for '{body_name}': pos={body_payload['position']}"},
                        {"json": {"body": body_name, **body_payload}},
                    ],
                }

            bodies = {}
            for i in range(model.nbody):
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                bodies[name] = {
                    "position": data.xpos[i].tolist(),
                    "quaternion": data.xquat[i].tolist(),
                }

        return {
            "status": "success",
            "content": [
                {"text": f"FK computed for {model.nbody} bodies"},
                {"json": {"bodies": bodies}},
            ],
        }

    # Total Mass

    def get_total_mass(self) -> dict[str, Any]:
        """Get total mass and per-body mass breakdown."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model = self._world._model

        total = float(mj.mj_getTotalmass(model))
        bodies = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            m = float(model.body_mass[i])
            if m > 0:
                bodies[name] = m

        return {
            "status": "success",
            "content": [
                {"text": f"Total mass: {total:.4f}kg ({len(bodies)} bodies with mass)"},
                {"json": {"total_mass": total, "bodies": bodies}},
            ],
        }

    # Export Model XML

    def export_xml(self, output_path: str | None = None) -> dict[str, Any]:
        """Export the current scene as canonical MJCF via ``spec.to_xml()``.

        Every code path in the MjSpec backend stashes the live ``MjSpec`` in
        ``_backend_state["spec"]`` (``create_world`` / ``load_scene`` /
        ``replace_scene_mjcf`` / ``patch_scene_mjcf`` / the ``inject_*``
        helpers all do this). The serialised XML reflects any runtime
        mutation, so no extra caching or round-tripping is needed.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        spec = self._world._backend_state.get("spec") if self._world._backend_state else None
        if spec is None:
            # Should never happen in the MjSpec backend. Surfacing as an
            # error is better than a C-level crash via mj_saveLastXML.
            return {
                "status": "error",
                "content": [
                    {"text": "No MjSpec tracked on this world - cannot export. This is a bug; please file an issue."}
                ],
            }

        try:
            # spec.to_xml() emits benign "Attach conflict ... keeping parent
            # value" chatter (Python UserWarning + raw fd-2 writes) for every
            # attached robot scene. Every other call site wraps it; export_xml
            # must too, or the noise leaks straight to the user's console.
            with filter_mujoco_attach_noise():
                xml = spec.to_xml()
        except Exception as e:
            return {"status": "error", "content": [{"text": f"spec.to_xml() failed: {e}"}]}

        if output_path:
            with open(output_path, "w") as f:
                f.write(xml)
            return {"status": "success", "content": [{"text": f"Model exported to {output_path}"}]}

        return {
            "status": "success",
            "content": [{"text": f"Model XML ({len(xml)} chars):\n{xml[:2000]}{'...' if len(xml) > 2000 else ''}"}],
        }
