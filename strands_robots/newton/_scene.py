"""Scene lifecycle — create_world, replicate, reset, step."""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ._core import NewtonBackend


def create_world(
    backend: NewtonBackend,
    gravity: Optional[Tuple[float, float, float]] = None,
    ground_plane: bool = True,
    up_axis: str = "y",
) -> Dict[str, Any]:
    """Create a new simulation world."""
    backend._lazy_init()
    newton = backend._newton

    if gravity is not None:
        gravity_val = tuple(gravity)
    else:
        if up_axis == "z":
            gravity_val = (0.0, 0.0, -9.81)
        else:
            gravity_val = (0.0, -9.81, 0.0)
    backend._gravity = gravity_val

    # Determine scalar gravity magnitude and up_axis vector
    gx, gy, gz = gravity_val
    gravity_mag = -(abs(gx) + abs(gy) + abs(gz))  # negative by convention
    if gravity_mag == 0.0:
        gravity_mag = -9.81

    if up_axis == "z" or abs(gz) >= abs(gy):
        up_vec = (0.0, 0.0, 1.0)
        gravity_mag = -abs(gz) if gz != 0 else -9.81
    else:
        up_vec = (0.0, 1.0, 0.0)
        gravity_mag = -abs(gy) if gy != 0 else -9.81

    try:
        backend._builder = newton.ModelBuilder(gravity=gravity_mag)
    except (TypeError, AttributeError):
        backend._builder = newton.ModelBuilder()
    # Set scalar gravity and up_vector
    try:
        backend._builder.gravity = gravity_mag
    except Exception:
        pass
    try:
        backend._builder.up_vector = up_vec
        backend._builder.up_axis = up_axis
    except Exception:
        pass

    backend._ground_plane_requested = ground_plane
    if ground_plane:
        _add_ground_plane(backend._builder)

    backend._world_created = True
    backend._step_count = 0
    backend._sim_time = 0.0

    return {
        "success": True,
        "message": "World created.",
        "world_info": {
            "gravity": gravity_val,
            "ground_plane": ground_plane,
            "solver": backend._config.solver,
            "device": backend._config.device,
            "physics_dt": backend._config.physics_dt,
            "broad_phase": backend._config.broad_phase,
        },
    }


def recreate_builder(backend: NewtonBackend) -> None:
    """Recreate the model builder to clear any partial state from failed loads."""
    newton = backend._newton
    gravity_mag = getattr(backend._builder, "gravity", -9.81)
    up_vec = getattr(backend._builder, "up_vector", (0.0, 0.0, 1.0))
    try:
        backend._builder = newton.ModelBuilder(gravity=gravity_mag)
    except (TypeError, AttributeError):
        backend._builder = newton.ModelBuilder()
    try:
        backend._builder.gravity = gravity_mag
    except Exception:
        pass
    try:
        backend._builder.up_vector = up_vec
    except Exception:
        pass
    if getattr(backend, "_ground_plane_requested", True):
        _add_ground_plane(backend._builder)
    logger.debug("Builder recreated (cleared partial state)")


def replicate(backend: NewtonBackend, num_envs: Optional[int] = None) -> Dict[str, Any]:
    """Clone the scene across parallel environments."""
    backend._ensure_world()

    if not backend._robots and not backend._cloths:
        return {"success": False, "message": "Nothing to replicate."}

    if num_envs is not None:
        backend._config.num_envs = num_envs
    else:
        num_envs = backend._config.num_envs

    newton = backend._newton
    logger.info("Replicating scene × %d environments …", num_envs)
    t0 = time.perf_counter()

    try:
        spacing = 2.0
        main_builder = backend._builder

        # Store per-world counts
        backend._joints_per_world = getattr(main_builder, "joint_count", 0)
        backend._bodies_per_world = getattr(main_builder, "body_count", 0)
        backend._dof_per_world = getattr(main_builder, "joint_count", 0)

        # Use Newton's replicate — instance method on a new builder
        replicated_builder = newton.ModelBuilder()
        replicated_builder.replicate(
            main_builder,
            world_count=num_envs,
            spacing=(spacing, 0.0, 0.0),
        )
        backend._builder = replicated_builder

        # Add ground plane to replicated builder
        if backend._ground_plane_requested:
            _add_ground_plane(backend._builder)

        _finalize_replicated_model(backend)

        backend._replicated = True
        elapsed = time.perf_counter() - t0
        logger.info("Replicated in %.3fs", elapsed)
        return {
            "success": True,
            "message": f"Replicated × {num_envs} in {elapsed:.3f}s",
        }
    except Exception as exc:
        logger.warning(
            "replicate() failed: %s, falling back to _finalize_model()", exc
        )
        finalize_model(backend)
        return {
            "success": True,
            "message": f"Finalized (replicate fallback): {exc}",
        }


def reset(backend: NewtonBackend, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
    """Reset simulation state. Supports per-environment reset."""
    if backend._model is None:
        return {"success": False, "message": "Model not finalised."}

    wp = backend._wp

    if backend._default_joint_q is None or backend._default_joint_qd is None:
        return {"success": False, "message": "No default joint state stored."}

    try:
        q_default = backend._default_joint_q
        qd_default = backend._default_joint_qd

        if env_ids is None:
            # Full reset
            backend._state_0.joint_q.assign(
                wp.array(q_default, dtype=wp.float32, device=backend._config.device)
            )
            backend._state_0.joint_qd.assign(
                wp.array(qd_default, dtype=wp.float32, device=backend._config.device)
            )
        else:
            # Per-environment reset
            current_q = backend._state_0.joint_q.numpy()
            current_qd = backend._state_0.joint_qd.numpy()

            if backend._joints_per_world > 0 and len(env_ids) > 0:
                for env_id in env_ids:
                    q_start = env_id * backend._joints_per_world
                    q_end = q_start + backend._joints_per_world
                    current_q[q_start:q_end] = q_default[: backend._joints_per_world]
                    current_qd[q_start:q_end] = qd_default[: backend._joints_per_world]
            else:
                logger.warning(
                    "Cannot per-env reset: joints_per_world=%d",
                    backend._joints_per_world,
                )

            backend._state_0.joint_q.assign(
                wp.array(current_q, dtype=wp.float32, device=backend._config.device)
            )
            backend._state_0.joint_qd.assign(
                wp.array(current_qd, dtype=wp.float32, device=backend._config.device)
            )

        # Re-evaluate FK
        if hasattr(backend._newton, "eval_fk"):
            backend._newton.eval_fk(
                backend._model,
                backend._state_0.joint_q,
                backend._state_0.joint_qd,
                backend._state_0,
            )

        backend._step_count = 0
        backend._sim_time = 0.0
        return {"success": True, "message": "Reset complete"}
    except Exception as exc:
        return {"success": False, "message": str(exc)}


def step(backend: NewtonBackend, actions: Any = None) -> Dict[str, Any]:
    """Advance simulation by one frame (with substeps)."""
    if backend._model is None:
        finalize_model(backend)

    wp = backend._wp
    dt = backend._config.physics_dt

    if actions is not None:
        _apply_actions(backend, actions)

    try:
        if backend._config.enable_cuda_graph and backend._cuda_graph is not None:
            # Use captured graph
            wp.capture_launch(backend._cuda_graph)
            backend._state_0, backend._state_1 = backend._state_1, backend._state_0
            backend._step_count += 1
            backend._sim_time += dt * backend._config.substeps
        elif backend._config.enable_differentiable:
            # Differentiable path: no CUDA graph, explicit substeps
            for _ in range(backend._config.substeps):
                _solver_step(backend, dt)
                backend._state_0, backend._state_1 = backend._state_1, backend._state_0
            backend._step_count += 1
            backend._sim_time += dt * backend._config.substeps
        else:
            # Standard path
            if (
                backend._config.enable_cuda_graph
                and backend._cuda_graph is None
                and "cuda" in str(backend._config.device)
            ):
                # Capture CUDA graph on first step
                with wp.ScopedCapture(device=backend._config.device) as capture:
                    for _ in range(backend._config.substeps):
                        _solver_step(backend, dt)
                        backend._state_0, backend._state_1 = backend._state_1, backend._state_0
                backend._cuda_graph = capture.graph
                logger.info("CUDA graph captured via ScopedCapture.")
                backend._step_count += 1
                backend._sim_time += dt * backend._config.substeps
            else:
                for _ in range(backend._config.substeps):
                    _solver_step(backend, dt)
                    backend._state_0, backend._state_1 = backend._state_1, backend._state_0
                backend._step_count += 1
                backend._sim_time += dt * backend._config.substeps

        return {
            "success": True,
            "sim_time": backend._sim_time,
            "step_count": backend._step_count,
        }
    except Exception as exc:
        logger.warning("step() failed: %s", exc)
        return {"success": False, "sim_time": backend._sim_time, "error": str(exc)}


def finalize_model(backend: NewtonBackend) -> None:
    """Finalise builder → Model, allocate states/solver/collision pipeline."""
    if backend._model is not None:
        return
    backend._ensure_world()

    logger.info("Finalising Newton model …")
    t0 = time.perf_counter()

    SolverCls = backend._get_solver_class()

    # Register custom attributes if needed (e.g., VBD/Style3D)
    if hasattr(SolverCls, "register_custom_attributes"):
        try:
            SolverCls.register_custom_attributes(backend._builder)
        except Exception as exc:
            logger.debug("register_custom_attributes: %s", exc)

    # Handle cloth solvers that need special attributes
    if backend._cloths and backend._config.solver in ("vbd", "style3d"):
        if hasattr(SolverCls, "register_custom_attributes"):
            try:
                SolverCls.register_custom_attributes(backend._builder)
            except Exception:
                pass

    _build_model_and_allocate(backend, SolverCls)

    elapsed = time.perf_counter() - t0
    logger.info("Model finalised in %.3fs", elapsed)


# ------------------------------------------------------------------
# Internal helpers
# ------------------------------------------------------------------


def _build_model_and_allocate(backend: NewtonBackend, SolverCls: Any) -> None:
    """Finalize builder into model and allocate states/solver/contacts."""
    newton = backend._newton

    # Finalize the model
    try:
        backend._model = backend._builder.finalize(
            device=backend._config.device,
            requires_grad=backend._config.enable_differentiable,
        )
    except TypeError:
        backend._model = backend._builder.finalize(device=backend._config.device)

    # Set soft contact parameters
    for attr, val in [
        ("soft_contact_ke", backend._config.soft_contact_ke),
        ("soft_contact_kd", backend._config.soft_contact_kd),
        ("soft_contact_mu", backend._config.soft_contact_mu),
        ("soft_contact_restitution", backend._config.soft_contact_restitution),
    ]:
        if hasattr(backend._model, attr):
            setattr(backend._model, attr, val)

    # Allocate states
    backend._state_0 = backend._model.state()
    backend._state_1 = backend._model.state()
    backend._control = backend._model.control()

    # Collision pipeline
    try:
        backend._collision_pipeline = newton.CollisionPipeline(
            backend._model,
            broad_phase=backend._config.broad_phase,
            soft_contact_margin=backend._config.soft_contact_margin,
        )
        backend._contacts = backend._collision_pipeline.contacts()
    except Exception as exc:
        logger.warning("CollisionPipeline init failed: %s", exc)

    # Solver
    try:
        backend._solver = SolverCls(backend._model)
    except TypeError:
        try:
            backend._solver = SolverCls()
        except TypeError:
            backend._solver = SolverCls(backend._model, backend._config.physics_dt)

    # Forward kinematics to get initial state
    try:
        newton.eval_fk(
            backend._model,
            backend._state_0.joint_q,
            backend._state_0.joint_qd,
            backend._state_0,
        )
    except Exception:
        pass

    # Store default joint state for resets
    try:
        backend._default_joint_q = backend._state_0.joint_q.numpy().copy()
        backend._default_joint_qd = backend._state_0.joint_qd.numpy().copy()
    except Exception:
        pass


def _finalize_replicated_model(backend: NewtonBackend) -> None:
    """Finalize a replicated builder (shared logic with finalize_model)."""
    SolverCls = backend._get_solver_class()

    if hasattr(SolverCls, "register_custom_attributes"):
        try:
            SolverCls.register_custom_attributes(backend._builder)
        except Exception:
            pass

    _build_model_and_allocate(backend, SolverCls)


def _solver_step(backend: NewtonBackend, dt: float) -> None:
    """One physics substep with collision pipeline.

    When dual-solver is enabled:
      1. Primary solver (MuJoCo) handles rigid bodies
      2. Secondary solver (VBD/XPBD) handles cloth particles

    Newton solver.step() signature (newton-sim >= 1.0):
        solver.step(state_in, state_out, control, contacts, dt)
    """
    # Clear forces
    try:
        backend._state_0.clear_forces()
    except (AttributeError, Exception):
        try:
            if hasattr(backend._state_0, "body_f"):
                backend._state_0.body_f.zero_()
        except Exception:
            pass

    # Collision detection
    if backend._collision_pipeline is not None:
        try:
            backend._collision_pipeline.collide(backend._state_0)
        except TypeError:
            try:
                backend._collision_pipeline.collide(
                    backend._model, backend._state_0, backend._contacts
                )
            except Exception as exc:
                logger.debug("pipeline.collide: %s", exc)
        except Exception as exc:
            logger.debug("pipeline.collide: %s", exc)

    # Secondary solver (cloth)
    if backend._secondary_solver is not None:
        try:
            backend._secondary_solver.step(
                backend._state_0, backend._state_1, backend._control, backend._contacts, dt
            )
        except TypeError:
            backend._secondary_solver.step(
                backend._model, backend._state_0, backend._state_1, dt, backend._control
            )

    # Primary solver — try new API first
    try:
        backend._solver.step(
            backend._state_0, backend._state_1, backend._control, backend._contacts, dt
        )
    except TypeError:
        backend._solver.step(
            backend._model, backend._state_0, backend._state_1, dt, backend._control
        )


def _apply_actions(backend: NewtonBackend, actions: Any) -> None:
    """Write actions into control / state joint_act arrays."""
    wp = backend._wp

    # Convert to numpy
    if hasattr(actions, "detach"):
        actions_np = actions.detach().cpu().numpy()
    elif isinstance(actions, np.ndarray):
        actions_np = actions
    else:
        actions_np = np.asarray(actions, dtype=np.float32)

    # Write to joint_act
    target = None
    if backend._control is not None and hasattr(backend._control, "joint_act"):
        target = backend._control.joint_act
    elif backend._state_0 is not None and hasattr(backend._state_0, "joint_act"):
        target = backend._state_0.joint_act

    if target is not None:
        flat = actions_np.flatten()
        n = min(
            len(flat), target.shape[0] if hasattr(target, "shape") else len(flat)
        )
        target.assign(
            wp.array(flat[:n], dtype=wp.float32, device=backend._config.device)
        )


def _add_ground_plane(builder: Any) -> None:
    """Add a ground plane to the builder, trying multiple API variants."""
    try:
        builder.add_ground_plane()
    except Exception:
        try:
            builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.0), body=-1)
        except Exception:
            pass
