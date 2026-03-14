"""Newton Physics GPU-Accelerated Simulation Backend for strands-robots.

Newton is a GPU-native physics engine built on NVIDIA Warp that provides:
  - 7 different solver backends (MuJoCo, Featherstone, XPBD, VBD, etc.)
  - Massive parallelism: 4096+ environments running simultaneously on a single GPU
  - Differentiable simulation for gradient-based policy optimization
  - CUDA graph support for minimal Python overhead
  - Native MJCF/URDF/USD parsing with automatic asset resolution
  - CollisionPipeline with broad-phase and soft-contact margin
  - Cloth, cable, soft-body, and MPM (granular/fluid) simulation
  - Contact, IMU, and tiled camera sensors
  - Jacobian-based inverse kinematics

Usage:
    from strands_robots.newton import NewtonBackend, NewtonConfig

    config = NewtonConfig(num_envs=4096, solver="featherstone", device="cuda:0")
    backend = NewtonBackend(config)
    backend.create_world()
    backend.add_robot("go2", urdf_path="path/to/go2.urdf")
    backend.replicate(num_envs=4096)

    for _ in range(1000):
        obs = backend.get_observation("go2")
        actions = policy(obs)
        backend.step(actions)

    backend.destroy()

Differentiable simulation:
    config = NewtonConfig(enable_differentiable=True, solver="semi_implicit")
    backend = NewtonBackend(config)
    ...
    result = backend.run_diffsim(
        num_steps=100,
        loss_fn=lambda states: distance_to_target(states[-1]),
        optimize_params="initial_velocity",
        lr=0.02, iterations=200,
    )
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from strands_robots._async_utils import _resolve_coroutine

logger = logging.getLogger(__name__)

# Lazily imported
_newton_module = None
_warp_module = None

SOLVER_MAP: Dict[str, str] = {
    "mujoco": "SolverMuJoCo",
    "featherstone": "SolverFeatherstone",
    "semi_implicit": "SolverSemiImplicit",
    "xpbd": "SolverXPBD",
    "vbd": "SolverVBD",
    "style3d": "SolverStyle3D",
    "implicit_mpm": "SolverImplicitMPM",
}

RENDER_BACKENDS = {"opengl", "rerun", "viser", "null", "none"}
BROAD_PHASE_OPTIONS = {"sap", "bvh", "none"}


# Known robot definitions for procedural construction when no model file is found.
# Each entry maps a robot name (or alias) to a list of (link_length, joint_axis) tuples
# representing a serial-chain manipulator.  The first element is the base link.
_PROCEDURAL_ROBOTS: Dict[str, Dict[str, Any]] = {
    "so100": {
        "aliases": [
            "so_100",
            "so-100",
            "so100_follower",
            "so100_leader",
            "trs_so_arm100",
        ],
        "num_joints": 6,
        # (link_length, radius, axis) per revolute joint — SO-100 is a 6-DOF desktop arm
        "links": [
            (0.05, 0.025, (0.0, 1.0, 0.0)),  # base rotation (yaw)
            (0.10, 0.020, (1.0, 0.0, 0.0)),  # shoulder pitch
            (0.10, 0.018, (1.0, 0.0, 0.0)),  # elbow pitch
            (0.05, 0.015, (0.0, 1.0, 0.0)),  # wrist yaw
            (0.04, 0.012, (1.0, 0.0, 0.0)),  # wrist pitch
            (0.03, 0.010, (0.0, 1.0, 0.0)),  # wrist roll / gripper
        ],
    },
    "so101": {
        "aliases": ["so_101", "so-101", "so101_follower", "so101_leader"],
        "num_joints": 6,
        "links": [
            (0.05, 0.025, (0.0, 1.0, 0.0)),
            (0.10, 0.020, (1.0, 0.0, 0.0)),
            (0.10, 0.018, (1.0, 0.0, 0.0)),
            (0.05, 0.015, (0.0, 1.0, 0.0)),
            (0.04, 0.012, (1.0, 0.0, 0.0)),
            (0.03, 0.010, (0.0, 1.0, 0.0)),
        ],
    },
    "koch": {
        "aliases": ["koch_follower", "koch_leader", "alexander_koch"],
        "num_joints": 6,
        "links": [
            (0.05, 0.025, (0.0, 1.0, 0.0)),
            (0.13, 0.020, (1.0, 0.0, 0.0)),
            (0.13, 0.018, (1.0, 0.0, 0.0)),
            (0.06, 0.015, (0.0, 1.0, 0.0)),
            (0.05, 0.012, (1.0, 0.0, 0.0)),
            (0.03, 0.010, (0.0, 1.0, 0.0)),
        ],
    },
}

# Build an alias → canonical-name lookup
_ROBOT_ALIAS_MAP: Dict[str, str] = {}
for _canonical, _spec in _PROCEDURAL_ROBOTS.items():
    _ROBOT_ALIAS_MAP[_canonical] = _canonical
    for _alias in _spec.get("aliases", []):
        _ROBOT_ALIAS_MAP[_alias] = _canonical


def _build_procedural_robot(
    builder: Any,
    wp: Any,
    name: str,
    position: Optional[Tuple[float, float, float]] = None,
    scale: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Build a robot procedurally using Newton ModelBuilder API.

    Creates a serial-chain manipulator from the _PROCEDURAL_ROBOTS registry
    when no URDF/MJCF/USD file is available.

    Returns robot_info dict on success, None if the robot name is not in the registry.
    """
    canonical = _ROBOT_ALIAS_MAP.get(name.lower())
    if canonical is None:
        # Also try stripping common prefixes/suffixes
        for prefix in ("trs_", "bi_"):
            stripped = name.lower().removeprefix(prefix)
            canonical = _ROBOT_ALIAS_MAP.get(stripped)
            if canonical:
                break
    if canonical is None:
        return None

    spec = _PROCEDURAL_ROBOTS[canonical]
    links = spec["links"]

    base_pos = position or (0.0, 0.0, 0.0)
    joint_count_before = getattr(builder, "joint_count", 0)
    body_count_before = getattr(builder, "body_count", 0)

    # Accumulate height for each link
    current_height = base_pos[1]
    parent_body = -1  # -1 = world

    for i, (link_len, radius, axis) in enumerate(links):
        link_len_scaled = link_len * scale
        radius_scaled = radius * scale

        # Position the body at the centre of the link
        body_y = current_height + link_len_scaled * 0.5
        body_pos = (base_pos[0], body_y, base_pos[2])

        body_idx = builder.add_body(
            xform=wp.transform(body_pos, wp.quat_identity()),
            armature=0.01,
        )

        # Add a capsule shape for the link
        builder.add_shape_capsule(
            body=body_idx,
            radius=radius_scaled,
            half_height=link_len_scaled * 0.5,
        )

        # Joint connecting to parent
        joint_pos = (base_pos[0], current_height, base_pos[2])
        builder.add_joint_revolute(
            parent=parent_body,
            child=body_idx,
            parent_xform=wp.transform(joint_pos, wp.quat_identity()),
            child_xform=wp.transform((0.0, -link_len_scaled * 0.5, 0.0), wp.quat_identity()),
            axis=axis,
            limit_lower=-3.14159,
            limit_upper=3.14159,
            target_ke=100.0,
            target_kd=10.0,
        )

        parent_body = body_idx
        current_height += link_len_scaled

    joint_count_after = getattr(builder, "joint_count", 0)
    body_count_after = getattr(builder, "body_count", 0)
    num_new_joints = joint_count_after - joint_count_before
    num_new_bodies = body_count_after - body_count_before

    # Register all new joints as a single articulation (required by Newton)
    if num_new_joints > 0:
        joint_indices = list(range(joint_count_before, joint_count_after))
        try:
            builder.add_articulation(joint_indices, label=name)
        except Exception as exc:
            logger.debug("add_articulation for '%s' failed: %s", name, exc)

    logger.info(
        "Procedurally built robot '%s' (canonical: %s) — %d joints, %d bodies",
        name,
        canonical,
        num_new_joints,
        num_new_bodies,
    )

    return {
        "name": name,
        "model_path": f"procedural:{canonical}",
        "format": "procedural",
        "position": position,
        "scale": scale,
        "num_joints": num_new_joints,
        "num_bodies": num_new_bodies,
        "joint_offset": joint_count_before,
        "body_offset": body_count_before,
    }


def _ensure_newton():
    """Lazily import Newton and Warp."""
    global _newton_module, _warp_module
    if _newton_module is not None:
        return

    try:
        import warp as wp

        _warp_module = wp
        logger.debug("Warp %s loaded.", getattr(wp, "__version__", "unknown"))
    except ImportError as exc:
        raise ImportError("warp-lang is required for the Newton backend. Install with: pip install warp-lang") from exc

    try:
        import newton

        _newton_module = newton
    except ImportError as exc:
        raise ImportError(
            "newton-sim is required for the Newton backend. Install with: pip install newton-sim"
        ) from exc


def _try_resolve_model_path(name: str) -> Optional[str]:
    """Attempt to resolve a model path via strands_robots.assets."""
    try:
        from strands_robots.assets import resolve_model_path

        path = resolve_model_path(name)
        logger.debug("Resolved '%s' → %s", name, path)
        return str(path)
    except Exception:
        return None


def _try_get_newton_asset(name: str) -> Optional[str]:
    """Try to resolve a Newton bundled asset."""
    try:
        _ensure_newton()
        from newton import examples as ne

        path = ne.get_asset(name)
        if os.path.exists(path):
            return path
    except Exception:
        pass
    return None


@dataclass
class NewtonConfig:
    """Configuration for the Newton simulation backend.

    Attributes:
        num_envs: Number of parallel environments (4096+ on GPU).
        device: Warp device string ("cuda:0", "cpu").
        solver: Solver backend name (see SOLVER_MAP).
        physics_dt: Physics timestep in seconds.
        substeps: Number of substeps per frame.
        render_backend: Rendering backend.
        enable_cuda_graph: Whether to capture CUDA graphs.
        enable_differentiable: Enable gradient tracking for diffsim.
        broad_phase: Broad-phase collision algorithm.
        soft_contact_margin: Soft-contact margin distance.
        soft_contact_ke: Contact stiffness.
        soft_contact_kd: Contact damping.
        soft_contact_mu: Contact friction.
        soft_contact_restitution: Contact restitution.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    solver: str = "mujoco"
    physics_dt: float = 0.005
    substeps: int = 1
    render_backend: str = "none"
    enable_cuda_graph: bool = False
    enable_differentiable: bool = False
    broad_phase: str = "sap"
    soft_contact_margin: float = 0.5
    soft_contact_ke: float = 10000.0
    soft_contact_kd: float = 10.0
    soft_contact_mu: float = 0.5
    soft_contact_restitution: float = 0.0

    def __post_init__(self):
        if self.solver not in SOLVER_MAP:
            raise ValueError(f"Unknown solver '{self.solver}'. Options: {list(SOLVER_MAP.keys())}")
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(f"Unknown render_backend '{self.render_backend}'. Options: {RENDER_BACKENDS}")
        if self.broad_phase not in BROAD_PHASE_OPTIONS:
            raise ValueError(f"Unknown broad_phase '{self.broad_phase}'. Options: {BROAD_PHASE_OPTIONS}")
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")


class NewtonBackend:
    """GPU-accelerated physics simulation backend powered by Newton / Warp.

    Implements the same interface as IsaacSimBackend and MuJoCoSimulation
    for transparent backend swapping.

    Key features beyond MuJoCo/Isaac:
      - 7 solver backends
      - 4096+ parallel envs on a single GPU
      - Differentiable simulation (wp.Tape)
      - CUDA graph capture
      - Cloth, cable, soft-body, MPM support
      - Contact, IMU, tiled camera sensors
      - Jacobian-based IK
    """

    def __init__(self, config: Optional[NewtonConfig] = None) -> None:
        if config is None:
            config = NewtonConfig()
        self._config = config
        self._newton = None
        self._wp = None
        self._builder = None
        self._model = None
        self._solver = None
        self._state_0 = None
        self._state_1 = None
        self._control = None
        self._contacts = None
        self._collision_pipeline = None
        self._renderer = None
        self._secondary_solver = None
        self._secondary_solver_name = None
        self._default_joint_q = None
        self._default_joint_qd = None
        self._joints_per_world = 0
        self._bodies_per_world = 0
        self._dof_per_world = 0
        self._diffsim_states = None
        self._diffsim_tape = None
        self._cuda_graph = None
        self._sensors: Dict[str, Any] = {}
        self._robots: Dict[str, Dict[str, Any]] = {}
        self._cloths: Dict[str, Any] = {}
        self._cables: Dict[str, Any] = {}
        self._world_created = False
        self._replicated = False
        self._step_count = 0
        self._sim_time = 0.0
        self._gravity = None

        logger.info(
            "NewtonBackend created — solver=%s, device=%s, num_envs=%d, dt=%.4f, broad_phase=%s",
            config.solver,
            config.device,
            config.num_envs,
            config.physics_dt,
            config.broad_phase,
        )

    # ------------------------------------------------------------------
    # Initialization helpers
    # ------------------------------------------------------------------

    def _lazy_init(self) -> None:
        if self._newton is not None:
            return
        _ensure_newton()
        self._newton = _newton_module
        self._wp = _warp_module
        try:
            self._wp.init()
            logger.info("Warp initialised on '%s'.", self._config.device)
        except Exception as exc:
            logger.warning(
                "Warp init on '%s' failed (%s), fallback to 'cpu'.",
                self._config.device,
                exc,
            )
            self._config.device = "cpu"

    def _get_solver_class(self, solver_name: Optional[str] = None) -> Any:
        solver_key = solver_name or self._config.solver
        class_name = SOLVER_MAP[solver_key]
        newton = self._newton
        solver_cls = getattr(getattr(newton, "solvers", newton), class_name, None)
        if solver_cls is None:
            raise RuntimeError(f"Solver '{solver_key}' not found in newton.solvers.")
        return solver_cls

    def _ensure_world(self) -> None:
        if not self._world_created:
            raise RuntimeError("World not created. Call create_world() first.")

    def _ensure_model(self) -> None:
        if self._model is None:
            raise RuntimeError("No model finalised. Add a robot and call step().")

    def _finalize_model(self) -> None:
        """Finalise builder → Model, allocate states/solver/collision pipeline."""
        if self._model is not None:
            return
        self._ensure_world()
        newton = self._newton

        logger.info("Finalising Newton model …")
        t0 = time.perf_counter()

        SolverCls = self._get_solver_class()

        # Register custom attributes if needed (e.g., VBD/Style3D)
        if hasattr(SolverCls, "register_custom_attributes"):
            try:
                SolverCls.register_custom_attributes(self._builder)
            except Exception as exc:
                logger.debug("register_custom_attributes: %s", exc)

        # Handle cloth solvers that need special attributes
        if self._cloths and self._config.solver in ("vbd", "style3d"):
            if hasattr(SolverCls, "register_custom_attributes"):
                try:
                    SolverCls.register_custom_attributes(self._builder)
                except Exception:
                    pass

        # Finalize the model
        try:
            self._model = self._builder.finalize(
                device=self._config.device,
                requires_grad=self._config.enable_differentiable,
            )
        except TypeError:
            self._model = self._builder.finalize(device=self._config.device)

        # Set soft contact parameters
        for attr, val in [
            ("soft_contact_ke", self._config.soft_contact_ke),
            ("soft_contact_kd", self._config.soft_contact_kd),
            ("soft_contact_mu", self._config.soft_contact_mu),
            ("soft_contact_restitution", self._config.soft_contact_restitution),
        ]:
            if hasattr(self._model, attr):
                setattr(self._model, attr, val)

        # Allocate states
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()

        # Collision pipeline — Newton requires model as first positional arg
        try:
            self._collision_pipeline = newton.CollisionPipeline(
                self._model,
                broad_phase=self._config.broad_phase,
                soft_contact_margin=self._config.soft_contact_margin,
            )
            self._contacts = self._collision_pipeline.contacts()
        except Exception as exc:
            logger.warning("CollisionPipeline init failed: %s", exc)

        # Solver — Newton solvers require model as first positional arg
        try:
            self._solver = SolverCls(self._model)
        except TypeError:
            try:
                self._solver = SolverCls()
            except TypeError:
                self._solver = SolverCls(self._model, self._config.physics_dt)

        # Forward kinematics to get initial state
        try:
            newton.eval_fk(
                self._model,
                self._state_0.joint_q,
                self._state_0.joint_qd,
                self._state_0,
            )
        except Exception:
            pass

        # Store default joint state for resets
        try:
            self._default_joint_q = self._state_0.joint_q.numpy().copy()
            self._default_joint_qd = self._state_0.joint_qd.numpy().copy()
        except Exception:
            pass

        elapsed = time.perf_counter() - t0
        logger.info("Model finalised in %.3fs", elapsed)

    # ------------------------------------------------------------------
    # Dual solver
    # ------------------------------------------------------------------

    def enable_dual_solver(
        self,
        rigid_solver: str = "mujoco",
        cloth_solver: str = "vbd",
        cloth_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Enable dual-solver mode: one solver for rigid bodies, another for cloth.

        This follows Newton's cloth_franka pattern where Featherstone handles
        the robot and VBD handles the cloth, each stepping independently.
        """
        if self._model is None:
            self._finalize_model()
        try:
            SecondaryCls = self._get_solver_class(cloth_solver)
            kwargs = {}
            if cloth_solver == "vbd":
                kwargs["iterations"] = cloth_iterations
            self._secondary_solver = SecondaryCls(**kwargs) if kwargs else SecondaryCls()
            self._secondary_solver_name = cloth_solver
            logger.info(
                "Dual solver enabled: %s (rigid) + %s (cloth)",
                self._config.solver,
                cloth_solver,
            )
            return {
                "success": True,
                "message": f"Dual solver: {self._config.solver} + {cloth_solver}",
            }
        except Exception as exc:
            logger.error("Dual solver failed: %s", str(exc))
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # World creation
    # ------------------------------------------------------------------

    def create_world(
        self,
        gravity: Optional[Tuple[float, float, float]] = None,
        ground_plane: bool = True,
        up_axis: str = "y",
    ) -> Dict[str, Any]:
        """Create a new simulation world."""
        self._lazy_init()
        newton = self._newton

        # Newton's ModelBuilder expects gravity as a scalar magnitude and
        # uses up_vector (default z-up) to compute the gravity vector.
        # We accept a 3-tuple for API compat but decompose it for Newton.
        if gravity is not None:
            gravity_val = tuple(gravity)
        else:
            if up_axis == "z":
                gravity_val = (0.0, 0.0, -9.81)
            else:
                gravity_val = (0.0, -9.81, 0.0)
        self._gravity = gravity_val

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
            self._builder = newton.ModelBuilder(gravity=gravity_mag)
        except (TypeError, AttributeError):
            self._builder = newton.ModelBuilder()
        # Set scalar gravity and up_vector
        try:
            self._builder.gravity = gravity_mag
        except Exception:
            pass
        try:
            self._builder.up_vector = up_vec
            self._builder.up_axis = up_axis
        except Exception:
            pass

        self._ground_plane_requested = ground_plane
        if ground_plane:
            try:
                self._builder.add_ground_plane()
            except Exception:
                try:
                    self._builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.0), body=-1)
                except Exception:
                    pass

        self._world_created = True
        self._step_count = 0
        self._sim_time = 0.0

        return {
            "success": True,
            "message": "World created.",
            "world_info": {
                "gravity": gravity_val,
                "ground_plane": ground_plane,
                "solver": self._config.solver,
                "device": self._config.device,
                "physics_dt": self._config.physics_dt,
                "broad_phase": self._config.broad_phase,
            },
        }

    def _recreate_builder(self) -> None:
        """Recreate the model builder to clear any partial state from failed loads."""
        newton = self._newton
        gravity_mag = getattr(self._builder, "gravity", -9.81)
        up_vec = getattr(self._builder, "up_vector", (0.0, 0.0, 1.0))
        try:
            self._builder = newton.ModelBuilder(gravity=gravity_mag)
        except (TypeError, AttributeError):
            self._builder = newton.ModelBuilder()
        try:
            self._builder.gravity = gravity_mag
        except Exception:
            pass
        try:
            self._builder.up_vector = up_vec
        except Exception:
            pass
        if getattr(self, "_ground_plane_requested", True):
            try:
                self._builder.add_ground_plane()
            except Exception:
                pass
        logger.debug("Builder recreated (cleared partial state)")

    # ------------------------------------------------------------------
    # Add robot
    # ------------------------------------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: Optional[str] = None,
        usd_path: Optional[str] = None,
        data_config: Optional[str] = None,
        position: Optional[Tuple[float, float, float]] = None,
        scale: float = 1.0,
    ) -> Dict[str, Any]:
        """Add a robot from URDF, MJCF, or USD.

        Phase 1.4: Now supports USD loading via add_usd().
        """
        self._ensure_world()

        # Resolve model path
        model_path = urdf_path or usd_path or data_config
        if model_path is None:
            model_path = name

        # Try to resolve path
        if not Path(str(model_path)).exists():
            resolved = _try_resolve_model_path(model_path)
            if resolved:
                model_path = resolved
            else:
                asset = _try_get_newton_asset(model_path)
                if asset:
                    model_path = asset

        model_path_obj = Path(str(model_path))
        if not model_path_obj.exists():
            # Try as a known robot name
            asset = _try_get_newton_asset(name)
            if asset:
                model_path = asset
                model_path_obj = Path(model_path)

        suffix = model_path_obj.suffix.lower() if model_path_obj.exists() else ""
        logger.info("Adding robot '%s' from %s", name, model_path)

        wp = self._wp
        xform = None
        if position is not None:
            xform = wp.transform(position, wp.quat_identity())

        try:
            joint_count_before = getattr(self._builder, "joint_count", 0)
            body_count_before = getattr(self._builder, "body_count", 0)

            fmt = suffix
            parse_kwargs: Dict[str, Any] = {}
            used_procedural = False

            if suffix in (".usda", ".usd"):
                self._builder.add_usd(
                    str(model_path),
                    xform=xform,
                    scale=scale,
                    **parse_kwargs,
                )
            elif suffix == ".urdf":
                parse_kwargs.update(
                    {
                        "collapse_fixed_joints": True,
                    }
                )
                # NOTE: armature, stiffness, damping, enable_self_collisions, floating
                # are NOT accepted by Newton's add_urdf() — they must be set on the
                # model/builder after loading. Only collapse_fixed_joints is supported.
                try:
                    self._builder.add_urdf(
                        str(model_path),
                        xform=xform,
                        scale=scale,
                        **parse_kwargs,
                    )
                except Exception as urdf_exc:
                    logger.warning(
                        "URDF load failed for '%s': %s — trying procedural",
                        model_path,
                        urdf_exc,
                    )
                    # Recreate builder to avoid partial state from failed URDF parse
                    self._recreate_builder()
                    proc_info = _build_procedural_robot(
                        self._builder,
                        self._wp,
                        name,
                        position=position,
                        scale=scale,
                    )
                    if proc_info is not None:
                        used_procedural = True
                        fmt = "procedural"
                    else:
                        raise urdf_exc
            elif suffix == ".xml":
                try:
                    self._builder.add_mjcf(
                        str(model_path),
                        xform=xform,
                        scale=scale,
                    )
                except Exception as mjcf_exc:
                    logger.warning(
                        "MJCF load failed for '%s': %s — trying procedural",
                        model_path,
                        mjcf_exc,
                    )
                    # Recreate builder to avoid partial state from failed MJCF parse
                    self._recreate_builder()
                    proc_info = _build_procedural_robot(
                        self._builder,
                        self._wp,
                        name,
                        position=position,
                        scale=scale,
                    )
                    if proc_info is not None:
                        used_procedural = True
                        fmt = "procedural"
                    else:
                        raise mjcf_exc
            else:
                # Try URDF first, then MJCF, then procedural fallback
                loaded = False
                for loader_name, loader_fn in [
                    ("urdf", self._builder.add_urdf),
                    ("mjcf", self._builder.add_mjcf),
                ]:
                    try:
                        loader_fn(str(model_path), xform=xform, scale=scale)
                        loaded = True
                        break
                    except Exception:
                        continue

                if not loaded:
                    # Recreate builder to avoid partial state from failed parsers
                    self._recreate_builder()
                    # Fallback: build robot procedurally from known definitions
                    proc_info = _build_procedural_robot(
                        self._builder,
                        self._wp,
                        name,
                        position=position,
                        scale=scale,
                    )
                    if proc_info is not None:
                        used_procedural = True
                        fmt = "procedural"
                    else:
                        raise FileNotFoundError(
                            f"No model file found for '{name}' and no procedural "
                            f"definition available. Provide urdf_path, usd_path, or "
                            f"add '{name}' to _PROCEDURAL_ROBOTS registry."
                        )

            if used_procedural:
                robot_info = proc_info
            else:
                joint_count_after = getattr(self._builder, "joint_count", 0)
                body_count_after = getattr(self._builder, "body_count", 0)
                num_new_joints = joint_count_after - joint_count_before
                num_new_bodies = body_count_after - body_count_before

                robot_info = {
                    "name": name,
                    "model_path": str(model_path),
                    "format": fmt,
                    "position": position,
                    "scale": scale,
                    "num_joints": num_new_joints,
                    "num_bodies": num_new_bodies,
                    "joint_offset": joint_count_before,
                    "body_offset": body_count_before,
                }

            self._robots[name] = robot_info

            # Invalidate model so it gets rebuilt
            self._model = None
            self._replicated = False
            joint_count_after = getattr(self._builder, "joint_count", 0)
            body_count_after = getattr(self._builder, "body_count", 0)
            self._joints_per_world = joint_count_after
            self._bodies_per_world = body_count_after

            num_j = robot_info.get("num_joints", 0)
            num_b = robot_info.get("num_bodies", 0)
            return {
                "success": True,
                "num_joints": num_j,
                "robot_info": robot_info,
                "message": f"Robot '{name}' added ({num_j} joints, {num_b} bodies)",
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Add cloth
    # ------------------------------------------------------------------

    def add_cloth(
        self,
        name: str,
        vertices: Optional[np.ndarray] = None,
        indices: Optional[np.ndarray] = None,
        usd_path: Optional[str] = None,
        usd_prim_path: Optional[str] = None,
        position: Optional[Tuple[float, float, float]] = None,
        rotation: Optional[Tuple[float, float, float, float]] = None,
        velocity: Optional[Tuple[float, float, float]] = None,
        density: float = 100.0,
        scale: float = 1.0,
        tri_ke: float = 10000.0,
        tri_ka: float = 10000.0,
        tri_kd: float = 10.0,
        edge_ke: float = 100.0,
        edge_kd: float = 0.0,
        particle_radius: float = 0.01,
    ) -> Dict[str, Any]:
        """Add a cloth mesh to the simulation.

        Supports either raw vertices/indices or loading from USD.
        Uses VBD solver for cloth simulation.
        """
        self._ensure_world()
        wp = self._wp

        # Load from USD if path provided
        if usd_path and vertices is None:
            try:
                from newton import usd as newton_usd
                from pxr import Usd

                stage = Usd.Stage.Open(usd_path)
                if usd_prim_path:
                    prim = stage.GetPrimAtPath(usd_prim_path)
                else:
                    prim = stage.GetDefaultPrim()
                    usd_prim_path = str(prim.GetPath())
                mesh = newton_usd.get_mesh(stage, prim)
                vertices = mesh.vertices
                indices = mesh.indices
            except Exception as exc:
                return {"success": False, "message": f"USD cloth load failed: {exc}"}

        if vertices is None or indices is None:
            return {
                "success": False,
                "message": "vertices and indices required for cloth.",
            }

        # Convert vertices to warp vec3
        v = [wp.vec3(*vert) for vert in vertices]

        rot = rotation if rotation is not None else wp.quat_identity()
        if isinstance(rotation, (list, tuple)) and not isinstance(rotation, type(wp.quat_identity())):
            if hasattr(wp, "quat"):
                rot = wp.quat(*rotation)

        self._builder.add_cloth_mesh(
            pos=position or (0.0, 0.0, 0.0),
            rot=rot,
            scale=scale,
            vel=velocity or (0.0, 0.0, 0.0),
            vertices=v,
            indices=indices.flatten() if hasattr(indices, "flatten") else indices,
            density=density,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            edge_ke=edge_ke,
            edge_kd=edge_kd,
            particle_radius=particle_radius,
        )

        cloth_info = {
            "name": name,
            "num_vertices": len(vertices),
            "num_triangles": len(indices) // 3,
            "density": density,
            "particle_radius": particle_radius,
        }
        self._cloths[name] = cloth_info
        self._model = None  # Invalidate

        logger.info(
            "Cloth '%s' added — %d vertices, %d triangles",
            name,
            len(vertices),
            len(indices) // 3,
        )
        return {
            "success": True,
            "message": f"Cloth '{name}' added",
            "cloth_info": cloth_info,
        }

    # ------------------------------------------------------------------
    # Add cable
    # ------------------------------------------------------------------

    def add_cable(
        self,
        name: str,
        start: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        end: Tuple[float, float, float] = (1.0, 0.0, 0.0),
        num_segments: int = 10,
        radius: float = 0.01,
        density: float = 100.0,
        bend_stiffness: float = 2.0,
        bend_damping: float = 0.5,
    ) -> Dict[str, Any]:
        """Add a cable/rope to the simulation using chained capsule bodies."""
        self._ensure_world()
        wp = self._wp

        try:
            start_vec = np.array(start)
            end_vec = np.array(end)
            direction = end_vec - start_vec
            length = np.linalg.norm(direction)
            seg_length = length / num_segments

            for i in range(num_segments):
                t = (i + 0.5) / num_segments
                pos = tuple(start_vec + direction * t)

                self._builder.add_body(
                    xform=wp.transform(pos, wp.quat_identity()),
                    armature=0.01,
                )
                self._builder.add_shape_capsule(
                    body=self._builder.body_count - 1,
                    radius=radius,
                    half_height=seg_length * 0.5,
                )

                if i > 0:
                    self._builder.add_joint_ball(
                        parent=self._builder.body_count - 2,
                        child=self._builder.body_count - 1,
                    )

            self._cables[name] = {
                "name": name,
                "num_segments": num_segments,
                "length": length,
                "radius": radius,
            }
            self._model = None
            logger.info(
                "Cable '%s' added — %d segments, length=%.2f.",
                name,
                num_segments,
                length,
            )
            return {"success": True, "message": f"Cable '{name}' added"}
        except Exception as exc:
            return {"success": False, "message": f"Cable creation failed: {exc}"}

    # ------------------------------------------------------------------
    # Add particles
    # ------------------------------------------------------------------

    def add_particles(
        self,
        name: str,
        positions: Union[np.ndarray, List],
        velocities: Optional[Union[np.ndarray, List]] = None,
        mass: float = 1.0,
        radius: float = 0.01,
    ) -> Dict[str, Any]:
        """Add particle-based entities (for MPM granular/fluid simulation)."""
        self._ensure_world()
        wp = self._wp

        if isinstance(positions, np.ndarray):
            positions = positions.tolist()

        count = 0
        for i, pos in enumerate(positions):
            vel = velocities[i] if velocities is not None and hasattr(velocities, "__iter__") else (0.0, 0.0, 0.0)
            if isinstance(vel, np.ndarray):
                vel = tuple(vel)
            try:
                self._builder.add_particle(
                    pos=wp.vec3(*tuple(pos)),
                    vel=wp.vec3(*tuple(vel)),
                    mass=mass,
                )
                count += 1
            except Exception as exc:
                logger.debug("add_particle failed at %d: %s", i, exc)

        self._model = None
        return {"success": True, "message": f"{count} particles added", "count": count}

    # ------------------------------------------------------------------
    # Replicate
    # ------------------------------------------------------------------

    def replicate(self, num_envs: Optional[int] = None) -> Dict[str, Any]:
        """Clone the scene across parallel environments."""
        self._ensure_world()

        if not self._robots and not self._cloths:
            return {"success": False, "message": "Nothing to replicate."}

        if num_envs is not None:
            self._config.num_envs = num_envs
        else:
            num_envs = self._config.num_envs

        newton = self._newton
        logger.info("Replicating scene × %d environments …", num_envs)
        t0 = time.perf_counter()

        try:
            spacing = 2.0
            main_builder = self._builder

            # Store per-world counts
            self._joints_per_world = getattr(main_builder, "joint_count", 0)
            self._bodies_per_world = getattr(main_builder, "body_count", 0)
            self._dof_per_world = getattr(main_builder, "joint_count", 0)

            # Use Newton's replicate — instance method on a new builder
            replicated_builder = newton.ModelBuilder()
            replicated_builder.replicate(
                main_builder,
                world_count=num_envs,
                spacing=(spacing, 0.0, 0.0),
            )
            self._builder = replicated_builder

            # Add ground plane to replicated builder
            if self._ground_plane_requested:
                try:
                    self._builder.add_ground_plane()
                except Exception:
                    try:
                        self._builder.add_shape_plane(plane=(0.0, 1.0, 0.0, 0.0), body=-1)
                    except Exception:
                        pass

            # Get solver class and register custom attributes
            SolverCls = self._get_solver_class()
            if hasattr(SolverCls, "register_custom_attributes"):
                try:
                    SolverCls.register_custom_attributes(self._builder)
                except Exception:
                    pass

            # Finalize replicated model
            try:
                self._model = self._builder.finalize(
                    device=self._config.device,
                    requires_grad=self._config.enable_differentiable,
                )
            except TypeError:
                self._model = self._builder.finalize(device=self._config.device)

            # Set soft contact parameters
            for attr, val in [
                ("soft_contact_ke", self._config.soft_contact_ke),
                ("soft_contact_kd", self._config.soft_contact_kd),
                ("soft_contact_mu", self._config.soft_contact_mu),
                ("soft_contact_restitution", self._config.soft_contact_restitution),
            ]:
                if hasattr(self._model, attr):
                    setattr(self._model, attr, val)

            # Allocate states
            self._state_0 = self._model.state()
            self._state_1 = self._model.state()
            self._control = self._model.control()

            # Collision pipeline — Newton requires model as first positional arg
            try:
                self._collision_pipeline = newton.CollisionPipeline(
                    self._model,
                    broad_phase=self._config.broad_phase,
                    soft_contact_margin=self._config.soft_contact_margin,
                )
                self._contacts = self._collision_pipeline.contacts()
            except Exception:
                pass

            # Solver — Newton solvers require model as first positional arg
            try:
                self._solver = SolverCls(self._model)
            except TypeError:
                try:
                    self._solver = SolverCls()
                except TypeError:
                    self._solver = SolverCls(self._model, self._config.physics_dt)

            # Forward kinematics
            try:
                newton.eval_fk(
                    self._model,
                    self._state_0.joint_q,
                    self._state_0.joint_qd,
                    self._state_0,
                )
            except Exception:
                pass

            # Store default joint state
            try:
                self._default_joint_q = self._state_0.joint_q.numpy().copy()
                self._default_joint_qd = self._state_0.joint_qd.numpy().copy()
            except (Exception, AttributeError):
                pass

            self._replicated = True
            elapsed = time.perf_counter() - t0
            logger.info("Replicated in %.3fs", elapsed)
            return {
                "success": True,
                "message": f"Replicated × {num_envs} in {elapsed:.3f}s",
            }
        except Exception as exc:
            logger.warning("replicate() failed: %s, falling back to _finalize_model()", exc)
            self._finalize_model()
            return {
                "success": True,
                "message": f"Finalized (replicate fallback): {exc}",
            }

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        """Reset simulation state. Supports per-environment reset."""
        if self._model is None:
            return {"success": False, "message": "Model not finalised."}

        wp = self._wp

        if self._default_joint_q is None or self._default_joint_qd is None:
            return {"success": False, "message": "No default joint state stored."}

        try:
            q_default = self._default_joint_q
            qd_default = self._default_joint_qd

            if env_ids is None:
                # Full reset
                self._state_0.joint_q.assign(wp.array(q_default, dtype=wp.float32, device=self._config.device))
                self._state_0.joint_qd.assign(wp.array(qd_default, dtype=wp.float32, device=self._config.device))
            else:
                # Per-environment reset
                current_q = self._state_0.joint_q.numpy()
                current_qd = self._state_0.joint_qd.numpy()

                if self._joints_per_world > 0 and len(env_ids) > 0:
                    for env_id in env_ids:
                        q_start = env_id * self._joints_per_world
                        q_end = q_start + self._joints_per_world
                        current_q[q_start:q_end] = q_default[: self._joints_per_world]
                        current_qd[q_start:q_end] = qd_default[: self._joints_per_world]
                else:
                    logger.warning(
                        "Cannot per-env reset: joints_per_world=%d",
                        self._joints_per_world,
                    )

                self._state_0.joint_q.assign(wp.array(current_q, dtype=wp.float32, device=self._config.device))
                self._state_0.joint_qd.assign(wp.array(current_qd, dtype=wp.float32, device=self._config.device))

            # Re-evaluate FK
            if hasattr(self._newton, "eval_fk"):
                self._newton.eval_fk(
                    self._model,
                    self._state_0.joint_q,
                    self._state_0.joint_qd,
                    self._state_0,
                )

            self._step_count = 0
            self._sim_time = 0.0
            return {"success": True, "message": "Reset complete"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, actions: Optional[Any] = None) -> Dict[str, Any]:
        """Advance simulation by one frame (with substeps)."""
        if self._model is None:
            self._finalize_model()

        wp = self._wp
        dt = self._config.physics_dt

        if actions is not None:
            self._apply_actions(actions)

        try:
            if self._config.enable_cuda_graph and self._cuda_graph is not None:
                # Use captured graph
                wp.capture_launch(self._cuda_graph)
                self._state_0, self._state_1 = self._state_1, self._state_0
                self._step_count += 1
                self._sim_time += dt * self._config.substeps
            elif self._config.enable_differentiable:
                # Differentiable path: no CUDA graph, explicit substeps
                for _ in range(self._config.substeps):
                    self._solver_step(dt)
                    self._state_0, self._state_1 = self._state_1, self._state_0
                self._step_count += 1
                self._sim_time += dt * self._config.substeps
            else:
                # Standard path
                if self._config.enable_cuda_graph and self._cuda_graph is None and "cuda" in str(self._config.device):
                    # Capture CUDA graph on first step
                    with wp.ScopedCapture(device=self._config.device) as capture:
                        for _ in range(self._config.substeps):
                            self._solver_step(dt)
                            self._state_0, self._state_1 = self._state_1, self._state_0
                    self._cuda_graph = capture.graph
                    logger.info("CUDA graph captured via ScopedCapture.")
                    self._step_count += 1
                    self._sim_time += dt * self._config.substeps
                else:
                    for _ in range(self._config.substeps):
                        self._solver_step(dt)
                        self._state_0, self._state_1 = self._state_1, self._state_0
                    self._step_count += 1
                    self._sim_time += dt * self._config.substeps

            return {
                "success": True,
                "sim_time": self._sim_time,
                "step_count": self._step_count,
            }
        except Exception as exc:
            logger.warning("step() failed: %s", exc)
            return {"success": False, "sim_time": self._sim_time, "error": str(exc)}

    def _solver_step(self, dt: float) -> None:
        """One physics substep with collision pipeline.

        When dual-solver is enabled:
          1. Primary solver (MuJoCo) handles rigid bodies
          2. Secondary solver (VBD/XPBD) handles cloth particles

        Newton solver.step() signature (newton-sim >= 1.0):
            solver.step(state_in, state_out, control, contacts, dt)
        """
        # Clear forces (body_f, particle_f exist but joint_f may not)
        try:
            self._state_0.clear_forces()
        except (AttributeError, Exception):
            # Fallback: manually zero body_f if available
            try:
                if hasattr(self._state_0, "body_f"):
                    self._state_0.body_f.zero_()
            except Exception:
                pass

        # Collision detection
        if self._collision_pipeline is not None:
            try:
                self._collision_pipeline.collide(self._state_0)
            except TypeError:
                try:
                    self._collision_pipeline.collide(self._model, self._state_0, self._contacts)
                except Exception as exc:
                    logger.debug("pipeline.collide: %s", exc)
            except Exception as exc:
                logger.debug("pipeline.collide: %s", exc)

        # Secondary solver (cloth)
        if self._secondary_solver is not None:
            try:
                self._secondary_solver.step(self._state_0, self._state_1, self._control, self._contacts, dt)
            except TypeError:
                self._secondary_solver.step(self._model, self._state_0, self._state_1, dt, self._control)

        # Primary solver — try new API first (state_in, state_out, control, contacts, dt)
        try:
            self._solver.step(self._state_0, self._state_1, self._control, self._contacts, dt)
        except TypeError:
            # Fallback to old API (model, state_in, state_out, dt, control)
            self._solver.step(self._model, self._state_0, self._state_1, dt, self._control)

    def _apply_actions(self, actions: Any) -> None:
        """Write actions into control / state joint_act arrays."""
        wp = self._wp

        # Convert to numpy
        if hasattr(actions, "detach"):
            actions_np = actions.detach().cpu().numpy()
        elif isinstance(actions, np.ndarray):
            actions_np = actions
        else:
            actions_np = np.asarray(actions, dtype=np.float32)

        # Write to joint_act
        target = None
        if self._control is not None and hasattr(self._control, "joint_act"):
            target = self._control.joint_act
        elif self._state_0 is not None and hasattr(self._state_0, "joint_act"):
            target = self._state_0.joint_act

        if target is not None:
            flat = actions_np.flatten()
            n = min(len(flat), target.shape[0] if hasattr(target, "shape") else len(flat))
            target.assign(wp.array(flat[:n], dtype=wp.float32, device=self._config.device))

    # ------------------------------------------------------------------
    # Differentiable simulation
    # ------------------------------------------------------------------

    def run_diffsim(
        self,
        num_steps: int = 100,
        loss_fn: Optional[Callable] = None,
        lr: float = 0.02,
        iterations: int = 20,
        optimize_param: str = "joint_qd",
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Run differentiable simulation with gradient-based optimization.

        Phase 1.2: Full wp.Tape integration following Newton's diffsim_ball pattern.

        Args:
            num_steps: Sim steps per iteration.
            loss_fn: Loss function taking list of states.
            lr: Learning rate.
            iterations: Optimisation iterations.
            optimize_param: Parameter to optimise.
            verbose: Print progress.
        """
        if self._model is None:
            self._finalize_model()

        if not self._config.enable_differentiable:
            return {"success": False, "message": "enable_differentiable=True required."}

        wp = self._wp
        newton = self._newton
        dt = self._config.physics_dt

        loss_history = []

        # Get parameter to optimize
        if optimize_param == "joint_qd":
            param = self._state_0.joint_qd
        elif optimize_param == "joint_q":
            param = self._state_0.joint_q
        elif optimize_param == "particle_qd":
            param = self._state_0.particle_qd
        elif optimize_param == "particle_q":
            param = self._state_0.particle_q
        elif optimize_param == "joint_act":
            param = self._control.joint_act if hasattr(self._control, "joint_act") else self._state_0.joint_act
        else:
            param = getattr(self._state_0, optimize_param, None)
            if param is None:
                return {
                    "success": False,
                    "message": f"Unknown parameter: {optimize_param}",
                }

        param.requires_grad = True

        for iteration in range(iterations):
            tape = wp.Tape()
            states = [self._model.state()]

            # Create collision pipeline once per iteration (not per timestep)
            diffsim_pipeline = None
            if self._collision_pipeline is not None:
                try:
                    diffsim_pipeline = newton.CollisionPipeline(
                        self._model,
                        soft_contact_margin=self._config.soft_contact_margin,
                    )
                except Exception:
                    pass

            with tape:
                for t in range(num_steps):
                    current = states[-1]
                    next_state = self._model.state()

                    if hasattr(current, "clear_forces"):
                        current.clear_forces()

                    # Collision — reuse pipeline allocated above
                    if diffsim_pipeline is not None:
                        try:
                            contacts = diffsim_pipeline.contacts()
                            diffsim_pipeline.collide(self._model, current, contacts)
                        except Exception:
                            pass

                    self._solver.step(self._model, current, next_state, dt, self._control)
                    states.append(next_state)

                if loss_fn is not None:
                    loss = loss_fn(states)
                else:
                    loss = wp.zeros(1, dtype=wp.float32, requires_grad=True)

            tape.backward(loss)
            current_loss = float(loss.numpy()[0]) if hasattr(loss, "numpy") else float(loss)
            loss_history.append(current_loss)

            # Gradient descent
            param_np = param.numpy()
            grad_np = tape.grad(param).numpy() if tape.grad(param) is not None else np.zeros_like(param_np)
            param_np -= lr * grad_np
            param.assign(wp.array(param_np, dtype=wp.float32, device=self._config.device))

            if verbose:
                logger.info("Iteration %d: loss=%.6f", iteration, current_loss)

            tape.zero()

        return {
            "success": True,
            "iterations": iterations,
            "final_loss": loss_history[-1] if loss_history else 0.0,
            "loss_history": loss_history,
        }

    # ------------------------------------------------------------------
    # Sensors
    # ------------------------------------------------------------------

    def add_sensor(
        self,
        name: str,
        sensor_type: str = "contact",
        body_name: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Add a sensor to the simulation.

        Args:
            name: Sensor name.
            sensor_type: "contact", "imu", or "tiled_camera".
            body_name: Body to attach sensor to.
        """
        if self._model is None:
            self._finalize_model()

        try:
            from newton.sensors import SensorContact, SensorIMU, SensorTiledCamera

            if sensor_type == "contact":
                shapes = kwargs.get("shapes", [0])
                sensor = SensorContact(sensing_obj_shapes=shapes, verbose=kwargs.get("verbose", False))
            elif sensor_type == "imu":
                sensor = SensorIMU(**kwargs)
            elif sensor_type == "tiled_camera":
                sensor = SensorTiledCamera(
                    width=kwargs.get("width", 640),
                    height=kwargs.get("height", 480),
                    **{k: v for k, v in kwargs.items() if k not in ("width", "height")},
                )
            else:
                return {
                    "success": False,
                    "message": f"Unknown sensor type: {sensor_type}",
                }

            self._sensors[name] = {"sensor": sensor, "type": sensor_type}
            return {
                "success": True,
                "message": f"Sensor '{name}' added ({sensor_type})",
            }
        except ImportError:
            return {"success": False, "message": "newton.sensors not available"}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    def read_sensor(self, name: str) -> Dict[str, Any]:
        """Read data from a named sensor."""
        if name not in self._sensors:
            return {"success": False, "message": f"Sensor '{name}' not found."}

        try:
            sensor_info = self._sensors[name]
            sensor = sensor_info["sensor"]
            data = sensor.evaluate(self._model, self._state_0, self._contacts)
            return {"success": True, "data": data}
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Inverse Kinematics
    # ------------------------------------------------------------------

    def solve_ik(
        self,
        robot_name: str,
        target_position: Tuple[float, float, float],
        target_rotation: Optional[Tuple[float, float, float, float]] = None,
        end_effector_body: Optional[int] = None,
        max_iterations: int = 100,
        tolerance: float = 0.001,
    ) -> Dict[str, Any]:
        """Solve inverse kinematics using Jacobian-based method.

        Phase 4: Following Newton's cloth_franka IK pattern.
        """
        if self._model is None:
            self._finalize_model()

        if robot_name not in self._robots:
            return {"success": False, "message": f"Robot '{robot_name}' not found."}

        wp = self._wp
        newton = self._newton
        target = np.array(target_position)

        try:
            for i in range(max_iterations):
                # Forward kinematics
                newton.eval_fk(
                    self._model,
                    self._state_0.joint_q,
                    self._state_0.joint_qd,
                    self._state_0,
                )
                body_q = self._state_0.body_q.numpy()

                # Get end effector position
                ee_idx = end_effector_body or (self._bodies_per_world - 1)
                ee_pos = body_q[ee_idx][:3]
                error = target - ee_pos
                error_norm = float(np.linalg.norm(error))

                if error_norm < tolerance:
                    return {
                        "success": True,
                        "iterations": i + 1,
                        "error": error_norm,
                        "joint_positions": self._state_0.joint_q.numpy().copy(),
                    }

                # Gradient-based IK step
                joint_q = self._state_0.joint_q.numpy().copy()
                step_size = min(0.1, error_norm)
                delta_q = np.zeros(len(joint_q), dtype=np.float32)

                # Numerical Jacobian
                for j in range(min(len(joint_q), self._joints_per_world)):
                    q_plus = joint_q.copy()
                    q_plus[j] += 0.001
                    temp_state = self._model.state()
                    temp_state.joint_q.assign(wp.array(q_plus, dtype=wp.float32, device=self._config.device))
                    temp_state.joint_qd.assign(self._state_0.joint_qd)
                    newton.eval_fk(self._model, temp_state.joint_q, temp_state.joint_qd, temp_state)
                    body_out = temp_state.body_q.numpy()
                    new_pos = body_out[ee_idx][:3]
                    jac_col = (new_pos - ee_pos) / 0.001
                    delta_q[j] = np.dot(jac_col, error) * step_size

                current_q = joint_q + delta_q
                self._state_0.joint_q.assign(wp.array(current_q, dtype=wp.float32, device=self._config.device))

            return {
                "success": True,
                "iterations": max_iterations,
                "error": float(np.linalg.norm(error)),
                "joint_positions": self._state_0.joint_q.numpy().copy(),
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Observation
    # ------------------------------------------------------------------

    def get_observation(self, robot_name: Optional[str] = None) -> Dict[str, Any]:
        """Return current observation for one or all robots."""
        if self._model is None:
            return {}

        observations = {}

        if robot_name:
            robots = {robot_name: self._robots.get(robot_name, {})}
        else:
            robots = self._robots

        try:
            for rname, rinfo in robots.items():
                obs: Dict[str, Any] = {}
                offset = rinfo.get("joint_offset", 0)
                n_joints = rinfo.get("num_joints", self._joints_per_world)

                try:
                    jq = self._state_0.joint_q.numpy().copy()
                    obs["joint_q"] = jq[offset : offset + n_joints] if n_joints > 0 else jq
                except Exception:
                    pass

                try:
                    jqd = self._state_0.joint_qd.numpy().copy()
                    obs["joint_qd"] = jqd[offset : offset + n_joints] if n_joints > 0 else jqd
                except Exception:
                    pass

                try:
                    obs["body_q"] = self._state_0.body_q.numpy().copy()
                except Exception:
                    pass

                try:
                    obs["body_qd"] = self._state_0.body_qd.numpy().copy()
                except Exception:
                    pass

                # Particle state (for cloth/MPM)
                if hasattr(self._model, "particle_count") and self._model.particle_count > 0:
                    try:
                        obs["particle_q"] = self._state_0.particle_q.numpy().copy()
                        obs["particle_qd"] = self._state_0.particle_qd.numpy().copy()
                    except Exception:
                        pass

                observations[rname] = obs

            return {
                "success": True,
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "observations": observations,
            }
        except Exception as exc:
            return {"success": False, "observations": observations, "error": str(exc)}

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(
        self,
        camera_name: Optional[str] = None,
        width: int = 1024,
        height: int = 768,
    ) -> Dict[str, Any]:
        """Render current scene state."""
        if self._model is None:
            return {"success": False, "message": "No model to render."}

        newton = self._newton
        backend = self._config.render_backend

        if self._renderer is None:
            try:
                viewer_mod = getattr(newton, "viewer", None)
                if viewer_mod is None:
                    viewer_mod = newton

                if backend == "opengl":
                    self._renderer = viewer_mod.ViewerGL()
                elif backend == "rerun":
                    self._renderer = viewer_mod.ViewerRerun()
                elif backend == "viser":
                    self._renderer = viewer_mod.ViewerViser()
                else:
                    self._renderer = viewer_mod.ViewerNull()

                if hasattr(self._renderer, "set_model"):
                    self._renderer.set_model(self._model)
                logger.info("Renderer initialised: %s", backend)
            except Exception as exc:
                logger.warning("Renderer init failed: %s", exc)
                return {"success": False, "message": str(exc)}

        try:
            if hasattr(self._renderer, "begin_frame"):
                self._renderer.begin_frame(self._sim_time)
            if hasattr(self._renderer, "log_state"):
                self._renderer.log_state(self._model, self._state_0)
            if hasattr(self._renderer, "end_frame"):
                self._renderer.end_frame()

            image = None
            if hasattr(self._renderer, "get_pixels"):
                image = self._renderer.get_pixels(width=width, height=height)
                logger.debug("Rendered frame %dx%d", width, height)

            return {
                "success": True,
                "backend": backend,
                "image": image,
                "sim_time": self._sim_time,
            }
        except Exception as exc:
            return {"success": False, "message": str(exc)}

    # ------------------------------------------------------------------
    # Policy loop
    # ------------------------------------------------------------------

    def run_policy(
        self,
        robot_name: str,
        policy_provider: str = "mock",
        instruction: str = "",
        duration: float = 10.0,
        record_video: Optional[str] = None,
        video_fps: int = 30,
        video_width: int = 1024,
        video_height: int = 768,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Run a policy loop for the given robot."""
        if robot_name not in self._robots:
            return {"success": False, "message": f"Robot '{robot_name}' not found."}

        if self._model is None:
            self._finalize_model()

        policy = self._create_policy(robot_name, policy_provider, instruction, **kwargs)
        dt = self._config.physics_dt * self._config.substeps
        num_steps = int(math.ceil(duration / dt))

        # Video recording setup
        writer = None
        frame_count = 0
        if record_video:
            import imageio

            os.makedirs(os.path.dirname(record_video) or ".", exist_ok=True)
            writer = imageio.get_writer(record_video, fps=video_fps, quality=8)
            frame_interval = max(1, int(1.0 / (video_fps * dt)))

        trajectory = []
        errors = []
        t0 = time.perf_counter()

        for i in range(num_steps):
            obs = self.get_observation(robot_name)
            try:
                coro_or_result = policy.get_actions(obs.get("observations", {}).get(robot_name, {}), instruction)
                actions = _resolve_coroutine(coro_or_result)
                self.step(actions)
                trajectory.append({"step": i, "sim_time": self._sim_time, "observation": obs})
            except Exception as exc:
                errors.append(str(exc))

            if writer and i % frame_interval == 0:
                r = self.render(width=video_width, height=video_height)
                img = r.get("image")
                if img is not None:
                    writer.append_data(img)
                    frame_count += 1

        wall_time = time.perf_counter() - t0

        result = {
            "success": True,
            "steps_executed": self._step_count,
            "sim_time": self._sim_time,
            "wall_time": wall_time,
            "realtime_factor": (float(self._sim_time / wall_time) if wall_time > 0 else 0.0),
            "trajectory": trajectory,
            "errors": errors,
        }

        if writer:
            writer.close()
            size_kb = os.path.getsize(record_video) // 1024 if os.path.exists(record_video) else 0
            result["video"] = {
                "path": record_video,
                "frames": frame_count,
                "fps": video_fps,
                "size_kb": size_kb,
            }

        return result

    def _create_policy(
        self,
        robot_name: str,
        policy_provider: str,
        instruction: str = "",
        **kwargs: Any,
    ):
        """Create a policy instance for the given robot."""
        if policy_provider == "mock":
            # Create a simple mock policy
            n_joints = self._robots.get(robot_name, {}).get("num_joints", 0)

            class _Mock:
                def get_actions(self, obs, instr):
                    return np.zeros(n_joints, dtype=np.float32)

            return _Mock()

        try:
            from strands_robots.policies import create_policy

            return create_policy(
                provider=policy_provider,
                robot_name=robot_name,
                instruction=instruction,
                **kwargs,
            )
        except Exception:
            # Fallback to mock
            n_joints = self._robots.get(robot_name, {}).get("num_joints", 0)

            class _Mock:
                def get_actions(self, obs, instr):
                    return np.zeros(n_joints, dtype=np.float32)

            return _Mock()

    # ------------------------------------------------------------------
    # State / Destroy
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return full simulation state."""
        state_data = {}
        for attr in (
            "joint_q",
            "joint_qd",
            "body_q",
            "body_qd",
            "particle_q",
            "particle_qd",
        ):
            if self._state_0 is not None:
                try:
                    arr = getattr(self._state_0, attr, None)
                    if arr is not None:
                        state_data[attr] = arr.numpy().copy()
                except Exception:
                    pass

        return {
            "success": True,
            "config": {
                "num_envs": self._config.num_envs,
                "device": self._config.device,
                "solver": self._config.solver,
                "physics_dt": self._config.physics_dt,
                "substeps": self._config.substeps,
                "broad_phase": self._config.broad_phase,
                "enable_cuda_graph": self._config.enable_cuda_graph,
                "enable_differentiable": self._config.enable_differentiable,
            },
            "sim_time": self._sim_time,
            "step_count": self._step_count,
            "world_created": self._world_created,
            "replicated": self._replicated,
            "robots": dict(self._robots),
            "cloths": dict(self._cloths),
            "cables": dict(self._cables),
            "sensors": list(self._sensors.keys()),
            "joints_per_world": self._joints_per_world,
            "bodies_per_world": self._bodies_per_world,
            "state": state_data,
        }

    def destroy(self) -> Dict[str, Any]:
        """Destroy the backend and release resources."""
        logger.info("Destroying NewtonBackend (step_count=%d) …", self._step_count)
        errors = []

        # Close renderer
        if self._renderer is not None:
            try:
                if hasattr(self._renderer, "close"):
                    self._renderer.close()
            except Exception as exc:
                errors.append(str(exc))

        # Clear all state
        for attr in (
            "_solver",
            "_secondary_solver",
            "_state_0",
            "_state_1",
            "_control",
            "_contacts",
            "_collision_pipeline",
            "_model",
            "_builder",
            "_cuda_graph",
            "_diffsim_states",
            "_diffsim_tape",
            "_default_joint_q",
            "_default_joint_qd",
        ):
            try:
                setattr(self, attr, None)
            except Exception:
                pass

        self._robots.clear()
        self._cloths.clear()
        self._cables.clear()
        self._sensors.clear()

        final_step = self._step_count
        final_time = self._sim_time
        self._sim_time = 0.0
        self._world_created = False
        self._replicated = False

        # Synchronize warp device
        if self._wp is not None:
            try:
                self._wp.synchronize()
            except Exception:
                pass

        logger.info("Destroyed. Ran %d steps (%.2fs sim time)", final_step, final_time)
        return {
            "success": True,
            "message": f"Destroyed. Ran {final_step} steps ({final_time:.2f}s sim time)",
            "steps_executed": final_step,
            "sim_time": final_time,
        }

    # ------------------------------------------------------------------
    # Dunder methods
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"NewtonBackend(solver={self._config.solver}, device={self._config.device}, "
            f"num_envs={self._config.num_envs}, robots={list(self._robots.keys())}, "
            f"cloths={list(self._cloths.keys())}, steps={self._step_count})"
        )

    def __del__(self):
        try:
            if self._world_created:
                self.destroy()
        except Exception:
            pass
