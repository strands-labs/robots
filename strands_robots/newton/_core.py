"""NewtonBackend — thin orchestrator that delegates to submodules.

GPU-accelerated physics simulation backend powered by Newton / Warp.
Implements the same interface as IsaacSimBackend and MujocoBackend
for transparent backend swapping.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

from ._registry import _ensure_newton, get_newton, get_warp
from ._types import SOLVER_MAP, NewtonConfig

logger = logging.getLogger(__name__)


class NewtonBackend:
    """GPU-accelerated physics simulation backend powered by Newton / Warp.

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
        self._newton = get_newton()
        self._wp = get_warp()
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

    # ------------------------------------------------------------------
    # Scene lifecycle (delegated to _scene)
    # ------------------------------------------------------------------

    def create_world(
        self,
        gravity: Optional[Tuple[float, float, float]] = None,
        ground_plane: bool = True,
        up_axis: str = "y",
    ) -> Dict[str, Any]:
        from ._scene import create_world
        return create_world(self, gravity=gravity, ground_plane=ground_plane, up_axis=up_axis)

    def replicate(self, num_envs: Optional[int] = None) -> Dict[str, Any]:
        from ._scene import replicate
        return replicate(self, num_envs=num_envs)

    def reset(self, env_ids: Optional[List[int]] = None) -> Dict[str, Any]:
        from ._scene import reset
        return reset(self, env_ids=env_ids)

    def step(self, actions: Optional[Any] = None) -> Dict[str, Any]:
        from ._scene import step
        return step(self, actions=actions)

    def _finalize_model(self) -> None:
        from ._scene import finalize_model
        finalize_model(self)

    def _recreate_builder(self) -> None:
        from ._scene import recreate_builder
        recreate_builder(self)

    # ------------------------------------------------------------------
    # Robot management (delegated to _robots)
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
        from ._robots import add_robot
        return add_robot(
            self, name, urdf_path=urdf_path, usd_path=usd_path,
            data_config=data_config, position=position, scale=scale,
        )

    # ------------------------------------------------------------------
    # Deformable objects (delegated to _objects)
    # ------------------------------------------------------------------

    def add_cloth(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        from ._objects import add_cloth
        return add_cloth(self, name, **kwargs)

    def add_cable(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        from ._objects import add_cable
        return add_cable(self, name, **kwargs)

    def add_particles(self, name: str, positions, **kwargs: Any) -> Dict[str, Any]:
        from ._objects import add_particles
        return add_particles(self, name, positions, **kwargs)

    # ------------------------------------------------------------------
    # Dual solver
    # ------------------------------------------------------------------

    def enable_dual_solver(
        self,
        rigid_solver: str = "mujoco",
        cloth_solver: str = "vbd",
        cloth_iterations: int = 10,
    ) -> Dict[str, Any]:
        """Enable dual-solver mode: one solver for rigid bodies, another for cloth."""
        if self._model is None:
            self._finalize_model()
        try:
            SecondaryCls = self._get_solver_class(cloth_solver)
            kwargs = {}
            if cloth_solver == "vbd":
                kwargs["iterations"] = cloth_iterations
            self._secondary_solver = (
                SecondaryCls(**kwargs) if kwargs else SecondaryCls()
            )
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
    # Rendering & observation (delegated to _rendering)
    # ------------------------------------------------------------------

    def render(self, camera_name=None, width=1024, height=768) -> Dict[str, Any]:
        from ._rendering import render
        return render(self, camera_name=camera_name, width=width, height=height)

    def get_observation(self, robot_name=None) -> Dict[str, Any]:
        from ._rendering import get_observation
        return get_observation(self, robot_name=robot_name)

    # ------------------------------------------------------------------
    # Recording (delegated to _recording)
    # ------------------------------------------------------------------

    def record_video(self, **kwargs: Any) -> Dict[str, Any]:
        from ._recording import record_video
        return record_video(self, **kwargs)

    # ------------------------------------------------------------------
    # Policy (delegated to _policy)
    # ------------------------------------------------------------------

    def run_policy(self, robot_name: str, **kwargs: Any) -> Dict[str, Any]:
        from ._policy import run_policy
        return run_policy(self, robot_name, **kwargs)

    # ------------------------------------------------------------------
    # Differentiable simulation (delegated to _diffsim)
    # ------------------------------------------------------------------

    def run_diffsim(self, **kwargs: Any) -> Dict[str, Any]:
        from ._diffsim import run_diffsim
        return run_diffsim(self, **kwargs)

    # ------------------------------------------------------------------
    # Sensors (delegated to _sensors)
    # ------------------------------------------------------------------

    def add_sensor(self, name: str, **kwargs: Any) -> Dict[str, Any]:
        from ._sensors import add_sensor
        return add_sensor(self, name, **kwargs)

    def read_sensor(self, name: str) -> Dict[str, Any]:
        from ._sensors import read_sensor
        return read_sensor(self, name)

    # ------------------------------------------------------------------
    # Inverse kinematics (delegated to _ik)
    # ------------------------------------------------------------------

    def solve_ik(self, robot_name: str, target_position, **kwargs) -> Dict[str, Any]:
        from ._ik import solve_ik
        return solve_ik(self, robot_name, target_position, **kwargs)

    # ------------------------------------------------------------------
    # State / Destroy
    # ------------------------------------------------------------------

    def get_state(self) -> Dict[str, Any]:
        """Return full simulation state."""
        state_data = {}
        for attr in (
            "joint_q", "joint_qd", "body_q", "body_qd",
            "particle_q", "particle_qd",
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

        if self._renderer is not None:
            try:
                if hasattr(self._renderer, "close"):
                    self._renderer.close()
            except Exception as exc:
                errors.append(str(exc))

        for attr in (
            "_solver", "_secondary_solver", "_state_0", "_state_1",
            "_control", "_contacts", "_collision_pipeline", "_model",
            "_builder", "_cuda_graph", "_diffsim_states", "_diffsim_tape",
            "_default_joint_q", "_default_joint_qd",
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
