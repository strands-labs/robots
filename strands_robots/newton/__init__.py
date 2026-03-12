"""Newton Physics Engine Integration for strands-robots.

Newton is a GPU-accelerated physics engine built on NVIDIA Warp that provides
high-performance parallel simulation for robotics applications. It wraps
MuJoCo-Warp as its primary backend and additionally supports six other solvers:
Featherstone, SemiImplicit, XPBD, VBD, Style3D, and ImplicitMPM.

Key capabilities:

- **Massive parallelism**: ``replicate()`` for 4096+ parallel environments on GPU.
- **Differentiable simulation**: Full gradient support via ``wp.Tape`` for
  sim-to-real transfer, trajectory optimization, and learned control.
- **CollisionPipeline**: Proper broad-phase collision detection with configurable
  margin and gradient support for differentiable contacts.
- **Same high-level API as simulation.py**: ``create_world``, ``add_robot``,
  ``step``, ``render``, and the familiar strands-robots interface.
- **Multiple solver backends**: MuJoCo, Featherstone, SemiImplicit, XPBD, VBD,
  Style3D, ImplicitMPM — choose the best fit for your task.
- **Cloth, cable, soft-body, MPM**: Full deformable physics support.
- **Sensors**: Contact, IMU, and tiled camera sensors.
- **Inverse kinematics**: Jacobian-based IK via autodiff.
- **CUDA graph capture**: ``wp.ScopedCapture`` for minimal Python overhead.
- **Gymnasium wrapper**: ``NewtonGymEnv`` for SB3/RL integration.

Requirements:
    - newton-sim (or newton from source)
    - warp-lang >= 1.11.0
"""

__all__ = [
    "NewtonBackend",
    "NewtonConfig",
    "NewtonGymEnv",
    "SOLVER_MAP",
    "RENDER_BACKENDS",
    "BROAD_PHASE_OPTIONS",
]


def __getattr__(name):
    """Lazy imports to avoid hard dependency on Newton / Warp."""
    if name == "NewtonBackend":
        from .newton_backend import NewtonBackend

        return NewtonBackend
    if name == "NewtonConfig":
        from .newton_backend import NewtonConfig

        return NewtonConfig
    if name == "NewtonGymEnv":
        from .newton_gym_env import NewtonGymEnv

        return NewtonGymEnv
    if name == "SOLVER_MAP":
        from .newton_backend import SOLVER_MAP

        return SOLVER_MAP
    if name == "RENDER_BACKENDS":
        from .newton_backend import RENDER_BACKENDS

        return RENDER_BACKENDS
    if name == "BROAD_PHASE_OPTIONS":
        from .newton_backend import BROAD_PHASE_OPTIONS

        return BROAD_PHASE_OPTIONS
    raise AttributeError(f"module 'strands_robots.newton' has no attribute {name!r}")
