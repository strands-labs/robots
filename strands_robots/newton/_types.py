"""Newton types, constants, and procedural robot definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

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
            raise ValueError(
                f"Unknown solver '{self.solver}'. Options: {list(SOLVER_MAP.keys())}"
            )
        if self.render_backend not in RENDER_BACKENDS:
            raise ValueError(
                f"Unknown render_backend '{self.render_backend}'. Options: {RENDER_BACKENDS}"
            )
        if self.broad_phase not in BROAD_PHASE_OPTIONS:
            raise ValueError(
                f"Unknown broad_phase '{self.broad_phase}'. Options: {BROAD_PHASE_OPTIONS}"
            )
        if self.physics_dt <= 0:
            raise ValueError("physics_dt must be positive")
        if self.num_envs < 1:
            raise ValueError("num_envs must be >= 1")
