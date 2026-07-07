"""Isaac Sim simulation configuration.

Central configuration dataclass for :class:`IsaacSimulation`. Controls
device selection, physics parameters, rendering, and headless mode.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any

# Supported render modes
RENDER_MODES = ("headless", "rtx_realtime", "rtx_pathtracing")

# Supported physics solvers in Isaac Sim
PHYSICS_SOLVERS = ("physx_gpu", "physx_cpu")


@dataclass
class IsaacConfig:
    """Configuration for :class:`IsaacSimulation`.

    Parameters
    ----------
    num_envs : int
        Number of parallel environments. Default 1. For fleet training,
        set to 1024 (Isaac is heavy per-env).
    device : str
        CUDA device string. ``"cuda:0"`` (default) or ``"cuda:N"``.
    headless : bool
        Run without GUI. Default True (required for cloud/CI runners).
    physics_dt : float
        Physics timestep in seconds. Default 1/120 s.
    rendering_dt : float
        Rendering timestep in seconds. Default 1/30 s.
    render_mode : str
        Rendering pipeline: ``"headless"`` (no rendering),
        ``"rtx_realtime"`` (fast, rasterization-based),
        ``"rtx_pathtracing"`` (slow, photorealistic). Default ``"headless"``.
    gravity : tuple[float, float, float]
        Gravity vector. Default (0.0, 0.0, -9.81) (Z-up convention).
    ground_plane : bool
        Whether to add a ground plane on ``create_world()``. Default True.
    stage_path : str
        USD stage path prefix. Default ``"/World"``.
    nucleus_url : str | None
        Override Omniverse Nucleus server URL. Default from env var
        ``STRANDS_ISAAC_NUCLEUS_URL`` or None (use Isaac defaults).
    camera_width : int
        Default camera width in pixels. Default 640.
    camera_height : int
        Default camera height in pixels. Default 480.
    enable_rtx_sensors : bool
        Enable RTX-accelerated sensors (camera, LiDAR). Default True.
    verbose : bool
        Enable verbose logging from Isaac Sim/Kit. Default False.
    extra : dict
        Escape-hatch for Isaac-specific or experimental options.
    """

    num_envs: int = 1
    device: str = "cuda:0"
    headless: bool = True
    physics_dt: float = 1.0 / 120.0
    rendering_dt: float = 1.0 / 30.0
    render_mode: str = "headless"
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    ground_plane: bool = True
    stage_path: str = "/World"
    nucleus_url: str | None = None
    camera_width: int = 640
    camera_height: int = 480
    enable_rtx_sensors: bool = True
    verbose: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize configuration."""
        # Validate render mode
        if self.render_mode not in RENDER_MODES:
            raise ValueError(f"Unknown render_mode {self.render_mode!r}. Supported: {RENDER_MODES}")

        # Validate device
        if not self.device.startswith("cuda"):
            raise ValueError(f"Isaac Sim requires a CUDA device, got {self.device!r}. Use 'cuda:0', 'cuda:1', etc.")

        # Validate num_envs
        if self.num_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {self.num_envs}")

        # Validate physics_dt
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be > 0, got {self.physics_dt}")

        # Validate rendering_dt
        if self.rendering_dt <= 0:
            raise ValueError(f"rendering_dt must be > 0, got {self.rendering_dt}")

        # Validate camera dimensions
        if self.camera_width < 1 or self.camera_height < 1:
            raise ValueError(f"camera dimensions must be >= 1, got {self.camera_width}x{self.camera_height}")

        # Resolve nucleus_url from environment if not explicitly set
        if self.nucleus_url is None:
            self.nucleus_url = os.environ.get("STRANDS_ISAAC_NUCLEUS_URL")

        # Resolve headless from environment if env says otherwise
        headless_env = os.environ.get("STRANDS_ISAAC_HEADLESS")
        if headless_env is not None:
            self.headless = headless_env.lower() in ("true", "1", "yes")

        # Resolve RTX pathtracing from environment
        rtx_env = os.environ.get("STRANDS_ISAAC_RTX_PATHTRACING")
        if rtx_env is not None and rtx_env.lower() in ("true", "1", "yes"):
            self.render_mode = "rtx_pathtracing"

    @classmethod
    def from_kwargs(cls, **kwargs: Any) -> IsaacConfig:
        """Construct IsaacConfig from kwargs, rejecting unknown keys eagerly.

        Equivalent to ``IsaacConfig(**kwargs)`` for the unknown-key behavior
        (dataclass ``__init__`` already raises ``TypeError`` on unexpected
        kwargs), but exposed as a named entry point so PR-4's
        ``IsaacSimulation.__init__`` can document its kwarg-validation
        contract by name rather than by inline ``dataclasses.fields()``
        reflection.

        Closes the R1 silent-drop bug (commit 32ef307) symmetrically across
        the ``config=None`` and ``config=<existing>`` construction paths in
        ``IsaacSimulation.__init__``.
        """
        return cls(**kwargs)
