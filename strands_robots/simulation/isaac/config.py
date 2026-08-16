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

# Spellings ``STRANDS_ISAAC_HEADLESS`` and ``STRANDS_ISAAC_RTX_PATHTRACING``
# accept. Both are documented as two-sided switches -- "truthy forces headless;
# falsy forces windowed" -- so both sides are enumerated, as four symmetric
# pairs, rather than only the opt-in one. The pairing is the point: with a
# truthy list and an open-ended falsy side, ``on`` and ``off`` both resolved to
# off, so the pair contradicted itself and the truthy side leaked into the falsy
# one. ``y``/``n`` and ``t``/``f`` are deliberately absent -- no measured caller
# sets them, and each addition widens a contract that is now total.
#
# Kept local to this module per AGENTS.md Key Convention 11: the two callers are
# in this file, and the one existing shared reader,
# ``simulation.safe_output.env_flag``, answers a different question (see
# ``_env_switch``).
ENV_SWITCH_ON = ("1", "true", "yes", "on")
ENV_SWITCH_OFF = ("0", "false", "no", "off")


def _env_switch(name: str) -> bool | None:
    """Read environment variable ``name`` as a two-sided on/off switch.

    Args:
        name: Environment variable to read.

    Returns:
        ``True`` for :data:`ENV_SWITCH_ON`, ``False`` for
        :data:`ENV_SWITCH_OFF` (case-insensitive, surrounding whitespace
        ignored), and ``None`` when the variable is unset or set to an empty
        value -- the shell's spelling of absent, which an undefined
        ``${{ vars.* }}`` interpolation in a GitHub Actions ``env:`` block also
        produces. ``None`` means the corresponding :class:`IsaacConfig` field
        is left as the caller set it.

    Raises:
        ValueError: ``name`` is set to any other spelling. Resolving an
            unrecognized spelling to a side would be a silent default on a
            value the caller got wrong (Key Convention 6), and the side it
            silently took was "off": read that way,
            ``STRANDS_ISAAC_HEADLESS=on`` opens a window on a runner that has
            no display, which is the outcome the variable exists to prevent.

            Narrowing the off side is the unavoidable half of this trade. While
            any unlisted spelling resolves to off, every spelling that means on
            resolves to off too, so the truthy side cannot be made whole
            without closing the falsy one. Refusing is what makes that visible
            instead of silent, and it is self-clearing: the message names both
            vocabularies and the variable is re-read on the next construction.

    ``safe_output.env_flag`` is deliberately not reused. It is documented as a
    one-sided opt-in and returns ``bool``, so it collapses "set to off" and
    "unset" into the same answer. That is correct for an ``ALLOW_ABS``-style
    flag, whose only job is to grant a permission, and wrong here: these two
    variables must be able to force the non-default side, so they need the
    third outcome ``None`` that a ``bool`` cannot carry.
    """
    raw = os.environ.get(name)
    if raw is None:
        return None
    value = raw.strip().lower()
    if not value:
        return None
    if value in ENV_SWITCH_ON:
        return True
    if value in ENV_SWITCH_OFF:
        return False
    raise ValueError(
        f"{name}={raw!r} is not a recognized switch value. Use one of {ENV_SWITCH_ON} "
        f"to turn it on or one of {ENV_SWITCH_OFF} to turn it off (case-insensitive); "
        f"leave it unset or empty to keep the value the IsaacConfig field already has. "
        f"It is refused rather than read as off because an unrecognized spelling is as "
        f"likely to mean on -- {name}=on would otherwise resolve to off."
    )


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
        ``STRANDS_ISAAC_HEADLESS`` overrides this field when set: one of
        ``("1", "true", "yes", "on")`` forces headless, one of
        ``("0", "false", "no", "off")`` forces windowed, empty or unset leaves
        the field alone, and any other spelling is refused.
    physics_dt : float
        Physics timestep in seconds. Default 1/120 s.
    rendering_dt : float
        Rendering timestep in seconds. Default 1/30 s.
    render_mode : str
        Rendering pipeline: ``"headless"`` (no rendering),
        ``"rtx_realtime"`` (fast, rasterization-based ``RayTracedLighting``),
        ``"rtx_pathtracing"`` (slow, photorealistic ``PathTracing``).
        Default ``"headless"``. The two RTX modes select the corresponding
        ``renderer`` in the ``SimulationApp`` launch config; because
        SimulationApp is a create-once process-wide singleton, the renderer
        is fixed by whichever world is created first in the process, and a
        later differing request is reported (warning) rather than applied.
        ``STRANDS_ISAAC_RTX_PATHTRACING`` set to one of
        ``("1", "true", "yes", "on")`` overrides this field with
        ``"rtx_pathtracing"``; one of ``("0", "false", "no", "off")``, empty or
        unset leaves the field alone, and any other spelling is refused.
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

        # Resolve headless from environment, which outranks the field when set
        headless_env = _env_switch("STRANDS_ISAAC_HEADLESS")
        if headless_env is not None:
            self.headless = headless_env

        # Resolve RTX pathtracing from environment. Only the on side acts: this
        # switch names one mode, so its off side is "do not force it" rather
        # than a second mode to select, and it leaves render_mode alone.
        if _env_switch("STRANDS_ISAAC_RTX_PATHTRACING"):
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
