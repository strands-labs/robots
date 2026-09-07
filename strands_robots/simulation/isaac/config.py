"""Isaac Sim simulation configuration.

Central configuration dataclass for :class:`IsaacSimulation`. Controls
device selection, physics parameters, rendering, and headless mode.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any

from strands_robots.utils import positive_count_error

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


#: A single USD prim name: an ASCII identifier, first character a letter or
#: underscore. This is the alphabet
#: :func:`strands_robots.simulation.isaac.joint_names._tf_make_valid_identifier`
#: already encodes as USD's own rule -- it replaces every character outside
#: ``[A-Za-z0-9_]``, and a first character outside ``[A-Za-z_]``, with ``_`` --
#: so the two spellings of the rule in this package agree by construction.
_PRIM_NAME_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")


def _stage_path_error(value: Any) -> str | None:
    """Return an error message if ``value`` cannot prefix a USD prim path.

    Every prim :class:`~strands_robots.simulation.isaac.simulation.IsaacSimulation`
    creates is addressed by a path interpolated from this prefix and an entity
    name -- ``f"{stage_path}/Robots/{name}"``, and the same shape for
    ``/Objects/`` and ``/Cameras/``. The name half of that f-string already has
    a domain: ``add_robot`` refuses a name that cannot address the robot it
    creates on the shared :func:`strands_robots.utils.entity_name_error`, which
    rejects a non-``str``, the empty string, and a string containing a NUL.
    This is the same domain for the other half, so a path the API records is
    one the API can address regardless of which component the caller got wrong.

    Args:
        value: The candidate :attr:`IsaacConfig.stage_path`.

    Returns:
        ``None`` when ``value`` is an absolute USD prim path with at least one
        component, every component a prim name; otherwise a message naming the
        field, the value, and the path it would have produced.

    Four spellings are refused, each measured against the unbound
    ``add_robot`` / ``remove_robot`` (no Isaac Sim or GPU needed -- the
    procedural branch touches no stage). Before this domain every one of them
    reported ``status="success"`` and recorded the interpolated string in
    ``_prim_registry``, which is what :meth:`destroy` releases and counts:

    * **Not a ``str``.** The f-string has no type requirement, so the value's
      ``repr``-ish text became part of the path: ``stage_path=None`` recorded
      ``None/Robots/arm`` -- the literal four characters -- ``stage_path=7``
      recorded ``7/Robots/arm``, and ``stage_path=["/World"]`` recorded
      ``['/World']/Robots/arm``. ``entity_name_error`` refuses a non-``str``
      name for the same reason; a non-``str`` prefix is the same value arriving
      through the other half.
    * **Not absolute.** ``stage_path="World"`` recorded ``World/Robots/arm``, a
      relative path. :meth:`IsaacSimulation.get_body_state` routes on exactly
      that distinction -- ``if body_name.startswith("/")`` takes the stage
      lookup, and ``elif "/" in body_name`` reads the value as
      ``robot_name/link_name`` -- so a caller handing back the path the API
      recorded is routed to the wrong branch, where ``World`` is looked up as a
      robot name. ``"/"`` alone is refused with it: the root names no
      component, and prefixing it yields ``//Robots/arm``.
    * **An empty component.** ``"/World/"`` recorded ``/World//Robots/arm`` and
      ``"/World//Sub"`` recorded ``/World//Sub/Robots/arm``. A doubled
      separator is not a path component, and a trailing separator is the
      likeliest way to write one, because the field is documented as a
      *prefix*.
    * **A component that is not a prim name.** ``"/My World"`` and
      ``"/World\x00x"`` were both recorded verbatim. This is the half that is
      not merely cosmetic: USD transcodes a prim name outside its identifier
      alphabet, so the prim does not land at the path that was recorded for it.
      This package already relies on that transcoding being real and
      deterministic -- ``demangle_usd_joint_names`` exists to undo it for
      joint names, where the URDF joint ``1`` imports as ``tn__1_`` -- and a
      recorded path the stage does not carry is a prim :meth:`destroy` counts
      and does not release.

    The three ``bool``/numeric-domain fields of this dataclass are validated in
    :meth:`IsaacConfig.__post_init__` and so is this one: the property is
    lexical, it needs no engine, and ``stage_path`` has exactly one consumer
    package. It is kept in this module rather than
    :mod:`strands_robots.utils` for the reason :data:`ENV_SWITCH_ON` states --
    AGENTS.md Key Convention 11 -- no other backend has a stage.
    """
    if not isinstance(value, str):
        return (
            f"IsaacConfig.stage_path must be a str, got {type(value).__name__} {value!r}. "
            f"Every prim path is interpolated from it, so this one would address "
            f"robots at {f'{value}/Robots/<name>'!r}. Use an absolute USD prim path "
            f"such as '/World'."
        )
    components = value.split("/")
    if not value.startswith("/"):
        return (
            f"IsaacConfig.stage_path must be an absolute USD prim path starting with '/', "
            f"got {value!r}, which would address robots at {f'{value}/Robots/<name>'!r} -- "
            f"a relative path. get_body_state() distinguishes an absolute prim path from a "
            f"'<robot>/<link>' pair by that leading '/', so a relative prefix makes the path "
            f"this backend records unusable as the key it records it for. Use '/World'."
        )
    if bad := [c for c in components[1:] if not _PRIM_NAME_RE.match(c)]:
        return (
            f"IsaacConfig.stage_path={value!r} is not a USD prim path: component "
            f"{bad[0]!r} is not a prim name (an ASCII identifier matching "
            f"[A-Za-z_][A-Za-z0-9_]*). It would address robots at "
            f"{f'{value}/Robots/<name>'!r}; USD transcodes a name outside that "
            f"alphabet, so the prim would not land at the path recorded for it, and "
            f"an empty component (a doubled or trailing '/') is not a component at "
            f"all. Use '/World' or '/World/<Identifier>'."
        )
    return None


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
        Gravity vector. Default (0.0, 0.0, -9.81) (Z-up convention). Read by
        ``create_world()``, which applies the same domain to it as to its own
        ``gravity=`` argument: three finite, non-boolean components, Z-aligned
        (``x`` and ``y`` both zero), or a real scalar taken as the z-component.
        The domain lives with the engine that spends the value - Isaac's
        ``PhysicsContext.set_gravity`` takes a signed scalar - and is shared
        with every other backend's gravity surface, so it is applied there
        rather than restated here.
    ground_plane : bool
        Whether to add a ground plane on ``create_world()``. Default True.
    stage_path : str
        USD stage path prefix, and the root every prim this backend creates is
        addressed under (``{stage_path}/Robots/{name}``, and the same shape for
        ``/Objects/`` and ``/Cameras/``). Default ``"/World"``. Must be an
        absolute USD prim path with at least one component, every component a
        prim name (an ASCII identifier matching ``[A-Za-z_][A-Za-z0-9_]*``) --
        the same requirement the name half of that path already carries via
        :func:`strands_robots.utils.entity_name_error`. A value outside that
        domain is refused on construction rather than interpolated into a path
        the stage cannot carry.
    nucleus_url : str | None
        Override Omniverse Nucleus server URL. Default from env var
        ``STRANDS_ISAAC_NUCLEUS_URL`` or None (use Isaac defaults).
    camera_width : int
        Default camera width in pixels, for every camera and render call that
        does not state one of its own. Default 640. Must be a positive integer
        on :func:`strands_robots.utils.positive_count_error` -- the same shared
        pixel floor ``add_camera`` and the render family already apply to the
        ``width`` they take instead of this default, so one resolution cannot be
        refused at the call site and accepted from the config.
    camera_height : int
        Default camera height in pixels. Default 480. Same domain as
        ``camera_width``.
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

        # Validate num_envs on the shared count domain
        # (:func:`strands_robots.utils.positive_count_error`) -- the same domain
        # ``camera_width`` / ``camera_height`` take a few lines below, and the
        # same one :meth:`~strands_robots.simulation.isaac.IsaacSimulation.replicate`
        # applies to the ``num_envs`` *argument* it takes instead of this
        # default, so the two owners of one environment count reach one verdict.
        # The hand-rolled ``< 1`` test this replaces is the shape that domain's
        # docstring warns about, and the shape ``RLTrainSpec.num_envs`` was
        # moved off for the same reason: it tests only the floor, so it read
        # ``True`` as a count of 1 while refusing ``False``, and let ``4.0``,
        # ``2.7``, ``nan`` and ``inf`` through to be stored and reported as an
        # environment count -- the init log formats this field with ``%d``, so a
        # stored ``2.7`` was announced as ``num_envs=2`` and a stored ``nan``
        # made that logging call raise. A ``str``, ``None`` or a list raised
        # ``TypeError`` from the comparison itself, naming neither the field nor
        # a remedy.
        if (envs_err := positive_count_error(self.num_envs, "num_envs", type(self).__name__)) is not None:
            raise ValueError(envs_err)

        # Validate physics_dt
        if self.physics_dt <= 0:
            raise ValueError(f"physics_dt must be > 0, got {self.physics_dt}")

        # Validate rendering_dt
        if self.rendering_dt <= 0:
            raise ValueError(f"rendering_dt must be > 0, got {self.rendering_dt}")

        # Validate camera dimensions on the shared pixel floor
        # (:func:`strands_robots.utils.positive_count_error`) that ``add_camera``
        # and ``_render_frame`` already apply to the ``width`` / ``height``
        # arguments they take *instead of* these defaults, so the two owners of
        # one resolution reach one verdict. The hand-rolled ``< 1`` pair this
        # replaces is the shape that domain's docstring warns about: it tests
        # only the floor, so it read ``True`` as a width of 1, let ``640.0``,
        # ``640.5``, ``nan`` and ``inf`` through to ``np.zeros((h, w, 3))`` as
        # ``TypeError: 'float' object cannot be interpreted as an integer`` --
        # raised out of ``render``, whose contract is a ``{"status": "error"}``
        # dict -- and raised ``TypeError`` from the comparison itself for a
        # ``str`` or ``None``, rather than naming the field. Each field is
        # graded separately so the message names the one to fix.
        for param, value in (("camera_width", self.camera_width), ("camera_height", self.camera_height)):
            if (dim_err := positive_count_error(value, param, type(self).__name__)) is not None:
                raise ValueError(dim_err)

        # Validate stage_path. It is the other half of every prim path this
        # backend interpolates; the name half is already refused on the shared
        # ``entity_name_error`` domain at each creation site.
        if (stage_err := _stage_path_error(self.stage_path)) is not None:
            raise ValueError(stage_err)

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
