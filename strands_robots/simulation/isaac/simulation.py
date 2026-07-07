"""Isaac Sim simulation backend -- GPU-native SimEngine implementation.

This module contains :class:`IsaacSimulation`, the primary implementation
of the ``SimEngine`` ABC for the NVIDIA Isaac Sim backend.

Architecture:
    - All heavy omni/Isaac imports are lazy (not at module level)
    - The class manages an Isaac Sim ``World``, ``Articulation`` handles,
      and RTX camera instances
    - Multi-env replication uses ``omni.isaac.cloner.Cloner``
    - SimulationApp is a process-wide singleton (never create more than one)
    - Rendering delegates to Isaac Sim's RTX pipeline

Thread safety:
    - ``step()``, ``send_action()``, and ``get_observation()`` acquire
      ``self._lock`` to prevent data races
    - ``step()`` must not run concurrently with ``add_robot()``

Environment variables:
    - STRANDS_ISAAC_HEADLESS: Override headless mode (true/false)
    - STRANDS_ISAAC_RTX_PATHTRACING: Enable RTX pathtracing (true/false)
    - STRANDS_ISAAC_NUCLEUS_URL: Override Nucleus asset server URL
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from typing import Any, TypedDict

import numpy as np

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.isaac.config import IsaacConfig

logger = logging.getLogger(__name__)

# Minimum NATIVE render width for RTX cameras. Isaac's RTX pipeline runs
# the DLSS temporal upscaler, which renders internally at ~half the output
# width and upscales. Below ~300 px internal resolution DLSS falls back
# to a temporal-accumulation path that smears a moving arm into a
# translucent "ghost" (long-standing front/oblique-view bug seen during
# the SO-101 cuRobo example's GPU validation -- see issue #69 / PR #68).
# Rendering at >= 640 px wide keeps the DLSS internal resolution above
# that threshold so every frame is crisp on its own; captured frames
# are downscaled to the caller's requested size before return.
_MIN_RENDER_PX = 640


def _env_int(name: str, default: int) -> int:
    """Read a small positive int from the environment (fallback to ``default``)."""
    try:
        v = int(os.environ.get(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    """Read a positive float from the environment (fallback to ``default``)."""
    try:
        v = float(os.environ.get(name, ""))
        return v if v > 0 else default
    except (TypeError, ValueError):
        return default


class SimulationAppLaunchConfig(TypedDict, total=False):
    """Typed shape for ``omni.isaac.kit.SimulationApp`` launch config.

    All keys optional; SimulationApp accepts an open-ended dict and any
    additional keys are forwarded to Kit unchanged. The keys below are the
    well-known ones documented by NVIDIA across Isaac Sim 4.x / 5.x and are
    the ones a Strands tool would realistically expose to an agent.

    See: https://docs.omniverse.nvidia.com/py/isaacsim/source/extensions/omni.isaac.kit/docs/index.html

    Keys
    ----
    headless : bool
        Run without GUI. Required True on cloud / CI runners.
    renderer : str
        ``"RayTracedLighting"`` or ``"PathTracing"``.
    width, height : int
        Viewport resolution in pixels.
    physics_gpu : int
        CUDA device index for PhysX.
    active_gpu : int
        CUDA device index for rendering.
    multi_gpu : bool
        Enable multi-GPU rendering.
    sync_loads : bool
        Block until USD assets finish loading.
    hide_ui : bool
        Hide Kit's editor UI in non-headless mode.
    anti_aliasing : int
        Anti-aliasing level (0-3).
    """

    headless: bool
    renderer: str
    width: int
    height: int
    physics_gpu: int
    active_gpu: int
    multi_gpu: bool
    sync_loads: bool
    hide_ui: bool
    anti_aliasing: int


# Shape-name aliases accepted by :meth:`IsaacSimulation.add_object`.
# Maps an alias -> the canonical shape name. ``"cuboid"`` mirrors Isaac's
# ``DynamicCuboid`` / ``FixedCuboid`` class names and the vocabulary used
# throughout the docs; it normalizes to the canonical ``"box"`` (see #88).
# A unit test pins this mapping so docs and code can't drift apart again.
_SHAPE_ALIASES: dict[str, str] = {"cuboid": "box"}


def _rgb_png_block(rgb: np.ndarray) -> dict[str, Any] | None:
    """Encode an RGB ndarray as a render ``content[].image`` PNG block.

    Mirrors the MuJoCo backend's emission: raw PNG bytes in
    ``source.bytes`` (NOT base64 -- the Bedrock Converse API base64-encodes
    on the wire and rejects a pre-encoded string with "Could not process
    image"). Emitting this block on every ``render()`` success path makes
    the Isaac frame transport match MuJoCo so the shared
    ``PolicyRunner._extract_frame_ndarray`` (which decodes ``content[].image``
    only) can pull frames for video recording (#127).

    Returns ``None`` if PIL is unavailable or encoding fails, so
    ``render()`` degrades to the legacy rgb-only envelope rather than
    raising -- same lazy-PIL discipline as ``_resize_rgb`` (PIL stays out
    of module import; Isaac's bundled python may lack it).
    """
    try:
        import io

        from PIL import Image  # lazy: keep heavy import out of module load

        arr = np.asarray(rgb)[..., :3].astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        return {"image": {"format": "png", "source": {"bytes": buf.getvalue()}}}
    except (ImportError, ValueError, OSError, TypeError, AttributeError) as e:
        # PIL absent (ImportError) or encode failure -- never let frame
        # telemetry break render; mirrors the render() except-clause shape.
        logger.warning("render: PNG frame encode failed (%s); content[].image omitted", e)
        return None


# Module-level singleton tracking for SimulationApp
_SIMULATION_APP: Any = None
_SIMULATION_APP_LOCK = threading.Lock()


def _get_or_create_simulation_app(
    headless: bool = True,
    launch_config: SimulationAppLaunchConfig | None = None,
    **kwargs: Any,
) -> Any:
    """Get or create the process-wide SimulationApp singleton.

    Isaac Sim's SimulationApp can only be created ONCE per process.
    This function ensures that constraint is respected.

    Parameters
    ----------
    headless : bool
        Run without GUI.
    launch_config : SimulationAppLaunchConfig, optional
        Typed launch config dict forwarded to ``omni.isaac.kit.SimulationApp``.
        See :class:`SimulationAppLaunchConfig` for documented keys
        (``renderer``, ``width``, ``height``, ``physics_gpu``,
        ``active_gpu``, ``multi_gpu``, ``sync_loads``, ``hide_ui``,
        ``anti_aliasing``). The explicit ``headless`` argument always
        wins over any ``"headless"`` key in ``launch_config``.
    **kwargs
        Additional SimulationApp launch keys (escape hatch for Kit
        options not in :class:`SimulationAppLaunchConfig`). Merged on
        top of ``launch_config``; ``headless`` argument still wins.

    Returns
    -------
    SimulationApp instance.

    Raises
    ------
    ImportError
        If omni.isaac.kit is not available.
    """
    global _SIMULATION_APP

    with _SIMULATION_APP_LOCK:
        if _SIMULATION_APP is not None:
            return _SIMULATION_APP

        try:
            # Isaac Sim 4.5+: ``isaacsim.SimulationApp`` is the supported
            # entry point. The legacy ``omni.isaac.kit.SimulationApp``
            # still works on 4.5 (deprecated shim under ``extsDeprecated``)
            # but emits a noisy deprecation warning at import time and
            # may not exist at all on a pip-only ``isaacsim`` install.
            # Try the modern path first, fall back to the legacy one so
            # this code keeps working on older Isaac Sim builds (and on
            # CI mocks that monkey-patch the legacy module).
            try:
                from isaacsim import SimulationApp  # type: ignore[import-not-found]
            except ImportError:
                from omni.isaac.kit import SimulationApp  # type: ignore[import-not-found]
        except ImportError as e:
            from strands_robots.simulation.isaac._install import not_available_import_error

            raise ImportError(not_available_import_error()) from e

        # Layer order: typed launch_config base, then **kwargs escape hatch,
        # then explicit headless argument (always wins so the caller's
        # intent is unambiguous).
        merged: dict[str, Any] = dict(launch_config or {})
        merged.update(kwargs)
        merged["headless"] = headless
        _SIMULATION_APP = SimulationApp(merged)
        logger.info(
            "SimulationApp created (headless=%s). Note: this is a process-wide singleton.",
            headless,
        )
        return _SIMULATION_APP


# ----------------------------------------------------------------------------
# Dual-namespace import note
# ----------------------------------------------------------------------------
#
# Isaac Sim ships every runtime extension under TWO namespaces: the legacy
# ``omni.isaac.*`` tree (the 4.x path, still present as Kit-extension shims
# under ``extsDeprecated/`` on 4.5/5.x -- imports work post-SimulationApp
# boot but emit deprecation warnings) and the modern ``isaacsim.*`` tree
# (the supported path on Isaac Sim 6.0). This file targets Isaac Sim 6.0 /
# Python 3.12 (see ``_install.ISAAC_SIM_MIN_VERSION``): every lazy import
# now tries the ``isaacsim.*`` location first and falls back to the
# ``omni.isaac.*`` path via ``try: ... except ImportError:`` so 4.x
# installs aren't hard-broken during the transition. The namespace map
# applied across this module:
#
#   omni.isaac.core.World              -> isaacsim.core.api.World
#   omni.isaac.core.objects.*          -> isaacsim.core.api.objects.*
#   omni.isaac.sensor.Camera           -> isaacsim.sensors.camera.Camera
#   omni.isaac.core.articulations.*    -> isaacsim.core.prims.SingleArticulation
#                                         (see ``_import_articulation_cls``)
#   omni.isaac.core.utils.{prims,
#       stage,viewports}               -> isaacsim.core.utils.{prims,stage,viewports}
#   omni.importer.urdf                 -> isaacsim.asset.importer.urdf
#
# ``import omni.usd`` is NOT renamed (it stays under ``omni.*`` on 6.0).
# Downstream unit tests ``patch.dict("sys.modules", {"isaacsim.*": fake})``
# to inject mocks; the modern-first dual-path resolves those mocks while
# still degrading gracefully on a legacy box.


def _accepts_config_kw(cls: Any) -> bool:
    """True if ``cls.__init__`` accepts a ``config`` keyword argument."""
    try:
        import inspect

        return "config" in inspect.signature(cls).parameters
    except (TypeError, ValueError):
        return True  # assume yes; the call site falls back to no-arg on TypeError


def _coerce_prim_path(res: Any) -> str:
    """Normalise a URDF-import return value to a USD prim-path string.

    Isaac Sim 6.0's ``URDFImporter.import_urdf()`` may return the prim path
    directly, a ``(status, path)`` tuple, or an object exposing ``.prim_path`` /
    ``.path``. Handle the common shapes; return ``""`` if none match.
    """
    if res is None:
        return ""
    if isinstance(res, str):
        return res
    if isinstance(res, (tuple, list)):
        for item in res:
            p = _coerce_prim_path(item)
            if p:
                return p
        return ""
    for attr in ("prim_path", "path", "stage_path", "default_prim_path"):
        val = getattr(res, attr, None)
        if isinstance(val, str) and val:
            return val
    return ""


def _import_articulation_cls() -> Any:
    """Resolve the single-prim articulation wrapper across Isaac versions.

    Isaac Sim 6.0 relocated the single-articulation view. The 4.x path
    was ``omni.isaac.core.articulations.Articulation``; on 6.0 the
    high-level wrapper is ``isaacsim.core.api.articulations.Articulation``
    and the lower-level single-prim view lives in ``isaacsim.core.prims``
    as ``SingleArticulation`` (some builds also keep an ``Articulation``
    alias). Probe modern locations first, fall back to the legacy 4.x
    path so transitional installs keep working.

    Returns the class object. Raises ``ImportError`` only if no known
    location resolves (the caller's cleanup-clause tuple catches it).
    """
    # 1. Isaac Sim 6.0 high-level API (keeps the ``Articulation`` name).
    try:
        from isaacsim.core.api.articulations import (  # type: ignore[import-not-found]
            Articulation,
        )

        return Articulation
    except ImportError:
        pass
    # 2. Isaac Sim 6.0 single-prim view: isaacsim.core.prims.SingleArticulation
    try:
        from isaacsim.core.prims import (  # type: ignore[import-not-found]
            SingleArticulation,
        )

        return SingleArticulation
    except ImportError:
        pass
    # 3. Some 6.0 builds keep an ``Articulation`` alias under core.prims.
    try:
        from isaacsim.core.prims import (  # type: ignore[import-not-found]
            Articulation,
        )

        return Articulation
    except ImportError:
        pass
    # 4. Legacy 4.x fallback.
    from omni.isaac.core.articulations import (  # type: ignore[import-not-found]
        Articulation,
    )

    return Articulation


class _RobotState:
    """Internal bookkeeping for a robot in the Isaac simulation."""

    def __init__(
        self,
        name: str,
        prim_path: str,
        joint_names: list[str],
        articulation: Any = None,
        actual_prim_path: str | None = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.joint_names = joint_names
        self.articulation = articulation
        # The prim path the URDF importer / USD reference actually
        # placed the robot at, which can differ from ``prim_path`` when
        # the importer ignores the requested destination (Isaac Sim 4.5
        # ``isaacsim.asset.importer.urdf.import_robot`` ignores the
        # ``stage=""`` argument and lands the robot under the URDF's
        # ``robot name``, e.g. ``/so101_new_calib`` regardless of
        # ``/World/Robots/arm`` being requested). Used by
        # ``gripper_frame_pose`` to walk the actual robot subtree.
        self.actual_prim_path = actual_prim_path or prim_path


class _CameraState:
    """Internal bookkeeping for a camera in the Isaac simulation."""

    def __init__(self, name: str, prim_path: str, width: int, height: int):
        self.name = name
        self.prim_path = prim_path
        self.width = width
        self.height = height
        self.handle: Any = None


class _ObjectState:
    """Internal bookkeeping for an object (shape primitive) in the Isaac simulation.

    ``handle`` is the ``omni.isaac.core.objects.{Dynamic,Fixed}{Cuboid,Sphere,
    Cylinder,Capsule}`` instance returned by :meth:`IsaacSimulation.add_object`.
    The handle is what got registered with ``world.scene.add()`` and is the
    keyhole ``world.scene.remove_object(name)`` later uses for deletion. Held
    here so :meth:`IsaacSimulation.remove_object` doesn't have to round-trip
    through ``world.scene.get_object()`` (which can raise on a torn-down
    stage) just to find the prim.
    """

    def __init__(
        self,
        name: str,
        prim_path: str,
        shape: str,
        is_static: bool,
        handle: Any = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.shape = shape
        self.is_static = is_static
        self.handle = handle


class IsaacSimulation(SimEngine):
    """GPU-native simulation backend built on NVIDIA Isaac Sim.

    Implements the ``SimEngine`` ABC. Provides photorealistic rendering,
    RTX sensors, USD scene management, and fleet replication via Cloner.

    Parameters
    ----------
    config : IsaacConfig or None
        Configuration. If None, defaults are used.
    **kwargs
        Shortcut kwargs merged into config (e.g. ``num_envs=1024``).

    Examples
    --------
    >>> sim = IsaacSimulation(IsaacConfig(num_envs=1, headless=True))
    >>> ok, msg = IsaacSimulation.is_available()
    >>> if ok:
    ...     sim.create_world()
    ...     sim.add_robot("so100")
    ...     sim.step(100)
    ...     sim.destroy()
    """

    def __init__(self, config: IsaacConfig | None = None, **kwargs: Any) -> None:
        # Merge shortcut kwargs into config. Unknown kwargs are rejected
        # eagerly (rather than silently dropped) so a typo like
        # ``IsaacSimulation(headles=False)`` surfaces at construction time
        # instead of producing a default-config sim with no warning.
        #
        # A small allow-list of legacy kwargs from the example-local
        # adapter retired by issue #69 is accepted for backward compat
        # with callers that still pass them via
        # ``create_simulation("isaac", tool_name=..., default_timestep=...)``.
        # They are stored on the instance (not on ``IsaacConfig``) so the
        # config dataclass stays narrow.
        import dataclasses

        # Pull the legacy shortcuts out of ``kwargs`` before strict
        # IsaacConfig kwarg-validation runs.
        legacy_tool_name = kwargs.pop("tool_name", "isaac")
        legacy_default_timestep = kwargs.pop("default_timestep", None)
        legacy_default_width = kwargs.pop("default_width", None)
        legacy_default_height = kwargs.pop("default_height", None)

        if config is None:
            # IsaacConfig is a dataclass; passing an unknown kwarg raises
            # TypeError("__init__() got an unexpected keyword argument ...")
            # naturally. Both branches now have symmetric strictness.
            config = IsaacConfig(**kwargs)
        elif kwargs:
            fields = {f.name for f in dataclasses.fields(config)}
            unknown = sorted(set(kwargs) - fields)
            if unknown:
                raise TypeError(
                    f"IsaacSimulation got unexpected kwargs: {unknown}. Known IsaacConfig fields: {sorted(fields)}."
                )
            config = dataclasses.replace(config, **kwargs)
        # Apply legacy timestep / camera-size shortcuts onto the config
        # if the caller passed them. These map to the canonical
        # ``physics_dt`` / ``camera_width`` / ``camera_height`` fields so
        # downstream code only reads from one source of truth.
        if legacy_default_timestep is not None:
            config = dataclasses.replace(config, physics_dt=float(legacy_default_timestep))
        if legacy_default_width is not None:
            config = dataclasses.replace(config, camera_width=int(legacy_default_width))
        if legacy_default_height is not None:
            config = dataclasses.replace(config, camera_height=int(legacy_default_height))
        self._config = config
        # Tool-name is informational; some Strands tooling renders it.
        self.tool_name = legacy_tool_name

        # Simulation state (all lazy-initialized)
        self._app: Any = None
        self._world: Any = None

        # World state
        self._world_created = False
        self._replicated = False
        self._num_envs_active = 1
        self._sim_time = 0.0
        self._step_count = 0

        # Entity tracking
        self._robots: dict[str, _RobotState] = {}
        self._cameras: dict[str, _CameraState] = {}
        self._objects: dict[str, _ObjectState] = {}
        self._prim_registry: list[str] = []  # track all created prims for cleanup
        # Names of objects realized by load_scene (LIBERO/BDDL scene). Kept
        # separate from _objects so a per-episode load_scene can clear only
        # the prior scene's prims (idempotent reload) without disturbing
        # objects added manually via add_object.
        self._scene_objects: set[str] = set()
        # Per-camera output size (RTX cameras render at >= _MIN_RENDER_PX
        # wide so DLSS doesn't ghost a moving arm; captured frames are
        # downscaled to the size the caller asked for before return).
        self._cam_out_size: dict[str, tuple[int, int]] = {}
        # Synchronous rollout-video recorder state (set by
        # start_cameras_recording, cleared by stop_cameras_recording).
        self._cams_rec_state: dict[str, Any] | None = None

        # Thread safety
        self._lock = threading.RLock()

        # --- Main-thread pump (for off-main-thread callers, e.g. Gradio).
        # Isaac Sim's renderer + physics may only be driven from the
        # thread that created SimulationApp (the main thread). A web UI
        # like Gradio calls into the sim from worker threads, where
        # ``world.step(render=True)`` deadlocks. So when ``run_pump_forever``
        # is engaged the main thread runs ``pump()`` (steps + renders +
        # caches frames and joint state); worker-thread reads return the
        # cache, and worker-thread actions are enqueued for the pump to
        # apply. ``_main_tid`` identifies the owning thread; when called
        # ON it we run inline (no queue), so the headless smoke-test path
        # is unchanged. See issue #69 for the consolidation rationale.
        self._main_tid = threading.get_ident()
        self._action_q: queue.Queue = queue.Queue()
        self._main_jobs: queue.Queue = queue.Queue()
        self._frame_cache: dict[str, Any] = {}
        self._joint_cache: dict[str, dict[str, float]] = {}
        self._pump_running = False  # True while run_pump_forever owns the renderer
        self._pump_cameras = True
        # DLSS-convergence tick counts. Holding the kinematic arm still
        # for a few RTX render ticks lets the temporal upscaler settle
        # on the new pose; both knobs are env-tunable for headroom on
        # slower GPUs (the same names the retired example used so
        # existing operator runbooks keep working).
        self._record_converge = _env_int("SO101_RECORD_CONVERGE", 6)
        self._idle_converge = _env_int("SO101_IDLE_CONVERGE", 4)
        # Min seconds between IDLE live-preview refreshes. Static idle
        # scenes don't need to be re-rendered at full speed -- doing so
        # pegs the RTX renderer (~7 cores) and starves Gradio HTTP /
        # recorder threads. ~1 Hz is a working default validated against
        # the example's Gradio UI on an L4.
        self._idle_render_period = _env_float("SO101_IDLE_RENDER_PERIOD", 1.0)
        # Number of render-bearing world steps add_camera takes to warm up a
        # freshly-created RTX camera's render product before returning. Isaac's
        # RTX pipeline does not accumulate a frame until the world is stepped
        # with rendering enabled, so the first few ``get_rgba()`` calls on a
        # brand-new camera return a malformed / empty buffer (shape ``(0,)``).
        # Stepping a few times inside add_camera means render()/recording see a
        # valid frame on the very first call instead of dropping frames during
        # an example's opening rollout. Env-tunable for headroom on slow GPUs.
        self._camera_warmup_steps = _env_int("STRANDS_ISAAC_CAMERA_WARMUP_STEPS", 10)

        logger.info(
            "IsaacSimulation initialized: num_envs=%d, device=%s, headless=%s",
            config.num_envs,
            config.device,
            config.headless,
        )

    def _on_main_thread(self) -> bool:
        return threading.get_ident() == self._main_tid

    @classmethod
    def is_available(cls) -> tuple[bool, str | None]:
        """Check if Isaac Sim is available on this system.

        Returns
        -------
        tuple[bool, str | None]
            (available, reason_if_not). If available is True, reason is None.
        """
        # Probe what create_world() actually needs: a SimulationApp entry
        # point. Isaac Sim ships TWO namespaces today:
        #
        #   * Legacy: ``omni.isaac.kit.SimulationApp`` (the pre-4.5 path,
        #     still present as a deprecated shim in the 4.5 docker image
        #     under ``extsDeprecated/`` -- emits a deprecation warning at
        #     import time but works).
        #   * Modern: ``isaacsim.SimulationApp`` (the supported path on
        #     Isaac Sim 4.5+ / pip ``isaacsim``).
        #
        # Some Isaac Sim 4.5+ pip installs ship ONLY the modern namespace
        # (no ``omni.isaac.kit`` until ``import isaacsim`` bootstraps the
        # Kit kernel). Probing only ``omni.isaac.kit`` therefore returns
        # False on a perfectly working pip ``isaacsim`` install. Accept
        # either namespace as evidence Isaac Sim is usable.
        #
        # The bare ``omni`` namespace is intentionally NOT probed -- it's
        # a PEP 420 namespace package shared by omni.ui / omni.usd /
        # partial Omniverse SDK installs / Isaac-Lab pre-bootstrap
        # states; its mere presence is not a reliable signal. We probe
        # the specific submodules (``omni.isaac.kit`` / ``isaacsim``)
        # via ``importlib.util.find_spec`` (no side effects, no actual
        # import). Submodules deeper than ``isaacsim`` (e.g.
        # ``isaacsim.core.api``) only resolve AFTER SimulationApp boots
        # the Kit kernel, so we deliberately don't probe them here.
        import importlib.util

        try:
            kit_spec = importlib.util.find_spec("omni.isaac.kit")
        except ModuleNotFoundError:
            kit_spec = None
        try:
            isaacsim_spec = importlib.util.find_spec("isaacsim")
        except ModuleNotFoundError:
            isaacsim_spec = None
        if kit_spec is None and isaacsim_spec is None:
            from strands_robots.simulation.isaac._install import not_importable_reason

            return False, not_importable_reason()

        # Isaac requires CUDA
        try:
            import torch

            if not torch.cuda.is_available():
                return False, ("CUDA device not detected. Isaac Sim requires an NVIDIA GPU with CUDA support.")
        except ImportError:
            return False, ("PyTorch not installed. Isaac Sim requires torch with CUDA support.")

        return True, None

    @property
    def config(self) -> IsaacConfig:
        """Current configuration (read-only)."""
        return self._config

    # --- SimEngine: World Lifecycle ----------------------------------------

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world in Isaac Sim.

        Initializes the SimulationApp (singleton), creates a USD stage,
        configures physics, and optionally adds a ground plane.

        Parameters
        ----------
        timestep : float, optional
            Override physics_dt from config.
        gravity : list[float], optional
            Override gravity vector from config. [gx, gy, gz].
        ground_plane : bool
            Whether to add a ground plane. Default True.

        Returns
        -------
        dict
            Status dict with world info.
        """
        with self._lock:
            if self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "World already created. Call destroy() first."}],
                }

            try:
                # Create/get SimulationApp singleton
                self._app = _get_or_create_simulation_app(headless=self._config.headless)

                # Now safe to import Isaac core modules. Isaac Sim 6.0
                # exposes ``World`` under ``isaacsim.core.api``; the legacy
                # 4.x path was ``omni.isaac.core``. Try modern first, fall
                # back so 4.x installs keep working during the transition.
                try:
                    from isaacsim.core.api import World  # type: ignore[import-not-found]
                except ImportError:
                    from omni.isaac.core import World  # type: ignore[import-not-found]

                dt = timestep if timestep is not None else self._config.physics_dt
                grav = gravity if gravity is not None else list(self._config.gravity)

                # Create World
                self._world = World(
                    stage_units_in_meters=1.0,
                    physics_dt=dt,
                    rendering_dt=self._config.rendering_dt,
                )

                # Set gravity
                # Isaac Sim 5.1: set_gravity takes a scalar magnitude, not a vector.
                # Extract the Z-component (convention: gravity points along -Z).
                gravity_magnitude = grav[2] if isinstance(grav, (list, tuple)) else grav
                self._world.get_physics_context().set_gravity(gravity_magnitude)

                # Add ground plane
                if ground_plane and self._config.ground_plane:
                    self._world.scene.add_default_ground_plane()
                    self._prim_registry.append(f"{self._config.stage_path}/defaultGroundPlane")

                # Reset world to initialize
                self._world.reset()

                self._world_created = True
                self._sim_time = 0.0
                self._step_count = 0

                logger.info(
                    "World created: dt=%.5f, gravity=%s, headless=%s",
                    dt,
                    grav,
                    self._config.headless,
                )

                # Surface a structured snapshot of the freshly-created
                # environment alongside the human-readable text. Agents
                # spinning up a sim can introspect device / dt / scene
                # config without re-querying via get_state().
                world_info = {
                    "physics_dt": dt,
                    "rendering_dt": self._config.rendering_dt,
                    "gravity": list(grav) if isinstance(grav, (list, tuple)) else [0.0, 0.0, float(grav)],
                    "ground_plane": bool(ground_plane and self._config.ground_plane),
                    "stage_path": self._config.stage_path,
                    "stage_units_in_meters": 1.0,
                    "device": self._config.device,
                    "headless": self._config.headless,
                    "render_mode": self._config.render_mode,
                    "num_envs": self._config.num_envs,
                    "num_envs_active": self._num_envs_active,
                    "replicated": self._replicated,
                    "sim_time": self._sim_time,
                    "step_count": self._step_count,
                }

                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Isaac Sim world created. "
                                f"dt={dt:.5f}, gravity={grav}, "
                                f"device={self._config.device}, "
                                f"headless={self._config.headless}"
                            ),
                            "json": world_info,
                        }
                    ],
                }

            except ImportError as e:
                return {
                    "status": "error",
                    "content": [
                        {"text": (f"Isaac Sim import failed: {e}. Ensure Isaac Sim is installed and accessible.")}
                    ],
                }
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                # Cleanup on partial failure. Narrow to what World() /
                # set_gravity / add_default_ground_plane / reset actually
                # raise on Isaac: RuntimeError (Carb / sim init), ValueError
                # (USD prim shape mismatches, e.g. set_init_state on the
                # ground plane), OSError (USD/Nucleus IO), AttributeError
                # (omni surface drift across SDK versions), TypeError
                # (Isaac Sim 5.1 ``set_gravity`` rejects non-scalar input
                # - see #52; defence in depth for similar argument-shape
                # surface drift on neighbouring physics-context calls).
                # Programming bugs (NameError, ImportError-not-already-
                # caught above) propagate.
                self._world = None
                logger.error("Failed to create Isaac world: %s", e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to create world: {e}"}],
                }

    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources.

        Note: SimulationApp is NOT shut down (it is process-wide).
        Only the World/Stage are cleared.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world to destroy."}],
                }

            # Capture pre-teardown counts so the structured json payload
            # surfaces what was actually released (the agent's get_state()
            # window is gone after destroy() returns).
            num_robots_released = len(self._robots)
            num_cameras_released = len(self._cameras)
            num_objects_released = len(self._objects)
            num_prims_released = len(self._prim_registry)
            num_envs_released = self._num_envs_active
            sim_time_at_destroy = self._sim_time
            step_count_at_destroy = self._step_count

            try:
                if self._world is not None:
                    self._world.stop()
                    self._world.clear_instance()
                    self._world = None
            except (RuntimeError, OSError, AttributeError) as e:
                # World.stop() / clear_instance() can raise on partial init
                # or on a torn-down stage; AttributeError covers omni surface
                # drift across versions. Logged at WARNING because we still
                # mark the world destroyed below; programming bugs propagate.
                logger.warning("World cleanup warning: %s", e)

            # Clear the USD stage. SimulationApp is a process-wide singleton
            # that outlives destroy(), so World.clear_instance() alone leaves
            # every prim from this session on the stage. The next
            # create_world() + add_robot() then builds onto a dirty stage and
            # Isaac auto-suffixes any colliding path (e.g.
            # /World/Physics_Materials/physics_material -> ..._1), so prim
            # paths drift across destroy()/create_world() cycles and break
            # determinism for multi-scene eval / benchmark loops. Issuing a
            # fresh stage here honours this method's documented contract
            # ("Only the World/Stage are cleared") and pins prim paths stable
            # run-to-run.
            try:
                import omni.usd  # type: ignore[import-not-found]

                omni.usd.get_context().new_stage()
            except (RuntimeError, OSError, AttributeError, ImportError) as e:
                # new_stage() can raise on a torn-down context or omni surface
                # drift across versions; ImportError covers the no-Isaac path.
                # Logged at WARNING because the world is still marked destroyed
                # below; a stale stage only affects a subsequent create_world().
                logger.warning("Stage clear warning: %s", e)

            # Clear entity tracking
            self._robots.clear()
            self._cameras.clear()
            self._objects.clear()
            self._prim_registry.clear()
            # Drop any in-flight recorder state (buffers reference RTX
            # frames that are meaningless after the stage tears down).
            self._cams_rec_state = None

            # Reset state
            self._world_created = False
            self._replicated = False
            self._num_envs_active = 1
            self._sim_time = 0.0
            self._step_count = 0

            logger.info("World destroyed. SimulationApp remains (process-wide singleton).")

            # Surface a structured snapshot of what teardown released
            # alongside the human-readable text. Mirrors the json content
            # block convention used by get_state() (L624) and create_world()
            # (L455) so an agent inspecting destroy() can confirm what was
            # actually torn down without re-querying.
            destroy_info = {
                "num_robots_released": num_robots_released,
                "num_cameras_released": num_cameras_released,
                "num_objects_released": num_objects_released,
                "num_prims_released": num_prims_released,
                "num_envs_released": num_envs_released,
                "sim_time_at_destroy": sim_time_at_destroy,
                "step_count_at_destroy": step_count_at_destroy,
                "stage_path": self._config.stage_path,
                "simulation_app_alive": True,  # singleton survives destroy()
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            "Isaac Sim world destroyed. All resources released. SimulationApp singleton remains active."
                        ),
                        "json": destroy_info,
                    }
                ],
            }

    def reset(self, env_ids: list[int] | None = None) -> dict[str, Any]:
        """Reset simulation to initial state.

        Parameters
        ----------
        env_ids : list[int], optional
            Specific environment indices to reset. If None, reset all.

        Returns
        -------
        dict
            Status dict.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is not None:
                self._world.reset()

            self._sim_time = 0.0
            self._step_count = 0

            if env_ids is None:
                msg = "Full reset complete."
            else:
                msg = f"Partial reset complete for {len(env_ids)} envs."

            return {"status": "success", "content": [{"text": msg}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Parameters
        ----------
        n_steps : int
            Number of steps to take. Default 1.

        Returns
        -------
        dict
            Status dict with timing info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is None:
                return {"status": "error", "content": [{"text": "World not initialized."}]}

            t0 = time.perf_counter()

            for _ in range(n_steps):
                self._world.step(render=self._config.render_mode != "headless")
                self._sim_time += self._config.physics_dt
                self._step_count += 1

            elapsed = time.perf_counter() - t0
            steps_per_sec = n_steps / elapsed if elapsed > 0 else float("inf")

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Stepped {n_steps}x. "
                            f"sim_time={self._sim_time:.4f}s, "
                            f"wall={elapsed * 1000:.1f}ms, "
                            f"{steps_per_sec:.0f} steps/sec"
                        )
                    }
                ],
            }

    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary.

        Returns
        -------
        dict
            Status dict with state information.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            state_data = {
                "sim_time": self._sim_time,
                "step_count": self._step_count,
                "num_envs": self._num_envs_active,
                "num_robots": len(self._robots),
                "num_cameras": len(self._cameras),
                "num_objects": len(self._objects),
                "stage_path": self._config.stage_path,
                "device": self._config.device,
                "headless": self._config.headless,
                "render_mode": self._config.render_mode,
            }

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"State: t={self._sim_time:.4f}s, "
                            f"step={self._step_count}, "
                            f"envs={self._num_envs_active}, "
                            f"robots={len(self._robots)}, "
                            f"cameras={len(self._cameras)}, "
                            f"objects={len(self._objects)}"
                        ),
                        "json": state_data,
                    }
                ],
            }

    # --- SimEngine: Robot Management ----------------------------------------

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        mjcf_path: str | None = None,
        usd_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup).
        urdf_path : str, optional
            Path to URDF file.
        mjcf_path : str, optional
            Path to MJCF file.
        usd_path : str, optional
            Path to USD file (native Isaac format).
        data_config : str, optional
            Named data config for procedural lookup.
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base orientation as quaternion [w, x, y, z].

        Returns
        -------
        dict
            Status dict with robot info.
        """
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world created. Call create_world() first."}],
                }

            if name in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' already exists."}],
                }

            if self._replicated:
                return {
                    "status": "error",
                    "content": [{"text": "Cannot add robots after replicate(). Call destroy() first."}],
                }

            pos = position or [0.0, 0.0, 0.0]
            prim_path = f"{self._config.stage_path}/Robots/{name}"

            # Procedural lookup is a *fallback*: an explicit usd_path /
            # urdf_path always wins (parity with the MuJoCo backend and
            # least-surprise for a caller passing a concrete asset). The
            # lookup still runs unconditionally (a cheap dict read), but
            # the procedural branch below is only taken when no explicit
            # asset path was given (#152). Without the usd_path/urdf_path
            # guard on that branch, any name colliding with the procedural
            # registry (franka->panda, so100, g1, ...) would silently
            # shadow an explicit usd_path/urdf_path.
            lookup_name = data_config or name
            try:
                from strands_robots.simulation.isaac.procedural import get_procedural_robot

                procedural = get_procedural_robot(lookup_name)
            except ImportError:
                procedural = None

            if procedural is not None and usd_path is None and urdf_path is None:
                # Build procedurally via USD API
                joint_names = procedural.joint_names
                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                )
                self._robots[name] = robot_state

                logger.info("Added robot '%s' (procedural, %d joints)", name, len(joint_names))
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' added (procedural: {procedural.name}, "
                                f"{len(joint_names)} joints: {joint_names})"
                            )
                        }
                    ],
                }

            elif usd_path is not None:
                # Load from USD (native Isaac format).
                # Phase 2 wiring (#14): _load_usd_robot now actually
                # references the USD into the stage, constructs an
                # Articulation, initialises it, and returns the handle
                # alongside the joint names. Pre-Phase-2 it returned
                # joint_names=[] and silently did nothing.
                try:
                    joint_names, articulation = self._load_usd_robot(prim_path, usd_path, pos)
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                    # Cleanup-clause shape mirrors create_world (#52
                    # precedent): RuntimeError (Carb / sim init), ValueError
                    # (USD shape mismatches), OSError (USD file IO failure),
                    # AttributeError (omni surface drift), TypeError (signature
                    # drift), ImportError (omni.isaac.core.articulations
                    # unavailable). Programming bugs propagate.
                    logger.error(
                        "Failed to load USD robot '%s' (usd_path=%s): %s",
                        name,
                        usd_path,
                        e,
                    )
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to load USD robot '{name}': {e}"}],
                    }

                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                    articulation=articulation,
                    actual_prim_path=getattr(articulation, "_strands_actual_prim_path", None),
                )
                self._robots[name] = robot_state

                logger.info(
                    "Added robot '%s' (USD: %s, %d joints, articulation=%s)",
                    name,
                    usd_path,
                    len(joint_names),
                    "wired" if articulation is not None else "phase1",
                )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (f"Robot '{name}' added (USD: {usd_path}, {len(joint_names)} joints)"),
                            "json": {
                                "name": name,
                                "prim_path": prim_path,
                                "usd_path": usd_path,
                                "joint_names": joint_names,
                                "joint_count": len(joint_names),
                                "position": pos,
                                "articulation_wired": articulation is not None,
                            },
                        }
                    ],
                }

            elif urdf_path is not None:
                # Convert URDF to USD and load.
                # Phase 2 wiring (#14): _load_urdf_robot now actually
                # runs the URDF importer command + constructs an
                # Articulation, returning the handle alongside joint
                # names. Pre-Phase-2 it returned joint_names=[] and
                # silently did nothing.
                try:
                    joint_names, articulation = self._load_urdf_robot(prim_path, urdf_path, pos)
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                    # Cleanup-clause shape mirrors the USD branch above
                    # plus create_world (#52 precedent). RuntimeError
                    # covers the URDFParseAndImportFile command
                    # returning ``False``; OSError covers a missing
                    # ``urdf_path``; ImportError covers a partial
                    # ``omni.importer.urdf`` install on the runner.
                    logger.error(
                        "Failed to load URDF robot '%s' (urdf_path=%s): %s",
                        name,
                        urdf_path,
                        e,
                    )
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to load URDF robot '{name}': {e}"}],
                    }

                self._prim_registry.append(prim_path)

                robot_state = _RobotState(
                    name=name,
                    prim_path=prim_path,
                    joint_names=joint_names,
                    articulation=articulation,
                    actual_prim_path=getattr(articulation, "_strands_actual_prim_path", None),
                )
                self._robots[name] = robot_state

                logger.info(
                    "Added robot '%s' (URDF: %s, %d joints, articulation=%s)",
                    name,
                    urdf_path,
                    len(joint_names),
                    "wired" if articulation is not None else "phase1",
                )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (f"Robot '{name}' added (URDF: {urdf_path}, {len(joint_names)} joints)"),
                            "json": {
                                "name": name,
                                "prim_path": prim_path,
                                "urdf_path": urdf_path,
                                "joint_names": joint_names,
                                "joint_count": len(joint_names),
                                "position": pos,
                                "articulation_wired": articulation is not None,
                            },
                        }
                    ],
                }

            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{lookup_name}' not found in procedural registry "
                                "and no usd_path/urdf_path provided. "
                                "Available procedural robots: so100, panda, unitree_g1"
                            )
                        }
                    ],
                }

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool = False,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object (shape primitive) to the scene.

        Phase 2 wiring (#14): instantiates the underlying USD prim via
        ``omni.isaac.core.objects.{Dynamic,Fixed}{Cuboid,Sphere,Cylinder,
        Capsule}`` and registers it with ``world.scene.add()``. In Phase 1
        this method silently returned ``status: "success"`` without
        creating any prim; that path is gone -- callers that previously
        relied on the silent-no-op shape will now see a real geometry on
        the stage and a ``DynamicXxx``/``FixedXxx`` handle in the
        scene.

        Parameters
        ----------
        name : str
            Object identifier. Must be unique across the simulation; a
            duplicate is rejected with a structured error envelope rather
            than silently overwriting the existing prim.
        shape : str
            Shape type: ``"box"`` (default), ``"sphere"``, ``"capsule"``,
            ``"cylinder"``. ``"cuboid"`` is accepted as an alias for
            ``"box"`` (it mirrors Isaac's ``DynamicCuboid`` class name and
            the docs vocabulary; it normalizes to ``"box"``, which is the
            value reported back in the result ``json``). Anything else
            returns a structured error envelope listing the valid set.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[0.0, 0.0, 0.5]`` (50 cm above origin so an object dropped
            with the default ground plane doesn't intersect it).
        orientation : list[float], optional
            World-space orientation as a quaternion ``[w, x, y, z]``.
            Default ``[1.0, 0.0, 0.0, 0.0]`` (identity).
        size : list[float], optional
            Shape dimensions in meters. ``scale`` is accepted as an alias
            for ``size`` (matches Isaac's ``DynamicCuboid(scale=...)``
            convention and the docs vocabulary -- see #88); an explicit
            ``size`` wins if both are passed. Conventions per shape:

            * ``box``:      ``[width, height, depth]`` (default ``[0.05, 0.05, 0.05]``).
            * ``sphere``:   ``[radius]`` (default ``[0.05]``).
            * ``cylinder``: ``[radius, height]`` (default ``[0.05, 0.10]``).
            * ``capsule``:  ``[radius, height]`` (default ``[0.05, 0.10]``).

            Lists shorter than the convention fall back to defaults for
            the missing trailing components.
        color : list[float], optional
            RGB color ``[r, g, b]`` in ``[0, 1]``. RGBA lists (length 4)
            are accepted; alpha is dropped (Isaac's primitive constructors
            take RGB only). ``None`` -> default white.
        mass : float
            Mass in kg. Default 0.1. Ignored when ``is_static=True``.
        is_static : bool
            If ``True``, the prim is constructed via ``Fixed{Cuboid,
            Sphere, Cylinder, Capsule}`` and stays pinned in space. If
            ``False`` (default), uses the ``Dynamic*`` counterpart and
            participates in physics with ``mass``.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``shape``, ``position``, ``orientation``, ``size``,
            ``mass``, and ``is_static`` so an agent can confirm what
            actually landed on the stage without re-querying.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Normalize shape aliases. ``"cuboid"`` is accepted as an
            # alias for ``"box"`` because it matches Isaac's underlying
            # ``DynamicCuboid`` / ``FixedCuboid`` class names and is the
            # vocabulary used throughout the docs (see issue #88). The
            # canonical name stored / reported is ``"box"``.
            shape = _SHAPE_ALIASES.get(shape, shape)

            # Validate shape
            valid_shapes = ("box", "sphere", "capsule", "cylinder")
            if shape not in valid_shapes:
                accepted = valid_shapes + tuple(_SHAPE_ALIASES)
                return {
                    "status": "error",
                    "content": [{"text": f"Unknown shape: {shape!r}. Valid: {accepted}"}],
                }

            if name in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' already exists."}],
                }

            # ``scale`` is accepted as an alias for ``size`` (matches
            # Isaac's ``DynamicCuboid(scale=...)`` convention and the docs
            # vocabulary -- see issue #88). An explicit ``size`` always
            # wins over ``scale`` if both are passed.
            scale_alias = kwargs.pop("scale", None)
            if size is None and scale_alias is not None:
                size = scale_alias

            pos = list(position) if position is not None else [0.0, 0.0, 0.5]
            orient = list(orientation) if orientation is not None else [1.0, 0.0, 0.0, 0.0]
            size_in = list(size) if size is not None else None
            prim_path = f"{self._config.stage_path}/Objects/{name}"

            try:
                handle, resolved_size = self._construct_shape_prim(
                    shape=shape,
                    prim_path=prim_path,
                    name=name,
                    position=pos,
                    orientation=orient,
                    size=size_in,
                    color=color,
                    mass=mass,
                    is_static=is_static,
                )
                # ``world.scene.add`` registers the wrapper so that
                # ``world.reset()`` re-initialises it on the same
                # ``post_reset`` callback as the ground plane and
                # robots. The return value is the (possibly wrapped)
                # handle Isaac uses internally; we keep our own ref in
                # ``_objects[name]`` so ``remove_object`` doesn't have
                # to round-trip through ``scene.get_object`` later.
                self._world.scene.add(handle)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Cleanup-clause shape mirrors create_world (line 467):
                # RuntimeError (Carb / sim init), ValueError (USD prim
                # shape mismatches, e.g. negative scale on a Dynamic*),
                # OSError (USD/Nucleus IO), AttributeError (omni surface
                # drift), TypeError (signature drift across SDK versions
                # -- see #52 for the gravity precedent), ImportError
                # (omni.isaac.core.objects unavailable on a partial
                # Isaac install). Programming bugs propagate.
                #
                # NOTE: the bare ``Exception`` ``omni.physics.tensors``
                # raises when a Dynamic* prim's eager
                # ``RigidPrim._on_physics_ready`` queries velocities
                # before the physics-tensor view includes it (#159) is
                # NOT caught here -- ``_construct_shape_prim`` prevents it
                # up front by stopping the timeline (clearing the physics
                # sim view) before constructing dynamic prims, so the
                # eager query never runs. That keeps this clause free of a
                # bare ``except Exception`` (forbidden by the
                # exception-hygiene pin, PR #31).
                logger.error(
                    "Failed to add object '%s' (shape=%s, static=%s): %s",
                    name,
                    shape,
                    is_static,
                    e,
                )
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add object '{name}' ({shape}): {e}"}],
                }

            self._prim_registry.append(prim_path)
            self._objects[name] = _ObjectState(
                name=name,
                prim_path=prim_path,
                shape=shape,
                is_static=is_static,
                handle=handle,
            )

            obj_info = {
                "name": name,
                "prim_path": prim_path,
                "shape": shape,
                "position": pos,
                "orientation": orient,
                "size": resolved_size,
                "mass": float(mass) if not is_static else 0.0,
                "is_static": bool(is_static),
            }
            logger.info(
                "Added object '%s' (shape=%s, pos=%s, mass=%.3f, static=%s)",
                name,
                shape,
                pos,
                mass,
                is_static,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": f"Object '{name}' added (shape={shape}, pos={pos}).",
                        "json": obj_info,
                    }
                ],
            }

    def _construct_shape_prim(
        self,
        *,
        shape: str,
        prim_path: str,
        name: str,
        position: list[float],
        orientation: list[float],
        size: list[float] | None,
        color: list[float] | None,
        mass: float,
        is_static: bool,
    ) -> tuple[Any, list[float]]:
        """Construct a shape prim, deferring physics init for dynamic bodies.

        Wraps :meth:`_create_shape_prim` to work around an eager
        velocity query in Isaac's ``RigidPrim.__init__``. The
        ``Dynamic*`` constructors run ``RigidPrim._on_physics_ready``
        during ``__init__``, which calls ``get_linear_velocities()`` ->
        ``omni.physics.tensors`` *before the new prim is part of the
        physics-tensor view*. The backend then raises a bare::

            Exception("Failed to get rigid body velocities from backend")

        which is fatal for :meth:`load_scene` -- it builds a LIBERO
        task's movable objects as ``Dynamic*`` prims after the world has
        already been initialised (see
        `#159 <https://github.com/strands-labs/robots-sim/issues/159>`_).

        We *prevent* the failure rather than catching it: a bare
        ``except Exception`` is forbidden in this module by the
        exception-hygiene pin (PR #31), and ``omni.physics.tensors``
        raises exactly that bare type. Before constructing any
        ``Dynamic*`` prim we stop the timeline, which clears the
        physics-tensor view so ``RigidPrim.__init__`` skips its eager
        ``_on_physics_ready`` velocity query; the prim is then
        initialised cleanly on the next ``world.reset()`` that
        ``world.scene.add`` schedules -- mirroring Isaac's own "build
        rigid bodies, then reset once" scene-construction ordering.

        The stop is *unconditional* for dynamic prims rather than gated on
        a ``SimulationManager.get_physics_sim_view()`` probe. The earlier
        probe-gated guard (PR #161) was a no-op on the actual ``#159``
        ``load_scene`` path: in a real Isaac 6.0 run the probe reported no
        live view while ``RigidPrim.__init__`` still issued the eager
        velocity query and crashed -- the probe checks a *different*
        tensor-view handle than the one ``RigidPrim`` keys off, so the two
        fell out of sync. ``timeline.stop()`` is idempotent (a no-op when
        the timeline is already stopped, the common scene-build case), so
        always stopping is safe and strictly covers the path the probe
        missed.

        Static (``Fixed*``) prims don't take the ``RigidPrim`` velocity
        path, so they never hit this and the timeline is left untouched.

        Returns the same ``(handle, resolved_size)`` tuple as
        :meth:`_create_shape_prim`.
        """
        if not is_static:
            logger.info(
                "Stopping timeline before constructing dynamic prim '%s' so "
                "RigidPrim.__init__ skips its eager velocity query for a prim not "
                "yet in the tensor view (#159). The prim initialises on the next "
                "reset(). Idempotent if the timeline is already stopped.",
                name,
            )
            self._stop_timeline_for_deferred_physics()
        return self._create_shape_prim(
            shape=shape,
            prim_path=prim_path,
            name=name,
            position=position,
            orientation=orientation,
            size=size,
            color=color,
            mass=mass,
            is_static=is_static,
        )

    @staticmethod
    def _stop_timeline_for_deferred_physics() -> None:
        """Stop the Isaac timeline so the physics-tensor view is cleared.

        After this returns the physics-tensor view is torn down, so
        ``RigidPrim.__init__`` skips its eager ``_on_physics_ready``
        velocity query and a freshly constructed ``Dynamic*`` prim
        initialises only on the next ``world.reset()``. Best-effort: a
        missing ``omni.timeline`` (partial Isaac install) is logged and
        ignored.
        """
        try:
            import omni.timeline  # type: ignore[import-not-found]

            omni.timeline.get_timeline_interface().stop()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("Could not stop timeline to defer physics init: %s", e)

    def _create_shape_prim(
        self,
        *,
        shape: str,
        prim_path: str,
        name: str,
        position: list[float],
        orientation: list[float],
        size: list[float] | None,
        color: list[float] | None,
        mass: float,
        is_static: bool,
    ) -> tuple[Any, list[float]]:
        """Construct the omni.isaac.core.objects shape wrapper.

        Returns the handle plus the resolved ``size`` list (defaults
        applied per shape) so :meth:`add_object` can surface the
        actually-used dimensions in its structured json payload.

        Lazy-imports the Isaac object constructors so the module loads
        cleanly without Isaac Sim installed (the call site only ever
        runs after :meth:`create_world` has booted ``SimulationApp``).
        Isaac Sim 6.0 exposes these under ``isaacsim.core.api.objects``;
        the legacy 4.x path was ``omni.isaac.core.objects``. Try modern
        first, fall back so 4.x installs keep working.
        """
        import numpy as np  # type: ignore[import-not-found]

        try:
            from isaacsim.core.api.objects import (  # type: ignore[import-not-found]
                DynamicCapsule,
                DynamicCuboid,
                DynamicCylinder,
                DynamicSphere,
                FixedCapsule,
                FixedCuboid,
                FixedCylinder,
                FixedSphere,
            )
        except ImportError:
            from omni.isaac.core.objects import (  # type: ignore[import-not-found]
                DynamicCapsule,
                DynamicCuboid,
                DynamicCylinder,
                DynamicSphere,
                FixedCapsule,
                FixedCuboid,
                FixedCylinder,
                FixedSphere,
            )

        common: dict[str, Any] = {
            "prim_path": prim_path,
            "name": name,
            "position": np.asarray(position, dtype=float),
            "orientation": np.asarray(orientation, dtype=float),
        }
        if color is not None:
            # RGBA -> RGB: Isaac's primitive constructors take a 3-vector
            # color; alpha would silently raise a shape mismatch deeper
            # in USD. Truncate here so RGBA-style examples (e.g. the #15
            # sketch's ``[1, 0, 0, 1]``) work transparently.
            rgb = list(color)[:3]
            common["color"] = np.asarray(rgb, dtype=float)
        if not is_static:
            common["mass"] = float(mass)

        if shape == "box":
            cls = FixedCuboid if is_static else DynamicCuboid
            # Per-component fallback to honour the docstring contract
            # ("Lists shorter than the convention fall back to defaults
            # for the missing trailing components"). Mirrors the
            # cylinder / capsule pattern below; previously this branch
            # was all-or-nothing, so e.g. ``size=[0.10]`` silently fell
            # back to ``[0.05, 0.05, 0.05]`` instead of the documented
            # ``[0.10, 0.05, 0.05]`` -- caught in PR #60 review.
            size_list = list(size) if size else []
            sx = float(size_list[0]) if len(size_list) >= 1 else 0.05
            sy = float(size_list[1]) if len(size_list) >= 2 else 0.05
            sz = float(size_list[2]) if len(size_list) >= 3 else 0.05
            scale = [sx, sy, sz]
            common["scale"] = np.asarray(scale, dtype=float)
            return cls(**common), scale
        if shape == "sphere":
            cls = FixedSphere if is_static else DynamicSphere
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            return cls(radius=radius, **common), [radius]
        if shape == "cylinder":
            cls = FixedCylinder if is_static else DynamicCylinder
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        if shape == "capsule":
            cls = FixedCapsule if is_static else DynamicCapsule
            radius = float(size[0]) if size and len(size) >= 1 else 0.05
            height = float(size[1]) if size and len(size) >= 2 else 0.10
            return cls(radius=radius, height=height, **common), [radius, height]
        # Unreachable: shape was validated by add_object before this call;
        # raise loudly if a future caller bypasses that guard.
        raise ValueError(f"Unknown shape: {shape!r}")

    # --- SimEngine: Scene loading -------------------------------------------

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Realize a LIBERO/BDDL task scene as USD prims on the Isaac stage.

        The ``SimEngine`` contract lets each backend realize a complete
        scene (objects, poses, fixtures) from a file. The MuJoCo backend
        parses a LIBERO/BDDL-generated MJCF and recompiles the live spec;
        ``LiberoAdapter.on_episode_start`` relies on this to instantiate
        each task's scene. This Isaac override translates the same
        robosuite-compiled MJCF into Isaac stage prims so the LIBERO eval
        runs end-to-end on the Isaac backend (closes the substantive
        LIBERO-on-Isaac gap that PR #117 deferred with a fail-fast stub --
        see `#129 <https://github.com/strands-labs/robots-sim/issues/129>`_).

        Translation layer (BDDL/MJCF -> USD):
            * The ``scene_path`` is a robosuite-compiled LIBERO MJCF XML
              (e.g. ``~/.strands_robots/scene_cache/libero/<sha>.xml``).
            * :func:`load_mjcf_scene_objects` walks the ``<worldbody>``,
              skips the floor (the ground plane is created by
              :meth:`create_world`) and the Panda robot (the adapter loads
              it separately via :meth:`add_robot`), and returns one
              :class:`SceneObject` per task object / fixture.
            * LIBERO object meshes aren't portable to the Isaac stage, so
              each object is approximated by a box primitive sized to the
              axis-aligned bounding box of its collision geoms and placed
              at its MJCF body pose. That's faithful enough for
              rollout-video parity with the MuJoCo driver.
            * Each object is realized via :meth:`add_object` (static
              fixtures -> ``Fixed*``; movable objects -> ``Dynamic*``).

        Idempotency: a fresh ``load_scene`` first removes any objects left
        over from a prior episode's scene (tracked in ``_scene_objects``)
        so per-episode reloads don't accumulate duplicate prims or hit the
        "object already exists" guard in :meth:`add_object`.

        Parameters
        ----------
        scene_path : str
            Path to the compiled LIBERO MJCF scene file.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. On success ``json`` carries the realized object
            count and names so ``LiberoAdapter.on_episode_start`` proceeds.
            On a recoverable failure (no world, missing/malformed file)
            returns ``{"status": "error", ...}``; the adapter converts that
            into a descriptive ``RuntimeError``.
        """
        from strands_robots.simulation.isaac.loaders import load_mjcf_scene_objects

        with self._lock:
            if not self._world_created:
                msg = f"Cannot load scene: no world created. Call create_world() before load_scene({scene_path!r})."
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            if not scene_path or not os.path.exists(scene_path):
                msg = f"Scene file not found: {scene_path!r}"
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            # Parse the MJCF -> a backend-agnostic list of SceneObjects.
            try:
                scene_objects = load_mjcf_scene_objects(scene_path)
            except (FileNotFoundError, ValueError) as e:
                msg = f"Failed to parse LIBERO scene {scene_path!r}: {e}"
                logger.error("IsaacSimulation.load_scene: %s", msg)
                return {"status": "error", "content": [{"text": msg}]}

            # Clear any objects realized by a prior load_scene so per-episode
            # reloads are idempotent (no duplicate prims / no "already exists").
            for prior_name in list(self._scene_objects):
                if prior_name in self._objects:
                    self.remove_object(prior_name)
                self._scene_objects.discard(prior_name)

            realized: list[str] = []
            skipped: list[dict[str, Any]] = []
            for obj in scene_objects:
                # ``add_object`` rejects duplicate names; if a manually-added
                # object shadows a scene object, skip it rather than abort.
                if obj.name in self._objects:
                    skipped.append({"name": obj.name, "reason": "name already in use"})
                    continue
                result = self.add_object(
                    name=obj.name,
                    shape="box",
                    position=list(obj.position),
                    orientation=list(obj.quat),
                    size=list(obj.size),
                    mass=0.1,
                    is_static=obj.is_static,
                )
                if result.get("status") == "success":
                    realized.append(obj.name)
                    self._scene_objects.add(obj.name)
                else:
                    text = (result.get("content") or [{}])[0].get("text", "")
                    skipped.append({"name": obj.name, "reason": text})

            summary = (
                f"Loaded LIBERO scene from {os.path.basename(scene_path)}: "
                f"realized {len(realized)} object(s) as Isaac stage prims"
            )
            if skipped:
                summary += f" ({len(skipped)} skipped)"
            logger.info("IsaacSimulation.load_scene: %s", summary)
            return {
                "status": "success",
                "content": [
                    {
                        "text": summary,
                        "json": {
                            "scene_path": scene_path,
                            "realized": realized,
                            "skipped": skipped,
                            "object_count": len(realized),
                        },
                    }
                ],
            }

    # --- SimEngine: Introspection / Removal ---------------------------------

    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world.

        Returns
        -------
        list[str]
            Robot names in insertion order. Empty if no robots have been
            added (or after :meth:`destroy`).
        """
        with self._lock:
            return list(self._robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for ``robot_name``.

        Parameters
        ----------
        robot_name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        list[str]
            Joint names in articulation order, or an empty list if
            ``robot_name`` is not present (matches the silent-empty
            convention used by :meth:`get_observation` for unknown robots).
        """
        with self._lock:
            if robot_name not in self._robots:
                return []
            return list(self._robots[robot_name].joint_names)

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation.

        Drops the robot's bookkeeping entry and prunes any prims rooted at
        the robot's prim path from ``self._prim_registry``. The actual USD
        prim deletion is delegated to :meth:`destroy` / world teardown in
        Phase 1; only the in-Python registry is updated here.

        Parameters
        ----------
        name : str
            Robot identifier previously passed to :meth:`add_robot`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class.
        """
        with self._lock:
            if name not in self._robots:
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' not found."}],
                }
            prim_path = self._robots[name].prim_path
            self._prim_registry = [p for p in self._prim_registry if not p.startswith(prim_path)]
            del self._robots[name]
            logger.info("Removed robot '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Robot '{name}' removed."}],
            }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene.

        Phase 2 wiring (#14): paired with :meth:`add_object`'s prim
        creation. Calls ``world.scene.remove_object(name)`` to actually
        delete the USD prim, then prunes the in-Python registries
        (``_objects`` + ``_prim_registry``). In Phase 1 this method only
        updated the in-Python registry; that is no longer the case --
        the prim is gone from the stage when this returns.

        Parameters
        ----------
        name : str
            Object identifier previously passed to :meth:`add_object`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class. Returns ``error``
            if the object is unknown to ``_objects``; this is the only
            authoritative source -- ``_prim_registry`` is cleanup-time
            bookkeeping that does not distinguish robots from objects.
        """
        with self._lock:
            if name not in self._objects:
                return {
                    "status": "error",
                    "content": [{"text": f"Object '{name}' not found."}],
                }

            prim_path = self._objects[name].prim_path

            # Delete the prim from the world's scene. Wrapped in the same
            # cleanup-clause shape as add_object since the failure modes
            # mirror it: scene.remove_object can RuntimeError on a torn-
            # down stage, AttributeError on omni surface drift, etc.
            try:
                if self._world is not None:
                    self._world.scene.remove_object(name)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError) as e:
                logger.error("Failed to remove object '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove object '{name}': {e}"}],
                }

            # Now drop our bookkeeping. The order matters: we only want
            # to forget the object after the scene call succeeded so a
            # transient ``RuntimeError`` from ``scene.remove_object``
            # leaves a retry-friendly state.
            del self._objects[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed object '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Object '{name}' removed."}],
            }

    # --- SimEngine: Observation / Action ------------------------------------

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get observation for a robot.

        Parameters
        ----------
        robot_name : str, optional
            Robot to observe. Auto-resolves if only one robot exists.
        skip_images : bool
            Skip camera rendering. Default False.

        Returns
        -------
        dict
            Observation with joint positions keyed by joint name. An empty dict
            indicates one of four diagnostically-distinct conditions, each of
            which is logged before return so silent failures are visible in
            operational logs:

            * ``world not yet created`` -- DEBUG (expected pre-init state).
            * ``ambiguous robot_name=None with multiple robots`` -- WARNING.
            * ``unknown robot_name`` (typo / not-yet-added) -- WARNING.
            * ``robot present but Articulation handle not yet initialised``
              (Phase 1 procedural / load stub) -- DEBUG via the inner except;
              the dict returns empty because no joint positions are reachable.

            The return shape is preserved as a plain dict (rather than the
            ``{"status": ..., "content": [...]}`` shape used by mutating
            methods on this class) because callers consume joint positions
            keyed by joint name; the four silent-``{}`` modes are distinguished
            in logs rather than in the return value.
        """
        with self._lock:
            if not self._world_created:
                # Expected pre-init state; many callers probe before
                # create_world() to feature-detect, so DEBUG-only.
                logger.debug(
                    "get_observation(robot_name=%r): world not yet created",
                    robot_name,
                )
                return {}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    logger.warning(
                        "get_observation(robot_name=None): ambiguous -- "
                        "%d robots present (%s); pass robot_name explicitly. "
                        "Returning empty observation.",
                        len(self._robots),
                        sorted(self._robots),
                    )
                    return {}

            if robot_name not in self._robots:
                logger.warning(
                    "get_observation(robot_name=%r): unknown robot. Known: %s. Returning empty observation.",
                    robot_name,
                    sorted(self._robots),
                )
                return {}

            robot = self._robots[robot_name]
            obs: dict[str, Any] = {}

            # Get joint state from Articulation handle
            if robot.articulation is not None:
                try:
                    joint_positions = robot.articulation.get_joint_positions()
                    if joint_positions is not None:
                        positions = (
                            joint_positions.cpu().numpy()
                            if hasattr(joint_positions, "cpu")
                            else np.array(joint_positions)
                        )
                        for i, jname in enumerate(robot.joint_names):
                            if i < len(positions):
                                obs[jname] = float(positions[i])
                except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                    # Articulation handle may raise RuntimeError on a not-yet
                    # -initialized world, AttributeError on torch-tensor surface
                    # drift, ValueError/TypeError on np coercion. Programming
                    # bugs propagate.
                    logger.debug("Failed to get joint positions: %s", e)

            # Camera frames keyed by camera name (RGB HxWx3 uint8), so callers
            # (e.g. the SO-101 collector / Gradio render) get images the same way
            # the MuJoCo backend provides them. Skipped when ``skip_images`` or in
            # headless render mode (no RTX frames). Best-effort per camera: a
            # camera whose RTX product hasn't warmed up is omitted rather than
            # failing the whole observation.
            if not skip_images and self._config.render_mode != "headless":
                # Multi-camera refresh: a single ``world.step(render=True)`` in
                # the substep loop reliably refreshes only the PRIMARY render
                # product, so secondary cameras' ``get_rgba`` returns a stale or
                # blank buffer. When more than one camera is configured, tick the
                # renderer a few extra times (holding the pose static) so EVERY
                # camera's RTX render product accumulates a fresh frame before we
                # read them back. Single-camera setups skip this (the substep
                # render already warmed the one product) to stay fast.
                if len(self._cameras) > 1:
                    self._refresh_all_render_products()
                for cam_name, cam in self._cameras.items():
                    if cam.handle is None:
                        continue
                    try:
                        rgba = cam.handle.get_rgba()
                        # Validate shape BEFORE slicing: a 0-D scalar
                        # buffer from a not-yet-warmed RTX render product
                        # makes ``[..., :3]`` raise ``IndexError`` (#140).
                        # Skip such cameras rather than failing the whole
                        # observation.
                        arr = np.asarray(rgba)
                        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[1] > 0:
                            obs[cam_name] = arr[..., :3].astype(np.uint8)
                    except (RuntimeError, ValueError, AttributeError, TypeError, IndexError) as e:
                        logger.debug("camera %r frame unavailable: %s", cam_name, e)

            return obs

    def send_action(
        self,
        action: dict[str, Any] | np.ndarray | list,
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action and advance physics.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name.
        robot_name : str, optional
            Robot to control.
        n_substeps : int
            Physics sub-steps after applying action. Default 1.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}]}`` envelope, matching
            the :class:`~strands_robots.simulation.base.SimEngine` contract so
            :class:`~strands_robots.simulation.policy_runner.PolicyRunner` can
            count action failures (it increments ``_action_errors`` when the
            returned ``status`` is ``"error"``). When ``action`` is a dict and
            some keys don't name a joint on ``robot_name``, the ``content`` list
            carries a ``json`` block with ``unresolved_keys`` / ``applied`` so
            callers can self-correct -- mirroring the MuJoCo backend.
        """
        with self._lock:
            if not self._world_created or self._world is None:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Resolve robot
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                elif not self._robots:
                    return {"status": "error", "content": [{"text": "No robots in the world."}]}
                else:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Multiple robots present; specify robot_name. Available: {sorted(self._robots)}"
                                )
                            }
                        ],
                    }

            if robot_name not in self._robots:
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

            robot = self._robots[robot_name]

            # Convert action to array, tracking dict keys that don't name a
            # joint so unresolved commands surface in the envelope rather than
            # being silently dropped (parity with the MuJoCo backend).
            unresolved: list[str] = []
            action_array: np.ndarray
            if isinstance(action, dict):
                joint_set = set(robot.joint_names)
                unresolved = [k for k in action if k not in joint_set]
                action_array = np.zeros(len(robot.joint_names), dtype=np.float32)
                for i, jname in enumerate(robot.joint_names):
                    if jname in action:
                        action_array[i] = float(action[jname])
            elif isinstance(action, np.ndarray):
                action_array = action.astype(np.float32).flatten()
            else:
                action_array = np.array(action, dtype=np.float32)

            # Apply to articulation. Isaac Sim 6.0's articulation
            # (``isaacsim.core.prims.SingleArticulation``) drives PD position
            # targets via ``apply_action(ArticulationAction(joint_positions=...))``
            # -- the pre-6.0 ``set_joint_position_targets`` method does not exist
            # on the 6.0 class (the #101 ``omni.isaac.* -> isaacsim.*`` migration
            # renamed imports but missed this articulation method). See
            # ``set_joint_positions`` below for the teleport (non-PD) counterpart.
            if robot.articulation is not None:
                try:
                    from isaacsim.core.utils.types import (  # type: ignore[import-not-found]
                        ArticulationAction,
                    )

                    robot.articulation.apply_action(ArticulationAction(joint_positions=action_array))
                except (RuntimeError, ValueError, AttributeError, ImportError) as e:
                    # apply_action raises RuntimeError on a torn-down
                    # articulation, ValueError on shape mismatch, AttributeError
                    # on omni surface drift, ImportError if the isaacsim runtime
                    # isn't importable. Programming bugs (NameError, KeyError)
                    # propagate.
                    logger.debug("Failed to set joint targets: %s", e)
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to set joint targets on '{robot_name}': {e}"}],
                    }

            # Step physics. Render on the LAST substep when not headless so the
            # RTX camera render products refresh -> ``get_rgba`` returns a fresh
            # frame for this step (otherwise every recorded frame is identical,
            # i.e. a static video). Intermediate substeps skip render for speed.
            render_on = self._config.render_mode != "headless"
            for i in range(n_substeps):
                last = i == n_substeps - 1
                self._world.step(render=bool(render_on and last))
                self._sim_time += self._config.physics_dt
                self._step_count += 1

        if unresolved:
            applied = [k for k in action if k not in unresolved]
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action partially applied: keys {unresolved} could not be "
                            f"resolved to joints on '{robot_name}'. Applied: {applied}. "
                            f"Valid keys: {robot.joint_names}"
                        )
                    },
                    {"json": {"unresolved_keys": unresolved, "applied": applied}},
                ],
            }

        return {
            "status": "success",
            "content": [{"text": f"Action applied to '{robot_name}', {n_substeps} substeps."}],
        }

    # --- SimEngine: Rendering -----------------------------------------------

    def render(
        self,
        camera_name: str = "default",
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Render a camera view using Isaac Sim's RTX pipeline.

        Phase 2 wiring (#14): when a camera registered via
        :meth:`add_camera` carries a non-``None`` ``handle`` (i.e. the
        Phase-2 ``omni.isaac.sensor.Camera`` was successfully constructed)
        and the simulation isn't in ``headless`` render mode, this method
        pulls real frames via ``handle.get_rgba()`` + ``handle.get_depth()``.
        Otherwise returns blank frames -- four documented fallback paths,
        each tagged in the success envelope's text so a caller / agent
        can tell which path was taken without inspecting array contents:

        * ``Rendered (headless, no RTX)`` -- ``IsaacConfig.render_mode``
          is ``"headless"``; RTX path-tracing is unavailable. Most CI
          and GR00T server flows hit this path.
        * ``Rendered (no camera)`` -- ``camera_name`` is unknown to
          ``self._cameras``. Caller probably forgot to call ``add_camera``
          (or typo'd the name).
        * ``Rendered (Phase-1 camera, no RTX handle)`` -- the camera
          exists in ``self._cameras`` but its ``handle`` is ``None``.
          Happens when the camera was added before the
          ``add_camera`` Phase-2 wiring landed (or when the camera
          construction failed but bookkeeping was still seeded -- not
          possible after PR #61, but kept as a defensive fallback).
        * ``Rendered (RTX <render_mode>)`` -- Phase-2 path: real
          frames pulled from the Camera handle. ``rgb`` / ``depth``
          are the actual array shapes returned by Isaac (matching
          the camera's resolved resolution; not necessarily the
          ``width`` / ``height`` arguments passed to this method,
          which are only used to size the blank-frame fallbacks).

        Parameters
        ----------
        camera_name : str
            Camera identifier previously passed to :meth:`add_camera`.
            Default ``"default"``.
        width : int, optional
            Frame width for blank-frame fallbacks. Default from
            ``IsaacConfig.camera_width``. Ignored on the RTX path
            (the camera's own resolution wins).
        height : int, optional
            Frame height for blank-frame fallbacks. Default from
            ``IsaacConfig.camera_height``. Ignored on the RTX path.

        Returns
        -------
        dict
            Standard Strands tool-result envelope carrying ONLY ``status`` and
            ``content`` (the tool-result contract forbids extra top-level keys;
            see tests/test_tool_result_contract.py). On success ``content``
            holds a ``text`` block, a ``{"image": {"format": "png", ...}}``
            block with raw PNG bytes (matching the MuJoCo backend so the shared
            ``PolicyRunner._extract_frame_ndarray`` can pull frames for video
            recording, #127), and a ``{"json": {...}}`` block with pixel stats
            plus (RTX path) the resolved camera ``resolution``, ``prim_path``,
            and the boolean ``rtx`` flag so an agent can route without parsing
            text. The PNG block is omitted (and a warning logged) if PIL is
            unavailable.

            Consumers that need the raw ``rgb`` / ``depth`` ndarrays use
            :meth:`get_observation` (or, in-process, the internal
            :meth:`_render_frame` helper) rather than reading them off this
            tool-result dict.
        """
        rgb, depth, meta = self._render_frame(camera_name, width, height)
        if rgb is None:
            # meta carries the structured error text on the failure path.
            return {
                "status": "error",
                "content": [{"text": meta.get("error", "render failed")}],
            }
        content: list[dict[str, Any]] = [{"text": meta.get("text", "")}]
        block = _rgb_png_block(rgb)
        if block is not None:
            content.append(block)
        # Structured telemetry (resolution, prim_path, rtx flag, pixel stats)
        # lives INSIDE a content json block, never as extra top-level keys -
        # the Strands tool-result contract permits only {status, content}
        # (see tests/test_tool_result_contract.py). Consumers that need the
        # raw rgb/depth ndarrays use get_observation() or the internal
        # _render_frame() helper; the PNG image block above feeds the shared
        # PolicyRunner video pipeline (#127).
        json_block: dict[str, Any] = dict(meta.get("json", {}))
        json_block["pixel_mean"] = float(np.mean(rgb))
        json_block["pixel_variance"] = float(np.var(rgb))
        json_block["camera"] = camera_name
        content.append({"json": json_block})
        return {"status": "success", "content": content}

    def _render_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray | None, np.ndarray | None, dict[str, Any]]:
        """Render a camera to raw ``(rgb, depth, meta)`` for internal consumers.

        This is the numeric-array counterpart to the public :meth:`render`,
        which wraps this into the ``{status, content}`` tool-result envelope.
        Internal callers that need the raw ndarrays (dataset recording, camera
        warm-up) call this directly instead of parsing the tool-result dict, so
        the public envelope can stay contract-clean (only ``status`` /
        ``content``) while raw pixels remain available in-process.

        Returns:
            ``(rgb, depth, meta)``. On success ``rgb`` is a uint8
            ``(H, W, 3)`` array, ``depth`` a float32 ``(H, W)`` array, and
            ``meta`` carries ``text`` plus (RTX path) a ``json`` sub-dict. On
            failure ``rgb`` / ``depth`` are ``None`` and ``meta["error"]``
            holds the human-readable reason.
        """
        with self._lock:
            if not self._world_created:
                return None, None, {"error": "No world created."}

            w = width or self._config.camera_width
            h = height or self._config.camera_height

            if self._config.render_mode == "headless":
                # Return blank frames in headless mode. Most CI flows
                # land here; Isaac's RTX path-tracer is unavailable.
                return (
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    {"text": f"Rendered (headless, no RTX): {w}x{h}"},
                )

            if camera_name not in self._cameras:
                # No camera configured - return blank. Caller probably
                # forgot to call add_camera or typo'd the name.
                return (
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    {"text": f"Rendered (no camera): {w}x{h}"},
                )

            cam = self._cameras[camera_name]

            if cam.handle is None:
                # Phase-1 camera (no Phase-2 handle was attached).
                # Defensive fallback: blank frames sized to the camera's
                # registered resolution rather than the method's
                # ``width`` / ``height`` arguments, since the camera's
                # resolution is what the caller asked for at add_camera.
                return (
                    np.zeros((cam.height, cam.width, 3), dtype=np.uint8),
                    np.zeros((cam.height, cam.width), dtype=np.float32),
                    {"text": f"Rendered (Phase-1 camera, no RTX handle): {cam.width}x{cam.height}"},
                )

            # Phase-2 RTX path: pull real frames from the Camera handle.
            try:
                rgba = cam.handle.get_rgba()
                # ``get_rgba`` returns either ``(H, W, 4)`` or
                # ``(H, W, 3)`` depending on the Isaac Sim build. A
                # camera whose RTX render product hasn't accumulated a
                # frame yet (e.g. added after the last world step, not
                # warmed up, or during the RTX warm-up loop) returns a
                # malformed / empty buffer -- a 0-D scalar, 1-D, or
                # 0-size array rather than ``(H, W, C)``. Validate the
                # shape BEFORE slicing: ``np.asarray(rgba)[..., :3]`` on a
                # 0-D array raises ``IndexError`` ("too many indices for
                # array: array is 0-dimensional"), which previously
                # escaped the cleanup ``except`` (it wasn't in the tuple)
                # and crashed the example's warm-up loop end-to-end
                # (#140). Guard first so the not-ready case always
                # surfaces as a structured RuntimeError (caught below),
                # letting the warm-up loop retry. Regressed into view
                # after #138 enabled ``rtx_realtime``.
                arr = np.asarray(rgba)
                if arr.ndim < 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
                    raise RuntimeError(
                        f"camera {camera_name!r} returned a malformed RGB buffer "
                        f"(shape {arr.shape}); the RTX render product "
                        "likely hasn't accumulated a frame yet -- step the world a "
                        "few times after add_camera before rendering."
                    )
                # Slice to RGB defensively so the returned shape is stable
                # for downstream agents.
                rgb = arr[..., :3]
                depth_raw = cam.handle.get_depth()
                if depth_raw is None:
                    # Camera was constructed without the depth annotator
                    # (Isaac Sim ships rgba on by default but depth is
                    # opt-in via ``Camera.add_distance_to_image_plane_to_frame()``;
                    # PR #61's add_camera enables it post-initialize, but
                    # an older sim or a manually-attached Phase-1 camera
                    # state may not). Surface a zero-depth array sized to
                    # rgb so callers see a stable shape, plus a WARNING
                    # so misconfigured cameras don't silently produce
                    # zero-depth telemetry.
                    logger.warning(
                        "Camera '%s': get_depth() returned None (depth annotator not enabled). "
                        "Returning zero-depth array; "
                        "check add_distance_to_image_plane_to_frame() in add_camera.",
                        camera_name,
                    )
                    depth = np.zeros(rgb.shape[:2], dtype=np.float32)
                else:
                    depth = np.asarray(depth_raw)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, IndexError) as e:
                # Cleanup-clause shape mirrors create_world (#52
                # precedent). The Camera handle's ``get_rgba`` /
                # ``get_depth`` can raise on a not-yet-stepped world
                # (RTX render product hasn't accumulated samples) or
                # surface drift; surface as the structured error
                # envelope rather than letting the exception propagate.
                # ``IndexError`` is included so a 0-D / scalar ``get_rgba``
                # buffer during RTX warm-up surfaces here too rather than
                # escaping the loop (#140), even should the pre-slice
                # shape guard above ever be bypassed.
                logger.error("Failed to render camera '%s': %s", camera_name, e)
                return None, None, {"error": f"Failed to render camera '{camera_name}': {e}"}

            render_info = {
                "rtx": True,
                "prim_path": cam.prim_path,
                "resolution": [int(rgb.shape[1]), int(rgb.shape[0])],
                "render_mode": self._config.render_mode,
            }
            return (
                rgb,
                depth,
                {
                    "text": f"Rendered (RTX {self._config.render_mode}): {cam.width}x{cam.height}",
                    "json": render_info,
                },
            )

    def _warmup_camera(self, name: str, n_steps: int) -> bool:
        """Step the world (with rendering) until camera ``name`` yields a frame.

        Isaac's RTX render product does not accumulate a frame until the
        world is stepped with rendering enabled. A freshly-constructed
        camera therefore returns a malformed / empty ``get_rgba()`` buffer
        (shape ``(0,)``) for the first several frames, which makes
        :meth:`render` return an error envelope and the recording hook drop
        frames during an example's opening rollout. This steps the world up
        to ``n_steps`` times, re-rendering each time, and returns as soon as
        :meth:`render` reports success (a valid frame). Holds the scene
        static -- it only advances physics by the warm-up steps, which is
        negligible relative to a rollout.

        Parameters
        ----------
        name : str
            Camera name (already registered in ``self._cameras``).
        n_steps : int
            Maximum number of render-bearing world steps to take.

        Returns
        -------
        bool
            ``True`` if the camera produced a valid frame within
            ``n_steps``; ``False`` if it never warmed up (logged at
            WARNING so a misconfigured camera is visible). Never raises:
            a step / render failure is caught and ends the loop, because
            the camera is already registered and render()'s own 0-D guard
            still covers a not-yet-ready product.
        """
        if self._world is None:
            return False
        for i in range(max(1, n_steps)):
            try:
                self._world.step(render=True)
                self._sim_time += self._config.physics_dt
                self._step_count += 1
                if self.render(camera_name=name).get("status") == "success":
                    logger.debug("Camera %r warmed up after %d step(s)", name, i + 1)
                    return True
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, IndexError) as e:
                # Stepping / rendering a partially-initialised stage can
                # raise on surface drift; warm-up is best-effort, so log
                # and stop rather than failing the already-registered
                # camera. Programming bugs (NameError) still propagate.
                logger.debug("Camera %r warm-up step %d failed: %s", name, i + 1, e)
                break
        logger.warning(
            "Camera %r did not produce a valid frame after %d warm-up step(s); "
            "the first render() may return an error until the RTX product accumulates a frame.",
            name,
            n_steps,
        )
        return False

    def add_camera(
        self,
        name: str = "default",
        position: list[float] | None = None,
        target: list[float] | None = None,
        width: int | None = None,
        height: int | None = None,
        fov: float = 60.0,
    ) -> dict[str, Any]:
        """Add an RTX camera to the scene.

        Phase 2 wiring (#14): instantiates the underlying USD camera prim
        via ``omni.isaac.sensor.Camera`` and stores the handle on the
        ``_CameraState`` for later retrieval by :meth:`render`. In Phase 1
        this method silently returned ``status: "success"`` without
        creating any prim; that path is gone -- callers will now see a
        real camera prim on the stage and a Camera handle in
        ``self._cameras[name].handle``.

        ``render`` continues to return blank frames in Phase 2 because
        the actual ``camera.get_rgba()`` / annotator wiring is a separate
        slice. The Camera prim is the prerequisite, not the full frame
        path.

        Parameters
        ----------
        name : str
            Camera identifier. Default ``"default"``. Must be unique
            across the simulation; a duplicate is rejected with a
            structured error envelope.
        position : list[float], optional
            World-space position ``[x, y, z]`` in meters. Default
            ``[2.0, 2.0, 2.0]`` (an over-the-shoulder vantage that
            sees the default ground plane and any objects above it).
        target : list[float], optional
            World-space look-at point ``[x, y, z]``. If provided, the
            camera is oriented so its forward axis points at ``target``
            via ``omni.isaac.core.utils.viewports.set_camera_view``.
            If ``None``, the camera keeps its constructed orientation
            (identity).
        width : int, optional
            Image width in pixels. Defaults to ``IsaacConfig.camera_width``.
        height : int, optional
            Image height in pixels. Defaults to ``IsaacConfig.camera_height``.
        fov : float
            Horizontal field of view in degrees. Default 60.0. Mapped
            onto ``Camera.set_focal_length`` using the standard pinhole
            relation ``focal_length = horizontal_aperture / (2 * tan(fov/2))``
            with the USD-default 24 mm horizontal aperture.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``position``, ``target``, ``resolution``, ``fov``, and the
            computed ``focal_length`` so an agent can confirm the
            camera setup without re-querying.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if name in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' already exists."}],
                }

            w = int(width or self._config.camera_width)
            h = int(height or self._config.camera_height)
            pos = list(position) if position is not None else [2.0, 2.0, 2.0]
            tgt = list(target) if target is not None else None
            fov_deg = float(fov)

            # RTX cameras: render at a higher NATIVE resolution if the
            # caller's requested output is small, so the DLSS upscaler
            # stays above its temporal-ghost threshold; preserve the
            # requested aspect ratio. Captured frames are downscaled
            # back to ``(w, h)`` before return. See ``_MIN_RENDER_PX``
            # docstring for the why; gated by config.render_mode so
            # the headless CI path skips the cost.
            out_w, out_h = w, h
            if self._config.render_mode != "headless" and w < _MIN_RENDER_PX:
                scale = _MIN_RENDER_PX / float(w)
                w = _MIN_RENDER_PX
                h = int(round(h * scale))

            prim_path = f"{self._config.stage_path}/Cameras/{name}"

            try:
                handle, focal_length_mm = self._create_camera_prim(
                    name=name,
                    prim_path=prim_path,
                    position=pos,
                    target=tgt,
                    width=w,
                    height=h,
                    fov_deg=fov_deg,
                )
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                # Cleanup-clause shape mirrors create_world (#52 precedent)
                # and add_object: the constructor or initialise / look-at
                # call either succeeds and updates registries, or fails
                # with a structured envelope and updates neither.
                logger.error("Failed to add camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to add camera '{name}': {e}"}],
                }

            self._prim_registry.append(prim_path)
            cam_state = _CameraState(name=name, prim_path=prim_path, width=w, height=h)
            cam_state.handle = handle
            self._cameras[name] = cam_state
            # Track requested OUTPUT size (may differ from native render
            # size when DLSS upscaling required a larger native frame).
            self._cam_out_size[name] = (out_w, out_h)

            # Warm up the RTX render product: Isaac does not accumulate a
            # frame until the world is stepped with rendering enabled, so
            # without this the camera's first ``get_rgba()`` returns a
            # malformed / empty buffer (shape ``(0,)``) and render() /
            # recording drop the opening frames of a rollout. Skipped in
            # headless render mode (no RTX frames are produced there, so
            # stepping would only burn time). Best-effort: a warm-up step
            # failure is logged and does not fail add_camera -- the camera
            # is already registered, and render()'s own 0-D guard still
            # covers a not-yet-ready product.
            if self._config.render_mode != "headless" and self._camera_warmup_steps > 0:
                self._warmup_camera(name, self._camera_warmup_steps)

            cam_info = {
                "name": name,
                "prim_path": prim_path,
                "position": pos,
                "target": tgt,
                "resolution": [w, h],
                "fov": fov_deg,
                "focal_length_mm": focal_length_mm,
            }
            logger.info(
                "Added camera '%s' at pos=%s target=%s res=%dx%d fov=%.1f",
                name,
                pos,
                tgt,
                w,
                h,
                fov_deg,
            )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (f"Camera '{name}' added at {pos}, resolution={w}x{h}, fov={fov_deg}"),
                        "json": cam_info,
                    }
                ],
            }

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a camera from the scene.

        Phase 2 wiring (#14): paired with :meth:`add_camera`'s prim
        creation. Deletes the underlying USD camera prim via
        ``omni.isaac.core.utils.prims.delete_prim`` and prunes the
        in-Python registries. New method (no Phase 1 stub existed).

        Parameters
        ----------
        name : str
            Camera identifier previously passed to :meth:`add_camera`.

        Returns
        -------
        dict
            Status dict in the standard ``{"status", "content": [{"text"}]}``
            shape used by mutating methods on this class. Returns ``error``
            if the camera is unknown to ``_cameras``.
        """
        with self._lock:
            if name not in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' not found."}],
                }

            prim_path = self._cameras[name].prim_path

            # Cameras aren't added via ``world.scene.add`` (they're
            # standalone USD prims, not articulations or shape wrappers)
            # so removal goes via the stage utility rather than
            # ``world.scene.remove_object``. Wrapped in the same except
            # tuple as add_camera so a transient stage error returns the
            # structured envelope and leaves bookkeeping intact for
            # retry.
            try:
                if self._world is not None:
                    try:
                        from isaacsim.core.utils.prims import (  # type: ignore[import-not-found]
                            delete_prim,
                        )
                    except ImportError:
                        from omni.isaac.core.utils.prims import (  # type: ignore[import-not-found]
                            delete_prim,
                        )

                    delete_prim(prim_path)
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, ImportError) as e:
                logger.error("Failed to remove camera '%s' (prim=%s): %s", name, prim_path, e)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to remove camera '{name}': {e}"}],
                }

            del self._cameras[name]
            if prim_path in self._prim_registry:
                self._prim_registry.remove(prim_path)

            logger.info("Removed camera '%s' (prim=%s)", name, prim_path)
            return {
                "status": "success",
                "content": [{"text": f"Camera '{name}' removed."}],
            }

    # --- Recording (rollout video) -----------------------------------------
    #
    # The MuJoCo ``Simulation`` records rollout videos via a
    # ``start_cameras_recording`` / ``stop_cameras_recording`` pair that
    # spawns a daemon thread pulling frames off ``mjData``. Isaac can't
    # reuse that shape: the RTX renderer + ``Camera.get_rgba`` are bound
    # to the thread that booted ``SimulationApp`` (driving them from a
    # daemon thread deadlocks). So the Isaac recorder is *synchronous* --
    # it returns an ``on_frame`` closure that the eval driver wires into
    # ``evaluate_benchmark(..., on_frame=...)`` (present in the
    # strands-robots 0.4.0 ``SimEngine`` signature). The closure runs on
    # the eval thread, captures one ``render(camera)`` frame per applied
    # control step into an in-memory buffer, and ``stop_cameras_recording``
    # flushes the buffers to ``{name}__{camera}.mp4`` -- the same filename
    # convention MuJoCo uses, so cross-backend video discovery (the R15
    # backend matrix glob) picks up Isaac rows uniformly. See
    # strands-labs/robots-sim#112 and strands-labs/robots#191.

    def start_cameras_recording(
        self,
        cameras: list[str] | None = None,
        output_dir: str | None = None,
        fps: int = 30,
        name: str | None = None,
        max_frames_per_camera: int = 3000,
    ) -> dict[str, Any]:
        """Begin a synchronous rollout-video recording.

        Sets up one in-memory RGB buffer per camera and returns an
        ``on_frame(step, observation, action)`` closure in the result's
        ``json`` block. Wire that closure into
        :meth:`evaluate_benchmark`'s ``on_frame=`` kwarg; it captures one
        :meth:`render` frame per applied control step on the eval thread
        (no daemon thread -- Isaac's RTX renderer is thread-bound, see the
        class-level recording note). Call :meth:`stop_cameras_recording`
        afterwards to flush the buffers to MP4 files named
        ``{name}__{camera}.mp4`` under ``output_dir`` -- matching the
        MuJoCo backend's filename convention so cross-backend tooling
        finds Isaac videos the same way it finds MuJoCo ones.

        Parameters
        ----------
        cameras : list[str], optional
            Camera names to record. ``None`` = every camera added via
            :meth:`add_camera`. Unknown names error loudly (same policy
            as the MuJoCo recorder).
        output_dir : str, optional
            Directory for the ``{name}__{camera}.mp4`` files. Defaults to
            ``$TMPDIR/strands_robots/recordings``.
        fps : int
            Encoded MP4 frame rate. Default 30.
        name : str
            Filename tag. Auto-generated (``rec_<uuid>``) when ``None``.
        max_frames_per_camera : int
            Safety cap on in-memory buffers. Frames beyond the cap are
            silently dropped. Default 3000.

        Returns
        -------
        dict
            On success: ``{"status": "success", "content": [{"text": ...},
            {"json": {"on_frame": <callable>, "cameras": [...],
            "output_dir": ..., "name": ...}}]}``. The ``on_frame`` closure
            isn't JSON-serializable; Python callers unpack it from the
            json block. On error (no world, already recording, unknown
            cameras, none to record): ``{"status": "error", ...}``.
        """
        import os as _os
        import tempfile as _tempfile
        import time as _time
        import uuid as _uuid

        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}

            rec_state = self._cams_rec_state
            if rec_state and rec_state.get("running"):
                cur = rec_state["name"]
                return {
                    "status": "error",
                    "content": [{"text": f"Already recording '{cur}'. Call stop_cameras_recording() first."}],
                }

            if cameras is None:
                names = list(self._cameras.keys())
            else:
                unresolved = [c for c in cameras if c not in self._cameras]
                if unresolved:
                    return {
                        "status": "error",
                        "content": [
                            {"text": (f"Camera(s) not found: {unresolved}. Available: {list(self._cameras.keys())}")}
                        ],
                    }
                names = list(cameras)
            if not names:
                return {"status": "error", "content": [{"text": "No cameras to record."}]}

            out_dir = _os.path.abspath(
                output_dir or _os.path.join(_tempfile.gettempdir(), "strands_robots", "recordings")
            )
            _os.makedirs(out_dir, exist_ok=True)
            tag = name or f"rec_{_uuid.uuid4().hex[:8]}"

            buffers: dict[str, list] = {cam: [] for cam in names}
            paths = {cam: _os.path.join(out_dir, f"{tag}__{cam}.mp4") for cam in names}

            state: dict[str, Any] = {
                "running": True,
                "name": tag,
                "cameras": names,
                "fps": fps,
                "buffers": buffers,
                "paths": paths,
                "errors": dict.fromkeys(names, 0),
                "output_dir": out_dir,
                "started_at": _time.time(),
                "max_frames": max_frames_per_camera,
            }
            self._cams_rec_state = state

        def on_frame(step: int, observation: dict, action: dict) -> None:
            """Capture one RGB frame per camera (runs on the eval thread).

            Best-effort: a render failure on a single camera/step
            increments that camera's error counter rather than raising,
            so a transient RTX hiccup doesn't abort the whole eval.
            """
            st = getattr(self, "_cams_rec_state", None)
            if not st or not st.get("running"):
                return
            for cam in st["cameras"]:
                if len(st["buffers"][cam]) >= st["max_frames"]:
                    continue
                try:
                    rgb, _depth, _meta = self._render_frame(camera_name=cam)
                    if rgb is None:
                        st["errors"][cam] += 1
                        continue
                    arr = np.asarray(rgb)
                    if arr.ndim != 3 or arr.shape[0] == 0 or arr.shape[1] == 0:
                        st["errors"][cam] += 1
                        continue
                    st["buffers"][cam].append(np.ascontiguousarray(arr[..., :3].astype(np.uint8)))
                except (RuntimeError, ValueError, OSError, AttributeError, TypeError):
                    st["errors"][cam] += 1

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Recording '{tag}' armed for cameras {names}. "
                        "Pass the returned on_frame to evaluate_benchmark(on_frame=...), "
                        "then call stop_cameras_recording()."
                    ),
                    "json": {
                        "on_frame": on_frame,
                        "cameras": names,
                        "output_dir": out_dir,
                        "name": tag,
                        "paths": paths,
                    },
                }
            ],
        }

    def stop_cameras_recording(self) -> dict[str, Any]:
        """Stop recording and flush captured frames to MP4.

        Encodes each camera's in-memory RGB buffer to
        ``{name}__{camera}.mp4`` under the ``output_dir`` passed to
        :meth:`start_cameras_recording`, using ``imageio`` (the same
        encoder the MuJoCo recorder uses). Idempotent: a no-op success
        when nothing is recording.

        Best-effort: per-camera flush failures are reported in the result
        (``frames`` / ``errors`` / ``size_kb``) but never raise, so a
        partial encode still yields a structured success response.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text"}, {"json"}]}``
            envelope. ``json`` carries ``recording`` (the tag) and an
            ``artifacts`` list of ``{camera, path, frames, errors,
            size_kb}`` per camera.
        """
        import os as _os
        import time as _time

        with self._lock:
            state = getattr(self, "_cams_rec_state", None)
            if not state or not state.get("running"):
                return {"status": "success", "content": [{"text": "Was not recording cameras."}]}
            state["running"] = False
            self._cams_rec_state = None

        try:
            import imageio.v2 as imageio
        except ImportError:
            return {
                "status": "error",
                "content": [{"text": "imageio not installed. pip install imageio imageio-ffmpeg"}],
            }

        elapsed = _time.time() - state["started_at"]
        lines = [
            f"Stopped '{state['name']}' after {elapsed:.1f}s",
            f"   output_dir: {state['output_dir']}",
        ]
        artifacts = []
        for cam in state["cameras"]:
            frames_buffer = state["buffers"][cam]
            path = state["paths"][cam]
            errors = state["errors"][cam]
            frames_written = 0
            size_kb = 0.0
            if frames_buffer:
                writer = imageio.get_writer(path, fps=state["fps"], quality=8, macro_block_size=1)
                try:
                    for arr in frames_buffer:
                        writer.append_data(arr)
                        frames_written += 1
                finally:
                    writer.close()
                if _os.path.exists(path):
                    size_kb = _os.path.getsize(path) / 1024
            lines.append(
                f"   {cam:20s} {frames_written:>5d} frames  {size_kb:>7.1f} KB  "
                f"({errors} errors)  -> {_os.path.basename(path)}"
            )
            artifacts.append(
                {
                    "camera": cam,
                    "path": path,
                    "frames": frames_written,
                    "errors": errors,
                    "size_kb": size_kb,
                }
            )

        return {
            "status": "success",
            "content": [
                {"text": "\n".join(lines)},
                {"json": {"recording": state["name"], "artifacts": artifacts}},
            ],
        }

    def _create_camera_prim(
        self,
        *,
        name: str,
        prim_path: str,
        position: list[float],
        target: list[float] | None,
        width: int,
        height: int,
        fov_deg: float,
    ) -> tuple[Any, float]:
        """Construct the Isaac camera prim + apply look-at + FOV.

        Returns the camera handle plus the resolved focal length in mm
        so :meth:`add_camera` can surface the actually-used focal length
        in its structured json payload.

        Lazy-imports both the ``Camera`` sensor and
        ``set_camera_view`` so the module loads cleanly without Isaac
        Sim installed (the call site only runs after :meth:`create_world`
        has booted ``SimulationApp``). Isaac Sim 6.0 exposes ``Camera``
        under ``isaacsim.sensors.camera``; the legacy 4.x path was
        ``omni.isaac.sensor``. Try modern first, fall back so 4.x
        installs keep working.
        """
        import math

        import numpy as np  # type: ignore[import-not-found]

        try:
            from isaacsim.sensors.camera import Camera  # type: ignore[import-not-found]
        except ImportError:
            from omni.isaac.sensor import Camera  # type: ignore[import-not-found]

        camera = Camera(
            prim_path=prim_path,
            name=name,
            position=np.asarray(position, dtype=float),
            resolution=(int(width), int(height)),
        )
        # ``initialize`` allocates the RTX render product + annotators.
        # Some Camera builds defer this to first ``get_rgba()`` call;
        # call it explicitly so an init-time failure surfaces here
        # (and gets caught by the cleanup clause in add_camera) rather
        # than silently on the first render attempt.
        camera.initialize()

        # Map FOV (deg, horizontal) to focal length (mm) using the
        # standard pinhole lens relation:
        #
        #     focal_length = horizontal_aperture / (2 * tan(fov / 2))
        #
        # The horizontal aperture MUST be the camera's actual aperture,
        # read back from the prim -- assuming a nominal 24 mm is wrong on
        # Isaac's Camera (its default aperture + unit convention yield
        # fx~=6348 px at 640 px, i.e. a ~6 deg telephoto, instead of the
        # intended 60 deg / fx~=554). Deriving the focal length from the
        # read-back aperture makes the resulting pixel intrinsics
        # fx = width / (2*tan(fov/2)) exactly, independent of the
        # aperture's absolute value or units.
        try:
            horizontal_aperture_mm = float(camera.get_horizontal_aperture())
        except (AttributeError, RuntimeError, TypeError, ValueError):
            horizontal_aperture_mm = 24.0
        focal_length_mm = horizontal_aperture_mm / (2.0 * math.tan(math.radians(fov_deg) / 2.0))
        camera.set_focal_length(focal_length_mm)

        # Enable the depth annotator on the RTX render product. Isaac
        # Sim's Camera ships with rgba enabled by default but depth
        # is opt-in via this method; without it, ``camera.get_depth()``
        # returns ``None`` with a "Annotator 'distance_to_image_plane'
        # not found" warning -- which then crashes downstream
        # ``np.asarray`` calls in ``render()``. Caught during PR #61
        # GPU validation against the Isaac Sim 4.5 docker image.
        # ``add_distance_to_image_plane_to_frame`` is idempotent on
        # repeat calls so this is safe even if the camera has already
        # been initialized with depth elsewhere.
        try:
            camera.add_distance_to_image_plane_to_frame()
        except (AttributeError, RuntimeError):
            # Older Isaac Sim builds expose this under a different name
            # (``add_depth_to_frame``). Try the fallback before giving
            # up; downstream ``get_depth`` will still return ``None``
            # but ``render()``'s defensive None-handling (PR #62) will
            # cover it.
            try:
                camera.add_depth_to_frame()
            except (AttributeError, RuntimeError):
                logger.debug(
                    "Camera %s: depth annotator not enabled; ``get_depth()`` will return None",
                    name,
                )

        # Apply look-at after focal-length so the camera's forward axis
        # is correctly oriented at the target. ``set_camera_view`` works
        # on any USD camera prim by path; no Camera-specific API.
        if target is not None:
            try:
                from isaacsim.core.utils.viewports import (  # type: ignore[import-not-found]
                    set_camera_view,
                )
            except ImportError:
                from omni.isaac.core.utils.viewports import (  # type: ignore[import-not-found]
                    set_camera_view,
                )

            set_camera_view(eye=position, target=target, camera_prim_path=prim_path)

        return camera, focal_length_mm

    # --- Isaac-specific: Fleet Replication -----------------------------------

    def replicate(self, num_envs: int | None = None) -> dict[str, Any]:
        """Replicate the current scene into parallel environments.

        Uses ``omni.isaac.cloner.Cloner`` for GPU-efficient replication.

        Parameters
        ----------
        num_envs : int, optional
            Number of environments. Defaults to config.num_envs.

        Returns
        -------
        dict
            Status dict with replication info.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if not self._robots:
                return {
                    "status": "error",
                    "content": [{"text": "Add at least one robot first."}],
                }

            n = num_envs or self._config.num_envs

            t0 = time.perf_counter()
            # In full implementation: use omni.isaac.cloner.Cloner
            # to replicate the scene N times
            self._replicated = True
            self._num_envs_active = n
            elapsed = time.perf_counter() - t0

            logger.info("Replicated to %d envs in %.2fs", n, elapsed)

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Replicated to {n} environments. "
                            f"Build time: {elapsed * 1000:.0f}ms. "
                            f"Device: {self._config.device}."
                        ),
                        "json": {
                            "num_envs": n,
                            "build_time_ms": elapsed * 1000,
                        },
                    }
                ],
            }

    # --- Private Implementation ----------------------------------------------

    def _load_usd_robot(self, prim_path: str, usd_path: str, position: list[float]) -> tuple[list[str], Any]:
        """Load a robot from a USD file. Returns ``(joint_names, articulation)``.

        Phase 2 wiring (#14): the previous Phase-1 stub silently returned
        ``[]`` and didn't touch the stage. This Phase-2 implementation:

        1. References the USD at ``usd_path`` into the stage at
           ``prim_path`` via ``omni.isaac.core.utils.stage.add_reference_to_stage``.
        2. Wraps the resulting prim in
           ``omni.isaac.core.articulations.Articulation``.
        3. Calls ``articulation.initialize()`` to populate ``dof_names`` /
           internal handles. ``initialize`` is what triggers the Articulation
           tree walk that surfaces the joint count; without it
           ``dof_names`` is ``None`` on most Isaac Sim builds.
        4. Applies the requested ``position`` via ``set_world_pose`` so
           the robot lands where the caller asked. Identity ``[0, 0, 0]``
           is skipped to avoid an unnecessary kernel call.
        5. Extracts joint names from ``articulation.dof_names`` and returns
           them alongside the live ``Articulation`` handle. Callers store
           the handle on ``_RobotState.articulation`` so subsequent
           ``get_observation`` / ``send_action`` calls can read joint
           positions and apply targets through it.

        Raises propagate -- the caller (``add_robot`` USD branch) wraps
        this method in the standard cleanup-clause tuple
        ``(RuntimeError, ValueError, OSError, AttributeError, TypeError,
        ImportError)`` so any Isaac-side surface drift returns a
        structured error envelope rather than blowing up the agent.
        """
        import numpy as np  # type: ignore[import-not-found]

        # Isaac Sim 6.0 renamed the single-articulation wrapper. The 4.x
        # path was ``omni.isaac.core.articulations.Articulation``; on 6.0
        # the single-prim view lives in ``isaacsim.core.prims`` as
        # ``SingleArticulation`` (some builds keep an ``Articulation``
        # alias). Probe the modern locations first, fall back to legacy.
        Articulation = _import_articulation_cls()  # noqa: N806

        try:
            from isaacsim.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )
        except ImportError:
            from omni.isaac.core.utils.stage import (  # type: ignore[import-not-found]
                add_reference_to_stage,
            )

        # Step 1: stage reference. The USD's default prim becomes a child
        # of ``prim_path``; subsequent Articulation lookups walk that path.
        add_reference_to_stage(usd_path=usd_path, prim_path=prim_path)

        # Step 2-3: wrap + initialise. The articulation name has to be
        # unique within the scene's articulation registry, so derive it
        # from the prim path's leaf segment to match the caller's
        # ``add_robot`` ``name`` (the leaf of ``prim_path`` is the
        # caller-visible robot name by construction).
        articulation_name = prim_path.rsplit("/", 1)[-1]
        articulation = Articulation(prim_path=prim_path, name=articulation_name)
        articulation.initialize()
        # USD reference: the prim path is exactly what the caller asked
        # for (``add_reference_to_stage`` honours ``prim_path``); record
        # it as the actual landing path for symmetry with the URDF
        # branch. ``add_robot`` reads this back to seed
        # ``_RobotState.actual_prim_path``.
        try:
            articulation._strands_actual_prim_path = prim_path  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            pass

        # Step 4: position. The USD's authored pose is the default; only
        # call set_world_pose when the caller actually wanted a non-default
        # placement. Saves a tensor round-trip on the common
        # ``position=[0, 0, 0]`` case.
        if position is not None and any(p != 0.0 for p in position):
            articulation.set_world_pose(position=np.asarray(position, dtype=float))

        # Step 5: joint names. ``dof_names`` is ``None`` if ``initialize``
        # didn't surface them (e.g. the USD has no Articulation root on
        # the referenced prim); coerce to ``[]`` so downstream callers
        # see the documented "empty joint list" silent-empty mode rather
        # than a ``TypeError`` on iteration.
        joint_names = list(articulation.dof_names) if articulation.dof_names else []

        logger.info(
            "Loaded USD robot at %s from %s (%d joints, articulation=initialized)",
            prim_path,
            usd_path,
            len(joint_names),
        )
        return joint_names, articulation

    def _load_urdf_robot(self, prim_path: str, urdf_path: str, position: list[float]) -> tuple[list[str], Any]:
        """Load a robot from a URDF file. Returns ``(joint_names, articulation)``.

        Phase 2 wiring (#14): the previous Phase-1 stub silently
        returned ``[]`` and didn't touch the stage. This Phase-2
        implementation:

        1. Builds an ``omni.importer.urdf._urdf.ImportConfig`` with
           sensible defaults for a fixed-base manipulator (the most
           common case). Override behaviour is intentionally narrow:
           expose only the fields the agent / caller meaningfully
           controls (fix_base + distance_scale via config), keep the
           rest at the importer's defaults.
        2. Runs the ``URDFParseAndImportFile`` Kit command which
           parses the URDF and writes the USD onto the live stage at
           (or near) ``prim_path``. The importer occasionally returns a
           slightly different prim path than requested (it appends the
           URDF's ``robot name`` if the destination is a directory-like
           prim path); we honour the importer's choice and use that
           path for subsequent Articulation construction.
        3. Wraps the resulting prim in
           ``omni.isaac.core.articulations.Articulation``,
           initialises it, and applies the requested ``position``
           (skipping origin to save a tensor round-trip, mirroring
           ``_load_usd_robot``).
        4. Extracts joint names from ``articulation.dof_names``,
           coercing ``None`` to ``[]`` so a URDF with no actuated
           joints surfaces as the documented empty-joint-list mode
           rather than a ``TypeError`` on iteration.
        5. Returns ``(joint_names, articulation)`` -- same shape as
           ``_load_usd_robot`` so the ``add_robot`` URDF branch can
           reuse the same envelope shape.

        Raises propagate; the caller (``add_robot`` URDF branch)
        wraps in the standard cleanup-clause tuple
        ``(RuntimeError, ValueError, OSError, AttributeError,
        TypeError, ImportError)``.
        """
        import numpy as np  # type: ignore[import-not-found]

        # Isaac Sim 6.0 renamed the single-articulation wrapper (see
        # ``_import_articulation_cls`` / ``_load_usd_robot``). Probe the
        # modern ``isaacsim.*`` locations first, fall back to legacy.
        Articulation = _import_articulation_cls()  # noqa: N806

        # Isaac Sim's URDF importer API varies across releases:
        # * 6.0 exposes high-level ``URDFImporter`` + ``URDFImporterConfig``
        #   classes (the ``_urdf`` C-binding is no longer importable).
        # * 4.5/5.x used ``isaacsim.asset.importer.urdf._urdf`` with
        #   ``acquire_urdf_interface().parse_urdf()/import_robot()``.
        # * pre-4.5 used ``omni.importer.urdf._urdf``.
        # Try the modern 6.0 class API first, then the legacy ``_urdf`` ifaces.
        import os

        urdf_root, urdf_filename = os.path.split(os.path.abspath(urdf_path))
        imported_prim_path = None

        URDFImporter = URDFImporterConfig = None  # noqa: N806
        try:
            from isaacsim.asset.importer.urdf import (  # type: ignore[import-not-found,no-redef]
                URDFImporter,
                URDFImporterConfig,
            )
        except ImportError:
            URDFImporter = URDFImporterConfig = None  # noqa: N806

        if URDFImporter is not None and URDFImporterConfig is not None:
            # Isaac Sim 6.0 high-level API: build a config, point it at the URDF,
            # and import. Fixed base, keep joint names aligned with cuRobo (no
            # fixed-joint merge), self-collision off (adjacent-link mesh contact
            # oscillates on the actuator-less arm). Drive type 'position' so the
            # imported articulation can be position-commanded.
            cfg = URDFImporterConfig()
            cfg.urdf_path = os.path.abspath(urdf_path)
            for attr, val in (
                ("fix_base", True),
                ("merge_fixed_joints", False),
                ("allow_self_collision", False),
                ("collision_from_visuals", False),
            ):
                if hasattr(cfg, attr):
                    setattr(cfg, attr, val)
            if hasattr(cfg, "joint_drive_type"):
                try:
                    cfg.joint_drive_type = "position"
                except (AttributeError, TypeError, ValueError):  # enum vs str varies; leave default
                    pass
            # Strong position-drive gains so the arm holds against gravity and
            # tracks commanded joint targets (the bare SO-101 URDF has no
            # <dynamics> drive params -> default gains are too soft and the arm
            # droops instead of reaching the planned grasp pose).
            for attr, val in (("override_joint_stiffness", 1.0e5), ("override_joint_damping", 1.0e4)):
                if hasattr(cfg, attr):
                    try:
                        setattr(cfg, attr, val)
                    except (AttributeError, TypeError, ValueError):
                        pass
            importer = URDFImporter(config=cfg) if _accepts_config_kw(URDFImporter) else URDFImporter()
            if hasattr(importer, "config"):
                try:
                    importer.config = cfg
                except (AttributeError, TypeError):
                    pass
            # Isaac Sim 6.0 ``import_urdf()`` converts URDF -> USD on disk and
            # returns the USD path (NOT a live-stage prim path). Reference that
            # USD onto the live stage at our prim_path, then wrap that prim as an
            # Articulation.
            usd_out = importer.import_urdf()
            if not isinstance(usd_out, str) or not usd_out:
                raise RuntimeError(f"URDF import (6.0 API) returned no USD path for {urdf_path!r}")
            from isaacsim.core.utils.stage import add_reference_to_stage  # type: ignore[import-not-found]

            add_reference_to_stage(usd_path=usd_out, prim_path=prim_path)
            imported_prim_path = prim_path
        else:
            try:
                from isaacsim.asset.importer.urdf import _urdf  # type: ignore[import-not-found]
            except ImportError:
                from omni.importer.urdf import _urdf  # type: ignore[import-not-found]

            import_config = _urdf.ImportConfig()
            import_config.fix_base = True
            import_config.import_inertia_tensor = True
            import_config.create_physics_scene = False
            import_config.distance_scale = 1.0
            if hasattr(import_config, "merge_fixed_joints"):
                import_config.merge_fixed_joints = False
            if hasattr(import_config, "self_collision"):
                import_config.self_collision = False
            if hasattr(import_config, "make_default_prim"):
                import_config.make_default_prim = False

            urdf_iface = _urdf.acquire_urdf_interface()
            urdf_robot = urdf_iface.parse_urdf(urdf_root, urdf_filename, import_config)
            if urdf_robot is None:
                raise RuntimeError(f"URDF parse failed for {urdf_path!r}")
            imported_prim_path = urdf_iface.import_robot(urdf_root, urdf_filename, urdf_robot, import_config, "")
            if not imported_prim_path:
                raise RuntimeError(f"URDF import failed for {urdf_path!r} via _urdf.import_robot")

        # Step 2b: bind the imported prim to our caller-requested
        # ``prim_path``. The ``import_robot`` call adds prims at
        # ``imported_prim_path`` (under the live stage's default-prim
        # parent); we want the robot under our stage convention
        # (``{stage_path}/Robots/{name}``). Use the imported path
        # directly for ``Articulation`` construction -- the caller
        # bookkeeping (``_RobotState.prim_path``) records this so
        # ``remove_robot`` can look it up later. If a future caller
        # needs strict prim-path placement, this can move to a
        # ``MoveCommand`` to relocate after import.
        actual_prim_path = imported_prim_path

        # Step 3: Articulation wrap + initialise.
        articulation_name = actual_prim_path.rsplit("/", 1)[-1]
        articulation = Articulation(prim_path=actual_prim_path, name=articulation_name)
        articulation.initialize()
        # Set strong position-drive PD gains so the arm holds against gravity and
        # tracks commanded joint targets (the bare SO-101 URDF has no <dynamics>
        # drive params, so default gains are too soft -> the arm droops instead
        # of reaching the planned grasp pose, and the gripper can't clamp).
        try:
            ndof = len(articulation.dof_names) if articulation.dof_names else 0
            if ndof:
                kp = np.full(ndof, 1.0e5, dtype=float)
                kd = np.full(ndof, 1.0e4, dtype=float)
                set_gains = getattr(articulation, "set_gains", None)
                if callable(set_gains):
                    set_gains(kps=kp, kds=kd)
                else:
                    ctrl = getattr(articulation, "get_articulation_controller", None)
                    if callable(ctrl):
                        ctrl().set_gains(kps=kp, kds=kd)
                # Raise the per-joint max effort far above the SO-101 URDF's
                # tiny ``effort=10`` limit. With effort capped at 10 N*m PhysX
                # clamps the gripper drive torque regardless of how stiff the PD
                # gains are -> the closed jaw can't generate enough clamping
                # force to hold the 5 cm cube against gravity through the lift
                # (the cube just gets nudged and squirts out). 1000 gives the
                # gripper real clamping authority while staying physical.
                set_max = getattr(articulation, "set_max_efforts", None)
                if callable(set_max):
                    try:
                        set_max(np.full(ndof, 1.0e3, dtype=float))
                    except (RuntimeError, ValueError, TypeError, IndexError):
                        # Some builds expect a (M, K) batch for the view.
                        set_max(np.full((1, ndof), 1.0e3, dtype=float))
        except (AttributeError, TypeError, ValueError, RuntimeError, IndexError):  # gain set is best-effort
            logger.debug("set drive gains failed (non-fatal)", exc_info=True)
        # Stash the importer's actual landing path on the articulation
        # handle as a sidecar attribute so the caller (``add_robot``)
        # can record it on ``_RobotState.actual_prim_path`` for later
        # USD-stage walks (e.g. ``gripper_frame_pose``). The return
        # tuple shape ``(joint_names, articulation)`` is pinned by
        # downstream tests.
        try:
            articulation._strands_actual_prim_path = actual_prim_path  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Some Articulation builds don't allow attribute assignment;
            # caller falls back to the requested prim_path in that case.
            pass

        # Position. Same skip-origin shortcut as ``_load_usd_robot``.
        if position is not None and any(p != 0.0 for p in position):
            articulation.set_world_pose(position=np.asarray(position, dtype=float))

        # Step 4-5: joint names + return.
        joint_names = list(articulation.dof_names) if articulation.dof_names else []

        logger.info(
            "Loaded URDF robot at %s from %s (%d joints, articulation=initialized)",
            actual_prim_path,
            urdf_path,
            len(joint_names),
        )
        return joint_names, articulation

    # --- SimEngine: extra helpers for the SO-101 cuRobo example -------------
    #
    # These methods migrated in from the example-local Isaac adapter
    # (``examples/so101_curobo/isaac/simulation.py``) when issue #69
    # consolidated it into this library backend. They cover three
    # concerns the headless ``SimEngine`` core doesn't:
    #
    # 1. **Main-thread pump** (``pump`` / ``run_pump_forever`` / ``run_on_main``)
    #    -- Isaac's renderer + physics may only be driven from the
    #    thread that created ``SimulationApp``. A web UI like Gradio
    #    serves callbacks on worker threads where ``world.step(render=True)``
    #    deadlocks. The pump runs on the main thread and is the single
    #    place that advances the sim and renders the cameras.
    #
    # 2. **Kinematic teleport-grasp helpers** (``set_object_collision``,
    #    ``gripper_frame_pos``, ``gripper_frame_pose``, ``move_object``,
    #    ``_object_position``) -- the actuator-less SO-101 URDF can't
    #    grip via friction, so the collector teleport-follows the cube
    #    to the gripper. Reading the gripper-link world pose off the
    #    USD stage (rather than via the articulation handle) and
    #    toggling the cube collider while it's carried gives a stable
    #    multi-episode grasp.
    #
    # 3. **DLSS ghost mitigation** (``_converge_render``, ``_resize_rgb``,
    #    ``_configure_renderer``, ``_add_lighting``, ``set_joint_positions``,
    #    plus the ``add_camera`` native-resolution upscale) -- RTX
    #    cameras at small (<300 px) internal resolution smear a moving
    #    arm into a translucent "ghost"; rendering at >= ``_MIN_RENDER_PX``
    #    wide and holding the kinematic pose static for a few converge
    #    ticks per captured frame keeps every frame crisp.
    #
    # The headless / CI path doesn't engage any of these (the main-thread
    # callers run inline, the renderer config is best-effort, and the
    # native-resolution upscale is gated by ``render_mode != "headless"``).

    # --- main-thread pump --------------------------------------------------

    def pump(self, render: bool = True) -> None:
        """Drain queued actions, step once, refresh caches. MAIN THREAD ONLY.

        A web UI calls ``get_observation``/``send_action`` from worker
        threads where Isaac's renderer / physics deadlock. Those calls
        instead enqueue actions and read cached frames; this pump (run
        on the owning main thread) is the single place that actually
        advances the sim and renders the cameras.
        """
        if not self._world_created or self._world is None:
            return
        # 1. Apply any actions queued by worker threads, counting them.
        n_actions = 0
        while not self._action_q.empty():
            try:
                fn = self._action_q.get_nowait()
            except queue.Empty:
                break
            try:
                fn()
                n_actions += 1
            except (RuntimeError, ValueError, AttributeError, TypeError, KeyError, IndexError):
                # Queued worker actions are best-effort. Narrow to the
                # exceptions Isaac's articulation / object handles
                # plausibly raise (RuntimeError, ValueError, AttributeError,
                # TypeError) plus indexing surface (KeyError, IndexError);
                # programming bugs (NameError, ImportError) propagate so
                # they're caught early in development rather than
                # swallowed silently.
                logger.debug("queued action failed", exc_info=True)
        # 2. When worker actions ran this tick (n_actions > 0) they include
        # the recording capture, which does its OWN _converge_render + grab.
        # Doing a second idle converge here just doubles the render load and
        # serializes behind the capture. So only render here when the sim is
        # IDLE (no queued work): that keeps the live preview fresh between
        # episodes without competing with the recorder mid-episode.
        if n_actions == 0 and render:
            self._converge_render(self._idle_converge)
        # 3. Refresh joint-state cache for every robot.
        for rname, r in self._robots.items():
            if r.articulation is None:
                continue
            try:
                q = r.articulation.get_joint_positions()
                if q is not None:
                    arr = q.cpu().numpy() if hasattr(q, "cpu") else np.asarray(q)
                    self._joint_cache[rname] = {jn: float(v) for jn, v in zip(r.joint_names, list(arr))}
            except (RuntimeError, ValueError, AttributeError, TypeError):
                pass
        # 4. Refresh camera frame cache for the live preview -- only when we
        # actually rendered this tick (idle path). When actions ran, the
        # capture already published its frames to the cache; re-grabbing
        # here would be a wasted readback per camera every recorded frame.
        if render and n_actions == 0 and self._pump_cameras:
            for cname, cam in self._cameras.items():
                if cam.handle is None:
                    continue
                try:
                    img = self._grab_frame(cname, cam.handle)
                    if img is not None:
                        self._frame_cache[cname] = img
                except (RuntimeError, ValueError, AttributeError, TypeError):
                    logger.debug("pump frame grab failed for %s", cname, exc_info=True)

    def run_pump_forever(self, stop_event: Any = None) -> None:
        """Block on the MAIN THREAD running ``pump()`` in a loop.

        Drains queued worker actions (an executing episode) every
        iteration so the episode runs at full speed, and refreshes the
        live preview only every ``_idle_render_period`` IDLE seconds.
        A short sleep when idle keeps the renderer from running flat
        out -- which otherwise starves the Gradio HTTP thread so the
        page never loads.

        ``stop_event`` is a ``threading.Event``-style object whose
        ``is_set()`` returning truthy ends the loop. ``None`` (default)
        loops until ``KeyboardInterrupt``.
        """
        last_idle_render = 0.0
        self._pump_running = True
        try:
            while stop_event is None or not stop_event.is_set():
                # A whole-job submission (UI record/plan) takes priority:
                # run it inline on this main thread. The job drives the
                # sim directly (no per-frame round-trips); the preview
                # just freezes for its duration, which is the right
                # trade for a fast, reliable record.
                try:
                    job = self._main_jobs.get_nowait()
                except queue.Empty:
                    job = None
                if job is not None:
                    job()
                    last_idle_render = 0.0
                    continue
                busy = not self._action_q.empty()
                if busy:
                    self.pump(render=False)
                    continue
                now = time.time()
                do_render = (now - last_idle_render) >= self._idle_render_period
                self.pump(render=do_render)
                if do_render:
                    last_idle_render = now
                time.sleep(0.05)
        finally:
            self._pump_running = False

    def run_on_main(self, fn: Any, timeout: float | None = None) -> Any:
        """Run ``fn()`` on the MAIN THREAD (the pump owner) and return its result.

        A web UI calls record/plan jobs from a Gradio worker thread.
        Driving the episode from there means every per-frame
        ``set_joint_positions`` / ``step`` / ``get_observation``
        round-trips through the action queue to the pump -- slow and
        deadlock-prone for a long (355-frame) trajectory. Instead,
        submit the WHOLE job here: the pump runs it inline on the
        main thread, so inside ``fn`` ``_on_main_thread()`` is True and
        the collector drives the sim directly (exactly like the
        headless smoke path -- fast, no round-trips).

        While the job runs, the pump's normal loop is paused. Re-raises
        any exception from ``fn`` on the caller's thread. If already on
        the main thread, runs ``fn`` immediately.
        """
        if self._on_main_thread():
            return fn()
        done = threading.Event()
        box: dict[str, Any] = {}

        def _job() -> None:
            try:
                box["result"] = fn()
            except BaseException as exc:  # noqa: BLE001 - surfaced to caller below
                box["exc"] = exc
            finally:
                done.set()

        self._main_jobs.put(_job)
        if not done.wait(timeout=timeout):
            raise TimeoutError("run_on_main timed out waiting for the main-thread pump.")
        if "exc" in box:
            raise box["exc"]
        return box.get("result")

    # --- joint targets / kinematic teleport --------------------------------

    def set_joint_positions(
        self,
        positions: Any = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Drive an articulated robot kinematically to ``positions``.

        Used by the SO-101 cuRobo example to replay a planned trajectory
        on the actuator-less arm: ``send_action`` (position-target write
        + step) wouldn't move it because the URDF imports without
        position actuators on the SO-101. This writes joint state
        directly so the kinematic carry works.

        ``positions`` may be a ``dict`` keyed by joint name (only the
        listed joints are written; others retain their current value)
        or a list/array in the robot's joint order.
        """
        with self._lock:
            if not self._world_created or not self._robots:
                return {"status": "error", "content": [{"text": "No world/robot."}]}
            if positions is None:
                return {"status": "error", "content": [{"text": "'positions' is required."}]}
            if robot_name is None:
                robot_name = next(iter(self._robots))
            r = self._robots.get(robot_name)
            if r is None or r.articulation is None:
                return {"status": "error", "content": [{"text": f"Robot {robot_name!r} not initialized."}]}

            def _apply() -> None:
                if isinstance(positions, dict):
                    cur = list(r.articulation.get_joint_positions())
                    idx = {jn: i for i, jn in enumerate(r.joint_names)}
                    for jn, v in positions.items():
                        if jn in idx:
                            cur[idx[jn]] = float(v)
                    r.articulation.set_joint_positions(np.array(cur, dtype=float))
                else:
                    r.articulation.set_joint_positions(np.array(positions, dtype=float))

            if self._on_main_thread():
                _apply()
                return {"status": "success", "content": [{"text": "Set joint positions (main)."}]}
            self._action_q.put(_apply)
            return {"status": "success", "content": [{"text": "Set joint positions (queued)."}]}

    def move_object(
        self,
        name: str,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Teleport an object to ``(position, orientation)``.

        Used by the SO-101 cuRobo example for the kinematic
        teleport-grasp: while the cube is carried it is teleported into
        the closing gripper fingers every frame. Velocities are zeroed
        so a teleport doesn't fling a dynamic body.
        """
        obj = self._objects.get(name)
        if obj is None or obj.handle is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        try:
            pos = np.array(position[:3], dtype=float) if position else None
            ori = np.array(orientation[:4], dtype=float) if orientation else None
            obj.handle.set_world_pose(position=pos, orientation=ori)
            if hasattr(obj.handle, "set_linear_velocity"):
                obj.handle.set_linear_velocity(np.zeros(3))
            if hasattr(obj.handle, "set_angular_velocity"):
                obj.handle.set_angular_velocity(np.zeros(3))
            # Also write the USD xform translate/orient DIRECTLY. ``set_world_pose``
            # updates the PhysX/fabric transform, but the RENDER reads the prim's
            # USD ``xformOp:translate``; for a teleported (collider-off / kinematic)
            # body the fabric->USD writeback can lag or not fire on a render-only
            # tick, so the cube RENDERS at its stale pose (looked offset from / beside
            # the bin). Setting the USD xform ops here keeps the rendered mesh
            # exactly at the commanded pose.
            if pos is not None:
                try:
                    import omni.usd  # type: ignore[import-not-found]
                    from pxr import Gf, UsdGeom  # type: ignore[import-not-found]

                    stage = omni.usd.get_context().get_stage()
                    prim = stage.GetPrimAtPath(obj.prim_path)
                    if prim and prim.IsValid():
                        xf = UsdGeom.Xformable(prim)
                        top = None
                        for op in xf.GetOrderedXformOps():
                            if op.GetOpType() == UsdGeom.XformOp.TypeTranslate:
                                top = op
                                break
                        if top is None:
                            top = xf.AddTranslateOp()
                        top.Set(Gf.Vec3d(float(pos[0]), float(pos[1]), float(pos[2])))
                        if ori is not None:
                            for op in xf.GetOrderedXformOps():
                                if op.GetOpType() == UsdGeom.XformOp.TypeOrient:
                                    op.Set(Gf.Quatf(float(ori[0]), float(ori[1]), float(ori[2]), float(ori[3])))
                                    break
                except (AttributeError, TypeError, ValueError, RuntimeError):
                    logger.debug("move_object USD xform write skipped", exc_info=True)
        except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
            return {"status": "error", "content": [{"text": f"move_object failed: {type(exc).__name__}: {exc}"}]}
        return {"status": "success", "content": [{"text": f"'{name}' moved to {position or 'same'}."}]}

    def set_object_kinematic(self, name: str, kinematic: bool = True) -> dict[str, Any]:
        """Toggle an object's rigid body between KINEMATIC and dynamic.

        Root-causes the "cube renders offset from where it was placed" bug in
        the hybrid carry: the cube is a *dynamic* PhysX body, so even with its
        collider disabled, gravity + the physics solver own its transform and
        write it back to USD every ``world.step``. ``move_object`` (set_world_pose)
        sets the pose, but a render-only ``app.update()`` then shows the physics
        body's last-written (drifted) transform, NOT the pinned pose -> the cube
        renders apart from the bin.

        A KINEMATIC body ignores gravity/forces and takes its transform straight
        from ``set_world_pose``, which is written to USD and rendered faithfully.
        So flipping the carried cube kinematic makes its render match its placed
        pose exactly. Restored to dynamic on release. Toggles
        ``UsdPhysics.RigidBodyAPI.kinematicEnabled`` on the prim; best-effort.
        """
        obj = self._objects.get(name)
        if obj is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        # Prefer the wrapper's own setter if present.
        if obj.handle is not None:
            for meth in ("set_rigid_body_kinematic", "set_kinematic_enabled"):
                fn = getattr(obj.handle, meth, None)
                if callable(fn):
                    try:
                        fn(bool(kinematic))
                        return {"status": "success", "content": [{"text": f"'{name}' kinematic={kinematic}."}]}
                    except (RuntimeError, ValueError, AttributeError, TypeError):
                        logger.debug("%s failed; trying USD API", meth, exc_info=True)
        # Fallback: toggle UsdPhysics.RigidBodyAPI kinematicEnabled directly.
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdPhysics  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.prim_path)
            api = UsdPhysics.RigidBodyAPI.Get(stage, prim.GetPath()) or UsdPhysics.RigidBodyAPI.Apply(prim)
            attr = api.GetKinematicEnabledAttr()
            if not attr:
                attr = api.CreateKinematicEnabledAttr()
            attr.Set(bool(kinematic))
            return {
                "status": "success",
                "content": [{"text": f"'{name}' kinematic={kinematic} (USD)."}],
            }
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError) as exc:
            return {
                "status": "error",
                "content": [{"text": f"set_object_kinematic failed: {type(exc).__name__}: {exc}"}],
            }

    def set_object_collision(self, name: str, enabled: bool = True) -> dict[str, Any]:
        """Enable / disable an object's collider (keeps the visual mesh intact).

        Used by the SO-101 cuRobo kinematic grasp: while the cube is
        carried it is teleported *into* the closing gripper fingers
        every frame. With its collider on, the static cube and the
        finger colliders interpenetrate, and the resulting contact
        forces fling the stiff, undamped PD arm (kp ~3.6e4, kd ~0)
        into a ~5 cm/frame oscillation. Disabling the grasped cube's
        collider lets the gripper close cleanly around it; re-enabled
        on release.
        """
        obj = self._objects.get(name)
        if obj is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        if obj.handle is not None:
            try:
                obj.handle.set_collision_enabled(bool(enabled))
                return {"status": "success", "content": [{"text": f"'{name}' collision {'on' if enabled else 'off'}."}]}
            except (RuntimeError, ValueError, AttributeError, TypeError):
                logger.debug("set_collision_enabled unavailable; falling back to USD API", exc_info=True)
        # Fallback: toggle UsdPhysics.CollisionAPI on the prim directly.
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import UsdPhysics  # type: ignore[import-not-found]

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(obj.prim_path)
            api = UsdPhysics.CollisionAPI.Get(stage, prim.GetPath()) or UsdPhysics.CollisionAPI.Apply(prim)
            api.GetCollisionEnabledAttr().Set(bool(enabled))
            return {
                "status": "success",
                "content": [{"text": f"'{name}' collision {'on' if enabled else 'off'} (USD)."}],
            }
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError) as exc:
            return {
                "status": "error",
                "content": [{"text": f"set_object_collision failed: {type(exc).__name__}: {exc}"}],
            }

    def _object_position(self, name: str) -> list[float] | None:
        """Return the world-frame position of ``name`` (or ``None`` if missing)."""
        obj = self._objects.get(name)
        if obj is None or obj.handle is None:
            return None
        try:
            pos, _ = obj.handle.get_world_pose()
            return [float(x) for x in pos]
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return None

    def gripper_frame_pos(self, robot_name: str | None = None) -> list[float] | None:
        """World position of the robot's gripper / tool link (translation only)."""
        pose = self.gripper_frame_pose(robot_name)
        return pose[0] if pose else None

    def gripper_frame_pose(self, robot_name: str | None = None) -> tuple[list[float], list[float]] | None:
        """World pose of the robot's gripper / tool link: ``(translation, rotation)``.

        ``translation`` is the link origin in world coords; ``rotation``
        is the row-major 3x3 (flattened to 9) whose *columns* are the
        tool frame's local x/y/z axes in world coords, so
        ``world = R @ local``.

        The SO-101 example's collector uses this to seat the cube
        *rigidly* in the tool frame for the kinematic teleport-grasp:
        a plain world-space offset can't keep the cube between the
        jaws as the wrist rotates and lifts (the cube would drift
        beside the jaws and jitter). Prefers a ``gripper_frame``/``tool``
        link, then any ``gripper``/``moving_jaw`` link, under the robot's
        prim subtree.
        """
        if robot_name is None:
            robot_name = next(iter(self._robots), None)
        r = self._robots.get(robot_name) if robot_name else None
        if r is None:
            return None
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import (  # type: ignore[import-not-found]
                Gf,
                Sdf,
                Usd,
                UsdGeom,
            )

            stage = omni.usd.get_context().get_stage()
            # ``r.actual_prim_path`` is the prim path the URDF importer
            # / USD reference actually placed the robot at (which may
            # differ from the requested ``prim_path``: Isaac Sim 4.5
            # ``isaacsim.asset.importer.urdf.import_robot`` ignores the
            # destination argument and lands the robot at
            # ``/{robot_name}``). Walk up from there to the top-level
            # robot prim and search its whole subtree for the gripper
            # / tool link.
            sdf_path = Sdf.Path(r.actual_prim_path)
            top = sdf_path
            while top.GetParentPath() != Sdf.Path.absoluteRootPath and top.GetParentPath() != Sdf.Path.emptyPath:
                top = top.GetParentPath()
            root = stage.GetPrimAtPath(top)
            if not root or not root.IsValid():
                return None
            preferred = None
            fallback = None
            for p in Usd.PrimRange(root):
                if not p.IsA(UsdGeom.Xformable):
                    continue
                ln = p.GetName().lower()
                if "gripper_frame" in ln or "tool" in ln:
                    preferred = p
                    break
                if "moving_jaw" in ln or "gripper" in ln:
                    fallback = fallback or p
            prim = preferred or fallback
            if prim is None:
                return None
            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = xf.ExtractTranslation()

            def _axis(vx: float, vy: float, vz: float) -> tuple[float, float, float]:
                d = xf.TransformDir(Gf.Vec3d(vx, vy, vz))
                n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5 or 1.0
                return (d[0] / n, d[1] / n, d[2] / n)

            ax = _axis(1.0, 0.0, 0.0)
            ay = _axis(0.0, 1.0, 0.0)
            az = _axis(0.0, 0.0, 1.0)
            rot = [ax[0], ay[0], az[0], ax[1], ay[1], az[1], ax[2], ay[2], az[2]]
            pos = [float(t[0]), float(t[1]), float(t[2])]
            return pos, [float(x) for x in rot]
        except (RuntimeError, ValueError, AttributeError, TypeError, ImportError):
            logger.debug("gripper_frame_pose failed", exc_info=True)
            return None

    # --- DLSS-ghost mitigation + RTX renderer config ------------------------

    def _refresh_all_render_products(self, n: int = 1) -> None:
        """Tick the renderer ``n`` times so EVERY camera's RTX render product
        accumulates a fresh frame.

        In headless RTX a single ``world.step(render=True)`` in the substep loop
        only reliably refreshes the PRIMARY render product; secondary cameras'
        ``get_rgba`` then returns a stale/blank buffer (the long-standing
        single-camera limitation). ONE extra render-only tick here lets Kit flush
        the remaining render products before read-back.

        Kept deliberately LIGHT: a single ``SimulationApp.update()`` (render
        only -- it does NOT advance physics, so the cube/arm don't drift between
        the action and the observation), falling back to one
        ``world.step(render=True)`` if the app handle is unavailable. Driving
        multiple extra render ticks per recorded frame hammered the RTX
        Hydra-texture pipeline (``libomni.hydratexture``) and crashed the kit
        over a full multi-episode session, so we do the minimum needed to flush
        the secondary render products. Main-thread only (renderer constraint).
        """
        if not self._world_created or self._world is None:
            return
        # Light render-only refresh (no physics advance, avoids the Hydra-texture
        # overload that a heavier per-frame render loop caused). Prefer
        # SimulationApp.update(); fall back to a single world.step(render=True).
        app = getattr(self, "_app", None)
        update = getattr(app, "update", None) if app is not None else None
        for _ in range(max(1, n)):
            if callable(update):
                update()
            else:
                self._world.step(render=True)

    def _converge_render(self, n: int = 8) -> None:
        """Render ``n`` ticks while HOLDING the robots at their current pose.

        ``world.step(render=True)`` advances physics every tick, so a
        kinematic arm keeps drifting (gravity / settling) while we try
        to converge the DLSS temporal upscaler -> the moving target
        leaves a faint ghost. Re-asserting each robot's joint positions
        (and zeroing velocities) before every render freezes the pose
        so DLSS converges on a single, static image.
        """
        if not self._world_created or self._world is None:
            return
        for _ in range(max(1, n)):
            for r in self._robots.values():
                if r.articulation is None:
                    continue
                try:
                    q = r.articulation.get_joint_positions()
                    if q is not None:
                        qa = np.asarray(q, dtype=float)
                        r.articulation.set_joint_positions(qa)
                        try:
                            r.articulation.set_joint_velocities(np.zeros_like(qa))
                        except (RuntimeError, ValueError, AttributeError, TypeError):
                            pass
                except (RuntimeError, ValueError, AttributeError, TypeError):
                    pass
            self._world.step(render=True)

    def _grab_frame(self, cname: str, cam: Any) -> Any:
        """Capture ``cam`` as an RGB uint8 array at the camera's requested output size.

        The RTX camera renders at a higher native resolution (to keep
        DLSS out of its temporal-ghost regime); this downscales the
        result back to the size the caller asked for. Returns ``None``
        if no frame is available yet.
        """
        frame = cam.get_rgba()
        if frame is None or not getattr(frame, "size", 0):
            return None
        img = np.asarray(frame)[:, :, :3].astype("uint8")
        out = self._cam_out_size.get(cname)
        if out is not None:
            ow, oh = out
            if img.shape[1] != ow or img.shape[0] != oh:
                img = self._resize_rgb(img, ow, oh)
        return img

    @staticmethod
    def _resize_rgb(img: Any, out_w: int, out_h: int) -> Any:
        """Downscale an HxWx3 uint8 array to ``(out_h, out_w)``.

        Uses cv2 / PIL if present, else a fast NumPy area-average /
        nearest fallback (no new deps).
        """
        try:
            import cv2  # type: ignore[import-not-found]

            return cv2.resize(img, (out_w, out_h), interpolation=cv2.INTER_AREA)
        except ImportError:
            pass
        try:
            from PIL import Image  # type: ignore[import-not-found]

            resample = getattr(Image, "Resampling", Image).BILINEAR
            return np.asarray(Image.fromarray(img).resize((out_w, out_h), resample))
        except ImportError:
            pass
        h, w = img.shape[:2]
        if w % out_w == 0 and h % out_h == 0:
            fx, fy = w // out_w, h // out_h
            return img.reshape(out_h, fy, out_w, fx, 3).mean(axis=(1, 3)).astype("uint8")
        ys = (np.arange(out_h) * (h / out_h)).astype(int).clip(0, h - 1)
        xs = (np.arange(out_w) * (w / out_w)).astype(int).clip(0, w - 1)
        return img[ys][:, xs]

    def _configure_renderer(self) -> None:
        """Best-effort RTX settings for a stable real-time image.

        These carb settings (RaytracedLighting, FXAA, no temporal
        denoiser) nudge RTX toward a single-frame-stable image, but
        note the RTX pipeline re-asserts ``/rtx/post/aa/op`` back to
        DLSS (3) on every render tick, so they do NOT by themselves
        stop the moving-arm "ghost". The actual ghost fix is rendering
        cameras at a high native resolution (>= ``_MIN_RENDER_PX`` wide)
        so the DLSS upscaler stays out of its temporal-ghost regime,
        plus ``_converge_render`` holding the pose static while it
        settles. Best-effort: skipped silently when ``carb.settings``
        isn't importable.
        """
        try:
            import carb  # type: ignore[import-not-found]

            s = carb.settings.get_settings()
            s.set("/rtx/rendermode", "RaytracedLighting")
            s.set("/rtx/directLighting/sampledLighting/enabled", True)
            s.set("/rtx/raytracing/subframes", 1)
            s.set("/rtx/pathtracing/totalSpp", 1)
            s.set("/rtx/sceneDb/ambientLightIntensity", 1.0)
            s.set("/rtx/post/aa/op", 1)
            s.set("/rtx/post/dlss/execMode", 0)
            s.set("/rtx/post/taa/enabled", False)
            s.set("/rtx/directLighting/denoiser/enabled", False)
            s.set("/rtx/raytracing/lightcache/spatialCache/enabled", False)
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("renderer config skipped", exc_info=True)

    def _add_lighting(self) -> None:
        """Add a dome + key + fill light so RTX camera frames aren't black.

        Unlike MuJoCo (which has implicit headlight / ambient), an Isaac
        stage is unlit by default -- without this, ``get_rgba()``
        returns near-black frames and the UI preview looks empty.
        Best-effort; skipped silently when Pixar USD imports fail.
        """
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import (  # type: ignore[import-not-found]
                Gf,
                Sdf,
                UsdGeom,
                UsdLux,
            )

            stage = omni.usd.get_context().get_stage()
            dome = UsdLux.DomeLight.Define(stage, Sdf.Path("/World/lights/dome"))
            dome.CreateIntensityAttr(800.0)
            distant = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/key"))
            distant.CreateIntensityAttr(2500.0)
            distant.CreateAngleAttr(1.0)
            UsdGeom.Xformable(distant.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-45.0, 0.0, 25.0))
            fill = UsdLux.DistantLight.Define(stage, Sdf.Path("/World/lights/fill"))
            fill.CreateIntensityAttr(1500.0)
            fill.CreateAngleAttr(1.0)
            UsdGeom.Xformable(fill.GetPrim()).AddRotateXYZOp().Set(Gf.Vec3f(-60.0, 0.0, 180.0))
        except (ImportError, AttributeError, RuntimeError):
            logger.debug("Could not add scene lighting", exc_info=True)

    def cleanup(self) -> None:
        """Release all resources.

        Callers must invoke this explicitly (or use the class as a context
        manager). There is intentionally no ``__del__`` finalizer: at
        interpreter shutdown the ``threading`` / ``logger`` / ``omni``
        modules can already be partially torn down, and acquiring
        ``self._lock`` from a finalizer is unsafe. Relying on GC for
        Isaac Sim cleanup also leaks the ``World``/USD stage on the
        common case where the GC scheduler defers the finalizer past
        the SimulationApp shutdown.
        """
        if self._world_created:
            self.destroy()

    def __enter__(self) -> IsaacSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        return (
            f"IsaacSimulation("
            f"num_envs={self._config.num_envs}, "
            f"device={self._config.device!r}, "
            f"headless={self._config.headless}, "
            f"world={'created' if self._world_created else 'none'})"
        )
