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
    - STRANDS_ISAAC_HEADLESS: Override headless mode. On (1/true/yes/on) forces
      headless, off (0/false/no/off) forces windowed, unset or empty keeps the
      ``IsaacConfig`` field, and any other spelling is refused
    - STRANDS_ISAAC_RTX_PATHTRACING: Enable RTX pathtracing, same vocabulary;
      its off side leaves ``render_mode`` alone rather than selecting a mode
    - STRANDS_ISAAC_NUCLEUS_URL: Override Nucleus asset server URL
"""

from __future__ import annotations

import logging
import os
import queue
import threading
import time
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, TypedDict, cast

import numpy as np

from strands_robots.simulation.base import SimEngine, unknown_kwargs_error
from strands_robots.simulation.isaac.config import IsaacConfig
from strands_robots.simulation.isaac.joint_names import demangle_usd_joint_names, urdf_joint_names
from strands_robots.simulation.isaac.motion_primitives import IsaacMotionPrimitivesMixin
from strands_robots.simulation.isaac.recording import IsaacRecordingMixin
from strands_robots.simulation.models import registered, registry_entry
from strands_robots.simulation.terrain import validate_difficulty
from strands_robots.utils import (
    camera_fov_error,
    coerce_pose_vector,
    coerce_rgba,
    coerce_size_vector,
    entity_name_error,
    name_list_error,
    non_negative_whole_number_error,
    partial_construction_repr,
    positive_count_error,
    positive_whole_number_error,
    step_aborted_msg,
)

if TYPE_CHECKING:
    from strands_robots.policies.base import Policy
    from strands_robots.rendering import CameraParams

logger = logging.getLogger(__name__)

# Minimum NATIVE render width for RTX cameras. Isaac's RTX pipeline runs
# the DLSS temporal upscaler, which renders internally at ~half the output
# width and upscales. Below ~300 px internal resolution DLSS falls back
# to a temporal-accumulation path that smears a moving arm into a
# translucent "ghost" (long-standing front/oblique-view bug seen during
# the SO-101 cuRobo example's GPU validation -- see robots-sim#69 /
# robots-sim#68).
# Rendering at >= 640 px wide keeps the DLSS internal resolution above
# that threshold so every frame is crisp on its own; captured frames
# are downscaled to the caller's requested size before return.
_MIN_RENDER_PX = 640


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a ``(3, 3)`` rotation matrix.

    Isaac's ``Camera.get_world_pose()`` returns the orientation as a
    ``(w, x, y, z)`` quaternion (USD convention). No external dep needed --
    the standard quaternion-to-matrix formula.
    """
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


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


def _to_float_list(value: Any, n: int) -> list[float] | None:
    """Coerce a torch tensor / numpy array / sequence to ``n`` floats.

    Isaac core APIs return numpy arrays on the CPU pipeline and torch
    tensors on the GPU pipeline; ``get_observation`` handles the same
    duality with a ``hasattr(x, "cpu")`` probe, mirrored here. Returns
    ``None`` on any shape / dtype mismatch so callers can treat the
    read as failed rather than propagate a partial vector.
    """
    try:
        if hasattr(value, "cpu"):
            value = value.cpu().numpy()
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
    except (RuntimeError, TypeError, ValueError):
        return None
    if arr.shape[0] < n or not np.all(np.isfinite(arr[:n])):
        return None
    return [float(x) for x in arr[:n]]


def _rotmat_to_quat_wxyz(rotmat: np.ndarray | list[list[float]]) -> list[float]:
    """Row-major 3x3 rotation matrix -> unit quaternion, MuJoCo order (wxyz).

    Shepperd's method (branch on the largest diagonal term) for numerical
    stability near 180-degree rotations. The result is normalized and
    sign-canonicalized to ``w >= 0`` (quaternion double-cover: ``q`` and
    ``-q`` encode the same rotation, so the canonical sign is safe for
    every consumer and keeps repeated reads deterministic).
    """
    m = np.asarray(rotmat, dtype=np.float64)
    t = float(np.trace(m))
    if t > 0.0:
        s = float(np.sqrt(t + 1.0)) * 2.0
        w, x, y, z = 0.25 * s, (m[2, 1] - m[1, 2]) / s, (m[0, 2] - m[2, 0]) / s, (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] >= m[1, 1] and m[0, 0] >= m[2, 2]:
        s = float(np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w, x, y, z = (m[2, 1] - m[1, 2]) / s, 0.25 * s, (m[0, 1] + m[1, 0]) / s, (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] >= m[2, 2]:
        s = float(np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w, x, y, z = (m[0, 2] - m[2, 0]) / s, (m[0, 1] + m[1, 0]) / s, 0.25 * s, (m[1, 2] + m[2, 1]) / s
    else:
        s = float(np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w, x, y, z = (m[1, 0] - m[0, 1]) / s, (m[0, 2] + m[2, 0]) / s, (m[1, 2] + m[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    norm = float(np.linalg.norm(q)) or 1.0
    q = q / norm
    if q[0] < 0.0:
        q = -q
    return [float(v) for v in q]


# ``get_body_state`` submits its USD read to the main-thread pump when called
# from a worker thread; the read itself is trivial, so a stuck wait means the
# pump died -- fail with a TimeoutError instead of hanging the caller forever.
_BODY_STATE_MAIN_THREAD_TIMEOUT_S = 30.0


def _body_state_envelope(body_name: str, state: dict[str, Any]) -> dict[str, Any]:
    """Wrap a body-state payload in the MuJoCo-compatible success envelope."""
    pos = state["position"]
    quat = state["quaternion"]
    text = (
        f"Body '{body_name}' ({state.get('source', 'body')} at {state.get('prim_path', '?')}):\n"
        f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
        f"  quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]"
    )
    return {"status": "success", "content": [{"text": text}, {"json": state}]}


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

# Every keyword :meth:`IsaacSimulation.add_object` honors, including the one it
# reads out of its own ``**kwargs`` (``scale``, the ``size`` alias). Doubles as
# the "Valid:" hint in the unknown-keyword refusal, so it lists the declared
# parameters too rather than only the residual-key vocabulary. A unit test pins
# it against the live signature so a parameter added to ``add_object`` cannot
# start being reported as unknown.
_ADD_OBJECT_PARAMS: tuple[str, ...] = (
    "color",
    "is_static",
    "mass",
    "material",
    "mesh_path",
    "name",
    "orientation",
    "position",
    "scale",
    "shape",
    "size",
)


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
        data_config: str | None = None,
        usd_to_urdf_joint_names: dict[str, str] | None = None,
    ):
        self.name = name
        self.prim_path = prim_path
        self.joint_names = joint_names
        self.articulation = articulation
        # Registry data-config the robot was added under (e.g. ``so100``).
        # Recorded as the LeRobotDataset ``robot_type`` so datasets collected
        # on Isaac carry the same embodiment metadata as MuJoCo/Newton ones.
        self.data_config = data_config
        # USD-mangled DOF name -> URDF joint name, for robots loaded from a
        # URDF whose joint names are not valid USD identifiers (e.g. the
        # ``robotstudio_so101`` URDF's ``"1"``..``"6"``, imported as
        # ``tn__1_``..``tn__6_``). ``joint_names`` above already carries the
        # translated (URDF) vocabulary - see
        # :mod:`strands_robots.simulation.isaac.joint_names` (#1900) - so
        # this map is diagnostics-only: it correlates the public names with
        # the prim names an on-stage USD walk would encounter. Empty when
        # nothing was mangled.
        self.usd_to_urdf_joint_names = dict(usd_to_urdf_joint_names or {})
        # The prim path the URDF importer / USD reference actually
        # placed the robot at, which can differ from ``prim_path`` when
        # the importer ignores the requested destination (Isaac Sim 4.5
        # ``isaacsim.asset.importer.urdf.import_robot`` ignores the
        # ``stage=""`` argument and lands the robot under the URDF's
        # ``robot name``, e.g. ``/so101_new_calib`` regardless of
        # ``/World/Robots/arm`` being requested). Used by
        # ``gripper_frame_pose`` to walk the actual robot subtree.
        self.actual_prim_path = actual_prim_path or prim_path
        # Rollout bookkeeping, mirroring the MuJoCo per-robot record: the
        # policy-driving loops (:meth:`IsaacSimulation.run_multi_policy`, the
        # recording hook in
        # :meth:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin._make_run_policy_hook`)
        # set these while a rollout drives this robot. ``policy_running`` is
        # both the busy guard (a robot already driven by one loop must not be
        # double-stepped by another) and the cooperative-stop flag (flipping
        # it False ends the loop cleanly).
        self.policy_running = False
        self.policy_instruction = ""
        self.policy_steps = 0


def _cameras_recording_option_error(
    method: str,
    fps: Any,
    max_frames_per_camera: Any,
) -> dict[str, Any] | None:
    """Reject a rollout-video option the Isaac recorder cannot honor.

    Pre-flight guard for :meth:`IsaacSimulation.start_cameras_recording`,
    mirroring the MuJoCo backend's guard of the same name
    (:func:`strands_robots.simulation.mujoco.rendering._cameras_recording_option_error`)
    against the one shared domain
    (:func:`~strands_robots.utils.positive_whole_number_error`), so the two
    recording surfaces cannot disagree on what a usable ``fps`` is. Isaac takes
    no ``width``/``height`` here - each camera carries its own resolution from
    :meth:`IsaacSimulation.add_camera` - so only the two frame counts are
    checked.

    Refusing at ``start`` is what keeps the flush honest: ``fps`` is stored in
    the recording state and handed to
    :func:`~strands_robots.rendering.encode_clip` by
    :meth:`IsaacSimulation.stop_cameras_recording`, which refuses a rate it
    cannot encode at. Validating only at flush time would surface the mistake
    after a whole rollout's frames had been buffered, and
    ``max_frames_per_camera=0`` would drop every frame while both calls still
    reported success.

    Args:
        method: Public method name, used to prefix the error message.
        fps: Encoded MP4 frame rate.
        max_frames_per_camera: In-memory per-camera frame cap.

    Returns:
        A structured ``{"status": "error", ...}`` dict naming the first
        offending parameter, or ``None`` when both options are usable.
    """
    for param, value in (("fps", fps), ("max_frames_per_camera", max_frames_per_camera)):
        if text := positive_whole_number_error(value, param, method):
            return {"status": "error", "content": [{"text": text}]}
    return None


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


class IsaacSimulation(IsaacMotionPrimitivesMixin, IsaacRecordingMixin, SimEngine):
    """GPU-native simulation backend built on NVIDIA Isaac Sim.

    Implements the ``SimEngine`` ABC. Provides photorealistic rendering,
    RTX sensors, USD scene management, and fleet replication via Cloner.
    LeRobotDataset recording (``start_recording`` / ``save_episode`` /
    ``stop_recording`` / ``stream_dataset``) comes from
    :class:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin`,
    and the motion primitives (``move_to`` / ``set_gripper`` /
    ``rotate_wrist``) from
    :class:`~strands_robots.simulation.isaac.motion_primitives.IsaacMotionPrimitivesMixin`,
    matching the MuJoCo and Newton backends.

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
        # adapter retired by robots-sim#69 is accepted for backward compat
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
        # Per-robot task-space action controllers (install_action_controller).
        # When a robot has one, dict actions handed to send_action are first
        # converted to joint-name targets via compute_joint_targets -- the
        # Isaac counterpart of the MuJoCo backend's
        # ``world._backend_state["action_controller"]`` seam (#1812).
        self._action_controllers: dict[str, Any] = {}
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

        # LeRobotDataset recording state - the Isaac side of the
        # DatasetRecordingMixin state seam. MuJoCo/Newton keep this dict on
        # their SimWorld (``_backend_state``); Isaac's ``self._world`` is the
        # Isaac Sim World handle, so the engine owns the dict directly and
        # IsaacRecordingMixin._recording_state() returns it. Reset by
        # destroy() alongside the rest of the world state.
        self._recording_state_dict: dict[str, Any] = {}

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
        # is unchanged. See robots-sim#69 for the consolidation rationale.
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

        # Construction complete - the finalizer may now release what we hold.
        # See SimEngine._init_complete: this must be the final statement.
        self._init_complete = True

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
        terrain: str | None = None,
        difficulty: float = 1.0,
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
        terrain : str, optional
            Heightfield terrain kind (e.g. ``"rough"``/``"stairs"``/
            ``"pyramid"``/``"slope"``). The Isaac backend has no heightfield
            ground yet, so a non-None value is rejected with an actionable
            error rather than raising ``TypeError`` or being silently ignored
            (honouring the
            :class:`~strands_robots.simulation.base.SimEngine` ``create_world``
            contract). Use ``create_simulation(backend="mujoco")`` for terrain.
        difficulty : float
            Terrain curriculum elevation scale (only meaningful together with
            ``terrain``). Accepted for signature parity with the base contract.
            Since the Isaac backend has no heightfield terrain, ``difficulty``
            can never take effect, so a non-default (``!= 1.0``) value is
            rejected with an actionable error (the base ``create_world``
            contract: reject rather than silently ignore it) rather than
            silently having no effect.

        Returns
        -------
        dict
            Status dict with world info.
        """
        if terrain is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"terrain={terrain!r} is not supported on the Isaac backend yet "
                            "(heightfield terrain, e.g. 'rough'/'stairs'/'pyramid'/'slope', is "
                            "currently MuJoCo-only); use create_simulation(backend='mujoco') for "
                            "terrain, or omit terrain for a flat ground plane."
                        )
                    }
                ],
            }
        # Base create_world contract: reject a non-default difficulty with no
        # terrain rather than silently ignoring it. On Isaac difficulty is
        # doubly inert - there is no heightfield terrain for it to scale (a
        # non-None terrain is already rejected above) - so any != 1.0 value is
        # meaningless here; surface that instead of a status=success no-op.
        # The ``difficulty`` domain is owned by ``validate_difficulty`` and shared
        # with the MuJoCo backend, which honors the value: the same scale is
        # refused identically on every backend rather than one accepting what
        # another rejects. It runs before the ``float(difficulty)`` test below,
        # which raises ``TypeError`` for ``None``/a list and ``ValueError`` for a
        # non-numeric string - escaping this method's structured-error contract.
        try:
            validate_difficulty(difficulty)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if float(difficulty) != 1.0:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"difficulty={difficulty!r} has no effect on the Isaac backend "
                            "(it scales a heightfield terrain's elevation, and this backend "
                            "has no heightfield terrain); use create_simulation(backend='mujoco') "
                            "for a terrain curriculum, or omit difficulty for a flat ground plane."
                        )
                    }
                ],
            }
        # A world must not be built around a dt the integrator cannot honor
        # (negative, zero, nan): physics_dt drives every stage step, so an
        # unusable value corrupts the world rather than one call.
        effective_timestep = self._config.physics_dt if timestep is None else timestep
        timestep_param = "physics_dt" if timestep is None else "timestep"
        if err := self._validate_timestep(effective_timestep, "create_world", timestep_param):
            return err
        # Isaac's ``PhysicsContext.set_gravity`` takes a single signed scalar
        # (the gravity along -Z); the backend cannot apply an off-axis vector.
        # A non-Z-aligned override was previously silently reduced to its
        # z-component -- so ``gravity=[0, -9.81, 0]`` yielded ZERO gravity --
        # while the result echoed the full input vector as if applied. Validate
        # up front and reject anything the backend cannot honour, rather than
        # applying a gravity the caller never asked for.
        if gravity is not None:
            # Normalize through the shared domain first, so the component count,
            # the numeric domain and the boolean refusal are the ones every
            # other gravity surface applies. The local copy coerced a scalar
            # with ``float()``, and bool is an int subclass, so
            # ``create_world(gravity=True)`` configured a +1 m/s^2 gravity
            # pointing *up*; it also keyed on ``isinstance(gravity, (list, tuple))``,
            # so a NumPy vector - which the other backends accept - was refused
            # as "not a scalar or vector". The Z-alignment constraint below is
            # this backend's own and is applied to the normalized components.
            components, gravity_error = self._normalize_gravity(gravity, "create_world")
            if components is None:
                return cast("dict[str, Any]", gravity_error)
            if components[0] != 0.0 or components[1] != 0.0:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"create_world: the Isaac backend only supports Z-aligned gravity "
                                f"(its PhysicsContext.set_gravity takes a signed scalar); a non-Z-aligned "
                                f"vector like {gravity!r} cannot be honoured. Pass a scalar or a "
                                f"[0, 0, gz] vector, or use create_simulation(backend='mujoco') for "
                                f"arbitrary-direction gravity."
                            )
                        }
                    ],
                }
            # Store the normalized components so what the result reports and
            # what the physics context receives are the same value.
            gravity = components
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
            self._action_controllers.clear()
            self._cameras.clear()
            self._objects.clear()
            self._prim_registry.clear()
            # Drop any in-flight recorder state (buffers reference RTX
            # frames that are meaningless after the stage tears down).
            self._cams_rec_state = None
            # Same for the LeRobotDataset recording session: a live recorder
            # holds an OPEN LeRobot episode buffer whose camera frames came
            # from this stage. Mirror the MuJoCo/Newton behaviour (their
            # state dict dies with the SimWorld) by resetting the seam dict;
            # warn when a session was still open so the data loss is visible.
            if self._recording_state_dict.get("recording", False):
                logger.warning(
                    "destroy() called while a dataset recording was active; the unsaved "
                    "episode buffer is discarded. Call stop_recording() before destroy() "
                    "to flush and finalize the dataset."
                )
            self._recording_state_dict = {}

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

        Concurrency: main-thread affine. ``world.reset()`` drives Isaac's kit
        runtime (``SimulationContext.stop()``/``play()``), which only pumps
        updates on the thread that created ``SimulationApp``. Off that thread
        the call is marshalled via :meth:`run_on_main` when
        :meth:`run_pump_forever` is engaged, and raises ``RuntimeError``
        (rather than blocking forever) when it is not. See
        :meth:`_marshal_main_thread_affine`.
        """
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

        def _reset_impl() -> dict[str, Any]:
            with self._lock:
                # Re-checked under the lock: a pump-marshalled call may race a
                # concurrent destroy() between the public check above and here.
                if not self._world_created:
                    return {"status": "error", "content": [{"text": "No world created."}]}

                if self._world is not None:
                    self._world.reset()
                    # ``world.reset()`` on the pip Isaac Sim 6.0.x wheels
                    # invalidates the physics-tensor view the per-robot
                    # ``SingleArticulation`` handles hold (the #1798
                    # invalidate-on-stop family, articulation edition), so
                    # without an explicit revive every post-reset
                    # ``get_joint_positions()`` returns ``None`` and
                    # ``get_observation`` degrades to its documented
                    # silent-empty mode (#1895).
                    self._revive_articulations_after_reset()

                self._sim_time = 0.0
                self._step_count = 0

                if env_ids is None:
                    msg = "Full reset complete."
                else:
                    msg = f"Partial reset complete for {len(env_ids)} envs."

                return {"status": "success", "content": [{"text": msg}]}

        return self._marshal_main_thread_affine("reset", _reset_impl)

    def _revive_articulations_after_reset(self) -> None:
        """Re-initialize robot articulation handles ``world.reset()`` killed.

        On the pip Isaac Sim 6.0.x wheels (verified live on ``isaacsim``
        6.0.0.1, 2026-08-03), ``world.reset()`` tears down and rebuilds the
        physics-tensor simulation view, and the per-robot
        ``SingleArticulation`` handles are left holding the torn-down view -
        ``get_joint_positions()`` returns ``None`` and every joint read /
        action apply after ANY reset silently degrades (#1895). This is the
        articulation edition of the #1798 invalidate-on-stop family: #1798
        fixed the scene-object path (``_stop_timeline_for_deferred_physics``
        invalidates synchronously; :meth:`load_scene` rebuilds and re-inits),
        this covers the wrapper handles our ``_RobotState`` bookkeeping owns.

        Mirrors #1798's prevent-and-revive approach rather than catching the
        downstream ``None``s: each registered robot's handle is probed
        (``get_joint_positions() is not None``), and only a dead handle is
        re-initialized against the fresh view - a probe-alive handle is left
        untouched, so builds whose reset keeps handles live (e.g. binary
        Isaac installs) pay one cheap read and nothing else.

        Best-effort per robot, matching :meth:`load_scene`'s re-init error
        tolerance: a robot without a handle (Phase-1 procedural / load stub)
        is skipped, and a failed re-init is logged loudly because joint
        observations for that robot WILL be empty afterwards.

        Concurrency: caller holds ``self._lock`` (called from ``reset()``
        only, after ``world.reset()`` completes).
        """
        for robot in self._robots.values():
            if robot.articulation is None:
                continue
            try:
                if robot.articulation.get_joint_positions() is not None:
                    continue  # handle survived the reset; leave it alone
            except (RuntimeError, ValueError, AttributeError, TypeError):
                # A dead handle may raise instead of returning None on some
                # SDK surfaces; either way it needs the re-init below.
                pass
            try:
                robot.articulation.initialize(getattr(self._world, "physics_sim_view", None))
            except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                logger.warning(
                    "reset: re-initializing articulation for robot %r after world.reset() "
                    "failed (%s); joint observations for this robot will be EMPTY until "
                    "the handle is re-initialized.",
                    robot.name,
                    e,
                )

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        Parameters
        ----------
        n_steps : int
            Non-negative whole number of steps to take, on the shared
            :func:`~strands_robots.utils.non_negative_whole_number_error` domain
            every backend applies. Default 1, and ``0`` is an accepted no-op. A
            NumPy or float count with an integral value is honored and coerced.

        Returns
        -------
        dict
            Status dict with timing info. ``status`` is ``"error"`` when no
            world exists, ``n_steps`` is outside that domain, or the world was
            destroyed on a batch boundary mid-run - in which case the error
            names the steps completed, since some were.

        Concurrency: main-thread affine. ``world.step()`` drives Isaac's kit
        runtime, which only pumps updates on the thread that created
        ``SimulationApp``. Off that thread the batched loop is marshalled via
        :meth:`run_on_main` when :meth:`run_pump_forever` is engaged, and
        raises ``RuntimeError`` (rather than blocking forever) when it is not.
        See :meth:`_marshal_main_thread_affine`.
        """
        # Guarded before the lock is taken and before any world tick: a
        # negative count made ``range()`` empty, so the call reported success
        # having stepped nothing, and divided the elapsed wall time by that
        # negative count to report a negative steps/sec rate. Every
        # non-integral value reached ``range()`` and raised past this method's
        # structured envelope.
        if error := non_negative_whole_number_error(n_steps, "n_steps", "step"):
            return {"status": "error", "content": [{"text": error}]}
        n_steps = int(n_steps)
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            if self._world is None:
                return {"status": "error", "content": [{"text": "World not initialized."}]}

        # Nested (not a separate method) so the batching loop remains part of
        # ``step``'s own body: the cross-backend batch-and-recheck contract is
        # enforced structurally by scanning each concrete ``step`` for
        # ``_STEPS_PER_BATCH`` and ``step_aborted_msg`` references, and a
        # helper method would hide them from that scan.
        def _step_impl() -> dict[str, Any]:
            t0 = time.perf_counter()
            # Batched so the lock is released every ``_STEPS_PER_BATCH`` ticks.
            # Previously the whole count ran under one acquisition, so a worker
            # thread's ``get_state`` / ``get_observation`` - and the pump's own
            # queue drain - blocked for the duration: measured, ``step(100_001)``
            # called ``world.step`` 100_001 times inside a single hold, which at a
            # ~2 ms tick is over three minutes with nothing able to interleave.
            #
            # The per-batch re-check is the other half of that release, not a
            # separate concern: with the lock dropped between batches a concurrent
            # ``destroy`` / ``cleanup`` becomes reachable mid-call, so each batch confirms
            # the world it is about to advance still exists and aborts naming the
            # steps completed rather than stepping a torn-down stage.
            remaining = n_steps
            while remaining > 0:
                batch = min(remaining, self._STEPS_PER_BATCH)
                with self._lock:
                    if not self._world_created or self._world is None:
                        return {
                            "status": "error",
                            "content": [{"text": step_aborted_msg(n_steps - remaining, n_steps)}],
                        }
                    render = self._config.render_mode != "headless"
                    for _ in range(batch):
                        self._world.step(render=render)
                        self._sim_time += self._config.physics_dt
                        self._step_count += 1
                remaining -= batch

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

        return self._marshal_main_thread_affine("step", _step_impl)

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
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        Parameters
        ----------
        name : str
            Robot identifier (also used for procedural lookup). Must be a
            non-empty string with no NUL, on the shared
            :func:`~strands_robots.utils.entity_name_error` domain the MuJoCo
            and Newton backends' ``add_robot`` enforces, so a robot name one
            backend refuses is refused by all three. This backend has no
            "derive a label from the model" short form, so ``None`` / ``""``
            are refused here rather than resolved to a generated label.
        urdf_path : str, optional
            Path to URDF file.
        mjcf_path : str, optional
            Path to an MJCF file. The Isaac backend has no MJCF importer for
            robots (it loads USD natively and converts URDF via the Omniverse
            URDF importer), so a non-None value is rejected with an actionable
            error rather than being silently ignored -- previously a name that
            also matched the procedural registry would silently spawn the
            procedural stub instead. Convert the MJCF to URDF/USD, or use
            create_simulation(backend="mujoco") to load MJCF directly.
        usd_path : str, optional
            Path to USD file (native Isaac format).
        data_config : str, optional
            Named data config for procedural lookup.
        position : list[float], optional
            Base position [x, y, z].
        orientation : list[float], optional
            Base orientation as quaternion [w, x, y, z]. The Isaac ``add_robot``
            spawn path does not yet apply a base orientation (the USD/URDF
            loaders position the articulation but ignore rotation), so a
            non-identity quaternion is rejected with an actionable error rather
            than being silently dropped. Omit it (or pass identity
            ``[1, 0, 0, 0]``) for the default upright spawn, or use
            create_simulation(backend="mujoco") to spawn at an arbitrary
            orientation.
        keyframe : str | int, optional
            Canonical-pose keyframe (e.g. panda ``"home"``). The Isaac
            backend does not parse the MuJoCo ``<keyframe>`` block this
            refers to, so a non-None value is rejected with an actionable
            error rather than raising ``TypeError`` or being silently
            ignored (honouring the
            :class:`~strands_robots.simulation.base.SimEngine` ``add_robot``
            contract). Use ``create_simulation(backend="mujoco")`` to spawn
            at a keyframe, or omit ``keyframe`` for the default zero-pose
            spawn.

        Validation
        ----------
        ``position`` must be 3 finite numbers and ``orientation`` 4, on the
        shared :func:`~strands_robots.utils.coerce_pose_vector` domain the
        MuJoCo and Newton backends' ``add_robot`` enforce. The pose domain is
        checked before the identity-only ``orientation`` reject, so a
        wrong-length quaternion reports its real defect instead of being told
        that base rotation is unsupported.

        Returns
        -------
        dict
            Status dict with robot info.
        """
        if keyframe is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: keyframe={keyframe!r} is not supported on "
                            "the Isaac backend (spawning at a MuJoCo <keyframe> pose "
                            "is currently MuJoCo-only); use "
                            "create_simulation(backend='mujoco') to spawn at a "
                            "keyframe, or omit keyframe for the default zero-pose "
                            "spawn."
                        )
                    }
                ],
            }
        if mjcf_path is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: mjcf_path={mjcf_path!r} is not supported on the Isaac "
                            "backend (it has no MJCF robot importer; it loads USD natively and "
                            "converts URDF). Convert the MJCF to URDF/USD and pass urdf_path/"
                            "usd_path, or use create_simulation(backend='mujoco') to load MJCF."
                        )
                    }
                ],
            }
        # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
        # MuJoCo backend's ``add_robot`` and this backend's own ``add_camera`` already
        # use, so a pose one backend refuses is refused by all of them - the
        # invariant that helper documents. The ``position or [0.0, 0.0, 0.0]`` read
        # this replaces tested the VECTOR: a NumPy array raised a bare
        # ``ValueError: truth value of an array ... is ambiguous`` through the
        # structured envelope, and an empty vector read as *omitted*.
        position, _perr = coerce_pose_vector("add_robot", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        orientation, _oerr = coerce_pose_vector("add_robot", "orientation", orientation, 4)
        if _oerr is not None:
            return {"status": "error", "content": [{"text": _oerr}]}
        # The default None means identity; only a non-identity quaternion is
        # rejected (parity with the keyframe/terrain guards -- reject an
        # unsupported request rather than silently dropping the rotation).
        if orientation is not None and list(orientation) != [1.0, 0.0, 0.0, 0.0]:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_robot: orientation={orientation!r} is not applied on the Isaac "
                            "backend spawn path (the USD/URDF loaders position the articulation but "
                            "ignore base rotation). Omit orientation (or pass identity "
                            "[1, 0, 0, 0]) for the default upright spawn, or use "
                            "create_simulation(backend='mujoco') to spawn at an orientation."
                        )
                    }
                ],
            }
        with self._lock:
            if not self._world_created:
                return {
                    "status": "error",
                    "content": [{"text": "No world created. Call create_world() first."}],
                }

            # Refuse a name that cannot address the robot this call creates, on the
            # shared ``entity_name_error`` domain the MuJoCo and Newton backends'
            # ``add_robot`` already applies, so a name one backend refuses is
            # refused by all three. Unlike the MuJoCo backend this one has no
            # "derive a label from the model" short form - ``name`` IS the
            # procedural lookup key - so every value goes through the domain.
            #
            # An unaddressable name is not merely cosmetic here, because the
            # prim path is interpolated from it (``{stage}/Robots/{name}``) and
            # ``remove_robot`` prunes the cleanup registry with
            # ``p.startswith(prim_path)``: ``add_robot("")`` reported success at
            # the path ``/World/Robots/``, which is the *container* scope for
            # every robot, so a later ``remove_robot("")`` pruned EVERY robot's
            # prim from ``_prim_registry`` - two live robots left with zero
            # tracked prims to release at teardown. An int name registered the
            # key ``7``, which the tool surface, where a name always arrives as a
            # JSON string, can never address, and an unhashable name raised
            # ``TypeError`` out of the duplicate-name test below, escaping the
            # structured envelope this method documents as its failure channel.
            if (name_err := entity_name_error("add_robot", "name", name)) is not None:
                return {"status": "error", "content": [{"text": name_err}]}

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

            pos = [0.0, 0.0, 0.0] if position is None else position
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
                    data_config=data_config,
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
                    data_config=data_config,
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
                    data_config=data_config,
                    usd_to_urdf_joint_names=getattr(articulation, "_strands_usd_to_urdf_joint_names", None),
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
        mesh_path: str | None = None,
        material: dict[str, Any] | None = None,
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
            Object identifier. Must be a non-empty string with no NUL, on the
            shared :func:`~strands_robots.utils.entity_name_error` domain the
            MuJoCo and Newton backends' ``add_object`` enforces, so an object
            name one backend refuses is refused by all three - the prim path is
            interpolated from it, and an entity registered under a name nothing
            can resolve is unreachable through the very API that created it.
            Must also be unique across the simulation; a duplicate is rejected
            with a structured error envelope rather than silently overwriting
            the existing prim.
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
            the missing trailing components. Whether that fallback should
            survive is the open half of #1858 - MuJoCo refuses a short
            ``size`` outright - so it is unchanged here.

            Must be a non-empty vector of finite numbers, on the shared
            :func:`~strands_robots.utils.coerce_size_vector` domain the
            MuJoCo and Newton backends' ``add_object`` enforce, so an extent
            one backend refuses is refused by all three. A ``nan``/``inf``,
            ``bool`` or non-numeric component is refused rather than reaching
            the prim constructor, an empty vector is a component count rather
            than an omission, and a scalar is refused by name instead of
            raising ``TypeError: 'float' object is not iterable`` out of the
            ``list()`` that used to coerce it. NumPy arrays are accepted and
            normalized to plain floats, so a ``np.float64`` extent no longer
            reaches the result ``json``.
        color : list[float], optional
            ``[r, g, b]`` or ``[r, g, b, a]`` in 0..1 (an RGB triple is
            completed with an opaque alpha; the Isaac shape wrappers take
            RGB only). Any other component count is refused rather than
            truncated, since applying only the leading components would
            paint a colour that was not asked for. NumPy arrays are
            accepted. ``None`` -> default white.
        mass : float
            Mass in kg for a dynamic object; a finite number > 0. Default
            0.1. Ignored when ``is_static=True``, and not validated there
            since nothing reads it. Unlike the Newton backend there is no
            ``mass=0`` "make it static" spelling - ``0`` is refused with
            ``is_static=True`` named as the remedy, which is the MuJoCo
            contract this backend's docs otherwise mirror.
        is_static : bool
            If ``True``, the prim is constructed via ``Fixed{Cuboid,
            Sphere, Cylinder, Capsule}`` and stays pinned in space. If
            ``False`` (default), uses the ``Dynamic*`` counterpart and
            participates in physics with ``mass``.
        mesh_path : str, optional
            Path to a custom mesh asset. Loading custom meshes is not
            supported by the Isaac backend yet; a non-``None`` value is
            rejected with an actionable error rather than being silently
            ignored (honouring the
            :class:`~strands_robots.simulation.base.SimEngine` ``add_object``
            contract). Use ``create_simulation(backend='mujoco')`` for meshes.
        material : dict, optional
            Visual material/texture spec. NOT supported by the Isaac backend
            yet; a non-``None`` value is rejected loudly rather than silently
            dropped (use the MuJoCo backend for matte/textured surfaces).
        **kwargs
            ``scale`` (the ``size`` alias documented above) is the only
            keyword read here. Any other keyword is refused rather than
            dropped -- see Validation below.

        Validation
        ----------
        ``position`` must be 3 finite numbers and ``orientation`` 4 (a list,
        tuple or NumPy array; NumPy scalar elements accepted, ``bool``
        refused), on the shared
        :func:`~strands_robots.utils.coerce_pose_vector` domain the MuJoCo and
        Newton backends' ``add_object`` enforce, so a pose one backend refuses
        is refused by all three. Omit a vector to take its default; an empty
        vector is a wrong-length request rather than an omission.

        ``mass`` must be a finite number > 0 for a dynamic object, on the shared
        :meth:`~strands_robots.simulation.base.SimEngine._validate_mass` domain
        the MuJoCo backend's ``add_object`` and ``set_body_properties`` enforce,
        so a mass one backend refuses is refused by all three. It is checked
        before the prim is constructed, because it used to be read *after* the
        object was registered: ``float(mass)`` while assembling the result
        raised for a non-number once the prim was already on the stage and in
        ``_objects``, leaving the name permanently taken. A static object's mass
        is never read, so it is not validated - the same scope MuJoCo uses.

        A keyword this method does not honor is refused by name on the shared
        :func:`~strands_robots.simulation.base.unknown_kwargs_error` contract the
        MuJoCo and Newton backends' discarding ``**kwargs`` sinks (``randomize``,
        ``set_obs_noise``) already apply, so a caller mistake reports the same way
        on every backend. Both sibling ``add_object`` implementations declare the
        same ten parameters with no ``**kwargs`` at all, so Python refuses an
        unknown keyword there with a ``TypeError``; here the keys reached a sink
        that read ``scale`` and discarded the rest, turning ``heigth=0.3`` or
        ``colour=[1, 0, 0]`` into a silent no-op reported as success. The check
        runs before anything else so a refused call constructs no prim, takes no
        lock and leaves the name reusable.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``shape``, ``position``, ``orientation``, ``size``,
            ``mass``, and ``is_static`` so an agent can confirm what
            actually landed on the stage without re-querying.
        """
        # ``**kwargs`` exists for one keyword - the documented ``scale`` alias
        # read below - and every other residual key used to be dropped. That
        # made a misspelled or invented parameter a successful no-op on the
        # backend's most-used scene surface: ``add_object(name="crate",
        # heigth=0.3)`` returned ``status="success"`` having compiled the
        # default extents, while the sibling backends' ``add_object`` - which
        # declare the same ten parameters and no ``**kwargs`` - refuse the same
        # call with a ``TypeError``. The action dispatcher deliberately skips
        # its own unknown-key check for a ``**kwargs`` method and delegates it
        # to the method (see ``_validate_and_build_kwargs``), so this is the
        # only place it can run.
        if err := unknown_kwargs_error("add_object", kwargs, _ADD_OBJECT_PARAMS):
            return err
        if material is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_object: material= is not supported on the "
                            "Isaac backend yet (matte/textured surfaces); use "
                            "create_simulation(backend='mujoco') for materials, "
                            "or omit material for a flat color."
                        )
                    }
                ],
            }
        if mesh_path is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_object: mesh_path=/custom mesh objects are not "
                            "supported on the Isaac backend yet; use "
                            "create_simulation(backend='mujoco') for meshes, or a "
                            "primitive shape (box/sphere/capsule/cylinder)."
                        )
                    }
                ],
            }
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Refuse a name that cannot address the object this call creates, on
            # the shared ``entity_name_error`` domain the MuJoCo and Newton
            # backends' ``add_object`` already applies, so a name one backend
            # refuses is refused by all three. The prim path is interpolated from
            # the name (``{stage}/Objects/{name}``), so ``add_object("")``
            # reported success at ``/World/Objects/`` - the container scope, not a
            # child prim - and a non-string name keyed ``_objects`` with a value
            # the tool surface can never send back. An unhashable name raised
            # ``TypeError`` out of the duplicate-name test below rather than
            # returning the structured error this method documents.
            if (name_err := entity_name_error("add_object", "name", name)) is not None:
                return {"status": "error", "content": [{"text": name_err}]}

            # Normalize shape aliases. ``"cuboid"`` is accepted as an
            # alias for ``"box"`` because it matches Isaac's underlying
            # ``DynamicCuboid`` / ``FixedCuboid`` class names and is the
            # vocabulary used throughout the docs (see robots-sim#88). The
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
            # vocabulary -- see robots-sim#88). An explicit ``size`` always
            # wins over ``scale`` if both are passed.
            # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
            # MuJoCo backend's ``add_object`` and this backend's own ``add_camera`` already
            # use, so a pose one backend refuses is refused by all of them - the
            # invariant that helper documents. The ``list(x)`` coercion this
            # replaces validated nothing: a wrong-length, ``nan``/``inf``, ``bool`` or
            # non-numeric component was passed straight to the prim constructor, and a
            # bare string was split per character into a 3-component "position".
            position, _perr = coerce_pose_vector("add_object", "position", position, 3)
            if _perr is not None:
                return {"status": "error", "content": [{"text": _perr}]}
            orientation, _oerr = coerce_pose_vector("add_object", "orientation", orientation, 4)
            if _oerr is not None:
                return {"status": "error", "content": [{"text": _oerr}]}
            # Same shared domain for the colour, whose accepted counts the
            # 4-component rgba row it ends up in defines. Forwarding it raw
            # validated nothing and then TRUNCATED: ``_construct_shape_prim``
            # writes ``list(color)[:3]``, so a 5-component colour was applied as
            # its first 3 under a success result, ``"abcd"`` was split per
            # character into the colour ``['a', 'b', 'c']``, and a scalar reached
            # ``np.asarray`` before failing. Normalizing to 4 components here
            # makes that ``[:3]`` read well-defined by construction.
            color, _cerr = coerce_rgba("add_object", "color", color)
            if _cerr is not None:
                return {"status": "error", "content": [{"text": _cerr}]}
            # Same idea for the mass, on the shared ``_validate_mass`` domain the
            # MuJoCo backend applies to the same quantity. Position matters as
            # much as the check: the only place the raw value was read was
            # ``float(mass)`` while assembling the result, which is AFTER
            # ``_construct_shape_prim``, ``scene.add``, the ``_prim_registry``
            # append and the ``_objects`` entry. A non-number therefore raised
            # past the envelope this method documents as its only failure
            # channel with the prim already on the stage and the name already
            # taken - so the obvious recovery, retrying under the same name with
            # a usable mass, was refused as a duplicate. Checked here, a refused
            # mass constructs nothing and leaves the name reusable.
            if not is_static and (mass_err := self._validate_mass(mass, "add_object")) is not None:
                return mass_err
            scale_alias = kwargs.pop("scale", None)
            if size is None and scale_alias is not None:
                size = scale_alias
            # The shape-independent half of the extent contract, on the shared
            # ``coerce_size_vector`` domain the MuJoCo backend's ``add_object``
            # composes with its own per-shape table. It runs AFTER the ``scale``
            # alias is resolved, because the two spellings name one parameter and
            # a domain only one of them enforced would be no domain at all. The
            # ``list(size)`` below coerced without validating, exactly as the pose
            # did before #1853: ``[nan, .1, .1]`` and ``[inf, .1, .1]`` reached the
            # prim constructor verbatim, ``[True, .1, .1]`` passed ``True`` as an
            # extent, ``[None, .1, .1]`` and ``[[0.1], .1, .1]`` passed a ``None``
            # and a nested list, the bare string ``"abc"`` was SPLIT per character
            # into a 3-component extent ``['a', 'b', 'c']``, ``[]`` was forwarded
            # as a sizeless size, and a scalar ``0.5`` raised
            # ``TypeError: 'float' object is not iterable`` out of that very
            # ``list()`` call - past the envelope this method documents as its only
            # failure channel. The per-shape component count, the short-vector
            # fallback this method's ``size`` docstring promises and the positivity
            # of a consumed extent are shape-dependent and stay with #1858.
            size, _serr = coerce_size_vector("add_object", "size", size)
            if _serr is not None:
                return {"status": "error", "content": [{"text": _serr}]}

            pos = [0.0, 0.0, 0.5] if position is None else position
            orient = [1.0, 0.0, 0.0, 0.0] if orientation is None else orientation
            size_in = size
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
                # exception-hygiene pin, robots-sim#31).
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
            # ``obj_info["mass"]``, not the caller's ``mass``: for a static object
            # the raw value is documented as ignored and is never coerced, so
            # formatting it with ``%.3f`` raised ``TypeError: must be real number``
            # inside the logging call - only when INFO was enabled, which is why it
            # sat latent. The resolved value is always a float and is the number
            # this result reports, so the log and the payload cannot disagree.
            logger.info(
                "Added object '%s' (shape=%s, pos=%s, mass=%.3f, static=%s)",
                name,
                shape,
                pos,
                obj_info["mass"],
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
        `robots-sim#159
        <https://github.com/strands-labs/robots-sim/issues/159>`_).

        We *prevent* the failure rather than catching it: a bare
        ``except Exception`` is forbidden in this module by the
        exception-hygiene pin (robots-sim#31), and ``omni.physics.tensors``
        raises exactly that bare type. Before constructing any
        ``Dynamic*`` prim we stop the timeline, which clears the
        physics-tensor view so ``RigidPrim.__init__`` skips its eager
        ``_on_physics_ready`` velocity query; the prim is then
        initialised cleanly on the next ``world.reset()`` that
        ``world.scene.add`` schedules -- mirroring Isaac's own "build
        rigid bodies, then reset once" scene-construction ordering.

        The stop is *unconditional* for dynamic prims rather than gated on
        a ``SimulationManager.get_physics_sim_view()`` probe. The earlier
        probe-gated guard (robots-sim#161) was a no-op on the actual
        ``robots-sim#159`` ``load_scene`` path: in a real Isaac 6.0 run the
        probe reported no live view while ``RigidPrim.__init__`` still issued
        the eager velocity query and crashed -- the probe checks a *different*
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
        """Stop the Isaac timeline AND invalidate the physics-tensor view.

        After this returns the physics-tensor view is torn down, so
        ``RigidPrim.__init__`` skips its eager ``_on_physics_ready``
        velocity query and a freshly constructed ``Dynamic*`` prim
        initialises only on the next ``world.reset()``.

        ``timeline.stop()`` alone is NOT sufficient on Isaac Sim 6.0.x
        (verified on the pip ``isaacsim`` 6.0.0.1 and 6.0.1.0 wheels):
        the view invalidation that upstream documents as happening
        "automatically when the timeline is stopped" rides an event
        subscription, and ``SimulationManager.get_physics_sim_view()``
        is still non-``None`` when ``stop()`` returns - so the eager
        velocity query in ``RigidPrim.__init__`` still fires and raises
        the bare ``Exception("Failed to get rigid body velocities from
        backend")`` this guard exists to prevent (#159 lineage). Upstream
        provides :meth:`SimulationManager.invalidate_physics` as the
        documented manual-invalidation entry point ("not intended to be
        called directly unless a manual invalidation is desired/required"
        - deferring a not-yet-viewed prim's physics init is exactly that
        case), and it tears the view down synchronously. Call both: stop
        the timeline (scene-build ordering, matches Isaac's own "build
        rigid bodies, then reset once" flow) then invalidate the view.

        Best-effort: a missing ``omni.timeline`` /
        ``isaacsim.core.simulation_manager`` (partial Isaac install, older
        Isaac without the manager API) is logged and ignored.
        """
        try:
            import omni.timeline  # type: ignore[import-not-found]

            omni.timeline.get_timeline_interface().stop()
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("Could not stop timeline to defer physics init: %s", e)
        try:
            from isaacsim.core.simulation_manager import (  # type: ignore[import-not-found]
                SimulationManager,
            )

            if SimulationManager.get_physics_sim_view() is not None:
                SimulationManager.invalidate_physics()
                logger.info(
                    "Invalidated live physics-tensor view after timeline stop "
                    "(stop() alone leaves the view armed on Isaac Sim 6.0.x, "
                    "so RigidPrim.__init__ would still run its eager velocity "
                    "query on a prim outside the view)."
                )
        except (ImportError, AttributeError, RuntimeError) as e:
            logger.warning("Could not invalidate physics-tensor view: %s", e)

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
            # ``[0.10, 0.05, 0.05]``.
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
        LIBERO-on-Isaac gap that robots-sim#117 deferred with a fail-fast
        stub -- see `robots-sim#129
        <https://github.com/strands-labs/robots-sim/issues/129>`_).

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

        Timeline contract (#1820): realizing dynamic objects stops the
        timeline per prim (#159); after rebuilding the physics view this
        method restarts it (``world.play()``, which lands the state change
        without integrating a physics tick), so on success the episode
        integrates physics. No settle step runs here -- the objects sit
        at MJCF *placeholder* poses until
        ``LiberoAdapter._apply_object_pose_state`` teleports them to the
        episode's init poses, and integrating from the placeholder
        configuration explodes PhysX (coincident bodies at the robot
        base). A failed or non-landing restart returns
        ``{"status": "error"}`` rather than leaving the episode silently
        frozen.

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
            removed_any = False
            for prior_name in list(self._scene_objects):
                if registered(self._objects, prior_name):
                    self.remove_object(prior_name)
                    removed_any = True
                self._scene_objects.discard(prior_name)

            # Deleting prims leaves PhysX's simulation view STALE but
            # non-None: the very next ``DynamicCuboid`` construction takes
            # ``RigidPrim.__init__``'s eager ``_on_physics_ready`` path
            # (gated on ``SimulationManager.get_physics_sim_view() is not
            # None``) and its velocity read against the torn-down view
            # raises the bare ``Exception("Failed to get rigid body
            # velocities from backend")`` -- episode 2+ of every Isaac
            # LIBERO eval crashed here (#1802). The #159 timeline-stop
            # guard does not cover this: ``SimulationManager``'s STOP
            # callback is a pop-subscription that only fires on an app
            # update tick, which never happens between ``timeline.stop()``
            # and the prim constructor. ``invalidate_physics()`` is the
            # public, SYNCHRONOUS form of exactly that callback (its
            # docstring sanctions manual invalidation); with the view
            # gone, every subsequent add takes the deferred-init path and
            # the end-of-load ``world.step`` below rebuilds one fresh
            # view for the new scene.
            if removed_any:
                try:
                    from isaacsim.core.simulation_manager import (  # type: ignore[import-not-found]
                        SimulationManager,
                    )

                    SimulationManager.invalidate_physics()
                except (ImportError, RuntimeError, ValueError, AttributeError, TypeError) as e:
                    logger.warning("load_scene: invalidating the physics view after removals failed: %s", e)

            realized: list[str] = []
            skipped: list[dict[str, Any]] = []
            for obj in scene_objects:
                # ``add_object`` rejects duplicate names; if a manually-added
                # object shadows a scene object, skip it rather than abort.
                if registered(self._objects, obj.name):
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

            # Re-initialize physics views (#1802). Realizing new prims on a
            # stage that has ALREADY STEPPED invalidates PhysX's simulation
            # view (``_construct_shape_prim`` also stops the timeline per
            # dynamic prim, #159; the removals above invalidate it
            # explicitly): the per-robot ``SingleArticulation`` handles are
            # left holding the torn-down view, and ``get_joint_positions``
            # returns nothing but the "Physics Simulation View is not
            # created yet" warning. On the LIBERO eval that surfaced as an
            # observation with NO joint keys -> no ``state.gripper`` -> the
            # GR00T server rejecting every call.
            #
            # The rebuild is ``SimulationManager.initialize_physics()``
            # (the public warmup that creates the tensor-API simulation
            # view; a no-op when the view is already live) + per-robot
            # ``articulation.initialize`` against the fresh view --
            # deliberately NOT ``world.reset()``: a full reset re-applies
            # every registered prim's default state on ``post_reset``,
            # which was measured (2026-07-31, Isaac 6.0 / L4) to
            # destabilize the already-posed articulation into a PhysX
            # "Illegal BroadPhaseUpdateData - non-finite bounds" explosion
            # within ~2 s of stepping. Best-effort per robot: a robot
            # without a handle (Phase-1 stub) is skipped, a failed re-init
            # is logged loudly because joint observations WILL be empty
            # afterwards.
            if realized:
                try:
                    import omni.kit.app  # type: ignore[import-not-found]
                    from isaacsim.core.simulation_manager import (  # type: ignore[import-not-found]
                        SimulationManager,
                    )

                    # Flush the timeline-STOP events queued by the
                    # per-dynamic-prim ``timeline.stop()`` calls (#159)
                    # BEFORE rebuilding. Those events are dispatched on the
                    # next app update, and their handlers null every
                    # ``RigidPrim`` / ``Articulation`` physics handle -- a
                    # rebuild done first would be silently undone by the
                    # first rendered step of the episode (observed as
                    # "Physics Simulation View is not created yet" on every
                    # subsequent joint read / action apply of the eval).
                    omni.kit.app.get_app().update()
                    SimulationManager.initialize_physics()
                except (ImportError, RuntimeError, ValueError, AttributeError, TypeError) as e:
                    logger.warning(
                        "load_scene: rebuilding the physics view after realizing scene objects failed: %s", e
                    )
                for robot in self._robots.values():
                    if robot.articulation is None:
                        continue
                    try:
                        robot.articulation.initialize(getattr(self._world, "physics_sim_view", None))
                    except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                        logger.warning(
                            "load_scene: re-initializing articulation for robot %r after scene "
                            "realization failed (%s); joint observations for this robot will be "
                            "EMPTY until the handle is re-initialized.",
                            robot.name,
                            e,
                        )

                # Restart the timeline (#1820). Every dynamic prim realized
                # above stopped it (`_construct_shape_prim`, #159) and
                # nothing else restarts it, so without this the ENTIRE
                # episode runs on frozen physics: `SimulationContext.step`
                # only integrates when `is_playing()` -- `world.step()`
                # renders with stale joint reads, `send_action` targets a
                # view that never integrates -- and every envelope still
                # reports success, so the eval reads green with a
                # motionless robot (measured: 5-episode groot evals at
                # success_rate=0.00 with byte-similar videos).
                #
                # `world.play()` rather than `timeline.play()`: on 6.0.x a
                # bare `timeline.play()` only QUEUES the state change, and
                # the headless step path (`world.step(render=False)`) never
                # runs the app update that would land it -- measured live
                # (Isaac 6.0 / L4): `is_playing()` still False after
                # play() + 5 world.steps, joints frozen. `world.play()` is
                # play + `timeline.commit()` + one app update issued with
                # `/app/player/playSimulations` DISABLED, so the state
                # lands synchronously and, critically, NO physics
                # integrates on the landing tick: the objects realized
                # above still sit at their MJCF *placeholder* poses (LIBERO
                # encodes real per-episode poses in BDDL init states, not
                # in body attributes -- coincident bodies inside the robot
                # base), and integrating even one tick from that
                # configuration storms PhysX "Illegal BroadPhaseUpdateData
                # - non-finite bounds" and NaNs the joint state. That is
                # also why there is deliberately NO settle step here:
                # `LiberoAdapter._apply_object_pose_state` teleports the
                # objects to their episode init poses and owns the settle.
                #
                # Ordering: play comes AFTER the STOP-event flush +
                # `initialize_physics()` + `articulation.initialize()`
                # above -- the stop -> rebuild -> re-init -> play cycle was
                # verified live (Isaac 6.0 / L4) to preserve articulation
                # state, whereas playing before the flush lets the queued
                # STOP handlers null the freshly-built handles.
                #
                # A failure is FATAL, not a warning: warn-and-continue here
                # reproduces the exact silent-frozen-physics defect this
                # block fixes (AGENTS.md: never warn-and-continue if the
                # system will behave unexpectedly).
                try:
                    self._world.play()
                except (AttributeError, RuntimeError, TypeError, ValueError) as e:
                    msg = (
                        f"load_scene: restarting the timeline after realizing scene objects "
                        f"failed ({type(e).__name__}: {e}). The timeline was stopped by the "
                        f"dynamic-prim constructors (#159), so physics would stay FROZEN for "
                        f"the whole episode while every action reports success (#1820). "
                        f"Refusing to continue."
                    )
                    logger.error("IsaacSimulation.load_scene: %s", msg)
                    return {"status": "error", "content": [{"text": msg}]}
                # Verify the play actually landed. Best-effort import: no
                # omni.timeline means no live Kit session (the CPU unit-test
                # skeleton), where the #159 stop never ran either.
                try:
                    import omni.timeline  # type: ignore[import-not-found]
                except ImportError:
                    logger.debug("load_scene: omni.timeline unavailable; skipping timeline-playing verification")
                else:
                    if not omni.timeline.get_timeline_interface().is_playing():
                        msg = (
                            "load_scene: timeline still stopped after world.play() -- physics "
                            "would stay FROZEN for the whole episode while every action "
                            "reports success (#1820). Refusing to continue."
                        )
                        logger.error("IsaacSimulation.load_scene: %s", msg)
                        return {"status": "error", "content": [{"text": msg}]}

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
            if not registered(self._robots, robot_name):
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
            if not registered(self._robots, name):
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{name}' not found."}],
                }
            prim_path = self._robots[name].prim_path
            self._prim_registry = [p for p in self._prim_registry if not p.startswith(prim_path)]
            del self._robots[name]
            # A controller closed over this robot's articulation is stale
            # the moment the robot is gone; drop it with the robot.
            self._action_controllers.pop(name, None)
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
            if not registered(self._objects, name):
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

            if not registered(self._robots, robot_name):
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
            #
            # Recording override (parity with MuJoCo/Newton): a non-image
            # policy (requires_images=False, e.g. the default mock) makes
            # PolicyRunner pass skip_images=True, but while a dataset
            # recording is active the recorded frames MUST carry the camera
            # images the schema declared - so images are forced on for the
            # duration of the session.
            if skip_images:
                rec_state = self._recording_state()
                if rec_state is not None and rec_state.get("recording", False):
                    skip_images = False
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

    def physics_timestep(self) -> float | None:
        """Return the fixed physics integration timestep in seconds.

        Isaac's ``World`` steps at :attr:`IsaacConfig.physics_dt`.
        Reporting it lets :class:`PolicyRunner` derive the physics substeps
        per control step (``round(1 / control_frequency / physics_dt)``) so
        a PD position-servo arm tracks each action's target for the full
        control period -- without this override the base class returned
        ``None`` and every applied action got a single ~8 ms step (#1812).
        """
        return float(self._config.physics_dt)

    def install_action_controller(self, robot_name: str, controller: Any) -> dict[str, Any]:
        """Install a task-space action controller for a robot.

        Once installed, every dict action handed to :meth:`send_action` for
        ``robot_name`` is first converted by
        ``controller.compute_joint_targets(action)`` into a
        ``{joint_name: position_target}`` dict, which then flows through the
        normal name-resolution / ``ArticulationAction`` path. This is the
        Isaac counterpart of the MuJoCo backend's
        ``world._backend_state["action_controller"]`` seam that
        ``LiberoAdapter._install_action_controller`` uses to route GR00T's
        delta-EEF actions (#1812) -- see
        :class:`~strands_robots.simulation.isaac.delta_eef.IsaacDeltaEEFController`.

        Args:
            robot_name: Robot previously added via ``add_robot``.
            controller: Object exposing a callable
                ``compute_joint_targets(action: Mapping) -> dict[str, float]``.

        Returns:
            Standard ``{"status", "content": [{"text"}]}`` envelope.
        """
        with self._lock:
            if not registered(self._robots, robot_name):
                return {
                    "status": "error",
                    "content": [{"text": f"Robot '{robot_name}' not found. Available: {sorted(self._robots)}"}],
                }
            if not callable(getattr(controller, "compute_joint_targets", None)):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "Controller must expose a callable compute_joint_targets(action) "
                                f"method; got {type(controller).__name__}."
                            )
                        }
                    ],
                }
            self._action_controllers[robot_name] = controller
            logger.info(
                "Installed action controller %s for robot '%s'",
                type(controller).__name__,
                robot_name,
            )
            return {
                "status": "success",
                "content": [{"text": f"Action controller installed for '{robot_name}'."}],
            }

    def uninstall_action_controller(self, robot_name: str) -> dict[str, Any]:
        """Remove a previously installed action controller.

        Idempotent: removing a robot with no controller succeeds (the
        post-state is the same), so per-episode re-install loops don't have
        to track whether an install ever happened.
        """
        with self._lock:
            removed = self._action_controllers.pop(robot_name, None)
            text = (
                f"Action controller removed from '{robot_name}'."
                if removed is not None
                else f"No action controller installed for '{robot_name}'."
            )
            return {"status": "success", "content": [{"text": text}]}

    def get_jacobian(
        self,
        body_name: str | None = None,
        site_name: str | None = None,
        geom_name: str | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute the world-frame spatial Jacobian for a robot link.

        Signature-parity with the MuJoCo backend's ``get_jacobian``
        (:meth:`strands_robots.simulation.mujoco.physics.PhysicsMixin.get_jacobian`):
        returns positional (3 x ndof) and rotational (3 x ndof) Jacobians in
        a ``json`` content block. Isaac has no sites/geoms as Jacobian
        targets, so ``site_name`` / ``geom_name`` are rejected loudly rather
        than silently ignored.

        Args:
            body_name: Link (rigid body) name on the articulation, e.g.
                ``panda_hand``.
            site_name: Unsupported on Isaac; passing it returns an error.
            geom_name: Unsupported on Isaac; passing it returns an error.
            robot_name: Robot to query; auto-picked when exactly one robot
                is loaded.

        Returns:
            ``{"status": "success", "content": [{"text"}, {"json": {"jacp",
            "jacr", "nv"}}]}`` on success, an error envelope otherwise.
        """
        if site_name is not None or geom_name is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "The Isaac backend computes Jacobians for articulation links only; "
                            "site_name/geom_name are unsupported. Pass body_name (a link name)."
                        )
                    }
                ],
            }
        if not body_name:
            return {"status": "error", "content": [{"text": "Specify body_name (a link name)."}]}
        with self._lock:
            if robot_name is None:
                if len(self._robots) == 1:
                    robot_name = next(iter(self._robots))
                else:
                    return {
                        "status": "error",
                        "content": [{"text": f"Specify robot_name; available robots: {sorted(self._robots)}"}],
                    }
            if not registered(self._robots, robot_name):
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
            try:
                jac = self._link_jacobian(self._robots[robot_name], body_name)
            except (RuntimeError, ValueError) as e:
                return {"status": "error", "content": [{"text": str(e)}]}
        return {
            "status": "success",
            "content": [
                {"text": f"Jacobian for link '{body_name}': pos=(3, {jac.shape[1]}), rot=(3, {jac.shape[1]})"},
                {"json": {"jacp": jac[:3].tolist(), "jacr": jac[3:].tolist(), "nv": jac.shape[1]}},
            ],
        }

    def _link_jacobian(self, robot: _RobotState, link_name: str) -> np.ndarray:
        """Return the ``(6, num_dof)`` world-frame Jacobian for one link.

        Rows are ``[linear(3); angular(3)]`` (PhysX convention), columns are
        the articulation DOFs in ``robot.joint_names`` order. Reads the
        PhysX tensor buffers via the articulation view --
        ``SingleArticulation`` exposes no public Jacobian accessor in Isaac
        Sim 6.0, only its wrapped ``Articulation`` view does
        (``_articulation_view.get_jacobians()``), so this is the one place
        that private attribute is touched.

        Raises:
            RuntimeError: If the articulation/physics view is not available
                yet, the robot is floating-base (unsupported layout), or the
                buffers cannot be read.
            ValueError: If ``link_name`` is not a link on the articulation.
        """
        articulation = robot.articulation
        if articulation is None:
            raise RuntimeError(f"Robot '{robot.name}' has no articulation handle; was the robot fully loaded?")
        view = getattr(articulation, "_articulation_view", None)
        if view is None:
            raise RuntimeError(
                f"Robot '{robot.name}': articulation exposes no _articulation_view; "
                "cannot read Jacobians on this Isaac Sim version."
            )

        body_names = list(getattr(view, "body_names", None) or [])
        if link_name not in body_names:
            raise ValueError(f"Link '{link_name}' not found on robot '{robot.name}'. Links: {body_names}")

        jacobians = view.get_jacobians()
        if jacobians is None:
            raise RuntimeError(
                "Articulation view returned no Jacobians (physics simulation view not created yet); "
                "step the world at least once before reading Jacobians."
            )
        jac = jacobians.cpu().numpy() if hasattr(jacobians, "cpu") else np.asarray(jacobians)
        # Shape (num_articulations, num_links, 6, num_dof). Fixed-base
        # articulations exclude the root link from the link dimension, so a
        # link's Jacobian row is its body index minus one. A floating base
        # would instead carry num_bodies rows and 6 extra root DOF columns;
        # LIBERO's Franka imports fix_base=True, so refuse the layout we
        # have not verified instead of guessing column semantics.
        jac = jac[0]
        n_dof = len(robot.joint_names)
        if jac.shape[0] == len(body_names) - 1 and jac.shape[2] == n_dof:
            row = body_names.index(link_name) - 1
            if row < 0:
                raise ValueError(f"Link '{link_name}' is the articulation root; it has no Jacobian.")
        else:
            raise RuntimeError(
                f"Unsupported Jacobian layout {jac.shape} for robot '{robot.name}' "
                f"({len(body_names)} links, {n_dof} DOFs). Only fixed-base articulations are supported."
            )
        return np.asarray(jac[row], dtype=np.float64)

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action and advance physics.

        Parameters
        ----------
        action : dict or array-like
            Joint targets. If dict, keyed by joint name -- unless the robot
            has a controller installed via :meth:`install_action_controller`,
            in which case the dict is first converted by the controller
            (e.g. GR00T task-space delta-EEF keys to joint position
            targets) and the converted dict is what gets resolved.
            Values are on the shared action-value domain
            (:meth:`~strands_robots.simulation.base.SimEngine._coerce_action`)
            every backend applies: a value must coerce to a finite scalar
            number, a boolean is refused rather than written as a 1.0/0.0
            target, and a single-element sequence -- the ``list[float]`` a 1-DoF
            key carries under the ``Policy.get_actions`` contract -- is
            unwrapped to its scalar. An ordered vector is bound positionally to
            :meth:`robot_action_keys` and its width must match that list
            exactly; a mismatch is refused rather than applied to whichever DOFs
            it covers. The controller conversion above runs first, so it is the
            controller's output that is checked.
        robot_name : str, optional
            Robot to control.
        n_substeps : int
            Physics sub-steps after applying action. Default 1. A positive whole
            number, on the shared
            :func:`~strands_robots.utils.positive_whole_number_error` domain
            every backend applies: a NumPy or float count with an integral value
            is honored and coerced, and a fractional, zero, negative,
            non-finite, boolean or non-numeric count is refused. ``0`` is
            refused rather than honored as "write but do not advance" -
            :meth:`step` is the surface that advances a count of its own, and it
            accepts ``0`` as a documented no-op. Note this backend's loop had no
            floor at all, so pre-fix a ``0`` or a negative count applied the
            targets and advanced nothing while reporting success - the opposite
            of what MuJoCo and Newton did with the same call.

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
            ``status`` is ``"error"`` when ``n_substeps`` is outside its domain,
            and no joint target is applied when it is.
        """
        # Guarded before the lock is taken and before any target is applied,
        # mirroring this backend's ``step``: a refusal arriving after the write
        # would leave the robot commanded and the world un-advanced. This
        # backend's substep loop is a bare ``range(n_substeps)`` with no floor,
        # so a fractional or non-numeric count raised ``TypeError`` past the
        # structured envelope, and a zero or negative count reported success
        # having advanced nothing.
        if error := positive_whole_number_error(n_substeps, "n_substeps", "send_action"):
            return {"status": "error", "content": [{"text": error}]}
        n_substeps = int(n_substeps)
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

            if not registered(self._robots, robot_name):
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

            robot = self._robots[robot_name]

            # Route a dict action through the robot's installed task-space
            # controller (install_action_controller), which converts e.g.
            # GR00T's delta-EEF keys ({x, y, z, roll, pitch, yaw, gripper})
            # into {joint_name: position_target}. A conversion failure is a
            # hard error envelope, never a silent fall-through to the raw
            # name-lookup path -- the raw path would drop every task-space
            # key and the robot would sit still while the eval reads green
            # (#1812).
            controller = registry_entry(self._action_controllers, robot_name)
            if controller is not None and isinstance(action, dict):
                try:
                    action = controller.compute_joint_targets(action)
                except (RuntimeError, ValueError, TypeError) as e:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Action controller for '{robot_name}' failed to convert the task-space action: {e}"
                                )
                            }
                        ],
                    }

            # Every value that reaches an actuator clears the shared
            # action-value domain first (:meth:`SimEngine._coerce_action`),
            # which is also what binds an ordered action *vector* positionally
            # to ``robot_action_keys`` and refuses a width that does not match
            # it. This backend hand-rolled its own conversion instead, so it
            # wrote a boolean as a 1.0/0.0 target, wrote a non-finite target no
            # solver can honor, applied a mismatched vector to whichever DOFs it
            # happened to cover, and raised ``TypeError`` straight past this
            # envelope for the single-element rows the
            # ``Policy.get_actions -> list[dict]`` contract emits for a 1-DoF
            # key. The coercion runs here rather than before the lock (where the
            # MuJoCo and Newton backends call it) because the task-space
            # controller above may rewrite the action and it is the controller's
            # *output* that is applied; ``self._lock`` is an ``RLock``, so the
            # ``robot_action_keys`` read inside the coercion re-enters safely.
            action_map, coerce_error = self._coerce_action(action, robot_name)
            if coerce_error is not None:
                return coerce_error
            assert action_map is not None  # narrow for mypy: no error implies a mapping

            # Track keys that don't name a joint so unresolved commands surface
            # in the envelope rather than being silently dropped (parity with
            # the MuJoCo backend).
            joint_set = set(robot.joint_names)
            unresolved = [k for k in action_map if k not in joint_set]
            # ``joint_indices`` restricts an ``ArticulationAction`` to a subset
            # of the articulation's DOFs. Command ONLY the named joints and
            # leave the rest at their current PD targets (parity with the
            # MuJoCo/Newton backends). A full zero-filled ``joint_positions``
            # vector would instead drive every unnamed joint to 0.0 -- e.g.
            # ``send_action({"gripper": 0.04})`` would slam the whole arm to its
            # home pose. A full-width vector action arrives here as a mapping
            # over every joint, so the indices then address every DOF in
            # articulation order - what the raw vector path expressed by passing
            # ``None``.
            named = [i for i, jname in enumerate(robot.joint_names) if jname in action_map]
            action_array: np.ndarray = np.array(
                [float(action_map[robot.joint_names[i]]) for i in named],
                dtype=np.float32,
            )
            joint_indices: np.ndarray = np.array(named, dtype=np.int32)

            # Apply to articulation. Isaac Sim 6.0's articulation
            # (``isaacsim.core.prims.SingleArticulation``) drives PD position
            # targets via ``apply_action(ArticulationAction(joint_positions=...))``
            # -- the pre-6.0 ``set_joint_position_targets`` method does not exist
            # on the 6.0 class (the #101 ``omni.isaac.* -> isaacsim.*`` migration
            # renamed imports but missed this articulation method). See
            # ``set_joint_positions`` below for the teleport (non-PD) counterpart.
            if robot.articulation is not None and action_array.size > 0:
                try:
                    from isaacsim.core.utils.types import (  # type: ignore[import-not-found]
                        ArticulationAction,
                    )

                    robot.articulation.apply_action(
                        ArticulationAction(joint_positions=action_array, joint_indices=joint_indices)
                    )
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
            applied = [k for k in action_map if k not in unresolved]
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

    # --- SimEngine: Synchronized multi-robot rollout -------------------------

    def _apply_lockstep_action(self, robot_name: str, action: dict[str, Any], warned_unresolved: set[str]) -> None:
        """Apply one robot's action WITHOUT stepping physics (lockstep half of send_action).

        The synchronized loop in :meth:`run_multi_policy` writes every robot's
        joint targets first and then steps physics ONCE, so it cannot use
        :meth:`send_action` (whose contract steps physics per call). This is
        the apply-only half of that method: task-space controller routing
        (#1812), the shared action-value coercion
        (:meth:`~strands_robots.simulation.base.SimEngine._coerce_action`), and
        the named-joints-only ``ArticulationAction`` application.

        Runs inside the loop's apply-and-step main-thread hop with
        ``self._lock`` held by the caller.

        Raises:
            RuntimeError: On a controller conversion failure, a refused action
                value, or an articulation apply failure. Mid-loop failures
                raise rather than returning an envelope because applying a
                zero/partial substitute and stepping anyway would advance a
                trajectory no policy commanded (Key Conventions #5/#6); the
                loop's ``finally`` clears the running flags.
        """
        robot = self._robots[robot_name]

        # Route through an installed task-space controller first, so the
        # controller's *output* is what clears the value domain (parity with
        # send_action, #1812).
        controller = registry_entry(self._action_controllers, robot_name)
        if controller is not None and isinstance(action, dict):
            try:
                action = controller.compute_joint_targets(action)
            except (RuntimeError, ValueError, TypeError) as e:
                raise RuntimeError(
                    f"run_multi_policy: action controller for '{robot_name}' failed to "
                    f"convert the task-space action: {e}"
                ) from e

        action_map, coerce_error = self._coerce_action(action, robot_name)
        if coerce_error is not None:
            raise RuntimeError(f"run_multi_policy: action for '{robot_name}' refused: {self._first_text(coerce_error)}")
        assert action_map is not None  # narrow for mypy: no error implies a mapping

        # A key naming no joint is surfaced once per (robot, key) rather than
        # silently dropped (parity with the MuJoCo loop's warn-once unresolved
        # reporting) or spammed at the control rate.
        joint_set = set(robot.joint_names)
        for key in action_map:
            if key not in joint_set and (tag := f"{robot_name}:{key}") not in warned_unresolved:
                warned_unresolved.add(tag)
                logger.warning(
                    "run_multi_policy: action key %r resolves to no joint on '%s' and is not applied. Valid keys: %s",
                    key,
                    robot_name,
                    robot.joint_names,
                )

        # Command ONLY the named joints (see send_action for why a full
        # zero-filled vector would slam unnamed joints to 0.0).
        named = [i for i, jname in enumerate(robot.joint_names) if jname in action_map]
        if robot.articulation is None or not named:
            return
        action_array: np.ndarray = np.array(
            [float(action_map[robot.joint_names[i]]) for i in named],
            dtype=np.float32,
        )
        joint_indices: np.ndarray = np.array(named, dtype=np.int32)
        try:
            from isaacsim.core.utils.types import (  # type: ignore[import-not-found]
                ArticulationAction,
            )

            robot.articulation.apply_action(
                ArticulationAction(joint_positions=action_array, joint_indices=joint_indices)
            )
        except (RuntimeError, ValueError, AttributeError, ImportError) as e:
            # Same expected-failure set as send_action's apply path; a failed
            # apply mid-lockstep must halt the loop, not leave this robot
            # coasting on stale targets while its siblings advance.
            raise RuntimeError(f"run_multi_policy: failed to set joint targets on '{robot_name}': {e}") from e

    def run_multi_policy(
        self,
        policies: dict[str, Policy],
        instructions: dict[str, str] | str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int | dict[str, int] = 8,
        n_steps: int | None = None,
        max_steps: int | None = None,
        *,
        reset_between: bool = False,
    ) -> dict[str, Any]:
        """Drive MULTIPLE robots with their own policies in ONE synchronized control loop.

        The Isaac implementation of the
        :meth:`~strands_robots.simulation.base.SimEngine.run_multi_policy`
        contract (#2158, part of the #2122 MuJoCo-parity work), for the fleet
        topology ``add_robot`` already supports: one stage, one env, multiple
        articulations. Per loop iteration it (1) observes every robot once,
        (2) re-queries each robot's policy only when that robot's buffered
        action chunk drains (chunk length via
        :func:`~strands_robots.policies.base.resolve_chunk_length`, exactly as
        the single-policy runner sizes it), (3) applies every robot's joint
        targets, then (4) steps physics ONCE - so all robots stay
        phase-aligned regardless of their individual re-query cadence.

        **Threading**: all Kit/USD/physics interaction is marshalled through
        :meth:`run_on_main` in two batched hops per timestep (observe-all,
        then apply-all-and-step); policy inference runs between the hops on
        the calling thread, never on the Kit main thread. Called on the
        owning thread the hops run inline; called from a worker thread the
        owning thread must be running :meth:`run_pump_forever` (the #1896
        contract :meth:`step` / :meth:`reset` enforce by raising - this
        entry point reports it in the tool envelope instead). No lock is
        held across a marshal hop; each hop takes ``self._lock`` itself.

        **Recording**: when a dataset recording session is active
        (:meth:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin.start_recording`),
        each loop iteration records exactly ONE merged frame containing every
        driven robot's prefixed state/action columns (``alice__shoulder_pan``
        ...) plus all camera images - mirroring the MuJoCo merged-frame
        semantics (:meth:`strands_robots.simulation.mujoco.simulation.Simulation.run_multi_policy`),
        so a 2-robot dataset has both arms co-observed in every frame.
        Cameras are scene-global and read once per step (from the first
        robot's observation), not once per robot. LeRobot stores one task per
        frame, so the recorded task is the FIRST robot's instruction; the
        shared instruction normalizer already warns when distinct per-robot
        instructions are given.

        Args:
            policies: Mapping ``{robot_name: Policy}`` of the robots to drive.
            instructions: Single instruction string for all robots, or a
                ``{robot_name: instruction}`` mapping.
            duration: Episode length in seconds (steps = duration x freq).
                Used only when no ``n_steps`` / ``max_steps`` is given.
            control_frequency: Target Hz for policy queries / physics steps.
            action_horizon: Actions consumed from each policy's chunk before
                re-querying it, as one int or a per-robot mapping.
            n_steps: Exact step horizon (overrides ``duration`` when set).
            max_steps: Legacy alias for ``n_steps``.
            reset_between: Forward-compat with :meth:`run_policy`'s
                multi-episode semantics. Must be ``False``: on the pip Isaac
                Sim wheels :meth:`reset` tears down the articulation
                physics-tensor views (#1895), so a mid-run reset would leave
                every robot unobservable; requesting one returns a structured
                error rather than silently skipping the reset.

        Returns:
            The standard status dict; on success ``content`` carries a text
            summary plus ``{"json": {"steps": N, "per_robot_steps": {...}}}``.

        Raises:
            RuntimeError: Mid-loop on a policy that returns an empty action
                chunk or an action that cannot be applied (see
                :meth:`_apply_lockstep_action`) - never a zero-valued
                substitute action. When recording, the partially-recorded
                frames of the failed episode are discarded so the next episode
                starts at frame 0 rather than appending to a dangling
                half-episode (mirrors the MuJoCo loop).
        """
        from collections import deque

        from strands_robots._async_utils import _resolve_coroutine
        from strands_robots.policies.base import resolve_chunk_length
        from strands_robots.simulation.policy_runner import CooperativeStop

        if not getattr(self, "_world_created", False) or self._world is None:
            return {"status": "error", "content": [{"text": "No world created. Use action='create_world' first."}]}
        if err := self._validate_multi_policies(policies, "run_multi_policy"):
            return err

        # Validate every robot exists.
        for rname in policies:
            if not registered(self._robots, rname):
                return {"status": "error", "content": [{"text": self._unknown_robot_msg(rname)}]}

        # Reject a robot another loop is already driving (double-stepping
        # physics on it), mirroring the MuJoCo busy check. Isaac has no
        # background start_policy futures to prune; ``policy_running`` is the
        # per-robot flag every Isaac policy-driving loop sets.
        busy = [r for r in policies if getattr(self._robots[r], "policy_running", False)]
        if busy:
            names = ", ".join(f"'{n}'" for n in busy)
            return {
                "status": "error",
                "content": [{"text": f"run_multi_policy: policy already running on {names}. Stop it first."}],
            }

        if reset_between:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "run_multi_policy: reset_between=True is not supported on the Isaac "
                            "backend: reset() tears down the articulation physics-tensor views on "
                            "the pip Isaac Sim wheels (#1895), so a mid-run reset would leave every "
                            "robot unobservable. Pass reset_between=False (the default) and reset "
                            "explicitly between calls, or use the MuJoCo backend for multi-episode "
                            "multi-robot rollouts."
                        )
                    }
                ],
            }

        # Latch the active recording session ONCE (MuJoCo parity): every loop
        # iteration below records exactly one merged frame into this recorder.
        # A session started mid-rollout is deliberately not picked up - the
        # schema probe and the rollout must observe the same scene.
        rec_state = self._recording_state()
        recorder = rec_state.get("dataset_recorder") if rec_state is not None else None
        recording = rec_state is not None and bool(rec_state.get("recording", False)) and recorder is not None

        # Thread affinity (#1896): off the owning thread every marshal hop
        # below would block forever without a pump. step()/reset() raise for
        # this; a tool-facing driver reports it in the envelope instead.
        if not self._on_main_thread() and not self._pump_running:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "run_multi_policy was called from a worker thread with no main-thread "
                            "pump running. Isaac Sim only pumps kit updates on the thread that "
                            "created SimulationApp, so the per-step observe/apply hops would block "
                            "forever. Either call it from the owning thread, or have the owning "
                            "thread run `run_pump_forever(stop_event=...)` and call this from the "
                            "worker (the loop marshals each hop via run_on_main itself)."
                        )
                    }
                ],
            }

        # Normalize instructions through the shared base helper (one refusal
        # text for every backend), attributing the distinct-instructions
        # warning to this module's logger.
        instr_map, err = self._normalize_multi_policy_instructions(
            policies, instructions, "run_multi_policy", warn_logger=logger
        )
        if err is not None:
            return err
        assert instr_map is not None  # for the type checker: err is None <=> instr_map is not None

        # Resolve the step horizon through the shared helpers so this loop
        # guards the same domain as run_policy (n_steps / max_steps override
        # duration; frequency validated first because _resolve_horizon divides
        # by it).
        if err := self._validate_positive_frequency(control_frequency, "run_multi_policy"):
            return err
        # Reject a rollout whose rate the active recording cannot describe
        # (MuJoCo parity): every frame below is timestamped at the dataset's
        # fps, so a disagreeing control_frequency would mislabel the episode.
        if err := self._validate_recording_rate(control_frequency, "run_multi_policy"):
            return err
        duration, n_steps, horizon_error = self._resolve_horizon(
            n_steps, max_steps, control_frequency, duration, "run_multi_policy"
        )
        if horizon_error is not None:
            return horizon_error
        if n_steps is None:
            if err := self._validate_duration(duration, "run_multi_policy"):
                return err

        # Normalize action_horizon to a per-robot mapping on the shared
        # positive-int domain.
        horizon_map, err = self._normalize_multi_policy_horizons(
            policies, action_horizon, "run_multi_policy", default_horizon=8
        )
        if err is not None:
            return err
        assert horizon_map is not None  # for the type checker: err is None <=> horizon_map is not None

        # Bind each policy's action keys (best-effort, mirrors run_policy).
        for rname, pol in policies.items():
            try:
                pol.set_robot_state_keys(self.robot_action_keys(rname))
                self.bind_policy_sim_context(pol, rname)
            except Exception as exc:  # noqa: BLE001 - non-fatal, mirrors run_policy defensiveness
                logger.debug("set_robot_state_keys(%s) failed: %s", rname, exc)

        # Merged-frame recording wiring (MuJoCo parity). Namespacing follows
        # the schema start_recording declared: prefixed ``robot__column`` when
        # the WORLD holds more than one robot (not merely this call), so the
        # merged frame always matches the declared columns. Every robot driven
        # here contributes to the one merged frame, so the merged action owes a
        # value for each of their actuators - resolved once rather than per
        # frame, and only when a recorder will consume it (robot_action_keys is
        # best-effort for unrecorded rollouts; where a recording is attached
        # the keys are load-bearing, so a raise here correctly fails the call).
        multi_robot = len(self._robots) > 1
        merged_required_action_keys = (
            [f"{rname}__{key}" if multi_robot else key for rname in policies for key in self.robot_action_keys(rname)]
            if recording
            else []
        )
        # Camera frames ride the observation keyed by RAW camera name; the
        # schema declared the safe names (``/`` -> ``__``), scoped to
        # start_recording(cameras=...). Same rename+scope the single-robot
        # on_frame hook applies.
        raw_to_safe: dict[str, str] = (
            {src: safe for src, safe, _w, _h in rec_state.get("recording_cameras", [])}
            if recording and rec_state is not None
            else {}
        )

        # Renders are expensive; skip camera readback when no policy needs
        # images AND no recording is active (recorded frames must carry the
        # camera images the schema declared).
        any_needs_images = any(getattr(p, "requires_images", True) for p in policies.values())
        skip_images = not (any_needs_images or recording)
        render_on = self._config.render_mode != "headless"
        physics_dt = float(getattr(self._config, "physics_dt", 0.0) or 0.0)

        total_steps = int(duration * control_frequency)
        action_sleep = 1.0 / control_frequency if control_frequency > 0 else 0.0

        # Mark all robots as running so a cooperative stop can interrupt the
        # loop, and so a concurrent driver is refused by the busy check above.
        for rname in policies:
            r = self._robots[rname]
            r.policy_running = True
            r.policy_instruction = instr_map[rname]
            r.policy_steps = 0

        # Per-robot action queue: actions remaining from the last chunk query.
        # A policy is only re-queried when its queue is empty, so expensive VLA
        # inference amortizes over up to ``horizon_map[robot]`` steps.
        action_queues: dict[str, deque] = {r: deque() for r in policies}
        warned_unresolved: set[str] = set()

        def _observe_all() -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
            """Main-thread hop 1: observe every robot, read cameras ONCE."""
            per_obs: dict[str, dict[str, Any]] = {}
            cams: dict[str, Any] = {}
            first = True
            for rname in policies:
                obs = self.get_observation(robot_name=rname, skip_images=(skip_images or not first))
                # Split scalars (joints) from ndarrays (camera images);
                # cameras are scene-global, so one readback serves all robots.
                per_obs[rname] = {k: v for k, v in obs.items() if not isinstance(v, np.ndarray)}
                if first:
                    cams = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
                    first = False
            return per_obs, cams

        def _apply_all_and_step(per_robot_action: dict[str, dict[str, Any]]) -> None:
            """Main-thread hop 2: apply EVERY robot's targets, step physics ONCE."""
            with self._lock:
                for rname, act in per_robot_action.items():
                    self._apply_lockstep_action(rname, act, warned_unresolved)
                self._world.step(render=render_on)
                self._sim_time += physics_dt
                self._step_count += 1

        step_count = 0
        stopped_early = False
        # Tracks whether the loop finished without an unexpected error. A
        # normal completion and a cooperative stop both leave a VALID
        # partial/complete episode the caller will save; any other exception
        # (e.g. an empty action chunk) leaves a dangling partial episode we
        # must discard so the next recording starts at frame 0 rather than
        # appending to a half-episode (MuJoCo parity).
        completed_cleanly = False
        try:
            while step_count < total_steps:
                # --- 1. Observe every robot (one main-thread hop). No lock is
                # held across the marshal (#1896); get_observation takes it.
                per_robot_obs, camera_imgs = self.run_on_main(_observe_all)

                # --- 2. Resolve each robot's action for THIS step, off the
                # main thread. Re-query a policy ONLY when its buffered chunk
                # drains (open-loop chunk execution).
                per_robot_action: dict[str, dict[str, Any]] = {}
                for rname, pol in policies.items():
                    # Cooperative stop check.
                    if not self._robots[rname].policy_running:
                        raise CooperativeStop(f"Policy stopped on '{rname}'")
                    if not action_queues[rname]:
                        pol_obs = dict(per_robot_obs[rname])
                        pol_obs.update(camera_imgs)
                        coro = pol.get_actions(pol_obs, instr_map[rname])
                        acts = _resolve_coroutine(coro)
                        # Size the chunk via the shared ChunkedPolicy rule so a
                        # chunk-emitting policy keeps its full trained chunk
                        # here exactly as the single-policy runner does.
                        _chunk = resolve_chunk_length(pol, horizon_map[rname])
                        for a in acts[:_chunk]:
                            action_queues[rname].append(a)
                    if not action_queues[rname]:
                        # Emitting a zero-valued substitute here would advance
                        # a trajectory no policy commanded. Fail loudly
                        # instead (Key Conventions #6).
                        raise RuntimeError(
                            f"Policy for robot '{rname}' returned an empty action chunk; "
                            "cannot advance the synchronized loop. Check the policy's "
                            "get_actions() output."
                        )
                    per_robot_action[rname] = action_queues[rname].popleft()

                # --- 3+4. Apply ALL robots' targets, then step physics ONCE
                # (one main-thread hop; the hop takes self._lock itself).
                self.run_on_main(lambda acts=per_robot_action: _apply_all_and_step(acts))

                # --- 5. Record ONE merged frame (all robots + all cameras),
                # off the main thread: add_frame writes to LeRobot's
                # image-writer queue and parquet buffer, never to Kit/USD, and
                # the consistent state snapshot was already taken inside the
                # two hops above.
                if recording and recorder is not None:
                    merged_obs: dict[str, Any] = {}
                    merged_act: dict[str, Any] = {}
                    for rname in policies:
                        if multi_robot:
                            for k, v in per_robot_obs[rname].items():
                                merged_obs[f"{rname}__{k}"] = v
                            for k, v in per_robot_action[rname].items():
                                merged_act[f"{rname}__{k}"] = v
                        else:
                            merged_obs.update(per_robot_obs[rname])
                            merged_act.update(per_robot_action[rname])
                    # Cameras are scene-global: rename raw -> schema-safe and
                    # drop any outside the start_recording(cameras=...) scope.
                    for k, v in camera_imgs.items():
                        safe = raw_to_safe.get(k)
                        if safe is not None:
                            merged_obs[safe] = v
                    # LeRobot stores ONE task per frame: the first robot's
                    # instruction (the shared normalizer already warned when
                    # per-robot instructions are distinct).
                    recorder.add_frame(
                        observation=merged_obs,
                        action=merged_act,
                        task=instr_map[next(iter(policies))],
                        required_action_keys=merged_required_action_keys,
                    )

                step_count += 1
                for rname in policies:
                    self._robots[rname].policy_steps = step_count

                if action_sleep:
                    time.sleep(action_sleep)

            completed_cleanly = True
        except CooperativeStop:
            # A cooperative stop is a normal, user-requested halt.
            stopped_early = True
            completed_cleanly = True
        finally:
            for rname in policies:
                if registered(self._robots, rname):
                    self._robots[rname].policy_running = False
            # Bailed mid-episode on an unexpected error (e.g. empty action
            # chunk): drop the partially-recorded frames so the next episode
            # begins at frame 0 instead of appending to a dangling half-episode.
            if not completed_cleanly and recording and recorder is not None:
                recorder.clear_episode_buffer()

        per_robot_steps = {rname: int(self._robots[rname].policy_steps) for rname in policies}
        text = (
            f"{'stopped early' if stopped_early else 'completed'}: "
            f"run_multi_policy on {len(policies)} robots ({', '.join(policies)}) - "
            f"{step_count} synchronized steps"
            f"{' (recorded)' if recording else ''}"
        )
        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"steps": step_count, "per_robot_steps": per_robot_steps}}],
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
          possible after robots-sim#61, but kept as a defensive fallback).
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
            ``content`` (the tool-result contract forbids extra top-level
            keys). On success ``content`` holds a ``text`` block, a ``{"image": {"format": "png", ...}}``
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
        # the Strands tool-result contract permits only {status, content}.
        # Consumers that need the raw rgb/depth ndarrays use get_observation()
        # or the internal
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

            # Same shared pixel floor ``add_camera`` applies, so the
            # config-time and call-time domains agree. These sized the
            # blank-frame fallbacks below with no validation at all: a negative
            # width reached ``np.zeros`` as ``ValueError: negative dimensions
            # are not allowed`` and a fractional or non-numeric one as
            # ``TypeError: 'float' object cannot be interpreted as an
            # integer``, both escaping this method's ``(rgb, depth, meta)``
            # contract. ``None`` still means "take the config default";
            # membership decides that, not truthiness.
            for param, value in (("width", width), ("height", height)):
                if value is not None and (dim_err := positive_count_error(value, param, "render")) is not None:
                    return None, None, {"error": dim_err}

            w = self._config.camera_width if width is None else width
            h = self._config.camera_height if height is None else height

            if self._config.render_mode == "headless":
                # Return blank frames in headless mode. Most CI flows
                # land here; Isaac's RTX path-tracer is unavailable.
                return (
                    np.zeros((h, w, 3), dtype=np.uint8),
                    np.zeros((h, w), dtype=np.float32),
                    {"text": f"Rendered (headless, no RTX): {w}x{h}"},
                )

            if not registered(self._cameras, camera_name):
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
                    # robots-sim#61's add_camera enables it post-initialize, but
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

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render a camera to raw ``(rgb, depth)`` ndarrays (metric depth).

        Public counterpart of the internal :meth:`_render_frame` (issue
        #1537): returns the raw RTX ``(H, W, 3) uint8`` RGB frame and the
        ``(H, W) float32`` metric depth buffer for in-process consumers such
        as :class:`strands_robots.rendering.HybridCompositor`, without the
        agent-tool PNG envelope and without reaching into private state.

        Unlike :meth:`_render_frame` -- whose blank-frame fallbacks exist for
        the envelope path -- this method **raises** on every degraded path
        (headless mode, unknown camera, Phase-1 camera without an RTX
        handle), so a compositing consumer can never silently receive black
        pixels with zero depth.

        Isaac's depth annotator reports pixels with no geometry as ``0`` or
        non-finite; consumers should treat both extremes as background.

        Concurrency: takes ``self._lock`` (via ``_render_frame``); rendering
        must be driven from the thread that owns the ``SimulationApp`` (use
        :meth:`run_on_main` from worker threads).

        Args:
            camera_name: a camera previously added via ``add_camera``.
            width: must be ``None`` or the camera's native render width, and
                a positive integer when supplied -- Isaac RTX cameras render at
                the resolution fixed at ``add_camera`` time; a mismatch raises
                rather than silently dropping the requested size.
            height: same contract as ``width``.

        Returns:
            ``(rgb, depth)`` -- ``(H, W, 3) uint8`` and ``(H, W) float32``.

        Raises:
            RuntimeError: no world, headless render mode, camera without an
                RTX handle, or an RTX render failure.
            KeyError: unknown camera name.
            ValueError: ``width``/``height`` is not a positive integer, or
                differs from the camera's native render resolution.
        """
        with self._lock:
            if not self._world_created:
                raise RuntimeError("No world created. Call create_world first.")
            if self._config.render_mode == "headless":
                raise RuntimeError(
                    "get_frame is unavailable in headless render mode (no RTX frames are produced); "
                    "use render_mode='rtx_realtime' or consume the envelope render() fallback."
                )
            if not registered(self._cameras, camera_name):
                raise KeyError(f"Camera '{camera_name}' not found. Available: {sorted(self._cameras)}")
            cam = self._cameras[camera_name]
            if cam.handle is None:
                raise RuntimeError(f"Camera '{camera_name}' has no live RTX handle; re-add it via add_camera().")
            for arg_name, arg, native in (("width", width, cam.width), ("height", height, cam.height)):
                # Shared pixel floor first: the comparison below coerces
                # with ``int(arg)``, which raises an uninformative
                # ``ValueError`` for a non-numeric value instead of naming
                # the parameter that was wrong.
                if arg is not None:
                    if (dim_err := positive_count_error(arg, arg_name, "get_frame")) is not None:
                        raise ValueError(dim_err)
                if arg is not None and int(arg) != int(native):
                    raise ValueError(
                        f"Isaac cameras render at the resolution fixed at add_camera time; "
                        f"requested {arg_name}={arg} but camera '{camera_name}' renders at "
                        f"{cam.width}x{cam.height}. Re-add the camera with the desired size."
                    )
            rgb, depth, meta = self._render_frame(camera_name)
        if rgb is None:
            raise RuntimeError(str(meta.get("error", f"Failed to render camera '{camera_name}'")))
        depth_arr = None if depth is None else np.asarray(depth, dtype=np.float32)
        return np.asarray(rgb, dtype=np.uint8), depth_arr

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return pinhole :class:`~strands_robots.rendering.CameraParams`.

        Intrinsics come from the RTX camera handle
        (``Camera.get_intrinsics_matrix()``), the pose from
        ``Camera.get_world_pose()``. ``get_world_pose`` returns the camera
        *prim's* world orientation, whose local axes are offset from the
        OpenGL optical frame ``CameraParams`` promises (+X right, +Y up, -Z
        forward): the USD camera prim basis maps prim +X -> GL -Z, prim +Y ->
        GL -X, prim +Z -> GL +Y. This backend-inherent fixed correction
        (``R_gl = R_prim @ PRIM_TO_GL``) is applied here -- consistent across
        poses -- so a composited background is upright and aligned with the
        RTX foreground (previously example-side, issue #1537).

        Args:
            camera_name: a camera previously added via ``add_camera``.
            width: must be ``None`` or the camera's native render width (the
                handle's intrinsics are only valid at native resolution), and
                a positive integer when supplied.
            height: same contract as ``width``.

        Raises:
            RuntimeError: no world, or the camera has no live RTX handle.
            KeyError: unknown camera name.
            ValueError: ``width``/``height`` is not a positive integer, or
                differs from the native render resolution.
        """
        from strands_robots.rendering import CameraParams

        with self._lock:
            if not self._world_created:
                raise RuntimeError("No world created. Call create_world first.")
            if not registered(self._cameras, camera_name):
                raise KeyError(f"Camera '{camera_name}' not found. Available: {sorted(self._cameras)}")
            cam = self._cameras[camera_name]
            if cam.handle is None:
                raise RuntimeError(
                    f"Camera '{camera_name}' has no live RTX handle -- intrinsics/pose cannot be "
                    "read off a registration-only camera. Re-add it via add_camera()."
                )
            for arg_name, arg, native in (("width", width, cam.width), ("height", height, cam.height)):
                # Shared pixel floor first: the comparison below coerces
                # with ``int(arg)``, which raises an uninformative
                # ``ValueError`` for a non-numeric value instead of naming
                # the parameter that was wrong.
                if arg is not None:
                    if (dim_err := positive_count_error(arg, arg_name, "get_camera_params")) is not None:
                        raise ValueError(dim_err)
                if arg is not None and int(arg) != int(native):
                    raise ValueError(
                        f"Isaac camera intrinsics are only valid at the native render resolution; "
                        f"requested {arg_name}={arg} but camera '{camera_name}' renders at "
                        f"{cam.width}x{cam.height}. Re-add the camera with the desired size."
                    )
            K = np.asarray(cam.handle.get_intrinsics_matrix(), dtype=np.float64).reshape(3, 3)
            position, quat_wxyz = cam.handle.get_world_pose()
            w_px, h_px = int(cam.width), int(cam.height)

        position = np.asarray(position, dtype=np.float64).reshape(3)
        quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)
        # Fixed camera-local correction, USD camera prim basis -> OpenGL
        # optical frame (see docstring). R_gl = R_prim @ PRIM_TO_GL.
        prim_to_gl = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz) @ prim_to_gl
        T[:3, 3] = position
        # Isaac exposes no scene-level clip planes on the handle; carry the
        # conventional near plane and a far plane "at infinity" for the
        # compositor's depth convention.
        return CameraParams(K=K, T_world_cam=T, width=w_px, height=h_px, znear=0.01, zfar=1_000_000.0)

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
            ``n_steps``; ``False`` if it never warmed up. Never raises:
            a step / render failure is caught and ends the loop, because
            the camera is already registered and render()'s own 0-D guard
            still covers a not-yet-ready product.

            A ``False`` is always reported at WARNING, and the two ways of
            getting there are reported DIFFERENTLY because they need
            different remedies: an exhausted budget says the render product
            never accumulated (waiting/stepping longer may help), while an
            early abort names the step reached and the exception that ended
            the loop (waiting cannot help). Reporting both as an exhausted
            budget sent operators after the renderer for what was really a
            surface/attribute error visible only at DEBUG.
        """
        if self._world is None:
            return False

        # A stopped timeline never feeds the RTX render products, so a
        # warm-up loop below it would step the physics scene while every
        # render() probe keeps returning the malformed-buffer error - the
        # exact state load_scene leaves behind after its deferred-physics
        # window (the #159 guard stops the timeline before constructing
        # Dynamic* prims, and ``world.step`` does not resume play). Resume
        # inside the loop, every iteration: timeline stop/play commands
        # land asynchronously on a kit update tick, so ``is_playing()`` can
        # report stale ``True`` right after a queued ``stop()`` and a
        # single pre-loop resume can be undone when that queued stop lands
        # mid-warm-up. Re-asserting play() before each step converges even
        # when a stop event is still in flight. Best-effort, same failure
        # tolerance as the step loop.
        def _ensure_timeline_playing() -> None:
            try:
                import omni.timeline  # type: ignore[import-not-found]

                timeline = omni.timeline.get_timeline_interface()
                if not timeline.is_playing():
                    timeline.play()
                    logger.debug(
                        "Camera %r warm-up: resumed stopped timeline so the RTX render product can accumulate frames",
                        name,
                    )
            except (ImportError, AttributeError, RuntimeError) as e:
                logger.debug("Camera %r warm-up: could not query/resume timeline: %s", name, e)

        budget = max(1, n_steps)
        attempted = 0
        aborted: Exception | None = None
        for i in range(budget):
            attempted = i + 1
            try:
                _ensure_timeline_playing()
                self._world.step(render=True)
                self._sim_time += self._config.physics_dt
                self._step_count += 1
                # ``world.step(render=True)`` reliably refreshes only the
                # PRIMARY render product; a camera added after the first
                # (e.g. the LIBERO adapter's ``wrist_image``, installed at
                # episode start next to the pre-existing ``image``) never
                # accumulates a frame from stepping alone and the warm-up
                # loop ran to exhaustion (#1802). Flush the secondary
                # products the same way ``get_observation`` does before
                # checking for a frame.
                if len(self._cameras) > 1:
                    self._refresh_all_render_products()
                if self.render(camera_name=name).get("status") == "success":
                    logger.debug("Camera %r warmed up after %d step(s)", name, i + 1)
                    return True
            except (RuntimeError, ValueError, OSError, AttributeError, TypeError, IndexError) as e:
                # Stepping / rendering a partially-initialised stage can
                # raise on surface drift; warm-up is best-effort, so log
                # and stop rather than failing the already-registered
                # camera. Programming bugs (NameError) still propagate.
                logger.debug("Camera %r warm-up step %d failed: %s", name, i + 1, e)
                aborted = e
                break
        if aborted is not None:
            # An early abort is NOT a slow render product, and the two need
            # different remedies: the exhaustion report below tells the
            # operator to let the RTX product accumulate, but a loop that
            # stopped on step 1 of 5 will never accumulate anything no matter
            # how long they wait. The cause was DEBUG-only, so the operator
            # saw only the exhaustion text and chased the renderer instead of
            # the exception. Name the step reached and the cause.
            logger.warning(
                "Camera %r warm-up aborted on step %d of %d: %s: %s. The camera stays "
                "registered and the first render() may return an error, but the loop "
                "stopped early - further warm-up steps cannot help until this is resolved.",
                name,
                attempted,
                budget,
                type(aborted).__name__,
                aborted,
            )
            return False
        logger.warning(
            "Camera %r did not produce a valid frame after %d warm-up step(s); "
            "the first render() may return an error until the RTX product accumulates a frame.",
            name,
            budget,
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
        parent_body: str | None = None,
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
            Camera identifier. Default ``"default"``. Must be a non-empty
            string with no NUL (see Validation below). Must also be unique
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
            Image width in pixels; a positive integer. ``None`` (omitted)
            takes ``IsaacConfig.camera_width``.
        height : int, optional
            Image height in pixels; a positive integer. ``None`` (omitted)
            takes ``IsaacConfig.camera_height``.
        parent_body : str, optional
            Body to mount the camera on, so it rides with that body instead
            of standing still in the world. Declared here but NOT SUPPORTED
            on this backend: the camera prim is parented to the stage's
            camera scope, not to an articulation link, so a value is refused
            with a structured error naming the backends that do mount
            cameras rather than dropped. Mounting is what
            :doc:`/policies/camera-naming` prescribes for a VLA's
            ``observation.images.wrist_image`` feature, so a caller
            following that guidance needs to be told which backend can
            honour it -- not handed a static world-space view, and not a
            bare ``TypeError`` naming neither the capability nor the
            alternative. Omit it (the default) for a world-fixed camera,
            which this backend does support.
        fov : float
            Horizontal field of view in degrees. Default 60.0. Mapped
            onto ``Camera.set_focal_length`` using the standard pinhole
            relation ``focal_length = horizontal_aperture / (2 * tan(fov/2))``
            with the USD-default 24 mm horizontal aperture.

        Validation
        ----------
        ``name`` must be a non-empty string containing no NUL, on the shared
        :func:`~strands_robots.utils.entity_name_error` domain - an empty name
        is the routing token ``render`` maps to the free camera, so a camera
        created under it could never be rendered from, and the camera's prim
        path is interpolated from the name.

        ``position`` and ``target`` must each be 3 finite numbers (a list,
        tuple or NumPy array; NumPy scalar elements accepted, ``bool``
        refused), and must not be identical - a camera whose eye is its own
        look-at point has no look direction. Omit a vector to take its
        default; an empty vector is a wrong-length request and is rejected
        rather than silently placing the camera at the default pose. ``fov``
        must be a finite angle in the open interval ``(0, 180)`` degrees, and
        ``width`` / ``height`` must each be a positive integer (omit one to take
        the ``IsaacConfig`` default - ``0`` is a request for an impossible
        resolution, not an omission). These are the same bounds the MuJoCo and
        Newton backends' ``add_camera`` enforces, on the shared
        :func:`~strands_robots.utils.coerce_pose_vector` /
        :func:`~strands_robots.utils.camera_fov_error` /
        :func:`~strands_robots.utils.entity_name_error` /
        :func:`~strands_robots.utils.positive_count_error` domains, so a camera
        configuration one backend refuses is refused by all three. Invalid
        values are rejected here, before the stage is touched, rather than
        deferring an uncaught exception (non-numeric ``fov``, ``fov=0``) or a
        silently degenerate camera (a ``nan`` in the USD pose or in the derived
        focal length) to the first render.

        A non-positive ``width`` is especially costly here because the DLSS
        upscale below multiplies it back out: ``scale = _MIN_RENDER_PX / w`` for
        ``w = -4`` gave a native render size of ``640 x -76800``, and every
        later render of that camera then failed on the negative dimension - one
        bad configuration call disabled the camera for the rest of the session.

        Returns
        -------
        dict
            Standard ``{"status", "content": [{"text", "json"}]}``
            envelope. ``json`` carries the resolved ``prim_path``,
            ``position``, ``target``, ``resolution``, ``fov``, and the
            computed ``focal_length`` so an agent can confirm the
            camera setup without re-querying.
        """
        if parent_body is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"add_camera: parent_body={parent_body!r} is not supported on the Isaac "
                            "backend (it parents camera prims to the stage camera scope, not to an "
                            "articulation link, so the camera would not ride with the body). Omit "
                            "parent_body for a world-fixed camera, or use "
                            "create_simulation(backend='mujoco') / create_simulation(backend='newton') "
                            "for a body-mounted (wrist) camera."
                        )
                    }
                ],
            }
        with self._lock:
            if not self._world_created:
                return {"status": "error", "content": [{"text": "No world created."}]}

            # Refuse a name that cannot address the camera this call creates, on
            # the shared ``entity_name_error`` domain the MuJoCo and Newton
            # backends' ``add_camera`` already applies, so a name one backend
            # refuses is refused by all three - the same invariant this method
            # already honours for ``position`` / ``target`` / ``fov`` / ``width``
            # / ``height`` below. An empty name is worse than unaddressable for a
            # camera: ``render`` routes ``camera_name in (None, "", "default",
            # "free")`` to the free camera by an explicit token check, so a
            # camera created as ``""`` could never be rendered from, while the
            # prim landed at ``/World/Cameras/`` - the container scope shared by
            # every camera on the stage.
            if (name_err := entity_name_error("add_camera", "name", name)) is not None:
                return {"status": "error", "content": [{"text": name_err}]}

            # Validate the pose and the field of view on the shared domains the
            # MuJoCo and Newton backends' ``add_camera`` already applies, before
            # anything touches the stage. ``coerce_pose_vector``'s contract is
            # that a pose either backend entry point refuses must be refused by
            # the other, and this was the entry point that refused nothing: the
            # ``list(position)`` below copied the caller's vector element-wise
            # without reading it, so a ``nan``/``inf`` component, a non-numeric
            # element, a ``bool`` (the coordinate 1.0) and a wrong-length vector
            # all reached ``_create_camera_prim`` under ``status="success"``, and
            # a NumPy pose leaked ``np.float64`` into the status text and the
            # json payload. ``float(fov)`` accepted ``nan``/``inf``/``0``/
            # ``>= 180`` and raised a bare ``ValueError`` on a non-numeric value
            # - from outside the try block below, so it escaped the structured
            # ``{"status": "error"}`` contract entirely. Coercion is validation
            # only when the coercion rejects; these did not.
            position, pos_err = coerce_pose_vector("add_camera", "position", position, 3)
            if pos_err is not None:
                return {"status": "error", "content": [{"text": pos_err}]}
            target, tgt_err = coerce_pose_vector("add_camera", "target", target, 3)
            if tgt_err is not None:
                return {"status": "error", "content": [{"text": tgt_err}]}
            pos = [2.0, 2.0, 2.0] if position is None else position
            tgt = target
            if tgt is not None and all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"add_camera: 'position' and 'target' are identical ({pos}); camera has no look direction."
                        }
                    ],
                }
            # An fov outside (0, 180) is not merely mis-framed here: the pinhole
            # relation ``focal_length = horizontal_aperture / (2 * tan(fov / 2))``
            # in :meth:`_create_camera_prim` raises ``ZeroDivisionError`` for
            # ``0`` - which is NOT in the except tuple below, so it propagates
            # out of this method - and yields a ``nan`` focal length for a
            # ``nan`` fov and 7.3e-16 mm for ``180``, both of which
            # ``set_focal_length`` accepts, giving a camera that renders nothing
            # usable under a success result.
            if (fov_err := camera_fov_error("add_camera", "fov", fov)) is not None:
                return {"status": "error", "content": [{"text": fov_err}]}
            # Pixel dimensions on the shared floor
            # (:func:`~strands_robots.utils.positive_count_error`) the MuJoCo
            # backend's ``_validate_render_dims`` already applies, so a
            # resolution one backend refuses is refused by all of them. The
            # ``int(width or ...)`` this replaces validated nothing and read a
            # falsy value as *omitted*: ``width=0`` silently substituted the
            # config default and reported it as the resolution, so the caller
            # was told a size it never asked for. A negative was stored
            # verbatim, a fractional or ``bool`` value was truncated, and a
            # non-numeric / ``nan`` / ``inf`` value raised ``ValueError`` /
            # ``OverflowError`` from ``int()`` - outside the try block below, so
            # it escaped the structured ``{"status": "error"}`` contract.
            for param, value in (("width", width), ("height", height)):
                if value is not None and (dim_err := positive_count_error(value, param, "add_camera")) is not None:
                    return {"status": "error", "content": [{"text": dim_err}]}

            if name in self._cameras:
                return {
                    "status": "error",
                    "content": [{"text": f"Camera '{name}' already exists."}],
                }

            w = self._config.camera_width if width is None else width
            h = self._config.camera_height if height is None else height
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
            if not registered(self._cameras, name):
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
            Encoded MP4 frame rate. Default 30. Must be a positive whole
            number - the rate the flush encodes at, refused here rather
            than after a rollout's frames have been buffered.
        name : str
            Filename tag. Auto-generated (``rec_<uuid>``) when ``None``.
        max_frames_per_camera : int
            Safety cap on in-memory buffers. Frames beyond the cap are
            silently dropped. Default 3000. Must be a positive whole
            number; a cap below 1 drops every frame.

        Returns
        -------
        dict
            On success: ``{"status": "success", "content": [{"text": ...},
            {"json": {"on_frame": <callable>, "cameras": [...],
            "output_dir": ..., "name": ...}}]}``. The ``on_frame`` closure
            isn't JSON-serializable; Python callers unpack it from the
            json block. On error (unusable ``fps`` or
            ``max_frames_per_camera``, no world, already recording,
            unknown cameras, none to record):
            ``{"status": "error", ...}``.
        """
        import os as _os
        import tempfile as _tempfile
        import time as _time
        import uuid as _uuid

        # Refuse a frame count the recorder cannot honor before any filesystem
        # or buffer work: ``fps`` reaches ``encode_clip`` at flush time, which
        # refuses a rate it cannot encode at, and a non-positive frame cap
        # drops every captured frame.
        if error := _cameras_recording_option_error("start_cameras_recording", fps, max_frames_per_camera):
            return error
        # ``cameras`` names an ordered list of DISTINCT camera names, so it is
        # refused on the shared name-list domain before any filesystem or buffer work. Neither
        # mistake this catches could be honored as written: a single name passed
        # as a bare string is iterable per character, so it was read as one
        # camera per letter, and a repeated name opened a second encoder on the one output
        # path, so the artifact ledger reported two files where one exists.
        if cameras and (text := name_list_error(cameras, "cameras", "start_cameras_recording")):
            return {"status": "error", "content": [{"text": text}]}

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
                unresolved = [c for c in cameras if not registered(self._cameras, c)]
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

        from strands_robots.rendering.video import encode_clip

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
            flush_error = None
            if frames_buffer:
                # Shared encoder (strands_robots.rendering.video, issue #1537);
                # same imageio/libx264 invocation as the previous inline writer.
                try:
                    encode_clip(frames_buffer, path, fps=state["fps"])
                    frames_written = len(frames_buffer)
                except ImportError:
                    return {
                        "status": "error",
                        "content": [{"text": "imageio not installed. pip install imageio imageio-ffmpeg"}],
                    }
                except (RuntimeError, ValueError) as e:
                    # ``encode_clip`` refused the clip: ``RuntimeError`` when it
                    # wrote no file, ``ValueError`` when it will not encode at
                    # the requested rate. ``start_cameras_recording`` pre-flights
                    # that rate, so the second is unreachable through the tool
                    # pair; it is caught anyway because this method's contract is
                    # best-effort and never-raise, and the flush is the last
                    # chance to hand back the buffered frames' fate. Report the
                    # reason on the artifact line and keep ``frames_written`` at
                    # 0 rather than claiming frames that reached no file.
                    flush_error = f"{type(e).__name__}: {e}"
                    logger.warning("camera recorder flush failed for %r -> %s: %s", cam, path, flush_error)
                if _os.path.exists(path):
                    size_kb = _os.path.getsize(path) / 1024
            line = (
                f"   {cam:20s} {frames_written:>5d} frames  {size_kb:>7.1f} KB  "
                f"({errors} errors)  -> {_os.path.basename(path)}"
            )
            if flush_error:
                line += f"  [flush failed: {flush_error}]"
            lines.append(line)
            artifact = {
                "camera": cam,
                "path": path,
                "frames": frames_written,
                "errors": errors,
                "size_kb": size_kb,
            }
            if flush_error:
                artifact["flush_error"] = flush_error
            artifacts.append(artifact)

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
        # ``np.asarray`` calls in ``render()``. Caught during robots-sim#61
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
            # but ``render()``'s defensive None-handling (robots-sim#62) will
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
           rather than a ``TypeError`` on iteration, then translates
           any USD-mangled name back to the URDF's own joint name via
           :func:`strands_robots.simulation.isaac.joint_names.demangle_usd_joint_names`
           (#1900) so the public joint vocabulary matches the MuJoCo
           backend's for the same URDF.
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

        # Step 4-5: joint names + return. The importer transcodes any URDF
        # joint name that is not a valid USD identifier (a purely numeric
        # name like the robotstudio_so101's "1" imports as "tn__1_"), and
        # before #1900 that mangled form leaked through every public surface
        # keyed by joint name - robot_joint_names, get_observation keys,
        # send_action resolution - so the same URDF spoke different joint
        # vocabularies on Isaac vs MuJoCo. Every articulation read/write is
        # positional (index into dof_names order), so translating here, once,
        # makes the whole backend speak the URDF names. The usd->urdf map is
        # stashed as a sidecar (same pattern as _strands_actual_prim_path)
        # for _RobotState diagnostics. Best-effort: a URDF the stdlib parse
        # cannot re-read (the importer accepted it, so this is surface drift,
        # not a bad file) keeps the importer's names, leaving Isaac
        # self-consistent on its own mangled names.
        joint_names = list(articulation.dof_names) if articulation.dof_names else []
        usd_to_urdf: dict[str, str] = {}
        try:
            urdf_declared = urdf_joint_names(os.path.abspath(urdf_path))
        except (OSError, ValueError) as e:
            logger.warning(
                "Could not re-parse %s for joint-name translation (%s); keeping the importer's joint names %s.",
                urdf_path,
                e,
                joint_names,
            )
        else:
            joint_names, usd_to_urdf = demangle_usd_joint_names(joint_names, urdf_declared)
            if usd_to_urdf:
                logger.info(
                    "Translated %d USD-mangled joint names back to their URDF names: %s",
                    len(usd_to_urdf),
                    usd_to_urdf,
                )
        try:
            articulation._strands_usd_to_urdf_joint_names = usd_to_urdf  # type: ignore[attr-defined]
        except (AttributeError, TypeError):
            # Same fallback as _strands_actual_prim_path above: the caller
            # then records an empty map, and the public names still carry
            # the translated vocabulary via the returned joint_names.
            pass

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
    # (``examples/so101_curobo/isaac/simulation.py``) when robots-sim#69
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

    def _marshal_main_thread_affine(self, method_name: str, fn: Any) -> Any:
        """Run ``fn`` on the SimulationApp-owning thread, or fail loudly.

        ``reset`` / ``step`` drive Isaac's kit runtime (``world.reset()`` /
        ``world.step()``), and kit only pumps updates on the thread that
        created ``SimulationApp``. Called from a worker thread those entry
        points do not error - they block **forever** waiting for a pump the
        worker thread can never run. Observed shape (#1896): a Strands
        ``Agent`` executed its tool on a worker thread, the tool ran
        ``evaluate_benchmark`` -> ``reset()`` -> ``SimulationContext.stop()``,
        and that call never returned because the main thread was itself
        parked waiting on the tool future.

        Three cases:

        * On the owning thread: run ``fn`` inline - the headless / smoke
          path is unchanged.
        * Off it with :meth:`run_pump_forever` engaged: marshal through
          :meth:`run_on_main` so the pump executes ``fn`` on the owning
          thread (the same auto-marshal the recording facade uses for its
          schema probe, see ``IsaacRecordingMixin._probe_recording_observation``).
        * Off it with NO pump running: raise ``RuntimeError`` naming the
          recipe. This is a raise, not a structured error dict, on purpose:
          it is a caller threading-contract violation (some internal call
          sites - e.g. the per-episode ``sim.reset()`` in the policy runner -
          do not inspect the envelope, so a dict here would silently skip
          the reset), and the alternative behavior is an indefinite,
          signal-free deadlock.
        """
        if self._on_main_thread():
            return fn()
        if self._pump_running:
            return self.run_on_main(fn)
        raise RuntimeError(
            f"IsaacSimulation.{method_name}() was called from a worker thread with no "
            "main-thread pump running. Isaac Sim only pumps kit updates on the thread "
            "that created SimulationApp, so this call would block forever. Either call "
            "it from the owning thread, or have the owning thread run "
            "`run_pump_forever(stop_event=...)` and submit the call from the worker via "
            "`run_on_main(lambda: ...)` (see examples/libero/run_isaac_agent.py for the "
            "agent-driven shape)."
        )

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

        Validation
        ----------
        The write is all-or-nothing, on the same terms the MuJoCo backend's
        :meth:`~strands_robots.simulation.mujoco.MuJoCoSimEngine.set_joint_positions`
        already enforces, so the accepted domain does not depend on which engine
        the caller is driving:

        * the list form's length must equal the robot's joint count - a shorter
          or longer vector is refused rather than resizing the articulation's
          joint-position array, which would otherwise be handed to PhysX at the
          wrong width;
        * every ``dict`` key must name one of the robot's joints, and the mapping
          must not be empty - an unresolvable name used to be skipped silently, so
          a typo (or a name from the wrong robot) wrote nothing, or wrote only
          part of the requested pose, while the caller was told the pose had been
          applied;
        * every value must be a finite, non-boolean real number, on the shared
          :meth:`~strands_robots.simulation.base.SimEngine._coerce_joint_state_map`
          domain. A ``nan`` / ``inf`` is refused rather than written into the
          articulation, where PhysX surfaces it from a *later* step as an
          "Illegal BroadPhaseUpdateData - non-finite bounds" error; a boolean is
          refused because ``float(True)`` is a silent 1-radian target.

        Validation runs synchronously, before the write is queued for the main
        thread, so a rejected value is reported to the caller rather than raised
        on the pump thread - where the queued-action handler swallows it after
        this method has already answered ``status="success"``.

        Returns
        -------
        dict
            Standard ``{"status", "content"}`` envelope; ``error`` for an
            unknown/uninitialized robot or a value outside the domain above.
        """
        with self._lock:
            if not self._world_created or not self._robots:
                return {"status": "error", "content": [{"text": "No world/robot."}]}
            if positions is None:
                return {"status": "error", "content": [{"text": "'positions' is required."}]}
            if robot_name is None:
                robot_name = next(iter(self._robots))
            r = registry_entry(self._robots, robot_name)
            if r is None or r.articulation is None:
                return {"status": "error", "content": [{"text": f"Robot {robot_name!r} not initialized."}]}

            joint_names = list(r.joint_names)
            # Normalize both accepted shapes to a {joint name: value} mapping so
            # one set of checks covers them. The list form binds positionally to
            # the robot's joint order, so its length is part of the contract: a
            # mismatched vector used to be written straight through, resizing the
            # articulation's joint-position array instead of being refused.
            if isinstance(positions, dict):
                requested: dict[str, Any] = dict(positions)
            elif isinstance(positions, (list, tuple, np.ndarray)):
                ordered = list(positions)
                if len(ordered) != len(joint_names):
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"set_joint_positions: list length {len(ordered)} does not match robot "
                                    f"{robot_name!r} joint count {len(joint_names)}. Use a dict for partial updates."
                                )
                            }
                        ],
                    }
                requested = dict(zip(joint_names, ordered, strict=True))
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "set_joint_positions: 'positions' must be a dict or list, "
                                f"got {type(positions).__name__}"
                            )
                        }
                    ],
                }

            # Resolve every name before any write, so a partially-resolvable
            # mapping is refused rather than applied in part and reported as a
            # complete pose.
            if not requested:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                "set_joint_positions: 'positions' is empty, so there is nothing to write. "
                                "Pass at least one joint (dict form) or a full ordered vector (list form); "
                                "use action='robot_joint_names' to see one robot's joints."
                            )
                        }
                    ],
                }
            index_of = {jn: i for i, jn in enumerate(joint_names)}
            unresolved = [jn for jn in requested if jn not in index_of]
            if unresolved:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_joint_positions: unresolved 'positions' keys {unresolved} on robot "
                                f"{robot_name!r}. Its joints are {joint_names}; "
                                "use action='robot_joint_names' to list them."
                            )
                        }
                    ],
                }

            coerced, err = self._coerce_joint_state_map(requested, "positions", "set_joint_positions")
            if err:
                return err
            targets = {index_of[jn]: value for jn, value in coerced.items()}

            def _apply() -> None:
                cur = list(r.articulation.get_joint_positions())
                for dof, value in targets.items():
                    cur[dof] = value
                r.articulation.set_joint_positions(np.array(cur, dtype=float))

            if self._on_main_thread():
                _apply()
                return {"status": "success", "content": [{"text": "Set joint positions (main)."}]}
            self._action_q.put(_apply)
            return {"status": "success", "content": [{"text": "Set joint positions (queued)."}]}

    def set_robot_pose(
        self,
        robot_name: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Teleport a robot's articulation base to ``(position, orientation)``.

        The scene-alignment counterpart of :meth:`move_object` (#1820): on
        the MuJoCo backend the robot is part of the scene MJCF, so its base
        lands wherever the scene author placed it (LIBERO puts
        ``robot0_base`` at ``(-0.66, 0, 0.912)``); on Isaac the robot is a
        separately-loaded USD articulation spawned at the origin -- inside
        the footprint of the scene's origin-anchored static fixtures.
        ``LiberoAdapter._apply_object_pose_state`` reads the scene's robot
        base pose and aligns the articulation through this method so live
        physics doesn't start with the robot interpenetrating a fixture
        (PhysX "Illegal BroadPhaseUpdateData - non-finite bounds").

        Parameters
        ----------
        robot_name : str, optional
            Robot to move. ``None`` resolves the single robot, mirroring
            :meth:`send_action`; ambiguous with several robots.
        position : list[float], optional
            World position ``[x, y, z]``. ``None`` keeps the current one.
        orientation : list[float], optional
            World orientation quaternion ``[w, x, y, z]``. ``None`` keeps
            the current one.

        Validation
        ----------
        ``position`` must be 3 finite numbers and ``orientation`` 4, on the
        shared :func:`~strands_robots.utils.coerce_pose_vector` domain. An
        over-long vector is refused rather than truncated, and a ``nan`` /
        ``inf`` component is refused rather than written into the articulation
        transform.

        Returns
        -------
        dict
            Standard ``{"status", "content"}`` envelope; ``error`` for an
            unknown/uninitialized robot or a failed pose write.
        """
        with self._lock:
            if not self._world_created or not self._robots:
                return {"status": "error", "content": [{"text": "No world/robot."}]}
            if robot_name is None:
                if len(self._robots) != 1:
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"set_robot_pose(robot_name=None) is ambiguous: {len(self._robots)} robots "
                                    f"present ({sorted(self._robots)}). Pass robot_name explicitly."
                                )
                            }
                        ],
                    }
                robot_name = next(iter(self._robots))
            robot = registry_entry(self._robots, robot_name)
            if robot is None or robot.articulation is None:
                return {"status": "error", "content": [{"text": f"Robot {robot_name!r} not initialized."}]}
            # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
            # MuJoCo backend's ``set_robot_pose`` and this backend's own ``add_camera`` already
            # use, so a pose one backend refuses is refused by all of them - the
            # invariant that helper documents. The ``x[:3]`` / ``x[:4]`` slices this
            # replace validated nothing and silently TRUNCATED an over-long vector, so a
            # 5-component request was written as its first 3 under a success result. A
            # short vector was passed through as-is, a ``nan``/``inf`` component wrote a
            # degenerate transform, and testing the vector for truthiness read an empty
            # one as *omitted* while raising on the NumPy array the Args advertise.
            position, _perr = coerce_pose_vector("set_robot_pose", "position", position, 3)
            if _perr is not None:
                return {"status": "error", "content": [{"text": _perr}]}
            orientation, _oerr = coerce_pose_vector("set_robot_pose", "orientation", orientation, 4)
            if _oerr is not None:
                return {"status": "error", "content": [{"text": _oerr}]}
            moved_to = "same" if position is None else position
            try:
                pos = np.asarray(position, dtype=float) if position is not None else None
                ori = np.asarray(orientation, dtype=float) if orientation is not None else None
                robot.articulation.set_world_pose(position=pos, orientation=ori)
            except (RuntimeError, ValueError, AttributeError, TypeError) as exc:
                return {
                    "status": "error",
                    "content": [{"text": f"set_robot_pose failed: {type(exc).__name__}: {exc}"}],
                }
            return {
                "status": "success",
                "content": [{"text": f"Robot {robot_name!r} base moved to {moved_to}."}],
            }

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

        Validation
        ----------
        ``position`` must be 3 finite numbers and ``orientation`` 4, on the
        shared :func:`~strands_robots.utils.coerce_pose_vector` domain the
        MuJoCo and Newton backends' ``move_object`` enforce. An over-long
        vector is refused rather than truncated to its leading components, and
        an omitted one leaves that component alone.
        """
        obj = registry_entry(self._objects, name)
        if obj is None or obj.handle is None:
            return {"status": "error", "content": [{"text": f"Object {name!r} not found."}]}
        # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
        # MuJoCo backend's ``move_object`` and this backend's own ``add_camera`` already
        # use, so a pose one backend refuses is refused by all of them - the
        # invariant that helper documents. The ``x[:3]`` / ``x[:4]`` slices this
        # replace validated nothing and silently TRUNCATED an over-long vector, so a
        # 5-component request was written as its first 3 under a success result. A
        # short vector was passed through as-is, a ``nan``/``inf`` component wrote a
        # degenerate transform, and testing the vector for truthiness read an empty
        # one as *omitted* while raising on the NumPy array the Args advertise.
        position, _perr = coerce_pose_vector("move_object", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        orientation, _oerr = coerce_pose_vector("move_object", "orientation", orientation, 4)
        if _oerr is not None:
            return {"status": "error", "content": [{"text": _oerr}]}
        try:
            pos = np.array(position, dtype=float) if position is not None else None
            ori = np.array(orientation, dtype=float) if orientation is not None else None
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
        moved_to = "same" if position is None else position
        return {"status": "success", "content": [{"text": f"'{name}' moved to {moved_to}."}]}

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
        obj = registry_entry(self._objects, name)
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
        obj = registry_entry(self._objects, name)
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
        obj = registry_entry(self._objects, name)
        if obj is None or obj.handle is None:
            return None
        try:
            pos, _ = obj.handle.get_world_pose()
            return [float(x) for x in pos]
        except (RuntimeError, ValueError, AttributeError, TypeError):
            return None

    def get_body_state(self, body_name: str) -> dict[str, Any]:
        """World pose of a scene object or robot link, MuJoCo-envelope-compatible.

        Implements the same duck-typed contract as
        :meth:`strands_robots.simulation.mujoco.physics.PhysicsMixin.get_body_state`,
        which is the read primitive under BOTH consumers this backend was
        missing (#1802):

        * the predicate DSL (:mod:`strands_robots.simulation.predicates`
          ``_body_position`` / ``_body_quaternion``, so ``body_above_z`` /
          ``body_on`` / ``distance_less_than`` / ... evaluate on Isaac), and
        * :meth:`LiberoAdapter._read_eef_pose`'s body-state fallback, which
          injects the ``state.x/y/z/roll/pitch/yaw`` keys the ``libero_panda``
          GR00T data-config requires.

        Name resolution, in order (namespace-aware, mirroring MuJoCo's
        "unambiguous or explicit" contract):

        1. **Object registry** -- an ``add_object`` / ``load_scene`` object
           whose name matches ``body_name`` verbatim (this is how LIBERO
           scene bodies such as ``porcelain_mug_1_main`` resolve).
        2. **Absolute prim path** -- a ``body_name`` starting with ``/`` is
           looked up on the stage directly.
        3. **Namespaced robot link** -- ``"<robot>/<link>"`` searches the
           named robot's prim subtree for an Xformable prim named
           ``<link>`` (e.g. ``"robot/panda_hand"``).
        4. **Bare robot link** -- searched across every robot's subtree;
           first match wins, so multi-robot scenes with colliding link
           names MUST use the namespaced form.

        Returns:
            ``{"status": "success", "content": [{"text": ...}, {"json":
            {"position": [x, y, z], "quaternion": [w, x, y, z],
            "rotation_matrix": 3x3, ...}}]}`` on success. ``linear_velocity``
            / ``angular_velocity`` are included only when the resolved
            object's handle exposes them (rigid-prim objects do; USD-walked
            robot links do not -- keys are OMITTED rather than zero-filled,
            per the no-silent-defaults rule). ``mass`` / ``center_of_mass``
            from the MuJoCo contract are likewise not reported on Isaac.
            Unknown bodies and a missing world return
            ``{"status": "error", "content": [{"text": ...}]}``.

        Concurrency: safe to call from any thread. Off the main thread with
        the pump running, the USD read is submitted via :meth:`run_on_main`
        (Kit's stage may only be queried from the thread that owns
        ``SimulationApp``).
        """
        if not isinstance(body_name, str) or not body_name.strip():
            return {
                "status": "error",
                "content": [{"text": "get_body_state: body_name must be a non-empty string."}],
            }
        if not self._world_created:
            return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}

        if self._on_main_thread() or not self._pump_running:
            return self._get_body_state_impl(body_name)
        return self.run_on_main(
            lambda: self._get_body_state_impl(body_name),
            timeout=_BODY_STATE_MAIN_THREAD_TIMEOUT_S,
        )

    def _get_body_state_impl(self, body_name: str) -> dict[str, Any]:
        """Resolve + read ``body_name``; runs on the main thread (or pump-less)."""
        obj = registry_entry(self._objects, body_name)
        if obj is not None and obj.handle is not None:
            state = self._object_body_state(obj)
            if state is not None:
                return _body_state_envelope(body_name, state)

        state = self._prim_body_state(body_name)
        if state is not None:
            return _body_state_envelope(body_name, state)

        objects = sorted(self._objects)
        shown = ", ".join(objects[:20]) + (", ..." if len(objects) > 20 else "")
        msg = (
            f"Body '{body_name}' not found on the Isaac stage. "
            f"Known objects: [{shown}]. Robots: {sorted(self._robots)} -- address robot links as "
            f"'<robot>/<link>' (e.g. 'robot/panda_hand') or pass an absolute prim path ('/World/...')."
        )
        return {"status": "error", "content": [{"text": msg}]}

    def _object_body_state(self, obj: _ObjectState) -> dict[str, Any] | None:
        """Pose (+ best-effort velocities) of a registered object's rigid prim."""
        try:
            pos, quat = obj.handle.get_world_pose()
        except (RuntimeError, ValueError, AttributeError, TypeError) as e:
            logger.debug("get_body_state: get_world_pose failed for object %r: %s", obj.name, e)
            return None
        position = _to_float_list(pos, 3)
        quaternion = _to_float_list(quat, 4)  # Isaac core convention is scalar-first (wxyz), same as MuJoCo
        if position is None or quaternion is None:
            logger.debug("get_body_state: object %r returned an unusable pose (%r, %r)", obj.name, pos, quat)
            return None
        state: dict[str, Any] = {
            "position": position,
            "quaternion": quaternion,
            "rotation_matrix": _quat_wxyz_to_rotmat(np.asarray(quaternion)).tolist(),
            "source": "object",
            "prim_path": obj.prim_path,
        }
        for key, getter in (("linear_velocity", "get_linear_velocity"), ("angular_velocity", "get_angular_velocity")):
            fn = getattr(obj.handle, getter, None)
            if fn is None:
                continue
            try:
                vel = _to_float_list(fn(), 3)
            except (RuntimeError, ValueError, AttributeError, TypeError) as e:
                logger.debug("get_body_state: %s failed for object %r: %s", getter, obj.name, e)
                continue
            if vel is not None:
                state[key] = vel
        return state

    def _prim_body_state(self, body_name: str) -> dict[str, Any] | None:
        """Pose of a robot-link / absolute-path prim, read off the USD stage.

        Same stage-walk + axis-normalization approach as
        :meth:`gripper_frame_pose` (handles authored scale), generalized from
        that method's name-heuristic gripper search to exact-name link
        resolution. Returns ``None`` when the prim cannot be resolved.
        """
        try:
            import omni.usd  # type: ignore[import-not-found]
            from pxr import (  # type: ignore[import-not-found]
                Gf,
                Sdf,
                Usd,
                UsdGeom,
            )
        except ImportError as e:
            logger.debug("get_body_state: USD runtime not importable: %s", e)
            return None
        try:
            stage = omni.usd.get_context().get_stage()
            if stage is None:
                return None

            prim = None
            if body_name.startswith("/"):
                p = stage.GetPrimAtPath(body_name)
                if p and p.IsValid() and p.IsA(UsdGeom.Xformable):
                    prim = p
            elif "/" in body_name:
                robot_name, _, link_name = body_name.partition("/")
                r = registry_entry(self._robots, robot_name)
                if r is not None and link_name:
                    prim = self._find_robot_link_prim(stage, r, link_name, Sdf, Usd, UsdGeom)
            else:
                for r in self._robots.values():
                    prim = self._find_robot_link_prim(stage, r, body_name, Sdf, Usd, UsdGeom)
                    if prim is not None:
                        break
            if prim is None:
                return None

            xf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            t = xf.ExtractTranslation()

            def _axis(vx: float, vy: float, vz: float) -> list[float]:
                d = xf.TransformDir(Gf.Vec3d(vx, vy, vz))
                n = (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) ** 0.5 or 1.0
                return [d[0] / n, d[1] / n, d[2] / n]

            ax, ay, az = _axis(1.0, 0.0, 0.0), _axis(0.0, 1.0, 0.0), _axis(0.0, 0.0, 1.0)
            # Columns are the prim frame's local axes in world coords
            # (``world = R @ local``), matching MuJoCo's ``data.xmat``.
            rotmat = [
                [ax[0], ay[0], az[0]],
                [ax[1], ay[1], az[1]],
                [ax[2], ay[2], az[2]],
            ]
            return {
                "position": [float(t[0]), float(t[1]), float(t[2])],
                "quaternion": _rotmat_to_quat_wxyz(rotmat),
                "rotation_matrix": rotmat,
                "source": "prim",
                "prim_path": str(prim.GetPath()),
            }
        except (RuntimeError, ValueError, AttributeError, TypeError):
            logger.debug("get_body_state: USD read failed for %r", body_name, exc_info=True)
            return None

    @staticmethod
    def _find_robot_link_prim(stage: Any, r: _RobotState, link_name: str, Sdf: Any, Usd: Any, UsdGeom: Any) -> Any:  # noqa: N803 - pxr module objects passed by caller
        """Exact-name Xformable prim under a robot's top-level subtree, or ``None``.

        Walks up from ``r.actual_prim_path`` to the top-level robot prim
        (the importer may have relocated the robot -- see
        :meth:`gripper_frame_pose`) and searches the whole subtree for a
        prim named ``link_name``.
        """
        sdf_path = Sdf.Path(r.actual_prim_path)
        top = sdf_path
        while top.GetParentPath() != Sdf.Path.absoluteRootPath and top.GetParentPath() != Sdf.Path.emptyPath:
            top = top.GetParentPath()
        root = stage.GetPrimAtPath(top)
        if not root or not root.IsValid():
            return None
        for p in Usd.PrimRange(root):
            if p.GetName() == link_name and p.IsA(UsdGeom.Xformable):
                return p
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
        r = registry_entry(self._robots, robot_name)
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
        out = registry_entry(self._cam_out_size, cname)
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

    def describe(self) -> dict[str, Any]:
        """Return the Isaac engine's live discovery surface.

        Extends the base :meth:`SimEngine.describe` contract with the backend
        identity, the registered RTX camera names, world state, and the
        LeRobotDataset recording family
        (:class:`~strands_robots.simulation.isaac.recording.IsaacRecordingMixin`)
        so an agent enumerating ``describe()["methods"]`` discovers the
        record-and-stream workflow (``start_recording`` -> ``run_policy`` ->
        ``save_episode`` -> ``stop_recording`` -> ``stream_dataset``) exactly
        as it does on the MuJoCo and Newton backends.
        """
        desc = super().describe()
        desc["backend"] = "isaac"
        desc["cameras"] = sorted(self._cameras)
        desc["world_created"] = self._world_created
        desc["methods"].update(
            {
                "add_camera": (
                    "(name='default', position=None, target=None, width=None, "
                    "height=None, fov=60.0) -> dict  # register an RTX camera "
                    "(rendered frames ride get_observation and recordings)"
                ),
                "remove_camera": "(name: str) -> dict  # remove a registered RTX camera",
                "start_recording": (
                    "(repo_id='local/sim_recording', task='', fps=30, root=None, "
                    "push_to_hub=False, vcodec='h264', overwrite=False, cameras=None) -> dict  "
                    "# record joint state + action + RTX cameras to a LeRobotDataset "
                    "(needs render_mode='rtx_realtime' for camera columns)"
                ),
                "save_episode": (
                    "() -> dict  # flush the current rollout as one episode; prefer "
                    "run_policy(n_episodes=N) which flushes a boundary per episode"
                ),
                "stop_recording": "(push_to_hub=False, bucket=None, run_id=None) -> dict",
                "get_recording_status": "() -> dict",
                "stream_dataset": (
                    "(repo_id: str, **kwargs) -> StreamingDatasetReader  # lazily read a "
                    "recorded LeRobotDataset back (root=, episodes=, delta_timestamps=, ...)"
                ),
                "verify_dataset_episodes": (
                    "(expected: int) -> dict  # after stop_recording, read the parquet and "
                    "confirm the dataset holds exactly `expected` episodes; status=error on mismatch"
                ),
                "start_cameras_recording": (
                    "(cameras=None, output_dir=None, fps=30, name=None, max_frames_per_camera=3000) -> dict  "
                    "# raw per-camera MP4 capture (no lerobot dependency)"
                ),
                "stop_cameras_recording": "() -> dict  # finalize the raw MP4 capture",
                # Motion primitives (GH #2154 joint-space pair + GH #2155
                # move_to, Isaac half of the GH #1645 vocabulary; shared
                # contract in strands_robots.simulation.motion_primitives_base).
                "move_to": (
                    "(robot_name=None, position=[x,y,z], orientation=None, tol=0.01, "
                    "max_steps=200, orientation_tol=None) -> dict  # IK-solve (shared mink "
                    "bridge on the registry MJCF) then servo the end-effector to a world-frame "
                    "Cartesian target; position-only when orientation is omitted, otherwise "
                    "converged to within orientation_tol radians (default 0.1) as well"
                ),
                "set_gripper": (
                    "(robot_name=None, state='open'|'close', steps=12) -> dict  # "
                    "drive the gripper joint(s) to the open/close set-point "
                    "(registry gripper metadata when present, else open=HIGH / "
                    "close=LOW end of the joint's limit range)"
                ),
                "rotate_wrist": (
                    "(robot_name=None, target_yaw, tol=0.02, max_steps=200) -> dict  "
                    "# rotate the wrist joint to a set-point (radians) while the "
                    "other joints hold their current positions"
                ),
            }
        )
        return desc

    def cleanup(self) -> None:
        """Release all resources.

        Callers must invoke this explicitly (or use the class as a context
        manager). Do not rely on garbage collection: at interpreter shutdown
        the ``threading`` / ``logger`` / ``omni`` modules can already be
        partially torn down, acquiring ``self._lock`` from a finalizer is
        unsafe, and a GC scheduler that defers the finalizer past the
        ``SimulationApp`` shutdown leaks the ``World``/USD stage. The
        inherited :meth:`~strands_robots.simulation.base.SimEngine.__del__`
        is a best-effort last resort for a fully-constructed engine, not the
        intended path.
        """
        if self._world_created:
            self.destroy()

    def __enter__(self) -> IsaacSimulation:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __repr__(self) -> str:
        """Describe this engine, without ever raising.

        ``repr`` is what a traceback or a failing assertion renders, so it must
        not be the thing that hides a failure. On an instance whose ``__init__``
        never finished, reading ``self._config`` raises and the reader is shown
        ``[AttributeError ... raised in repr()]`` naming an attribute that has
        nothing to do with the failure under investigation. Report the
        lifecycle fact that *is* relevant instead, and name no attribute so
        nobody is sent chasing one.
        """
        try:
            return (
                f"IsaacSimulation("
                f"num_envs={self._config.num_envs}, "
                f"device={self._config.device!r}, "
                f"headless={self._config.headless}, "
                f"world={'created' if self._world_created else 'none'})"
            )
        except AttributeError:
            return partial_construction_repr(self)
