"""Newton GPU-native simulation backend.

Implements :class:`~strands_robots.simulation.base.SimEngine` on top of
newton-physics/newton (NVIDIA Warp + MuJoCo-Warp). Newton ingests the same
MJCF assets the MuJoCo backend uses (resolved through
:mod:`strands_robots.assets`), builds a GPU model, and steps it with any of
Newton's rigid-body solvers. Rendering is done headlessly via Newton's
ray-traced ``SensorTiledCamera`` so it needs no display server.

The backend reuses the backend-agnostic policy orchestration
(``run_policy`` / ``eval_policy`` / ``replay_episode``) provided by the ABC -
it only implements the abstract physics primitives plus ``render``.

Lifecycle::

    from strands_robots.simulation import create_simulation

    sim = create_simulation("newton", solver="mujoco")
    sim.create_world()
    sim.add_robot("so100")
    sim.send_action({"Rotation": 0.5}, robot_name="so100")
    out = sim.render(width=320, height=240)   # out["image"] -> (H, W, 3) uint8
    sim.destroy()
"""

from __future__ import annotations

import logging
import math
import numbers
import os
import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from strands_robots.assets import resolve_model_path, resolve_robot_name
from strands_robots.registry.discovery import discover_urdf_path, list_urdf_discoverable
from strands_robots.simulation.base import SimEngine, reject_setup_kwargs
from strands_robots.simulation.ik import hint_matches_name
from strands_robots.simulation.model_registry import (
    list_available_models,
    resolve_model,
)
from strands_robots.simulation.model_registry import (
    register_urdf as _register_urdf,
)
from strands_robots.simulation.models import (
    SimCamera,
    SimObject,
    SimRobot,
    SimWorld,
    registered,
    registry_entry,
)
from strands_robots.simulation.newton.backend import (
    articulated_solver_error,
    articulated_solvers,
    ensure_newton,
    resolve_solver_class,
    solver_registry,
)
from strands_robots.simulation.newton.randomization import DomainRandomizationMixin
from strands_robots.simulation.newton.recording import NewtonRecordingMixin
from strands_robots.simulation.terrain import validate_difficulty
from strands_robots.utils import (
    FREE_CAMERA_TOKENS,
    camera_fov_error,
    coerce_pose_vector,
    coerce_rgba,
    coerce_size_vector,
    entity_name_error,
    is_boolean,
    non_negative_whole_number_error,
    positive_count_error,
    positive_whole_number_error,
    require_optional,
    reserved_camera_name_error,
    step_aborted_msg,
    tcp_port_error,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from strands_robots.rendering import CameraParams

logger = logging.getLogger(__name__)

# Newton's default control rate. MJCF position actuators settle within a few
# hundred substeps; 60 Hz frames with 10 substeps each matches the Newton
# example cadence and keeps position-servo arms tracking their targets.
_DEFAULT_TIMESTEP = 1.0 / 600.0

# Hint words for the best-guess gripper/EEF mount ``list_bodies`` advertises.
# Newton adds "jaw" (its own MJCF vocabulary) to the MuJoCo backend's set.
# Matched on word boundaries by
# :func:`~strands_robots.simulation.ik.hint_matches_name` - the same rule
# :func:`~strands_robots.simulation.ik.discover_ee_frame` applies - so the short
# hint "ee" cannot fire inside "knee" or "wheel".
_GRIPPER_BODY_HINTS = ("gripper", "hand", "jaw", "ee", "tool")

# Valid ``add_robot(source=...)`` selectors. ``None``/``"registry"`` resolve
# the curated registry + MJCF asset manager (the same path the MuJoCo backend
# uses); ``"robot_descriptions"`` resolves a URDF directly from the
# ``robot_descriptions`` package and loads it through Newton's native URDF
# importer. ``None`` additionally falls back to ``robot_descriptions`` when the
# registry has no asset, so the URDF-only long tail resolves without an
# explicit selector.
_ROBOT_SOURCES = (None, "registry", "robot_descriptions")


def _short_joint_name(label: str) -> str:
    """Reduce a hierarchical Newton joint label to its short joint name.

    Newton labels joints by their full body path
    (``so_arm100/worldbody/Base/.../Rotation``); the public observation /
    action schema uses the short trailing segment (``Rotation``) to match the
    MuJoCo backend. The same short name therefore maps to the same joint
    across both backends.

    Args:
        label: Full Newton joint label.

    Returns:
        The trailing path segment.
    """
    return label.rsplit("/", 1)[-1]


def _quat_rotate_inverse_wxyz(quat_wxyz: list[float], vec: list[float]) -> list[float]:
    """Express a WORLD-frame 3-vector in the body frame given a (w,x,y,z) quaternion.

    Computes ``R(q)^T @ vec`` (the standard "rotate by the inverse"), used to
    turn Newton's world-frame free-joint angular velocity into the BODY frame so
    ``base_ang_vel`` matches the MuJoCo backend and the IMU-gyro convention WBC /
    locomotion controllers consume. The quaternion is normalised internally; a
    ~zero-norm quaternion returns ``vec`` unchanged.
    """
    q = np.asarray(quat_wxyz, dtype=np.float64)
    norm = float(np.linalg.norm(q))
    if norm < 1e-8:
        return [float(v) for v in vec]
    w, x, y, z = q / norm
    v = np.asarray(vec, dtype=np.float64)
    q_vec = np.array([x, y, z], dtype=np.float64)
    a = v * (2.0 * w * w - 1.0)
    b = np.cross(q_vec, v) * (w * 2.0)
    c = q_vec * (float(np.dot(q_vec, v)) * 2.0)
    return [float(t) for t in (a - b + c)]


def _is_zero_mass_sentinel(mass: Any) -> bool:
    """Whether ``mass`` is Newton's documented "make it static" spelling.

    ``add_object`` documents ``mass=0`` as an alternative spelling of
    ``is_static=True``, and :meth:`NewtonSimEngine._add_object_to_builder`
    honours it by taking the static path. A zero mass is therefore a *mode*
    rather than a small mass, so it is not validated as a dynamic one - which is
    the single place this backend's accepted mass domain differs from the
    MuJoCo backend's.

    ``bool`` is excluded even though ``False == 0`` is true: a boolean is
    refused on every backend rather than being read as a spelling of the
    sentinel, since ``float(True)`` would otherwise mean a silent 1 kg body.

    Args:
        mass: The caller-supplied value.

    Returns:
        ``True`` only for a real, non-``bool`` number equal to zero.
    """
    return isinstance(mass, numbers.Real) and not is_boolean(mass) and float(mass) == 0.0


class NewtonSimEngine(DomainRandomizationMixin, NewtonRecordingMixin, SimEngine):
    """GPU-native simulation backend built on Newton (Warp / MuJoCo-Warp).

    One Newton model per instance. The world is rebuilt whenever robots or
    objects are added or removed, because Newton finalises an immutable
    ``Model`` from a ``ModelBuilder``. State is preserved across rebuilds
    where joint names still exist.

    Thread-safety: a single ``RLock`` serialises all model/state mutation so a
    ``PolicyRunner`` worker and the calling thread never race on Warp arrays.
    """

    def __init__(
        self,
        solver: str = "mujoco",
        default_timestep: float = _DEFAULT_TIMESTEP,
        substeps: int = 10,
        device: str | None = None,
        default_width: int = 640,
        default_height: int = 480,
        **kwargs: Any,
    ) -> None:
        """Construct a Newton simulation engine.

        Args:
            solver: Friendly solver name. One of
                :func:`~strands_robots.simulation.newton.backend.articulated_solvers`
                (default ``"mujoco"``, i.e. MuJoCo-Warp).
            default_timestep: Physics integration timestep in seconds.
            substeps: Physics substeps per :meth:`step` call.
            device: Warp device string (e.g. ``"cuda:0"`` or ``"cpu"``).
                ``None`` selects Warp's default device (GPU when available).
            default_width: Default render width in pixels.
            default_height: Default render height in pixels.
            **kwargs: Ignored; accepted for forward compatibility. Robot-setup
                arguments (``robot_name`` / ``robot``) are rejected rather than
                dropped - use ``Robot("so101", mode="sim")`` or ``add_robot``.

        Raises:
            ValueError: If ``solver`` is not a known solver name.
        """
        reject_setup_kwargs(kwargs)
        super().__init__()
        # State that teardown (destroy/cleanup/__del__) touches must be set
        # before any fallible construction step. Otherwise a partially built
        # engine -- e.g. ensure_newton() raising when warp is absent, or an
        # unknown solver -- leaves __del__ to acquire self._lock during GC and
        # raise AttributeError, masking the real construction error with log
        # noise. Initialising the lock and viewer handles first keeps destroy()
        # a clean no-op on a half-constructed instance.
        self._lock = threading.RLock()
        self._world: SimWorld | None = None
        # Interactive viewer handle (newton.viewer.ViewerGL/ViewerViser/
        # ViewerNull). None until open_viewer() is called; synced once per
        # control step from _advance() on the stepping thread.
        self._viewer: Any = None
        self._viewer_kind: str | None = None

        self._nt, self._wp = ensure_newton()
        if solver.lower() not in solver_registry():
            raise ValueError(f"Unknown Newton solver {solver!r}. Available: {sorted(solver_registry())}")
        # A solver Newton resolves is not automatically one that can drive a
        # robot. Reported here, where the caller names it, so the refusal
        # arrives before any world, model or solver is built rather than as a
        # Newton-internal error or a frozen world behind a successful add_robot.
        if solver_error := articulated_solver_error(solver):
            raise ValueError(solver_error)
        self._solver_name = solver.lower()
        self.default_timestep = default_timestep
        self.substeps = substeps
        self.device = device
        self.default_width = default_width
        self.default_height = default_height

        # Newton handles (rebuilt on every scene mutation via _rebuild).
        self._model: Any = None
        self._solver: Any = None
        self._state_0: Any = None
        self._state_1: Any = None
        self._control: Any = None
        # Ordered short joint names and their DOF indices in joint_q / target.
        self._joint_order: list[str] = []
        # Pending position targets keyed by (robot_name, short joint name).
        self._targets: dict[tuple[str, str], float] = {}
        # Coordinate index of each (robot, joint) in the global joint_q /
        # joint_target_q vector. For the revolute/prismatic joints of robot
        # arms one coordinate maps to one DOF, so this also indexes targets.
        self._joint_coord_index: dict[tuple[str, str], int] = {}
        # Short joint names per robot (rebuilt with the model).
        self._robot_joint_map: dict[str, list[str]] = {}
        # Coordinate-to-DOF index of each (robot, joint) in joint_qd (velocity).
        # Distinct from _joint_coord_index because free joints have more
        # coordinates (quaternion) than DOFs; arm joints are 1:1.
        self._joint_dof_index: dict[tuple[str, str], int] = {}
        # Short name of each robot's floating-base free joint (a humanoid's
        # named ``floating_base_joint``), when it has one. Used to surface the
        # 6-DoF base pose/twist as the structured base_* keys and to EXCLUDE the
        # free joint from the scalar joint state (its qpos is [xyz+quat], not a
        # single angle) - matching get_robot_state and the MuJoCo backend.
        self._robot_free_base_joint: dict[str, str] = {}
        # Ordered full body labels per robot (rebuilt with the model).
        self._robot_body_map: dict[str, list[str]] = {}
        # Parsed mesh geometry keyed by resolved mesh_path, so rebuilds do not
        # re-read mesh assets off disk on every scene mutation.
        self._mesh_cache: dict[str, tuple[Any, Any]] = {}
        # Domain-randomization spec + applied multipliers (see
        # strands_robots.simulation.newton.randomization). None until
        # randomize() is called; applied during _rebuild.
        self._dr: dict[str, Any] | None = None
        self._dr_applied: dict[str, Any] | None = None
        self._dr_light_dir: tuple[float, float, float] | None = None
        # Additive sensor-noise config + reproducible RNG (set_obs_noise).
        self._obs_noise: dict[str, float] | None = None
        self._obs_noise_rng: np.random.Generator | None = None

        logger.info("Newton simulation engine initialised (solver=%s)", self._solver_name)

        # Construction complete - the finalizer may now release what we hold.
        # See SimEngine._init_complete: this must be the final statement.
        self._init_complete = True

    # World lifecycle

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        """Create an empty Newton world.

        Args:
            timestep: Physics timestep in seconds (defaults to the engine's
                ``default_timestep``).
            gravity: Gravity vector ``[x, y, z]`` (default ``[0, 0, -9.81]``).
            ground_plane: Whether to add a ground plane.
            terrain: Heightfield terrain kind (e.g. ``"rough"``/``"stairs"``/``"pyramid"``/``"slope"``,
                MuJoCo backend only). The Newton backend has no heightfield
                ground yet, so a non-None value is rejected with an actionable
                error.
            difficulty: Terrain curriculum elevation scale (MuJoCo backend
                only, alongside ``terrain``); accepted for signature parity with
                the base contract. Since Newton has no heightfield terrain,
                ``difficulty`` can never take effect, so a non-default
                (``!= 1.0``) value is rejected with an actionable error (the base
                ``create_world`` contract: reject rather than silently ignore it)
                rather than silently having no effect.

        Returns:
            Status dict with a human-readable confirmation.
        """
        if terrain is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"terrain={terrain!r} is not supported on the Newton backend "
                            "(heightfield terrain, e.g. 'rough'/'stairs'/'pyramid'/'slope', is MuJoCo-only); use "
                            "create_simulation(backend='mujoco') for terrain, or omit terrain "
                            "for a flat ground plane."
                        )
                    }
                ],
            }
        # Base create_world contract: reject a non-default difficulty with no
        # terrain rather than silently ignoring it. On Newton difficulty is
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
                            f"difficulty={difficulty!r} has no effect on the Newton backend "
                            "(it scales a heightfield terrain's elevation, and this backend "
                            "has no heightfield terrain); use create_simulation(backend='mujoco') "
                            "for a terrain curriculum, or omit difficulty for a flat ground plane."
                        )
                    }
                ],
            }
        # Same contract as set_timestep / set_gravity (and the MuJoCo backend):
        # never build a world around a dt or gravity vector the setters would
        # refuse. The effective timestep is validated so an unusable engine
        # default surfaces under its own name rather than driving the solver.
        effective_timestep = self.default_timestep if timestep is None else timestep
        timestep_param = "default_timestep" if timestep is None else "timestep"
        if err := self._validate_timestep(effective_timestep, "create_world", timestep_param):
            return err
        if gravity is None:
            components = [0.0, 0.0, -9.81]
        else:
            normalized, gravity_error = self._normalize_gravity(gravity, "create_world")
            if normalized is None:
                return cast("dict[str, Any]", gravity_error)
            components = normalized
        with self._lock:
            self._world = SimWorld(
                timestep=float(effective_timestep),
                gravity=components,
                ground_plane=ground_plane,
            )
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Newton world created (solver={self._solver_name})."}]}

    def destroy(self) -> dict[str, Any]:
        """Destroy the world and release Newton/Warp handles."""
        with self._lock:
            self._close_viewer()
            self._world = None
            self._model = None
            self._solver = None
            self._state_0 = self._state_1 = self._control = None
            self._joint_order = []
            self._targets = {}
            self._dr = None
            self._dr_applied = None
            self._dr_light_dir = None
            self._obs_noise = None
            self._obs_noise_rng = None
        return {"status": "success", "content": [{"text": "Newton world destroyed."}]}

    def reset(self) -> dict[str, Any]:
        """Reset the world to its initial joint configuration."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        with self._lock:
            self._targets = {}
            self._world.sim_time = 0.0
            self._world.step_count = 0
            self._rebuild()
        return {"status": "success", "content": [{"text": "Newton world reset."}]}

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance the simulation by ``n_steps`` control steps.

        Args:
            n_steps: Non-negative whole control-step count, on the shared
                :func:`~strands_robots.utils.non_negative_whole_number_error`
                domain every backend applies. ``0`` is an accepted no-op, as it
                is on MuJoCo. A NumPy or float count with an integral value is
                honored and coerced.

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists, ``n_steps`` is outside that domain, or the world was
            destroyed on a batch boundary mid-run - in which case the error names
            the steps completed, since some were.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if error := non_negative_whole_number_error(n_steps, "n_steps", "step"):
            return {"status": "error", "content": [{"text": error}]}
        n_steps = int(n_steps)
        # ``_advance`` floors its count at 1 (``max(1, n_steps)``), so a zero
        # handed through would advance the world one step while this method
        # reported stepping none. Answered here rather than by moving the floor.
        # Both callers of ``_advance`` now guarantee a positive count - this
        # branch, and ``send_action`` on the shared
        # ``positive_whole_number_error`` domain - so the floor is unreachable
        # from either public surface and is retained as a defensive no-op rather
        # than as a contract anything reads.
        if n_steps == 0:
            return {"status": "success", "content": [{"text": "Stepped 0 step(s) (no-op)."}]}
        # Batched so the lock is released every ``_STEPS_PER_BATCH`` control
        # steps. Previously the whole count ran under one acquisition: measured,
        # ``step(100_001)`` advanced 100_001 control steps - 100_001 *
        # ``substeps`` solver steps - inside a single hold, blocking every other
        # locked method on the engine for the duration.
        #
        # The per-batch re-check is the other half of that release, not a
        # separate concern: with the lock dropped between batches a concurrent
        # ``destroy`` (which nulls ``_world`` and ``_model`` under this same
        # lock) becomes reachable mid-call, so each batch confirms the world it is about to
        # advance still exists and aborts naming the steps completed rather than
        # handing a finalized-then-freed model to the solver.
        remaining = n_steps
        while remaining > 0:
            batch = min(remaining, self._STEPS_PER_BATCH)
            with self._lock:
                if self._world is None or self._model is None:
                    return {
                        "status": "error",
                        "content": [{"text": step_aborted_msg(n_steps - remaining, n_steps)}],
                    }
                self._advance(batch)
            remaining -= batch
        return {"status": "success", "content": [{"text": f"Stepped {n_steps} step(s)."}]}

    def get_state(self) -> dict[str, Any]:
        """Return a human-readable world-state summary."""
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        w = self._world
        lines = [
            "Newton Simulation State",
            f"solver={self._solver_name} t={w.sim_time:.4f}s (step {w.step_count})",
            f"dt={w.timestep}s gravity={w.gravity}",
            f"Robots: {len(w.robots)} | Objects: {len(w.objects)} | DOFs: {self._model.joint_dof_count}",
        ]
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # Robot management

    def _resolve_asset(self, name: str, source: str | None) -> tuple[str | None, str | None]:
        """Resolve a robot name to an MJCF/URDF model path for the given source.

        Args:
            name: Robot name or alias to resolve.
            source: One of :data:`_ROBOT_SOURCES`. ``"robot_descriptions"``
                resolves a URDF directly; ``"registry"`` restricts to the
                curated registry / MJCF asset manager; ``None`` tries the
                registry first then falls back to a ``robot_descriptions`` URDF.

        Returns:
            ``(model_path, None)`` on success, or ``(None, error_text)`` with a
            human-readable reason on failure.
        """
        if source == "robot_descriptions":
            urdf = discover_urdf_path(name)
            if urdf:
                return urdf, None
            return None, (
                f"Could not resolve a URDF for '{name}' via robot_descriptions. "
                "See list_urdfs() for URDF-discoverable robots."
            )

        # source is None or "registry": curated registry / MJCF asset manager.
        try:
            resolved = resolve_robot_name(name)
            asset_path = resolve_model_path(resolved)
        except (ValueError, FileNotFoundError, KeyError) as exc:
            asset_path = None
            resolve_error: Exception | None = exc
        else:
            resolve_error = None
        if asset_path:
            return str(asset_path), None

        # Default selector: fall back to a robot_descriptions URDF before failing
        # so the URDF-only long tail resolves without an explicit source.
        if source is None:
            urdf = discover_urdf_path(name)
            if urdf:
                return urdf, None

        detail = f": {resolve_error}" if resolve_error else ""
        hint = "See list_robots() / list_urdfs()." if source is None else "See list_robots()."
        return None, f"Could not resolve a sim asset for robot '{name}'{detail}. {hint}"

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        keyframe: str | int | None = None,
        source: str | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the world from a registered name, MJCF, or URDF.

        Newton ingests both MJCF and URDF models natively, so a robot can be
        resolved three ways (see ``source``). The asset format is detected from
        the resolved path's extension - ``.urdf`` files load through Newton's
        URDF importer, everything else through the MJCF importer.

        Validation: ``position`` must be 3 finite numbers and ``orientation``
        4, on the shared :func:`~strands_robots.utils.coerce_pose_vector`
        domain the MuJoCo backend's ``add_robot`` uses - a NumPy array is
        accepted, a ``bool`` component and a ``nan``/``inf`` are not, and an
        empty vector is a wrong-length request rather than an omission.

        Args:
            name: Robot name in the registry / ``robot_descriptions``, or an
                arbitrary instance name when ``urdf_path`` points at an explicit
                MJCF/URDF file. Required, and a non-empty ``str`` containing no
                NUL (:func:`~strands_robots.utils.entity_name_error`); this
                backend has no derive-a-label short form.
            urdf_path: Optional explicit MJCF/URDF path. When given it wins
                outright and ``source`` is ignored.
            data_config: Accepted for ABC parity; unused by Newton.
            position: World position ``[x, y, z]`` (default origin).
            orientation: World orientation as a wxyz quaternion
                (default identity).
            source: Asset-resolution selector. ``None`` (default) tries the
                curated registry / MJCF asset manager first and falls back to a
                ``robot_descriptions`` URDF lookup; ``"registry"`` restricts to
                the registry; ``"robot_descriptions"`` resolves the URDF directly
                via ``robot_descriptions.<name>_description.URDF_PATH``. See
                :func:`~strands_robots.registry.discovery.list_urdf_discoverable`.
            keyframe: Canonical spawn pose. Not yet supported on the Newton
                backend (the MuJoCo backend applies a source ``<keyframe>``);
                passing a non-``None`` value is a clean error rather than a
                silent ignore.

        Returns:
            Status dict including the resolved joint names.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        # Refuse a name that cannot address the robot this call creates, on the
        # shared ``entity_name_error`` domain. Unlike the MuJoCo backend this
        # method documents no "derive a label" short form - ``name`` is required
        # and doubles as the registry/robot_descriptions key - so there is no
        # value to pass through here.
        if (name_err := entity_name_error("add_robot", "name", name)) is not None:
            return {"status": "error", "content": [{"text": name_err}]}

        # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
        # MuJoCo backend's ``add_robot`` and this backend's own ``add_camera`` already
        # use, so a pose one backend refuses is refused by all of them - the
        # invariant that helper documents. The ``x or <default>`` reads this
        # replaces tested the VECTOR, which is wrong twice over: a NumPy array
        # (what pose arithmetic produces, and what the Args below advertise)
        # raises a bare ``ValueError: truth value of an array ... is ambiguous``
        # straight through the structured envelope, and an empty vector is falsy
        # so it read as *omitted* - the default was substituted while the call
        # reported success. Nothing downstream caught the rest either: a
        # wrong-length, ``nan``/``inf``, ``bool`` or non-numeric component was
        # stored verbatim on the registry entry and handed to the solver rebuild.
        position, _perr = coerce_pose_vector("add_robot", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        orientation, _oerr = coerce_pose_vector("add_robot", "orientation", orientation, 4)
        if _oerr is not None:
            return {"status": "error", "content": [{"text": _oerr}]}
        if name in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' already exists."}]}
        if source not in _ROBOT_SOURCES:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unknown source {source!r}. "
                            f"Valid: {[s for s in _ROBOT_SOURCES if s is not None]} or None (default)."
                        )
                    }
                ],
            }
        if keyframe is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "keyframe= is not yet supported on the Newton backend; "
                            "spawn keyframe home poses are a MuJoCo-backend feature."
                        )
                    }
                ],
            }

        if urdf_path is not None:
            model_path = urdf_path
        else:
            resolved_path, error = self._resolve_asset(name, source)
            if resolved_path is None:
                return {"status": "error", "content": [{"text": error}]}
            model_path = resolved_path

        with self._lock:
            robot = SimRobot(
                name=name,
                urdf_path=model_path,
                position=[0.0, 0.0, 0.0] if position is None else position,
                orientation=[1.0, 0.0, 0.0, 0.0] if orientation is None else orientation,
                data_config=data_config,
            )
            self._world.robots[name] = robot
            try:
                self._rebuild()
            except Exception:
                del self._world.robots[name]
                self._rebuild()
                raise
            robot.joint_names = list(self._robot_joint_map.get(name, []))
        return {
            "status": "success",
            "content": [{"text": f"Added robot '{name}' ({len(robot.joint_names)} joints)."}],
        }

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot and rebuild the world."""
        if self._world is None or not registered(self._world.robots, name):
            return {"status": "error", "content": [{"text": f"Robot '{name}' not found."}]}
        with self._lock:
            del self._world.robots[name]
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Removed robot '{name}'."}]}

    def list_robots(self) -> list[str]:
        """Return the ordered names of robots in the world."""
        if self._world is None:
            return []
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered short joint names for ``robot_name``."""
        if self._world is None or not registered(self._world.robots, robot_name):
            return []
        return list(self._world.robots[robot_name].joint_names)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        """Return the scalar joint targets ``send_action`` resolves for ``robot_name``.

        Overrides :meth:`SimEngine.robot_action_keys`, whose default mirrors
        :meth:`robot_joint_names`. Newton's joint list carries a floating base's
        6-DoF free joint, which is not a commandable scalar: its coordinates are
        ``[xyz, quat_xyzw]``, so there is no single target to write. Every scalar
        surface on this backend already skips it and surfaces the base as the
        structured ``base_pos`` / ``base_quat`` / ``base_lin_vel`` /
        ``base_ang_vel`` signals instead - :meth:`get_observation`,
        :meth:`get_robot_state` and the recording schema
        (``_collect_recording_schema``). This method is the action-side half of
        that same exclusion.

        Leaving it in was silent on all four surfaces that read this list. It
        made ``send_action({free_joint: 0.5})`` write a scalar target for a
        6-DoF joint under a success result; it made the vector form reject an
        action of the width a recording actually holds (the recorded columns
        exclude the free joint, so the counts differed by one); and it left
        ``PolicyRunner.replay`` binding one more key than the recorded vector
        carries, which aborts a floating-base episode on frame 0 - so such a
        recording could be written but never replayed.

        A fixed-base robot has no free root, so this returns its joint names
        unchanged.
        """
        # ``getattr`` rather than a direct read: the recording paths already
        # reach this list on engines built via ``__new__`` (the solver-free test
        # harness), which never run ``__init__``. Mirrors
        # ``_collect_recording_schema``'s read of the same map.
        base_joint = getattr(self, "_robot_free_base_joint", {}).get(robot_name)
        return [jn for jn in self.robot_joint_names(robot_name) if jn != base_joint]

    # Object management

    def add_object(
        self,
        name: str,
        shape: str = "box",
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        size: list[float] | None = None,
        color: list[float] | None = None,
        mass: float = 0.1,
        is_static: bool | None = None,
        mesh_path: str | None = None,
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a primitive or mesh object to the scene.

        Validation: ``position`` must be 3 finite numbers and ``orientation``
        4 (a list, tuple or NumPy array; NumPy scalar elements accepted,
        ``bool`` refused). Omit a vector to take its default; an empty vector
        is a wrong-length request and is rejected rather than silently placing
        the object at the default pose. These are the same bounds the MuJoCo
        backend's ``add_object`` enforces, on the shared
        :func:`~strands_robots.utils.coerce_pose_vector` domain, so a pose one
        backend refuses is refused by both.

        ``size`` must be a non-empty vector of finite numbers, on the shared
        :func:`~strands_robots.utils.coerce_size_vector` domain the MuJoCo
        backend's ``add_object`` composes with its own per-shape table, so an
        extent one backend refuses is refused by both. An empty vector is a
        component count rather than an omission and is rejected instead of
        silently applying the default extent. How many components each shape
        needs, and whether a consumed extent must be positive, are shape-
        dependent and not yet unified across backends (#1858).

        ``mass`` must be a finite number > 0 for a dynamic object, the domain
        :meth:`~strands_robots.simulation.base.SimEngine._validate_mass`
        defines and the MuJoCo backend's ``add_object`` already enforces, so a
        mass one backend refuses is refused by both. ``mass=0`` is the one
        documented exception: it is this backend's alternative spelling of
        ``is_static=True`` and stays accepted. A static object's mass is never
        read, so it is not validated - the same scope MuJoCo uses.

        Args:
            name: Unique object name; a non-empty ``str`` containing no NUL, the
                same domain the MuJoCo backend's ``add_object`` accepts
                (:func:`~strands_robots.utils.entity_name_error`).
            shape: One of ``"box"``, ``"sphere"``, ``"capsule"``,
                ``"cylinder"``, or ``"mesh"``. ``"mesh"`` requires
                ``mesh_path``.
            position: World position ``[x, y, z]`` (default origin).
            orientation: wxyz quaternion (default identity).
            size: Half-extents (box) or ``[radius, ...]`` (others). For
                ``shape="mesh"`` this is the per-axis scale applied to the
                loaded geometry (default ``[1, 1, 1]`` -- the mesh's own units).
                Must be a non-empty vector of finite numbers; a ``nan``/``inf``,
                ``bool`` or non-numeric component is refused rather than handed
                to the solver rebuild, and an empty vector is refused rather
                than read as an omission. NumPy arrays are accepted.
            color: ``[r, g, b]`` or ``[r, g, b, a]`` in 0..1 (default
                mid-grey; alpha currently ignored by Newton shapes). Any
                other component count is refused rather than padded or
                truncated, since either would paint a colour that was not
                asked for. NumPy arrays are accepted.
            mass: Object mass in kg; a finite number > 0 for a dynamic
                object. ``0`` or ``is_static`` makes it static instead, and
                a static object's mass is not read. Any other value -
                negative, ``nan``, ``inf``, a ``bool`` or a non-number - is
                refused rather than handed to the solver rebuild.
            is_static: When True the object is fixed in the world.
                ``None`` (the default) means unspecified; Newton derives
                nothing from ``shape``, so it resolves to dynamic.
            mesh_path: Path to a mesh asset (``.obj`` / ``.stl`` / ``.glb`` /
                ``.usd`` -- anything ``trimesh.load`` accepts). Required and
                only used when ``shape="mesh"``; the mesh is loaded via
                ``trimesh`` and converted to a Newton collision/visual shape.
            material: Visual material/texture spec. NOT supported by the
                Newton backend yet; a non-``None`` value is rejected loudly
                rather than silently dropped (use the MuJoCo backend for
                matte/textured surfaces).

        Returns:
            Status dict.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        # Refuse a name that cannot address the object this call creates, on the
        # same shared domain the MuJoCo backend's ``add_object`` uses, so a name
        # one backend refuses is refused by both. It precedes the duplicate-name
        # test below, which is itself partial: ``name in self._world.objects``
        # raises ``TypeError: unhashable type`` for a list/dict/set name rather
        # than reaching the error path it guards.
        if (name_err := entity_name_error("add_object", "name", name)) is not None:
            return {"status": "error", "content": [{"text": name_err}]}

        # ``None`` means the caller did not specify, per
        # :meth:`~strands_robots.simulation.base.SimEngine.add_object`. Newton
        # has no shape-derived rule, so unspecified is dynamic. Resolved here so
        # the registered :class:`SimObject` carries a real bool rather than the
        # sentinel, which every later reader would have to re-interpret.
        is_static = False if is_static is None else is_static

        # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
        # MuJoCo backend's ``add_object`` and this backend's own ``add_camera`` already
        # use, so a pose one backend refuses is refused by all of them - the
        # invariant that helper documents. The ``x or <default>`` reads this
        # replaces tested the VECTOR, which is wrong twice over: a NumPy array
        # (what pose arithmetic produces, and what the Args below advertise)
        # raises a bare ``ValueError: truth value of an array ... is ambiguous``
        # straight through the structured envelope, and an empty vector is falsy
        # so it read as *omitted* - the default was substituted while the call
        # reported success. Nothing downstream caught the rest either: a
        # wrong-length, ``nan``/``inf``, ``bool`` or non-numeric component was
        # stored verbatim on the registry entry and handed to the solver rebuild.
        position, _perr = coerce_pose_vector("add_object", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        orientation, _oerr = coerce_pose_vector("add_object", "orientation", orientation, 4)
        if _oerr is not None:
            return {"status": "error", "content": [{"text": _oerr}]}
        # Same shared domain for the colour, whose accepted counts the 4-component
        # rgba row it ends up in defines. The ``color or <default>`` coalescing this
        # replaces was wrong three ways: a NumPy colour raised
        # ``ValueError: truth value of an array ... is ambiguous`` straight through
        # this method's only documented failure channel, ``[]`` read as *omitted* so
        # the default grey was painted under a success result, and a short colour was
        # stored verbatim - ``_add_object_to_builder`` then handed
        # ``tuple(obj.color[:3])`` a 1- or 2-component colour to the solver, at
        # rebuild time rather than at the call the caller can attribute it to.
        color, _cerr = coerce_rgba("add_object", "color", color)
        if _cerr is not None:
            return {"status": "error", "content": [{"text": _cerr}]}
        # The shape-independent half of the extent contract, on the shared
        # ``coerce_size_vector`` domain the MuJoCo backend's ``add_object``
        # composes with its own per-shape table. ``size or default_size`` below
        # was the LAST surviving truthiness coalesce on this constructor - the
        # other four vector parameters all test ``is None`` - and it carried the
        # same two defects that spelling always carries: ``np.array([.1,.1,.1])``
        # raised ``ValueError: truth value of an array ... is ambiguous`` through
        # this method's only documented failure channel, and ``[]`` is falsy so it
        # read as *omitted*, applying the default 5 cm extent under a success
        # result. Everything else was stored verbatim and reached the solver at
        # REBUILD time rather than at the call a caller can attribute it to:
        # ``[nan, .1, .1]`` and ``[inf, .1, .1]`` became a box half-extent,
        # ``[True, .1, .1]`` read ``True`` as an extent, ``[None, .1, .1]`` and
        # ``[[0.1], .1, .1]`` handed the shape builder a ``None`` and a nested
        # list, and the bare string ``"abc"`` was stored AS the size - raising
        # ``TypeError: can only concatenate str (not "list") to str`` from the box
        # branch, and silently building a sphere of ``radius='a'`` from the sphere
        # one. The per-shape component count, the short-vector question and the
        # positivity of a consumed extent are shape-dependent and stay with #1858.
        size, _serr = coerce_size_vector("add_object", "size", size)
        if _serr is not None:
            return {"status": "error", "content": [{"text": _serr}]}
        # A dynamic body's mass is the divisor of every force applied to it and
        # reaches the solver as ``builder.add_body(mass=...)``, so it goes
        # through the same shared domain the MuJoCo backend's ``add_object``
        # applies to the same quantity. Storing it raw was wrong in three ways.
        # ``nan``/``inf``/``True`` reached that call verbatim. A negative mass
        # silently took the ``obj.mass <= 0`` static path, making the body
        # immovable on a value only ``0`` is documented to mean. And a
        # non-number was stored, then raised
        # ``TypeError: '<=' not supported between instances of 'str' and 'int'``
        # out of BOTH consumers of that comparison - the rebuild and
        # ``list_objects`` - so one bad add left an already-registered object
        # that made a later, unrelated scene query raise.
        if not is_static and not _is_zero_mass_sentinel(mass):
            if (mass_err := self._validate_mass(mass, "add_object")) is not None:
                return mass_err
        if material is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "add_object: material= is not supported by the Newton "
                            "backend. Use the MuJoCo backend for matte/textured surfaces."
                        )
                    }
                ],
            }
        if name in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' already exists."}]}
        if shape not in ("box", "sphere", "capsule", "cylinder", "mesh"):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Unsupported shape {shape!r} for Newton backend. "
                            "Supported: box, sphere, capsule, cylinder, mesh."
                        )
                    }
                ],
            }
        if shape == "mesh":
            if not mesh_path:
                return {
                    "status": "error",
                    "content": [{"text": "add_object: shape='mesh' requires mesh_path=<path to .obj/.stl/.glb/.usd>."}],
                }
            resolved = Path(mesh_path).expanduser()
            if not resolved.is_file():
                return {
                    "status": "error",
                    "content": [{"text": f"add_object: mesh_path {mesh_path!r} does not exist or is not a file."}],
                }
            mesh_path = str(resolved)
        default_size = [1.0, 1.0, 1.0] if shape == "mesh" else [0.05, 0.05, 0.05]
        with self._lock:
            obj = SimObject(
                name=name,
                shape=shape,
                position=[0.0, 0.0, 0.0] if position is None else position,
                orientation=[1.0, 0.0, 0.0, 0.0] if orientation is None else orientation,
                size=default_size if size is None else size,
                color=[0.5, 0.5, 0.5, 1.0] if color is None else color,
                mass=mass,
                mesh_path=mesh_path,
                is_static=is_static,
            )
            self._world.objects[name] = obj
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Added {shape} object '{name}'."}]}

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object and rebuild the world."""
        if self._world is None or not registered(self._world.objects, name):
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        with self._lock:
            del self._world.objects[name]
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Removed object '{name}'."}]}

    def move_object(
        self, name: str, position: list[float] | None = None, orientation: list[float] | None = None
    ) -> dict[str, Any]:
        """Move an existing object to a new pose and rebuild the world.

        Mirrors the MuJoCo backend contract. Newton finalises an immutable
        model from a builder, so the object's stored pose is updated on its
        :class:`SimObject` and the model is rebuilt; live joint targets are
        preserved across the rebuild (see :meth:`_rebuild`).

        Validation: ``position`` must be 3 finite numbers and ``orientation``
        4, on the shared :func:`~strands_robots.utils.coerce_pose_vector`
        domain the MuJoCo backend's ``move_object`` uses. An omitted vector
        leaves that component alone; an empty one is refused rather than read
        as an omission.

        Args:
            name: Name of an object previously added via :meth:`add_object`.
            position: New world position ``[x, y, z]``. ``None`` keeps the
                current position.
            orientation: New wxyz quaternion. ``None`` keeps the current
                orientation.

        Returns:
            Status dict echoing the new position, or an error dict when no
            world exists or the object is unknown.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not registered(self._world.objects, name):
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        # Validate the pose vectors on the shared ``coerce_pose_vector`` domain the
        # MuJoCo backend's ``move_object`` and this backend's own ``add_camera`` already
        # use, so a pose one backend refuses is refused by all of them - the
        # invariant that helper documents. The ``x or <default>`` reads this
        # replaces tested the VECTOR, which is wrong twice over: a NumPy array
        # (what pose arithmetic produces, and what the Args below advertise)
        # raises a bare ``ValueError: truth value of an array ... is ambiguous``
        # straight through the structured envelope, and an empty vector is falsy
        # so it read as *omitted* - the default was substituted while the call
        # reported success. Nothing downstream caught the rest either: a
        # wrong-length, ``nan``/``inf``, ``bool`` or non-numeric component was
        # stored verbatim on the registry entry and handed to the solver rebuild.
        position, _perr = coerce_pose_vector("move_object", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        orientation, _oerr = coerce_pose_vector("move_object", "orientation", orientation, 4)
        if _oerr is not None:
            return {"status": "error", "content": [{"text": _oerr}]}
        with self._lock:
            obj = self._world.objects[name]
            if position is not None:
                obj.position = position
            if orientation is not None:
                obj.orientation = orientation
            self._rebuild()
        moved_to = "same" if position is None else position
        return {"status": "success", "content": [{"text": f"'{name}' moved to {moved_to}"}]}

    def list_objects(self) -> dict[str, Any]:
        """List objects in the world with their shape, pose, and mass.

        Returns:
            Agent-tool dict whose ``text`` block mirrors the MuJoCo backend's
            human-readable object listing.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}
        lines = ["Objects:\n"]
        for name, obj in self._world.objects.items():
            mass = "static" if obj.is_static or obj.mass <= 0 else f"{obj.mass}kg"
            suffix = f", mesh={obj.mesh_path}" if obj.shape == "mesh" and obj.mesh_path else ""
            lines.append(f"  - {name}: {obj.shape} at {obj.position}, {mass}{suffix}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # Observation / action

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Return joint positions plus any registered camera frames.

        The proprioceptive part maps each short joint name to its position
        (radians). When cameras have been registered via :meth:`add_camera`
        and ``skip_images`` is False, each camera is rendered and added to the
        observation keyed by camera name (an ``(H, W, 3)`` uint8 RGB ndarray),
        mirroring the MuJoCo backend so multi-camera policies see the same
        observation shape on either engine.

        Args:
            robot_name: Robot to observe. ``None`` resolves to the single
                robot when exactly one exists.
            skip_images: When True, skip camera rendering and return joint
                state only (used by control loops that do not need pixels).

        Returns:
            Mapping of short joint name to joint position (float), plus one
            entry per registered camera (name -> RGB ndarray) when
            ``skip_images`` is False. A robot with a floating base additionally
            carries ``base_pos`` (world x,y,z incl. height), ``base_quat``
            (orientation, w,x,y,z), ``base_lin_vel`` (m/s, WORLD frame) and
            ``base_ang_vel`` (rad/s, BODY frame - matching the MuJoCo backend and
            the IMU-gyro frame WBC / locomotion controllers consume) for
            locomotion controllers. Empty when no world exists or the robot is
            unknown.
        """
        if self._world is None or self._model is None:
            return {}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError:
            return {}
        if not registered(self._world.robots, robot_name):
            return {}
        if skip_images and self._world._backend_state.get("recording"):
            # T26: dataset recording needs every frame's image obs. Override
            # the policy's skip hint when an active recorder is attached so a
            # non-image policy (e.g. the default mock, requires_images=False)
            # does not silently record pixel-less frames for declared camera
            # features. Mirrors the MuJoCo backend's get_observation guard.
            skip_images = False
        with self._lock:
            joint_q = self._state_0.joint_q.numpy()
            joint_qd = self._state_0.joint_qd.numpy()
            robot_joints = self._world.robots[robot_name].joint_names
            free_short = self._robot_free_base_joint.get(robot_name)
            obs: dict[str, Any] = {}
            for jname in robot_joints:
                # A 6-DoF free joint (floating base) is not a scalar joint: its
                # coordinate 0 is the base x-position, so obs[jname] =
                # joint_q[idx] would report base-x as a joint angle (dropping the
                # rest of the pose + twist) - a degenerate scalar and a duplicate
                # of base_pos.x. Its full state is surfaced below as the
                # structured base_pos/base_quat/base_lin_vel/base_ang_vel keys,
                # matching get_robot_state and the MuJoCo backend.
                if jname == free_short:
                    continue
                idx = self._joint_coord_index.get((robot_name, jname))
                if idx is not None and idx < len(joint_q):
                    obs[jname] = float(joint_q[idx])
        # Joint-position sensor noise applies only to the float joint entries;
        # camera frames are added afterwards (and carry their own jitter via the
        # render path), so the result holds mixed float/ndarray values.
        obs_out: dict[str, Any] = dict(self._apply_joint_pos_noise(obs))
        # Floating-base IMU-style signals for a robot with a free root (a
        # humanoid / mobile base): ``base_quat`` (orientation, w,x,y,z) and
        # ``base_ang_vel`` (rad/s), consumed by WBC / locomotion controllers.
        # Additive and absent for fixed-base arms; left un-noised, matching the
        # MuJoCo backend's contract.
        base = self._free_base_pose(robot_name, joint_q, joint_qd)
        if base is not None:
            obs_out["base_pos"] = base["position"]
            obs_out["base_quat"] = base["quaternion"]
            obs_out["base_lin_vel"] = base["linear_velocity"]
            obs_out["base_ang_vel"] = base["angular_velocity"]
        if not skip_images:
            from strands_robots.simulation.policy_runner import _extract_frame_ndarray

            for cam_name in list(self._world.cameras):
                render_result = self.render(camera_name=cam_name)
                img = _extract_frame_ndarray(render_result)
                if img is not None:
                    obs_out[cam_name] = img
        return obs_out

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply position targets and advance physics by ``n_substeps``.

        ``action`` may be a ``{joint name: target}`` mapping or an ordered
        numeric vector (``list`` / ``tuple`` / 1-D ``numpy`` array) bound
        positionally to ``robot_action_keys(robot_name)`` - the same positional
        convention :meth:`replay_episode` uses. A vector whose length does not
        match that action-key count is rejected with an actionable error. Those
        keys are the robot's scalar joints; a floating base's 6-DoF free joint is
        not among them (see :meth:`robot_action_keys`), so it is neither bound
        positionally nor accepted as a mapping key.

        Args:
            action: Mapping of short joint name to target position (radians).
            robot_name: Robot to actuate. ``None`` resolves to the single
                robot when exactly one exists.
            n_substeps: Positive whole number of control steps to advance after
                writing targets, on the shared
                :func:`~strands_robots.utils.positive_whole_number_error` domain
                every backend applies. A NumPy or float count with an integral
                value is honored and coerced; a fractional, zero, negative,
                non-finite, boolean or non-numeric count is refused. ``0`` is
                refused rather than honored as "write but do not advance" -
                :meth:`step` is the surface that advances a count of its own,
                and it accepts ``0`` as a documented no-op.

        Returns:
            Status dict. When some keys cannot be resolved to joints, the
            ``content`` carries a ``json`` block with ``unresolved_keys`` and
            ``applied`` so callers can self-correct. ``status`` is ``"error"``
            when ``n_substeps`` is outside that domain, and no target is
            written when it is.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        # Refused before ``_write_targets``, because a refusal after the write
        # would leave the robot commanded and the world un-advanced. Pre-fix
        # this count reached ``_advance``'s ``max(1, n_steps)`` floor, so a
        # ``0`` or a ``-5`` advanced one control step under a success result;
        # on the solver-free path the same floor added a ``2.7`` or an ``inf``
        # straight into ``step_count``, and a non-numeric count raised
        # ``TypeError`` past this method's structured envelope.
        if error := positive_whole_number_error(n_substeps, "n_substeps", "send_action"):
            return {"status": "error", "content": [{"text": error}]}
        n_substeps = int(n_substeps)
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        action_map, coerce_error = self._coerce_action(action, robot_name)
        if coerce_error is not None:
            return coerce_error
        assert action_map is not None  # narrow for mypy: no error implies a mapping

        # Resolved through ``robot_action_keys``, not the raw joint list: a
        # floating base's free joint is in the latter and is not a commandable
        # scalar (see robot_action_keys). Reading the raw list here accepted
        # ``{free_joint: 0.5}`` under a success result and wrote a scalar target
        # for a 6-DoF joint, while every read surface skipped it - so the value
        # named no signal the backend reports and no column it records.
        valid = set(self.robot_action_keys(robot_name))
        unresolved = [k for k in action_map if k not in valid]
        applied = [k for k in action_map if k in valid]
        with self._lock:
            for jname in applied:
                self._targets[(robot_name, jname)] = float(action_map[jname])
            self._write_targets()
            self._advance(n_substeps)

        if unresolved:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action partially applied: keys {unresolved} are not commandable joints on "
                            f"'{robot_name}'. Applied: {applied}. Valid keys: {sorted(valid)}"
                        )
                    },
                    {"json": {"unresolved_keys": unresolved, "applied": applied}},
                ],
            }
        return {"status": "success", "content": [{"text": f"Action applied to '{robot_name}' ({len(applied)} keys)."}]}

    def physics_timestep(self) -> float | None:
        """Return the physics integration timestep in seconds."""
        if self._world is None:
            return None
        return float(self._world.timestep)

    def set_gravity(self, gravity: list[float] | float | int) -> dict[str, Any]:
        """Set the world gravity vector and rebuild so the solver applies it.

        Mirrors the MuJoCo backend: a scalar is interpreted as the z-component
        ``[0, 0, g]``; a 3-element list sets the full ``[x, y, z]`` vector in
        m/s^2. The accepted domain is
        :meth:`~strands_robots.simulation.base.SimEngine._normalize_gravity`,
        shared with ``create_world`` and with every other backend, so a value
        one gravity surface refuses is refused by all of them. Newton's solver snapshots gravity at construction time, so the
        model is rebuilt; this re-initialises the world to its rest pose, so
        prefer setting gravity before stepping. Live joint targets are
        preserved across the rebuild.

        Args:
            gravity: Scalar z-gravity, or a 3-element ``[x, y, z]`` vector.

        Returns:
            Status dict echoing the applied gravity, or an error dict when no
            world exists or the argument is not a finite 3-vector / scalar.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        # Normalize through the shared domain rather than a local copy of it,
        # so what this backend refuses and accepts cannot drift from
        # ``create_world`` above or from the other backends. The local copy
        # coerced a scalar with ``float()``, and bool is an int subclass, so
        # ``set_gravity(True)`` configured a +1 m/s^2 gravity pointing *up* and
        # reported success.
        components, gravity_error = self._normalize_gravity(gravity, "set_gravity")
        if components is None:
            return cast("dict[str, Any]", gravity_error)
        with self._lock:
            self._world.gravity = components
            self._rebuild()
        return {"status": "success", "content": [{"text": f"Gravity: {components}"}]}

    def set_timestep(self, timestep: float) -> dict[str, Any]:
        """Set the physics integration timestep in seconds.

        Mirrors the MuJoCo backend. Newton reads the timestep live on each
        :meth:`step` (``dt = timestep / substeps``), so no model rebuild is
        required and the change takes effect on the next step.

        Args:
            timestep: Positive integration timestep in seconds.

        Validation: ``timestep`` must be a finite number greater than zero.
        ``bool`` is refused explicitly (``True`` would otherwise act as a
        1-second step, 500x the default) and so is a non-numeric value. The
        domain is :meth:`~strands_robots.simulation.base.SimEngine._validate_timestep`,
        shared with :meth:`create_world` and with the MuJoCo setter, so a value
        one surface accepts is accepted by all of them.

        Returns:
            Status dict reporting the timestep and equivalent control rate, or
            an error dict when no world exists or the value is outside that
            domain.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        # Shared with create_world, and with the MuJoCo setter this method
        # mirrors, so no surface can install a dt another one refuses. The
        # hand-rolled float()/isfinite() pair this replaces had no bool arm, so
        # ``True`` coerced to a 1-second step under status="success".
        if err := self._validate_timestep(timestep, "set_timestep"):
            return err
        timestep = float(timestep)
        warn = " Warning: unusually large timestep (>0.1s); physics may be unstable" if timestep > 0.1 else ""
        with self._lock:
            self._world.timestep = timestep
        return {"status": "success", "content": [{"text": f"Timestep: {timestep}s ({1 / timestep:.0f}Hz){warn}"}]}

    # Rendering

    def add_camera(
        self,
        name: str,
        position: list[float] | None = None,
        target: list[float] | None = None,
        fov: float = 60.0,
        width: int = 640,
        height: int = 480,
        parent_body: str | None = None,
    ) -> dict[str, Any]:
        """Register a named camera for :meth:`render` and recording.

        Mirrors the MuJoCo backend so the same call site works on either
        engine. Newton cameras are ray-traced on demand by :meth:`render`
        (no model rebuild), so adding a camera never disturbs physics state.

        Orientation: the camera looks from ``position`` towards ``target``
        (an OpenGL-style look-at, see :meth:`_look_at_quat`). Degenerate
        cases (``position == target``) are rejected.

        Mounting (``parent_body``): when set to a body label (e.g. a robot's
        wrist such as the gripper body returned by :meth:`list_bodies`), the
        camera is mounted ON that body and rides along with it -- this models a
        realistic wrist/gripper camera for SO101/SO100-style data collection.
        In this mode ``position`` and ``target`` are interpreted in the body's
        LOCAL frame and resolved to world coordinates each render from the live
        body transform. Empty/``None`` = a world-fixed camera. Call
        :meth:`list_bodies` to discover valid mount points.

        Validation: ``position`` and ``target`` must each be 3 finite numbers
        (a list, tuple or NumPy array; NumPy scalar elements accepted, ``bool``
        refused). Omit a vector to take its default - an empty vector is a
        wrong-length request and is rejected rather than silently placing the
        camera at the default pose. ``fov`` must be a finite angle in ``(0,
        180)`` degrees. These are the same bounds the MuJoCo backend's
        ``add_camera`` enforces, on the shared
        :func:`~strands_robots.utils.pose_vector_error` /
        :func:`~strands_robots.utils.camera_fov_error` domains, so a camera
        configuration one backend refuses is refused by both. Invalid values are
        rejected here at config time rather than deferring an uncaught
        ``ValueError`` (non-numeric) or a silently degenerate camera
        (``nan``/``inf`` in the look-at basis or the derived intrinsics) to the
        first render.

        ``width`` and ``height`` must each be a positive integer, the same
        floor :meth:`render` and the MuJoCo backend's ``add_camera`` enforce
        (:func:`~strands_robots.utils.positive_count_error`). Newton ray-traces
        each frame on demand and has no offscreen framebuffer, so unlike MuJoCo
        there is no upper cap.

        Args:
            name: Unique camera name; a non-empty ``str`` containing no NUL, the
                same domain the MuJoCo backend's ``add_camera`` accepts
                (:func:`~strands_robots.utils.entity_name_error`). Duplicate
                names are rejected; remove the existing camera with
                :meth:`remove_camera` first.
            position: Camera eye ``[x, y, z]`` (world frame, or the parent
                body's local frame when ``parent_body`` is set). 3 finite
                numbers.
            target: Look-at point ``[x, y, z]`` (same frame as ``position``).
                3 finite numbers.
            fov: Vertical field of view in degrees; a finite angle in the open
                interval ``(0, 180)``.
            width: Render width in pixels for this camera; a positive integer.
            height: Render height in pixels for this camera; a positive integer.
            parent_body: Optional body label to mount the camera on. ``None``
                or empty leaves the camera world-fixed.

        Returns:
            Status dict confirming the registration, or an error dict when no
            world exists, the pose, ``fov`` or a pixel dimension is invalid, the
            name is taken, or the mount body is unknown. Never raises.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        # Refuse a name that cannot address the camera this call creates, on the
        # shared ``entity_name_error`` domain the MuJoCo backend's ``add_camera``
        # uses, so a camera name one backend refuses is refused by both.
        if (name_err := entity_name_error("add_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": name_err}]}

        # Validate shape, element type AND finiteness with the shared helpers the
        # MuJoCo backend uses, so a camera pose one backend refuses is refused by
        # both - the invariant ``pose_vector_error`` documents. The local
        # coercion this replaces only applied ``float(v)``: that caught a
        # non-numeric element but passed a ``nan``/``inf`` component (and a
        # ``bool``, silently reading ``True`` as the coordinate 1.0) straight
        # through to the registry. Nothing downstream catches it either - the
        # degenerate-orientation check below compares
        # ``abs(pos[i] - tgt[i]) < 1e-9``, which is False for a ``nan``, and
        # ``_look_at_quat`` then divides the view vector by a ``nan`` norm, so
        # ``render``/``get_frame`` return a frame from an all-NaN camera
        # quaternion under a success result.
        position, _perr = coerce_pose_vector("add_camera", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)
        if _terr is not None:
            return {"status": "error", "content": [{"text": _terr}]}
        pos = [1.0, 1.0, 1.0] if position is None else position
        tgt = [0.0, 0.0, 0.0] if target is None else target
        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"add_camera: 'position' and 'target' are identical ({pos}); camera has no look direction."
                    }
                ],
            }
        # Validate the field of view at config time, on the same shared domain
        # the MuJoCo backend uses. It was previously coerced by a bare
        # ``float(fov)`` inside the lock below, so a non-numeric value raised a
        # ``ValueError`` straight through the structured tool-result contract,
        # and ``nan``/``inf``/``0``/``>=180`` registered a degenerate camera
        # under a success result: :meth:`get_camera_params` derives the pinhole
        # intrinsics ``0.5 * h / tan(radians(fov) / 2)``, which is ``nan`` for a
        # ``nan`` fov and raises ``ZeroDivisionError`` for ``0``.
        if (e := camera_fov_error("add_camera", "fov", fov)) is not None:
            return {"status": "error", "content": [{"text": e}]}
        # Validate the pixel dimensions on the shared floor the MuJoCo backend's
        # ``_validate_render_dims`` already applies, so a resolution one backend
        # refuses is refused by all of them. They were previously coerced by a
        # bare ``int(width)`` inside the lock below, which validated nothing: a
        # non-numeric value raised ``ValueError`` straight through the structured
        # tool-result contract (``None`` a ``TypeError``, ``nan`` a
        # ``ValueError``, ``inf`` an ``OverflowError``), and ``0`` / a negative
        # registered a camera that cannot produce a frame under a success
        # result - deferring the failure to the first render, far from the call
        # that caused it. A fractional or ``bool`` value was silently truncated
        # (``2.7`` stored as 2, ``True`` as 1) while the success text below
        # echoed the value the caller passed, reporting a resolution that was
        # never registered.
        for _param, _value in (("width", width), ("height", height)):
            if (e := positive_count_error(_value, _param, "add_camera")) is not None:
                return {"status": "error", "content": [{"text": e}]}
        # Refuse a name this backend's own render entry points would resolve past.
        # Shared with the MuJoCo sibling, which routes the same tokens and until
        # now refused none of them, so the two cannot state the rule differently.
        if (reserved_err := reserved_camera_name_error("add_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": reserved_err}]}
        if name in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"add_camera: camera '{name}' already exists. Remove it first."}],
            }

        mount = parent_body or ""
        if mount:
            body_labels = list(self._model.body_label)
            if mount not in body_labels:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"add_camera: parent_body '{mount}' not found. Call list_bodies to "
                                f"discover mount points. Available bodies: {body_labels}"
                            )
                        }
                    ],
                }

        with self._lock:
            self._world.cameras[name] = SimCamera(
                name=name,
                position=pos,
                target=tgt,
                fov=float(fov),
                width=width,
                height=height,
                parent_body=mount,
            )
        where = f" mounted on '{mount}'" if mount else ""
        return {
            "status": "success",
            "content": [{"text": f"Camera '{name}' added ({width}x{height}, fov={fov}){where}."}],
        }

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a previously registered named camera.

        Args:
            name: Name of a camera added via :meth:`add_camera`.

        Returns:
            Status dict confirming removal, or an error dict when no world
            exists, ``name`` is a free-camera routing token, or the camera is
            unknown.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        # Same rule as ``add_camera``, at the other end of the name's life: a
        # routing token cannot be un-addressed, and ``list_cameras`` names it
        # unconditionally. It precedes the existence test because that test
        # answers "not found. Registered: [...]" for a name ``list_cameras``
        # advertises, which reads as a bookkeeping slip rather than the rule.
        if (reserved_err := reserved_camera_name_error("remove_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": reserved_err}]}
        if not registered(self._world.cameras, name):
            return {
                "status": "error",
                "content": [{"text": f"Camera '{name}' not found. Registered: {list(self._world.cameras)}"}],
            }
        with self._lock:
            del self._world.cameras[name]
        return {"status": "success", "content": [{"text": f"Camera '{name}' removed."}]}

    def list_cameras(self) -> list[str]:
        """Return all renderable camera names (the built-in ``'default'`` plus user cameras)."""
        return ["default", *(self._world.cameras if self._world else {})]

    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render the scene headlessly with Newton's ray-traced tiled camera.

        ``camera_name`` selects either the built-in three-quarter ``"default"``
        view (also ``None`` / ``""`` / ``"free"``) framing the world origin, or
        a named camera previously registered with :meth:`add_camera`. Named
        cameras render from their own eye/target/fov; world-fixed cameras use
        the stored pose directly, body-mounted cameras (``parent_body``) resolve
        their pose from the live body transform each call so a wrist camera
        tracks the arm.

        Args:
            camera_name: ``"default"`` (or ``None``/``""``/``"free"``) for the
                built-in view, or the name of a registered camera.
            width: Render width in pixels (defaults to the camera's width, or
                ``default_width`` for the built-in view).
            height: Render height in pixels (defaults to the camera's height, or
                ``default_height`` for the built-in view).

        Returns:
            Agent-tool dict with ``status`` and a ``content`` list. On success
            the content holds an ``image`` block carrying PNG bytes
            (``{"image": {"format": "png", "source": {"bytes": ...}}}``) plus a
            ``json`` block with pixel statistics, matching the MuJoCo backend so
            the shared ``PolicyRunner`` video pipeline consumes it unchanged.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        is_default = camera_name in FREE_CAMERA_TOKENS
        label = "default" if is_default else camera_name
        try:
            eye, target, fov_deg, w, h = self._resolve_camera_view(camera_name, width, height)
        except KeyError:
            return {
                "status": "error",
                "content": [{"text": f"Camera '{camera_name}' not found. Available: {self.list_cameras()}"}],
            }
        except ValueError as exc:
            return {"status": "error", "content": [{"text": f"Render failed: {exc}"}]}

        try:
            img = self._render_rgb(w, h, eye=eye, target=target, fov_deg=fov_deg)
        except Exception as exc:  # noqa: BLE001 - surface any render failure as a tool error
            return {"status": "error", "content": [{"text": f"Render failed: {exc}"}]}

        import io

        from PIL import Image

        buffer = io.BytesIO()
        Image.fromarray(img).save(buffer, format="PNG")
        png_bytes = buffer.getvalue()
        return {
            "status": "success",
            "content": [
                {"text": f"{w}x{h} from '{label}' at t={self._world.sim_time:.3f}s"},
                {"image": {"format": "png", "source": {"bytes": png_bytes}}},
                {
                    "json": {
                        "pixel_variance": float(np.var(img)),
                        "pixel_mean": float(np.mean(img)),
                        "camera": label,
                    }
                },
            ],
        }

    def _resolve_camera_pose(self, cam: SimCamera) -> tuple[tuple, tuple]:
        """Resolve a camera's world-frame ``(eye, target)`` from its :class:`SimCamera`.

        World-fixed cameras return their stored pose unchanged. Body-mounted
        cameras (``parent_body`` set) interpret ``position`` / ``target`` in the
        parent body's local frame and compose them with the live body transform
        so the camera tracks the body as it moves.

        Args:
            cam: The camera whose pose to resolve.

        Returns:
            Tuple of ``(eye, target)`` 3-tuples in world coordinates.

        Raises:
            ValueError: If the camera's mount body is no longer in the model.
        """
        if not cam.parent_body:
            return tuple(cam.position), tuple(cam.target)
        body_labels = list(self._model.body_label)
        if cam.parent_body not in body_labels:
            raise ValueError(f"camera '{cam.name}' mount body '{cam.parent_body}' is no longer in the model")
        idx = body_labels.index(cam.parent_body)
        with self._lock:
            tf = self._state_0.body_q.numpy()[idx]
        body_pos = np.asarray(tf[:3], dtype=np.float64)
        body_quat = np.asarray(tf[3:7], dtype=np.float64)  # warp xyzw
        eye = body_pos + self._rotate_vec_by_quat(body_quat, cam.position)
        target = body_pos + self._rotate_vec_by_quat(body_quat, cam.target)
        return tuple(eye.tolist()), tuple(target.tolist())

    @staticmethod
    def _rotate_vec_by_quat(q_xyzw: np.ndarray, v: Sequence[float]) -> np.ndarray:
        """Rotate a 3-vector by an ``(x, y, z, w)`` quaternion (Hamilton convention)."""
        x, y, z, w = (float(c) for c in q_xyzw)
        vv = np.asarray(v, dtype=np.float64)
        u = np.array([x, y, z], dtype=np.float64)
        rotated: np.ndarray = 2.0 * np.dot(u, vv) * u + (w * w - np.dot(u, u)) * vv + 2.0 * w * np.cross(u, vv)
        return rotated

    def _render_rgb(
        self,
        w: int,
        h: int,
        eye: Sequence[float] = (0.6, 0.6, 0.5),
        target: Sequence[float] = (0.0, 0.0, 0.15),
        fov_deg: float = 50.0,
    ) -> np.ndarray:
        """Render a view to an ``(H, W, 3)`` uint8 RGB ndarray.

        Args:
            w: Render width in pixels.
            h: Render height in pixels.
            eye: Camera position in world coordinates.
            target: Look-at point in world coordinates.
            fov_deg: Vertical field of view in degrees.

        Returns:
            Contiguous ``(H, W, 3)`` uint8 RGB array.

        Must be safe to call without holding ``self._lock`` from the caller;
        it acquires the lock internally.
        """
        with self._lock:
            sensors = self._nt.sensors
            cam = sensors.SensorTiledCamera(model=self._model)
            light_dir = self._wp.vec3f(*self._dr_light_dir) if self._dr_light_dir is not None else None
            cam.utils.create_default_light(enable_shadows=False, direction=light_dir)
            rays = cam.utils.compute_pinhole_camera_rays(w, h, math.radians(fov_deg))
            color = cam.utils.create_color_image_output(w, h, 1)
            q = self._look_at_quat(tuple(eye), tuple(target))
            wp = self._wp
            cam_tf = wp.array([[wp.transformf(wp.vec3f(*eye), wp.quatf(*q))]], dtype=wp.transformf)
            self._model.bvh_refit_shapes(self._state_0)
            cam.update(
                self._state_0,
                cam_tf,
                rays,
                color_image=color,
                clear_data=sensors.SensorTiledCamera.GRAY_CLEAR_DATA,
            )
            rgba = cam.utils.to_rgba_from_color(color).numpy()
        frame = rgba[0, 0] if rgba.ndim == 5 else rgba[0]
        frame = np.ascontiguousarray(frame[..., :3])
        return self._maybe_jitter_frame(frame)

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render a camera to raw ``(rgb, depth)`` ndarrays.

        Newton's ray-traced tiled camera has **no depth output path**, so the
        depth element is always ``None`` -- per the raw-frame API contract
        (issue #1537) this is explicit rather than a zero-filled buffer, and
        depth-dependent consumers (``HybridCompositor``) raise a clear error
        instead of silently compositing wrong pixels.

        Args:
            camera_name: ``"default"`` (or ``None``/``""``/``"free"``) for
                the built-in three-quarter view, or a camera registered via
                ``add_camera``.
            width: image width; ``None`` uses the camera's configured
                resolution (or the engine default for the built-in view).
            height: image height; same fallback as ``width``.

        Returns:
            ``(rgb, None)`` -- ``rgb`` is ``(H, W, 3) uint8``.

        Raises:
            RuntimeError: no world created.
            KeyError: unknown camera name.
            ValueError: the camera's mount body is no longer in the model.
        """
        if self._world is None or self._model is None:
            raise RuntimeError("No world. Call create_world first.")
        eye, target, fov_deg, w, h = self._resolve_camera_view(camera_name, width, height)
        rgb = self._render_rgb(w, h, eye=eye, target=target, fov_deg=fov_deg)
        return np.asarray(rgb, dtype=np.uint8), None

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return pinhole :class:`~strands_robots.rendering.CameraParams`.

        ``K`` is derived from the camera's vertical FOV with square pixels
        and a centered principal point (the same pinhole model
        ``_render_rgb`` traces rays with); ``T_world_cam`` is the OpenGL
        optical frame (+X right, +Y up, -Z forward) of the camera's live
        look-at pose (body-mounted cameras resolve against the current body
        transform). Newton exposes no scene clip planes; the params carry a
        conventional ``znear=0.01`` and a far plane "at infinity"
        (``zfar=1e6``) for the background-compositing depth convention.

        Args:
            camera_name: a camera registered via ``add_camera``. The built-in
                free view (``"default"``/``None``/``""``/``"free"``) is
                accepted and reports the fixed default vantage.
            width: image width to compute ``K`` for; ``None`` uses the
                camera's configured resolution.
            height: image height to compute ``K`` for.

        Raises:
            RuntimeError: no world created.
            KeyError: unknown camera name.
            ValueError: the camera's mount body is no longer in the model.
        """
        from strands_robots.rendering import CameraParams as _CameraParams

        if self._world is None or self._model is None:
            raise RuntimeError("No world. Call create_world first.")
        eye, target, fov_deg, w, h = self._resolve_camera_view(camera_name, width, height)

        fy = 0.5 * h / math.tan(math.radians(fov_deg) / 2.0)
        K = np.array([[fy, 0.0, 0.5 * w], [0.0, fy, 0.5 * h], [0.0, 0.0, 1.0]], dtype=np.float64)

        # Look-at basis in the OpenGL optical convention: -Z forward.
        fwd = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
        norm = float(np.linalg.norm(fwd))
        if norm < 1e-12:
            raise ValueError(f"camera '{camera_name}': eye and target coincide; look-at pose is undefined")
        fwd /= norm
        world_up = np.array([0.0, 0.0, 1.0])
        if abs(float(fwd @ world_up)) > 1.0 - 1e-6:
            world_up = np.array([0.0, 1.0, 0.0])  # straight up/down view: fall back
        right = np.cross(fwd, world_up)
        right /= np.linalg.norm(right)
        up = np.cross(right, fwd)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = np.stack([right, up, -fwd], axis=1)
        T[:3, 3] = np.asarray(eye, dtype=np.float64)
        return _CameraParams(K=K, T_world_cam=T, width=w, height=h, znear=0.01, zfar=1_000_000.0)

    def _resolve_camera_view(
        self, camera_name: str | None, width: int | None, height: int | None
    ) -> tuple[tuple[float, ...], tuple[float, ...], float, int, int]:
        """Resolve ``(eye, target, fov_deg, width, height)`` for a camera name.

        Shared by :meth:`render`, :meth:`get_frame` and
        :meth:`get_camera_params` so all three agree on the built-in default
        view and per-camera config fallbacks.

        A supplied ``width`` / ``height`` is validated on the same shared floor
        (:func:`~strands_robots.utils.positive_count_error`) that
        ``add_camera`` applies, so the config-time and call-time domains agree
        and every render entry point reports the same refusal. ``None`` means
        "take the camera's configured size"; membership decides that, not
        truthiness - reading ``0`` as omitted would render at the default
        resolution and report it as the requested one.

        Raises:
            KeyError: unknown camera name.
            ValueError: ``width`` / ``height`` is not a positive integer, or
                the camera's mount body is no longer in the model.
        """
        for param, value in (("width", width), ("height", height)):
            if value is not None and (text := positive_count_error(value, param, "render")) is not None:
                raise ValueError(text)
        if camera_name in FREE_CAMERA_TOKENS:
            return (
                (0.6, 0.6, 0.5),
                (0.0, 0.0, 0.15),
                50.0,
                self.default_width if width is None else width,
                self.default_height if height is None else height,
            )
        assert camera_name is not None  # the free-camera tokens were handled above
        cam = registry_entry(self._world.cameras, camera_name) if self._world is not None else None
        if cam is None:
            raise KeyError(f"Camera '{camera_name}' not found. Available: {self.list_cameras()}")
        eye, target = self._resolve_camera_pose(cam)
        return (
            eye,
            target,
            float(cam.fov),
            cam.width if width is None else width,
            cam.height if height is None else height,
        )

    # Interactive viewer

    _VIEWER_KINDS = ("auto", "gl", "viser", "null")

    @staticmethod
    def _display_available() -> bool:
        """Return True when a windowing display server is reachable.

        Mirrors the MuJoCo backend's headless check: on Linux an interactive
        OpenGL window needs ``DISPLAY`` (X11) or ``WAYLAND_DISPLAY`` (Wayland);
        macOS and Windows always have a native window server.
        """
        if sys.platform != "linux":
            return True
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))

    def open_viewer(
        self,
        viewer: str = "auto",
        *,
        port: int = 8080,
        width: int = 1280,
        height: int = 720,
    ) -> dict[str, Any]:
        """Open a live interactive viewer bound to the running Newton model.

        Brings the Newton backend to viewer parity with the MuJoCo backend's
        :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.open_viewer`.
        The viewer is fed one frame per control step from :meth:`_advance`, so
        it tracks the simulation live while ``step``, ``send_action``, or
        ``run_policy`` drive it from the calling thread.

        Args:
            viewer: Which Newton viewer to launch:

                * ``"gl"`` -- ``newton.viewer.ViewerGL`` interactive OpenGL
                  window; requires a display server.
                * ``"viser"`` -- ``newton.viewer.ViewerViser`` browser
                  dashboard served at ``http://localhost:<port>``; works
                  headless (no display required).
                * ``"null"`` -- ``newton.viewer.ViewerNull`` no-op sink (for
                  tests / benchmarks).
                * ``"auto"`` (default) -- ``"gl"`` when a display is present,
                  otherwise ``"viser"`` so headless hosts still get a live view.
            port: TCP port for the ``"viser"`` browser dashboard, an ``int``
                in ``[1, 65535]``. Read only by the ``"viser"`` branch, so
                ``"gl"`` and ``"null"`` ignore it. ``0`` is refused rather
                than forwarded as an ephemeral-bind request, because this
                surface advertises the requested port in the dashboard URL
                instead of reading the assigned one back.
            width: Window width in pixels for the ``"gl"`` viewer, an
                ``int`` ``>= 1`` on the same shared floor
                (:func:`~strands_robots.utils.positive_count_error`) that
                ``add_camera`` and the render family apply, so one pixel
                count cannot be refused for a frame and accepted for a
                window. Read only by the ``"gl"`` branch, so ``"viser"``
                and ``"null"`` ignore it.
            height: Window height in pixels, same domain as ``width``.

        Returns:
            Agent-tool ``status``/``content`` dict. On success the ``content``
            text reports the viewer kind and, for ``"viser"``, the dashboard
            URL.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        kind = viewer.lower().strip()
        if kind not in self._VIEWER_KINDS:
            return {
                "status": "error",
                "content": [{"text": f"Unknown viewer {viewer!r}. Available: {list(self._VIEWER_KINDS)}"}],
            }
        if self._viewer is not None:
            return {
                "status": "success",
                "content": [{"text": f"Viewer already open ({self._viewer_kind})."}],
            }
        has_display = self._display_available()
        if kind == "auto":
            kind = "gl" if has_display else "viser"
        if kind == "gl" and not has_display:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "Cannot open a 'gl' window: no display server (DISPLAY / "
                            "WAYLAND_DISPLAY unset). Use viewer='viser' for a headless "
                            "browser dashboard, or render(...) for offscreen frames."
                        )
                    }
                ],
            }
        # Only the viser branch reads ``port`` - it binds the dashboard and the
        # returned text advertises it - so the domain is applied on that branch
        # alone, as the policy providers apply theirs only when they dial.
        # Placed before the lock because a refusal must construct no viewer: the
        # viewer is a single slot, and a value forwarded verbatim that happens
        # not to raise inside ``ViewerViser`` fills it, after which the obvious
        # retry with a usable port is refused as "Viewer already open".
        if kind == "viser" and (port_error := tcp_port_error(port, "port", type(self).__name__)) is not None:
            return {"status": "error", "content": [{"text": port_error}]}
        # Only the gl branch reads ``width`` / ``height``, so the domain is
        # applied on that branch alone, exactly as ``port`` is applied on the
        # viser branch above. The floor is the one ``add_camera`` and the
        # render family already share, so a resolution this backend refuses
        # for a frame is not accepted for a window. Before the lock for the
        # same single-slot reason: a value forwarded verbatim that happens not
        # to raise inside ``ViewerGL`` fills the one viewer slot, and the
        # retry with a usable size is then answered "Viewer already open"
        # under ``status="success"`` - so the caller is left with the window
        # they did not ask for and no way to replace it.
        if kind == "gl":
            for _param, _value in (("width", width), ("height", height)):
                if (dim_error := positive_count_error(_value, _param, "open_viewer")) is not None:
                    return {"status": "error", "content": [{"text": dim_error}]}
        with self._lock:
            try:
                vmod = self._nt.viewer
                if kind == "gl":
                    handle = vmod.ViewerGL(width=width, height=height)
                elif kind == "viser":
                    handle = vmod.ViewerViser(port=port)
                else:
                    handle = vmod.ViewerNull()
                handle.set_model(self._model)
                self._viewer = handle
                self._viewer_kind = kind
                # Prime one frame so the initial pose is visible immediately.
                self._sync_viewer()
            except Exception as exc:  # noqa: BLE001 - surface any viewer launch failure as a tool error
                self._viewer = None
                self._viewer_kind = None
                return {"status": "error", "content": [{"text": f"Viewer launch failed: {exc}"}]}
        text = f"Newton '{kind}' viewer opened."
        if kind == "viser":
            url = getattr(self._viewer, "url", None) or f"http://localhost:{port}"
            text = f"Newton 'viser' viewer streaming at {url}"
        return {"status": "success", "content": [{"text": text}]}

    def _sync_viewer(self) -> None:
        """Push the current state to the open viewer (one frame).

        Must be called with ``self._lock`` held. A no-op when no viewer is
        open. If the viewer window has been closed by the user, or logging a
        frame raises, the handle is released so stepping continues unimpeded.
        """
        if self._viewer is None or self._world is None or self._state_0 is None:
            return
        try:
            if not self._viewer.is_running():
                self._close_viewer()
                return
            self._viewer.begin_frame(self._world.sim_time)
            self._viewer.log_state(self._state_0)
            self._viewer.end_frame()
        except Exception as exc:  # noqa: BLE001 - a dead viewer must never break stepping
            logger.warning("Newton viewer sync failed, closing viewer: %s", exc)
            self._close_viewer()

    def _close_viewer(self) -> None:
        """Close and release the viewer handle if one is open (lock-agnostic)."""
        if self._viewer is not None:
            try:
                self._viewer.close()
            except Exception as exc:  # noqa: BLE001 - best-effort teardown
                logger.warning("Newton viewer close raised (ignored): %s", exc)
            self._viewer = None
            self._viewer_kind = None

    def close_viewer(self) -> dict[str, Any]:
        """Close the interactive viewer if one is open.

        Returns:
            Always a success status dict; closing when no viewer is open is a
            no-op.
        """
        with self._lock:
            self._close_viewer()
        return {"status": "success", "content": [{"text": "Viewer closed."}]}

    # Robot / scene discovery

    def _free_base_pose(self, robot_name: str, joint_q: Any, joint_qd: Any) -> dict[str, list[float]] | None:
        """Return a robot's floating-base 6-DoF pose + twist, or None.

        For a robot whose root is a free joint (a humanoid's named
        ``floating_base_joint``), returns a dict with ``position`` (xyz),
        ``quaternion`` (w,x,y,z), ``linear_velocity`` (world frame) and
        ``angular_velocity`` (BODY frame), mirroring the MuJoCo backend's
        ``base`` entry. Newton stores the free joint's coordinates as
        ``[xyz, quat_xyzw]`` and its DOFs as ``[linear(3), angular(3)]`` in the
        WORLD frame, so the quaternion is reordered ``xyzw -> wxyz`` to match the
        MuJoCo (w,x,y,z) contract, the linear velocity maps directly (world on
        both backends), and the angular velocity is rotated world -> body so it
        matches MuJoCo's body-frame convention (the IMU-gyro frame WBC /
        locomotion controllers consume). Returns None for a fixed-base robot.

        Args:
            robot_name: The robot to query.
            joint_q: The world's ``joint_q`` array (already ``.numpy()``).
            joint_qd: The world's ``joint_qd`` array (already ``.numpy()``).

        Returns:
            The base pose/twist dict, or None when the robot has no free root.
        """
        base_jname = self._robot_free_base_joint.get(robot_name)
        if base_jname is None:
            return None
        q = self._joint_coord_index.get((robot_name, base_jname))
        d = self._joint_dof_index.get((robot_name, base_jname))
        if q is None or d is None or q + 7 > len(joint_q) or d + 6 > len(joint_qd):
            return None
        quat_wxyz = [
            float(joint_q[q + 6]),
            float(joint_q[q + 3]),
            float(joint_q[q + 4]),
            float(joint_q[q + 5]),
        ]
        # Newton stores the free joint's angular velocity in the WORLD frame,
        # while the MuJoCo backend reports it in the BODY frame (MuJoCo's
        # free-joint qvel angular block is local) - the IMU-gyro convention WBC /
        # locomotion controllers consume via get_observation's base_ang_vel.
        # Rotate world -> body so base_ang_vel agrees across both backends; the
        # linear velocity stays world-frame on both, so it maps directly.
        ang_world = [float(joint_qd[d + 3]), float(joint_qd[d + 4]), float(joint_qd[d + 5])]
        return {
            "position": [float(v) for v in joint_q[q : q + 3]],
            "quaternion": quat_wxyz,
            "linear_velocity": [float(v) for v in joint_qd[d : d + 3]],
            "angular_velocity": _quat_rotate_inverse_wxyz(quat_wxyz, ang_world),
        }

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        """Return per-joint position and velocity for a robot.

        Mirrors the MuJoCo backend: the ``json`` block carries a ``state``
        mapping of short joint name to ``{"position", "velocity"}`` (radians
        and radians/second for revolute joints). Positions are read from
        ``joint_q`` via the per-joint coordinate index and velocities from
        ``joint_qd`` via the per-joint DOF index (the two indices differ once
        a free-floating object adds a quaternion coordinate).

        A robot with a floating base (a 6-DoF free root, e.g. a humanoid's
        named ``floating_base_joint``) additionally carries a ``"base"`` entry
        with ``position`` (xyz), ``quaternion`` (w,x,y,z), ``linear_velocity``
        and ``angular_velocity``. The free joint is NOT reported as a scalar
        joint (its coordinates are [xyz + quat], not a single angle), and the
        base ``quaternion``/``angular_velocity`` match get_observation's
        ``base_quat``/``base_ang_vel`` for the same robot (``angular_velocity``
        is in the BODY frame, ``linear_velocity`` in the WORLD frame, matching
        the MuJoCo backend).

        Args:
            robot_name: Robot to query. ``None`` resolves to the sole robot
                when exactly one exists.

        Returns:
            Agent-tool dict with a human-readable ``text`` block and a
            ``json`` block ``{"state": {joint: {"position", "velocity"}}}``.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        with self._lock:
            joint_q = self._state_0.joint_q.numpy()
            joint_qd = self._state_0.joint_qd.numpy()
            state: dict[str, dict[str, float]] = {}
            base_jname = self._robot_free_base_joint.get(robot_name)
            for jname in self._world.robots[robot_name].joint_names:
                # A FREE joint (6-DoF floating base) has no scalar hinge value:
                # its coordinates are [xyz + quaternion]. Reading joint_q[start]
                # as a "position" reports the base x-coordinate and drops the
                # orientation, so skip it and surface a structured ``base`` below.
                if jname == base_jname:
                    continue
                q_idx = self._joint_coord_index.get((robot_name, jname))
                d_idx = self._joint_dof_index.get((robot_name, jname))
                pos = float(joint_q[q_idx]) if q_idx is not None and q_idx < len(joint_q) else 0.0
                vel = float(joint_qd[d_idx]) if d_idx is not None and d_idx < len(joint_qd) else 0.0
                state[jname] = {"position": pos, "velocity": vel}
        state = self._apply_state_noise(state)
        # Floating base: surface the full 6-DoF pose + twist under a ``base``
        # key (sibling of ``state``, left un-noised), consistent with
        # get_observation's base_quat / base_ang_vel and the MuJoCo backend.
        base = self._free_base_pose(robot_name, joint_q, joint_qd)

        text = f"'{robot_name}' state (t={self._world.sim_time:.3f}s):\n"
        for jnt, vals in state.items():
            text += f"{jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"
        if base is not None:
            p_, q_ = base["position"], base["quaternion"]
            lv_, av_ = base["linear_velocity"], base["angular_velocity"]
            text += (
                f"base: pos=[{p_[0]:.4f}, {p_[1]:.4f}, {p_[2]:.4f}], "
                f"quat=[{q_[0]:.4f}, {q_[1]:.4f}, {q_[2]:.4f}, {q_[3]:.4f}], "
                f"lin_vel=[{lv_[0]:.4f}, {lv_[1]:.4f}, {lv_[2]:.4f}], "
                f"ang_vel=[{av_[0]:.4f}, {av_[1]:.4f}, {av_[2]:.4f}]\n"
            )
        json_payload: dict[str, Any] = {"state": state}
        if base is not None:
            json_payload["base"] = base
        return {"status": "success", "content": [{"text": text}, {"json": json_payload}]}

    def list_robots_info(self) -> dict[str, Any]:
        """Pretty-printed robot listing (dict-shaped, for agent display).

        Distinct from :meth:`list_robots` (which returns ``list[str]`` for the
        SimEngine ABC). Mirrors the MuJoCo backend's per-robot summary.

        Returns:
            Agent-tool dict whose ``text`` block lists each robot's asset,
            world position, joint count, and config.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not self._world.robots:
            return {"status": "success", "content": [{"text": "No robots. Use add_robot."}]}
        lines = ["Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            lines.append(
                f"  - {name} ({Path(robot.urdf_path).name})\n"
                f"    Position: {robot.position}, Joints: {len(robot.joint_names)}, "
                f"Config: {robot.data_config or 'direct'}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def list_bodies(self, robot_name: str | None = None) -> dict[str, Any]:
        """List Newton body labels, optionally scoped to one robot.

        This is the discovery surface for resolving a robot's end-effector /
        mount body without guessing. Newton labels bodies by their full MJCF
        path (``so_arm100/.../Moving_Jaw``); the ``json`` block returns the
        full labels and, when ``robot_name`` is given, a best-guess
        ``gripper_body`` one of whose trailing segment's *name components* is
        ``gripper``, ``hand``, ``jaw``, ``ee``, or ``tool``. Hints match
        components rather than bare substrings, so a short hint cannot fire
        inside an unrelated word: a ``knee`` or a drive ``wheel`` is not an
        end-effector because ``ee`` occurs in its name.

        Args:
            robot_name: When set, return only that robot's bodies. When
                omitted, return every body label in the world.

        Returns:
            Agent-tool dict with a ``text`` block and a ``json`` block
            ``{"bodies": [...]}`` (plus ``"gripper_body"`` when scoped).
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        if robot_name is not None:
            if not registered(self._world.robots, robot_name):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"list_bodies: robot '{robot_name}' not found. "
                                f"Known robots: {list(self._world.robots.keys())}"
                            )
                        }
                    ],
                }
            bodies = list(self._robot_body_map.get(robot_name, []))
        else:
            bodies = list(self._model.body_label)

        json_payload: dict[str, Any] = {"bodies": bodies}
        if robot_name is not None:
            gripper_body: str | None = None
            for name in bodies:
                short = _short_joint_name(name)
                if any(hint_matches_name(hint, short) for hint in _GRIPPER_BODY_HINTS):
                    gripper_body = name
                    break
            json_payload["gripper_body"] = gripper_body

        text = "Bodies:\n" + "\n".join(f"  - {b}" for b in bodies) if bodies else "No bodies."
        return {"status": "success", "content": [{"text": text}, {"json": json_payload}]}

    def get_features(self, robot_name: str | None = None) -> dict[str, Any]:
        """Describe the model's joints / bodies / robots for the agent.

        Mirrors the MuJoCo backend's ``features`` json schema with values
        sourced from the finalized Newton model. Newton drives joints through
        position targets rather than named MuJoCo actuators, so
        ``actuator_names`` echoes the robot's joint names and ``camera_names``
        is the single headless ``"default"`` view.

        Args:
            robot_name: When set, scope joint / robot listings to that robot.

        Returns:
            Agent-tool dict with a ``text`` summary and a ``json`` block
            ``{"features": {...}}``.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        m = self._model
        if robot_name is not None:
            if not registered(self._world.robots, robot_name):
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
            joint_names = list(self._world.robots[robot_name].joint_names)
            scoped = {robot_name: self._world.robots[robot_name]}
        else:
            joint_names = list(self._joint_order)
            scoped = dict(self._world.robots)

        robots_info = {
            rname: {
                "joint_names": list(robot.joint_names),
                "n_joints": len(robot.joint_names),
                "data_config": robot.data_config,
                "source": Path(robot.urdf_path).name,
            }
            for rname, robot in scoped.items()
        }
        actuator_names = list(joint_names)
        camera_names = ["default"]
        features = {
            "n_bodies": int(m.body_count),
            "n_joints": int(m.joint_count),
            "n_dofs": int(m.joint_dof_count),
            "timestep": float(self._world.timestep),
            "solver": self._solver_name,
            "joint_names": joint_names,
            "actuator_names": actuator_names,
            "camera_names": camera_names,
            "robots": robots_info,
        }
        lines = [
            "Simulation Features (Newton)",
            f"Joints ({m.joint_count}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"DOFs: {m.joint_dof_count} | Bodies: {m.body_count}",
            f"Solver: {self._solver_name} | Cameras: default",
            f"Timestep: {self._world.timestep}s ({1 / self._world.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(f"{rname}: {rinfo['n_joints']} joints ({rinfo['source']})")
        return {"status": "success", "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}]}

    def list_urdfs(self) -> dict[str, Any]:
        """List robot assets resolvable by this backend.

        Returns the union of the curated registry / MJCF asset manager listing
        and the URDF long tail discoverable through ``robot_descriptions``.
        Because Newton loads URDF natively, the URDF-only descriptions (which the
        MuJoCo backend cannot use) are first-class here.

        Returns:
            Agent-tool dict whose ``text`` block is the formatted registry
            listing plus the ``robot_descriptions`` URDF names, and whose
            ``json`` block exposes ``robot_descriptions_urdf`` (the sorted URDF
            long tail) for programmatic use.
        """
        urdf_robots = list_urdf_discoverable()
        text = list_available_models()
        if urdf_robots:
            text += (
                "\n\nURDF-native via robot_descriptions "
                f"(Newton loads through add_urdf, {len(urdf_robots)} robots):\n  " + ", ".join(urdf_robots)
            )
        return {
            "status": "success",
            "content": [
                {"text": text},
                {"json": {"robot_descriptions_urdf": urdf_robots}},
            ],
        }

    def register_urdf(self, data_config: str, urdf_path: str) -> dict[str, Any]:
        """Register an MJCF/URDF asset under ``data_config`` in the registry.

        Validates ``urdf_path`` before handing it to the shared registry so a
        missing or unreadable file surfaces a clear error here rather than
        deep inside the Newton MJCF importer.

        Args:
            data_config: Registry name to bind the asset to.
            urdf_path: Filesystem path to the MJCF/URDF asset.

        Returns:
            Status dict reporting the resolved path, or an error dict when the
            file is missing / unreadable.
        """
        if not urdf_path:
            return {"status": "error", "content": [{"text": "register_urdf: 'urdf_path' must be a non-empty string."}]}
        path = Path(urdf_path)
        if not path.exists():
            return {"status": "error", "content": [{"text": f"register_urdf: file not found: {urdf_path}"}]}
        if not path.is_file():
            return {"status": "error", "content": [{"text": f"register_urdf: not a file: {urdf_path}"}]}
        try:
            with path.open("rb"):
                pass
        except OSError as exc:
            return {"status": "error", "content": [{"text": f"register_urdf: cannot read {urdf_path}: {exc}"}]}
        _register_urdf(data_config, urdf_path)
        resolved = resolve_model(data_config)
        return {
            "status": "success",
            "content": [{"text": f"Registered '{data_config}' -> {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"}],
        }

    # Introspection

    def describe(self) -> dict[str, Any]:
        """Return a discovery surface describing this backend's capabilities.

        This surface is built from scratch rather than by extending
        :meth:`~strands_robots.simulation.base.SimEngine.describe`, so every
        method the base contract advertises and this backend delivers has to be
        listed here explicitly -- including the facades inherited unchanged from
        the ABC. Extending the base surface instead would also advertise
        ``get_contacts`` and ``load_scene``, which this backend does not
        implement.

        Returns:
            Dict with the backend name, active solver, the solvers that can
            drive an articulated robot (the accepted domain of ``solver=``),
            device, current robot / object counts, and the ``methods`` mapping a
            caller enumerates to find a capability without guessing its name.
        """
        device = str(self._wp.get_device(self.device)) if self.device else str(self._wp.get_device())
        bodies = list(self._model.body_label) if self._model is not None else []
        return {
            "backend": "newton",
            "solver": self._solver_name,
            "available_solvers": sorted(articulated_solvers()),
            "device": device,
            "robots": self.list_robots(),
            "objects": list(self._world.objects) if self._world else [],
            "bodies": bodies,
            "cameras": self.list_cameras(),
            "timestep": self._world.timestep if self._world else self.default_timestep,
            "gravity": list(self._world.gravity) if self._world else [0.0, 0.0, -9.81],
            "world_created": self._world is not None and self._model is not None,
            "methods": {
                "add_robot": (
                    "(name: str, urdf_path=None, position=None, orientation=None, source=None) -> dict  "
                    "(source: None=registry then robot_descriptions URDF fallback, "
                    "'registry'=registry only, 'robot_descriptions'=URDF via "
                    "robot_descriptions.<name>_description.URDF_PATH; see list_urdfs)"
                ),
                "add_object": (
                    "(name: str, shape='box', position=None, orientation=None, size=None, "
                    "color=None, mass=0.1, is_static=None, mesh_path=None) -> dict  "
                    "(add a manipulable object -- box/sphere/.../mesh -- to the scene)"
                ),
                "remove_robot": (
                    "(name: str) -> dict  (remove a robot and rebuild the world; the inverse "
                    "of add_robot, completing the add/remove pair alongside remove_object)"
                ),
                "remove_object": "(name: str) -> dict  (remove a previously added object)",
                "get_robot_state": "(robot_name: str | None = None) -> dict (per-joint position + velocity)",
                "get_state": (
                    "() -> dict (whole-world snapshot: sim time, step count, timestep, gravity, "
                    "robot / object / camera / body / joint / actuator counts)"
                ),
                "get_observation": "(robot_name: str | None = None, *, skip_images: bool = False) -> dict",
                "send_action": (
                    "(action: dict, robot_name: str | None = None, n_substeps: int = 1) -> dict"
                    "  # n_substeps must be a positive whole number; use step() to advance without commanding"
                ),
                "run_policy": "(robot_name: str, policy_provider='mock', n_episodes=1, ...) -> dict",
                "eval_policy": (
                    "(robot_name: str | None = None, policy_provider='mock', n_episodes=1, "
                    "max_steps=300, success_fn=None, seed=None, ...) -> dict  (multi-episode "
                    "success-rate evaluation -- the scoring sibling of run_policy)"
                ),
                "start_policy": (
                    "(robot_name: str | None = None, policy_provider='mock', duration=10.0, ...) -> dict  "
                    "(the base contract's synchronous passthrough to run_policy on this backend; "
                    "only MuJoCo runs a policy on a background thread)"
                ),
                "replay_episode": (
                    "(repo_id: str, robot_name=None, episode=0, root=None, speed=1.0, "
                    "action_key_map=None) -> dict  (replay a recorded LeRobotDataset episode "
                    "through the sim)"
                ),
                "evaluate_benchmark": (
                    "(benchmark_name: str, robot_name=None, policy_provider='mock', "
                    "n_episodes=1, seed=None, video=None, ...) -> dict  (score a registered "
                    "success/failure/dense_reward benchmark over a rollout; max_steps comes "
                    "from the benchmark, not a parameter)"
                ),
                "register_builtin_benchmarks": (
                    "() -> dict  (register the shipped built-in velocity-tracking locomotion "
                    "benchmarks so they appear in list_benchmarks and can be run via "
                    "evaluate_benchmark)"
                ),
                "list_benchmarks": (
                    "() -> dict  (enumerate registered benchmarks -- names, supported robots, "
                    "default robot, max_steps -- the source of the benchmark_name evaluate_benchmark expects)"
                ),
                "register_benchmark_from_file": (
                    "(benchmark_name: str, spec_path: str) -> dict  (author a declarative "
                    "success/failure/dense_reward benchmark spec as YAML/JSON at runtime and register it)"
                ),
                "list_robots": "() -> list[str]  (ordered robot names -- what describe() reports under 'robots')",
                "list_robots_info": "() -> dict (pretty robot listing)",
                "list_bodies": "(robot_name: str | None = None) -> dict (body labels + gripper_body)",
                "list_objects": "() -> dict",
                "move_object": "(name: str, position=None, orientation=None) -> dict",
                "get_features": "(robot_name: str | None = None) -> dict (joints/bodies/robots schema)",
                "list_urdfs": "() -> dict (registry + robot_descriptions URDF long tail; json.robot_descriptions_urdf)",
                "register_urdf": "(data_config: str, urdf_path: str) -> dict",
                "set_gravity": "(gravity: list[float] | float) -> dict",
                "set_timestep": "(timestep: float) -> dict (seconds; takes effect on the next step)",
                "render": "(camera_name='default', width=None, height=None) -> dict (named camera or 'default')",
                "add_camera": (
                    "(name: str, position=None, target=None, fov=60.0, width=640, height=480, "
                    "parent_body=None) -> dict  (register a named camera; parent_body mounts it "
                    "on a body for a wrist cam -- see list_bodies)"
                ),
                "remove_camera": "(name: str) -> dict  (remove a registered named camera)",
                "list_cameras": "() -> list[str]  (renderable camera names incl. 'default')",
                "open_viewer": (
                    "(viewer='auto'|'gl'|'viser'|'null', *, port=8080, width=1280, height=720) -> dict  "
                    "(open a live interactive viewer; 'gl' window needs a display, 'viser' "
                    "streams a browser dashboard at localhost:port headless)"
                ),
                "close_viewer": "() -> dict  (close the interactive viewer)",
                "randomize": (
                    "(randomize_colors=True, randomize_lighting=True, randomize_physics=False, "
                    "mass_range=(0.5, 2.0), friction_range=(0.5, 1.5), color_range=(0.1, 1.0), "
                    "seed=None) -> dict (domain randomization; json block carries applied multipliers)"
                ),
                "set_obs_noise": (
                    "(joint_pos_std=0.0, joint_vel_std=0.0, camera_jitter_px=0.0, seed=None) -> dict "
                    "(additive Gaussian sensor noise on observations + rendered frames)"
                ),
                "create_world": (
                    "(timestep=None, gravity=None, ground_plane=True, terrain=None, "
                    "difficulty=1.0) -> dict  (create a fresh simulation world - the "
                    "world-lifecycle entry point that precedes add_robot / add_object; "
                    "gravity is [gx, gy, gz], ground_plane lays a floor)"
                ),
                "destroy": (
                    "() -> dict  (tear down the world and release all resources, joining "
                    "any running background policy first; the inverse of create_world)"
                ),
                "reset": "() -> dict (restore baseline joint configuration)",
                "step": "(n_steps: int = 1) -> dict",
                "start_recording": (
                    "(repo_id='local/sim_recording', task='', fps=30, root=None, "
                    "push_to_hub=False, vcodec='h264', overwrite=False) -> dict  "
                    "(record joint state + action + named cameras to a LeRobotDataset)"
                ),
                "save_episode": (
                    "() -> dict  (flush the current rollout as one episode; call once per "
                    "run_policy to get N episodes instead of one merged episode. Prefer "
                    "run_policy(n_episodes=N) which flushes a boundary per episode)"
                ),
                "stop_recording": "(push_to_hub=False, bucket=None, run_id=None) -> dict",
                "get_recording_status": "() -> dict",
                "verify_dataset_episodes": (
                    "(expected: int) -> dict  (after stop_recording, read the parquet and "
                    "confirm the dataset holds exactly `expected` episodes; status=error on mismatch)"
                ),
            },
            "note": (
                "robot_name defaults to the sole robot when only one exists for "
                "get_observation, send_action, get_robot_state, get_features, run_policy. "
                "With multiple robots, pass robot_name explicitly (from 'robots')."
            ),
        }

    def cleanup(self, policy_stop_timeout: float | None = None) -> None:
        """Release resources (alias for :meth:`destroy`)."""
        self.destroy()

    def __enter__(self) -> NewtonSimEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.destroy()

    # Internal helpers

    def _look_at_quat(self, eye: Sequence[float], target: Sequence[float], up: Sequence[float] = (0, 0, 1)) -> tuple:
        """Build an (x, y, z, w) quaternion orienting a camera at ``eye`` to look at ``target``.

        Uses an OpenGL-style camera frame (camera looks down its local -Z).

        When ``up`` is parallel to the view axis (e.g. a top-down camera
        directly above its target with the default world-up), an alternate up
        vector is chosen so the basis stays well-defined instead of producing a
        NaN quaternion.

        Args:
            eye: Camera position.
            target: Point the camera looks at.
            up: World up vector.

        Returns:
            Quaternion as ``(x, y, z, w)`` for ``warp.quatf``.

        Raises:
            ValueError: If ``eye`` and ``target`` coincide, leaving the view
                direction undefined.
        """
        e = np.asarray(eye, dtype=np.float64)
        t = np.asarray(target, dtype=np.float64)
        u = np.asarray(up, dtype=np.float64)
        f = t - e
        f_norm = np.linalg.norm(f)
        if f_norm < 1e-9:
            raise ValueError(f"look_at: eye and target coincide ({eye!r}); camera direction is undefined.")
        f /= f_norm
        z = -f
        x = np.cross(u, z)
        # When ``up`` is (near-)parallel to the view axis -- e.g. a top-down
        # camera placed directly above its target with the default world-up --
        # ``cross(up, z)`` collapses to ~0 and normalising it would yield a
        # NaN quaternion (a silently garbage camera pose). Fall back to an
        # alternate up vector that is guaranteed non-parallel to ``z``.
        if np.linalg.norm(x) < 1e-6:
            alt_up = np.array([1.0, 0.0, 0.0]) if abs(z[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            x = np.cross(alt_up, z)
        x /= np.linalg.norm(x)
        y = np.cross(z, x)
        r = np.stack([x, y, z], axis=1)
        return self._quat_from_matrix(r)

    @staticmethod
    def _quat_from_matrix(r: np.ndarray) -> tuple:
        """Convert a 3x3 rotation matrix to an (x, y, z, w) quaternion."""
        trace = np.trace(r)
        if trace > 0:
            s = math.sqrt(trace + 1.0) * 2
            w = 0.25 * s
            x = (r[2, 1] - r[1, 2]) / s
            y = (r[0, 2] - r[2, 0]) / s
            z = (r[1, 0] - r[0, 1]) / s
        elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
            s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2
            w = (r[2, 1] - r[1, 2]) / s
            x = 0.25 * s
            y = (r[0, 1] + r[1, 0]) / s
            z = (r[0, 2] + r[2, 0]) / s
        elif r[1, 1] > r[2, 2]:
            s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2
            w = (r[0, 2] - r[2, 0]) / s
            x = (r[0, 1] + r[1, 0]) / s
            y = 0.25 * s
            z = (r[1, 2] + r[2, 1]) / s
        else:
            s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2
            w = (r[1, 0] - r[0, 1]) / s
            x = (r[0, 2] + r[2, 0]) / s
            y = (r[1, 2] + r[2, 1]) / s
            z = 0.25 * s
        return (x, y, z, w)

    def _advance(self, n_steps: int) -> None:
        """Run ``n_steps`` control steps (each ``substeps`` solver steps).

        Must be called with ``self._lock`` held.

        The ``max(1, n_steps)`` floor is defensive only: both public callers -
        :meth:`step` and :meth:`send_action` - refuse or answer a non-positive
        count before reaching here, so the floor cannot be observed through
        either. It is kept rather than removed so a future internal caller
        cannot advance a negative count, and it is not a contract: a caller
        wanting "advance nothing" gets it from ``step(0)``, which returns
        without calling this method at all.
        """
        assert self._world is not None
        if self._solver is None:
            self._world.step_count += max(1, n_steps)
            self._world.sim_time += self._world.timestep * max(1, n_steps)
            self._sync_viewer()
            return
        dt = self._world.timestep / self.substeps
        for _ in range(max(1, n_steps)):
            for _ in range(self.substeps):
                self._state_0.clear_forces()
                self._solver.step(self._state_0, self._state_1, self._control, None, dt)
                self._state_0, self._state_1 = self._state_1, self._state_0
            self._world.sim_time += self._world.timestep
            self._world.step_count += 1
        self._sync_viewer()

    def _apply_gravity(self) -> None:
        """Write the world's gravity vec3 onto the finalized model.

        Must be called with ``self._lock`` held, after ``builder.finalize`` and
        before the solver is constructed (Newton solvers snapshot gravity at
        construction). Newton stores one gravity vec3 per parallel world; the
        same world-gravity vector is broadcast to each of them. A no-op when no
        model or gravity buffer exists yet.
        """
        assert self._world is not None and self._model is not None
        grav = self._model.gravity
        if grav is None:
            return
        n_worlds = grav.shape[0]
        vec = np.tile(np.asarray(self._world.gravity, dtype=np.float32), (n_worlds, 1))
        self._model.gravity = self._wp.array(vec, dtype=self._wp.vec3, device=self._model.device)

    def _write_targets(self) -> None:
        """Push pending position targets into the Newton control buffer.

        Must be called with ``self._lock`` held.
        """
        if self._control is None or self._control.joint_target_q is None:
            return
        tgt = self._control.joint_target_q.numpy()
        for (robot_name, jname), value in self._targets.items():
            idx = self._joint_coord_index.get((robot_name, jname))
            if idx is not None and idx < len(tgt):
                tgt[idx] = value
        self._control.joint_target_q = self._wp.array(tgt, dtype=self._wp.float32, device=self._model.device)

    def _rebuild(self) -> None:
        """(Re)build the Newton model from the current world state.

        Newton finalises an immutable model from a builder, so every scene
        mutation triggers a full rebuild. Joint targets that still reference
        existing joints are preserved. Must be called with ``self._lock`` held.
        """
        assert self._world is not None
        nt, wp = self._nt, self._wp
        builder = nt.ModelBuilder()
        solver_cls = resolve_solver_class(self._solver_name)
        if hasattr(solver_cls, "register_custom_attributes"):
            solver_cls.register_custom_attributes(builder)

        # Track coordinate index of each robot's joints in the global joint_q.
        self._joint_coord_index = {}
        self._joint_dof_index = {}
        self._robot_joint_map = {}
        self._robot_free_base_joint = {}
        self._robot_body_map = {}

        for robot_name, robot in self._world.robots.items():
            label_before = len(builder.joint_label)
            body_before = builder.body_count
            xform = wp.transform(
                wp.vec3(*robot.position),
                self._wxyz_to_wp_quat(robot.orientation),
            )
            model_path = str(robot.urdf_path)
            if model_path.lower().endswith(".urdf"):
                # Newton ingests URDF natively; floating=None leaves the root
                # fixed to the world (correct for table-mounted arms like panda).
                builder.add_urdf(model_path, xform=xform, collapse_fixed_joints=True)
            else:
                builder.add_mjcf(model_path, xform=xform, collapse_fixed_joints=True)
            new_labels = builder.joint_label[label_before:]
            # Map each joint to its coordinate index in joint_q and its DOF index
            # in joint_qd. These are NOT the joint's ordinal position: a floating
            # base (a free joint) spans 7 coordinates (xyz + quaternion) and 6
            # DOFs, a ball joint 4/3, while a revolute/prismatic joint spans 1/1.
            # So once a robot has a multi-coordinate joint (e.g. a humanoid whose
            # root is a free joint), every joint after it is offset. Read the
            # authoritative per-joint coordinate/DOF starts the builder tracks
            # (joint_q_start / joint_qd_start) instead of assuming one coordinate
            # and one DOF per joint - otherwise get_observation and
            # get_robot_state report the base coordinates for the first child
            # joint and shift the reading of every joint after it.
            q_start = builder.joint_q_start
            qd_start = builder.joint_qd_start
            short_names: list[str] = []
            for offset, label in enumerate(new_labels):
                short = _short_joint_name(label)
                short_names.append(short)
                joint_index = label_before + offset
                self._joint_coord_index[(robot_name, short)] = int(q_start[joint_index])
                self._joint_dof_index[(robot_name, short)] = int(qd_start[joint_index])
                # A free root (6-DoF floating base) is not a scalar joint: its
                # coordinates are [xyz + quaternion]. Remember it so state/obs
                # queries surface a structured base pose (see _free_base_pose).
                if (
                    builder.joint_type[joint_index] == self._nt.JointType.FREE
                    and robot_name not in self._robot_free_base_joint
                ):
                    self._robot_free_base_joint[robot_name] = short
            self._robot_joint_map[robot_name] = short_names
            self._robot_body_map[robot_name] = list(builder.body_label[body_before:])

        for obj in self._world.objects.values():
            self._add_object_to_builder(builder, obj)

        if self._world.ground_plane:
            builder.add_ground_plane()

        # Apply active domain randomization (mass/friction/colors) to the
        # builder arrays before the immutable model is finalized.
        self._apply_domain_randomization(builder)

        self._model = builder.finalize(device=self.device)
        # Newton solvers snapshot gravity at construction, and ModelBuilder only
        # expresses gravity as a scalar magnitude along its up-axis (silently
        # dropping any non-axis-aligned component). Write the world's full
        # gravity vec3 onto the finalized model BEFORE building the solver so an
        # arbitrary gravity vector configured via create_world / set_gravity is
        # honoured instead of being ignored.
        self._apply_gravity()
        # Rigid-body solvers (notably SolverMuJoCo) require at least one joint.
        # An empty world (ground plane only) has none, so defer solver creation
        # until a robot is added; stepping is a no-op until then.
        self._solver = solver_cls(self._model) if self._model.joint_dof_count > 0 else None
        self._state_0 = self._model.state()
        self._state_1 = self._model.state()
        self._control = self._model.control()
        nt.eval_fk(self._model, self._model.joint_q, self._model.joint_qd, self._state_0)
        self._joint_order = [name for names in self._robot_joint_map.values() for name in names]

        # Re-apply targets that still reference live joints.
        self._targets = {k: v for k, v in self._targets.items() if k in self._joint_coord_index}
        self._write_targets()

        # An open viewer holds the previous (now stale) model; rebind it to
        # the freshly finalized model so it keeps tracking the live scene.
        if self._viewer is not None:
            try:
                self._viewer.set_model(self._model)
            except Exception as exc:  # noqa: BLE001 - never let a dead viewer break a rebuild
                logger.warning("Newton viewer rebind failed, closing viewer: %s", exc)
                self._close_viewer()

    def _add_object_to_builder(self, builder: Any, obj: SimObject) -> None:
        """Add one :class:`SimObject` primitive to a Newton builder."""
        wp = self._wp
        xform = wp.transform(wp.vec3(*obj.position), self._wxyz_to_wp_quat(obj.orientation))
        if obj.is_static or obj.mass <= 0:
            body = -1
        else:
            body = builder.add_body(xform=xform, mass=obj.mass)
        shape_xform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat_identity()) if body >= 0 else xform
        color = tuple(obj.color[:3])
        size = obj.size
        if obj.shape == "box":
            hx, hy, hz = (size + [0.05, 0.05, 0.05])[:3]
            builder.add_shape_box(body, xform=shape_xform, hx=hx, hy=hy, hz=hz, color=color)
        elif obj.shape == "sphere":
            builder.add_shape_sphere(body, xform=shape_xform, radius=size[0], color=color)
        elif obj.shape == "capsule":
            radius = size[0]
            half_height = size[1] if len(size) > 1 else size[0]
            builder.add_shape_capsule(body, xform=shape_xform, radius=radius, half_height=half_height, color=color)
        elif obj.shape == "cylinder":
            radius = size[0]
            half_height = size[1] if len(size) > 1 else size[0]
            builder.add_shape_cylinder(body, xform=shape_xform, radius=radius, half_height=half_height, color=color)
        elif obj.shape == "mesh":
            vertices, indices = self._load_mesh_geometry(obj.mesh_path)
            mesh = self._nt.Mesh(vertices, indices)
            sx, sy, sz = (size + [1.0, 1.0, 1.0])[:3]
            builder.add_shape_mesh(body, xform=shape_xform, mesh=mesh, scale=wp.vec3(sx, sy, sz), color=color)

    def _load_mesh_geometry(self, mesh_path: str | None) -> tuple[Any, Any]:
        """Load a mesh asset into ``(vertices, indices)`` arrays for Newton.

        The mesh file is parsed once via ``trimesh`` and cached by path, so
        rebuilds triggered by scene mutations (``move_object`` / ``add_object``)
        do not re-read the asset off disk. ``trimesh.load(..., force="mesh")``
        flattens any multi-geometry scene into a single triangle mesh; the
        returned indices are flattened (3 per triangle) as ``newton.Mesh``
        expects.

        Args:
            mesh_path: Absolute path to a mesh asset (resolved by
                :meth:`add_object`).

        Returns:
            Tuple of ``(vertices, indices)`` -- an ``(N, 3)`` float32 array of
            vertices and a flat int32 array of triangle indices.

        Raises:
            ValueError: If ``mesh_path`` is missing or the asset has no
                triangle geometry.
            ImportError: If ``trimesh`` (the ``sim-newton`` extra) is absent.
        """
        if not mesh_path:
            raise ValueError("mesh object has no mesh_path; add it via add_object(shape='mesh', mesh_path=...).")
        cached = self._mesh_cache.get(mesh_path)
        if cached is not None:
            return cached
        trimesh = require_optional(
            "trimesh", extra="sim-newton", purpose="loading mesh assets for the Newton add_object backend"
        )
        loaded = trimesh.load(mesh_path, force="mesh")  # type: ignore[attr-defined]
        vertices = np.ascontiguousarray(loaded.vertices, dtype=np.float32)
        indices = np.ascontiguousarray(loaded.faces, dtype=np.int32).reshape(-1)
        if vertices.size == 0 or indices.size == 0:
            raise ValueError(f"Mesh {mesh_path!r} has no triangle geometry (empty vertices/faces).")
        self._mesh_cache[mesh_path] = (vertices, indices)
        return vertices, indices

    def _wxyz_to_wp_quat(self, wxyz: list[float]) -> Any:
        """Convert a wxyz quaternion (SimRobot convention) to a warp xyzw quatf."""
        w, x, y, z = wxyz
        return self._wp.quat(x, y, z, w)
