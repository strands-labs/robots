"""MuJoCo Simulation backend - AgentTool orchestrator + shared state host.

Architecture notes (honest version, see GH #118)

The ``Simulation`` class uses multiple-inheritance to compose four mixins
(``PhysicsMixin``, ``RenderingMixin``, ``RecordingMixin``, ``RandomizationMixin``)
on top of the ``SimEngine`` ABC and the Strands ``AgentTool`` base. The
split keeps each module navigable (:mod:`physics` ~1150 lines,
:mod:`rendering` ~730, etc.) but the mixin boundaries describe *where code lives*, NOT the
coupling graph.

Every mixin reaches back into this class for the same shared state:

    self._world              - SimWorld handle (model + data + bookkeeping)
    self._lock               - RLock serializing ALL model/data access
    self._mj                 - cached ``mujoco`` module reference
    self._policy_threads     - per-robot Future dict (GH #114)
    self._renderer_tls       - thread-local renderer cache (macOS CGL)
    self._executor           - ThreadPoolExecutor for async policies

AND the cross-cutting helpers:

    self._require_world()              - "is the world live?" guard
    self._require_no_running_policy()  - scene-mutation safety gate
    self._prune_done_futures()         - cleanup of stale Future refs
    self._active_policy_robots()       - introspection + prune

Mixins declare these via ``if TYPE_CHECKING`` stubs so mypy accepts the
attribute lookups. This is NOT a Protocol - mixins are not enforceable;
the contract is *documentary*. The stubs exist so edits to the helpers
in this file propagate to the mixin type-checks without manual sync.

The alternative (extract a ``_SimulationState`` dataclass + pass it to
mixins) was explored and rejected: threading the state through every
method would blow up the diff across every mutation call, and mypy
narrowing of ``state.world._model`` after a ``_require_world(state)``
call does not work any better than narrowing through a bound method
(same limitation that led commit f5c8518 to back out the helper-based
dedup).

So: the split is honest about being for file-size, not for decoupling.
"""

import contextlib
import inspect
import json
import logging
import numbers
import os
import re
import threading
import time
from collections.abc import AsyncGenerator, Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots.simulation.base import SimEngine, close_match_hint, reject_setup_kwargs
from strands_robots.simulation.ik import hint_matches_name
from strands_robots.simulation.model_registry import (
    count_sim_robots,
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
    SimStatus,
    SimWorld,
    registered,
    registry_entry,
)
from strands_robots.simulation.mujoco.backend import (
    _NO_WORLD_MSG,
    _ensure_mujoco,
    filter_mujoco_attach_noise,
    mj_name_to_id,
)
from strands_robots.simulation.mujoco.manipulation import ManipulationMixin
from strands_robots.simulation.mujoco.motion_primitives import MotionPrimitivesMixin
from strands_robots.simulation.mujoco.physics import PhysicsMixin, _coerce_rgba
from strands_robots.simulation.mujoco.randomization import RandomizationMixin
from strands_robots.simulation.mujoco.recording import RecordingMixin
from strands_robots.simulation.mujoco.rendering import RenderingMixin
from strands_robots.simulation.mujoco.scene_ops import (
    eject_body_from_scene,
    eject_camera_from_scene,
    eject_robot_from_scene,
    inject_camera_into_scene,
    inject_object_into_scene,
    inject_robot_into_scene,
    install_compiled_model,
    patch_scene_mjcf,
    persist_world_option,
    replace_scene_mjcf,
    reposition_body_in_scene,
)
from strands_robots.simulation.mujoco.spec_builder import (
    SpecBuilder,
    _validate_size,
    material_spec_error,
)
from strands_robots.simulation.policy_runner import CooperativeStop
from strands_robots.simulation.terrain import SUPPORTED_TERRAINS, validate_difficulty, validate_terrain
from strands_robots.teleop_mixin import TeleopMixin
from strands_robots.utils import (
    camera_fov_error,
    coerce_pose_vector,
    entity_name_error,
    finite_vector_error,
    non_negative_whole_number_error,
    positive_finite_number_error,
    positive_whole_number_error,
    published_string_error,
    reserved_camera_name_error,
    sequence_length,
    step_aborted_msg,
)

if TYPE_CHECKING:
    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)


def _drop_unrecorded_cameras(observation: dict[str, Any], recorded: set[str] | None) -> dict[str, Any]:
    """Filter an observation down to the cameras the caller chose to record.

    Drops image arrays (ndarray with ``ndim >= 2``) whose key is not in
    ``recorded`` so the recorded frame matches a schema scoped via
    ``start_recording(cameras=...)``. Scalar state values (joint positions /
    velocities) and 1-D arrays are always kept.

    Args:
        observation: Raw observation dict (camera_name -> image, joint -> float).
        recorded: Set of camera keys to keep, or ``None`` to record every
            camera (the legacy default - the dict is returned unchanged).

    Returns:
        The observation unchanged when ``recorded`` is ``None``; otherwise a new
        dict without the excluded camera arrays.
    """
    if recorded is None:
        return observation
    import numpy as np

    return {
        k: v for k, v in observation.items() if not (isinstance(v, np.ndarray) and v.ndim >= 2 and k not in recorded)
    }


def _jnt_qpos_width(mj: Any, jnt_type: int) -> int:
    """qpos slice width for a MuJoCo joint type (free=7, ball=4, slide/hinge=1)."""
    if jnt_type == int(mj.mjtJoint.mjJNT_FREE):
        return 7
    if jnt_type == int(mj.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _jnt_dof_width(mj: Any, jnt_type: int) -> int:
    """qvel/dof slice width for a MuJoCo joint type (free=6, ball=3, slide/hinge=1).

    The velocity counterpart of :func:`_jnt_qpos_width`: a free joint spends 7
    ``qpos`` entries (3 translation + a wxyz quaternion) but only 6 ``qvel``
    entries (3 linear + 3 angular), and a ball joint 4 against 3. Reading a
    velocity slice at the ``qpos`` width would run past the joint and into its
    neighbour's.
    """
    if jnt_type == int(mj.mjtJoint.mjJNT_FREE):
        return 6
    if jnt_type == int(mj.mjtJoint.mjJNT_BALL):
        return 3
    return 1


def _compiled_geom_extent(mj: Any, model: Any, geom_name: str) -> list[float] | None:
    """Full extent in meters of a compiled geom's local bounding box.

    Reads MuJoCo's own ``geom_aabb`` row (centre plus half-extent per local
    axis) rather than re-deriving the extent from the request, so the number
    describes the geometry that actually compiled. For a primitive that
    reproduces the caller's ``size``; for a mesh it is the asset's own extent,
    which no request component defines.

    Args:
        mj: the cached ``mujoco`` module, for the ``mjtObj`` enum.
        model: a compiled ``MjModel``.
        geom_name: name of the geom to measure. Resolved through
            :func:`~strands_robots.simulation.mujoco.backend.mj_name_to_id`, so a
            name that is not a string reports no extent rather than reaching the
            binding.

    Returns:
        ``[x, y, z]`` full extents, or ``None`` when ``geom_name`` resolves to
        no geom -- a caller then reports no extent rather than a wrong one.
    """
    geom_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_GEOM, geom_name)
    if geom_id < 0:
        return None
    aabb = model.geom_aabb[geom_id]
    return [round(float(2.0 * aabb[3 + axis]), 4) for axis in range(3)]


def _validated_mesh_handle(mesh: Any) -> Any:
    """Normalize the constructor ``mesh`` argument to a stoppable client or None.

    ``mesh`` is a hook for an already-started mesh client (see
    :func:`strands_robots.mesh.init_mesh`), not a boolean opt-in switch: the
    engine only ever calls ``.stop()`` on it during teardown. A truthy value
    without a callable ``stop`` cannot be honored - ``cleanup()`` would abort
    on it and leave the MuJoCo world, renderers, and the executor alive - so
    it is rejected at construction, where the caller can still fix the call.

    Args:
        mesh: The raw constructor argument. Falsy (``None`` / ``False``) means
            "standalone, never joined a mesh".

    Returns:
        ``None`` for a falsy value, otherwise ``mesh`` unchanged.

    Raises:
        TypeError: If ``mesh`` is truthy but exposes no callable ``stop``. The
            message names the two supported ways to attach a mesh.
    """
    if not mesh:
        return None
    if callable(getattr(mesh, "stop", None)):
        return mesh
    raise TypeError(
        f"mesh={mesh!r} is not a mesh client: mesh= takes an already-started client "
        f"exposing .stop() (got {type(mesh).__name__}), because the engine only stops it "
        "during teardown. To join a mesh use Robot(name, mode='sim', mesh=True), which "
        "resolves the STRANDS_MESH kill switch and attaches a client; or attach one "
        "yourself after construction: sim.mesh = init_mesh(sim, peer_id=...)."
    )


def _resolve_policy_stop_timeout(policy_stop_timeout: float | None, default: float) -> float:
    """Seconds :meth:`MuJoCoSimEngine.cleanup` waits per live policy future.

    ``None`` is the documented "no preference" spelling and resolves to
    ``default``. A value outside the domain
    :func:`strands_robots.utils.positive_finite_number_error` accepts cannot
    express a preference either, so it resolves to ``default`` as well and the
    reason is logged against the parameter it came from.

    Only a positive finite budget can be honored.
    ``concurrent.futures.Future.result`` measures its wait as
    ``time.monotonic() + timeout``, so ``0``, a negative value and ``nan``
    expire it before the first check, and ``inf`` - the spelling that reads as
    "wait as long as it takes" - raises ``OverflowError`` out of that
    arithmetic. Each of those abandons a policy worker that may still be inside
    ``mj_step`` on the world ``cleanup`` is about to free, which is the
    stale-pointer window the bounded join exists to close; a non-real budget
    additionally makes the ``%.1f`` in the join's own warning raise, so the
    record that would have reported the skipped wait is dropped. Resolving to
    ``default`` keeps the join, so a budget the join cannot measure costs the
    caller the faster teardown they asked for rather than the protection.

    Returning a plain ``float`` is load-bearing rather than cosmetic: the shared
    guard accepts any real scalar, and ``Future.result`` raises ``TypeError``
    for a ``np.float32`` budget read out of a config array.

    Args:
        policy_stop_timeout: Caller-supplied budget in seconds, or ``None``.
        default: Budget to use when none was supplied, or when the supplied one
            is outside the domain the join can measure.

    Returns:
        Seconds to wait per live policy future - always a positive finite
        ``float``.
    """
    if policy_stop_timeout is None:
        return float(default)
    reason = positive_finite_number_error(policy_stop_timeout, "policy_stop_timeout", "cleanup")
    if reason is None:
        return float(policy_stop_timeout)
    logger.warning(
        "%s Waiting the default %.1fs instead, so the bounded join that keeps a live policy "
        "worker's pointers out of the freed world still happens.",
        reason,
        default,
    )
    return float(default)


# Actions consumed open-loop from each policy's chunk before it is re-queried,
# when the caller gives no per-robot override. Single-sourced so the signature
# default and the per-robot mapping fallback cannot drift apart.
_DEFAULT_ACTION_HORIZON = 8

# Hint words for the best-guess gripper/EEF mount ``list_bodies`` advertises.
# Matched on word boundaries by
# :func:`~strands_robots.simulation.ik.hint_matches_name` - the same rule
# :func:`~strands_robots.simulation.ik.discover_ee_frame` applies - so the short
# hint "ee" cannot fire inside "knee" or "wheel".
_GRIPPER_BODY_HINTS = ("gripper", "hand", "ee", "tool")

# The ``create_world`` parameters a LIVE world can still adopt, paired with the
# published action that applies each one in place without discarding the scene.
# ``ground_plane`` / ``terrain`` / ``difficulty`` are absent by construction:
# they shape the compiled scene at creation time and have no setter, so the only
# way to change them is to build a new world. Single-sourced here so the refusal
# below cannot advertise a setter that does not exist - the dead end it exists to
# close.
_WORLD_PARAM_SETTERS: tuple[tuple[str, str], ...] = (
    ("timestep", "set_timestep"),
    ("gravity", "set_gravity"),
)


_TOOL_SPEC_PATH = Path(__file__).parent / "tool_spec.json"

# Tool schema is 357 lines of JSON. `tool_spec` property is on the LLM hot path
# (called on every `strands` invocation). Load once at import, not per access.
with open(_TOOL_SPEC_PATH) as _f:
    _TOOL_SPEC_SCHEMA: dict[str, Any] = json.load(_f)

# The actions the schema advertises to a model, derived from the schema rather
# than restated beside it: a second list would be a second thing to keep true,
# and the failure it produces - refusing an action the model was told to use -
# is exactly what the guard below exists to prevent.
_PUBLISHED_ACTIONS: frozenset[str] = frozenset(_TOOL_SPEC_SCHEMA["properties"]["action"]["enum"])


def _published_string_params(field_aliases: dict[str, str]) -> frozenset[str]:
    """Method parameters the schema publishes as a JSON string.

    Derived from the schema for the same reason ``_PUBLISHED_ACTIONS`` is: a
    second list would be a second thing to keep true, and a field published
    without being covered here is one whose refusal is a raw ``TypeError`` from
    inside the method body.

    Wire names are mapped through *field_aliases* because the dispatcher
    validates a payload whose keys have already been rewritten to method
    parameter names. ``action`` is excluded: both agent entry points refuse a
    non-string action before dispatch is reached.

    Args:
        field_aliases: The dispatcher's wire-name to parameter-name map.

    Returns:
        The parameter names a caller must supply as a string.
    """
    props: dict[str, Any] = _TOOL_SPEC_SCHEMA["properties"]
    return frozenset(
        field_aliases.get(wire, wire)
        for wire, prop in props.items()
        if wire != "action" and prop.get("type") == "string"
    )


# Every field name the schema publishes. Derived from the schema for the same
# reason ``_PUBLISHED_ACTIONS`` is, and used to decide which spelling a refusal
# may name: a model constrained to this schema can emit no other.
_PUBLISHED_PARAMS: frozenset[str] = frozenset(_TOOL_SPEC_SCHEMA["properties"]) - {"action"}


def _reported_param_name(param: str, field_aliases: Mapping[str, str], received: Mapping[str, Any]) -> str:
    """The spelling a refusal names when it is about method parameter *param*.

    The dispatcher rewrites ``_FIELD_ALIASES`` to method parameter names before
    validating, so a refusal that echoes the parameter it validated can name a
    field no caller ever wrote. ``apply_force``'s torque is the case with no way
    back: it arrives as the schema's ``torque_vec``, is validated as ``torque``,
    and ``torque`` is not a published property at all -- so a model told
    "Parameter 'torque' must be a list of 3 numbers" was sent to a field it
    cannot emit.

    Preference order, most specific first:

    1. *param* itself when the payload carries it, since that is what the caller
       wrote.
    2. The alias for *param* that the payload carries, for a field that arrived
       under a wire name.
    3. The spelling the schema publishes, for a name the payload does not carry
       at all -- an entry in a ``Valid:`` list, or a missing required parameter.

    Args:
        param: The method parameter name the dispatcher validated.
        field_aliases: The dispatcher's wire-name to parameter-name map.
        received: The payload as the caller sent it, before alias rewriting.

    Returns:
        The field name to put in the refusal. Falls back to *param* when no
        published spelling exists, which keeps a Python-only parameter reportable
        under the only name it has.
    """
    if param in received:
        return param
    for wire, target in field_aliases.items():
        if target == param and wire in received:
            return wire
    if param in _PUBLISHED_PARAMS:
        return param
    return next(
        (wire for wire, target in field_aliases.items() if target == param and wire in _PUBLISHED_PARAMS),
        param,
    )


class MuJoCoSimEngine(
    TeleopMixin,
    PhysicsMixin,
    RenderingMixin,
    RecordingMixin,
    RandomizationMixin,
    ManipulationMixin,
    MotionPrimitivesMixin,
    SimEngine,
    AgentTool,
):
    """Programmatic MuJoCo simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language -> tool actions.

    **Stateful session.** One MuJoCo world per instance; actions form an
    implicit state machine starting with ``create_world``. Tools that mutate
    the scene (``add_robot``, ``remove_robot``, ``add_object``, ``remove_object``, ``move_object``, ``add_camera``, ``remove_camera``,
    ``load_scene``) are NOT safe to call while a policy is running via
    ``start_policy`` - stop it first. Call ``destroy()`` or ``cleanup()`` at
    session end to release the ThreadPoolExecutor, temp dirs, and MuJoCo
    resources.
    """

    def __init__(
        self,
        tool_name: str = "mujoco_simulation",
        default_timestep: float = 0.002,
        default_width: int = 640,
        default_height: int = 480,
        mesh: Any = None,
        peer_id: str | None = None,
        ros2_bridge: bool = False,
        ros2_domain: int = 0,
        **kwargs,
    ):
        """Construct a MuJoCo Simulation AgentTool.

        Args:
            tool_name: Identifier surfaced to the agent and used as the
                thread-name prefix for the executor.
            default_timestep: Default physics timestep (seconds). Can be
                overridden via ``create_world(timestep=...)``.
            default_width: Default render width (pixels) used when a
                caller does not pass explicit dimensions to ``render``.
            default_height: Default render height (pixels).
            mesh: Optional mesh-networking hook: an already-started mesh
                client exposing ``.stop()`` (see
                :func:`strands_robots.mesh.init_mesh`), which ``cleanup()``
                stops to detach this Simulation from the peer network before
                tearing down the MuJoCo world. Falsy (``None``, the default)
                keeps the Simulation standalone - all mesh code paths are
                no-ops. A truthy value that is not stoppable (notably
                ``mesh=True``) is rejected with a ``TypeError``: it is not a
                boolean opt-in switch, and the engine has nothing to stop.
                To join a mesh, use ``Robot(name, mode="sim", mesh=True)``,
                which resolves the ``STRANDS_MESH`` kill switch and attaches a
                client. The attribute is plain (not a property), so consumers
                may also attach a client after construction.
            peer_id: Stable identifier the mesh transport uses to
                address this Simulation. Opaque to MuJoCo itself; only
                consulted when ``mesh`` is truthy.
            ros2_bridge: When True, publish per-robot ``joint_states`` and
                camera ``image_raw`` on a ROS 2 domain every ``step``, so
                external ROS 2 nodes can subscribe to the running simulation.
                Requires ``rclpy`` (system ROS 2 / the official docker image);
                an :class:`ImportError` is raised here if it is missing.
                Defaults to False - the sim never touches ROS 2.
            ros2_domain: ROS 2 domain id (``ROS_DOMAIN_ID``) to publish on.
                Only an ``int`` in ``[0, 232]`` names a domain - the RTPS port
                map has no room above that - and a value outside it is refused
                with a :class:`ValueError` during construction whether or not
                ``ros2_bridge`` is set, so a backend that only publishes later
                still rejects it up front. Defaults to ``0``.
            **kwargs: Accepted and ignored, for cross-backend forward
                compatibility. The shared ``create_simulation`` / ``Robot``
                factory forwards one superset of keyword arguments to whichever
                backend is selected; MuJoCo tolerates and drops backend-specific
                kwargs it does not use (e.g. ``num_envs`` / ``device`` meant for
                GPU backends) so an identical call resolves across backends.
                Mirrors ``NewtonSimEngine``'s forward-compatible contract. Note
                they are NOT passed to ``super().__init__()`` (``AgentTool``
                takes no constructor arguments). Robot-setup arguments
                (``robot_name`` / ``robot``) are rejected here rather than
                dropped - a constructor builds an empty engine, so use
                ``Robot("so101", mode="sim")`` or ``add_robot`` instead.
        """
        reject_setup_kwargs(kwargs)
        super().__init__()
        self._init_ros_bridge(ros2_bridge=ros2_bridge, ros2_domain=ros2_domain)
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        # Mesh attributes are stored plainly (no property wrapper) so
        # downstream code can swap in a real mesh client after
        # construction without a setter dance. See the ``mesh`` /
        # ``peer_id`` docstring entries above for the contract.
        self.mesh: Any = _validated_mesh_handle(mesh)
        self.peer_id: str | None = peer_id

        # Additive sensor-noise config + reproducible RNG (set_obs_noise).
        # None until configured; the noise is then applied on every
        # get_observation / get_robot_state and every rendered frame.
        # Mirrors the Newton backend so a set_obs_noise(...) call behaves
        # identically on both engines. Type-declared on RandomizationMixin.
        self._obs_noise = None
        self._obs_noise_rng = None

        self._world: SimWorld | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{tool_name}_sim")
        # Per-robot Future refs for *active* policies. Completed futures are
        # pruned by ``_active_policy_futures()``/``_prune_done_futures()`` so
        # the dict never grows unboundedly and never reports stale "running".
        self._policy_threads: dict[str, Future] = {}
        # Capture rate of each rollout in ``_policy_threads``, recorded where
        # the Future is tracked so it is readable from another thread the
        # instant ``start_policy`` returns (``start_recording`` compares against
        # it). ``_policy_threads`` stays the sole authority on which rollouts
        # are live: this table is swept to it by ``_prune_done_futures``, so it
        # can never report a rate for a rollout that is not running.
        self._policy_rates: dict[str, float] = {}
        self._shutdown_event = threading.Event()
        # ``self._lock`` (RLock) serializes ALL access to MuJoCo
        # ``model``/``data`` arrays - both reads and writes. MuJoCo arrays
        # are NOT safe for concurrent reads during mutation (a racing
        # mj_step can produce torn/stale values). The lock is acquired:
        #
        #   * In ``_dispatch_action`` - so every agent-dispatched action
        #     is automatically serialized.
        #   * In ``send_action`` / ``get_observation`` - so the
        #     PolicyRunner worker thread is also serialized against the
        #     agent's dispatch thread.
        #   * RLock allows nested acquisition (methods that also acquire
        #     the lock internally are harmless when called via dispatch).
        self._lock = threading.RLock()

        self._viewer_handle = None
        self._viewer_thread = None

        # Thread-local renderer cache - MuJoCo Renderer uses thread-local GL
        # contexts (CGL on macOS, GLX on Linux). Sharing renderers across
        # threads causes SIGSEGV in cgl.free(). Each thread gets its own.
        self._renderer_tls = threading.local()
        self._renderer_model = None

        # Fail fast: verify MuJoCo is importable at construction time
        # so consumers catch missing-dependency errors immediately.
        self._mj = _ensure_mujoco()
        logger.info("MuJoCo simulation tool '%s' initialized", tool_name)

        # Construction complete - the finalizer may now release what we hold.
        # See SimEngine._init_complete: this must be the final statement.
        self._init_complete = True

    # Public Properties - read-only introspection.
    # WARNING: callers MUST NOT mutate the returned objects without holding
    # self._lock. Prefer using action methods which serialize automatically.

    @property
    def mj_model(self):
        """Read-only access to the MuJoCo model (mujoco.MjModel).

        Callers must NOT mutate the model without holding self._lock.
        Use action methods (set_gravity, set_timestep, etc.) instead.

        Warning: reads also race with a running PolicyRunner worker's mj_step
        (which mutates model arrays in-place for warm-start caches). For agent
        flows via stream()/dispatch, serialization is handled automatically.
        Direct Python consumers should either read between steps or accept
        that values may be momentarily stale during a running policy.
        """
        return self._world._model if self._world else None

    @property
    def mj_data(self):
        """Read-only access to the MuJoCo data (mujoco.MjData).

        Callers must NOT mutate data without holding self._lock.
        Use action methods (send_action, step, etc.) instead.

        Warning: reads race with a running PolicyRunner worker's mj_step.
        For agent flows via stream()/dispatch, the lock is held automatically.
        Direct Python consumers should use the action API or accept stale reads.
        """
        return self._world._data if self._world else None

    # Robot-compatible interface

    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get full observation for a robot: joint state + all attached cameras.

        See :meth:`SimEngine.get_observation` for the schema contract.
        Thread-safety: acquires self._lock to prevent torn reads while a
        concurrent mj_step is mutating data arrays.
        """
        if self._world is None or self._world._model is None:
            return {}
        if robot_name is None:
            if not self._world.robots:
                return {}
            robot_name = next(iter(self._world.robots))
        if not registered(self._world.robots, robot_name):
            return {}
        if skip_images and self._world is not None and self._world._backend_state.get("recording"):
            # T26: dataset recording needs every frame's image obs. Override
            # the policy's skip hint when an active recorder is attached -- but
            # only when the recorder actually keeps images. A recording scoped
            # to no cameras (``start_recording(cameras=[])``) writes a dataset
            # with no image features at all, and the frame hook drops every
            # image array through ``_drop_unrecorded_cameras`` before add_frame.
            # Overriding the hint there renders every scene camera once per
            # control step only to discard the pixels, which on a robot like
            # ``aloha`` (7 scene cameras) is the dominant cost of an
            # action-only rollout. ``None`` means "record every camera" (the
            # legacy default), so it still forces the render.
            rec_cams = self._world._backend_state.get("recording_cameras")
            if rec_cams is None or len(rec_cams) > 0:
                skip_images = False
        with self._lock:
            obs = self._get_sim_observation(robot_name, skip_images=skip_images)
        # Additive sensor noise (set_obs_noise). Exact no-op / same dict when
        # unconfigured, so the default path is byte-for-byte unchanged.
        return self._apply_obs_noise(obs)

    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action to simulation (Robot ABC compatible).

        ``action`` is normally a ``{joint/actuator name: value}`` mapping, but an
        ordered numeric vector (``list`` / ``tuple`` / 1-D ``numpy`` array) is
        also accepted and bound positionally to ``robot_action_keys(robot_name)``
        - the same positional convention :meth:`replay_episode` uses - so a
        policy's raw action vector can be applied directly. Those are the robot's
        *actuator* keys, which diverge from its joint names when it has
        passive/mimic joints or a tendon gripper. A vector whose length does not
        match the robot's actuator count is rejected with an actionable error
        rather than silently truncated.

        Thread-safety: acquires self._lock around ctrl writes + mj_step,
        as documented in the :class:`~strands_robots.simulation.base.SimEngine` contract. Concurrent calls
        from the agent's dispatch thread and a PolicyRunner worker are
        serialized here.

        Args:
            action: Actuator command, as a mapping or an ordered vector.
            robot_name: Robot to actuate. ``None`` resolves to the single robot
                when exactly one exists.
            n_substeps: Positive whole number of physics steps to advance after
                writing the targets, on the shared
                :func:`~strands_robots.utils.positive_whole_number_error` domain
                every backend applies. A NumPy or float count with an integral
                value is honored and coerced; a fractional, zero, negative,
                non-finite, boolean or non-numeric count is refused. ``0`` is
                refused rather than honored as "write but do not advance" -
                :meth:`step` is the surface that advances a count of its own,
                and it accepts ``0`` as a documented no-op.

        Returns:
            Dict with ``status`` ("success" or "error") and ``content``.
            When some action keys could not be resolved to actuators/joints,
            the ``content`` list includes a ``json`` block with an
            ``unresolved_keys`` list (and ``applied``) so callers can
            self-correct instead of silently losing commands. ``status`` is
            ``"error"`` when ``n_substeps`` is outside that domain, and nothing
            is written when it is.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # Refused before a single ctrl value is written, because a refusal that
        # arrived after the write would leave the robot commanded and the world
        # un-advanced - the one state this surface must never report an error
        # from. ``_apply_sim_action`` floors its loop at ``max(1, n_substeps)``
        # but adds the raw count to ``step_count``, so pre-fix a ``0`` ran one
        # ``mj_step`` and recorded none, a ``-5`` ran one and moved the counter
        # *backwards*, and a ``nan`` ran one and made ``step_count`` ``nan`` for
        # the rest of the world's life. A fractional or non-numeric count
        # reached ``range()`` and raised ``TypeError`` straight past this
        # method's structured envelope, after the write.
        if error := positive_whole_number_error(n_substeps, "n_substeps", "send_action"):
            return {"status": "error", "content": [{"text": error}]}
        n_substeps = int(n_substeps)
        if robot_name is None:
            if not self._world.robots:
                return {"status": "error", "content": [{"text": "No robots in the world."}]}
            robot_name = next(iter(self._world.robots))
        if not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
        action_map, coerce_error = self._coerce_action(action, robot_name)
        if coerce_error is not None:
            return coerce_error
        assert action_map is not None  # narrow for mypy: no error implies a mapping
        with self._lock:
            self._unresolved_action_keys: list[str] = []
            self._apply_sim_action(robot_name, action_map, n_substeps=n_substeps)
            unresolved = self._unresolved_action_keys
        applied = [k for k in action_map if k not in unresolved]
        if unresolved:
            # Surface the actual valid actuator names so the user can
            # self-correct without inspecting the MJCF by hand.
            valid_keys = self._get_valid_action_keys(self._world.robots[robot_name].namespace or "")
            hint = f" Valid keys: {valid_keys}" if valid_keys else ""
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action partially applied: keys {unresolved} could not be "
                            f"resolved to actuators or joints on '{robot_name}'. "
                            f"Applied: {applied}. Use individual joint/actuator names "
                            f"as dict keys.{hint}"
                        )
                    },
                    {"json": {"unresolved_keys": unresolved, "applied": applied}},
                ],
            }
        return {"status": "success", "content": [{"text": f"Action applied to '{robot_name}' ({len(applied)} keys)."}]}

    def physics_timestep(self) -> float | None:
        """Physics integration timestep (seconds) of the active world.

        Lets :class:`PolicyRunner` substep at the control rate so position-
        servo arms track each action. Returns ``None`` when no world exists.
        """
        if self._world is None:
            return None
        return float(self._world.timestep)

    # World Management

    def _cheap_robot_count(self) -> int:
        """Count available sim robot models (delegated to model_registry)."""
        try:
            return count_sim_robots()
        except (ImportError, OSError, AttributeError) as e:
            logger.warning("Could not count sim robots: %s", e)
            return 0

    def _world_contents(self) -> str:
        """Name what a live world holds, for a refusal whose remedy would drop it.

        Returns:
            A human-readable inventory such as ``"robots: so101; objects:
            none"``. Reported by :meth:`_world_exists_error` so ``destroy`` is
            offered with its cost stated rather than as a bare instruction.
        """
        world = self._world
        robots = ", ".join(world.robots) if world is not None and world.robots else "none"
        objects = ", ".join(world.objects) if world is not None and world.objects else "none"
        return f"robots: {robots}; objects: {objects}"

    def _world_exists_error(
        self,
        *,
        timestep: float | None,
        gravity: list[float] | None,
        ground_plane: bool,
        terrain: str | None,
        difficulty: float,
    ) -> dict[str, Any]:
        """Refuse a second ``create_world``, routing by what the caller asked for.

        A world cannot be rebuilt under a live scene, so the call is refused.
        Which remedy applies, though, depends entirely on the arguments: the
        parameters in :data:`_WORLD_PARAM_SETTERS` can be applied to the world
        that already exists, while ``ground_plane`` / ``terrain`` /
        ``difficulty`` are compiled in at creation and can only be changed by
        building a new world. ``reset`` applies NONE of them - it restores the
        initial state at the values the world was built with - so advertising it
        as an alternative to ``create_world`` sends a caller who asked for a
        different world to a call that reports success and changes nothing.

        Args:
            timestep: The ``create_world`` argument, unmodified.
            gravity: The ``create_world`` argument, unmodified.
            ground_plane: The ``create_world`` argument, unmodified.
            terrain: The ``create_world`` argument, unmodified.
            difficulty: The ``create_world`` argument, unmodified.

        Returns:
            A ``{status: "error", content: [...]}`` tool result naming the live
            world's contents and only the remedies that can satisfy the request.
        """
        requested = {"timestep": timestep, "gravity": gravity}
        contents = self._world_contents()
        lines = [f"create_world: a world already exists ({contents})."]

        in_place = [
            f"{param}={requested[param]!r} with {action}"
            for param, action in _WORLD_PARAM_SETTERS
            if requested[param] is not None
        ]
        structural = []
        if terrain is not None:
            structural.append(f"terrain={terrain!r}")
        if not ground_plane:
            structural.append("ground_plane=False")
        if float(difficulty) != 1.0:
            structural.append(f"difficulty={difficulty!r}")

        if in_place:
            lines.append(f"Apply {' and '.join(in_place)} on the live world; its contents stay.")
        if structural:
            lines.append(
                f"{' and '.join(structural)} can only be set when a world is built: "
                f"destroy (this discards {contents}), then create_world with it."
            )
        if not in_place and not structural:
            lines.append(
                "It is ready to use: add_robot / add_object build on it and reset restarts "
                f"the rollout in place. destroy, then create_world, starts empty and discards {contents}."
            )
        return {"status": "error", "content": [{"text": " ".join(lines)}]}

    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        """Create a new simulation world.

        ``terrain`` lays down a deterministic heightfield instead of the flat
        ground plane, so a floating-base/locomotion robot is spawned and
        evaluated on non-flat ground: ``"rough"`` = smoothed value-noise bumps,
        ``"stairs"`` = a flight of discrete step plateaus rising along +x,
        ``"pyramid"`` = concentric step plateaus rising toward the centre,
        ``"slope"`` = a constant-grade inclined ramp (see
        :mod:`strands_robots.simulation.terrain`). Only applies when
        ``ground_plane=True``.

        ``difficulty`` scales the terrain's peak elevation (``1.0`` = full
        height, ``<1`` gentler, ``>1`` harsher) - the curriculum knob a
        trainer ramps across resets. It is only meaningful with a ``terrain``;
        ``difficulty != 1.0`` with no ``terrain`` is rejected (it would have no
        effect) and must be a finite value ``> 0``.

        A floating-base robot added to a terrain world is spawned SEATED on
        the local terrain surface (its base is raised by the heightfield
        height beneath it) at ``add_robot`` and on every ``reset()``, rather
        than at the flat-ground keyframe height that would leave its feet
        buried below the raised terrain.

        A world can only be built once: a second call while one is live is
        refused rather than rebuilding under the live scene (``Robot("so101")``
        returns an instance whose world is already created and populated, so this
        refusal is the first thing such a caller meets). The refusal names what
        the live world holds and routes by the arguments actually passed:
        ``timestep`` / ``gravity`` are applied to the live world with
        :meth:`set_timestep` / :meth:`set_gravity`, keeping its contents, while
        ``ground_plane`` / ``terrain`` / ``difficulty`` are compiled in at
        creation and need :meth:`destroy` first. :meth:`reset` applies no
        ``create_world`` parameter - it restores the initial state at the values
        the world was built with - so it is never offered as a way to obtain a
        different world.

        ``timestep`` and ``gravity`` are validated exactly as
        :meth:`set_timestep` / :meth:`set_gravity` validate them - a finite
        ``timestep > 0`` (``0`` is an error, not a silent fallback to the
        engine default) and a 3-element finite ``gravity`` vector (or a real
        scalar taken as the z-component). A value MuJoCo cannot integrate is
        rejected with a structured error instead of being compiled into
        ``model.opt``. ``None`` selects the engine default for either.
        """
        # mujoco verified at __init__

        if terrain is not None:
            try:
                validate_terrain(terrain)
            except ValueError as exc:
                return {"status": "error", "content": [{"text": str(exc)}]}
        try:
            validate_difficulty(difficulty)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if terrain is None and float(difficulty) != 1.0:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"difficulty={difficulty!r} has no effect without a terrain "
                            "(it scales a heightfield's elevation); pass a terrain "
                            f"({'/'.join(repr(t) for t in sorted(SUPPORTED_TERRAINS))}) as well, "
                            "or omit difficulty for a flat ground plane."
                        )
                    }
                ],
            }

        if self._world is not None and self._world._model is not None:
            return self._world_exists_error(
                timestep=timestep,
                gravity=gravity,
                ground_plane=ground_plane,
                terrain=terrain,
                difficulty=difficulty,
            )

        # Validate the physics parameters on the same terms set_timestep /
        # set_gravity enforce: a world must not be created with a dt or a
        # gravity vector the setters would refuse. The effective timestep is
        # checked (not just the argument) so an unusable engine default is
        # reported under its own name instead of compiling into the world.
        effective_timestep = self.default_timestep if timestep is None else timestep
        timestep_param = "default_timestep" if timestep is None else "timestep"
        if err := self._validate_timestep(effective_timestep, "create_world", timestep_param):
            return err
        if gravity is None:
            _gravity = [0.0, 0.0, -9.81]
        else:
            normalized, gravity_error = self._normalize_gravity(gravity, "create_world")
            if normalized is None:
                return cast("dict[str, Any]", gravity_error)
            _gravity = normalized

        self._world = SimWorld(
            timestep=float(effective_timestep),
            gravity=_gravity,
            ground_plane=ground_plane,
            terrain=terrain,
            terrain_difficulty=float(difficulty),
        )

        self._world.cameras["default"] = SimCamera(
            name="default",
            position=[1.5, 1.5, 1.2],
            target=[0.0, 0.0, 0.3],
            width=self.default_width,
            height=self.default_height,
        )

        self._compile_world()

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        "Simulation world created\n"
                        f"Timestep: {self._world.timestep}s ({1 / self._world.timestep:.0f}Hz physics)\n"
                        f"Gravity: {self._world.gravity}\n"
                        f"Default camera ready\n"
                        f"Robot models: {self._cheap_robot_count()} available\n"
                        "Add robots: action='add_robot' (urdf_path or data_config)\n"
                        "Add objects: action='add_object'\n"
                        "List URDFs: action='list_urdfs'"
                    )
                }
            ],
        }

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a complete scene from an MJCF XML (or URDF) file.

        Replaces the currently-live spec with one parsed from disk. The
        loaded spec becomes the source of truth, so downstream
        ``add_object`` / ``add_camera`` / ``add_robot`` calls mutate it via
        ``spec.recompile(model, data)`` and preserve the on-disk scene.

        Notes:

        * ``_backend_state["scene_loaded"] = True`` stays as a marker for
          introspection (and for downstream callers that still check it,
          though the scene_ops path is now uniform across both entry
          points).
        * ``_backend_state["scene_base_dir"]`` is recorded in case any
          consumer needs the original source directory (e.g. for mesh path
          resolution in followup inject operations on files with relative
          mesh paths).
        """
        if err := self._require_no_running_policy("load_scene"):
            return err
        mj = self._mj

        if not os.path.exists(scene_path):
            return {"status": "error", "content": [{"text": f"Scene file not found: {scene_path}"}]}

        # Compile the new scene into LOCAL model/data first. A malformed MJCF
        # must NOT destroy the currently-live world: previously self._world was
        # reassigned to a fresh SimWorld BEFORE the spec compiled, so a bad
        # file discarded the live scene and still returned an error dict. We
        # only swap self._world in after a successful compile + forward.
        try:
            # Load the scene as a live MjSpec - this gives us a mutable AST
            # for downstream add_object/add_robot operations, matching the
            # contract produced by _compile_world for fresh worlds.
            spec = SpecBuilder.from_file(scene_path)
            with filter_mujoco_attach_noise():
                model = spec.compile()
            data = mj.MjData(model)
            # Forward the freshly-allocated MjData so derived state
            # (xpos / xquat / xmat / sensor data) is populated. Without
            # this, ``Renderer.update_scene`` finds the body transforms
            # unset and returns a skybox-only gradient on the first
            # render call after load_scene. Forwarding here populates
            # that derived state before the first render.
            #
            # Cost: O(model.nbody) - negligible for typical scenes.
            # Failure here is genuinely a bug in the loaded MJCF
            # (e.g. inconsistent qpos vs joint definitions), so let it
            # propagate to the ``except`` below where it gets converted
            # to a structured error response - with the old world intact.
            mj.mj_forward(model, data)
        except Exception as e:
            logger.error("Failed to load scene: %s", e)
            return {"status": "error", "content": [{"text": f"Failed to load scene: {e}"}]}

        # Compile succeeded - atomically swap in the new world under the lock
        # so a concurrent render/recorder thread never observes a half-built
        # world (load_scene runs under the blanket dispatch lock, so this
        # acquisition is a reentrant no-op there and the real guard when the
        # method is called directly).
        with self._lock:
            world = SimWorld()
            world._backend_state["spec"] = spec
            install_compiled_model(world, model, data)
            world.status = SimStatus.IDLE

            # Cache the canonical serialisation; legacy readers use this.
            try:
                with filter_mujoco_attach_noise():
                    world._backend_state["xml"] = spec.to_xml()
            except Exception as xml_err:
                logger.debug("spec.to_xml() on loaded scene failed: %s", xml_err)

            world._backend_state["scene_loaded"] = True
            world._backend_state["scene_base_dir"] = os.path.dirname(os.path.abspath(scene_path))
            self._world = world

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Scene loaded from {os.path.basename(scene_path)}\n"
                        f"Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}\n"
                        "Use action='get_state' to inspect, action='step' to simulate"
                    )
                }
            ],
        }

    def replace_scene_mjcf(self, xml: str) -> dict[str, Any]:
        """Atomically replace the entire scene with agent-authored MJCF.

        Validated by actually compiling it via ``mujoco.MjSpec.from_string``
        and ``spec.compile()``. On failure returns a standard error dict with
        MuJoCo's compiler error verbatim; on success the old ``_world._model``,
        ``_world._data`` and ``_world._backend_state['spec']`` are replaced.

        Note: ``self._world.robots`` / ``objects`` / ``cameras`` registries
        are LEFT UNTOUCHED. The raw MJCF can express elements that those
        dataclasses can't (``<tendon>``, ``<equality>``, ``<pair>``, etc.) -
        the agent is responsible for reconciling the registry with the new
        scene if it cares.

        Use this as an escape hatch when the ``add_object`` / ``add_robot``
        vocabulary is insufficient. For additive changes, prefer those
        methods - they keep the registry in sync.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("replace_scene_mjcf"):
            return err

        try:
            replace_scene_mjcf(self._world, xml)
        except (ValueError, RuntimeError) as e:
            return {"status": "error", "content": [{"text": f"MJCF compile failed: {e}"}]}

        model = self._world._model
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Scene replaced via raw MJCF\n"
                        f"Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}, Cameras: {model.ncam}\n"
                        "Warning: world.robots / world.objects / world.cameras registries were NOT updated - "
                        "they describe our previous Python-side view of the scene."
                    )
                }
            ],
        }

    def patch_scene_mjcf(self, ops: list[dict[str, Any]]) -> dict[str, Any]:
        """Apply a list of structured ops to the live MjSpec atomically.

        Each op is a small dict. Supported kinds::

            {"op": "add_body",      "parent": "world", "name": "foo", "pos": [0,0,1]}
            {"op": "add_geom",      "body": "foo",     "type": "sphere", "size": [0.1]}
            {"op": "add_site",      "body": "foo",     "name": "tip",    "pos": [0,0,0.2]}
            {"op": "set_body_pos",  "name": "foo",     "pos": [1,0,1]}
            {"op": "set_body_quat", "name": "foo",     "quat": [1,0,0,0]}
            {"op": "delete_body",   "name": "foo"}

        Each op accepts only the keys it reads, and every other key is
        refused::

            add_body       op, parent, name, pos, quat
            add_geom       op, body, type, size, rgba, name, pos, quat
            add_site       op, body, name, pos, size, rgba
            set_body_pos   op, name, pos
            set_body_quat  op, name, quat
            delete_body    op, name

        Rejecting the rest is not pedantry: every field has a fallback default
        (``pos`` the origin, ``quat`` identity, ``type`` ``"box"``, ``parent``
        the worldbody), so a misspelled key would apply that default and report
        success - ``{"op": "set_body_pos", "name": "crate", "position": [...]}``
        would move the body to the origin instead of to the requested pose. The
        error names the op, the unrecognised key, a close match where one
        exists, and the keys that op accepts.

        Every numeric field an op writes is held to the domain the
        scene-construction calls apply to the same buffer: a ``pos`` of exactly
        3 finite components, a ``quat`` of 4, a ``rgba`` of 3 (read as RGB and
        completed with an opaque alpha, exactly as ``add_object(color=...)``
        does) or 4, and a ``size`` of finite components in the count its shape
        consumes. MuJoCo bakes a ``nan``/``inf`` component into the model
        without complaint, so an unchecked one is accepted and only surfaces as
        a poisoned physics state several successful calls later::

            sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate",
                                   "pos": [float("nan"), 0, 0.3]}])
            # status=error: set_body_pos: 'pos' must contain finite numbers
            #               (no nan/inf), got [nan, 0, 0.3]

            sim.patch_scene_mjcf([{"op": "set_body_pos", "name": "crate",
                                   "pos": [0.4, 0.9]}])
            # status=error: set_body_pos: 'pos' must be a 3-element vector,
            #               got 2 ([0.4, 0.9])

        The width matters as much as the components: these two ops assign the
        field as a spec attribute rather than passing it as a constructor
        keyword, and MuJoCo reports a mismatch there by dumping a C++ overload
        table that names neither the op nor the field.

        The three ops that CLAIM a name - ``add_body``, ``add_geom`` and
        ``add_site`` - hold it to the same
        :func:`~strands_robots.utils.entity_name_error` domain that
        ``add_object`` / ``add_camera`` / ``add_robot`` apply, so a name one
        door refuses is refused at all of them. Supplying ``""`` or a name
        carrying a NUL leaves an entity this API cannot then address: MuJoCo
        reads a name only up to the first NUL, so ``{"op": "add_body", "name":
        "a\x00b"}`` used to report success while compiling the body as ``a`` -
        ``list_bodies`` showed only ``a``, and a later op that genuinely asked
        for ``a`` was refused as a repeated name the caller never used. An
        empty ``add_geom`` name compiled an unnamed geom, which
        ``set_geom_properties(geom_name=...)`` cannot reach at all. Omitting
        ``name`` on ``add_geom`` is unaffected and still produces an unnamed
        geom, which is legal MJCF; the three ops that instead LOOK a name UP
        (``set_body_pos`` / ``set_body_quat`` / ``delete_body``) are unchanged,
        because a name that addresses nothing is honestly absent there and
        those ops already say so.

        The whole batch is applied, then the spec is recompiled once. If any
        op fails, the batch is rejected and the world is rolled back to its
        pre-patch state (from a deep copy of the spec). Use this for fast
        iterative edits; use ``replace_scene_mjcf`` when you need to express
        MJCF elements not covered by the supported op vocabulary.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("patch_scene_mjcf"):
            return err

        try:
            applied = patch_scene_mjcf(self._world, ops)
        except (ValueError, RuntimeError) as e:
            return {"status": "error", "content": [{"text": f"MJCF patch failed: {e}"}]}

        model = self._world._model
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Patched scene: {applied} op(s) applied\n"
                        f"Bodies: {model.nbody}, Joints: {model.njnt}, Actuators: {model.nu}, Cameras: {model.ncam}\n"
                        "Warning: world.robots / world.objects / world.cameras registries were NOT updated."
                    )
                }
            ],
        }

    def _compile_world(self) -> None:
        """Build the MjSpec from ``self._world`` and compile it to MjModel.

        Stashes the live ``MjSpec`` in ``_backend_state["spec"]`` so every
        subsequent scene mutation uses ``spec.recompile(model, data)`` in
        place - that preserves existing joint state automatically, replacing
        the legacy XML-round-trip helpers in :mod:`scene_ops`.

        Also exports ``spec.to_xml()`` to ``_backend_state["xml"]`` for any
        consumer that still reads the raw MJCF string (e.g. ``load_scene``
        compatibility paths).
        """
        mj = self._mj
        assert self._world is not None  # only called after create_world
        spec = SpecBuilder.build(self._world)
        self._world._backend_state["spec"] = spec
        with filter_mujoco_attach_noise():
            model = spec.compile()
        install_compiled_model(self._world, model, mj.MjData(model))
        # Forward the freshly-allocated MjData so derived state
        # (xpos / xquat / xmat) is populated - same rationale as in
        # ``load_scene`` (#168). Without this, the first
        # render after ``_compile_world`` returns the skybox-only
        # gradient because body transforms are zero-initialised.
        mj.mj_forward(self._world._model, self._world._data)
        try:
            with filter_mujoco_attach_noise():
                self._world._backend_state["xml"] = spec.to_xml()
        except Exception as xml_err:
            # spec.to_xml() is best-effort - if it fails we still have a
            # valid compiled model. The cached XML is a convenience for
            # tooling, not a correctness invariant.
            logger.debug("spec.to_xml() failed: %s", xml_err)
        self._world.status = SimStatus.IDLE

    # Robot Management

    @staticmethod
    def _ensure_meshes(model_path: str, robot_name: str) -> dict[str, Any] | None:
        """Check if mesh files referenced by a model XML exist; auto-download if missing.

        Returns ``None`` on success (meshes present or downloaded cleanly) and
        a standard error dict on auto-download failure. Caller MUST propagate
        the error dict back to the agent - previously the return value was
        ignored and the error was silently swallowed, leaving the agent to
        hit a cryptic 'mesh not found' from MuJoCo instead.

        A reference is resolved the way MuJoCo resolves it (see
        :func:`strands_robots.assets.download._mjcf_mesh_candidates`): against
        the MAIN model file's directory plus the model's mesh subdirectory, and
        the subdirectory is read once for the whole model because
        ``<compiler>`` applies across ``<include>``. Resolving against the
        directory of whichever fragment declared the mesh - a location MuJoCo
        does not accept - reports a present mesh as absent, which costs a full
        ``force=True`` re-download on every ``add_robot`` and refuses the robot
        outright when that download cannot run.
        """
        # One owner for the resolution rule: the download path applies the same
        # rule to decide whether a robot's assets need fetching, so a second
        # copy here could disagree with it about the same model.
        from strands_robots.assets.download import _mjcf_mesh_candidates, _mjcf_mesh_subdir

        model_dir = os.path.dirname(os.path.abspath(model_path))

        files_to_check = [model_path]
        try:
            with open(model_path) as _f:
                top_content = _f.read()
            for inc in re.findall(r'<include\s+file="([^"]+)"', top_content):
                inc_path = os.path.join(model_dir, inc)
                if os.path.exists(inc_path):
                    files_to_check.append(inc_path)
        except (OSError, UnicodeDecodeError):
            # An unreadable top-level model contributes no includes to scan.
            # MuJoCo names the unreadable file itself on the load that follows,
            # which is a better report than anything this check could invent.
            pass

        # (fragment directory relative to model_dir, fragment text)
        fragments: list[tuple[str, str]] = []
        for xml_path in files_to_check:
            try:
                with open(xml_path) as _f:
                    content = _f.read()
            except (OSError, UnicodeDecodeError):
                # Same reasoning: a fragment we cannot read declares no mesh
                # references, and MuJoCo reports it on load.
                continue
            frag_dir = os.path.dirname(os.path.abspath(xml_path))
            rel_dir = os.path.relpath(frag_dir, model_dir)
            fragments.append(("" if rel_dir == os.curdir else rel_dir, content))

        mesh_subdir = _mjcf_mesh_subdir(*(text for _rel, text in fragments))

        missing = False
        for rel_dir, content in fragments:
            for mf in re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content):
                if not any(os.path.exists(p) for p in _mjcf_mesh_candidates(mf, model_dir, mesh_subdir, rel_dir)):
                    missing = True
                    break
            if missing:
                break

        if not missing:
            return None

        logger.info("Downloading mesh files for '%s' from MuJoCo Menagerie (first time only)...", robot_name)
        try:
            from strands_robots.assets import resolve_robot_name
            from strands_robots.assets.download import download_robots

            canonical = resolve_robot_name(robot_name)
            download_robots(names=[canonical], force=True)
        except (ImportError, OSError) as e:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Auto-download failed for '{robot_name}': {e}. "
                            f"Install robot_descriptions: pip install strands-robots[sim-mujoco]"
                        )
                    }
                ],
            }
        return None

    def _attach_robot_to_mesh(self, robot: SimRobot) -> None:
        """Best-effort: register *robot* as its own peer on the parent's mesh.

        When the parent ``Simulation`` is on a Zenoh mesh (``self.mesh`` set
        and ``self.peer_id`` populated), every robot added via ``add_robot``
        becomes addressable on the mesh in its own right - the agent can
        ``robot_mesh tell target=<robot.peer_id>`` instead of having to ask
        the sim container to route by ``robot_name``.

        Stays a no-op (and silently swallows failures) when:

        * the parent sim never joined a mesh (``self.mesh`` is falsy), or
        * ``init_mesh`` returns ``None`` because ``STRANDS_MESH=false``, or
        * ``zenoh`` is not installed, or
        * any unexpected exception bubbles up from the mesh stack.

        On success, mutates ``robot.mesh`` + ``robot.peer_id`` in place so
        ``remove_robot`` / ``cleanup`` can tear it down later.
        """
        if not self.mesh:
            # Sim itself isn't on a mesh - nothing to attach to. Stays a
            # no-op so unit tests that construct a bare ``Simulation``
            # without zenoh keep working.
            return
        try:
            # Local import to avoid pulling zenoh into the import graph for
            # users who run the sim entirely off-mesh.
            from strands_robots.mesh import init_mesh

            # Derive a stable peer_id from the parent sim + robot name so
            # the same robot in two different sims still gets distinct ids.
            # Format: ``<parent_peer_id>__<robot_name>`` e.g.
            # ``so100_sim-a1b2c3d4__so100``. Keeps the parent's uuid suffix
            # so collisions across processes stay impossible.
            parent_id = self.peer_id or "sim"
            child_peer_id = f"{parent_id}__{robot.name}"

            # We pass the SimRobot dataclass as the owner. Mesh is duck-
            # typed and only needs ``hasattr`` accesses, so the dataclass
            # works even though it has no ``tool_name_str`` etc.
            child_mesh = init_mesh(
                robot,
                peer_id=child_peer_id,
                # "sim", not "robot": presence publishes this as robot_type,
                # and a simulated arm announcing itself as real hardware makes
                # every consumer (dashboard badges, fleet-agent "is this real
                # hardware?" checks, e-stop triage) treat a sim as actuating
                # metal. Parent sim peers already announce "sim"; the child
                # is the same MuJoCo world scoped to one robot.
                peer_type="sim",
                mesh=True,
            )
            if child_mesh is not None:
                robot.mesh = child_mesh
                robot.peer_id = child_mesh.peer_id
                # Bridge: give the SimRobot a _world reference so the child
                # Mesh's _read_state() can extract per-robot joint positions
                # from the MuJoCo world data (without this, the child mesh
                # publishes only presence heartbeats - no state topic).
                robot._world = self._world
                # Parent-sim backref: the child peer's Mesh._dispatch
                # delegates execute/start to this Simulation (with
                # robot_name pre-bound) - a bare SimRobot has no run_policy
                # of its own, so without this the addressable child peer
                # answered "unknown action: execute".
                robot._sim_parent = self
        except Exception as exc:  # noqa: BLE001 - mesh enrichment is best-effort
            logger.warning(
                "Failed to attach robot %r to mesh (sim peer_id=%s): %s",
                robot.name,
                self.peer_id,
                exc,
            )

    def _detach_robot_from_mesh(self, robot: SimRobot) -> None:
        """Stop *robot*'s mesh peer if it has one. Best-effort, no-raise."""
        m = getattr(robot, "mesh", None)
        if not m:
            return
        try:
            m.stop()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "Failed to stop mesh peer for robot %r (peer_id=%s): %s",
                robot.name,
                getattr(robot, "peer_id", "?"),
                exc,
            )
        finally:
            robot.mesh = None
            robot.peer_id = ""

    @staticmethod
    def _unknown_model_msg(requested: str) -> str:
        """Build the 'model could not be resolved' error for a robot name.

        Three conditions reach this message and they have different remedies, so
        it diagnoses which one it is instead of reporting them all as a bad name:

        * The registry does not know ``requested`` - a typo or an unknown robot.
          Names the closest sim-loadable registry keys via
          :func:`close_match_hint` so the caller can fix it in place without a
          discovery round-trip. The pool is deliberately the ``mode="sim"``
          listing rather than the whole registry: a suggestion this engine
          cannot spawn sends the caller straight back here, and the registry
          holds hardware-only entries close enough to be suggested (the sole
          suggestion offered for ``earthrover`` was ``hope_jr``, which is itself
          hardware-only, so the one remedy on offer reproduced the same
          refusal). ``close_match_hint`` already drops a suggestion identical to
          ``requested`` for the same reason - it carries no information and
          displaces a real one out of the three slots.
        * The registry knows ``requested`` and the entry declares a hardware
          backend and no simulation asset - a real robot strands drives over
          LeRobot that has no model to load. The name is already correct, so
          spelling suggestions are the wrong advice here too; names the hardware
          entry point instead, the way
          :func:`~strands_robots.robot.Robot` already answers a leader-arm name
          with the teleoperator entry point rather than the registry listing.
        * The registry knows ``requested`` and its model XML is simply not on
          disk. Here the name is already correct, so spelling suggestions are
          the wrong advice - ``difflib`` ranks an exact match first, so this was
          the one case that got told "Did you mean: <the name it just
          refused>". Names the asset path the resolver looked for and the
          remedy, split on the entry's own ``auto_download`` posture: an entry
          with ``auto_download: false`` is never fetched automatically (the
          asset has to be placed by hand), any other entry had a download
          attempted by :func:`~strands_robots.assets.manager.resolve_model_path`
          before it gave up, so retrying it through the ``download_assets``
          tool is what surfaces why. Mirrors the registration-time wording in
          :func:`~strands_robots.registry.user_registry.register_robot`, which
          already reports a missing asset directory this way.

        Suggestions and the asset probe are both best-effort: a registry that
        cannot be read degrades to the bare form rather than propagating.
        """
        known: list[str] = []
        try:
            from strands_robots.registry import list_robots as _list_robots

            # mode="sim" so a suggestion is a name this engine can actually
            # spawn. A user model added through ``register_urdf`` is absent from
            # every ``list_robots`` mode, so narrowing the pool drops nothing
            # that was suggestable before.
            known = [r.get("name", "") for r in _list_robots(mode="sim") if r.get("name")]
        except Exception:  # noqa: BLE001 - suggestions are best-effort
            known = []

        # Probed independently of the suggestion list so an unreadable registry
        # listing cannot mask the more specific diagnosis, and vice versa.
        asset_gap: tuple[str, str, str, bool, list[str]] | None = None
        hardware_only: tuple[str, str] | None = None
        try:
            from strands_robots.assets.manager import get_search_paths, is_robot_asset_present
            from strands_robots.registry import get_robot as _get_robot
            from strands_robots.registry import resolve_name as _resolve_name

            # ``requested`` may be an alias; resolve to the canonical key the
            # asset entry hangs off. No type test on ``requested`` here - a name
            # that cannot be a registry key raises and is caught, which keeps
            # the availability listing above ungated on the name's type.
            canonical = _resolve_name(requested)
            entry = ((_get_robot(canonical) or {}) if canonical else {}) or {}
            asset = entry.get("asset") or {}
            if asset and not is_robot_asset_present(canonical):
                asset_gap = (
                    canonical,
                    str(asset.get("dir", "")),
                    str(asset.get("model_xml", "")),
                    asset.get("auto_download") is False,
                    [str(path) for path in get_search_paths()],
                )
            elif entry and not asset:
                # Registered, correct, and simply not a simulation robot. The
                # LeRobot type is what the hardware route is keyed on, so it is
                # quoted when the entry declares one.
                hardware_only = (canonical, str((entry.get("hardware") or {}).get("lerobot_type") or ""))
        except Exception:  # noqa: BLE001 - the diagnosis is best-effort
            asset_gap = None
            hardware_only = None

        if asset_gap is not None:
            canonical, asset_dir, model_xml, never_downloads, search_paths = asset_gap
            relative = f"{asset_dir}/{model_xml}"
            searched = ", ".join(f"'{path}'" for path in search_paths)
            msg = (
                f"Robot '{requested}' is registered but its model file is not on disk: "
                f"no '{relative}' under {searched}."
                if search_paths
                else (
                    f"Robot '{requested}' is registered but its model file is not on disk "
                    f"(expected '{relative}' on an asset search path)."
                )
            )
            if never_downloads:
                msg += (
                    f" This entry declares auto_download=false, so its asset is never fetched "
                    f"automatically - create that directory and place '{model_xml}' inside it."
                )
            else:
                msg += f" Fetch it with the download_assets tool (robots='{canonical}')."
            return msg

        if hardware_only is not None:
            canonical, lerobot_type = hardware_only
            typed = f" (LeRobot type '{lerobot_type}')" if lerobot_type else ""
            return (
                f"Robot '{requested}' is registered for real hardware only{typed}: its registry "
                f"entry declares no simulation asset, so there is no model to load. The name is "
                f"already correct, so there is no spelling to fix - drive it as hardware with "
                f"Robot('{canonical}', mode='real'), or pass urdf_path= to supply a model of your "
                f"own. Use list_robots(mode='sim') to see the robots this backend can spawn."
            )

        msg = f"No model found for '{requested}'."
        msg += close_match_hint(requested, known)
        msg += " Use action='list_urdfs' to see all available robots."
        return msg

    def _unknown_object_msg(self, requested: object) -> str:
        """Actionable 'object not found' message: name it, offer a close-match,
        and point at the discovery surface - consistent with the camera
        render/record error paths and ``_unknown_model_msg`` (#1299) rather
        than a dead-end "Object 'X' not found."."""
        known = list(self._world.objects.keys()) if self._world is not None else []
        msg = f"Object '{requested}' not found."
        if known:
            msg += close_match_hint(requested, known)
            msg += f" Available objects: {known}. Use action='list_objects' to see all."
        else:
            msg += " No objects in the scene; add one with action='add_object'."
        return msg

    def _unknown_camera_msg(self, requested: object) -> str:
        """Actionable 'camera not found' message for ``remove_camera`` - lists the
        renderable cameras (like the render/record error paths already do) plus a
        close-match, so a typo is fixable in-place without a discovery round-trip.

        The recovery hint names the canonical ``list_cameras`` action (the name
        in ``tool_spec.json`` and ``describe()``), not the internal
        ``list_cameras_info`` method the dispatcher aliases it to - so a blind
        agent following the hint learns the same action the discovery surface
        teaches, mirroring the ``list_objects`` hint in ``_unknown_object_msg``."""
        known = self._list_camera_names()
        msg = f"Camera '{requested}' not found."
        if known:
            msg += close_match_hint(requested, known)
            msg += f" Available: {known}. Use action='list_cameras' to see all."
        return msg

    def _unknown_robot_msg(self, requested: object) -> str:
        """Actionable 'robot not found' message: name it, offer a close-match,
        and list the robots in the world - consistent with ``_unknown_object_msg`` /
        ``_unknown_camera_msg`` / ``_unknown_model_msg`` (#1299/#1303) rather than a
        dead-end "Robot 'X' not found." that forces an agent driving the API blind
        into a discovery round-trip on every typo."""
        known = list(self._world.robots.keys()) if self._world is not None else []
        msg = f"Robot '{requested}' not found."
        if known:
            msg += close_match_hint(requested, known)
            msg += f" Available robots: {known}. Use action='list_robots' to see all."
        else:
            msg += " No robots in the scene; add one with action='add_robot'."
        return msg

    def _unknown_action_msg(self, requested: str) -> str:
        """Actionable 'unknown action' message: name it, offer a close-match over
        the published enum, and point at where that enum is written - consistent
        with ``_unknown_model_msg`` / ``_unknown_object_msg`` /
        ``_unknown_camera_msg`` / ``_unknown_robot_msg`` rather than a dead-end
        "Unknown action: X." that forces an agent driving the tool blind to
        re-read its own schema to recover from a one-character typo.

        ``action`` is the parameter every call must supply and the only one with
        no usable default, so it is where a typo is most likely to land - and it
        was the one parameter whose refusal named neither a candidate nor a way
        to find one, while a misspelled *robot* one frame away got both.

        The suggestion is drawn from the published enum rather than from every
        dispatchable method, because the enum is what an agent was handed: a
        name outside it is refused at this boundary even when it resolves
        (#2093), so offering one would send the caller to a second refusal. The
        count travels with the pointer so a caller who gets no suggestion still
        learns that the vocabulary is closed and enumerated, not open-ended.
        """
        msg = f"Unknown action: {requested}."
        msg += close_match_hint(requested, sorted(_PUBLISHED_ACTIONS))
        msg += (
            f" This tool publishes {len(_PUBLISHED_ACTIONS)} actions in the 'action' enum "
            "of its schema; see tool_spec for the actions you can use."
        )
        return msg

    def add_robot(
        self,
        name: str | None = None,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation via XML round-trip composition.

        Instead of replacing the entire world model, this method merges the
        robot's bodies, actuators, assets, and sensors into the existing scene
        XML.  This preserves previously-created world state (gravity, objects,
        cameras, other robots).

        That preservation covers the scene's DYNAMIC state, not just its
        contents: an arm already in the world keeps the pose it is in and the
        actuator setpoints holding it there, an object stays where it settled or
        was carried to, a latched ``apply_force`` wrench persists, and the clock
        keeps counting. Only the robot being added is placed at a defined
        configuration - its ``keyframe`` pose, or the zero configuration - so
        composing a scene incrementally cannot undo what has already happened in
        it. Use :meth:`reset` to return the whole world to its initial state.

        ``name`` is the instance label used to address this robot later
        (``run_policy(robot_name=...)``, ``get_robot_state``, etc.). It is
        OPTIONAL: when omitted (``None``, or ``""``) it is auto-derived from
        ``data_config`` (or the URDF filename), with a numeric suffix appended
        if that label is already taken -- so ``add_robot(data_config="so101")``
        twice yields ``so101`` and ``so101_2`` instead of erroring. Any other
        value must be a ``str`` containing no NUL: those two are the documented
        derive-a-label short form, while another falsy value (``0``, ``[]``)
        would take the same branch and report success under a label that was
        never asked for, and a non-string label keys the registry with a value
        the agent-tool surface - where a name arrives as JSON - cannot
        address.

        ``keyframe`` (name ``str`` or index ``int``) spawns the robot in a
        canonical pose declared by a ``<keyframe>`` in its source model
        (e.g. panda ``"home"``) instead of the all-zero configuration, and that
        state is restored by ``reset()``. The keyed actuator command is applied
        with the pose, so a gravity-loaded arm HOLDS its home configuration
        instead of sagging out of it on the first step. An unknown keyframe is a
        hard error naming the available keyframes; ``None`` (default) keeps the
        zero pose.

        A ``name``/``data_config`` that resolves to no model is reported as an
        actionable error naming the requested robot, offering close-match
        suggestions and pointing at ``list_urdfs`` (plus the
        ``data_config=``/``urdf_path=`` options) -- not a dead-end "supply a
        model source". The bare model-source message is kept only when no
        ``name`` was supplied at all.

        ``position`` (3 elements) and ``orientation`` (4-element wxyz quaternion)
        are validated up front: a wrong-length, non-numeric, or non-finite
        (nan/inf) vector returns an actionable ``{"status": "error"}`` and
        leaves the simulation unchanged, rather than baking a degenerate pose
        into the robot's base transform. NumPy scalar components are accepted.

        ``position`` OFFSETS the model's own authored root pose rather than
        replacing it: it is written as the attach frame's translation, which
        MuJoCo composes with the ``pos`` the model's root body declares. A
        ground-bolted arm declares ``pos="0 0 0"``, so for those the offset IS
        the world position - but a locomotion model is authored standing, and
        ``position=[0, 0, 0.4]`` on a Unitree Go2 (base ``pos`` ``z=0.445``)
        compiles its base at ``z=0.845``. This differs from
        :meth:`add_object`, whose ``position`` does place its body at exactly
        that world point. The returned message reports the MEASURED world
        position of the robot's root body, and names the request and the
        model's own offset beside it whenever the two differ, so a spawn that
        did not land where it was asked is visible in the result instead of
        having to be measured with :meth:`get_body_state`.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("add_robot"):
            return err

        # Refuse a name that cannot address the robot this call creates. ``None``
        # and ``""`` are the documented "derive a label from the model" short
        # form and are passed through to the derivation below; every other value
        # goes through the shared domain. That closes two gaps at once: an int
        # name compiled its bodies under the ``7/`` namespace but keyed the
        # registry with the int ``7``, which the tool surface - where a name
        # always arrives as a JSON string - can never address, and ANY other
        # falsy value (``0``, ``[]``, ``{}``) fell into the derive branch below
        # and reported success under a label the caller never asked for.
        if not (name is None or (isinstance(name, str) and name == "")):
            if (name_err := entity_name_error("add_robot", "name", name)) is not None:
                return {"status": "error", "content": [{"text": name_err}]}

        # Validate the caller-supplied base pose before it is baked into the
        # robot's frame pos/quat. Without this, `add_robot` shares the numeric
        # -vector failure classes already guarded on `add_object` / `move_object`
        # / `add_camera`: a nan/inf `position`/`orientation` is written verbatim
        # into the base transform and propagated across the whole physics state
        # by `mj_forward` while `add_robot` still reports `status="success"`
        # (silent corruption); a wrong-length vector yields a generic "Failed to
        # inject robot" with no hint that the length was wrong; and a non-numeric
        # element raises a bare MuJoCo `add_frame(): incompatible function
        # arguments` TypeError that escapes the structured-error contract. NumPy
        # scalar components are accepted.
        position, e = coerce_pose_vector("add_robot", "position", position, 3)
        if e is not None:
            return {"status": "error", "content": [{"text": e}]}
        orientation, e = coerce_pose_vector("add_robot", "orientation", orientation, 4)
        if e is not None:
            return {"status": "error", "content": [{"text": e}]}

        # Remember whether the caller supplied a `name` (vs an auto-derived
        # label): an explicit name that resolves to no model is a
        # mistyped/unknown robot, not a 'you forgot a model source' case.
        explicit_name = name
        # Auto-derive an instance label when the caller didn't supply one.
        # Friction fix: ``name`` used to be required, so a natural
        # ``add_robot(data_config="so101")`` failed with "requires parameter
        # 'name'". Now the label defaults to the model name (deduped).
        if not name:
            base = data_config or (os.path.splitext(os.path.basename(urdf_path))[0] if urdf_path else None) or "robot"
            name = base
            i = 2
            while name in self._world.robots:
                name = f"{base}_{i}"
                i += 1

        if name in self._world.robots:
            taken = ", ".join(sorted(self._world.robots)) or "(none)"
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Robot '{name}' already exists. Pick a different "
                            f"name, or omit name= to auto-number. Existing: {taken}."
                        )
                    }
                ],
            }

        # Resolution precedence:
        #   1. explicit `urdf_path` (anything on disk).
        #   2. `data_config` looked up in the model registry.
        #   3. DEPRECATED: `name` looked up in the registry (undocumented
        #      fallback kept for one release with a DeprecationWarning).
        # Pass `data_config` for new code; the `name`-as-registry-key path
        # will be removed.
        resolved_path = urdf_path
        if not resolved_path and data_config:
            resolved_path = resolve_model(data_config)
            if not resolved_path:
                return {
                    "status": "error",
                    "content": [{"text": self._unknown_model_msg(data_config)}],
                }
        elif not resolved_path and name:
            # deprecated fallback - try registry by instance name.
            resolved_path = resolve_model(name)
            if resolved_path:
                logger.info(
                    "add_robot: resolved model via instance name '%s'. "
                    "Prefer: add_robot(name='<instance_label>', data_config='%s')",
                    name,
                    name,
                )
                self._add_robot_deprecation_hint: str | None = (
                    f"Hint: add_robot(name='{name}') resolved via deprecated "
                    f"name-as-registry-key fallback. Prefer: "
                    f"add_robot(name='<instance_label>', data_config='{name}')."
                )

        if not resolved_path:
            # A caller-provided `name` that resolves to no model is almost
            # always a mistyped/unknown robot (the deprecated
            # name-as-registry-key short form). Surface the same actionable
            # "no model found (did you mean ...?) / list_urdfs" error the
            # data_config path and the Robot() factory give, instead of a
            # dead-end "supply urdf_path or data_config". The bare
            # "supply a model source" message is kept only for the no-name
            # case (auto-derived label, nothing to resolve).
            if explicit_name:
                msg = self._unknown_model_msg(explicit_name)
                msg += (
                    f" Or pass data_config=<registered model> or urdf_path=<file> "
                    f"to add '{explicit_name}' as an instance under that label."
                )
                return {"status": "error", "content": [{"text": msg}]}
            return {"status": "error", "content": [{"text": "Either urdf_path or data_config is required."}]}
        if not os.path.exists(resolved_path):
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        mj = self._mj

        robot = SimRobot(
            name=name,
            urdf_path=resolved_path,
            # ``is None`` means "omitted" -> the documented default. ``or`` would
            # additionally read a NumPy pose as ambiguous (a bare ValueError) and
            # an empty vector as omitted; coerce_pose_vector already rejected
            # the latter and normalized the former to plain floats.
            position=[0.0, 0.0, 0.0] if position is None else position,
            orientation=[1.0, 0.0, 0.0, 0.0] if orientation is None else orientation,
            data_config=data_config,
            namespace=f"{name}/",
        )

        try:
            # Propagate auto-download failure back to the agent instead of
            # silently eating it (previously this dict was discarded and
            # the next MuJoCo load threw a cryptic 'mesh not found').
            mesh_err = self._ensure_meshes(resolved_path, data_config or name)
            if mesh_err is not None:
                self._world.robots.pop(name, None)
                return mesh_err

            # Resolve the requested spawn keyframe from the robot's SOURCE
            # model BEFORE mutating the scene, so an unknown keyframe fails
            # cleanly (naming the available keyframes) without leaving a
            # half-added robot behind.
            home_by_short: dict[str, list[float]] | None = None
            home_actuators_by_short: dict[str, tuple[float, list[float]]] | None = None
            if keyframe is not None:
                home_by_short, home_actuators_by_short, kf_err = self._keyframe_home_state(resolved_path, keyframe)
                if kf_err is not None:
                    return kf_err

            # Register the robot BEFORE attach so scene_ops can re-discover
            # its joint/actuator IDs inside the merged model.
            self._world.robots[name] = robot
            # Track robot base path for asset path resolution.
            if not self._world._backend_state.get("robot_base_xml"):
                self._world._backend_state["robot_base_xml"] = resolved_path

            # Compose into the live spec via spec.attach(). The helper sets
            # robot.joint_names from the source spec (pre-namespacing) and
            # then scene_ops._recompile_preserving_state resolves the
            # post-attach joint/actuator IDs on the compiled model.
            ok = inject_robot_into_scene(self._world, robot, resolved_path)
            if not ok:
                del self._world.robots[name]
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject robot '{name}' into scene."}],
                }

            # Discover cameras that the robot's source MJCF declared. The
            # compiled model already has them namespaced under
            # ``{robot.name}/<cam_name>``. We probe the post-compile model
            # instead of the source, which avoids loading a second model
            # just for introspection.
            #
            # The probe walks the MERGED model, so it sees every camera in the
            # scene - the overview camera, cameras a caller added, and the
            # cameras of robots already attached. Only the ones under THIS
            # robot's namespace are its own, hence the prefix test: without it
            # a camera another robot contributed is registered a second time
            # under its qualified name and recorded as belonging to the robot
            # being added now, which makes ``origin_robot`` (the key
            # ``remove_robot`` cleans up by) name the wrong robot.
            pfx = robot.namespace or f"{name}/"
            model = self._world._model
            for i in range(model.ncam):
                cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name or not cam_name.startswith(pfx):
                    continue
                # Key on the short name - the registry is keyed on it and we
                # re-attach the namespace when passing to the renderer. When
                # another robot already claimed that short alias (two arms both
                # declaring ``wrist``), key on the qualified name instead: it is
                # unique per robot by construction, so the camera stays
                # addressable and owned rather than being dropped.
                short = cam_name[len(pfx) :]
                key = short if not registered(self._world.cameras, short) else cam_name
                if key != short:
                    logger.info(
                        "Robot '%s' camera '%s' registered as '%s': the short name '%s' is "
                        "already taken by camera '%s'.",
                        name,
                        cam_name,
                        key,
                        short,
                        getattr(registry_entry(self._world.cameras, short), "name", short),
                    )
                self._world.cameras[key] = SimCamera(
                    name=cam_name,
                    camera_id=i,
                    width=self.default_width,
                    height=self.default_height,
                    origin_robot=name,
                )

            # Leave the freshly-added robot in a clean, deterministic state:
            # the zero configuration by default, or -- when a spawn keyframe was
            # requested -- that keyframe's canonical home pose. Callers that want
            # a pre-settled pose call step().
            #
            # Scoped to THIS robot. ``mj_resetData`` would supply the same clean
            # state, but for the whole world: the arm a caller just parked with
            # send_action goes limp and collapses, an object that has settled or
            # been carried somewhere teleports back to its declared spawn, a
            # latched apply_force wrench is dropped, and the clock rewinds - all
            # reported as a successful "add a robot". That contradicts this
            # method's own contract ("preserves previously-created world state
            # (gravity, objects, cameras, other robots)"), and the world-wide
            # reset is what forced the home-pose re-apply that used to follow it:
            # a partial repair that only covered robots spawned with a keyframe.
            self._reset_robot_to_reference(robot)
            if home_by_short:
                self._apply_home_state_to_robot(robot, home_by_short, home_actuators_by_short or {})
            # Seat only the new base: the others are wherever the scene left
            # them, and re-seating a base that is not at its reference height
            # would stack the terrain offset onto it again.
            self._seat_floating_bases_on_terrain(only=robot)
            mj.mj_forward(self._world._model, self._world._data)

            # Attach the robot to the mesh as its own peer so the agent can
            # address it directly (e.g. ``robot_mesh tell target=<peer_id>``)
            # rather than going through the sim container. Best-effort: a
            # mesh failure must not prevent ``add_robot`` from returning a
            # working robot. Only attempt when the parent sim is itself
            # already on a mesh.
            self._attach_robot_to_mesh(robot)

            source = f"data_config='{data_config}'" if data_config else os.path.basename(resolved_path)
            mesh_line = f"\nMesh peer: {robot.peer_id}" if robot.peer_id else ""
            hint = getattr(self, "_add_robot_deprecation_hint", None)
            self._add_robot_deprecation_hint = None
            hint_line = f"\nWarning: {hint}" if hint else ""
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Robot '{name}' added to simulation\n"
                            f"Source: {source} -> {os.path.basename(resolved_path)}\n"
                            f"Position: {self._describe_robot_placement(robot)}\n"
                            f"Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                            f"Actuators: {len(robot.actuator_ids)}\n"
                            f"Cameras: {list(self._world.cameras.keys())}"
                            f"{mesh_line}\n"
                            f"Run policy: action='run_policy', robot_name='{name}'"
                            f"{hint_line}"
                        )
                    }
                ],
            }
        except Exception as e:
            # Clean up on failure
            self._world.robots.pop(name, None)
            logger.error("Failed to add robot '%s': %s", name, e)
            return {"status": "error", "content": [{"text": f"Failed to load: {e}"}]}

    def _keyframe_home_state(
        self, resolved_path: str, keyframe: str | int
    ) -> tuple[
        dict[str, list[float]] | None,
        dict[str, tuple[float, list[float]]] | None,
        dict[str, Any] | None,
    ]:
        """Read a robot's ``<keyframe>`` home state from its SOURCE model.

        Returns ``(qpos_by_short_joint, actuators_by_short_name, None)`` -- each
        source joint's short name mapped to its qpos slice, and each source
        actuator's short name mapped to its keyed ``(ctrl, act)`` pair -- or
        ``(None, None, error_result)`` when the source model cannot be compiled
        or the keyframe name/index is unknown (the error names the available
        keyframes so the caller can fix it).

        A MuJoCo ``<key>`` pairs a pose with the actuator command that HOLDS
        that pose, and ``mj_resetDataKeyframe`` -- MuJoCo's own definition of
        what a keyframe restores -- writes both. Reading ``key_qpos`` alone
        leaves a gravity-loaded arm standing at its home pose while every
        actuator is commanded to the zero configuration, so the first step
        drives it off home: the pose is not self-holding, which is the whole
        point of a canonical home. 28 of the 31 built-in registry robots that
        ship a ``<keyframe>`` declare a non-zero ``ctrl`` in it.

        The keyed command is captured VERBATIM and never classified by actuator
        type. The MJCF author chose those numbers for those actuators, so a
        servo's setpoint, a motor's torque and a stateful actuator's activation
        each carry across as whatever quantity their own actuator reads -- the
        same reason ``_snapshot_scene_state`` carries ``ctrl``/``act`` as one
        pair per named actuator on the eject path.

        Not read here, because a per-robot apply cannot own them:

        * ``qvel`` -- the robot-scoped reset that runs immediately before the
          apply (:meth:`_reset_robot_to_reference`) zeroes it deliberately, so
          that a freshly added robot starts from a defined configuration and
          "callers that want a pre-settled pose call step()". Spawning a robot
          already in motion is a different contract, and no built-in registry
          keyframe declares a non-zero ``qvel``.
        * ``time`` and the mocap pools -- world-scope buffers with no slice that
          belongs to one robot. :meth:`reset` owns the clock.
        """
        mj = self._mj
        fname = os.path.basename(resolved_path)
        # ``bool`` is an ``int`` subclass; reject it explicitly so True/False is
        # never silently taken as keyframe index 1/0.
        if isinstance(keyframe, bool):
            return (
                None,
                None,
                {
                    "status": "error",
                    "content": [{"text": "keyframe must be a keyframe name (str) or index (int), not a bool."}],
                },
            )
        try:
            src = mj.MjModel.from_xml_path(resolved_path)
        except Exception as e:  # noqa: BLE001 - surface any compile failure to the caller
            return (
                None,
                None,
                {
                    "status": "error",
                    "content": [{"text": f"Cannot read keyframe from '{fname}': {e}"}],
                },
            )
        names = [mj.mj_id2name(src, mj.mjtObj.mjOBJ_KEY, i) for i in range(src.nkey)]
        if src.nkey == 0:
            return (
                None,
                None,
                {
                    "status": "error",
                    "content": [
                        {"text": f"Model '{fname}' declares no <keyframe>; cannot apply keyframe={keyframe!r}."}
                    ],
                },
            )
        if isinstance(keyframe, int):
            if keyframe < 0 or keyframe >= src.nkey:
                return (
                    None,
                    None,
                    {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"keyframe index {keyframe} out of range; '{fname}' has "
                                    f"{src.nkey} keyframe(s): {names}."
                                )
                            }
                        ],
                    },
                )
            idx = keyframe
        else:
            if keyframe not in names:
                avail = ", ".join(repr(n) for n in names)
                return (
                    None,
                    None,
                    {
                        "status": "error",
                        "content": [{"text": f"Keyframe {keyframe!r} not found in '{fname}'. Available: {avail}."}],
                    },
                )
            idx = names.index(keyframe)
        kq = src.key_qpos[idx]
        home: dict[str, list[float]] = {}
        for j in range(src.njnt):
            jn = mj.mj_id2name(src, mj.mjtObj.mjOBJ_JOINT, j)
            if not jn:
                continue
            adr = int(src.jnt_qposadr[j])
            width = _jnt_qpos_width(mj, int(src.jnt_type[j]))
            home[jn] = [float(x) for x in kq[adr : adr + width]]
        kc = src.key_ctrl[idx]
        ka = src.key_act[idx]
        actuators: dict[str, tuple[float, list[float]]] = {}
        for a in range(int(src.nu)):
            an = mj.mj_id2name(src, mj.mjtObj.mjOBJ_ACTUATOR, a)
            if not an:
                # Nothing names it, so it cannot be matched inside the merged
                # model -- the same limit ``_snapshot_scene_state`` records for
                # an unnamed actuator. Its joint keeps the keyed pose; only the
                # command holding it is lost, so say which one rather than
                # failing the spawn (no built-in registry robot has one).
                logger.debug(
                    "_keyframe_home_state: actuator id %d in '%s' has no name; its keyed ctrl/act is not applied",
                    a,
                    fname,
                )
                continue
            act_adr = int(src.actuator_actadr[a])
            act_num = int(src.actuator_actnum[a])
            act_vals = [float(x) for x in ka[act_adr : act_adr + act_num]] if act_adr >= 0 else []
            actuators[an] = (float(kc[a]), act_vals)
        return home, actuators, None

    def _apply_home_state_to_robot(
        self,
        robot: SimRobot,
        home_by_short: dict[str, list[float]],
        actuators_by_short: dict[str, tuple[float, list[float]]],
    ) -> None:
        """Write a captured keyframe home state onto ``robot`` in the live model
        and record it on the robot so :meth:`reset` can restore it.

        Both halves of the keyframe are written: the pose, and the actuator
        command that holds the pose (see :meth:`_keyframe_home_state` for why
        one without the other is not a home pose). The applied state is recorded
        keyed by NAMESPACED name, which is what :meth:`_restore_home_state`
        resolves against the merged model.

        Matching is by name under ``robot.namespace``, so only the joints and
        actuators this robot contributed are written and a robot already in the
        world keeps the pose it is in and the setpoints holding it there -- the
        guarantee :meth:`add_robot` states for the scene it joins.

        The caller runs ``mj_forward`` afterwards.
        """
        mj = self._mj
        assert self._world is not None and self._world._model is not None and self._world._data is not None
        model = self._world._model
        data = self._world._data
        pfx = robot.namespace or ""
        stored: dict[str, list[float]] = {}
        for j in range(model.njnt):
            jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, j)
            if not jn:
                continue
            short = jn[len(pfx) :] if pfx and jn.startswith(pfx) else jn
            vals = home_by_short.get(short)
            if vals is None:
                continue
            adr = int(model.jnt_qposadr[j])
            width = _jnt_qpos_width(mj, int(model.jnt_type[j]))
            if len(vals) != width:
                # width mismatch (unexpected for a matching source model): skip
                # defensively rather than corrupt an adjacent joint's slice.
                continue
            data.qpos[adr : adr + width] = vals
            stored[jn] = vals
        robot.home_qpos = stored
        stored_actuators: dict[str, tuple[float, list[float]]] = {}
        for a in range(int(model.nu)):
            an = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, a)
            if not an:
                continue
            short = an[len(pfx) :] if pfx and an.startswith(pfx) else an
            keyed = actuators_by_short.get(short)
            if keyed is None:
                continue
            ctrl_val, act_vals = keyed
            data.ctrl[a] = ctrl_val
            act_adr = int(model.actuator_actadr[a])
            act_num = int(model.actuator_actnum[a])
            if act_adr < 0 or len(act_vals) != act_num:
                # The activation width the source model declared does not match
                # the merged one (unexpected for a matching source model): keep
                # the setpoint, and leave the fresh zero rather than write a
                # mismatched slice into a neighbouring actuator's activation.
                if act_vals or act_num:
                    logger.warning(
                        "_apply_home_state_to_robot: act width mismatch for actuator %r (%d!=%d), skipping activation",
                        an,
                        len(act_vals),
                        act_num,
                    )
                stored_actuators[an] = (ctrl_val, [])
                continue
            for i, v in enumerate(act_vals):
                data.act[act_adr + i] = v
            stored_actuators[an] = (ctrl_val, act_vals)
        robot.home_actuators = stored_actuators

    def _reset_robot_to_reference(self, robot: SimRobot) -> None:
        """Put one robot's joints and velocities at the model's reference
        configuration, leaving the rest of the world untouched.

        A freshly added robot needs a defined starting configuration - the
        recompile that merged it in leaves its new joints at whatever the
        compiler initialized them to. ``mj_resetData`` supplies one, but it
        supplies it for the WHOLE world: every other robot's pose and actuator
        setpoints, every settled object's position, latched wrenches and the
        clock all go back to their reference values too. On a scene that is
        being built up incrementally - the documented way to compose one - that
        turns "add a robot" into "rewind the scene", which is the opposite of
        what :meth:`add_robot` promises its caller.

        Scoping the reset to the robot being added keeps both halves true: the
        new arm starts from a known configuration and nothing else in the world
        moves.

        Actuator setpoints are deliberately NOT this function's business. A
        freshly added robot's ``ctrl``/``act`` entries are new, and defining a
        new entry is the recompile's job -- ``_recompile_preserving_state``
        zeroes the tail its positional state transfer leaves undefined, before
        anything reads it. Repeating that here by iterating ``actuator_ids``
        would be both narrower and less safe, because the two answer different
        questions: ``actuator_ids`` is ownership -- which robot may command an
        actuator -- whereas the tail is a memory fact about which entries the
        positional transfer never wrote. ``mj_checkCtrl`` reads the whole buffer,
        so the obligation is positional and unconditional, while ownership is
        derived and not guaranteed to cover the tail:
        :func:`~strands_robots.simulation.mujoco.scene_ops.robot_owned_actuator_ids`
        returns an empty list for an actuator that is neither namespace-prefixed
        nor joint-driven -- the fixed tendon that couples a gripper's fingers,
        say, whose transmission is gated out of the driven-joint match by design.
        Sourcing initialization from it would make this method silently wrong
        wherever ownership is not a complete cover.

        Args:
            robot: The robot to reset. Its ``joint_ids`` are the post-recompile
                ids resolved by
                :func:`~strands_robots.simulation.mujoco.scene_ops._recompile_preserving_state`,
                which looks joints up by name and so needs no id-space caveat.

        Notes:
            The caller holds the model lock and runs ``mj_forward`` afterwards.
        """
        mj = self._mj
        assert self._world is not None and self._world._model is not None and self._world._data is not None
        model = self._world._model
        data = self._world._data
        for jid in robot.joint_ids:
            jnt_type = int(model.jnt_type[jid])
            adr = int(model.jnt_qposadr[jid])
            width = _jnt_qpos_width(mj, jnt_type)
            # ``qpos0`` is the reference configuration ``mj_resetData`` writes,
            # so reading it per joint reproduces that value without touching a
            # joint this robot does not own.
            data.qpos[adr : adr + width] = model.qpos0[adr : adr + width]
            dadr = int(model.jnt_dofadr[jid])
            data.qvel[dadr : dadr + _jnt_dof_width(mj, jnt_type)] = 0.0

    def _restore_home_state(self) -> None:
        """Re-apply every robot's captured keyframe home state onto the live
        data: the pose, and the actuator command that holds it.

        ``mj_resetData`` zeroes ``ctrl`` and ``act`` along with everything else,
        so restoring the pose alone puts a gravity-loaded arm at its home
        configuration with every actuator commanded to zero -- it sags off home
        over the first steps of the new rollout, which is the state a policy's
        FIRST inference of every episode then sees.

        A no-op for robots spawned without a keyframe (both records are empty).
        The caller holds the model lock and runs ``mj_forward`` afterwards.
        """
        mj = self._mj
        assert self._world is not None and self._world._model is not None and self._world._data is not None
        model = self._world._model
        data = self._world._data
        for robot in self._world.robots.values():
            hq = getattr(robot, "home_qpos", None)
            if not hq:
                continue
            for jn, vals in hq.items():
                jid = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, jn)
                if jid < 0:
                    continue
                adr = int(model.jnt_qposadr[jid])
                data.qpos[adr : adr + len(vals)] = vals
            for an, (ctrl_val, act_vals) in getattr(robot, "home_actuators", {}).items():
                aid = mj_name_to_id(model, mj.mjtObj.mjOBJ_ACTUATOR, an)
                if aid < 0:
                    continue
                data.ctrl[aid] = ctrl_val
                act_adr = int(model.actuator_actadr[aid])
                if act_adr < 0 or len(act_vals) != int(model.actuator_actnum[aid]):
                    continue
                data.act[act_adr : act_adr + len(act_vals)] = act_vals

    def _describe_robot_placement(self, robot: SimRobot) -> str:
        """Describe where ``robot`` actually stands, and why it differs.

        ``add_robot`` used to echo the requested ``position`` back as the
        robot's placement. For a model whose root body declares a non-zero
        ``pos`` that names a place the robot is not: ``position=[0, 0, 0.4]``
        on a Unitree Go2 compiles its base at ``z=0.845``, and the reported
        ``0.4`` is the one number the caller had to go on. The sibling call
        ``add_object(position=...)`` does place its body at exactly that world
        point, so one parameter name meant two different things depending on
        which entity it addressed, with nothing in the result to say so.

        Report the measured world position, and - only when it differs from
        the request - name the request and the model's own root offset beside
        it, so the caller can see both what it asked for and what the model
        added. A robot whose root offset is zero (every ground-bolted arm) or
        whose roots cannot be reduced to one pose keeps the original one-vector
        form, so their messages are unchanged.
        """
        requested = list(robot.position or (0.0, 0.0, 0.0))
        actual = self._robot_root_world_position(robot)
        if actual is None:
            return f"{requested}"
        offset = [round(a - r, 4) for a, r in zip(actual, requested, strict=False)]
        if not any(abs(component) > 1e-9 for component in offset):
            return f"{actual}"
        return f"{actual} (position={requested} + model root offset {offset})"

    def _robot_root_world_position(self, robot: SimRobot) -> list[float] | None:
        """Return the world position of ``robot``'s single root body, or ``None``.

        ``position`` is written as the attach FRAME's translation, and MuJoCo
        COMPOSES that frame with the model's own authored root pose - it does
        not replace it. So for any model whose root body declares a non-zero
        ``pos`` (30 of the 55 single-root robots in the built-in registry, e.g.
        the Unitree Go2 base at ``z=0.445``, the JVRC pelvis at ``z=1.4``) the
        robot does not stand where ``position`` names, and the requested vector
        alone cannot tell the caller where it does stand. Read the compiled
        placement back out of ``data.xpos`` so the answer is measured rather
        than assumed - the same value :meth:`get_body_state` would report.

        ``None`` when there is no compiled world yet, or when the robot has no
        single root body: an ``aloha`` attaches two independent arm bases and an
        ``rby1`` six, and a set of roots has no one base pose to name. Requires
        a preceding ``mj_forward`` so ``data.xpos`` is current.
        """
        mj = self._mj
        world = self._world
        if world is None or world._model is None or world._data is None:
            return None
        model, data = world._model, world._data
        prefix = robot.namespace or ""
        roots = [
            body
            for body in range(1, model.nbody)
            if int(model.body_parentid[body]) == 0
            and (name := mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, body)) is not None
            and name.startswith(prefix)
        ]
        if len(roots) != 1:
            return None
        return [round(float(v), 4) for v in data.xpos[roots[0]]]

    def _seat_floating_bases_on_terrain(self, only: SimRobot | None = None) -> None:
        """Raise each floating-base robot onto the local terrain surface.

        ``create_world(terrain=...)`` lays a heightfield whose surface rises up
        to ``TERRAIN_ELEVATION * difficulty`` above ``z=0``, but a robot's model
        spawns its free base at the flat-ground keyframe height (e.g. the
        Unitree Go2 base at ``z=0.445``, feet ~``z=0.02``). On a terrain world
        that leaves the feet BELOW the heightfield -- the robot spawns *buried*
        in the ground, with penetration that grows with the curriculum
        ``difficulty`` -- contradicting the terrain feature's stated purpose of
        spawning a locomotion robot ON non-flat ground. Offset each floating
        base's ``z`` by the terrain height beneath its ``(x, y)`` so it is
        seated on the surface (feet just clear of it), the correct initial
        state for a locomotion policy and a terrain-difficulty curriculum.

        A flat ground plane (``_ground_height_at`` returns ``0.0``) is a no-op,
        so non-terrain worlds are byte-for-byte unchanged; a fixed-base arm (no
        free joint) is skipped. Called once per spawn / reset cycle right after
        the home-pose restore (which returns each base to its flat keyframe z),
        so it starts from a known base height and is idempotent. ``only``
        restricts the pass to a single robot, which is what ``add_robot`` needs:
        seating is only idempotent for a base that is at its reference height,
        and a robot that has already walked somewhere is not - re-seating it
        would add the terrain offset to its current z a second time. Handles both a
        NAMED floating base (a humanoid's ``floating_base_joint``) and an
        UNNAMED ``<freejoint>`` (a mobile base) via
        :meth:`_robot_free_base_joint_id`. The caller holds the model lock and
        runs ``mj_forward`` afterwards.
        """
        world = self._world
        if world is None or world._model is None or world._data is None:
            return
        model = world._model
        data = world._data
        if model.nhfield == 0:  # flat ground plane -- nothing to seat onto
            return
        targets = [only] if only is not None else list(world.robots.values())
        for robot in targets:
            jid = self._robot_free_base_joint_id(model, robot)
            if jid < 0:  # fixed-base arm: no floating base to seat
                continue
            adr = int(model.jnt_qposadr[jid])
            ground = self._ground_height_at(float(data.qpos[adr]), float(data.qpos[adr + 1]))
            if ground:
                data.qpos[adr + 2] = float(data.qpos[adr + 2]) + ground

    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot and every element it injected (bodies, actuators,
        sensors, equality/tendon refs) from the MJCF scene, then recompile.

        Previously remove_robot only popped the Python-side dict entry,
        leaving the robot's MJCF in place. That blocked re-adding a robot
        with the same name (MuJoCo rejects duplicates on compile) and left
        stale bodies in the physics loop.

        Concurrency (GH #114): this is a *global-scope* mutation - the XML
        round-trip reallocates ``model``/``data`` and invalidates cached
        actuator/joint IDs held by every running PolicyRunner. We stop the
        target robot's own policy first (cooperatively), then require no
        OTHER robot is running a policy.
        """
        if self._world is None or not registered(self._world.robots, name):
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(name)}]}

        # Step 1: cooperatively stop THIS robot's policy if running.
        # Has to happen before the global check so remove_robot works even
        # when the target robot has an active policy (the common case).
        if registered(self._policy_threads, name):
            self._world.robots[name].policy_running = False
            fut = self._policy_threads[name]
            timeout = self._DEFAULT_POLICY_STOP_TIMEOUT
            with contextlib.suppress(Exception):
                # ``result`` raises either the worker's own exception - it has
                # exited, which is all we needed - or the join timeout. Which
                # one happened is decided by ``fut.done()`` below rather than by
                # the exception type, because the two are not distinguishable
                # that way: ``socket.timeout`` IS ``TimeoutError``, so a policy
                # server that stops answering raises the same class the join
                # does. Typing on the class would refuse a removal whose worker
                # has already exited.
                fut.result(timeout=timeout)
            if not fut.done():
                # The cooperative stop lapsed: the worker is still live (it is
                # blocked somewhere the stop flag is not read yet - inside a
                # policy inference call, typically). Keep the tracking entry.
                # It is the single record every bound on a live worker reads:
                # the global gate in step 2, ``list_policies_running``, and
                # cleanup's own bounded join. Deleting it here does not end the
                # worker, it only makes the worker unobservable - the gate then
                # admits every later scene mutation (``add_object``,
                # ``set_timestep``, ``add_robot``) for the rest of the session,
                # and cleanup's ``executor.shutdown(wait=False)`` runs on the
                # premise that all policy workers were drained.
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Robot '{name}' still has a live policy worker after waiting "
                                f"{timeout:.1f}s for it to stop, so the scene rebuild was refused: "
                                "it would reallocate the model/data that worker holds. The "
                                "cooperative stop has been requested and the worker exits at its "
                                "next control tick - retry action='remove_robot' (until it does, "
                                "action='list_policies_running' still reports it)."
                            )
                        }
                    ],
                }
            del self._policy_threads[name]

        # Step 2: after stopping our own, there must be no OTHER policy
        # running - an XML round-trip will invalidate cached IDs everywhere.
        if err := self._require_no_running_policy("remove_robot"):
            return err

        # An active attachment referencing one of this robot's bodies would be
        # silently lost by the scene rebuild (weld) or dangle (kinematic).
        # Fail fast so the caller detaches deliberately.
        if (attached_child := self.attachment_involving(name)) is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Robot '{name}' is referenced by an active attachment "
                            f"(child '{attached_child}'). Call detach_bodies first."
                        )
                    }
                ],
            }

        # Pop the robot from the registry BEFORE the rebuild - eject_robot_from_scene
        # rebuilds the spec from the remaining world.robots dict, so the robot
        # we want to drop must no longer be in it.
        robot_obj = self._world.robots[name]

        # The cooperative stop + no-running-policy gate above guarantee no
        # PolicyRunner worker is mid-mj_step: step 1 refuses outright when its
        # own robot's worker outlived the stop budget, so reaching here means
        # every worker is done. The XML round-trip below still
        # reallocates model/data, so serialize it under self._lock to exclude
        # the render/recorder daemon (rendering.py reads mjData under the same
        # lock). remove_robot is dispatched WITHOUT the blanket lock (see
        # _SELF_LOCKING_ACTIONS), so this acquisition is the real critical
        # section, not a reentrant no-op.
        with self._lock:
            del self._world.robots[name]

            # Detach the robot's per-peer mesh (if any) BEFORE the XML rebuild
            # so external peers see the peer leave the mesh promptly. This is
            # the inverse of the announce in ``add_robot`` / ``_attach_robot_to_mesh``.
            self._detach_robot_from_mesh(robot_obj)

            ejected = eject_robot_from_scene(self._world, name)
        if not ejected:
            # Unlikely - rebuild from world state with one fewer robot.
            return {
                "status": "error",
                "content": [{"text": f"Failed to eject robot '{name}' from scene."}],
            }

        return {"status": "success", "content": [{"text": f"Robot '{name}' removed."}]}

    def list_robots(self) -> list[str]:
        """Return ordered robot names (SimEngine ABC).

        For the user-facing agent-tool action (rich dict output) see
        :meth:`list_robots_info`, which the dispatcher aliases to the
        ``list_robots`` action string.
        """
        if self._world is None or not self._world.robots:
            return []
        return list(self._world.robots.keys())

    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Ordered joint names for ``robot_name`` (SimEngine ABC)."""
        if self._world is None or not registered(self._world.robots, robot_name):
            return []
        return list(self._world.robots[robot_name].joint_names)

    def robot_action_keys(self, robot_name: str) -> list[str]:
        """Actuator short-names a policy should emit for ``robot_name``.

        Overrides :meth:`SimEngine.robot_action_keys`. Returns the actuator
        names scoped to the robot's namespace - exactly the keys
        :meth:`send_action` resolves - rather than the joint names. This
        matters whenever a robot's actuator set differs from its joint set:

        * passive/mimic finger joints have no driving actuator (so keying by
          joint name emits keys that resolve to nothing), and
        * a tendon-driven gripper is an *actuator* with no matching joint name.

        Keying a policy by :meth:`robot_joint_names` in those cases makes the
        affected DOFs silently no-op. The namespace short-names are produced by
        :meth:`_get_valid_action_keys`, which strips the trailing-slash
        namespace prefix (e.g. ``"xarm7/"``).
        """
        if self._world is None or not registered(self._world.robots, robot_name):
            return []
        namespace = self._world.robots[robot_name].namespace or ""
        return self._get_valid_action_keys(namespace)

    def bind_policy_sim_context(self, policy: Any, robot_name: str) -> None:
        """Hand the compiled MjModel + robot namespace to policies that opt in.

        Enables zero-config IK for eef/cartesian-delta policies (e.g.
        ``VeraPolicy``): the policy auto-discovers its end-effector frame from
        the model scoped to this robot's namespace. No-op for policies without
        ``set_sim_context``; never fails a rollout on a binding error.
        """
        ctx = getattr(policy, "set_sim_context", None)
        if not callable(ctx):
            return
        if self._world is None or self._world._model is None:
            return
        if not registered(self._world.robots, robot_name):
            return
        namespace = self._world.robots[robot_name].namespace or ""
        try:
            ctx(self._world._model, namespace)
        except Exception as exc:  # noqa: BLE001 - non-fatal (mirrors set_robot_state_keys)
            logger.debug("bind_policy_sim_context(%s) failed: %s", robot_name, exc)

    def _maybe_install_wbc_torque_control(self, policy: Any, robot_name: str) -> Callable[[], None] | None:
        """Auto-install the WBC torque shim when a WBCPolicy drives a servo scene.

        Overrides :meth:`SimEngine._maybe_install_wbc_torque_control`. WBC emits
        joint-position targets; on the stock position-servo Unitree G1 those
        targets fight the uniform ``kp=500`` servo gain and override SONIC's
        tuned per-joint PD, so ``sim.run_policy(policy_provider="wbc")`` would
        otherwise fall over within a fraction of a second. When the driven
        actuators are position-servo (see
        :func:`~strands_robots.policies.wbc.wbc_uses_position_servo`) and no
        action controller is already installed, this wires up
        :func:`~strands_robots.policies.wbc.install_wbc_torque_control` and
        returns its :meth:`uninstall` so the scene is restored after the run.

        ``policy`` may be the ``WBCPolicy`` itself or any wrapper that declares
        it through :attr:`~strands_robots.policies.base.Policy.children` - a
        :class:`~strands_robots.policies.composite.CompositePolicy` driving the
        legs from WBC and the arms from a manipulation policy, or a
        :class:`~strands_robots.policies.persistent.PersistentPolicy` holding it
        warm. The shim is resolved by walking that tree
        (:func:`~strands_robots.policies.base.iter_policy_tree`), because the
        physics it corrects is a property of the WBC policy driving the joints,
        not of the type of object handed to ``run_policy``.

        Returns ``None`` (no-op) in five cases, in the order they are checked:
        ``[wbc]`` is not installed; no ``WBCPolicy`` appears in ``policy``'s
        tree; the sim has no compiled world; a controller is already registered
        (a manual install always wins); or
        :func:`~strands_robots.policies.wbc.wbc_uses_position_servo` finds no
        position-servo actuator, meaning the driven actuators are already torque
        motors or none of the WBC joints resolve in this scene.
        """
        try:
            from strands_robots.policies.base import iter_policy_tree
            from strands_robots.policies.wbc import (
                WBCPolicy,
                install_wbc_torque_control,
                wbc_uses_position_servo,
            )
        except ImportError:
            return None

        # The shim is keyed on the WBC policy actually driving the joints, which
        # may sit inside a wrapper (composite / persistent) that is not itself a
        # WBCPolicy. Walk the declared tree instead of type-testing the argument.
        wbc_policy = next((p for p in iter_policy_tree(policy) if isinstance(p, WBCPolicy)), None)
        if wbc_policy is None:
            return None
        world = self._world
        if world is None or world._model is None:
            return None
        backend_state = getattr(world, "_backend_state", None)
        if isinstance(backend_state, dict) and backend_state.get("action_controller") is not None:
            return None  # a manually-installed controller wins
        if not wbc_uses_position_servo(self, wbc_policy, robot_name):
            return None

        controller = install_wbc_torque_control(self, wbc_policy, robot_name)
        logger.info(
            "run_policy: auto-installed WBC torque control on %r (position-servo "
            "actuators detected). WBC emits joint-position targets the stock servo "
            "gain would override; the torque shim applies SONIC's per-joint PD law "
            "so the gait is stable. Pass wbc_install_torque_control=False to opt out.",
            robot_name,
        )

        # ``uninstall`` releases both halves of the install - the registration it
        # made and the actuator gains - so the caller of the *documented manual*
        # API gets the same teardown this hook does, from one implementation.
        return controller.uninstall

    def list_robots_info(self) -> dict[str, Any]:
        """Agent-tool action: pretty-printed robot listing.

        Separate from :meth:`list_robots` (which returns ``list[str]`` for
        the SimEngine ABC) because the dispatcher needs a dict-shaped
        response for user display.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if not self._world.robots:
            return {"status": "success", "content": [{"text": "No robots. Use action='add_robot'."}]}

        lines = ["Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            status = "running" if robot.policy_running else "idle"
            lines.append(
                f"  - {name} ({os.path.basename(robot.urdf_path)})\n"
                f"    Position: {robot.position}, Joints: {len(robot.joint_names)}, "
                f"Config: {robot.data_config or 'direct'}, Status: {status}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def describe(self) -> dict[str, Any]:
        """Return a machine-readable summary of this MuJoCo engine's live state.

        Overrides :meth:`SimEngine.describe` to include MuJoCo-specific camera
        names from the loaded world plus sim-time / world status, so agents can
        discover available robots, cameras, and methods without guessing.
        """
        base = super().describe()

        # Renderable camera names, incl. the built-in "default" free view.
        # Delegates to list_cameras() so the discovery surface is identical to
        # the Newton backend and always advertises "default" (which render()
        # accepts) regardless of the loaded MJCF's own camera names.
        base["cameras"] = self.list_cameras()

        bodies: list[str] = []
        if self._world is not None and self._world._model is not None:
            mj = self._mj
            model = self._world._model
            for i in range(model.nbody):
                body_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                if body_name:
                    bodies.append(body_name)
        base["bodies"] = bodies
        base["methods"]["list_bodies"] = "(robot_name: str | None = None) -> dict (camera mount points)"
        # Scene-construction + object-manipulation siblings that the base
        # discovery surface omits. describe() already advertises add_object /
        # remove_object and add_robot, but an agent enumerating how to build
        # and vary a scene from describe() alone could not discover the
        # alternative scene entry point (load_scene), the object siblings
        # (list_objects / move_object), or domain randomization - all
        # first-class facades it would otherwise have to guess by name.
        base["methods"]["load_scene"] = (
            "(scene_path: str) -> dict  # load a complete scene from an MJCF "
            "(or URDF) file; the alternative scene-construction entry point to "
            "add_robot - downstream add_object/add_camera/add_robot then mutate "
            "the loaded scene"
        )
        base["methods"]["list_objects"] = (
            "() -> dict  # enumerate objects added to the scene (the object sibling of list_robots / list_cameras)"
        )
        base["methods"]["move_object"] = (
            "(name: str, position=None, orientation=None) -> dict  # reposition "
            "an existing object; position is [x, y, z] meters, orientation is a "
            "[w, x, y, z] quaternion (either may be omitted to leave it unchanged)"
        )
        # Manipulation primitives (GH #1533): runtime grasp-assist attach/
        # detach, URDF-arm actuation, and the anti-explosion dynamics reset.
        base["methods"]["attach_bodies"] = (
            "(parent: str, child: str, mode='weld', torquescale=1.0) -> dict  # "
            "rigidly attach child to parent at their CURRENT relative pose "
            "(grasp-assist; mode='weld' adds an equality constraint, "
            "mode='kinematic' teleport-follows every step). NOT a physical grasp"
        )
        base["methods"]["detach_bodies"] = "(parent: str, child: str) -> dict  # release an attach_bodies attachment"
        base["methods"]["actuate_robot"] = (
            "(robot_name: str, kp=100.0, damping=2.0, armature=0.01, "
            "gravity_compensation=True, disable_self_collision=False) -> dict  # "
            "add position-servo actuators to an actuator-less (URDF-loaded) arm "
            "so send_action / run_policy can drive it (kp: number for all "
            "hinge/slide joints, or {joint: kp} for a subset)"
        )
        base["methods"]["zero_dynamics"] = (
            "(robot_name: str | None = None) -> dict  # zero qvel/qacc/warmstart "
            "(world-wide, or one robot's joints) after kinematic qpos writes"
        )
        # Analytic motion primitives (GH #1645): the agent-facing staging/
        # transport/release vocabulary around a learned policy (Harness VLA).
        base["methods"]["move_to"] = (
            "(robot_name=None, position, orientation=None, tol=0.01, "
            "max_steps=200, orientation_tol=None) -> dict  # move the "
            "end-effector to a world-frame [x, y, z] target via IK "
            "(position-only when orientation is omitted - right for <6-DOF "
            "arms); an orientation is CONVERGED to within orientation_tol "
            "radians (default 0.1), not just fed to the solver; NOT "
            "collision-aware; returns reached/residuals, structured error "
            "naming the out-of-reach component when the pose is unreachable"
        )
        base["methods"]["set_gripper"] = (
            "(robot_name=None, state='open'|'close', steps=12) -> dict  # "
            "drive the gripper actuator(s) to the open/close set-point "
            "(registry gripper metadata when present, else open=HIGH / "
            "close=LOW end of the actuator ctrlrange, or of the driven joint's "
            "range when the MJCF left the ctrlrange unset)"
        )
        base["methods"]["rotate_wrist"] = (
            "(robot_name=None, target_yaw, tol=0.02, max_steps=200) -> dict  "
            "# rotate the wrist-yaw joint to a set-point (radians) while the "
            "other arm joints hold position"
        )
        # Robot-registry + robot-removal surface. describe() calls add_robot
        # "the first scene-construction step" and advertises the object/camera
        # remove halves (remove_object / remove_camera), but not the robot
        # inverse remove_robot, nor how a caller discovers or extends the set of
        # names add_robot(name=...) accepts. All three are first-class MuJoCo
        # tool-spec + action-dispatcher actions: list_urdfs enumerates the
        # registered robot descriptions, register_urdf adds a new one so
        # add_robot can spawn it by name, and remove_robot ejects a robot (and
        # every MJCF element it injected) - completing the add/remove symmetry
        # that remove_object / remove_camera already establish. Listing them
        # here reveals the robot-registry surface from one describe() call
        # instead of by guessing. (Newton exposes the same trio and has the same
        # gap; deferred here to keep this diff MuJoCo-scoped like the sibling
        # describe() families.)
        base["methods"]["list_urdfs"] = (
            "() -> dict  # enumerate the robot descriptions registered for "
            "add_robot(name=...) (the source of truth for the names add_robot "
            "accepts, the robot-registry sibling of list_robots)"
        )
        base["methods"]["register_urdf"] = (
            "(data_config: str, urdf_path: str) -> dict  # register a URDF under "
            "a data_config name so add_robot(name=data_config) can spawn it; the "
            "way to extend the add_robot registry with a custom robot"
        )
        base["methods"]["remove_robot"] = (
            "(name: str) -> dict  # remove a robot and every MJCF element it "
            "injected (bodies, actuators, sensors), then recompile; the inverse "
            "of add_robot (cooperatively stops that robot's policy first, and "
            "requires no other robot is running a policy)"
        )
        base["methods"]["randomize"] = (
            "(randomize_colors=True, randomize_lighting=True, "
            "randomize_physics=False, randomize_positions=False, "
            "position_noise=0.02, seed=None, ...) -> dict  # domain randomization "
            "(each axis opt-in; no flags = no-op). Destructive - recompile to undo"
        )
        # Scene-construction cameras: the SO-101 rollout rig is built with
        # add_camera before run_policy, so the discovery surface must name it
        # (and its inverse) rather than leave a caller to guess.
        base["methods"]["add_camera"] = (
            "(name: str, position=None, target=None, fov=60.0, width=640, "
            "height=480, parent_body=None) -> dict  # attach a camera to the "
            "scene; parent_body mounts it on a moving link (e.g. a wrist cam)"
        )
        base["methods"]["remove_camera"] = "(name: str) -> dict  # remove a camera added via add_camera"
        base["methods"]["list_cameras"] = "() -> list[str]  # renderable camera names incl. the built-in 'default' view"
        # Rendering siblings of "render" (advertised in tool_spec.json + the
        # action dispatcher) that the base discovery surface omits. Listing
        # them here lets an agent enumerate the full render surface from
        # describe() rather than discovering depth / multi-view by guessing.
        base["methods"]["render_depth"] = (
            "(camera_name='default', width=None, height=None) -> dict "
            "(viewable grayscale depth PNG image block + metric depth_min/depth_max stats)"
        )
        base["methods"]["get_world_point"] = (
            "(camera_name='default', pixels=[[u, v], ...], width=None, height=None) -> dict  # "
            "pixel-to-world grounding: unprojects each [u, v] pixel through the metric depth "
            "buffer and returns the MEDIAN world [x, y, z] over the valid samples (plus "
            "per-pixel points and n_valid). Pick pixels ON the visible surface of the target; "
            "avoid rims/edges/reflections/background; sample several pixels on the same "
            "surface; re-localize after any robot/camera/object motion. The deployment-shaped "
            "alternative to get_body_state (works identically with a real RGB-D camera)"
        )
        base["methods"]["render_all"] = "(cameras=None, width=None, height=None) -> dict (one image block per camera)"
        base["methods"]["set_obs_noise"] = (
            "(joint_pos_std=0.0, joint_vel_std=0.0, camera_jitter_px=0.0, seed=None) -> dict "
            "(additive Gaussian sensor noise on joint observations + rendered frames)"
        )
        # Recording / dataset-collection surface (LeRobotDataset, [lerobot] extra).
        # Exposed here so agents discover the explicit episode-boundary workflow
        # (start_recording -> run_policy + save_episode per episode -> stop_recording)
        # instead of guessing method names.
        base["methods"]["start_recording"] = (
            "(repo_id='local/sim_recording', task='', fps=30, root=None, "
            "push_to_hub=False, vcodec='h264', overwrite=False, cameras=None) -> dict  "
            "(cameras= scopes the recorded LeRobotDataset to a subset of the "
            "scene's cameras; None records every camera)"
        )
        base["methods"]["save_episode"] = (
            "() -> dict  (flush current rollout as one episode; call once per "
            "run_policy to get N episodes instead of one merged episode. For "
            "the common case prefer run_policy(n_episodes=N) which flushes a "
            "boundary per episode automatically)"
        )
        base["methods"]["stop_recording"] = "(push_to_hub=False, bucket=None, run_id=None) -> dict"
        base["methods"]["get_recording_status"] = "() -> dict"
        base["methods"]["verify_dataset_episodes"] = (
            "(expected: int) -> dict  (after stop_recording, read the parquet "
            "and confirm the dataset holds exactly `expected` episodes; "
            "status=error on mismatch)"
        )

        # Plain-MP4 camera-recording surface ([sim-mujoco] extra, no lerobot).
        # The block above advertises the LeRobotDataset recording family
        # (start_recording -> save_episode -> stop_recording), which needs the
        # [lerobot] extra and writes a parquet+MP4 dataset. But an agent that
        # only wants a raw MP4 per camera -- or that lacks lerobot entirely --
        # had no way to discover the dependency-free recorder from describe()
        # alone. These three are first-class MuJoCo tool-spec + action-dispatcher
        # actions; listing them completes the recording surface with the plain
        # start/stop/status trio alongside the dataset trio.
        base["methods"]["start_cameras_recording"] = (
            "(cameras=None, output_dir=None, fps=30, width=None, height=None, "
            "name=None, max_frames_per_camera=3000) -> dict  # start a "
            "dependency-free background recorder that writes one MP4 per camera "
            "(no lerobot / dataset); cameras=None records every camera. The "
            "raw-MP4 sibling of start_recording's LeRobotDataset"
        )
        base["methods"]["stop_cameras_recording"] = (
            "() -> dict  # stop start_cameras_recording, flush each camera's "
            "buffer to an MP4, and report per-camera frame counts + paths; "
            "idempotent. The inverse of start_cameras_recording"
        )
        base["methods"]["get_cameras_recording_status"] = (
            "() -> dict  # inspect an in-progress start_cameras_recording "
            "(elapsed time, per-camera frame counts); reports idle when none is active"
        )

        # Physics-introspection / grounding surface. The discovery surface
        # teaches how to build a scene, run a policy, and record a dataset, but
        # previously gave no way to discover how to READ the physics result --
        # so an agent that ran a rollout could not learn how to verify it (read
        # a body's world pose, check whether the gripper is in contact, query a
        # sensor) without guessing method names. These are public MuJoCo methods
        # the tool spec + action dispatcher already expose; listing them here
        # lets one describe() call reveal the read/verify surface alongside the
        # act/record surface.
        base["methods"]["get_body_state"] = (
            "(body_name: str) -> dict  # world pose (xpos/xquat/xmat) + linear "
            "and angular velocity of a body; ground grasp/lift/move claims on "
            "this delta, not on a caption"
        )
        base["methods"]["forward_kinematics"] = (
            "(body_name: str | None = None) -> dict  # world poses of every "
            "body (or one) computed from the current qpos"
        )
        base["methods"]["get_contacts"] = (
            "() -> dict  # active geom-geom contacts at the current step (verify a grasp / detect a collision)"
        )
        base["methods"]["get_contact_forces"] = "() -> dict  # per-contact normal + friction force magnitudes"
        base["methods"]["get_sensor_data"] = (
            "(sensor_name: str | None = None) -> dict  # MuJoCo sensor readouts "
            "(jointpos/jointvel, accelerometer, gyro, force, torque, touch, "
            "rangefinder, framequat, ...)"
        )
        base["methods"]["get_energy"] = "() -> dict  # kinetic + potential energy of the system"
        base["methods"]["get_mass_matrix"] = (
            "() -> dict  # joint-space inertia matrix M(q); json carries the "
            "DOF-indexed diagonal plus dof_joint_names naming each entry's joint"
        )
        base["methods"]["inverse_dynamics"] = (
            "() -> dict  # gravity + Coriolis/bias compensation torques for the "
            "current pose (mj_inverse at zero desired acceleration)"
        )
        base["methods"]["get_jacobian"] = (
            "(body_name=None, site_name=None, geom_name=None) -> dict  # "
            "translational + rotational Jacobian of a body/site/geom for IK/control; "
            "columns are whole-model DOFs, and json carries dof_joint_names naming "
            "the joint that owns each column"
        )
        base["methods"]["get_total_mass"] = "() -> dict  # total mass of the model"
        base["methods"]["get_ground_height"] = (
            "(x, y) -> dict  # local terrain surface height (world z) beneath (x, y); "
            "0.0 on flat ground -- place objects/cameras/goals on create_world(terrain=...) ground"
        )
        base["methods"]["raycast"] = (
            "(origin: list[float], direction: list[float], exclude_body=-1, "
            "include_static=True) -> dict  # first geom hit along a world-frame "
            "ray + hit distance"
        )
        base["methods"]["multi_raycast"] = (
            "(origin: list[float], directions: list[list[float]], "
            "exclude_body=-1, include_static=True) -> dict  # batch raycast from one origin "
            "(e.g. a lidar fan); "
            "all-or-nothing - a direction it cannot cast refuses the batch instead of "
            "reporting that bearing as a miss"
        )

        # Physics-tuning / domain-perturbation WRITE surface -- the complement
        # of the physics-introspection READ family above. describe() teaches how
        # to build a scene, run a policy, read the physics result, and checkpoint
        # state, but previously listed no way to discover how to VARY the physics
        # engine itself: randomize gravity or the integration timestep, perturb a
        # body's mass or a geom's color/friction/size for domain randomization +
        # sim2real, or apply an external wrench for push-recovery / perturbation
        # testing. These are public MuJoCo methods the tool spec + action
        # dispatcher already expose (the engine's own guidance points a caller at
        # "set_gravity, set_timestep, etc."), and they are the write siblings of
        # the coarse-grained randomize() facade; listing them here reveals the
        # physics-write surface from one describe() call instead of by guessing.
        base["methods"]["set_gravity"] = (
            "(gravity: list[float] | float | int) -> dict  # set the world "
            "gravity vector [gx, gy, gz] (a scalar is taken as the -z magnitude); "
            "domain-randomize gravity or simulate reduced/zero-g"
        )
        base["methods"]["set_timestep"] = (
            "(timestep: float) -> dict  # set the physics integration timestep "
            "in seconds (smaller = more accurate, slower); the per-action substep "
            "count run_policy derives from control_frequency rescales with it"
        )
        base["methods"]["set_body_properties"] = (
            "(body_name: str, mass: float | None = None) -> dict  # set a body's "
            "mass (its inertia scales with the mass); domain-randomize dynamics"
        )
        base["methods"]["set_geom_properties"] = (
            "(geom_name=None, geom_id=None, color=None, friction=None, size=None) "
            "-> dict  # set a geom's color (RGB or RGBA), friction (3: sliding, "
            "torsional, rolling) or size (every component the geom type defines: "
            "sphere 1, capsule/cylinder 2, box/ellipsoid/plane 3); "
            "domain-randomize appearance + contact dynamics (identify the geom by "
            "name or id)"
        )
        base["methods"]["apply_force"] = (
            "(body_name: str, force=None, torque=None, point=None) -> dict  # "
            "latch an external force/torque wrench on a body; MuJoCo re-applies "
            "it on every subsequent step until the next apply_force for THAT "
            "body, so several bodies can hold wrenches at once (push-recovery / "
            "disturbance-rejection perturbation testing, wind, thrusters); "
            "apply_force(body, force=[0,0,0]) stops one body, reset() stops all"
        )

        # Sim-state checkpoint + direct pose-setting surface. describe() teaches
        # how to build a scene, run a policy, and read the physics result, but
        # previously gave no way to discover how to CHECKPOINT/RESTORE the whole
        # physics state or teleport the robot to a pose without stepping the
        # actuators -- so an agent setting up a deterministic initial condition
        # (or A/B-testing two rollouts from the same start) had to guess these
        # names. All four are first-class actions in the tool spec + action
        # dispatcher; listing them here reveals the state-management surface
        # alongside the act / read surfaces.
        base["methods"]["save_state"] = (
            "(name='default') -> dict  # checkpoint the full physics state "
            "(qpos/qvel/act/ctrl + sim time + step count) under a name; restore "
            "it later with load_state"
        )
        base["methods"]["load_state"] = (
            "(name='default') -> dict  # restore a checkpoint saved by save_state "
            "(errors if the name is unknown or a policy is running)"
        )
        base["methods"]["set_joint_positions"] = (
            "(positions: dict[str, float] | list[float], robot_name=None, hold=False) -> dict "
            "# write qpos directly and run forward kinematics (teleport / set an "
            "initial pose, bypassing the actuators). dict is per-joint; list is "
            "ordered and must match one robot's joint count (see get_features). "
            "The write is all-or-nothing: a dict key that is not a joint of the "
            "model is an error, not a silent skip (see robot_joint_names). "
            "Kinematic only: a joint held by a position servo is pulled back "
            "toward the servo's existing setpoint by the next step, and the "
            "success text names those joints; hold=True moves the matching "
            "position-servo setpoints with the pose so it survives stepping"
        )
        base["methods"]["set_joint_velocities"] = (
            "(velocities: dict[str, float] | list[float], robot_name=None) -> dict "
            "# write qvel directly (set an initial dynamic state); dict or "
            "ordered-list form mirrors set_joint_positions, including its "
            "all-or-nothing rejection of unknown joint names"
        )

        # Background-policy lifecycle. MuJoCo overrides start_policy to run in a
        # background thread (non-blocking), unlike the base engine's synchronous
        # passthrough -- so an agent that discovers start_policy here and launches
        # a rollout has no way to discover how to STOP it or see WHAT is running
        # without guessing these names. Both are first-class actions in the tool
        # spec + action dispatcher; listing them completes the start/stop/list
        # lifecycle on the discovery surface (the resource-management sibling of
        # run_policy's blocking rollout).
        base["methods"]["stop_policy"] = (
            "(robot_name: str) -> dict  # cooperatively stop the background "
            "policy started by start_policy on robot_name; idempotent (succeeds "
            "with 'Was not running' when none is active). The inverse of start_policy"
        )
        base["methods"]["list_policies_running"] = (
            "() -> dict  # names of robots currently running a background policy "
            "(inspect concurrent-policy state when driving two or more arms in one scene)"
        )

        # Multi-robot rollout + per-robot action/joint introspection. describe()
        # advertises run_policy (drive ONE robot with a created policy) and the
        # background start/stop/list lifecycle, but omits run_multi_policy -- the
        # facade that drives SEVERAL robots, each with its OWN Policy, in one
        # synchronized control loop that co-observes every robot into one merged
        # frame per timestep (the correct path for bimanual / handover / multi-
        # agent data collection; independent start_policy threads instead
        # interleave single-robot frames into a shared recorder). A caller
        # assembling its {robot_name: Policy} map also has to know exactly what
        # each robot's policy must emit, so this block advertises the two
        # per-robot introspection primitives alongside it: robot_action_keys
        # (the actuator short-names send_action resolves -- NOT always the joint
        # names, since passive/mimic fingers have no actuator and a tendon
        # gripper is an actuator with no joint) and robot_joint_names (the
        # ordered observation.state joint vector). Without these three an agent
        # could drive one robot from describe() but had to guess how to drive
        # many, or key a multi-robot policy by joint name and watch tendon/mimic
        # DOFs silently no-op. (get_features exposes the whole-scene view; these
        # return the exact per-robot lists a policy is keyed on. Newton exposes
        # the same trio and has the same gap; deferred to keep this diff
        # MuJoCo-scoped like the sibling describe() families.)
        base["methods"]["run_multi_policy"] = (
            "(policies: dict[str, Policy], instructions='' | dict, duration=10.0, "
            "control_frequency=50.0, action_horizon=8 | dict, n_steps=None, "
            "max_steps=None) -> dict  # drive MULTIPLE robots, each with its own "
            "Policy, in one synchronized loop that records ALL robots into ONE "
            "merged frame per timestep (prefixed state/action, e.g. "
            "'alice__shoulder_pan'); the concurrent multi-robot sibling of "
            "run_policy for bimanual / handover / multi-agent data collection. "
            "policies keys and action_horizon dict keys are robot names from the "
            "'robots' list; per-robot instructions/horizon are supported"
        )
        base["methods"]["robot_action_keys"] = (
            "(robot_name: str) -> list[str]  # the actuator short-names a policy "
            "should emit as its action-dict keys for robot_name -- the exact keys "
            "send_action resolves. NOT always the joint names: passive/mimic "
            "fingers have no driving actuator and a tendon gripper is an actuator "
            "with no joint, so keying by robot_joint_names makes those DOFs "
            "silently no-op. Use this to key each policy in a run_multi_policy map"
        )
        base["methods"]["robot_joint_names"] = (
            "(robot_name: str) -> list[str]  # the ordered joint names for "
            "robot_name -- the names of the observation.state vector a policy "
            "reads (the observation sibling of robot_action_keys' action side)"
        )

        # MJCF-editing surface. create_world / destroy (the world lifecycle) are
        # advertised on the base SimEngine.describe() contract; here we add the
        # MuJoCo-specific MJCF-editing family: patch or wholesale-replace the
        # live MjSpec and serialize the scene back to XML. All three are
        # first-class actions in the tool spec + action dispatcher, completing
        # the discovery surface with MJCF-authoring operations alongside the
        # build / act / read surfaces. (The URDF/model registry trio --
        # register_urdf / list_urdfs / remove_robot -- is advertised with the
        # robot-registry family earlier in describe().)
        base["methods"]["patch_scene_mjcf"] = (
            "(ops: list[dict]) -> dict  # apply structured edits to the live "
            "MjSpec atomically then recompile once (rolled back if any op fails). "
            "ops: add_body/add_geom/add_site/set_body_pos/set_body_quat/delete_body. "
            "The fast iterative-edit sibling of replace_scene_mjcf"
        )
        base["methods"]["replace_scene_mjcf"] = (
            "(xml: str) -> dict  # atomically replace the whole scene with "
            "agent-authored MJCF (compiled + validated; MuJoCo's compiler error "
            "returned verbatim on failure). Escape hatch when add_robot/add_object "
            "cannot express an element; leaves world.robots/objects/cameras "
            "registries untouched (caller reconciles)"
        )
        base["methods"]["export_xml"] = (
            "(output_path: str | None = None) -> dict  # serialise the current "
            "scene (incl. runtime mutations) to canonical MJCF via spec.to_xml(); "
            "writes to output_path when given, else returns the XML inline. The "
            "read sibling of replace_scene_mjcf"
        )
        # Teleoperation surface (TeleopMixin, shared with the hardware Robot).
        # describe() teaches how to build a scene and drive it with a policy,
        # but gave no way to discover the OTHER actuation source: driving a sim
        # robot from an attached teleoperator (a real leader arm, gamepad, or
        # keyboard) - the leader->follower / human-demonstration workflow that
        # feeds data collection. All six are public facade methods on the sim;
        # without them a caller could not learn from describe() how to attach an
        # input device, run the teleop loop, or stop it, and had to guess these
        # names. Listing them here reveals the teleop lifecycle (attach ->
        # teleoperate -> stop) as the human-driven sibling of run_policy.
        base["methods"]["attach_teleop"] = (
            "(device_or_spec, *, name=None, method=None, map_fn=None, **kwargs) "
            "-> Simulation  # attach a teleoperator (lazy - no hardware touched "
            "until teleoperate). device_or_spec is a built lerobot Teleoperator "
            "or a type string ('so101_leader', 'gamepad', 'keyboard'); map_fn "
            "remaps a real leader's joint names onto the sim robot's actuators. "
            "Returns self for chaining"
        )
        base["methods"]["teleoperate"] = (
            "(*, names=None, robot_name=None, hz=50.0, publish=False, "
            "block=False, duration=None) -> dict  # drive the sim from its "
            "attached teleoperator(s): each tick polls get_action(), applies "
            "map_fn, merges (last-wins), and send_action()s the result. "
            "block=False runs a background loop and returns immediately; "
            "duration stops it after N seconds; publish=True also mirrors the "
            "stream to the mesh. The human-driven sibling of run_policy"
        )
        base["methods"]["stop_teleoperate"] = (
            "() -> dict  # stop the background teleop loop, any mesh publishers, "
            "and disconnect every device; frame/error stats. The inverse of teleoperate"
        )
        base["methods"]["get_teleoperate_status"] = (
            "() -> dict  # local teleop-loop status (running, frames, errors, "
            "actual hz, attached devices); distinct from the mesh teleop status"
        )
        base["methods"]["list_teleops"] = (
            "() -> dict  # attached teleoperators and their connection state "
            "(the teleop sibling of list_robots / list_cameras)"
        )
        base["methods"]["detach_teleop"] = (
            "(name: str | None = None) -> dict  # detach one teleoperator by "
            "name, or all when name is None; disconnects each and stops the loop "
            "if it would be left with no devices. The inverse of attach_teleop"
        )

        # Live interactive viewer. describe() teaches how to build a scene, drive
        # it with a policy, and read/checkpoint the result, but gave no way to
        # discover how to OPEN a live window on the running model for human
        # inspection (watch a rollout, debug a pose, hand-verify a scene). Both
        # are first-class actions in the tool spec + action dispatcher, so a
        # caller enumerating the sim's contract from describe() alone had to
        # guess their names. open_viewer launches a passive MuJoCo viewer (an
        # interactive OpenGL window that requires a local display -- it errors on
        # a headless host, where render()/render_all() capture frames instead)
        # and close_viewer is its teardown, completing the discovery surface with
        # the human-inspection sibling of the render family.
        base["methods"]["open_viewer"] = (
            "() -> dict  # launch a passive interactive MuJoCo viewer window bound "
            "to the running model (mujoco.viewer.launch_passive); requires a local "
            "display -- errors on a headless host, where render()/render_all() "
            "capture frames instead. Idempotent ('Viewer already open' if one is up)"
        )
        base["methods"]["close_viewer"] = (
            "() -> dict  # close the interactive viewer opened by open_viewer; "
            "idempotent (succeeds even if none is open). The inverse of open_viewer"
        )

        if self._world is not None:
            base["sim_time"] = self._world.sim_time
            base["world_created"] = True
        else:
            base["world_created"] = False

        return base

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        """Return a robot's per-joint position/velocity, plus base pose for a floating base.

        The canonical name parameter is ``robot_name``. The router accepts
        ``name`` as an alias (bidirectional) so legacy LLM calls keep working,
        but new tool specs should document only robot_name.

        The ``json`` payload carries ``{"state": {joint: {"position", "velocity"}}}``
        for the scalar (hinge/slide) joints. A robot with a floating base (a
        6-DoF free joint - a humanoid's named ``floating_base_joint`` or a mobile
        base's unnamed ``<freejoint/>`` like LeKiwi) additionally carries a
        ``"base"`` entry with ``position`` (xyz), ``quaternion`` (w,x,y,z),
        ``linear_velocity`` and ``angular_velocity``. The free joint is NOT
        reported as a scalar joint (its qpos is [xyz+quat], not a single angle),
        and the base ``quaternion``/``angular_velocity`` match get_observation's
        ``base_quat``/``base_ang_vel`` for the same robot.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}
        if not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}

        mj = self._mj
        robot = self._world.robots[robot_name]
        model, data = self._world._model, self._world._data

        # Namespace-aware joint lookup (see add_robot / _apply_sim_action).
        pfx = robot.namespace or ""
        state = {}
        free_jnt_id = -1  # the robot's floating-base free joint, if any
        for jnt_name in robot.joint_names:
            jnt_id = -1
            if pfx:
                jnt_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jnt_id < 0:
                jnt_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id < 0:
                continue
            # A FREE joint (6-DoF floating base, e.g. a humanoid's named
            # ``floating_base_joint``) has no scalar hinge/slide value: its qpos
            # is [xyz(3) + quat(4)] and qvel is [linvel(3) + angvel(3)]. Reading
            # qpos[jnt_qposadr] as a "position" reports the base x-coordinate as a
            # joint angle and silently drops the orientation, so record it and
            # surface a structured ``base`` entry below instead.
            if model.jnt_type[jnt_id] == mj.mjtJoint.mjJNT_FREE:
                free_jnt_id = jnt_id
                continue
            state[jnt_name] = {
                "position": float(data.qpos[model.jnt_qposadr[jnt_id]]),
                "velocity": float(data.qvel[model.jnt_dofadr[jnt_id]]),
            }

        # Additive sensor noise (set_obs_noise); no-op when unconfigured. Runs
        # over the scalar joints only; the floating-base pose/twist below is left
        # un-noised, matching get_observation's base_quat / base_ang_vel contract.
        state = self._apply_state_noise(state)

        # Floating base: surface the full 6-DoF pose + twist under a ``base`` key,
        # consistent with get_observation's base_quat / base_ang_vel. Recovered
        # from the kinematic tree when the free joint is unnamed and therefore
        # absent from ``joint_names`` (e.g. a mobile base like LeKiwi).
        if free_jnt_id < 0:
            free_jnt_id = self._robot_base_free_joint(model, robot, pfx)
        base: dict[str, list[float]] | None = None
        if free_jnt_id >= 0:
            qadr = int(model.jnt_qposadr[free_jnt_id])
            vadr = int(model.jnt_dofadr[free_jnt_id])
            base = {
                "position": [float(v) for v in data.qpos[qadr : qadr + 3]],
                "quaternion": [float(v) for v in data.qpos[qadr + 3 : qadr + 7]],
                "linear_velocity": [float(v) for v in data.qvel[vadr : vadr + 3]],
                "angular_velocity": [float(v) for v in data.qvel[vadr + 3 : vadr + 6]],
            }

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

    def list_bodies(self, robot_name: str | None = None) -> dict[str, Any]:
        """List MuJoCo body names available as camera/sensor mount points.

        This is the discovery surface for ``add_camera(parent_body=...)``.
        Robot bodies are namespaced ``<robot>/<body>`` (e.g. ``so101/gripper``
        is the canonical wrist-camera mount for SO101/SO100 arms). Without this
        action an agent has to guess the body name, mount a camera against a
        non-existent body, read the failure, and retry -- a wasted turn. Call
        ``list_bodies`` first to resolve the exact mount point deterministically.

        Args:
            robot_name: When set, return only that robot's bodies (its
                namespace prefix). When omitted, return every body in the
                world (the global ``world`` body plus all robots/objects).

        Returns:
            ``{"status", "content": [{"text"}, {"json": {"bodies": [...]}}]}``.
            ``bodies`` is the ordered list of body names; the ``text`` block
            mirrors it for human display. When ``robot_name`` is given the
            json also carries ``"gripper_body"`` -- the best-guess gripper/
            end-effector mount (a body one of whose *name components* is
            ``gripper``, ``hand``, ``ee``, or ``tool``), or ``null`` if none
            matches. Hints match components rather than bare substrings, so a
            short hint cannot fire inside an unrelated word: a ``knee`` or a
            drive ``wheel`` is not an end-effector because ``ee`` occurs in its
            name.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = self._mj
        model = self._world._model

        prefix = ""
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
            prefix = self._world.robots[robot_name].namespace or ""

        bodies = [
            name
            for i in range(model.nbody)
            if (name := mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)) and (not prefix or name.startswith(prefix))
        ]

        json_payload: dict[str, Any] = {"bodies": bodies}
        if robot_name is not None:
            gripper_body: str | None = None
            for name in bodies:
                short = name.rsplit("/", 1)[-1]
                if any(hint_matches_name(hint, short) for hint in _GRIPPER_BODY_HINTS):
                    gripper_body = name
                    break
            json_payload["gripper_body"] = gripper_body

        text = (
            f"Bodies ({len(bodies)}): {bodies}\n"
            "Use any of these as add_camera(parent_body=...) to mount a "
            "wrist/gripper camera."
        )
        if robot_name is not None and json_payload.get("gripper_body"):
            text += f"\nGripper/EEF mount: '{json_payload['gripper_body']}'"

        return {"status": "success", "content": [{"text": text}, {"json": json_payload}]}

    # Object Management

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
        """Add a primitive or mesh object to the live MuJoCo scene.

        The ``size`` argument is the **full extent in meters** along each local
        axis -- *not* MuJoCo's native half-extent. It is halved internally when
        the geom is compiled, so a box created with ``size=[0.05, 0.05, 0.05]``
        is a 5 cm cube (MuJoCo stores ``geom_size == [0.025, 0.025, 0.025]``).
        Per-shape ``size`` semantics (all values are full extents / diameters in
        meters):

        * ``box`` / ``ellipsoid``: ``[x, y, z]`` full edge lengths per axis.
        * ``sphere``: ``size[0]`` is the diameter (``size[1:]`` ignored).
        * ``cylinder`` / ``capsule``: ``size[0]`` diameter, ``size[2]`` full
          height (``size[1]`` ignored).
        * ``plane``: ``size[0]`` / ``size[1]`` are visual half-widths; planes are
          infinite for collision and are forced static.
        * ``mesh``: ``size`` is ignored -- the asset's own units define the
          extent (requires ``mesh_path``). Because no component is consumed, the
          success text reports the compiled extent read back off the geom
          instead of echoing the request.

        A free (non-static) body rests on a horizontal support at
        ``rest_z = support_top + size_z / 2`` -- e.g. a 5 cm cube on a table
        whose top is at ``z = 0.75`` settles with its body origin at
        ``z = 0.775``. Assuming half-extent makes objects look like they "sink
        into" a support when the rest height was simply mis-computed.

        On a ``create_world(terrain=...)`` world the local ground beneath
        ``(x, y)`` is *elevated* (up to ``TERRAIN_ELEVATION * difficulty`` above
        ``z = 0``), so this flat-support formula -- which assumes a support at
        ``z = 0`` -- leaves an object spawned there buried in the heightfield
        (it then sinks through instead of resting on the surface). Query the
        local surface with :meth:`get_ground_height` and add the shape's rest
        offset (e.g. ``size_z / 2`` for a box) to place the object on the
        terrain: ``position=[x, y, get_ground_height(x, y)["content"][1]["json"]["height"] + size_z / 2]``.

        Args:
            name: Unique object name. Its geom is injected as ``"<name>_geom"``.
                Must be a non-empty ``str`` that contains no NUL: an empty name
                is MuJoCo's own sentinel for an unnamed body (so
                ``get_body_state`` could not find the object afterwards), a NUL
                truncates the name the model compiles under while the registry
                keeps the full string, and a non-string name is not addressable
                through the agent-tool surface, where a name arrives as JSON.
            shape: ``"box"``, ``"sphere"``, ``"cylinder"``, ``"capsule"``,
                ``"ellipsoid"``, ``"plane"``, or ``"mesh"``.
            position: World position ``[x, y, z]`` of the body origin (default
                origin).
            orientation: wxyz quaternion (default identity).
            size: Full extents in meters per the per-shape table above. Defaults
                to ``[0.05, 0.05, 0.05]`` (a 5 cm box) when omitted. Must carry
                every component the shape consumes -- 3 for ``box`` /
                ``ellipsoid`` / ``cylinder`` / ``capsule``, at least 1 for
                ``sphere`` / ``plane`` -- and at most 3 in total. A partial
                vector (``size=[0.5]`` on a box) is rejected rather than
                completed from a backend default, because a completed vector
                compiles a differently-sized object while reporting success.
                Every consumed component must be > 0; a non-positive extent is
                rejected.
            color: ``[r, g, b]`` or ``[r, g, b, a]`` in 0..1 (default mid-grey).
                An RGB triple is completed with an opaque alpha -- the one
                component the geom's rgba row defines a default for. Any other
                component count, including an empty vector, is rejected rather
                than completed from the backend default, because a completed
                colour paints a surface the caller never asked for while
                reporting success.
            mass: Body mass in kg for dynamic objects (default 0.1); must be a
                finite number > 0 (the same domain ``set_body_properties``
                enforces). Ignored when ``is_static``.
            is_static: Fix the body in the world. ``None`` (the default)
                means unspecified: ``shape="plane"`` resolves it to True, every
                other shape to False. A plane cannot be dynamic, so an explicit
                ``is_static=False`` there is refused rather than overridden --
                which is why the default is ``None`` and not ``False``.
            mesh_path: Mesh asset path; required and only used when
                ``shape="mesh"``. The asset defines the geom's extent, and
                MuJoCo collides a mesh geom as its **convex hull** -- not as the
                triangles that render. For a convex asset the two coincide; for
                a concave one (a room shell, a tray, a shelf, a bowl) the hull
                fills every cavity, so an object dropped "inside" rests on the
                filled hull and a camera still shows the open interior. To get
                load-bearing concave geometry, decompose the asset into convex
                parts and add one mesh object per part. The success text names
                the hull for this reason.
            material: Optional MuJoCo material for the object's geom, given as
                a mapping of material attribute to value. The accepted keys are
                ``builtin``, ``reflectance``, ``rgb1``, ``rgb2``, ``shininess``,
                ``specular``, ``texdim``, ``texrepeat`` and ``texture``
                (:data:`~strands_robots.simulation.mujoco.spec_builder.MATERIAL_KEYS`
                is the single source of truth). A key outside that vocabulary is
                refused - with a "did you mean" suggestion and the accepted list
                - rather than dropped, and so is ``material={}``, which would
                otherwise report success having compiled the default material.
                ``rgb1`` / ``rgb2`` / ``texdim`` only colour or size a
                procedural texture, so passing one without ``builtin`` is
                refused too: the texture they configure would never be
                generated. ``None`` (default) leaves the geom on the flat
                ``color`` rgba.

        Returns:
            Agent-tool status dict. ``{"status": "success", ...}`` on success;
            ``{"status": "error", ...}`` when no world exists, a policy is
            running, ``name`` is not a usable entity name (see above), the name
            is taken, ``position``/``orientation``/``color``/
            ``size`` contains a non-finite (``nan``/``inf``) or non-numeric
            element or ``position``/``orientation``/``color`` is the wrong length
            (3 / 4 / 3-or-4), ``size`` has a non-positive extent or a component
            count the shape cannot consume, ``mass`` is not a finite number > 0
            for a dynamic object, ``shape="mesh"`` is missing ``mesh_path``, or
            the recompile fails.

        Example:
            >>> sim.add_object("cube", shape="box", size=[0.05, 0.05, 0.05])  # 5 cm cube
            >>> # on a table whose top is at z=0.75, the cube rests at z=0.775

        ``material`` (optional): a visual material/texture spec. When ``None``
        the object renders with the flat ``color`` rgba (unchanged behaviour).
        When set, a real MuJoCo material/texture is attached so the surface can
        be matte (``{"specular": 0, "shininess": 0}``) or carry an image
        texture (``{"texture": "/abs/path.png", "texrepeat": [2, 2]}``) or a
        procedural builtin (``{"builtin": "checker", "rgb1": [...], "rgb2":
        [...]}``). See :meth:`SpecBuilder._build_material` for the full schema;
        an invalid texture path or unknown builtin fails loudly (no silent
        fallback to flat plastic). Only the keys in
        :data:`~strands_robots.simulation.mujoco.spec_builder.MATERIAL_KEYS`
        are accepted -- a misspelled or unsupported key (``"rgb_1"``,
        ``"roughness"``) is rejected rather than dropped, because a dropped key
        renders the default glossy surface while still reporting success.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("add_object"):
            return err

        # Refuse a name that cannot address the object this call creates, before
        # anything is registered under it. Previously ``add_object("")``
        # succeeded and ``get_body_state(body_name="")`` then reported the body
        # absent (MuJoCo reads "" as unnamed), a NUL name registered one string
        # while the body compiled under another, and a non-string name was
        # entered in the registry and only then raised out of MuJoCo's
        # ``add_body`` - leaving the world holding an entry for a body that does
        # not exist, through a raise the tool-result contract does not allow.
        # The duplicate-name test below is itself partial for an unhashable
        # name, so this has to come first.
        if (name_err := entity_name_error("add_object", "name", name)) is not None:
            return {"status": "error", "content": [{"text": name_err}]}

        if name in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' exists."}]}

        # planes are infinite and must be static.  Explicit
        # is_static=False for a plane is an error; None or True both
        # resolve to True. Non-plane shapes default to dynamic.
        if shape == "plane":
            if is_static is False:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "add_object: shape='plane' requires is_static=True (planes are infinite and cannot have dynamic mass)."
                        }
                    ],
                }
            is_static = True
        elif is_static is None:
            is_static = False

        # A mesh geom names an asset that must be loaded from a file; without a
        # path there is nothing to register, so fail fast with an actionable
        # message rather than letting the recompile refuse the unresolved geom.
        if shape == "mesh" and not mesh_path:
            return {
                "status": "error",
                "content": [{"text": "add_object: shape='mesh' requires mesh_path (path to an STL/OBJ asset)."}],
            }

        # Validate every caller-supplied numeric vector is finite BEFORE we bake
        # it into the compiled MJCF. Without this: a nan/inf position or
        # orientation is written verbatim into the object's freejoint qpos and
        # mj_forward then propagates it through the whole physics state
        # (reporting success while silently poisoning the sim); a nan/inf size
        # aborts the recompile with a cryptic "spec recompile refused"; and a
        # non-numeric element (e.g. ["a", ...]) raises a bare TypeError inside
        # MuJoCo's add_geom or the size <= 0 comparison, escaping the
        # structured-error contract. NumPy scalar components are accepted.
        position, e = coerce_pose_vector("add_object", "position", position, 3)
        if e is not None:
            return {"status": "error", "content": [{"text": e}]}
        orientation, e = coerce_pose_vector("add_object", "orientation", orientation, 4)
        if e is not None:
            return {"status": "error", "content": [{"text": e}]}
        if size is not None and (e := finite_vector_error("add_object", "size", size)) is not None:
            return {"status": "error", "content": [{"text": e}]}

        # 'color' targets the geom's 4-component rgba row, so its component
        # count is part of the contract - the same one set_geom_properties
        # enforces on a live geom. Without the count check an RGB triple (or any
        # other partial vector) reached MuJoCo's add_geom and aborted the
        # recompile with "spec recompile refused", hiding the actionable reason,
        # and an empty vector fell through the `color or <default>` coalescing
        # below and painted the default grey under a success result.
        color_rgba: list[float] | None = None
        if color is not None:
            color_rgba, color_err = _coerce_rgba(color, "add_object")
            if color_err is not None:
                return color_err

        # 'size' is the full extent in meters per the docstring; reject a
        # non-positive (degenerate) extent before mutating scene state so the
        # caller gets a clear error rather than a confusing recompile failure.
        if size is not None and (size_err := _validate_size(shape, list(size))) is not None:
            return {"status": "error", "content": [{"text": size_err}]}

        # A dynamic body's mass divides every force acting on it, so a value
        # outside (0, inf) is unusable: inf compiled successfully and made the
        # first step produce nan, which the shared state vector then spread to
        # every OTHER body in the world, while 0/negative/nan aborted the
        # recompile with a generic "spec recompile refused" that named neither
        # the parameter nor MuJoCo's mjMINVAL invariant. Checked only for a
        # dynamic body, which is where the value reaches body_mass - a static
        # body has no mass in the compiled model, and the result says "static"
        # rather than quoting a mass, so nothing is silently dishonored there.
        if not is_static:
            if (mass_err := self._validate_mass(mass, "add_object")) is not None:
                return mass_err
            # MuJoCo additionally refuses to compile a moving body lighter than
            # mjMINVAL ("mass and inertia of moving bodies must be larger than
            # mjMINVAL"). Name that floor here so the common case is rejected by
            # parameter name rather than by a recompile. It is a necessary but
            # not a sufficient condition: the same invariant also covers the
            # INERTIA, which the compiler integrates from the geom's shape, so a
            # mass above this floor can still integrate to an inertia below it
            # on a very small geom. Reproducing that bound here would mean
            # reimplementing MuJoCo's per-shape integration, so the residual
            # case is reported with the compiler's own message instead (see
            # scene_ops.inject_object_into_scene).
            minimum = float(_ensure_mujoco().mjMINVAL)
            if float(mass) < minimum:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"add_object: 'mass' must be >= MuJoCo's mjMINVAL ({minimum:g} kg) "
                                f"for a dynamic body, got {float(mass)!r}; a lighter body cannot be "
                                "integrated. Use is_static=True for an immovable body."
                            )
                        }
                    ],
                }

        # A material key the builder cannot honor (typo / another renderer's
        # field name) would otherwise be dropped and the object would compile
        # with MuJoCo's glossy defaults while this call reported success.
        if material is not None and (mat_err := material_spec_error(name, material)) is not None:
            return {"status": "error", "content": [{"text": mat_err}]}

        obj = SimObject(
            name=name,
            shape=shape,
            # ``is None`` means "omitted" -> the documented default. ``or`` would
            # additionally read a NumPy pose as ambiguous (a bare ValueError) and
            # an empty vector as omitted; coerce_pose_vector already rejected
            # the latter and normalized the former to plain floats.
            position=[0.0, 0.0, 0.0] if position is None else position,
            orientation=[1.0, 0.0, 0.0, 0.0] if orientation is None else orientation,
            # ``size is None`` means "omitted" -> the documented 5 cm default. An
            # explicitly supplied vector is passed through verbatim: ``or`` would
            # read an empty list as omitted and substitute the default for a
            # request _validate_size has already rejected. Elements are floated
            # for the same reason poses are - ``list(np.array([0.05] * 3))``
            # keeps NumPy scalars, which leak as ``np.float64(0.05)`` into this
            # object's status text and ``list_objects``.
            size=[0.05, 0.05, 0.05] if size is None else [float(v) for v in size],
            # ``color is None`` means "omitted" -> the documented mid-grey
            # default. A supplied colour arrives here already coerced to 4
            # components; ``or`` would read an empty list as omitted and paint
            # the default for a request the rgba contract has already rejected.
            color=[0.5, 0.5, 0.5, 1.0] if color_rgba is None else color_rgba,
            mass=mass,
            mesh_path=mesh_path,
            material=material,
            is_static=is_static,
        )
        self._world.objects[name] = obj

        # Every scene mutation goes through spec.recompile() - no branching
        # on robots / scene_loaded, and no XML round-trip. MjSpec preserves
        # existing joint state automatically on recompile.
        try:
            if not inject_object_into_scene(self._world, obj):
                # Injection returned False (compile error). Clean up.
                self._world.objects.pop(name, None)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject '{name}': spec recompile refused."}],
                }
        except (ValueError, RuntimeError) as e:
            self._world.objects.pop(name, None)
            return {
                "status": "error",
                "content": [{"text": f"Failed to inject '{name}' into live scene: {e}"}],
            }

        # A mesh consumes no 'size' component (``_SIZE_LAYOUT["mesh"]`` is 0),
        # so echoing the request back reports an extent this add never applied:
        # the default read as a 5 cm object for an asset of any size, and an
        # explicit vector read as honoured. Report what compiled instead -- the
        # asset's own extent, and the collision geometry, which for every mesh
        # geom is its convex hull rather than the surface that renders. Both are
        # what a caller placing a robot or an object against the asset needs, and
        # neither is derivable from the request. Primitive shapes keep echoing
        # ``size``: there it is the extent, and the geom compiles to it.
        if shape == "mesh":
            extent = _compiled_geom_extent(self._mj, self._world._model, f"{name}_geom")
            geometry = "extent unavailable" if extent is None else f"extent={extent}m from the asset"
            detail = f"{geometry} (collision uses its convex hull)"
        else:
            detail = f"size={obj.size}"

        return {
            "status": "success",
            "content": [
                {
                    "text": f"'{name}' added: {shape} at {obj.position}, {detail}, {'static' if is_static else f'{mass}kg'}"
                }
            ],
        }

    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove a named object from the live scene.

        Deletes the object from the world registry and ejects its body from the
        MjSpec, recompiling the model while preserving the state of the
        remaining bodies.

        A camera mounted on the object (``add_camera(parent_body=name)``) is
        removed with it: its pose is expressed in that body's frame, so the
        recompile drops the camera element and the registry entry goes too.
        ``list_cameras`` therefore never advertises a camera the renderer
        cannot resolve. Each dropped camera is named in a warning, matching
        :meth:`remove_robot`.

        Args:
            name: The object name (as passed to ``add_object``).

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists, ``name`` is unknown, or a policy is running.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if not registered(self._world.objects, name):
            return {"status": "error", "content": [{"text": self._unknown_object_msg(name)}]}
        if err := self._require_no_running_policy("remove_object"):
            return err
        # An active attachment referencing this body would make the ejection
        # recompile fail (dangling weld) or silently drop a kinematic carry.
        # Fail fast so the caller detaches deliberately.
        if (attached_child := self.attachment_involving(name)) is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Object '{name}' is referenced by an active attachment "
                            f"(child '{attached_child}'). Call detach_bodies first."
                        )
                    }
                ],
            }
        del self._world.objects[name]
        # spec-based path: eject_body_from_scene looks up the body in the
        # live MjSpec, deletes it, and recompiles preserving remaining state.
        eject_body_from_scene(self._world, name)
        return {"status": "success", "content": [{"text": f"'{name}' removed."}]}

    def move_object(
        self, name: str, position: list[float] | None = None, orientation: list[float] | None = None
    ) -> dict[str, Any]:
        """Move an existing object to a new pose.

        ``position`` is ``[x, y, z]`` in meters; ``orientation`` is a
        ``[w, x, y, z]`` quaternion. Either may be omitted to leave that
        component unchanged.

        Two paths, transparent to the caller:

        * **Dynamic objects** (``is_static=False``, created with a freejoint)
          are moved cheaply by writing ``data.qpos`` + a forward pass. The
          object is placed **at rest** at the new pose (its freejoint velocity
          is zeroed), consistent with ``add_object`` and ``reset``.
        * **Static objects** (``is_static=True``, welded to the worldbody with
          no DOF) cannot be moved through ``data.qpos``; they are repositioned
          by editing the spec body pose and recompiling the scene (preserving
          other joints' state), just like ``add_object`` / ``remove_object``.

        ``position`` must be a 3-element vector and ``orientation`` a 4-element
        wxyz quaternion; each must contain only finite real numbers. The vector
        itself may be a list, a tuple or a NumPy array (pose arithmetic produces
        arrays), and its elements may be NumPy scalars. A wrong-length,
        non-numeric, or nan/inf pose is rejected up front rather than raising a
        bare ``ValueError`` past the tool-result contract or silently writing a
        non-finite value into ``data.qpos`` (which ``mj_forward`` would propagate
        through the whole physics state). "Supplied" means "not ``None``": an
        empty vector is a wrong-length request, not an omission, so it is
        rejected instead of leaving the component unchanged under a success
        result. The returned text names the components actually applied.

        Returns ``status="error"`` if the object is unknown, the pose is
        invalid, or the static-body recompile fails - it never reports success
        without actually moving the object.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if not registered(self._world.objects, name):
            return {"status": "error", "content": [{"text": self._unknown_object_msg(name)}]}
        # Guard: move_object writes qpos + calls mj_forward, racing a running policy.
        if err := self._require_no_running_policy("move_object"):
            return err

        # Validate the pose BEFORE mutating any state. Both the dynamic path
        # (raw data.qpos write) and the static path (spec reposition) otherwise
        # let a wrong-length / non-numeric vector raise a bare ValueError past
        # the tool-result contract, or write a nan/inf straight into qpos where
        # mj_forward silently poisons the whole physics state. Only validate a
        # component that is actually supplied (None leaves it unchanged; the
        # move logic below treats a falsy value as "no change").
        position, perr = coerce_pose_vector("move_object", "position", position, 3)
        if perr is not None:
            return {"status": "error", "content": [{"text": perr}]}
        orientation, oerr = coerce_pose_vector("move_object", "orientation", orientation, 4)
        if oerr is not None:
            return {"status": "error", "content": [{"text": oerr}]}

        mj = self._mj
        # move_object writes data.qpos/qvel + mj_forward (dynamic path) or
        # recompiles the scene (static path); serialize both against a
        # concurrent mj_step/render under self._lock, matching every sibling
        # mutator (send_action, step, reset, set_gravity). The
        # _require_no_running_policy gate above stops a policy worker; the
        # lock additionally excludes the render/recorder daemon.
        with self._lock:
            model, data = self._world._model, self._world._data
            jnt_id = mj_name_to_id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
            if jnt_id >= 0:
                # Dynamic object: a freejoint carries its pose, so move it cheaply
                # through data.qpos + a forward pass (no recompile).
                qpos_addr = model.jnt_qposadr[jnt_id]
                moved = False
                if position is not None:
                    data.qpos[qpos_addr : qpos_addr + 3] = position
                    self._world.objects[name].position = position
                    moved = True
                if orientation is not None:
                    data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
                    self._world.objects[name].orientation = orientation
                    moved = True
                if moved:
                    # Place the object AT REST at the new pose. A freejoint retains
                    # its 6-DOF linear+angular velocity across a bare data.qpos
                    # write, so without this a repositioned object keeps its prior
                    # momentum: a settling object teleports and immediately shoots
                    # off, and an eval/benchmark loop that repositions objects
                    # between episodes starts each episode with the object drifting
                    # (silently non-reproducible). This matches add_object (spawns
                    # at rest), reset (zeroes velocities), and the Newton backend
                    # (rebuilds from the builder at rest).
                    dof_addr = model.jnt_dofadr[jnt_id]
                    data.qvel[dof_addr : dof_addr + 6] = 0.0
                mj.mj_forward(model, data)
            elif position is not None or orientation is not None:
                # Static object: welded to the worldbody with no freejoint, so it
                # has no data.qpos slice to write - the old code fell through here
                # and returned success while moving nothing (a silent no-op).
                # Reposition it by editing the spec body pose and recompiling
                # (preserving other joints' state), mirroring add_object /
                # remove_object.
                if not reposition_body_in_scene(self._world, name, position=position, orientation=orientation):
                    return {
                        "status": "error",
                        "content": [{"text": f"Failed to reposition '{name}' in the live scene."}],
                    }
                if position is not None:
                    self._world.objects[name].position = position
                if orientation is not None:
                    self._world.objects[name].orientation = orientation

        # Report what was actually applied. ``position or 'same'`` claimed
        # "moved to same" for an orientation-only move (which DID change the
        # pose) and raised on a NumPy position.
        applied = [
            f"{label} {vec}" for label, vec in (("position", position), ("orientation", orientation)) if vec is not None
        ]
        return {
            "status": "success",
            "content": [{"text": f"'{name}' moved to {', '.join(applied) if applied else 'same'}"}],
        }

    def list_objects(self) -> dict[str, Any]:
        """List every object in the scene with its shape, position, and mass.

        Returns:
            A ``{status, content}`` tool result whose text enumerates the
            objects (or reports that there are none). ``status`` is
            ``"error"`` when no world exists.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}

        lines = ["Objects:\n"]
        for name, obj in self._world.objects.items():
            lines.append(f"  - {name}: {obj.shape} at {obj.position}, {'static' if obj.is_static else f'{obj.mass}kg'}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def list_cameras_info(self) -> dict[str, Any]:
        """Agent-facing camera listing (tool-result form of :meth:`list_cameras`).

        Completes the discovery surface alongside ``list_robots`` /
        ``list_objects`` / ``list_bodies`` so an agent can enumerate the
        renderable cameras (what ``render`` / ``start_recording`` can target)
        instead of guessing names or triggering a "camera not found" error.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        cams = self.list_cameras()
        lines = ["Cameras (renderable):\n"] + [f"  - {c}" for c in cams]
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # Camera Management

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
        """Add a camera to the scene (MJCF ``<camera>`` injection).

        Naming: ``add_object(name="X", ...)`` injects its geom as
        ``"X_geom"`` in MJCF, so cameras share the name table only with
        other cameras and body names - not with object geoms. A name this
        engine already registered is rejected upfront; a name the loaded
        scene's MJCF declares is invisible to that check (``load_scene``
        replaces the registry, not the MJCF) and is refused by the compiler
        instead - either way the add is refused and the camera the scene
        already had keeps its own pose.

        Orientation: ``target`` is baked into the camera's ``xyaxes``
        attribute so the rendered view looks at that point (not just
        forward-facing). Degenerate cases (target == position) error.

        Mounting (``parent_body``): when set to a body name (e.g. a robot's
        gripper such as ``"so101/gripper"``), the camera is attached TO that
        body and rides along with it -- this is how a realistic wrist/gripper
        camera is modelled for SO101/SO100-style data collection. In this
        mode ``position`` and ``target`` are in the body's LOCAL frame. Call
        ``list_bodies`` (optionally with ``robot_name``) to discover valid
        mount points before placing a camera; robot bodies are namespaced
        ``<robot>/<body>`` (e.g. ``so101/gripper`` is the SO101 wrist mount).

        Validation: ``name`` must be a non-empty ``str`` containing no NUL, and
        must not be one of the free-camera routing tokens
        (:data:`~strands_robots.utils.FREE_CAMERA_TOKENS` - ``None``, ``""``,
        ``"default"``, ``"free"``). ``render``/``render_depth``/``get_frame``
        resolve every one of them to the free camera by an explicit token check,
        so a camera created under any of them could never be rendered from even
        though it is registered, compiled into the model and listed by
        ``list_cameras``; a non-string name is additionally not addressable
        through the agent-tool surface. This is the same set the Newton backend's
        ``add_camera`` refuses, on the shared
        :func:`~strands_robots.utils.reserved_camera_name_error` domain, because
        that backend routes the same tokens. The Isaac backend does not route
        them - its ``get_frame`` looks the name up directly - so ``"default"``
        there is an ordinary camera name and stays accepted. ``position`` and ``target``
        must each be 3 finite numbers (a list, tuple or NumPy array; NumPy
        scalar elements accepted). Omit a vector to take its default - an empty
        vector is a wrong-length request and is rejected rather than silently
        placing the camera at the default pose. ``fov`` must be a finite angle in ``(0, 180)``
        degrees; and ``width``/``height`` must be positive ints within the
        offscreen framebuffer cap (same bounds ``render`` enforces). Invalid
        values are rejected here at config time with an actionable error rather
        than deferring an uncaught ``TypeError`` (non-numeric), a silently
        degenerate camera (``nan``/``inf`` baked into ``xyaxes``), or a cryptic
        spec-recompile / GL failure to the first render.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("add_camera"):
            return err

        # Refuse a name that cannot address the camera this call creates, on the
        # shared ``entity_name_error`` domain. An empty name is worse here than
        # for an object: ``render``/``get_frame`` route ``camera_name in (None,
        # "", "default", "free")`` to the FREE camera by an explicit token
        # check, so a camera registered as "" could never be rendered from. It
        # precedes the duplicate-name test for the same reason it does in
        # ``add_object`` - that test is partial for an unhashable name.
        if (name_err := entity_name_error("add_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": name_err}]}

        # Refuse a name this backend's own render entry points resolve past. The
        # three of them (``render`` / ``render_depth`` / ``get_frame``) select the
        # free camera for every ``FREE_CAMERA_TOKENS`` member by an explicit token
        # check, so claiming one produced a camera that is registered, compiled
        # into the model and offered by ``list_cameras`` - and that every render
        # of silently answers with the free view instead, under a success result.
        # ``entity_name_error`` above covers only the two falsy tokens, which is
        # why this is a second guard rather than a widening of that domain: the
        # other two are perfectly addressable *names* that this backend alone
        # cannot address as *cameras*.
        #
        # It precedes the duplicate-name test because that test answered for
        # ``"default"`` and answered misleadingly: ``create_world`` registers the
        # built-in free view under that name, so the refusal was "already exists.
        # Remove it first." - and following that prescription succeeded, leaving
        # the scene with an unreachable camera where the advertised free-view
        # alias had been.
        if (reserved_err := reserved_camera_name_error("add_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": reserved_err}]}

        # Validate position / target shape before we bake them into XML.
        # Membership, not truthiness: ``position or <default>`` raised a bare
        # ValueError on a NumPy pose (what the docstring above advertises as
        # accepted) and read an empty vector as "omitted", quietly placing the
        # camera at the default [1, 1, 1] under a success result.
        position, _perr = coerce_pose_vector("add_camera", "position", position, 3)
        if _perr is not None:
            return {"status": "error", "content": [{"text": _perr}]}
        target, _terr = coerce_pose_vector("add_camera", "target", target, 3)
        if _terr is not None:
            return {"status": "error", "content": [{"text": _terr}]}
        # ``coerce_pose_vector`` above owns the pose rule for both parameters:
        # anything reaching here is either a validated 3-vector of plain floats or
        # ``None``, so the defaults below - and the element-wise comparison after
        # them - operate on numbers that are already known to be finite.
        pos = [1.0, 1.0, 1.0] if position is None else position
        tgt = [0.0, 0.0, 0.0] if target is None else target
        # Degenerate orientation: position == target means no well-defined look direction.
        if all(abs(pos[i] - tgt[i]) < 1e-9 for i in range(3)):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"add_camera: 'position' and 'target' are identical ({pos}); camera has no look direction."
                    }
                ],
            }

        # Validate the field of view up front. MuJoCo's ``fovy`` must be a
        # finite angle in the open interval (0, 180) degrees; otherwise the
        # spec recompile aborts deep inside ``inject_camera_into_scene`` with a
        # cryptic "spec recompile refused", or - for fov <= 0 - silently
        # registers a degenerate camera that renders nothing useful. The domain
        # lives in the shared ``camera_fov_error`` so the Newton backend's
        # ``add_camera`` cannot drift from this one.
        if (e := camera_fov_error("add_camera", "fov", fov)) is not None:
            return {"status": "error", "content": [{"text": e}]}

        # Validate the render resolution baked into this camera the same way
        # ``render`` validates its dims, so a bad size fails at config time with
        # a clear message instead of deferring a cryptic GL/Renderer error (or a
        # silent non-positive dimension) to the first rollout that renders it.
        if dim_err := self._validate_render_dims(width, height):
            text = dim_err["content"][0]["text"].replace("render:", "add_camera:", 1)
            return {"status": "error", "content": [{"text": text}]}

        # reject duplicate camera names.  Previously a second
        # add_camera(name=existing) silently overwrote the registry entry but
        # left the XML's <camera> unchanged, so the old pose stuck around for
        # rendering.  Explicit error avoids the surprise.
        if name in self._world.cameras:
            return {
                "status": "error",
                "content": [{"text": f"add_camera: camera '{name}' already exists. Remove it first."}],
            }

        # Validate the mount target up front so the user gets a clear,
        # actionable error (the names of available bodies) instead of the
        # generic "spec recompile refused" that inject_camera_into_scene
        # surfaces when SpecBuilder.add_camera raises deep inside the recompile.
        if parent_body:
            mj = self._mj
            model = self._world._model
            if mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, parent_body) < 0:
                available = [
                    mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                    for i in range(model.nbody)
                    if mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i)
                ]
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"add_camera: parent_body '{parent_body}' not found. "
                                f"Robot bodies are namespaced '<robot>/<body>'. "
                                f"Call list_bodies to discover mount points. Available bodies: {available}"
                            )
                        }
                    ],
                }

        cam = SimCamera(
            name=name,
            position=pos,
            target=tgt,
            fov=fov,
            width=width,
            height=height,
            parent_body=parent_body or "",
        )
        self._world.cameras[name] = cam

        # Spec-based path: inject_camera_into_scene adds the camera to the
        # live spec and recompiles preserving state.
        try:
            if not inject_camera_into_scene(self._world, cam):
                self._world.cameras.pop(name, None)
                return {
                    "status": "error",
                    "content": [{"text": f"Failed to inject camera '{name}': spec recompile refused."}],
                }
        except (ValueError, RuntimeError) as e:
            self._world.cameras.pop(name, None)
            return {
                "status": "error",
                "content": [{"text": f"Failed to inject camera '{name}' into live scene: {e}"}],
            }

        mount_note = f" (mounted on '{parent_body}')" if parent_body else ""
        return {"status": "success", "content": [{"text": f"Camera '{name}' added at {cam.position}{mount_note}"}]}

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a named camera from the live scene.

        Deletes the camera element from the MjSpec and recompiles so ``ncam`` in
        the live model matches, then drops the Python-side registry entry.

        The registry entry goes only once the recompile has been accepted. A
        refused ``spec.recompile`` leaves the live model untouched, so dropping
        the entry first would report a removal the model had not applied:
        ``list_cameras`` would stop naming a camera ``render`` and
        ``get_camera_params`` still resolve, and the delete would land later,
        applied by whichever unrelated mutation next recompiles successfully.
        This mirrors :meth:`add_camera`, which rolls a refused add back out and
        reports it rather than claiming the camera exists.

        Args:
            name: The camera name (as passed to ``add_camera``).

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists, ``name`` is a free-camera routing token, ``name`` is
            unknown, a policy is running, or the scene would not recompile
            without the camera -- in that last case the camera is still
            registered and the scene is unchanged.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # Same rule as ``add_camera``, at the other end of the name's life, and for
        # the same reason: a routing token cannot be un-addressed. ``render`` /
        # ``render_depth`` / ``get_frame`` / ``get_camera_params`` keep resolving
        # the token past the registry, and ``list_cameras`` names it
        # unconditionally, so dropping the entry removes no camera - it removes
        # only the recordable/observable alias, leaving every other surface still
        # advertising it. It precedes the existence test because that test can
        # answer for ``"default"`` (``create_world`` registers the free view under
        # that name) and answering it succeeds.
        if (reserved_err := reserved_camera_name_error("remove_camera", "name", name)) is not None:
            return {"status": "error", "content": [{"text": reserved_err}]}
        if not registered(self._world.cameras, name):
            return {"status": "error", "content": [{"text": self._unknown_camera_msg(name)}]}
        if err := self._require_no_running_policy("remove_camera"):
            return err
        cam = self._world.cameras[name]

        if self._world._backend_state.get("spec") is not None:
            # Use the namespaced MuJoCo name if we have it (camera came from
            # a robot's URDF), else the short name.
            if not eject_camera_from_scene(self._world, cam.name or name):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Camera '{name}' was not removed: the scene would not "
                                "recompile without it. The camera is still registered and "
                                "the scene is unchanged; the compiler's reason is logged."
                            )
                        }
                    ],
                }

        del self._world.cameras[name]
        return {"status": "success", "content": [{"text": f"Camera '{name}' removed."}]}

    # Simulation Control

    # Ceiling on the total steps one call may request. Distinct from the
    # inherited ``SimEngine._STEPS_PER_BATCH`` granularity, which bounds how
    # long the lock is held: this bounds how much work is accepted at all, and
    # its value is MuJoCo's own because a per-step cost is. Isaac and Newton
    # have no equivalent - see #1871 and the note on ``_STEPS_PER_BATCH``.
    _MAX_STEPS_PER_CALL = 100_000

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance the simulation by ``n_steps`` physics steps.

        Args:
            n_steps: Non-negative whole step count (``0`` is an accepted no-op),
                on the shared
                :func:`~strands_robots.utils.non_negative_whole_number_error`
                domain every backend applies. A NumPy or float count with an
                integral value is honored and coerced; a fractional, negative,
                non-finite, boolean or non-numeric count is refused.

        Returns:
            A ``{status, content}`` tool result reporting the elapsed sim time
            and total step count. ``status`` is ``"error"`` when no world
            exists, ``n_steps`` is outside that domain, the count exceeds
            :attr:`_MAX_STEPS_PER_CALL`, or the world was destroyed on a batch
            boundary mid-run - in which case the error names the steps
            completed, since some were.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # Refuse before coercing: ``int()`` alone truncates a fractional count
        # to a different number of steps under a success result, reads a
        # boolean as one step, and raises ``OverflowError`` on ``inf`` - which
        # the previous ``except (TypeError, ValueError)`` did not catch, so an
        # infinite count escaped this method's structured envelope entirely.
        if error := non_negative_whole_number_error(n_steps, "n_steps", "step"):
            return {"status": "error", "content": [{"text": error}]}
        n_steps = int(n_steps)
        if n_steps == 0:
            return {
                "status": "success",
                "content": [
                    {"text": f"+0 steps (no-op) | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}
                ],
            }
        if n_steps > self._MAX_STEPS_PER_CALL:
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"step: n_steps={n_steps} exceeds max {self._MAX_STEPS_PER_CALL}. Break into smaller calls."
                    }
                ],
            }
        mj = self._mj
        # Kinematic attachments (attach_bodies mode="kinematic") follow their
        # parent every physics step. Resolved once here; the per-step call is
        # a fast no-op when the registry is empty.
        has_kinematic_attachments = bool(self._world._backend_state.get("kinematic_attachments"))
        # Process in batches, releasing lock between batches so stop_policy
        # and other actions can interleave on long runs.
        remaining = n_steps
        while remaining > 0:
            batch = min(remaining, self._STEPS_PER_BATCH)
            with self._lock:
                # Re-checked per batch, not once before the loop: releasing the
                # lock is exactly what lets ``cleanup`` win its bounded
                # world-handoff acquire (GH #116) in the gap between two
                # batches, after which ``self._world._model`` raised
                # ``AttributeError`` past this method's structured envelope
                # having already advanced part of the count. Same pairing as
                # ``_primitive_abort_reason``, whose loops release the lock on
                # this schedule for the same reason.
                if self._world is None or self._world._model is None or self._world._data is None:
                    return {
                        "status": "error",
                        "content": [{"text": step_aborted_msg(n_steps - remaining, n_steps)}],
                    }
                for _ in range(batch):
                    mj.mj_step(self._world._model, self._world._data)
                    if has_kinematic_attachments:
                        self._apply_kinematic_attachments()
                if has_kinematic_attachments:
                    # Re-forward so the carried bodies' derived state (xpos,
                    # cam xforms) reflects the final teleport for the next
                    # render/observation.
                    mj.mj_forward(self._world._model, self._world._data)
                self._world.sim_time = self._world._data.time
                self._world.step_count += batch
            remaining -= batch
        self._publish_ros_telemetry()
        return {
            "status": "success",
            "content": [{"text": f"+{n_steps} steps | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}],
        }

    def reset(self) -> dict[str, Any]:
        """Reset the world to its initial state, beginning a new rollout.

        Restores the initial pose -- together with the actuator command that
        holds it, for a robot spawned with ``add_robot(keyframe=...)`` -- and
        re-forwards derived state so the world is
        render- and observation-ready on return. If a recording is active with
        buffered frames, flushes them as their own episode first, so a
        ``run_policy`` + ``reset`` collection loop records one episode per
        rollout instead of merging every rollout into episode 0.

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists or a policy is running (a reset during a live
            ``mj_step`` can segfault).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # reset during a running policy races mj_step -> SEGFAULT risk
        if err := self._require_no_running_policy("reset"):
            return err

        # A reset re-initializes the world, which is the start of a new
        # rollout. If a recording is active with buffered (unsaved) frames,
        # flush them as their own episode BEFORE the teleport mixes the next
        # rollout into the same buffer. Without this, a run_policy + reset
        # collection loop silently merges every rollout into a single
        # episode_index=0 (total_episodes stuck at 1) - a data-integrity bug
        # for downstream training/eval that slices by episode. This mirrors
        # stop_recording, which already auto-flushes the trailing episode.
        # save_episode() is a no-op on an empty buffer, so resets that are not
        # preceded by recorded frames (e.g. eval_policy's internal per-episode
        # resets, which do not feed the recorder) are unaffected. To DISCARD a
        # partial rollout instead of flushing it, call clear_episode_buffer()
        # before reset().
        flush_note = ""
        if self._world._backend_state.get("recording", False):
            recorder = self._world._backend_state.get("dataset_recorder")
            pending = getattr(recorder, "episode_frame_count", 0) if recorder is not None else 0
            if pending > 0:
                save_result = self.save_episode()
                if save_result.get("status") != "success":
                    # save_episode failed -> the recorder poisoned itself and
                    # the facade already cleared the recording flag. Surface
                    # the failure rather than resetting into an undefined state.
                    return save_result
                flush_note = save_result["content"][0]["text"] + " "

        mj = self._mj
        with self._lock:
            mj.mj_resetData(self._world._model, self._world._data)
            # mj_resetData restores qpos/qvel/ctrl but zeroes ALL derived
            # kinematics (xpos, site_xpos, geom_xpos, cam_xpos, ...) until the
            # next mj_step/mj_forward. A consumer that reads state or RENDERS
            # before the next step - e.g. eval_policy's per-episode loop, which
            # calls get_observation() immediately after reset() and before the
            # first send_action - would otherwise see every body collapsed at
            # the origin: a degenerate camera frame and zeroed Cartesian state
            # feeding the policy's FIRST inference of every episode. Forward
            # here so reset() leaves a fully consistent, render-ready state,
            # matching the mj_resetData -> mj_forward idiom used by
            # _compile_world and load_scene.
            #
            # Re-apply any per-robot keyframe home state captured at add_robot
            # time (mj_resetData alone drops it back to the zero configuration),
            # so a keyframe spawn is sticky across resets -- mirroring how a
            # benchmark restores its canonical start pose each episode. That
            # covers the actuator command the keyframe pairs with the pose as
            # well: without it every episode begins with the servos commanded to
            # zero, so the arm is already sagging off home as the policy takes
            # its first inference of the episode.
            self._restore_home_state()
            self._seat_floating_bases_on_terrain()
            mj.mj_forward(self._world._model, self._world._data)
            self._world.sim_time = 0.0
            self._world.step_count = 0
            # Flip policy_running flag inside the lock so a racing worker
            # thread cannot slip in one more mj_step between reset and flag
            # flip.
            for r in self._world.robots.values():
                r.policy_running = False
                r.policy_steps = 0
        return {"status": "success", "content": [{"text": f"{flush_note}Reset to initial state."}]}

    def get_state(self) -> dict[str, Any]:
        """Summarize the current simulation state.

        Reports sim time, step count, timestep, gravity, and counts of robots,
        objects, cameras, bodies, joints, and actuators (plus recording status
        when a recording is active).

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        lines = [
            "Simulation State",
            f"t={self._world.sim_time:.4f}s (step {self._world.step_count})",
            f"dt={self._world.timestep}s | g={self._world.gravity}",
            f"Robots: {len(self._world.robots)} | Objects: {len(self._world.objects)} | Cameras: {len(self._world.cameras)}",
        ]
        if self._world._model:
            lines.append(
                f"Bodies: {self._world._model.nbody} | Joints: {self._world._model.njnt} | Actuators: {self._world._model.nu}"
            )
        if self._world._backend_state.get("recording", False):
            lines.append(f"[recording] {len(self._world._backend_state['trajectory'])} steps")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def destroy(self) -> dict[str, Any]:
        """Destroy the world and release all resources.

        Delegates to cleanup() which properly joins running policy Futures
        before nulling self._world - prevents SIGSEGV from workers holding
        stale model/data pointers.
        """
        if self._world is None:
            return {"status": "success", "content": [{"text": "No world to destroy."}]}
        # Sensor-noise config is engine-level; clear it on destroy so a
        # recreated world starts noise-free (parity with the Newton backend).
        self._obs_noise = None
        self._obs_noise_rng = None
        self.cleanup()
        return {"status": "success", "content": [{"text": "World destroyed."}]}

    def _close_main_thread_renderers(self) -> None:
        """Close any renderers this thread owns and drop the TLS cache.

        Only safe for the main thread because ``mujoco.Renderer`` binds a
        CGL/GLX context to the thread that created it; closing from another
        thread can SIGSEGV in ``cgl.free()``. Worker threads drop their
        renderers via ``threading.Thread`` teardown.
        """
        tls = getattr(self, "_renderer_tls", None)
        if tls is None:
            return
        renderers = getattr(tls, "renderers", None)
        if renderers:
            for r in list(renderers.values()):
                try:
                    r.close()
                except Exception:
                    pass
            renderers.clear()
        # Forget the model marker so the next _get_renderer() rebuilds fresh.
        if hasattr(tls, "model"):
            tls.model = None

    def set_gravity(self, gravity: list[float] | float | int) -> dict[str, Any]:
        """Set the world gravity vector.

        Accepts either a 3-element ``[x, y, z]`` list or a bare real scalar,
        which is interpreted as the z-component (``[0, 0, z]``). The scalar
        may be any :class:`numbers.Real` -- a Python ``int`` / ``float`` or a
        NumPy scalar such as ``np.float32`` / ``np.int64`` (e.g. a value read
        from a config array or produced by ``np.degrees(...)``). A NumPy
        array is treated as the vector form, not a scalar.

        The value is recorded in the scene spec as well as ``model.opt``, which is
        compiled from it: without that, the next scene recompile - triggered by any
        ``add_object`` / ``add_camera`` / ``add_robot`` call - would silently restore
        the value the scene was created with.

        Args:
            gravity: Gravity as ``[x, y, z]`` (m/s^2) or a real scalar z-component.

        Returns:
            Agent-tool ``status`` / ``content`` dict. Errors (structured, never
            raised) on a missing world, a running policy, a non-3-element or
            non-numeric vector, or non-finite components.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # set_gravity during a running policy races the worker thread
        if err := self._require_no_running_policy("set_gravity"):
            return err
        # Shape/dtype/finiteness live in the shared normalizer so create_world
        # accepts exactly what this setter accepts (a 3-element vector, or a
        # real scalar - including a NumPy scalar - as the z-component).
        components, gravity_error = self._normalize_gravity(gravity, "set_gravity")
        if components is None:
            return cast("dict[str, Any]", gravity_error)
        with self._lock:
            # model.opt is compiled from spec.option, so a gravity written only
            # into the model is restored to the scene's declared value by the
            # next recompile. Record it in the spec first, so a scene that
            # cannot carry the change is refused before anything is touched.
            if reason := persist_world_option(self._world, gravity=components):
                return {"status": "error", "content": [{"text": f"set_gravity: {reason}"}]}
            self._world._model.opt.gravity[:] = components
            self._world.gravity = components
        return {"status": "success", "content": [{"text": f"Gravity: {components}"}]}

    def set_timestep(self, timestep: float) -> dict[str, Any]:
        """Set the physics integration timestep in seconds.

        The value is recorded in the scene spec as well as ``model.opt``, which is
        compiled from it: without that, the next scene recompile - triggered by any
        ``add_object`` / ``add_camera`` / ``add_robot`` call - would silently restore
        the value the scene was created with.

        Args:
            timestep: A finite positive float.

        Returns:
            A ``{status, content}`` tool result. ``status`` is ``"error"`` when
            no world exists, a policy is running, or ``timestep`` is
            non-positive or non-finite.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_timestep"):
            return err
        # Shared with create_world so a world cannot be created with a dt this
        # setter would refuse; warn (not reject) on huge-but-usable values.
        if err := self._validate_timestep(timestep, "set_timestep"):
            return err
        timestep = float(timestep)
        warn = ""
        if timestep > 0.1:
            warn = f" Warning: unusually large timestep (>{0.1}s); physics may be unstable"
        with self._lock:
            # Same as set_gravity: model.opt is derived from spec.option, so the
            # step has to be recorded there to survive the next recompile.
            if reason := persist_world_option(self._world, timestep=timestep):
                return {"status": "error", "content": [{"text": f"set_timestep: {reason}"}]}
            self._world._model.opt.timestep = timestep
            self._world.timestep = timestep
        return {"status": "success", "content": [{"text": f"Timestep: {timestep}s ({1 / timestep:.0f}Hz){warn}"}]}

    # Viewer

    def open_viewer(self) -> dict[str, Any]:
        """Launch a passive interactive MuJoCo viewer window bound to the running model.

        Opens ``mujoco.viewer.launch_passive`` on the live model/data so a human
        can watch a rollout, debug a pose, or hand-verify a scene. The viewer is
        an interactive OpenGL window and therefore **requires a local display**:
        on a headless host it fails with a viewer error, where
        :meth:`render` / :meth:`render_all` capture frames instead. Idempotent --
        succeeds with "Viewer already open" if one is already up. The inverse is
        :meth:`close_viewer`.

        Returns:
            Agent-tool ``status``/``content`` dict.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "No simulation to view."}]}
        from strands_robots.simulation.mujoco.backend import _mujoco_viewer

        if _mujoco_viewer is None:
            return {"status": "error", "content": [{"text": "mujoco.viewer not available."}]}
        if self._viewer_handle is not None:
            return {"status": "success", "content": [{"text": "Viewer already open."}]}
        try:
            self._viewer_handle = _mujoco_viewer.launch_passive(self._world._model, self._world._data)
            return {"status": "success", "content": [{"text": "Interactive viewer opened."}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Viewer failed: {e}"}]}

    def _close_viewer(self) -> None:
        if self._viewer_handle is not None:
            try:
                self._viewer_handle.close()
            except Exception:
                pass
            self._viewer_handle = None

    def close_viewer(self) -> dict[str, Any]:
        """Close the interactive viewer opened by :meth:`open_viewer`.

        Idempotent -- succeeds even when no viewer is open. The inverse of
        :meth:`open_viewer`.

        Returns:
            Agent-tool ``status``/``content`` dict.
        """
        self._close_viewer()
        return {"status": "success", "content": [{"text": "Viewer closed."}]}

    # URDF Registry

    def list_urdfs(self) -> dict[str, Any]:
        """List every robot/URDF known to the registry (built-in + user-registered).

        This is the discovery entry point an agent uses to learn what it can
        spawn without guessing a model name; ``register_urdf`` adds a new one.

        Read the ``Sim`` column: the registry also holds hardware-only entries -
        robots strands drives over LeRobot that have no simulation model - and
        those names are listed with ``Sim`` blank. ``add_robot`` and
        ``load_scene`` accept the names marked ``Sim``; a hardware-only name is
        refused with the hardware entry point instead. ``list_robots(mode="sim")``
        returns just the spawnable subset.
        """
        return {"status": "success", "content": [{"text": list_available_models()}]}

    def register_urdf(self, data_config: str, urdf_path: str) -> dict[str, Any]:
        """validate urdf_path before handing it to the registry.

        The router already rejects missing required params, so the
        no-args case produces a friendly 'requires parameter ...' message
        without hitting this body.
        """
        if not urdf_path:
            return {
                "status": "error",
                "content": [{"text": "register_urdf: 'urdf_path' must be a non-empty string."}],
            }
        p = Path(urdf_path)
        if not p.exists():
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: file not found: {urdf_path}"}],
            }
        if not p.is_file():
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: not a file: {urdf_path}"}],
            }
        try:
            # Smoke-check readability - mj.MjModel.from_xml_path will surface a
            # better error later, but permission issues are worth catching now.
            with p.open("rb"):
                pass
        except OSError as e:
            return {
                "status": "error",
                "content": [{"text": f"register_urdf: cannot read {urdf_path}: {e}"}],
            }

        _register_urdf(data_config, urdf_path)
        resolved = resolve_model(data_config)
        return {
            "status": "success",
            "content": [{"text": f"Registered '{data_config}' -> {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"}],
        }

    # Introspection

    def get_features(self, robot_name: str | None = None) -> dict[str, Any]:
        """Describe the simulation's joints / actuators / cameras / robots.

        If ``robot_name`` is given, the joint / actuator / camera listings
        are restricted to that robot (its namespaced MuJoCo names).  The
        ``robots`` map is also filtered to just that entry.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = self._mj
        model = self._world._model

        # All-model name pools
        all_joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        all_joint_names = [n for n in all_joint_names if n]
        all_actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        all_actuator_names = [n for n in all_actuator_names if n]
        all_camera_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        all_camera_names = [n for n in all_camera_names if n]

        if robot_name is not None:
            if not registered(self._world.robots, robot_name):
                return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
            robot = self._world.robots[robot_name]
            ns = (getattr(robot, "namespace", "") or "").rstrip("/")
            prefix = f"{ns}/" if ns else ""

            def _scoped(pool: list[str]) -> list[str]:
                if not prefix:
                    # Single-robot scene with no namespace: return the robot's own
                    # joints/actuators from the robot model rather than the pool.
                    return pool
                return [n for n in pool if n.startswith(prefix)]

            joint_names = robot.joint_names or _scoped(all_joint_names)
            actuator_names = _scoped(all_actuator_names)
            camera_names = _scoped(all_camera_names)

            robots_info = {
                robot_name: {
                    "joint_names": robot.joint_names,
                    "n_joints": len(robot.joint_names),
                    "n_actuators": len(robot.actuator_ids),
                    "data_config": robot.data_config,
                    "source": os.path.basename(robot.urdf_path),
                }
            }
        else:
            joint_names = all_joint_names
            actuator_names = all_actuator_names
            camera_names = all_camera_names

            robots_info = {}
            for rname, robot in self._world.robots.items():
                robots_info[rname] = {
                    "joint_names": robot.joint_names,
                    "n_joints": len(robot.joint_names),
                    "n_actuators": len(robot.actuator_ids),
                    "data_config": robot.data_config,
                    "source": os.path.basename(robot.urdf_path),
                }

        features = {
            "n_bodies": model.nbody,
            "n_joints": model.njnt,
            "n_actuators": model.nu,
            "n_cameras": model.ncam,
            "timestep": model.opt.timestep,
            "joint_names": joint_names,
            "actuator_names": actuator_names,
            "camera_names": camera_names,
            "robots": robots_info,
        }

        lines = [
            "Simulation Features",
            f"Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
            f"Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
            f"Timestep: {model.opt.timestep}s ({1 / model.opt.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(f"{rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}],
        }

    # AgentTool Interface

    def _unpublished_action_error(self, action: str) -> dict[str, Any] | None:
        """Refuse an action the tool schema does not advertise to a model.

        The router (:meth:`_dispatch_action`) resolves by ``getattr`` with no
        allowlist, so every public method stays callable from Python. That
        breadth is deliberate, but it is a *Python* contract: a model is handed
        the ``action`` enum and nothing else, so an action outside it is one
        the model was never told exists, and dispatching it makes the tool's
        behaviour wider than the contract it publishes (#2093).

        Two verdicts, because a caller needs different things from them. A name
        that resolves to no method is a typo. A name that resolves to a method
        held back from the enum is a curation decision, and reporting that as
        "unknown" sends a reader hunting for a misspelling that is not there -
        the method is right where they left it.

        Args:
            action: The action name as the entry point received it.

        Returns:
            An error dict when the action is not published, else ``None`` to
            let dispatch proceed. A non-string is not this guard's verdict to
            give: the entry points already answer it, and ``__call__`` answers
            it before reaching here.
        """
        if not isinstance(action, str) or action in _PUBLISHED_ACTIONS or action in self._ACTION_ALIASES:
            return None

        # Probe the class, never the instance: the type carries properties as
        # well as methods, and evaluating one to word a refusal would run
        # engine code on the error path.
        target = self._ACTION_ALIASES.get(action, action)
        if not action.startswith("_") and callable(getattr(type(self), target, None)):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Action '{action}' is not available to an agent. It exists on this "
                            "simulation but is deliberately not published in tool_spec, so it is "
                            "reachable from Python only. See tool_spec for the actions you can use."
                        )
                    }
                ],
            }
        return {"status": "error", "content": [{"text": self._unknown_action_msg(action)}]}

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        """Dispatch an action directly: ``sim(action="render", camera_name="topdown")``.

        This makes the Simulation usable as a plain callable in addition to
        being a Strands ``AgentTool``. It mirrors the agent-facing dispatch
        path: the same validation, field-aliasing, and per-action method
        routing apply, and the return value is the standard ``{"status",
        "content"}`` dict.

        Mirroring the *agent* path is what this form is for, so it is narrower
        than ``_dispatch_action`` in exactly one way: an action the tool schema
        does not publish is refused here rather than routed (#2093). Reaching a
        deliberately Python-only capability is what its own method is for -
        ``sim.get_observation(...)`` rather than ``sim(action=...)``.

        The README markets ``Robot("so100")`` as something you can drive with
        ``robot(action="...")``; without this method that contract raised
        ``TypeError: 'MuJoCoSimEngine' object is not callable``. Keyword
        arguments are forwarded as the action's parameters.

        Args:
            action: The action name (e.g. ``"create_world"``, ``"add_robot"``,
                ``"render"``). Required; an empty/blank action returns an
                error dict rather than raising, to match the tool contract.
            **kwargs: Parameters for the action (e.g. ``camera_name="top"``).

        Returns:
            ``{"status": "success"|"error", "content": [...]}``.
        """
        if not action or not isinstance(action, str) or not action.strip():
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "Calling a Simulation requires action=. "
                            "Example: sim(action='render', camera_name='topdown'). "
                            "See tool_spec for the full action list."
                        )
                    }
                ],
            }
        requested = action.strip()
        refusal = self._unpublished_action_error(requested)
        if refusal is not None:
            return refusal
        return self._dispatch_action(requested, kwargs)

    @property
    def tool_name(self) -> str:
        """The tool name the agent invokes this simulation under.

        Defaults to ``"mujoco_simulation"`` but is settable via the
        ``tool_name`` constructor argument so several engines can coexist in
        one agent's tool registry under distinct names.
        """
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        """The Strands ``AgentTool`` category for this tool (always ``"simulation"``)."""
        return "simulation"

    def _require_world(self) -> dict[str, Any] | None:
        """Return the unified 'no world' error, or None if the world is live.

        The "world is live" predicate is ``self._world`` plus a compiled
        ``_model`` and allocated ``_data`` - the partial state ``load_scene``
        leaves behind on a compile failure (``_world`` set, model/data still
        ``None``) counts as *not* live. Methods that touch model/data inline the
        same condition for mypy narrowing, but every guard - inline or via this
        helper - returns the single :data:`_NO_WORLD_MSG` string so the contract
        reads identically across the facade.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        return None

    def _prune_done_futures(self) -> None:
        """Drop completed Future refs from self._policy_threads.

        Without this, list_policies_running and stale-active checks see
        historical entries forever (see GH #120).
        """
        done = [k for k, f in self._policy_threads.items() if f.done()]
        for k in done:
            self._policy_threads.pop(k, None)
        # The capture-rate table is keyed the same way and is only meaningful
        # while the rollout's Future is tracked, so it follows _policy_threads
        # here rather than at each of that dict's own removal sites
        # (remove_robot deletes an entry, cleanup clears them all).
        for stale in [k for k in self._policy_rates if not registered(self._policy_threads, k)]:
            self._policy_rates.pop(stale, None)

    def _active_policy_robots(self) -> list[str]:
        """Names of robots with a live (not-done) policy Future.

        Prunes stale entries as a side-effect so the returned list is
        authoritative. Callers can introspect via ``list_policies_running``.
        """
        self._prune_done_futures()
        return list(self._policy_threads.keys())

    def _active_rollout_rates(self) -> dict[str, float]:
        """Capture rate of every ``start_policy`` rollout still in flight.

        Overrides :meth:`SimEngine._active_rollout_rates` so
        ``start_recording`` can refuse a dataset rate a running rollout is not
        capturing at. Prunes stale entries first (via
        :meth:`_prune_done_futures`, which also sweeps the rate table against
        ``_policy_threads``) so a finished rollout cannot block a recording.
        """
        self._prune_done_futures()
        return dict(self._policy_rates)

    def _require_no_running_policy(self, action_name: str, robot_name: str | None = None) -> dict[str, Any] | None:
        """Return an error dict if a disallowed policy is running, else None.

        Two scopes (GH #114):

        * ``robot_name=None`` (default) - **global scope**. Used by scene
          mutations that touch the whole XML / model pointer (``add_robot``,
          ``remove_robot``, ``add_object``, ``remove_object``, ``move_object``,
          ``add_camera``, ``remove_camera``, ``load_scene``, ``set_gravity``,
          ``set_timestep``). An XML round-trip swaps ``self._world._model``
          and ``self._world._data``; any live PolicyRunner worker holding
          pointers to the old arrays will segfault when it next calls
          ``mj_step``. Hard-fail.

        * ``robot_name="..."`` - **per-robot scope**. Used by actions that
          are safe to run while *other* robots' policies are active
          (start_policy on the same robot, stop_policy, etc.). Policies on
          different robots can execute concurrently because MuJoCo physics
          is serialized by ``self._lock`` and each robot writes to a
          disjoint slice of ``data.ctrl[]``.
        """
        self._prune_done_futures()
        if robot_name is not None:
            fut = registry_entry(self._policy_threads, robot_name)
            if fut is not None and not fut.done():
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Cannot '{action_name}' on '{robot_name}' while its policy is running. "
                                f"Stop it first: action='stop_policy', name='{robot_name}'."
                            )
                        }
                    ],
                }
            return None

        active = [name for name, f in self._policy_threads.items() if not f.done()]
        if active:
            names = ", ".join(f"'{n}'" for n in active)
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Cannot '{action_name}' while a policy is running on {names}. "
                            "Stop it first: action='stop_policy'."
                        )
                    }
                ],
            }
        return None

    @property
    def tool_spec(self) -> ToolSpec:
        """The Strands ``ToolSpec`` (name, description, JSON input schema) the agent sees.

        The description enumerates the full action surface (create_world,
        add_robot, run_policy, render, ...) and the input schema is the
        module-level ``_TOOL_SPEC_SCHEMA`` cached at import time, so building
        the spec is allocation-cheap on every access.
        """
        # schema cached at module load; see _TOOL_SPEC_SCHEMA
        return {
            "name": self.tool_name_str,
            "description": (
                "Programmatic MuJoCo simulation environment (stateful session). "
                "One world per instance; actions form an implicit state machine starting with "
                "create_world. Scene mutations (add_robot, remove_robot, add_object, remove_object, "
                "move_object, add_camera, remove_camera, load_scene) are blocked while a policy "
                "is running - stop it first. Create worlds, add robots from URDF "
                "(direct path or auto-resolve from data_config name), add objects, run VLA policies, "
                "render cameras, record trajectories, domain randomize. "
                "Same Policy ABC as real robot control - sim and real with zero code changes. "
                "Actions (77 total): "
                "[World] create_world, load_scene, reset, get_state, destroy, export_xml; "
                "[Robots] add_robot, remove_robot, list_robots, get_robot_state, list_bodies; "
                "[Objects] add_object, remove_object, move_object, list_objects; "
                "[Cameras] add_camera, remove_camera, list_cameras; "
                "[Policy] run_policy, start_policy, stop_policy, eval_policy, replay_episode, list_policies_running; "
                "[Rendering] render, render_depth, render_all, get_world_point, open_viewer, close_viewer; "
                "[Physics] step, set_gravity, set_timestep, set_joint_positions, set_joint_velocities, "
                "apply_force, get_contacts, get_contact_forces, get_body_state, get_energy, "
                "get_total_mass, get_ground_height, get_sensor_data, get_jacobian, get_mass_matrix, inverse_dynamics, "
                "forward_kinematics, save_state, load_state, set_body_properties, set_geom_properties; "
                "[Manipulation] attach_bodies, detach_bodies, actuate_robot, zero_dynamics; "
                "[Motion primitives] move_to (Cartesian EE transport via IK; not collision-aware), "
                "set_gripper (open/close set-point), rotate_wrist (wrist-yaw set-point holding position); "
                "[Scene MJCF] replace_scene_mjcf, patch_scene_mjcf, raycast, multi_raycast; "
                "[Recording] start_recording, stop_recording, get_recording_status, "
                "start_cameras_recording, stop_cameras_recording, get_cameras_recording_status; "
                "[Randomize] randomize, set_obs_noise (additive Gaussian sensor noise on observations and rendered frames); "
                "[Benchmark] list_benchmarks, register_benchmark_from_file, register_builtin_benchmarks, evaluate_benchmark; "
                "[Registry] list_urdfs, register_urdf, get_features. "
                "Call destroy() at session end to release resources."
            ),
            "inputSchema": {"json": _TOOL_SPEC_SCHEMA},
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        """Dispatch one agent tool call and yield its single ``ToolResultEvent``.

        Reads the ``action`` (and its arguments) from ``tool_use["input"]``,
        runs it against this stateful session, and yields exactly one result
        event carrying the originating ``toolUseId``. Any exception is caught
        and surfaced as a ``status="error"`` result rather than propagating
        past dispatch, per the agent-tool contract.
        """
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            requested = input_data.get("action", "")
            result = self._unpublished_action_error(requested) or self._dispatch_action(requested, input_data)
            yield ToolResultEvent(dict(toolUseId=tool_use_id, **result))  # type: ignore[typeddict-item]
        except Exception as e:
            yield ToolResultEvent(
                {
                    "toolUseId": tool_use.get("toolUseId", ""),
                    "status": "error",
                    "content": [{"text": f"Sim error: {e}"}],
                }
            )

    # Policy orchestration overrides (MuJoCo-specific wiring)

    def start_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: "Policy | None" = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Start policy execution on a background thread (non-blocking).

        MuJoCo override: reuses the ThreadPoolExecutor owned by
        ``Simulation`` so agent tools can kick off long-running policies
        without blocking the event loop.

        Concurrency (GH #114): multiple policies can run simultaneously on
        *different* robots. MuJoCo's ``mj_step`` and ``ctrl[]`` writes are
        still serialized via ``self._lock`` (MuJoCo ``model``/``data`` are
        not thread-safe for concurrent mutation), but each robot owns a
        disjoint slice of ``data.ctrl[]`` so there's no semantic conflict.

        A second ``start_policy`` on the *same* robot is still rejected.

        accepts ``n_steps`` (primary) or legacy ``max_steps`` as an
        alternate horizon specification; run_policy converts to duration.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}
        if not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}

        # Per-robot gate: another policy running on a DIFFERENT robot is fine.
        if err := self._require_no_running_policy("start_policy", robot_name=robot_name):
            return err

        # Validate the step horizon synchronously, before submitting to the
        # executor. run_policy runs on a background thread, so a malformed
        # horizon (duration <= 0, n_steps <= 0, control_frequency <= 0) would
        # otherwise be reported only inside the future - the caller would
        # receive a false "started" success and the robot would be left marked
        # as running. Both horizon knobs are covered: the step count via
        # _resolve_horizon, and the wall-clock duration it falls back to (the
        # default path, when n_steps is omitted) via _validate_duration. The
        # seed is covered for the same reason and was not: an unusable one
        # reached NumPy on the worker thread, where the raise was swallowed, so
        # the caller read "started" and the rollout ran unseeded.
        if err := self._validate_positive_frequency(control_frequency, "start_policy"):
            return err
        resolved_duration, resolved_n_steps, horizon_error = self._resolve_horizon(
            n_steps, max_steps, control_frequency, duration, "start_policy"
        )
        if horizon_error is not None:
            return horizon_error
        if resolved_n_steps is None:
            if err := self._validate_duration(resolved_duration, "start_policy"):
                return err
        if err := self._validate_action_horizon(action_horizon, "start_policy"):
            return err
        if err := self._validate_seed(seed, "start_policy"):
            return err
        # Same reason as the horizon guards above: a malformed video config
        # would otherwise be rejected inside the future, after the caller has
        # already been told the policy started and the robot marked running.
        if err := self._validate_video_config(video, "start_policy"):
            return err
        # Likewise for the provider keyword bags: a non-mapping policy_config /
        # policy_kwargs only fails when create_policy / get_actions splats it,
        # i.e. inside the future, so without this guard the caller receives a
        # false "started" for a rollout that never produced an action.
        if err := self._validate_policy_mapping(policy_config, "policy_config", "start_policy"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "start_policy"):
            return err
        # Validate the rate against the open recording synchronously, before
        # the executor submit below - a refusal after submit would report
        # "started" to a caller whose rollout cannot be recorded correctly.
        if err := self._validate_recording_rate(control_frequency, "start_policy"):
            return err

        # Concurrent multi-robot policies run on disjoint ctrl slices (physics
        # serialized by _lock). For SYNCHRONIZED multi-robot *recording* (both
        # robots captured in one merged frame per step), use run_multi_policy
        # instead - independent start_policy threads each step physics and write
        # frames separately, which is correct for live control but interleaves
        # for a shared recorder. start_policy while recording is left to the
        # caller's intent (run_multi_policy is the recommended recording path).
        # Resolve the provider synchronously. run_policy performs this check
        # too, but it runs on the worker below: a raise there is captured in
        # the future and this method would still report "Policy started",
        # leaving the caller with a success for a rollout that never began.
        if err := self._unresolvable_policy_provider_error(policy_provider, policy_config):
            return err

        future = self._executor.submit(
            self.run_policy,
            robot_name,
            policy_provider=policy_provider,
            policy_config=policy_config,
            instruction=instruction,
            duration=duration,
            control_frequency=control_frequency,
            action_horizon=action_horizon,
            fast_mode=fast_mode,
            video=video,
            policy_object=policy_object,
            n_steps=n_steps,
            max_steps=max_steps,
            policy_kwargs=policy_kwargs,
            seed=seed,
        )
        self._policy_threads[robot_name] = future
        self._policy_rates[robot_name] = float(control_frequency)

        return {
            "status": "success",
            "content": [{"text": f"Policy started on '{robot_name}' (async)"}],
        }

    def _make_run_policy_hook(self, robot_name: str, instruction: str):
        """MuJoCo override: recording + policy_running flag + lock.

        Returns an ``on_frame(step, obs, action)`` closure that:
        * flips ``robot.policy_running`` so ``stop_policy`` can interrupt,
        * appends to ``_backend_state["trajectory"]`` when recording,
        * forwards frames to the LeRobot ``dataset_recorder`` if attached,
        * raises ``PolicyStopped`` when the user calls ``stop_policy``.
        """
        import numpy as np

        from strands_robots.simulation.models import TrajectoryStep

        world = self._world
        if world is None or not registered(world.robots, robot_name):
            return None

        robot = world.robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0

        lock = self._lock

        # Action columns this rollout is responsible for: the driven robot's own
        # actuators. A declared column the policy never produced cannot be written
        # as a placeholder without persisting a command nobody issued, so
        # ``add_frame`` refuses it.
        #
        # Resolved on the first recorded frame and cached, rather than up front:
        # ``robot_action_keys`` is explicitly best-effort for the runner's
        # fail-fast probe (a backend quirk or a mid-rollout teardown may make it
        # raise, and that must not mask the primary "robot has not moved" signal),
        # so the hook must not call it for a rollout that is not recording. Where a
        # recording IS attached the keys are load-bearing - without them the frame
        # cannot be checked - so a raise there correctly fails the recording.
        action_key_cache: dict[bool, list[str]] = {}

        def _required_action_keys(prefixed: bool) -> list[str]:
            """Action columns this frame owes the recorder, resolved once."""
            cached = action_key_cache.get(prefixed)
            if cached is None:
                keys = self.robot_action_keys(robot_name)
                cached = [f"{robot_name}__{key}" for key in keys] if prefixed else list(keys)
                action_key_cache[prefixed] = cached
            return cached

        # N4: stream per-step telemetry on the mesh. publish_step existed with
        # consumers (robot_mesh watch, dashboards) but ZERO producers - no
        # rollout ever emitted it. Rate-limited to ~10 Hz to respect the
        # transport caps. Prefer the robot's own child-peer mesh (per-robot
        # topic), fall back to the parent sim's mesh.
        _mesh = getattr(robot, "mesh", None) or getattr(self, "mesh", None)
        _stream_state = {"last": 0.0}
        from strands_robots.mesh.session import stream_min_period_from_env

        # inf when step telemetry is off / misconfigured, so the throttle below
        # simply never fires. A bare division here killed run_policy hook setup
        # on STRANDS_MESH_STREAM_HZ=0.
        _stream_min_period = stream_min_period_from_env()

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            # Cooperative cancellation: stop_policy flips this flag.
            if not robot.policy_running:
                raise CooperativeStop(f"Policy stopped on '{robot_name}'")

            robot.policy_steps = step + 1

            if _mesh is not None:
                _now = time.time()
                if _now - _stream_state["last"] >= _stream_min_period:
                    _stream_state["last"] = _now
                    try:
                        _mesh.publish_step(step, observation, action, instruction=instruction)
                    except Exception:  # noqa: BLE001 - telemetry must not kill the rollout
                        pass

            with lock:
                if world._backend_state.get("recording", False):
                    world._backend_state["trajectory"].append(
                        TrajectoryStep(
                            timestamp=time.time(),
                            sim_time=world.sim_time,
                            robot_name=robot_name,
                            observation={k: v for k, v in observation.items() if not isinstance(v, np.ndarray)},
                            action=action,
                            instruction=instruction,
                        )
                    )
                    rec = world._backend_state.get("dataset_recorder")
                    if rec is not None:
                        # Honor the start_recording(cameras=...) scope: drop image
                        # arrays for cameras the caller chose not to record so the
                        # frame matches the (already scoped) dataset schema. None
                        # means record every camera (legacy default).
                        rec_cams = world._backend_state.get("recording_cameras")
                        observation = _drop_unrecorded_cameras(observation, rec_cams)
                        # In multi-robot scenes start_recording() declares
                        # the dataset schema with per-robot-prefixed joint ids
                        # (``alice__shoulder_pan``) so each agent has unique state/
                        # action columns. But _get_sim_observation() and the action
                        # dict use SHORT joint names (``shoulder_pan``). Without
                        # remapping here, add_frame() looks up the prefixed schema
                        # keys, finds nothing, and writes all-zero state/action
                        # vectors silently. Prefix scalar obs + action keys to match
                        # the schema. Camera values (ndarray) keep their (already
                        # namespaced) names - dataset_recorder normalizes '/'->'__'.
                        if len(world.robots) > 1:
                            obs_keyed = {
                                (k if isinstance(v, np.ndarray) else f"{robot_name}__{k}"): v
                                for k, v in observation.items()
                            }
                            act_keyed = {f"{robot_name}__{k}": v for k, v in action.items()}
                            rec.add_frame(
                                observation=obs_keyed,
                                action=act_keyed,
                                task=instruction,
                                required_action_keys=_required_action_keys(True),
                            )
                        else:
                            rec.add_frame(
                                observation=observation,
                                action=action,
                                task=instruction,
                                required_action_keys=_required_action_keys(False),
                            )

        return _hook

    def run_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: dict[str, Any] | None = None,
        policy_object: "Policy | None" = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        max_onframe_failures: int | None = None,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        n_episodes: int = 1,
        reset_between: bool = True,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        wbc_install_torque_control: bool = True,
        stop_when: dict[str, Any] | Callable[[SimEngine], bool] | None = None,
    ) -> dict[str, Any]:
        """MuJoCo ``run_policy`` override: pre-flight world check + graceful stop.

        Delegates to :meth:`SimEngine.run_policy` but clears the MuJoCo
        ``policy_running`` flag in a ``finally`` clause and swallows
        ``_PolicyStopped`` (which the ``on_frame`` hook raises on user
        cancellation) into a normal "policy stopped" result.

        forwards ``n_steps`` / ``max_steps`` to the base so LLM callers
        can specify horizon in steps rather than wall-clock seconds.

        ``n_episodes`` / ``reset_between`` are forwarded to
        :meth:`SimEngine.run_policy` for first-class multi-episode dataset
        collection: ``n_episodes > 1`` runs that many rollouts back-to-back,
        flushing a ``save_episode`` boundary after each (when recording) and
        resetting between episodes. See :meth:`SimEngine.run_policy`.

        ``stop_when`` (the semantic early-return predicate clause) is
        forwarded verbatim to :meth:`SimEngine.run_policy`, which compiles and
        validates it against the closed predicate registry; see its docstring
        for the schema and the ``stopped_reason`` telemetry contract.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        try:
            robot_name = self._resolve_single_robot(robot_name)
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}

        try:
            return super().run_policy(
                robot_name,
                policy_provider=policy_provider,
                policy_config=policy_config,
                instruction=instruction,
                duration=duration,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=video,
                policy_object=policy_object,
                n_steps=n_steps,
                max_steps=max_steps,
                max_onframe_failures=max_onframe_failures,
                control_substeps=control_substeps,
                policy_kwargs=policy_kwargs,
                seed=seed,
                n_episodes=n_episodes,
                reset_between=reset_between,
                async_rtc=async_rtc,
                rtc_inference_timeout_s=rtc_inference_timeout_s,
                wbc_install_torque_control=wbc_install_torque_control,
                stop_when=stop_when,
            )
        finally:
            if self._world is not None and registered(self._world.robots, robot_name):
                self._world.robots[robot_name].policy_running = False

    def run_multi_policy(
        self,
        policies: dict[str, "Policy"],
        instructions: dict[str, str] | str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int | dict[str, int] = _DEFAULT_ACTION_HORIZON,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Drive MULTIPLE robots with their own policies in a SINGLE
        synchronized control loop, recording ALL robots into ONE merged frame
        per timestep.

        This is the correct path for concurrent multi-robot data collection
        (e.g. two SO-100 arms doing a handover, or a bimanual setup). Unlike
        launching two independent ``start_policy`` threads - which each step
        physics and call ``add_frame`` separately, so the shared recorder
        receives interleaved single-robot frames (B4) - this loop:

        1. Observes every robot (joints) and renders all cameras ONCE.
        2. Queries each robot's policy for its action.
        3. Applies every robot's ctrl writes, then steps physics ONCE.
        4. Records ONE frame containing ALL robots' prefixed state/action
           (``alice__shoulder_pan`` ...) plus all camera images.

        So a 2-robot dataset has both arms co-observed in every frame - usable
        for bimanual / multi-agent policy training.

        Args:
            policies: Mapping ``{robot_name: Policy}``. Each robot must exist in
                the scene. Order defines the state/action column order.
            instructions: Either a single instruction string applied to all
                robots, or a ``{robot_name: instruction}`` mapping whose keys
                must name robots driven by this call (i.e. keys of ``policies``);
                a robot omitted from the mapping gets an empty instruction. A
                key matching no driven robot is rejected rather than silently
                dropped. The frame's recorded task is the first robot's
                instruction (LeRobot stores one task per frame).
            duration: Episode length in seconds (steps = duration x freq).
                Used only when no ``n_steps`` / ``max_steps`` is given. Must be
                a finite positive number; a non-positive, non-finite, or
                non-numeric value is a caller error, not a zero-step rollout
                reported as a success.
            control_frequency: Target Hz for policy action queries / physics.
                Must be a positive number.
            action_horizon: How many actions to consume from each policy's
                returned chunk before re-querying it (open-loop chunk
                execution, mirrors ``run_policy``). Either a single int applied
                to all robots, or a ``{robot_name: horizon}`` mapping for
                per-robot control. The effective per-robot chunk length is
                ``max(action_horizon, policy.actions_per_step)`` (see
                ``strands_robots.policies.resolve_chunk_length``): a
                chunk-emitting policy keeps its full trained chunk even when
                ``action_horizon`` is smaller, identical to ``run_policy``, so a
                model is never re-queried out-of-distribution in one driver but
                not the other. A robot's policy is only re-queried when its
                action queue drains - so an expensive VLA emitting a 30-action
                chunk with ``action_horizon=30`` runs inference ~once per 30
                steps instead of every step. Physics still advances ONE step per
                loop iteration, keeping all robots phase-aligned regardless of
                their individual re-query cadence. Every horizon must be a
                positive integer - the same guard ``run_policy`` /
                ``start_policy`` / ``eval_policy`` enforce - and mapping keys
                must name robots driven by this call; a robot omitted from the
                mapping uses the default horizon (8).
            n_steps: Alternate horizon in steps (overrides duration when set).
            max_steps: Legacy alias for ``n_steps``.

        Returns:
            Standard status dict with per-robot step counts.
        """
        import numpy as np

        from strands_robots._async_utils import _resolve_coroutine
        from strands_robots.policies.base import resolve_chunk_length

        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._validate_multi_policies(policies, "run_multi_policy"):
            return err

        # Validate every robot exists.
        for rname in policies:
            if not registered(self._world.robots, rname):
                return {"status": "error", "content": [{"text": self._unknown_robot_msg(rname)}]}

        # Reject if any of these robots already has a running async policy
        # (would double-step physics on that robot).
        self._prune_done_futures()
        busy = [r for r in policies if (f := registry_entry(self._policy_threads, r)) is not None and not f.done()]
        if busy:
            names = ", ".join(f"'{n}'" for n in busy)
            return {
                "status": "error",
                "content": [{"text": f"run_multi_policy: policy already running on {names}. Stop it first."}],
            }

        # Normalize instructions to a per-robot mapping through the shared
        # base helper (one refusal text for every backend), passing this
        # module's logger so the distinct-instructions one-task-per-frame
        # warning stays attributed to the MuJoCo loop that records the frame.
        instr_map, err = self._normalize_multi_policy_instructions(
            policies, instructions, "run_multi_policy", warn_logger=logger
        )
        if err is not None:
            return err
        assert instr_map is not None  # for the type checker: err is None <=> instr_map is not None

        # Resolve horizon (n_steps / max_steps override duration) through the
        # shared helpers, so this loop guards the same domain as run_policy: a
        # hand-rolled check only fired on the n_steps path, leaving the default
        # duration path to compute total_steps = int(duration * frequency) = 0
        # and report a rollout that never ran as a success.
        if err := self._validate_positive_frequency(control_frequency, "run_multi_policy"):
            return err
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

        # Normalize action_horizon to a per-robot mapping through the shared
        # base helper, which validates every horizon on the same positive-int
        # domain run_policy / start_policy / eval_policy use.
        horizon_map, err = self._normalize_multi_policy_horizons(
            policies, action_horizon, "run_multi_policy", default_horizon=_DEFAULT_ACTION_HORIZON
        )
        if err is not None:
            return err
        assert horizon_map is not None  # for the type checker: err is None <=> horizon_map is not None

        # Bind robot_state_keys for each policy (per-robot action keys -- the
        # actuators send_action resolves, not the joints; see robot_action_keys).
        for rname, pol in policies.items():
            try:
                pol.set_robot_state_keys(self.robot_action_keys(rname))
                self.bind_policy_sim_context(pol, rname)
            except Exception as exc:  # noqa: BLE001 - non-fatal, mirrors run_policy defensiveness
                logger.debug("set_robot_state_keys(%s) failed: %s", rname, exc)

        multi_robot = len(self._world.robots) > 1
        recorder = self._world._backend_state.get("dataset_recorder")
        recording = bool(self._world._backend_state.get("recording", False)) and recorder is not None
        # Every robot driven here contributes to the one merged frame, so the
        # merged action owes a value for each of their actuators. Resolved once
        # rather than per frame, and only when a recorder will consume it (see the
        # note on ``robot_action_keys`` being best-effort in _make_run_policy_hook).
        merged_required_action_keys = (
            [f"{rname}__{key}" if multi_robot else key for rname in policies for key in self.robot_action_keys(rname)]
            if recording
            else []
        )

        # Whether ANY policy needs images (renders are expensive; skip if none
        # need them AND we're not recording - recording always needs frames).
        any_needs_images = any(getattr(p, "requires_images", True) for p in policies.values())
        skip_images = not (any_needs_images or recording)

        total_steps = int(duration * control_frequency)
        action_sleep = 1.0 / control_frequency if control_frequency > 0 else 0.0

        # Mark all robots as running so stop_policy can interrupt the loop.
        for rname in policies:
            r = self._world.robots[rname]
            r.policy_running = True
            r.policy_instruction = instr_map[rname]
            r.policy_steps = 0

        # Per-robot action queue: actions remaining from the last chunk query.
        # A policy is only re-queried when its queue is empty, so expensive VLA
        # inference amortizes over up to ``horizon_map[robot]`` steps.
        from collections import deque

        action_queues: dict[str, deque] = {r: deque() for r in policies}

        step_count = 0
        stopped_early = False
        # Tracks whether the loop finished without an unexpected error. A normal
        # completion and a cooperative stop both leave a VALID partial/complete
        # episode the caller will save; any other exception (e.g. an empty action
        # chunk) leaves a dangling partial episode we must discard so the next
        # recording starts at frame 0 rather than appending to a half-episode.
        completed_cleanly = False
        try:
            while step_count < total_steps:
                # --- 1. Observe every robot + render cameras ONCE (under lock).
                # get_observation renders ALL cameras, so we only need to fetch
                # the camera images once (from any robot's observation); per-robot
                # we keep the scalar joint values.
                per_robot_obs: dict[str, dict[str, Any]] = {}
                camera_imgs: dict[str, Any] = {}
                first = True
                for rname in policies:
                    obs = self.get_observation(robot_name=rname, skip_images=(skip_images or not first))
                    # Split scalars (joints) from ndarrays (camera images).
                    scal = {k: v for k, v in obs.items() if not isinstance(v, np.ndarray)}
                    per_robot_obs[rname] = scal
                    if first:
                        camera_imgs = {k: v for k, v in obs.items() if isinstance(v, np.ndarray)}
                        first = False

                # --- 2. Get each robot's action for THIS step.
                # Re-query a policy ONLY when its action queue is empty (open-loop
                # chunk execution). Between re-queries we replay the buffered
                # chunk - so an expensive VLA runs inference once per horizon
                # steps, not every step. Observation is still gathered every step
                # (cheap) so recorded frames carry live state, and the chunk's
                # first action is computed from a fresh observation.
                per_robot_action: dict[str, dict[str, Any]] = {}
                for rname, pol in policies.items():
                    # Cooperative stop check.
                    if not self._world.robots[rname].policy_running:
                        raise CooperativeStop(f"Policy stopped on '{rname}'")
                    if not action_queues[rname]:
                        pol_obs = dict(per_robot_obs[rname])
                        # Give the policy this robot's camera view(s) too.
                        pol_obs.update(camera_imgs)
                        coro = pol.get_actions(pol_obs, instr_map[rname])
                        acts = _resolve_coroutine(coro)
                        # Buffer the policy's chunk; re-query only when drained.
                        # Size the chunk via the shared ChunkedPolicy rule so a
                        # chunk-emitting policy (actions_per_step == N) keeps its
                        # full trained chunk here exactly as the single-policy
                        # runner does - clamping to action_horizon alone would
                        # drop the chunk tail and force an out-of-distribution
                        # re-query for one driver but not the other.
                        _chunk = resolve_chunk_length(pol, horizon_map[rname])
                        for a in acts[:_chunk]:
                            action_queues[rname].append(a)
                    if not action_queues[rname]:
                        # A policy yielded no actions (empty chunk) -- emitting an
                        # empty action dict here would remap downstream to all-zero
                        # ctrl, silently corrupting the recording with dead frames.
                        # Fail loudly instead (Key Conventions #6: no silent
                        # zero-valued action on failure).
                        raise RuntimeError(
                            f"Policy for robot '{rname}' returned an empty action chunk; "
                            "cannot advance the synchronized loop. Check the policy's "
                            "get_actions() output."
                        )
                    per_robot_action[rname] = action_queues[rname].popleft()

                # --- 3. Apply ALL robots' ctrl, then step physics ONCE.
                with self._lock:
                    mj = self._mj
                    for rname, act in per_robot_action.items():
                        # write ctrl only (no per-robot mj_step) - we step once
                        # after every robot's ctrl is set.
                        robot = self._world.robots[rname]
                        pfx = robot.namespace or ""
                        self._apply_action_by_name(self._world._model, self._world._data, act, pfx, mj)
                    mj.mj_step(self._world._model, self._world._data)
                    # Kinematic attachments (attach_bodies mode="kinematic")
                    # follow their parent every physics step, on this
                    # synchronized loop as much as on the single-robot policy
                    # path. Fast no-op when none are registered.
                    self._apply_kinematic_attachments()
                    self._world.sim_time = self._world._data.time
                    self._world.step_count += 1
                    if hasattr(self, "_viewer_handle") and self._viewer_handle is not None:
                        self._viewer_handle.sync()

                # --- 4. Record ONE merged frame (all robots + all cameras).
                # ``recording`` already implies ``recorder is not None`` (see its
                # definition), but the explicit check narrows the Optional for the
                # type checker at the add_frame call below.
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
                    # Cameras are scene-global (already namespaced if injected
                    # per-robot); keep ndarray keys as-is, minus any the caller
                    # excluded via start_recording(cameras=...).
                    rec_cams = self._world._backend_state.get("recording_cameras")
                    merged_obs.update(_drop_unrecorded_cameras(camera_imgs, rec_cams))
                    task = instr_map[next(iter(policies))]
                    # add_frame writes to LeRobot's image-writer queue and parquet
                    # buffer; it does not touch MuJoCo model/data. The consistent
                    # state snapshot was already taken under self._lock in steps 1
                    # and 3, and merged_obs/merged_act are plain copies, so holding
                    # the physics lock across frame writeout would needlessly starve
                    # other lock holders (viewer sync, concurrent tool reads).
                    recorder.add_frame(
                        observation=merged_obs,
                        action=merged_act,
                        task=task,
                        required_action_keys=merged_required_action_keys,
                    )

                step_count += 1
                for rname in policies:
                    self._world.robots[rname].policy_steps = step_count

                if action_sleep:
                    time.sleep(action_sleep)

            completed_cleanly = True
        except CooperativeStop:
            # A cooperative stop is a normal, user-requested halt: the frames
            # captured so far are valid and the caller will save_episode them.
            stopped_early = True
            completed_cleanly = True
        finally:
            for rname in policies:
                if registered(self._world.robots, rname):
                    self._world.robots[rname].policy_running = False
            # Bailed mid-episode on an unexpected error (e.g. empty action
            # chunk): drop the partially-recorded frames so the next episode
            # begins at frame 0 instead of appending to a dangling half-episode.
            if not completed_cleanly and recording and recorder is not None:
                recorder.clear_episode_buffer()

        text = (
            f"{'stopped early' if stopped_early else 'completed'}: "
            f"run_multi_policy on {len(policies)} robots ({', '.join(policies)}) - "
            f"{step_count} synchronized steps"
            f"{' (recorded)' if recording else ''}"
        )
        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"steps": step_count}}],
        }

    # Action name aliases (tool-action -> method-name)
    # Dispatched actions that manage their own locking and MUST NOT run
    # under the blanket dispatch RLock (see _dispatch_action). Each one
    # acquires self._lock internally around its own model/data access.
    # The motion primitives (move_to / set_gripper / rotate_wrist) lock per
    # control tick (the step() pattern) so stop_policy / renders can
    # interleave during a long primitive.
    _SELF_LOCKING_ACTIONS: frozenset[str] = frozenset(
        {"step", "stop_policy", "remove_robot", "move_to", "set_gripper", "rotate_wrist"}
    )

    _ACTION_ALIASES = {
        "list_robots": "list_robots_info",
        "list_cameras": "list_cameras_info",
    }

    # Input field name -> method parameter name (syntactic sugar for the LLM)
    _FIELD_ALIASES = {
        "checkpoint_name": "name",
        "torque_vec": "torque",
        # Back-compat for README param names that predate the canonical
        # signatures (GH #373 friction #5). Customers copy-pasting older docs
        # passed ``camera_names=`` / ``joint_positions=``; accept them as
        # aliases so the call doesn't raise "unexpected keyword argument".
        "camera_names": "cameras",
        "joint_positions": "positions",
    }

    # Fields the schema publishes as a string. The dispatcher refuses a
    # non-string one before the method body assumes it is a string; see
    # :func:`strands_robots.utils.published_string_error` for the failures that
    # reached the caller without this.
    _PUBLISHED_STRING_PARAMS: frozenset[str] = _published_string_params(_FIELD_ALIASES)

    # Params the router passes through but not every method declares.
    # These are used for cross-cutting concerns (e.g. video on run_policy)
    # and must not be reported as "unknown" by the router.
    _ROUTER_PASSTHROUGH = {"action"}

    # Vector params with the component counts the target buffer can honor (for
    # dimension validation before numpy/MuJoCo sees them). 3 = xyz unless noted.
    # A param with several honorable counts lists them all, so the router never
    # rejects a vector the method itself accepts.
    _VECTOR_PARAM_LENGTHS: dict[str, tuple[int, ...]] = {
        "position": (3,),
        "target": (3,),
        "origin": (3,),
        "force": (3,),
        "torque": (3,),
        "torque_vec": (3,),
        "gravity": (3,),
        "direction": (3,),
        "point": (3,),
        "orientation": (4,),  # quaternion (w,x,y,z)
        "color": (3, 4),  # rgb (opaque alpha) or rgba
    }

    def _validate_and_build_kwargs(
        self,
        action: str,
        method_name: str,
        sig: inspect.Signature,
        remapped: dict[str, Any],
        received: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Validate input against method signature; return (kwargs, error_result).

        Exactly one of the tuple elements is non-None.

        Args:
            action: The action being dispatched, used in error text.
            method_name: The method *action* resolves to.
            sig: That method's signature, which the validation binds against.
            remapped: The payload with field aliases already rewritten to
                parameter names.
            received: The payload as the caller sent it, before that rewriting.
                A refusal names the spelling found here, so a caller who used an
                alias is not sent looking for a key their payload never had.
        """
        received = remapped if received is None else received
        # Strip self + VAR_POSITIONAL (*args) + VAR_KEYWORD (**kwargs) for signature
        # introspection; **kwargs methods accept arbitrary inputs, so we skip the
        # unknown-key check for them. Those methods own the check instead: the
        # forwarding sinks (attach_teleop, stream_dataset) hand the residual keys
        # to their callee, and the discarding ones (randomize, set_obs_noise)
        # reject them via unknown_kwargs_error - otherwise skipping here would
        # make a misspelled parameter a silent no-op on exactly those actions.
        named_params = {
            n: p
            for n, p in sig.parameters.items()
            if n != "self" and p.kind not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        }
        method_has_var_keyword = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        method_param_names = set(named_params)
        accepted_field_names = method_param_names | set(self._FIELD_ALIASES.keys()) | self._ROUTER_PASSTHROUGH

        # run_policy folds flat video keys into a structured `video` dict; those
        # flat keys are legitimate at the router boundary even though run_policy
        # itself takes `video=`.
        if action == "run_policy":
            accepted_field_names |= {"output_path", "fps", "camera_name"}

        # name/robot_name are aliased in both directions in the legacy router;
        # allow either here so we don't flag the alias as unknown.
        if "name" in method_param_names:
            accepted_field_names.add("robot_name")
        if "robot_name" in method_param_names:
            accepted_field_names.add("name")

        # 1) Unknown kwargs (skipped for **kwargs methods which legitimately passthrough)
        unknown = [] if method_has_var_keyword else [k for k in remapped if k not in accepted_field_names]
        if unknown:
            valid_sorted = sorted(
                _reported_param_name(param, self._FIELD_ALIASES, received) for param in method_param_names - {"action"}
            )
            return None, {
                "status": "error",
                "content": [
                    {"text": (f"Unknown parameter '{unknown[0]}' for action '{action}'. Valid: {valid_sorted}")}
                ],
            }

        # 2) Scalar string type validation. The schema publishes these as
        # strings, and every value at this boundary arrives as JSON, so a
        # non-string one is a caller error - not something to carry into the
        # method body, which assumes a string and fails with a raw TypeError /
        # AttributeError that names neither the parameter nor a remedy. ``None``
        # is "unset" and is left to the parameter default.
        for sparam in sorted(self._PUBLISHED_STRING_PARAMS & remapped.keys()):
            svalue = remapped[sparam]
            if svalue is None:
                continue
            # Name the spelling the caller used. A field can arrive under an
            # alias (``checkpoint_name`` -> ``name``), and reporting the
            # canonical parameter would send them looking for a key their
            # payload does not contain.
            reported = _reported_param_name(sparam, self._FIELD_ALIASES, received)
            smsg = published_string_error(svalue, reported, f"Action '{action}'")
            if smsg is not None:
                return None, {"status": "error", "content": [{"text": smsg}]}

        # 3) Vector dimension validation (applies before method runs)
        for vparam, accepted_lens in self._VECTOR_PARAM_LENGTHS.items():
            if vparam not in remapped:
                continue
            val = remapped[vparam]
            if val is None:
                continue
            expected_len = " or ".join(str(n) for n in accepted_lens)
            reported_vparam = _reported_param_name(vparam, self._FIELD_ALIASES, received)
            n_components = sequence_length(val)
            if n_components is None:
                return None, {
                    "status": "error",
                    "content": [{"text": f"Parameter '{reported_vparam}' must be a list of {expected_len} numbers."}],
                }
            if n_components not in accepted_lens:
                return None, {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Parameter '{reported_vparam}' must be a list of {expected_len} numbers, "
                                f"got {n_components}."
                            )
                        }
                    ],
                }
            for i, component in enumerate(val):
                # Accept any real scalar, including NumPy types (np.float32 /
                # np.int64 / ...): vector params like position / gravity / point
                # naturally arrive from an observation or mj_data (a NumPy
                # array), so `[float(x) for x in obs[...]]` is not required of
                # the caller. isinstance(_, (int, float)) rejected those (only
                # np.float64 subclasses float). numbers.Real still rejects
                # bool / np.bool_ (np.bool_ is not a Real) and non-numeric junk.
                if not isinstance(component, numbers.Real) or isinstance(component, bool):
                    return None, {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Parameter '{reported_vparam}'[{i}] must be numeric, "
                                    f"got {type(component).__name__}."
                                )
                            }
                        ],
                    }

        # 4) Build kwargs + check required params
        kwargs: dict[str, Any] = {}
        for param_name, param in named_params.items():
            if param_name == "name" and "name" not in remapped and "robot_name" in remapped:
                kwargs["name"] = remapped["robot_name"]
            elif param_name == "robot_name" and "robot_name" not in remapped and "name" in remapped:
                kwargs["robot_name"] = remapped["name"]
            elif param_name in remapped:
                kwargs[param_name] = remapped[param_name]
            elif param.default is inspect.Parameter.empty:
                reported_missing = _reported_param_name(param_name, self._FIELD_ALIASES, received)
                return None, {
                    "status": "error",
                    "content": [{"text": f"Action '{action}' requires parameter '{reported_missing}'."}],
                }

        # 5) Residual keys for **kwargs methods. The unknown-key check above is
        # skipped for them, so dropping the keys here too would leave the input
        # unvalidated AND unused: a forwarding sink would never see the option
        # the caller asked for, and a discarding sink could not reject a
        # misspelling. Hand them over and let the method decide.
        if method_has_var_keyword:
            kwargs.update({k: v for k, v in remapped.items() if k not in accepted_field_names})

        return kwargs, None

    def _dispatch_action(self, action: str, d: dict[str, Any]) -> dict[str, Any]:
        """Route action to the matching method with full input validation.

        Validation layer:
          * unknown top-level params are rejected with a friendly message,
          * missing required params produce a "requires parameter X" error
            (no raw Python ``TypeError``),
          * a field the schema publishes as a string is refused unless it is
            one, before the method body assumes it,
          * vector params have length + numeric dtype checked before the
            value reaches numpy / MuJoCo,
          * and every one of those refusals names the field by the spelling the
            caller sent, or by the one the schema publishes - see
            :func:`_reported_param_name`, since the alias rewriting above runs
            first and a refusal echoing the rewritten name can point at a field
            the schema does not carry.

        Policy-provider kwargs are nested under ``policy_config`` (never
        top-level) so the dispatcher stays backend-agnostic.

        Resolution is ``getattr`` by name with no allowlist, so the router is
        deliberately wider than ``tool_spec.json``'s ``action`` enum: the enum is
        the curated agent-facing subset, and a public method absent from it stays
        callable from Python through this method. That breadth is intended, and
        it is accounted for: every public method of this class is either
        published in that enum or recorded as deliberately Python-only, and one
        that is neither fails the backend's tool-spec guards. So a newly added
        method cannot become dispatchable-but-unadvertised in silence, and a
        deliberate omission stays distinguishable from an oversight.

        That width no longer reaches an agent. Both agent-facing entry points
        (:meth:`__call__` and :meth:`stream`) refuse a non-enum action first,
        via :meth:`_unpublished_action_error`, so what a model can invoke is
        exactly what it was advertised - the resolution of #2093. Calling this
        method directly bypasses that refusal deliberately: it is the Python
        path, and the width is what makes it useful there.
        """
        method_name = self._ACTION_ALIASES.get(action, action)
        method = getattr(self, method_name, None)

        if method is None or action.startswith("_"):
            return {"status": "error", "content": [{"text": f"Unknown action: {action}"}]}

        cache = getattr(self, "_sig_cache", None)
        if cache is None:
            self._sig_cache = cache = {}
        if method_name not in cache:
            cache[method_name] = inspect.signature(method)
        sig = cache[method_name]

        # Field-alias rewriting (before validation so the validator sees
        # canonical names).
        remapped = {k: v for k, v in d.items() if k != "action"}
        for field_key, param_key in self._FIELD_ALIASES.items():
            if field_key in remapped and param_key not in remapped:
                remapped[param_key] = remapped.pop(field_key)

        # Fold flat video keys into `video` dict for the rollout/eval actions.
        if action in ("run_policy", "start_policy", "eval_policy", "evaluate_benchmark") and "video" not in remapped:
            _video_flat: dict[str, Any] = {}
            if "output_path" in remapped:
                _video_flat["path"] = remapped.pop("output_path")
            if "fps" in remapped:
                _video_flat["fps"] = remapped.pop("fps")
            # camera_name is shared with render(); only treat as video camera
            # when paired with an output path.
            if _video_flat.get("path") and "camera_name" in remapped:
                _video_flat["camera"] = remapped.pop("camera_name")
            if _video_flat.get("path"):
                remapped["video"] = _video_flat

        kwargs, err = self._validate_and_build_kwargs(action, method_name, sig, remapped, d)
        if err is not None:
            return err
        assert kwargs is not None
        # Most dispatched actions are serialized under self._lock (RLock): the
        # single chokepoint that prevents concurrent reads/writes to MuJoCo
        # model/data from the agent thread while a PolicyRunner worker is
        # mid-mj_step. Individual methods that also acquire the lock are
        # harmless (RLock is reentrant on the same thread).
        #
        # A few actions in _SELF_LOCKING_ACTIONS must run WITHOUT the blanket
        # lock because they manage their own concurrency:
        #   * remove_robot joins on the target robot's PolicyRunner Future;
        #     that worker needs self._lock to observe the cooperative-stop
        #     flag, so holding the lock across the join deadlocks (the join
        #     then swallows the TimeoutError and rebuilds the scene under a
        #     still-live worker holding stale model/data ids).
        #   * step acquires+releases self._lock in bounded batches so
        #     stop_policy and the recorder/MJPEG threads can interleave on long
        #     runs; the blanket lock would hold it for the whole call and
        #     defeat the batching.
        #   * stop_policy only flips a bool and needs no lock; keeping it off
        #     the blanket lock lets it interrupt a long-running step.
        if method_name in self._SELF_LOCKING_ACTIONS:
            return method(**kwargs)
        with self._lock:
            return method(**kwargs)

    def stop_policy(self, robot_name: str = "") -> dict[str, Any]:
        """Stop a running policy on the given robot (cooperative cancellation).

        Counterpart to :meth:`start_policy`. Flips the robot's
        ``policy_running`` flag; the background loop in
        :meth:`_run_policy_loop` sees it and raises :class:`PolicyStopped`
        which is caught cleanly inside :meth:`start_policy`.

        idempotent - if the robot exists but no policy is running, we
        still return success with 'Was not running' so callers can call
        stop_policy unconditionally. The only error case is an unknown
        robot_name.

        empty robot_name returns a clear error instead of a silent
        match against the first robot.
        """
        if not robot_name:
            return {
                "status": "error",
                "content": [{"text": "stop_policy requires 'robot_name'."}],
            }
        if self._world is None or not registered(self._world.robots, robot_name):
            return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
        robot = self._world.robots[robot_name]
        was_running = robot.policy_running
        robot.policy_running = False
        msg = f"Stopped on '{robot_name}'" if was_running else f"Was not running on '{robot_name}'"
        return {"status": "success", "content": [{"text": msg}]}

    def list_policies_running(self) -> dict[str, Any]:
        """Return the names of robots currently running a policy.

        Useful for inspecting concurrent-policy state when running two or
        more VLA arms in the same scene (GH #114). Always returns a
        success dict so the LLM can parse it uniformly. Prunes stale
        completed Future entries as a side effect.
        """
        active = self._active_policy_robots()
        if not active:
            return {
                "status": "success",
                "content": [{"text": "No policies running."}],
            }
        robot_lines = "\n".join(f"  - {n}" for n in active)
        return {
            "status": "success",
            "content": [{"text": f"Active policies ({len(active)}):\n{robot_lines}"}],
        }

    # Cleanup

    # Cooperative-stop budget (seconds) - how long a caller-facing action waits
    # for a policy worker to notice ``policy_running = False`` and exit. A
    # policy worker might be mid-step when cleanup is called; give it bounded
    # time to see the cooperative-stop flag and exit cleanly before we null the
    # world and its in-flight ``mj_step`` segfaults on a nulled
    # ``_model``/``_data``. ``remove_robot`` waits the same budget on the one
    # worker it stops itself, so the two paths cannot drift to different
    # answers about how long the protocol gets. Bounded rather than open-ended
    # so a wedged worker can never hang the caller; each path then decides what
    # to do with a lapse - cleanup logs and continues teardown, remove_robot
    # refuses the scene rebuild.
    # Override in tests via ``cleanup(policy_stop_timeout=...)`` if needed.
    _DEFAULT_POLICY_STOP_TIMEOUT = 5.0

    # Bounded wait for the world-handoff lock (seconds). A motion primitive
    # holds ``self._lock`` only for one control tick (a few physics substeps,
    # sub-millisecond); the longest holder is ``step``, bounded by
    # ``_STEPS_PER_BATCH`` ``mj_step`` calls per acquisition rather than by the
    # whole requested count. So this ceiling is generous; it exists so cleanup can
    # never hang the host process on a wedged lock holder - the same tradeoff
    # ``_DEFAULT_POLICY_STOP_TIMEOUT`` makes for a wedged policy worker.
    _WORLD_HANDOFF_LOCK_TIMEOUT = 5.0

    def cleanup(self, policy_stop_timeout: float | None = None) -> None:
        """Release every resource owned by this Simulation instance.

        Concurrency (GH #116): nulling ``self._world`` while a policy worker
        thread is still inside ``mj_step(world._model, world._data)`` is a
        SIGSEGV waiting to happen. Previously cleanup called
        ``executor.shutdown(wait=False)`` right after setting
        ``self._world = None``, which meant the worker could still be
        holding stale pointers to freed arrays. The
        ``policy_running = False`` flag was flipped but never awaited.

        New order:
          1. Signal every live policy to stop (``policy_running = False``).
          2. Await each outstanding Future with a bounded timeout - the
             ``on_frame`` hook sees the flag at the top of its next call
             and raises ``CooperativeStop`` which short-circuits run_policy.
          3. Any Future still not-done after the timeout: we log a warning
             and proceed - at that point the worker is wedged somewhere
             outside MuJoCo and a stale-pointer segfault is the lesser evil
             than hanging the host process on exit.
          4. Only AFTER workers have unwound do we null ``self._world``
             and tear down renderers / the viewer / the executor. The
             nulling itself happens under ``self._lock`` (bounded acquire):
             an analytic motion primitive runs on its caller's thread rather
             than through the executor, so the join in step 2 cannot cover
             it and the lock is the only thing that keeps the handoff from
             landing inside one of its control ticks.

        Args:
            policy_stop_timeout: Seconds to wait per active policy future - a
                positive finite number, validated by the same rule every other
                span of time is (:func:`strands_robots.utils.positive_finite_number_error`).
                ``None`` (default) uses ``_DEFAULT_POLICY_STOP_TIMEOUT`` (5s).
                Set to a small value in tests that want fast teardown. A budget
                the join cannot measure - ``0``, a negative value, ``nan``,
                ``inf`` or a non-real - is reported and resolved to that same
                default rather than refused: every one of them expires the wait
                before its first check, and abandoning a live worker is the
                stale-pointer window step 2 exists to close, so the safe budget
                is the honest reading of "no usable preference". Teardown always
                completes; see :func:`_resolve_policy_stop_timeout`.

        Note:
            Every name this method needs is bound at module scope. A
            finalizer calls this during interpreter shutdown, where the
            import system is already gone, so a function-local import here
            raises before the first teardown step and the ``__del__`` safety
            net releases nothing at all - reported only as a warning naming
            the interpreter rather than anything the caller can act on.
        """
        # Detach from the mesh network first (if attached). A truthy
        # ``self.mesh`` is any object exposing ``.stop()``; falsy values
        # (the default) mean this Simulation never joined a mesh and
        # there's nothing to release. Done BEFORE stopping policies so
        # peer-visible state is torn down cleanly even if the policy
        # teardown below hits the fallback ``wait=False`` path.
        #
        # Each robot added via ``add_robot`` may have
        # its own per-peer mesh (see ``_attach_robot_to_mesh``). Stop those
        # FIRST so external peers see them leave before the sim container
        # itself goes down - leaving the inverse order ("sim drops, robots
        # linger") would create zombie peer entries in remote ``get_peers``
        # results until their heartbeats expire.
        # Stop any local teleoperation loop + disconnect attached devices
        # (TeleopMixin) before mesh teardown. Best-effort.
        # Tear down the ROS 2 telemetry bridge (if any) before other teardown
        # so external subscribers see the node leave cleanly.
        with contextlib.suppress(Exception):
            self._shutdown_ros_bridge()
        if getattr(self, "_teleop_running", False) or getattr(self, "_teleops", None):
            with contextlib.suppress(Exception):
                self.stop_teleoperate()
        if self._world is not None:
            for r in list(self._world.robots.values()):
                self._detach_robot_from_mesh(r)
        if self.mesh:
            # Best-effort, mirroring the per-robot ``_detach_robot_from_mesh``
            # loop above: a mesh client that fails to stop (transport already
            # closed, peer registry error) must not abort the teardown of the
            # MuJoCo world, renderers, and the executor below.
            try:
                self.mesh.stop()
            except Exception as exc:  # noqa: BLE001 - teardown continues regardless
                logger.warning(
                    "cleanup: failed to stop mesh client (peer_id=%s): %s",
                    self.peer_id or "?",
                    exc,
                )
            finally:
                self.mesh = None

        timeout = _resolve_policy_stop_timeout(policy_stop_timeout, self._DEFAULT_POLICY_STOP_TIMEOUT)

        # Step 1 + 2: cooperative stop + bounded join BEFORE nulling world.
        # The ``policy_running`` flag is read by the MuJoCo-specific
        # ``_make_run_policy_hook`` at the top of its next call; setting
        # it here makes the worker raise CooperativeStop at its next step.
        if self._world is not None:
            for r in self._world.robots.values():
                r.policy_running = False

        # Prune completed futures so we only wait on genuinely-live ones.
        self._prune_done_futures()
        if self._policy_threads:
            for robot_name, fut in list(self._policy_threads.items()):
                try:
                    fut.result(timeout=timeout)
                except Exception as e:
                    # result() raises either the worker's exception OR a
                    # TimeoutError. Log and continue - we want cleanup to
                    # finish even on pathological workers.
                    logger.warning(
                        "cleanup: policy on '%s' did not stop within %.1fs: %s",
                        robot_name,
                        timeout,
                        e,
                    )
            self._policy_threads.clear()

        # Step 3: hand the world off UNDER ``self._lock``.
        #
        # The join above covers every worker submitted to the executor, but a
        # motion primitive (``move_to`` / ``set_gripper`` / ``rotate_wrist``)
        # runs on its CALLER's thread: no Future awaits it, and the
        # ``policy_running`` flag signalled in step 1 never applies to it
        # (primitives refuse to start while a policy runs, so they never set
        # it). ``self._lock`` is the only synchronisation such a primitive
        # has - it takes the lock per control tick to re-check the world,
        # step physics, then write back ``sim_time`` / ``step_count``.
        # Nulling ``self._world`` outside the lock therefore lands INSIDE a
        # tick, and the write-back raises ``AttributeError`` out of an API
        # documented never to raise. Holding the lock makes the handoff
        # atomic against a tick, so the primitive's next
        # ``_primitive_abort_reason`` check reports the structured
        # "world was destroyed ... aborting" error instead - the outcome that
        # helper is written to produce. ``load_scene`` brackets its own world
        # handoff the same way and for the same reason.
        #
        # The join must stay OUTSIDE the lock: a live policy worker takes the
        # lock per step, so awaiting it while holding the lock deadlocks. The
        # acquire is bounded for the same reason step 2 is bounded - cleanup
        # must not hang the host process on a wedged lock holder.
        handoff_locked = self._lock.acquire(timeout=self._WORLD_HANDOFF_LOCK_TIMEOUT)
        if not handoff_locked:
            logger.warning(
                "cleanup: world-handoff lock still held after %.1fs; nulling the world anyway "
                "(a concurrent control tick may raise).",
                self._WORLD_HANDOFF_LOCK_TIMEOUT,
            )
        try:
            if self._world:
                self._world = None
        finally:
            if handoff_locked:
                self._lock.release()

        self._close_viewer()
        # close main-thread renderers before dropping the TLS object.
        # Renderers created on worker threads release their GL contexts
        # when those threads terminate; calling close() cross-thread
        # SIGSEGVs in cgl.free(), so we stay on main.
        self._close_main_thread_renderers()
        if hasattr(self, "_renderer_tls"):
            self._renderer_tls = threading.local()
        # Step 4: shut the executor down now that all our policy futures
        # are either completed or abandoned. wait=False is OK at this
        # point because we've already drained policy workers above - any
        # remaining thread is render / observation work that's safe to
        # outlive us.
        self._executor.shutdown(wait=False)
        self._shutdown_event.set()

    def __enter__(self) -> "Simulation":
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()


# Backward-compatible aliases (this engine originally shipped as ``Simulation``)
Simulation = MuJoCoSimEngine
MuJoCoSimulation = MuJoCoSimEngine
