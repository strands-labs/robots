"""MuJoCo Simulation backend - AgentTool orchestrator + shared state host.

Architecture notes (honest version, see GH #118)

The ``Simulation`` class uses multiple-inheritance to compose four mixins
(``PhysicsMixin``, ``RenderingMixin``, ``RecordingMixin``, ``RandomizationMixin``)
on top of the ``SimEngine`` ABC and the Strands ``AgentTool`` base. The
split keeps each file navigable (physics.py ~1150 lines, rendering.py ~730,
etc.) but the mixin boundaries describe *where code lives*, NOT the
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

import inspect
import json
import logging
import math
import os
import re
import threading
import time
from collections.abc import AsyncGenerator
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots.simulation.base import SimEngine
from strands_robots.simulation.model_registry import (
    count_sim_robots,
    list_available_models,
    resolve_model,
)
from strands_robots.simulation.model_registry import (
    register_urdf as _register_urdf,
)
from strands_robots.simulation.models import SimCamera, SimObject, SimRobot, SimStatus, SimWorld
from strands_robots.simulation.mujoco.backend import _ensure_mujoco
from strands_robots.simulation.mujoco.physics import PhysicsMixin
from strands_robots.simulation.mujoco.randomization import RandomizationMixin
from strands_robots.simulation.mujoco.recording import RecordingMixin
from strands_robots.simulation.mujoco.rendering import RenderingMixin
from strands_robots.simulation.mujoco.scene_ops import (
    eject_body_from_scene,
    eject_robot_from_scene,
    inject_camera_into_scene,
    inject_object_into_scene,
    inject_robot_into_scene,
    patch_scene_mjcf,
    replace_scene_mjcf,
)
from strands_robots.simulation.mujoco.spec_builder import SpecBuilder
from strands_robots.simulation.policy_runner import CooperativeStop

if TYPE_CHECKING:
    from strands_robots.policies import Policy

logger = logging.getLogger(__name__)

_TOOL_SPEC_PATH = Path(__file__).parent / "tool_spec.json"

# Tool schema is 357 lines of JSON. `tool_spec` property is on the LLM hot path
# (called on every `strands` invocation). Load once at import, not per access.
with open(_TOOL_SPEC_PATH) as _f:
    _TOOL_SPEC_SCHEMA: dict[str, Any] = json.load(_f)


class MuJoCoSimEngine(
    PhysicsMixin,
    RenderingMixin,
    RecordingMixin,
    RandomizationMixin,
    SimEngine,
    AgentTool,
):
    """Programmatic MuJoCo simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language → tool actions.

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
        mesh: bool = False,
        peer_id: str | None = None,
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
            mesh: Optional mesh-networking hook. Falsy (default) keeps
                the Simulation standalone - all mesh code paths are
                no-ops. When set to a live mesh-client object exposing
                ``.stop()``, ``cleanup()`` will detach this Simulation
                from the peer network before tearing down the MuJoCo
                world. The attribute is plain (not a property), so
                consumers may attach a client after construction.
            peer_id: Stable identifier the mesh transport uses to
                address this Simulation. Opaque to MuJoCo itself; only
                consulted when ``mesh`` is truthy.
            **kwargs: Forwarded to ``AgentTool.__init__`` for subclass
                compatibility.
        """
        super().__init__()
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        # Mesh attributes are stored plainly (no property wrapper) so
        # downstream code can swap in a real mesh client after
        # construction without a setter dance. See the ``mesh`` /
        # ``peer_id`` docstring entries above for the contract.
        self.mesh: Any = mesh if mesh else None
        self.peer_id: str | None = peer_id

        self._world: SimWorld | None = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{tool_name}_sim")
        # Per-robot Future refs for *active* policies. Completed futures are
        # pruned by ``_active_policy_futures()``/``_prune_done_futures()`` so
        # the dict never grows unboundedly and never reports stale "running".
        self._policy_threads: dict[str, Future] = {}
        self._shutdown_event = threading.Event()
        # ``self._lock`` (RLock) serializes ALL access to MuJoCo
        # ``model``/``data`` arrays — both reads and writes. MuJoCo arrays
        # are NOT safe for concurrent reads during mutation (a racing
        # mj_step can produce torn/stale values). The lock is acquired:
        #
        #   * In ``_dispatch_action`` — so every agent-dispatched action
        #     (all 37 handlers) is automatically serialized.
        #   * In ``send_action`` / ``get_observation`` — so the
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

    # Public Properties — read-only introspection.
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
        if robot_name not in self._world.robots:
            return {}
        if skip_images and self._world is not None and self._world._backend_state.get("recording"):
            # T26: dataset recording needs every frame's image obs. Override
            # the policy's skip hint when an active recorder is attached.
            skip_images = False
        with self._lock:
            return self._get_sim_observation(robot_name, skip_images=skip_images)

    def send_action(self, action: dict[str, Any], robot_name: str | None = None, n_substeps: int = 1) -> None:
        """Apply action to simulation (Robot ABC compatible).

        Thread-safety: acquires self._lock around ctrl writes + mj_step,
        as documented in base.py's SimEngine contract. Concurrent calls
        from the agent's dispatch thread and a PolicyRunner worker are
        serialized here.
        """
        if self._world is None or self._world._model is None:
            return
        if robot_name is None:
            if not self._world.robots:
                return
            robot_name = next(iter(self._world.robots))
        if robot_name not in self._world.robots:
            return
        with self._lock:
            self._apply_sim_action(robot_name, action, n_substeps=n_substeps)

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

    def create_world(
        self, timestep: float | None = None, gravity: list[float] | None = None, ground_plane: bool = True
    ) -> dict[str, Any]:
        """Create a new simulation world."""
        # mujoco verified at __init__

        if self._world is not None and self._world._model is not None:
            return {
                "status": "error",
                "content": [{"text": "World already exists. Use action='destroy' first, or action='reset'."}],
            }

        if gravity is None:
            _gravity = [0.0, 0.0, -9.81]
        elif isinstance(gravity, (int, float)):
            _gravity = [0.0, 0.0, float(gravity)]
        else:
            _gravity = list(gravity)

        self._world = SimWorld(
            timestep=timestep or self.default_timestep,
            gravity=_gravity,
            ground_plane=ground_plane,
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
                        "🌍 Simulation world created\n"
                        f"⚙️ Timestep: {self._world.timestep}s ({1 / self._world.timestep:.0f}Hz physics)\n"
                        f"🌐 Gravity: {self._world.gravity}\n"
                        f"📷 Default camera ready\n"
                        f"🤖 Robot models: {self._cheap_robot_count()} available\n"
                        "💡 Add robots: action='add_robot' (urdf_path or data_config)\n"
                        "💡 Add objects: action='add_object'\n"
                        "💡 List URDFs: action='list_urdfs'"
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

        try:
            self._world = SimWorld()
            # Load the scene as a live MjSpec - this gives us a mutable AST
            # for downstream add_object/add_robot operations, matching the
            # contract produced by _compile_world for fresh worlds.
            spec = SpecBuilder.from_file(scene_path)
            self._world._backend_state["spec"] = spec
            self._world._model = spec.compile()
            self._world._data = mj.MjData(self._world._model)
            # Forward the freshly-allocated MjData so derived state
            # (xpos / xquat / xmat / sensor data) is populated. Without
            # this, ``Renderer.update_scene`` finds the body transforms
            # unset and returns a skybox-only gradient on the first
            # render call after load_scene - the bug-D pattern that
            # rounds 11/12/13 in #168 chased through several wrong
            # directions before #168 verification isolated it to
            # this missing forward call (#168).
            #
            # Cost: O(model.nbody) - negligible for typical scenes.
            # Failure here is genuinely a bug in the loaded MJCF
            # (e.g. inconsistent qpos vs joint definitions), so let it
            # propagate to the outer ``except Exception`` below where
            # it gets converted to a structured error response.
            mj.mj_forward(self._world._model, self._world._data)
            self._world.status = SimStatus.IDLE

            # Cache the canonical serialisation; legacy readers use this.
            try:
                self._world._backend_state["xml"] = spec.to_xml()
            except Exception as xml_err:
                logger.debug("spec.to_xml() on loaded scene failed: %s", xml_err)

            self._world._backend_state["scene_loaded"] = True
            self._world._backend_state["scene_base_dir"] = os.path.dirname(os.path.abspath(scene_path))

            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🌍 Scene loaded from {os.path.basename(scene_path)}\n"
                            f"🦴 Bodies: {self._world._model.nbody}, 🔩 Joints: {self._world._model.njnt}, ⚡ Actuators: {self._world._model.nu}\n"
                            "💡 Use action='get_state' to inspect, action='step' to simulate"
                        )
                    }
                ],
            }
        except Exception as e:
            logger.error("Failed to load scene: %s", e)
            return {"status": "error", "content": [{"text": f"Failed to load scene: {e}"}]}

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
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Use action='create_world' first."}]}
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
                        f"🔄 Scene replaced via raw MJCF\n"
                        f"🦴 Bodies: {model.nbody}, 🔩 Joints: {model.njnt}, ⚡ Actuators: {model.nu}, 📷 Cameras: {model.ncam}\n"
                        "⚠️ world.robots / world.objects / world.cameras registries were NOT updated - "
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

        The whole batch is applied, then the spec is recompiled once. If any
        op fails, the batch is rejected and the world is rolled back to its
        pre-patch state (from an XML snapshot). Use this for fast iterative
        edits; use ``replace_scene_mjcf`` when you need to express MJCF
        elements not covered by the supported op vocabulary.
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Use action='create_world' first."}]}
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
                        f"🩹 Patched scene: {applied} op(s) applied\n"
                        f"🦴 Bodies: {model.nbody}, 🔩 Joints: {model.njnt}, ⚡ Actuators: {model.nu}, 📷 Cameras: {model.ncam}\n"
                        "⚠️ world.robots / world.objects / world.cameras registries were NOT updated."
                    )
                }
            ],
        }

    def _compile_world(self) -> None:
        """Build the MjSpec from ``self._world`` and compile it to MjModel.

        Stashes the live ``MjSpec`` in ``_backend_state["spec"]`` so every
        subsequent scene mutation uses ``spec.recompile(model, data)`` in
        place - that preserves existing joint state automatically, replacing
        the legacy XML-round-trip helpers in ``scene_ops.py``.

        Also exports ``spec.to_xml()`` to ``_backend_state["xml"]`` for any
        consumer that still reads the raw MJCF string (e.g. ``load_scene``
        compatibility paths).
        """
        mj = self._mj
        assert self._world is not None  # only called after create_world
        spec = SpecBuilder.build(self._world)
        self._world._backend_state["spec"] = spec
        self._world._model = spec.compile()
        self._world._data = mj.MjData(self._world._model)
        # Forward the freshly-allocated MjData so derived state
        # (xpos / xquat / xmat) is populated - same rationale as in
        # ``load_scene`` (#168). Without this, the first
        # render after ``_compile_world`` returns the skybox-only
        # gradient because body transforms are zero-initialised.
        mj.mj_forward(self._world._model, self._world._data)
        try:
            self._world._backend_state["xml"] = spec.to_xml()
        except Exception as xml_err:
            # spec.to_xml() is best-effort - if it fails we still have a
            # valid compiled model. The cached XML is a convenience for
            # tooling, not a correctness invariant.
            logger.debug("spec.to_xml() failed: %s", xml_err)
        self._world.status = SimStatus.IDLE

    def _recompile_world(self) -> dict[str, Any]:
        """Rebuild MjModel from scratch via :meth:`_compile_world`.

        This is the "nuke and pave" path used when the world config changes
        in a way that can't be expressed as a spec mutation (e.g. clearing
        every body). For incremental changes (add/remove body, camera),
        prefer ``_recompile_preserving_state`` in ``scene_ops.py`` which
        goes through ``spec.recompile(model, data)`` and preserves joint
        state.
        """
        try:
            self._compile_world()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"Recompile failed: {e}"}]}

    # Robot Management

    @staticmethod
    def _ensure_meshes(model_path: str, robot_name: str) -> dict[str, Any] | None:
        """Check if mesh files referenced by a model XML exist; auto-download if missing.

        Returns ``None`` on success (meshes present or downloaded cleanly) and
        a standard error dict on auto-download failure. Caller MUST propagate
        the error dict back to the agent - previously the return value was
        ignored and the error was silently swallowed, leaving the agent to
        hit a cryptic 'mesh not found' from MuJoCo instead.
        """
        model_dir = os.path.dirname(os.path.abspath(model_path))

        files_to_check = [model_path]
        try:
            with open(model_path) as _f:
                top_content = _f.read()
            for inc in re.findall(r'<include\s+file="([^"]+)"', top_content):
                inc_path = os.path.join(model_dir, inc)
                if os.path.exists(inc_path):
                    files_to_check.append(inc_path)
        except Exception:
            pass

        missing = False
        for xml_path in files_to_check:
            try:
                with open(xml_path) as _f:
                    content = _f.read()
            except Exception:
                continue

            mesh_files = re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content)
            if not mesh_files:
                continue

            meshdir_match = re.search(r'meshdir="([^"]*)"', content)
            meshdir = meshdir_match.group(1) if meshdir_match else ""
            xml_dir = os.path.dirname(os.path.abspath(xml_path))

            for mf in mesh_files:
                if not os.path.exists(os.path.join(xml_dir, meshdir, mf)):
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
        except (ImportError, FileNotFoundError, OSError) as e:
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
        becomes addressable on the mesh in its own right — the agent can
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
            # Sim itself isn't on a mesh — nothing to attach to. Stays a
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
                peer_type="robot",
                mesh=True,
            )
            if child_mesh is not None:
                robot.mesh = child_mesh
                robot.peer_id = child_mesh.peer_id
        except Exception as exc:  # noqa: BLE001 — mesh enrichment is best-effort
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

    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation via XML round-trip composition.

        Instead of replacing the entire world model, this method merges the
        robot's bodies, actuators, assets, and sensors into the existing scene
        XML.  This preserves previously-created world state (gravity, objects,
        cameras, other robots).
        """
        if self._world is None:
            return {"status": "error", "content": [{"text": "No world. Use action='create_world' first."}]}
        if err := self._require_no_running_policy("add_robot"):
            return err
        if name in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' already exists."}]}

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
                    "content": [
                        {
                            "text": f"No model found for '{data_config}'.\n💡 Use action='list_urdfs' to see available robots"
                        }
                    ],
                }
        elif not resolved_path and name:
            # deprecated fallback - try registry by instance name.
            import warnings as _warnings

            resolved_path = resolve_model(name)
            if resolved_path:
                _warnings.warn(
                    f"add_robot: resolving model via instance name '{name}' is deprecated; "
                    "pass data_config='<registry-key>' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if not resolved_path:
            return {"status": "error", "content": [{"text": "Either urdf_path or data_config is required."}]}
        if not os.path.exists(resolved_path):
            return {"status": "error", "content": [{"text": f"File not found: {resolved_path}"}]}

        mj = self._mj

        robot = SimRobot(
            name=name,
            urdf_path=resolved_path,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
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
            pfx = robot.namespace or ""
            model = self._world._model
            for i in range(model.ncam):
                cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name:
                    continue
                # Strip the robot namespace for our Python-side key - the
                # registry is keyed on the short name and we re-attach the
                # namespace when passing to the renderer.
                short = cam_name[len(pfx) :] if cam_name.startswith(pfx) else cam_name
                if short not in self._world.cameras:
                    self._world.cameras[short] = SimCamera(
                        name=cam_name,
                        camera_id=i,
                        width=self.default_width,
                        height=self.default_height,
                        origin_robot=name,
                    )

            # leave the freshly-added robot in a clean, deterministic
            # zero state (qpos=qvel=ctrl=0) rather than silently settling
            # under gravity for 100 steps. Callers that want a pre-settled
            # pose should call step()/reset() explicitly. This makes
            # `add_robot` -> `get_robot_state` observations meaningful for
            # learning pipelines that expect t=0 to be a canonical start.
            mj.mj_resetData(self._world._model, self._world._data)
            self._world.sim_time = 0.0
            self._world.step_count = 0
            mj.mj_forward(self._world._model, self._world._data)

            # Attach the robot to the mesh as its own peer so the agent can
            # address it directly (e.g. ``robot_mesh tell target=<peer_id>``)
            # rather than going through the sim container. Best-effort: a
            # mesh failure must not prevent ``add_robot`` from returning a
            # working robot. Only attempt when the parent sim is itself
            # already on a mesh.
            self._attach_robot_to_mesh(robot)

            source = f"data_config='{data_config}'" if data_config else os.path.basename(resolved_path)
            mesh_line = f"\n🌐 Mesh peer: {robot.peer_id}" if robot.peer_id else ""
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"🤖 Robot '{name}' added to simulation\n"
                            f"📁 Source: {source} → {os.path.basename(resolved_path)}\n"
                            f"📍 Position: {robot.position}\n"
                            f"🔩 Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                            f"⚡ Actuators: {len(robot.actuator_ids)}\n"
                            f"📷 Cameras: {list(self._world.cameras.keys())}"
                            f"{mesh_line}\n"
                            f"💡 Run policy: action='run_policy', robot_name='{name}'"
                        )
                    }
                ],
            }
        except Exception as e:
            # Clean up on failure
            self._world.robots.pop(name, None)
            logger.error("Failed to add robot '%s': %s", name, e)
            return {"status": "error", "content": [{"text": f"Failed to load: {e}"}]}

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
        if self._world is None or name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{name}' not found."}]}

        # Step 1: cooperatively stop THIS robot's policy if running.
        # Has to happen before the global check so remove_robot works even
        # when the target robot has an active policy (the common case).
        if name in self._policy_threads:
            self._world.robots[name].policy_running = False
            try:
                self._policy_threads[name].result(timeout=5.0)
            except Exception:
                pass
            del self._policy_threads[name]

        # Step 2: after stopping our own, there must be no OTHER policy
        # running - an XML round-trip will invalidate cached IDs everywhere.
        if err := self._require_no_running_policy("remove_robot"):
            return err

        # Pop the robot from the registry BEFORE the rebuild - eject_robot_from_scene
        # rebuilds the spec from the remaining world.robots dict, so the robot
        # we want to drop must no longer be in it.
        robot_obj = self._world.robots[name]
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

        return {"status": "success", "content": [{"text": f"🗑️ Robot '{name}' removed."}]}

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
        if self._world is None or robot_name not in self._world.robots:
            return []
        return list(self._world.robots[robot_name].joint_names)

    def list_robots_info(self) -> dict[str, Any]:
        """Agent-tool action: pretty-printed robot listing.

        Separate from :meth:`list_robots` (which returns ``list[str]`` for
        the SimEngine ABC) because the dispatcher needs a dict-shaped
        response for user display.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if not self._world.robots:
            return {"status": "success", "content": [{"text": "No robots. Use action='add_robot'."}]}

        lines = ["🤖 Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            status = "🟢 running" if robot.policy_running else "⚪ idle"
            lines.append(
                f"  • {name} ({os.path.basename(robot.urdf_path)})\n"
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

        cameras: list[str] = []
        if self._world is not None and self._world._model is not None:
            mj = self._mj
            model = self._world._model
            for i in range(model.ncam):
                cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name:
                    cameras.append(cam_name)
        base["cameras"] = cameras

        if self._world is not None:
            base["sim_time"] = self._world.sim_time
            base["world_created"] = True
        else:
            base["world_created"] = False

        return base

    def get_robot_state(self, robot_name: str | None = None) -> dict[str, Any]:
        """canonical name parameter is ``robot_name``. The router
        accepts ``name`` as an alias (bidirectional) so legacy LLM calls
        keep working, but new tool specs should document only robot_name."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        robot_name = self._resolve_single_robot(robot_name)
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        mj = self._mj
        robot = self._world.robots[robot_name]
        model, data = self._world._model, self._world._data

        # Namespace-aware joint lookup (see add_robot / _apply_sim_action).
        pfx = robot.namespace or ""
        state = {}
        for jnt_name in robot.joint_names:
            jnt_id = -1
            if pfx:
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, pfx + jnt_name)
            if jnt_id < 0:
                jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                state[jnt_name] = {
                    "position": float(data.qpos[model.jnt_qposadr[jnt_id]]),
                    "velocity": float(data.qvel[model.jnt_dofadr[jnt_id]]),
                }

        text = f"🤖 '{robot_name}' state (t={self._world.sim_time:.3f}s):\n"
        for jnt, vals in state.items():
            text += f"{jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"

        return {"status": "success", "content": [{"text": text}, {"json": {"state": state}}]}

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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object to the simulation."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if err := self._require_no_running_policy("add_object"):
            return err
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

        obj = SimObject(
            name=name,
            shape=shape,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            size=size or [0.05, 0.05, 0.05],
            color=color or [0.5, 0.5, 0.5, 1.0],
            mass=mass,
            mesh_path=mesh_path,
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

        return {
            "status": "success",
            "content": [
                {
                    "text": f"📦 '{name}' added: {shape} at {obj.position}, size={obj.size}, {'static' if is_static else f'{mass}kg'}"
                }
            ],
        }

    def remove_object(self, name: str) -> dict[str, Any]:
        if self._world is None or name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        if err := self._require_no_running_policy("remove_object"):
            return err
        del self._world.objects[name]
        # spec-based path: eject_body_from_scene looks up the body in the
        # live MjSpec, deletes it, and recompiles preserving remaining state.
        eject_body_from_scene(self._world, name)
        return {"status": "success", "content": [{"text": f"🗑️ '{name}' removed."}]}

    def move_object(
        self, name: str, position: list[float] | None = None, orientation: list[float] | None = None
    ) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"Object '{name}' not found."}]}
        # Guard: move_object writes qpos + calls mj_forward, racing a running policy.
        if err := self._require_no_running_policy("move_object"):
            return err

        mj = self._mj
        model, data = self._world._model, self._world._data

        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if jnt_id >= 0:
            qpos_addr = model.jnt_qposadr[jnt_id]
            if position:
                data.qpos[qpos_addr : qpos_addr + 3] = position
                self._world.objects[name].position = position
            if orientation:
                data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
                self._world.objects[name].orientation = orientation
            mj.mj_forward(model, data)

        return {"status": "success", "content": [{"text": f"📍 '{name}' moved to {position or 'same'}"}]}

    def list_objects(self) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}

        lines = ["📦 Objects:\n"]
        for name, obj in self._world.objects.items():
            lines.append(f"  • {name}: {obj.shape} at {obj.position}, {'static' if obj.is_static else f'{obj.mass}kg'}")
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
        other cameras and body names - not with object geoms. Duplicate
        camera names are rejected upfront.

        Orientation: ``target`` is baked into the camera's ``xyaxes``
        attribute so the rendered view looks at that point (not just
        forward-facing). Degenerate cases (target == position) error.

        Mounting (``parent_body``): when set to a body name (e.g. a robot's
        gripper such as ``"so101/gripper"``), the camera is attached TO that
        body and rides along with it -- this is how a realistic wrist/gripper
        camera is modelled for SO101/SO100-style data collection. In this
        mode ``position`` and ``target`` are in the body's LOCAL frame. Use
        ``list_robots`` / ``get_robot_state`` to discover body names; robot
        bodies are namespaced ``<robot>/<body>``.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if err := self._require_no_running_policy("add_camera"):
            return err

        # validate position / target shape before we bake them into XML.
        pos = position or [1.0, 1.0, 1.0]
        tgt = target or [0.0, 0.0, 0.0]
        for _lbl, _vec in (("position", pos), ("target", tgt)):
            try:
                if len(_vec) != 3:
                    return {
                        "status": "error",
                        "content": [{"text": f"add_camera: '{_lbl}' must be 3 elements [x,y,z], got {len(_vec)}"}],
                    }
            except TypeError:
                return {"status": "error", "content": [{"text": f"add_camera: '{_lbl}' must be a list of 3 numbers"}]}
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
            if mj.mj_name2id(model, mj.mjtObj.mjOBJ_BODY, parent_body) < 0:
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
                                f"Robot bodies are namespaced '<robot>/<body>'. Available bodies: {available}"
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
        return {"status": "success", "content": [{"text": f"📷 Camera '{name}' added at {cam.position}{mount_note}"}]}

    def remove_camera(self, name: str) -> dict[str, Any]:
        """Remove a named camera from the live scene.

        Pops the Python-side registry entry and then deletes the camera
        from the MjSpec via :func:`SpecBuilder.remove_camera` so future
        renders/compiles no longer see it.
        """
        if self._world is None or name not in self._world.cameras:
            return {"status": "error", "content": [{"text": f"Camera '{name}' not found."}]}
        if err := self._require_no_running_policy("remove_camera"):
            return err
        cam = self._world.cameras.pop(name)

        spec = self._world._backend_state.get("spec")
        if spec is not None:
            # Use the namespaced MuJoCo name if we have it (camera came from
            # a robot's URDF), else the short name.
            mj_name = cam.name or name
            SpecBuilder.remove_camera(spec, mj_name)
            # Recompile so nbody/ncam in _model match the new spec.
            try:
                self._world._model, self._world._data = spec.recompile(self._world._model, self._world._data)
                try:
                    self._world._backend_state["xml"] = spec.to_xml()
                except Exception:
                    pass
            except (ValueError, RuntimeError) as e:
                logger.warning("remove_camera recompile failed: %s", e)

        return {"status": "success", "content": [{"text": f"🗑️ Camera '{name}' removed."}]}

    # Simulation Control

    _MAX_STEPS_PER_CALL = 100_000  # Hard ceiling to prevent unbounded lock hold.
    _STEPS_PER_BATCH = 1000  # Release lock every N steps for cancellation.

    def step(self, n_steps: int = 1) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        # reject negative, accept zero as no-op
        if not isinstance(n_steps, int):
            try:
                n_steps = int(n_steps)
            except (TypeError, ValueError):
                return {
                    "status": "error",
                    "content": [{"text": f"step: n_steps must be an integer, got {type(n_steps).__name__}"}],
                }
        if n_steps < 0:
            return {"status": "error", "content": [{"text": f"step: n_steps must be >= 0, got {n_steps}"}]}
        if n_steps == 0:
            return {
                "status": "success",
                "content": [
                    {"text": f"⏩ +0 steps (no-op) | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}
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
        # Process in batches, releasing lock between batches so stop_policy
        # and other actions can interleave on long runs.
        remaining = n_steps
        while remaining > 0:
            batch = min(remaining, self._STEPS_PER_BATCH)
            with self._lock:
                for _ in range(batch):
                    mj.mj_step(self._world._model, self._world._data)
                self._world.sim_time = self._world._data.time
                self._world.step_count += batch
            remaining -= batch
        return {
            "status": "success",
            "content": [
                {"text": f"⏩ +{n_steps} steps | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}
            ],
        }

    def reset(self) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        # reset during a running policy races mj_step -> SEGFAULT risk
        if err := self._require_no_running_policy("reset"):
            return err
        mj = self._mj
        with self._lock:
            mj.mj_resetData(self._world._model, self._world._data)
            self._world.sim_time = 0.0
            self._world.step_count = 0
            # Flip policy_running flag inside the lock so a racing worker
            # thread cannot slip in one more mj_step between reset and flag
            # flip.
            for r in self._world.robots.values():
                r.policy_running = False
                r.policy_steps = 0
        return {"status": "success", "content": [{"text": "🔄 Reset to initial state."}]}

    def get_state(self) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        lines = [
            "🌍 Simulation State",
            f"🕐 t={self._world.sim_time:.4f}s (step {self._world.step_count})",
            f"⚙️ dt={self._world.timestep}s | 🌐 g={self._world.gravity}",
            f"🤖 Robots: {len(self._world.robots)} | 📦 Objects: {len(self._world.objects)} | 📷 Cameras: {len(self._world.cameras)}",
        ]
        if self._world._model:
            lines.append(
                f"🦴 Bodies: {self._world._model.nbody} | 🔩 Joints: {self._world._model.njnt} | ⚡ Actuators: {self._world._model.nu}"
            )
        if self._world._backend_state.get("recording", False):
            lines.append(f"🔴 Recording: {len(self._world._backend_state['trajectory'])} steps")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def destroy(self) -> dict[str, Any]:
        """Destroy the world and release all resources.

        Delegates to cleanup() which properly joins running policy Futures
        before nulling self._world — prevents SIGSEGV from workers holding
        stale model/data pointers.
        """
        if self._world is None:
            return {"status": "success", "content": [{"text": "No world to destroy."}]}
        self.cleanup()
        return {"status": "success", "content": [{"text": "🗑️ World destroyed."}]}

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
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        # set_gravity during a running policy races the worker thread
        if err := self._require_no_running_policy("set_gravity"):
            return err
        # validate length/dtype before numpy broadcast
        if isinstance(gravity, (int, float)):
            gravity = [0.0, 0.0, float(gravity)]
        try:
            if len(gravity) != 3:
                return {
                    "status": "error",
                    "content": [
                        {"text": f"set_gravity: 'gravity' must be a 3-element list [x,y,z], got {len(gravity)}"}
                    ],
                }
            gravity = [float(g) for g in gravity]
        except (TypeError, ValueError) as e:
            return {
                "status": "error",
                "content": [{"text": f"set_gravity: 'gravity' must be a 3-element list of numbers ({e})"}],
            }
        if not all(math.isfinite(g) for g in gravity):
            return {
                "status": "error",
                "content": [{"text": f"set_gravity: all components must be finite, got {gravity}"}],
            }
        with self._lock:
            self._world._model.opt.gravity[:] = gravity
            self._world.gravity = gravity
        return {"status": "success", "content": [{"text": f"🌐 Gravity: {gravity}"}]}

    def set_timestep(self, timestep: float) -> dict[str, Any]:
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        if err := self._require_no_running_policy("set_timestep"):
            return err
        # reject non-positive; warn on huge values
        try:
            timestep = float(timestep)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "content": [{"text": f"set_timestep: must be a positive number, got {timestep!r}"}],
            }
        if not math.isfinite(timestep) or timestep <= 0:
            return {
                "status": "error",
                "content": [{"text": f"set_timestep: must be a finite positive number, got {timestep}"}],
            }
        warn = ""
        if timestep > 0.1:
            warn = f" ⚠️ unusually large timestep (>{0.1}s); physics may be unstable"
        with self._lock:
            self._world._model.opt.timestep = timestep
            self._world.timestep = timestep
        return {"status": "success", "content": [{"text": f"⏱️ Timestep: {timestep}s ({1 / timestep:.0f}Hz){warn}"}]}

    # Viewer

    def open_viewer(self) -> dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "No simulation to view."}]}
        from strands_robots.simulation.mujoco.backend import _mujoco_viewer

        if _mujoco_viewer is None:
            return {"status": "error", "content": [{"text": "mujoco.viewer not available."}]}
        if self._viewer_handle is not None:
            return {"status": "success", "content": [{"text": "👁️ Viewer already open."}]}
        try:
            self._viewer_handle = _mujoco_viewer.launch_passive(self._world._model, self._world._data)
            return {"status": "success", "content": [{"text": "👁️ Interactive viewer opened."}]}
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
        self._close_viewer()
        return {"status": "success", "content": [{"text": "👁️ Viewer closed."}]}

    # URDF Registry

    def list_urdfs(self) -> dict[str, Any]:
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
            "content": [{"text": f"📋 Registered '{data_config}' → {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"}],
        }

    # Introspection

    def get_features(self, robot_name: str | None = None) -> dict[str, Any]:
        """Describe the simulation's joints / actuators / cameras / robots.

        If ``robot_name`` is given, the joint / actuator / camera listings
        are restricted to that robot (its namespaced MuJoCo names).  The
        ``robots`` map is also filtered to just that entry.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

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
            if robot_name not in self._world.robots:
                return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
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
            "🔍 Simulation Features",
            f"🦴 Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"⚡ Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
            f"📷 Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
            f"⏱️ Timestep: {model.opt.timestep}s ({1 / model.opt.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(
                f"🤖 {rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})"
            )

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}],
        }

    # AgentTool Interface

    def __call__(self, action: str = "", **kwargs: Any) -> dict[str, Any]:
        """Dispatch an action directly: ``sim(action="render", camera_name="topdown")``.

        This makes the Simulation usable as a plain callable in addition to
        being a Strands ``AgentTool``. It mirrors the agent-facing dispatch
        path (``_dispatch_action``): the same validation, field-aliasing, and
        per-action method routing apply, and the return value is the standard
        ``{"status", "content"}`` dict.

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
        return self._dispatch_action(action.strip(), kwargs)

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "simulation"

    def _require_world(self) -> dict[str, Any] | None:
        """Return unified 'no world' error or None if world is live.

        Replaces scattered ``"No simulation."`` / ``"No world."`` strings. Every
        action that touches ``self._world`` / ``self._world._model`` /
        ``self._world._data`` should call this first.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {
                "status": "error",
                "content": [{"text": ("No world. Call create_world (or load_scene) first.")}],
            }
        return None

    def _prune_done_futures(self) -> None:
        """Drop completed Future refs from self._policy_threads.

        Without this, list_policies_running and stale-active checks see
        historical entries forever (see GH #120).
        """
        done = [k for k, f in self._policy_threads.items() if f.done()]
        for k in done:
            self._policy_threads.pop(k, None)

    def _active_policy_robots(self) -> list[str]:
        """Names of robots with a live (not-done) policy Future.

        Prunes stale entries as a side-effect so the returned list is
        authoritative. Callers can introspect via ``list_policies_running``.
        """
        self._prune_done_futures()
        return list(self._policy_threads.keys())

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
            fut = self._policy_threads.get(robot_name)
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
                "Same Policy ABC as real robot control - sim ↔ real with zero code changes. "
                "Actions (61 total): "
                "[World] create_world, load_scene, reset, get_state, destroy, export_xml; "
                "[Robots] add_robot, remove_robot, list_robots, get_robot_state; "
                "[Objects] add_object, remove_object, move_object, list_objects; "
                "[Cameras] add_camera, remove_camera; "
                "[Policy] run_policy, start_policy, stop_policy, eval_policy, replay_episode, list_policies_running; "
                "[Rendering] render, render_depth, render_all, open_viewer, close_viewer; "
                "[Physics] step, set_gravity, set_timestep, set_joint_positions, set_joint_velocities, "
                "apply_force, get_contacts, get_contact_forces, get_body_state, get_energy, "
                "get_total_mass, get_sensor_data, get_jacobian, get_mass_matrix, inverse_dynamics, "
                "forward_kinematics, save_state, load_state, set_body_properties, set_geom_properties; "
                "[Scene MJCF] replace_scene_mjcf, patch_scene_mjcf, raycast, multi_raycast; "
                "[Recording] start_recording, stop_recording, get_recording_status, "
                "start_cameras_recording, stop_cameras_recording, get_cameras_recording_status; "
                "[Randomize] randomize; "
                "[Registry] list_urdfs, register_urdf, get_features. "
                "Call destroy() at session end to release resources."
            ),
            "inputSchema": {"json": _TOOL_SPEC_SCHEMA},
        }

    async def stream(
        self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
    ) -> AsyncGenerator[ToolResultEvent, None]:
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            result = self._dispatch_action(input_data.get("action", ""), input_data)
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
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}
        robot_name = self._resolve_single_robot(robot_name)
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}

        # Per-robot gate: another policy running on a DIFFERENT robot is fine.
        if err := self._require_no_running_policy("start_policy", robot_name=robot_name):
            return err

        # Concurrent multi-robot policies run on disjoint ctrl slices (physics
        # serialized by _lock). For SYNCHRONIZED multi-robot *recording* (both
        # robots captured in one merged frame per step), use run_multi_policy
        # instead — independent start_policy threads each step physics and write
        # frames separately, which is correct for live control but interleaves
        # for a shared recorder. start_policy while recording is left to the
        # caller's intent (run_multi_policy is the recommended recording path).
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
        )
        self._policy_threads[robot_name] = future

        return {
            "status": "success",
            "content": [{"text": f"🚀 Policy started on '{robot_name}' (async)"}],
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
        if world is None or robot_name not in world.robots:
            return None

        robot = world.robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0

        lock = self._lock

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            # Cooperative cancellation: stop_policy flips this flag.
            if not robot.policy_running:
                raise CooperativeStop(f"Policy stopped on '{robot_name}'")

            robot.policy_steps = step + 1

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
                        # In multi-robot scenes start_recording() declares
                        # the dataset schema with per-robot-prefixed joint ids
                        # (``alice__shoulder_pan``) so each agent has unique state/
                        # action columns. But _get_sim_observation() and the action
                        # dict use SHORT joint names (``shoulder_pan``). Without
                        # remapping here, add_frame() looks up the prefixed schema
                        # keys, finds nothing, and writes all-zero state/action
                        # vectors silently. Prefix scalar obs + action keys to match
                        # the schema. Camera values (ndarray) keep their (already
                        # namespaced) names — dataset_recorder normalizes '/'→'__'.
                        if len(world.robots) > 1:
                            obs_keyed = {
                                (k if isinstance(v, np.ndarray) else f"{robot_name}__{k}"): v
                                for k, v in observation.items()
                            }
                            act_keyed = {f"{robot_name}__{k}": v for k, v in action.items()}
                            rec.add_frame(observation=obs_keyed, action=act_keyed, task=instruction)
                        else:
                            rec.add_frame(observation=observation, action=action, task=instruction)

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
    ) -> dict[str, Any]:
        """MuJoCo ``run_policy`` override: pre-flight world check + graceful stop.

        Delegates to :meth:`SimEngine.run_policy` but clears the MuJoCo
        ``policy_running`` flag in a ``finally`` clause and swallows
        ``_PolicyStopped`` (which the ``on_frame`` hook raises on user
        cancellation) into a normal "policy stopped" result.

        forwards ``n_steps`` / ``max_steps`` to the base so LLM callers
        can specify horizon in steps rather than wall-clock seconds.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

        robot_name = self._resolve_single_robot(robot_name)

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
            )
        finally:
            if self._world is not None and robot_name in self._world.robots:
                self._world.robots[robot_name].policy_running = False

    def run_multi_policy(
        self,
        policies: dict[str, "Policy"],
        instructions: dict[str, str] | str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int | dict[str, int] = 8,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Drive MULTIPLE robots with their own policies in a SINGLE
        synchronized control loop, recording ALL robots into ONE merged frame
        per timestep.

        This is the correct path for concurrent multi-robot data collection
        (e.g. two SO-100 arms doing a handover, or a bimanual setup). Unlike
        launching two independent ``start_policy`` threads — which each step
        physics and call ``add_frame`` separately, so the shared recorder
        receives interleaved single-robot frames (B4) — this loop:

        1. Observes every robot (joints) and renders all cameras ONCE.
        2. Queries each robot's policy for its action.
        3. Applies every robot's ctrl writes, then steps physics ONCE.
        4. Records ONE frame containing ALL robots' prefixed state/action
           (``alice__shoulder_pan`` …) plus all camera images.

        So a 2-robot dataset has both arms co-observed in every frame — usable
        for bimanual / multi-agent policy training.

        Args:
            policies: Mapping ``{robot_name: Policy}``. Each robot must exist in
                the scene. Order defines the state/action column order.
            instructions: Either a single instruction string applied to all
                robots, or a ``{robot_name: instruction}`` mapping. The frame's
                recorded task is the first robot's instruction (LeRobot stores
                one task per frame).
            duration: Episode length in seconds (steps = duration × freq).
            control_frequency: Target Hz for policy action queries / physics.
            action_horizon: How many actions to consume from each policy's
                returned chunk before re-querying it (open-loop chunk
                execution, mirrors ``run_policy``). Either a single int applied
                to all robots, or a ``{robot_name: horizon}`` mapping for
                per-robot control. A robot's policy is only re-queried when its
                action queue drains — so an expensive VLA emitting a 30-action
                chunk with ``action_horizon=30`` runs inference ~once per 30
                steps instead of every step. Physics still advances ONE step per
                loop iteration, keeping all robots phase-aligned regardless of
                their individual re-query cadence.
            n_steps: Alternate horizon in steps (overrides duration when set).
            max_steps: Legacy alias for ``n_steps``.

        Returns:
            Standard status dict with per-robot step counts.
        """
        import numpy as np

        from strands_robots._async_utils import _resolve_coroutine

        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}
        if not policies:
            return {"status": "error", "content": [{"text": "run_multi_policy: 'policies' is empty."}]}

        # Validate every robot exists.
        for rname in policies:
            if rname not in self._world.robots:
                return {"status": "error", "content": [{"text": f"Robot '{rname}' not found."}]}

        # Reject if any of these robots already has a running async policy
        # (would double-step physics on that robot).
        self._prune_done_futures()
        busy = [r for r in policies if (f := self._policy_threads.get(r)) is not None and not f.done()]
        if busy:
            names = ", ".join(f"'{n}'" for n in busy)
            return {
                "status": "error",
                "content": [{"text": f"run_multi_policy: policy already running on {names}. Stop it first."}],
            }

        # Normalize instructions to a per-robot mapping.
        if isinstance(instructions, str):
            instr_map = {r: instructions for r in policies}
        else:
            instr_map = {r: instructions.get(r, "") for r in policies}

        # LeRobot stores ONE task string per frame. If the caller supplied
        # distinct per-robot instructions (e.g. {alice: 'pour', bob: 'catch'})
        # only the first robot's task is recorded -- a downstream pipeline that
        # splits by task string would mis-attribute the other robots' frames.
        # Surface this as a known limitation rather than losing it silently.
        _distinct_tasks = {t for t in instr_map.values() if t}
        if len(_distinct_tasks) > 1:
            logger.warning(
                "run_multi_policy: %d distinct per-robot instructions supplied (%s) but "
                "LeRobot records one task per frame; only '%s' (robot '%s') will be stored. "
                "Per-robot task columns are not yet supported.",
                len(_distinct_tasks),
                sorted(_distinct_tasks),
                instr_map[next(iter(policies))],
                next(iter(policies)),
            )

        # Resolve horizon (n_steps / max_steps override duration).
        if n_steps is None and max_steps is not None:
            n_steps = int(max_steps)
        if n_steps is not None:
            if n_steps <= 0 or control_frequency <= 0:
                return {
                    "status": "error",
                    "content": [{"text": "run_multi_policy: n_steps and control_frequency must be > 0."}],
                }
            duration = float(n_steps) / float(control_frequency)

        # Normalize action_horizon to a per-robot mapping. >=1 actions are
        # consumed open-loop from each policy's chunk before re-querying.
        if isinstance(action_horizon, dict):
            horizon_map = {r: max(1, int(action_horizon.get(r, 8))) for r in policies}
        else:
            h = max(1, int(action_horizon))
            horizon_map = {r: h for r in policies}

        # Bind robot_state_keys for each policy (per-robot joint names).
        for rname, pol in policies.items():
            try:
                pol.set_robot_state_keys(self.robot_joint_names(rname))
            except Exception as exc:  # noqa: BLE001 - non-fatal, mirrors run_policy defensiveness
                logger.debug("set_robot_state_keys(%s) failed: %s", rname, exc)

        multi_robot = len(self._world.robots) > 1
        recorder = self._world._backend_state.get("dataset_recorder")
        recording = bool(self._world._backend_state.get("recording", False)) and recorder is not None

        # Whether ANY policy needs images (renders are expensive; skip if none
        # need them AND we're not recording — recording always needs frames).
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
                # chunk — so an expensive VLA runs inference once per horizon
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
                        # Buffer up to this robot's horizon; re-query when drained.
                        for a in acts[: horizon_map[rname]]:
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
                        # write ctrl only (no per-robot mj_step) — we step once
                        # after every robot's ctrl is set.
                        robot = self._world.robots[rname]
                        pfx = robot.namespace or ""
                        self._apply_action_by_name(self._world._model, self._world._data, act, pfx, mj)
                    mj.mj_step(self._world._model, self._world._data)
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
                    # per-robot); keep ndarray keys as-is.
                    merged_obs.update(camera_imgs)
                    task = instr_map[next(iter(policies))]
                    # add_frame writes to LeRobot's image-writer queue and parquet
                    # buffer; it does not touch MuJoCo model/data. The consistent
                    # state snapshot was already taken under self._lock in steps 1
                    # and 3, and merged_obs/merged_act are plain copies, so holding
                    # the physics lock across frame writeout would needlessly starve
                    # other lock holders (viewer sync, concurrent tool reads).
                    recorder.add_frame(observation=merged_obs, action=merged_act, task=task)

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
                if rname in self._world.robots:
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
        return {"status": "success", "content": [{"text": text}], "steps": step_count}

    # Action name aliases (tool-action -> method-name)
    _ACTION_ALIASES = {
        "list_robots": "list_robots_info",
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

    # Params the router passes through but not every method declares.
    # These are used for cross-cutting concerns (e.g. video on run_policy)
    # and must not be reported as "unknown" by the router.
    _ROUTER_PASSTHROUGH = {"action"}

    # Vector params with expected length (for dimension validation before
    # numpy/MuJoCo sees them). Length 3 = xyz unless noted.
    _VECTOR_PARAM_LENGTHS: dict[str, int] = {
        "position": 3,
        "target": 3,
        "origin": 3,
        "force": 3,
        "torque": 3,
        "torque_vec": 3,
        "gravity": 3,
        "direction": 3,
        "point": 3,
        "orientation": 4,  # quaternion (w,x,y,z)
        "color": 4,  # rgba
    }

    def _validate_and_build_kwargs(
        self, action: str, method_name: str, sig: inspect.Signature, remapped: dict[str, Any]
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Validate input against method signature; return (kwargs, error_result).

        Exactly one of the tuple elements is non-None.
        """
        # Strip self + VAR_POSITIONAL (*args) + VAR_KEYWORD (**kwargs) for signature
        # introspection; **kwargs methods accept arbitrary inputs, so we skip the
        # unknown-key check for them.
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
            valid_sorted = sorted(method_param_names - {"action"})
            return None, {
                "status": "error",
                "content": [
                    {"text": (f"Unknown parameter '{unknown[0]}' for action '{action}'. Valid: {valid_sorted}")}
                ],
            }

        # 2) Vector dimension validation (applies before method runs)
        for vparam, expected_len in self._VECTOR_PARAM_LENGTHS.items():
            if vparam not in remapped:
                continue
            val = remapped[vparam]
            if val is None:
                continue
            if not hasattr(val, "__len__"):
                return None, {
                    "status": "error",
                    "content": [{"text": f"Parameter '{vparam}' must be a list of {expected_len} numbers."}],
                }
            if len(val) != expected_len:
                return None, {
                    "status": "error",
                    "content": [
                        {"text": (f"Parameter '{vparam}' must be a list of {expected_len} numbers, got {len(val)}.")}
                    ],
                }
            for i, component in enumerate(val):
                if not isinstance(component, (int, float)) or isinstance(component, bool):
                    return None, {
                        "status": "error",
                        "content": [
                            {"text": (f"Parameter '{vparam}'[{i}] must be numeric, got {type(component).__name__}.")}
                        ],
                    }

        # 3) Build kwargs + check required params
        kwargs: dict[str, Any] = {}
        for param_name, param in named_params.items():
            if param_name == "name" and "name" not in remapped and "robot_name" in remapped:
                kwargs["name"] = remapped["robot_name"]
            elif param_name == "robot_name" and "robot_name" not in remapped and "name" in remapped:
                kwargs["robot_name"] = remapped["name"]
            elif param_name in remapped:
                kwargs[param_name] = remapped[param_name]
            elif param.default is inspect.Parameter.empty:
                return None, {
                    "status": "error",
                    "content": [{"text": f"Action '{action}' requires parameter '{param_name}'."}],
                }

        return kwargs, None

    def _dispatch_action(self, action: str, d: dict[str, Any]) -> dict[str, Any]:
        """Route action to the matching method with full input validation.

        Validation layer:
          * unknown top-level params are rejected with a friendly message,
          * missing required params produce a "requires parameter X" error
            (no raw Python ``TypeError``),
          * vector params have length + numeric dtype checked before the
            value reaches numpy / MuJoCo.

        Policy-provider kwargs are nested under ``policy_config`` (never
        top-level) so the dispatcher stays backend-agnostic.
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

        # Fold flat video keys into `video` dict for run_policy/start_policy.
        if action in ("run_policy", "start_policy") and "video" not in remapped:
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

        kwargs, err = self._validate_and_build_kwargs(action, method_name, sig, remapped)
        if err is not None:
            return err
        assert kwargs is not None
        # All dispatched actions are serialized under self._lock (RLock).
        # This is the single chokepoint that prevents concurrent reads/writes
        # to MuJoCo model/data from the agent thread while a PolicyRunner
        # worker is mid-mj_step. Individual methods that also acquire the
        # lock are harmless (RLock is reentrant on the same thread).
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
        if self._world is None or robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"Robot '{robot_name}' not found."}]}
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
                "content": [{"text": "⚪ No policies running."}],
            }
        robot_lines = "\n".join(f"  • 🟢 {n}" for n in active)
        return {
            "status": "success",
            "content": [{"text": f"🟢 Active policies ({len(active)}):\n{robot_lines}"}],
        }

    # Cleanup

    # Default cleanup shutdown timeout (seconds). A policy worker might be
    # mid-step when cleanup is called; give it bounded time to see the
    # cooperative-stop flag and exit cleanly before we null the world and
    # its in-flight ``mj_step`` segfaults on a nulled ``_model``/``_data``.
    # Override in tests via ``cleanup(policy_stop_timeout=...)`` if needed.
    _DEFAULT_POLICY_STOP_TIMEOUT = 5.0

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
             and tear down renderers / the viewer / the executor.

        Args:
            policy_stop_timeout: Seconds to wait per active policy future.
                ``None`` (default) uses
                ``_DEFAULT_POLICY_STOP_TIMEOUT`` (5s). Set to a small value
                in tests that want fast teardown.
        """
        # Detach from the mesh network first (if attached). A truthy
        # ``self.mesh`` is any object exposing ``.stop()``; falsy values
        # (the default) mean this Simulation never joined a mesh and
        # there's nothing to release. Done BEFORE stopping policies so
        # peer-visible state is torn down cleanly even if the policy
        # teardown below hits the fallback ``wait=False`` path.
        #
        # PR #101 follow-up: each robot added via ``add_robot`` may have
        # its own per-peer mesh (see ``_attach_robot_to_mesh``). Stop those
        # FIRST so external peers see them leave before the sim container
        # itself goes down — leaving the inverse order ("sim drops, robots
        # linger") would create zombie peer entries in remote ``get_peers``
        # results until their heartbeats expire.
        if self._world is not None:
            for r in list(self._world.robots.values()):
                self._detach_robot_from_mesh(r)
        if self.mesh:
            self.mesh.stop()

        timeout = policy_stop_timeout if policy_stop_timeout is not None else self._DEFAULT_POLICY_STOP_TIMEOUT

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

        # Step 3: now it's safe to null the world. Any worker still alive
        # at this point has already escaped MuJoCo (we've confirmed via
        # fut.result()), so a nulled _model / _data is no longer racy.
        if self._world:
            self._world = None

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

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception:
            pass


# Backward-compatible aliases (PR #85 shipped as ``Simulation``)
Simulation = MuJoCoSimEngine
MuJoCoSimulation = MuJoCoSimEngine
