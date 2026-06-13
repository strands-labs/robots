"""Simulation ABC - backend-agnostic interface for all simulation engines.

Every simulation backend (MuJoCo, Isaac, Newton) implements this interface.
Agent tools and the Robot() factory interact through these methods only -
they never touch backend-specific APIs directly.

Usage::

    from strands_robots.simulation import Simulation  # returns MuJoCo by default

    # Or explicitly:
    from strands_robots.simulation.mujoco import MuJoCoSimulation

    # Future:
    from strands_robots.simulation.isaac import IsaacSimulation
    from strands_robots.simulation.newton import NewtonSimulation
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.policies import Policy

# PolicyRunner and VideoConfig are used by run_policy / replay / eval_policy.
# We could defer these with inline lazy imports (and historically did), but
# policy_runner.py only imports `SimEngine` from base under TYPE_CHECKING so
# the runtime cycle doesn't actually exist. Keep the imports at module level
# to break the AST-visible cycle that static analysers flag.
#
# Note (#191): we deliberately do NOT import ``OnFrame`` here, even under
# ``TYPE_CHECKING`` — CodeQL's ``py/unsafe-cyclic-import`` rule walks
# ``TYPE_CHECKING`` blocks too and would flag the static cycle (
# policy_runner.py imports SimEngine from base under TYPE_CHECKING,
# so importing OnFrame from policy_runner here closes the loop in the
# AST). Instead, we reference ``OnFrame`` in the ``evaluate_benchmark``
# signature as a *string* annotation; ``from __future__ import
# annotations`` (already in effect) makes that a no-op at runtime.
from strands_robots.simulation.policy_runner import PolicyRunner, VideoConfig

logger = logging.getLogger(__name__)


class SimEngine(ABC):
    """Abstract base class for simulation engines.

    Defines the contract that all backends (MuJoCo, Isaac, Newton) must
    implement. This is the *programmatic* API - the AgentTool layer
    wraps it with tool_spec/stream for LLM access.

    Method categories:

    **Required** (``@abstractmethod``): Core simulation loop - world
    lifecycle, entity management, observation/action, rendering, robot
    discovery. Every physics engine must implement these to be usable.

    **Provided** (concrete base-class methods): Policy orchestration
    (``run_policy`` / ``start_policy`` / ``replay_episode`` / ``eval_policy``)
    is implemented once in this ABC as a facade over the abstract primitives.
    Backends inherit them for free by implementing the primitives. They
    *may* override for backend-specific optimisations (e.g. GPU-batched
    policy inference on Isaac).

    **Optional** (default raises ``NotImplementedError``): Higher-level
    features - scene loading, domain randomization, contact queries.
    Backends opt in by overriding only what they support.

    Lifecycle::

        sim = SomeEngine()
        sim.create_world()
        sim.add_robot("so100", data_config="so100")
        sim.add_object("cube", shape="box", position=[0.3, 0, 0.05])

        # Control loop
        obs = sim.get_observation("so100")
        sim.send_action({"joint_0": 0.5}, robot_name="so100")
        sim.step(n_steps=10)

        # Render
        result = sim.render(camera_name="default")

        # Cleanup
        sim.destroy()
    """

    def _resolve_single_robot(self, robot_name: str | None) -> str:
        """Resolve an optional robot name to a concrete one.

        None + exactly one robot -> that robot.
        None + zero robots -> ValueError.
        None + many robots -> ValueError listing the candidates so the
        caller can recover in zero extra calls.

        Args:
            robot_name: Explicit robot name (returned unchanged) or None.

        Returns:
            Resolved robot name string.

        Raises:
            ValueError: When robot_name is None and the resolution is
                ambiguous or impossible.
        """
        if robot_name is not None:
            return robot_name
        names = self.list_robots()
        if len(names) == 1:
            return names[0]
        if len(names) == 0:
            raise ValueError("No robots registered in the simulation. Add a robot first (add_robot or Robot factory).")
        raise ValueError(f"Multiple robots registered; specify robot_name. Available: {names}")

    # World lifecycle

    @abstractmethod
    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
    ) -> dict[str, Any]:
        """Create a new simulation world."""
        ...

    @abstractmethod
    def destroy(self) -> dict[str, Any]:
        """Destroy the simulation world and release resources."""
        ...

    @abstractmethod
    def reset(self) -> dict[str, Any]:
        """Reset simulation to initial state."""
        ...

    @abstractmethod
    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps."""
        ...

    @abstractmethod
    def get_state(self) -> dict[str, Any]:
        """Get full simulation state summary."""
        ...

    # Robot management

    @abstractmethod
    def add_robot(
        self,
        name: str,
        urdf_path: str | None = None,
        data_config: str | None = None,
        position: list[float] | None = None,
        orientation: list[float] | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation."""
        ...

    @abstractmethod
    def remove_robot(self, name: str) -> dict[str, Any]:
        """Remove a robot from the simulation."""
        ...

    @abstractmethod
    def list_robots(self) -> list[str]:
        """Return ordered list of robot names currently in the world.

        Used by the backend-agnostic ``PolicyRunner`` to resolve a
        default robot when the caller omits ``robot_name``.
        """
        ...

    @abstractmethod
    def robot_joint_names(self, robot_name: str) -> list[str]:
        """Return ordered joint names for ``robot_name``.

        Used by ``Policy.set_robot_state_keys`` and by
        ``PolicyRunner.replay`` to map dataset action-vector indices to
        named joints. Order must match the backend's action ordering.
        """
        ...

    # Object management

    @abstractmethod
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
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Add an object to the scene."""
        ...

    @abstractmethod
    def remove_object(self, name: str) -> dict[str, Any]:
        """Remove an object from the scene."""
        ...

    # Observation / Action

    @abstractmethod
    def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
        """Get full observation for a robot: joint state + all attached cameras.

        Unified observation consumed by :class:`Policy` and
        :class:`~strands_robots.simulation.policy_runner.PolicyRunner`.
        Backends MUST return a dict with the following schema; extra keys
        are allowed.

        Schema:
            - ``"<joint_name>"`` (float): One entry per joint on the robot,
              keyed by the *short* joint name (e.g. ``"shoulder_pan"``).
              The schema is stable regardless of multi-robot namespacing
              at the physics-engine level.
            - ``"<camera_name>"`` (np.ndarray): One RGB uint8 frame per
              camera associated with the robot, keyed by camera name.
              Shape ``(H, W, 3)``. Cameras whose render fails MAY be
              omitted; joint state MUST still be returned.

        Single-camera rendering is :meth:`render`'s job, not this method's.
        For batched multi-robot observation (future Isaac / Newton), add a
        separate ``get_observations(robot_names)`` method - do NOT extend
        this one.

        Args:
            robot_name: Which robot to observe. If ``None`` and exactly one
                robot exists, that robot is used; otherwise returns ``{}``.

        Returns:
            Observation dict per schema above. Returns ``{}`` if the world
            is not yet created or ``robot_name`` is unknown.
        """
        ...

    @abstractmethod
    def send_action(self, action: dict[str, Any], robot_name: str | None = None, n_substeps: int = 1) -> None:
        """Apply action and advance physics by n_substeps.

        Contract: each call writes actuator/ctrl values and then runs
        ``n_substeps`` physics steps (e.g. mj_step). PolicyRunner.run()
        relies on this — it calls send_action once per control step and
        does NOT call sim.step() separately.

        Backends are responsible for internal thread-safety (e.g.
        MuJoCo acquires self._lock here). PolicyRunner does not manage
        locks.
        """
        ...

    def physics_timestep(self) -> float | None:
        """Return the physics integration timestep in seconds, or ``None``.

        Used by :class:`PolicyRunner` to convert a policy's ``control_frequency``
        into the number of physics substeps per control step
        (``round(1 / control_frequency / physics_timestep)``) so a
        position-servo robot actually tracks each action's target before the
        next action overwrites ``ctrl``. Backends that cannot report a fixed
        timestep return ``None`` and the runner falls back to ``n_substeps=1``.
        """
        return None

    # Rendering

    @abstractmethod
    def render(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> dict[str, Any]:
        """Render a camera view.

        Returns dict with ``"image"`` key (numpy array, RGB uint8) and
        optional ``"depth"`` key (float32 depth map). Resolution comes
        from camera config unless ``width``/``height`` are given.
        """
        ...

    # Policy orchestration (concrete facade, not abstract)

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
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
        max_onframe_failures: int | None = None,
        control_substeps: int | None = None,
    ) -> dict[str, Any]:
        """Run a policy loop in the simulation (blocking).

        Default implementation delegates to the backend-agnostic
        :class:`~strands_robots.simulation.policy_runner.PolicyRunner`.
        Backends MAY override for backend-specific optimisations
        (e.g. GPU-batched policy inference on Isaac).

        Args:
            robot_name: Robot to control.
            policy_provider: Name passed to
                :func:`strands_robots.policies.create_policy`.
            policy_config: Opaque dict of provider-specific kwargs
                (``observation_mapping``, ``action_mapping``, ``host``,
                ``port``, ``api_token``, ``pretrained_name_or_path``,
                ``trust_remote_code``, ``actions_per_step``,
                ``use_processor``, ``processor_overrides``, ``device``,
                ...). Forwarded verbatim to ``create_policy``.
            instruction: Natural-language instruction for the policy.
            duration: Wall-clock seconds to run.
            control_frequency: Target Hz for policy queries.
            action_horizon: Max actions per policy call.
            fast_mode: Skip real-time sleep between steps.
            video: Optional video-recording config dict. Accepted keys:
                ``path`` (str, output MP4 - required to enable recording),
                ``fps`` (int, default 30), ``camera`` (str, default backend
                default), ``width`` (int, default 640), ``height`` (int,
                default 480). See :class:`~strands_robots.simulation.policy_runner.VideoConfig`.
                For extension points beyond video (custom telemetry,
                dataset recording), backends plug into
                ``PolicyRunner.run``'s ``on_frame`` hook via
                :meth:`_make_run_policy_hook`.

        Returns:
            Standard status dict.
        """
        from strands_robots.policies import create_policy

        robot_name = self._resolve_single_robot(robot_name)

        # accept n_steps (or legacy max_steps) as an alternate horizon
        # specification. duration = n_steps / control_frequency. If both
        # are passed, n_steps wins (primary per DoD).
        if n_steps is None and max_steps is not None:
            n_steps = int(max_steps)
        if n_steps is not None:
            if n_steps <= 0:
                return {
                    "status": "error",
                    "content": [{"text": f"run_policy: n_steps must be > 0, got {n_steps}."}],
                }
            if control_frequency <= 0:
                return {
                    "status": "error",
                    "content": [{"text": "run_policy: control_frequency must be > 0 when n_steps is used."}],
                }
            duration = float(n_steps) / float(control_frequency)

        if robot_name not in self.list_robots():
            return {
                "status": "error",
                "content": [{"text": f"Robot '{robot_name}' not found."}],
            }

        if policy_object is not None:
            # Pre-built policy path - skip the expensive create_policy call.
            # Caller is responsible for policy.set_robot_state_keys(...) if needed,
            # but we set it here defensively so the semantics match the provider path.
            policy = policy_object
        else:
            policy = create_policy(policy_provider, **(policy_config or {}))
        policy.set_robot_state_keys(self.robot_joint_names(robot_name))

        on_frame = self._make_run_policy_hook(robot_name, instruction)

        return PolicyRunner(self).run(
            robot_name,
            policy,
            instruction=instruction,
            duration=duration,
            control_frequency=control_frequency,
            action_horizon=action_horizon,
            fast_mode=fast_mode,
            video=VideoConfig.from_dict(video),
            on_frame=on_frame,
            max_onframe_failures=max_onframe_failures,
            control_substeps=control_substeps,
        )

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
        policy_object: Policy | None = None,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Start policy execution in a background thread (non-blocking).

        Default implementation: synchronous passthrough to ``run_policy``.
        Backends that support true background execution (like MuJoCo via
        its ``ThreadPoolExecutor``) should override.

        accepts ``n_steps`` (primary) or legacy ``max_steps`` as an
        alternate to ``duration``. See ``run_policy`` for conversion rules.
        """
        robot_name = self._resolve_single_robot(robot_name)
        return self.run_policy(
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

    def replay_episode(
        self,
        repo_id: str,
        robot_name: str | None = None,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        action_key_map: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a LeRobotDataset episode via ``PolicyRunner.replay``.

        Override per backend for optimised replay (e.g. direct ctrl
        writes) only when measured necessary.
        """

        return PolicyRunner(self).replay(
            repo_id,
            robot_name=robot_name,
            episode=episode,
            root=root,
            speed=speed,
            action_key_map=action_key_map,
        )

    def eval_policy(
        self,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 1,
        max_steps: int = 300,
        success_fn: str | None = None,
    ) -> dict[str, Any]:
        """Multi-episode policy evaluation via ``PolicyRunner.evaluate``.

        ``robot_name`` is required - eval_policy used to silently pick
        the first robot, which is surprising in multi-robot scenes.
        ``n_episodes`` default lowered from 10 to 1 (callers opt in to
        longer evals explicitly).
        """
        from strands_robots.policies import create_policy

        if not robot_name:
            return {
                "status": "error",
                "content": [{"text": "eval_policy requires 'robot_name'."}],
            }
        robots = self.list_robots()
        if not robots:
            return {"status": "error", "content": [{"text": "No robots in sim. Add one first."}]}
        if robot_name not in robots:
            return {
                "status": "error",
                "content": [{"text": f"Robot '{robot_name}' not found."}],
            }
        resolved_robot = robot_name

        policy = create_policy(policy_provider, **(policy_config or {}))
        policy.set_robot_state_keys(self.robot_joint_names(resolved_robot))

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            max_steps=max_steps,
            success_fn=success_fn,
        )

    # Benchmark protocol facades

    def evaluate_benchmark(
        self,
        benchmark_name: str,
        robot_name: str | None = None,
        policy_provider: str = "mock",
        policy_config: dict[str, Any] | None = None,
        instruction: str = "",
        n_episodes: int = 1,
        seed: int | None = None,
        action_horizon: int = 8,
        on_frame: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Run a registered :class:`BenchmarkProtocol` against the current sim.

        Benchmark-agnostic evaluation entry point. Looks up ``benchmark_name``
        in the global benchmark registry, validates robot compatibility, and
        forwards to :meth:`PolicyRunner.evaluate` with the spec.
        ``max_steps`` comes from the benchmark (not a parameter here).

        Args:
            benchmark_name: Key from :func:`register_benchmark` /
                :func:`register_benchmark_from_file`.
            robot_name: Robot to evaluate. If ``None`` and the benchmark has
                exactly one supported robot that matches a loaded robot, that
                robot is picked; otherwise returns an error.
            policy_provider: Policy provider name (forwarded to
                :func:`create_policy`).
            policy_config: Provider-specific kwargs.
            instruction: Natural-language instruction for the policy.
            n_episodes: Number of episodes.
            seed: Master RNG seed for per-episode reproducibility.
            action_horizon: How many actions to consume from each
                ``policy.get_actions(...)`` chunk before re-querying the
                policy. Default ``8`` matches NVIDIA's upstream
                GR00T LIBERO eval (``MultiStepWrapper`` with
                ``n_action_steps=8``) — the policy commits to 8 actions
                before re-observing, which is what GR00T-N1.7-LIBERO
                checkpoints were trained against. Set to ``1`` for
                closed-loop receding-horizon control (re-observe every
                step; matches OpenVLA-style eval). Values < 1 are
                rejected with a structured error. ``on_step`` and
                success/failure checks run after EACH applied action,
                so per-step rewards and early termination work
                correctly regardless of horizon.
            on_frame: Optional ``(step, observation, action) -> None``
                hook fired per applied control step on the eval thread,
                immediately after ``sim.send_action``. Use this for
                synchronous recording or telemetry when the eval is
                dispatched from a thread distinct from the script main
                (e.g. Strands ``Agent`` tool dispatch under asyncio) —
                the daemon-thread recorder
                (:meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording`)
                races ``mjData`` mutations on the eval thread under that
                pattern and produces 2-3% frame-capture rates with
                greenish GL clear-colour artifacts. Pair with
                :meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording_synchronous`
                for the recorder side. See #191.

        Returns:
            Standard status dict. On success, carries per-episode cumulative
            reward + aggregate success_rate / avg_reward / avg_steps in the
            JSON payload.
        """
        from strands_robots.policies import create_policy
        from strands_robots.simulation.benchmark import get_benchmark

        if not isinstance(action_horizon, int) or action_horizon < 1:
            return {
                "status": "error",
                "content": [
                    {"text": (f"evaluate_benchmark: action_horizon must be a positive integer, got {action_horizon!r}")}
                ],
            }

        spec = get_benchmark(benchmark_name)
        if spec is None:
            from strands_robots.simulation.benchmark import list_benchmarks as _list

            available = sorted(_list().keys())
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"evaluate_benchmark: no benchmark registered under "
                            f"{benchmark_name!r}. Registered: {available}. "
                            "Call register_benchmark_from_file or register_benchmark first."
                        )
                    }
                ],
            }

        robots = self.list_robots()
        if not robots:
            return {"status": "error", "content": [{"text": "No robots in sim. Add one first."}]}

        resolved_robot = robot_name
        if not resolved_robot:
            # Try to pick a robot. Prefer single-robot scenes; multi-robot
            # scenes require explicit selection.
            if len(robots) == 1:
                resolved_robot = robots[0]
            else:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"evaluate_benchmark: 'robot_name' is required when the sim has "
                                f"multiple robots. Loaded: {robots}"
                            )
                        }
                    ],
                }
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": f"Robot '{resolved_robot}' not found. Loaded: {robots}"}],
            }

        policy = create_policy(policy_provider, **(policy_config or {}))
        policy.set_robot_state_keys(self.robot_joint_names(resolved_robot))

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            spec=spec,
            seed=seed,
            action_horizon=action_horizon,
            on_frame=on_frame,
        )

    def list_benchmarks(self) -> dict[str, Any]:
        """Enumerate registered benchmarks.

        Returns a standard status dict whose JSON payload contains the
        :func:`~strands_robots.simulation.benchmark.list_benchmarks`
        metadata snapshot. Safe to call from any backend; the registry is
        engine-agnostic.
        """
        from strands_robots.simulation.benchmark import list_benchmarks as _list

        snapshot = _list()
        if not snapshot:
            text = "No benchmarks registered. Use register_benchmark_from_file to add one."
        else:
            lines = [f"Registered benchmarks ({len(snapshot)}):"]
            for name, meta in snapshot.items():
                lines.append(
                    f"  • {name}: {meta['class']} "
                    f"(robots={meta['supported_robots'] or 'any'}, "
                    f"default={meta['default_robot']}, "
                    f"max_steps={meta['max_steps']})"
                )
            text = "\n".join(lines)
        return {
            "status": "success",
            "content": [{"text": text}, {"json": {"benchmarks": snapshot}}],
        }

    def register_benchmark_from_file(
        self,
        benchmark_name: str,
        spec_path: str,
    ) -> dict[str, Any]:
        """Load a declarative benchmark spec from disk and register it.

        Wraps :func:`strands_robots.simulation.benchmark_spec.register_benchmark_from_file`
        so agents can author benchmarks as YAML / JSON at runtime. Parsing
        errors surface as structured error dicts rather than exceptions.
        """
        from strands_robots.simulation.benchmark_spec import (
            register_benchmark_from_file as _register,
        )

        if not benchmark_name:
            return {
                "status": "error",
                "content": [{"text": "register_benchmark_from_file: 'benchmark_name' must be non-empty."}],
            }
        if not spec_path:
            return {
                "status": "error",
                "content": [{"text": "register_benchmark_from_file: 'spec_path' must be non-empty."}],
            }
        try:
            benchmark = _register(benchmark_name, spec_path)
        except FileNotFoundError as e:
            return {"status": "error", "content": [{"text": f"register_benchmark_from_file: {e}"}]}
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"register_benchmark_from_file: {e}"}]}
        except ImportError as e:
            # YAML support requires pyyaml; surface the install hint verbatim.
            return {"status": "error", "content": [{"text": f"{e}"}]}
        except Exception as e:  # noqa: BLE001 - defensive catch-all with clear message
            return {
                "status": "error",
                "content": [{"text": f"register_benchmark_from_file: unexpected error: {e}"}],
            }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📋 Registered benchmark '{benchmark_name}' from {spec_path}\n"
                        f"  class: {type(benchmark).__name__}\n"
                        f"  supported_robots: {benchmark.supported_robots or 'any'}\n"
                        f"  default_robot: {benchmark.default_robot}\n"
                        f"  max_steps: {benchmark.max_steps}"
                    )
                }
            ],
        }

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Override to return an ``on_frame(step, obs, action)`` callable.

        Used by backends that want to layer in recording / telemetry
        without subclassing :class:`PolicyRunner`. Default: no hook.

        Args:
            robot_name: Robot being controlled this run.
            instruction: Instruction passed to this run.

        Returns:
            Callable or ``None``.
        """
        return None

    # Optional overrides (have default no-op implementations)

    def load_scene(self, scene_path: str) -> dict[str, Any]:
        """Load a complete scene from file. Override per backend."""
        raise NotImplementedError("load_scene not implemented by this backend")

    def randomize(self, **kwargs: Any) -> dict[str, Any]:
        """Apply domain randomization.

        Concrete backends define their own parameter signatures.
        Override per backend.
        """
        raise NotImplementedError("randomize not implemented by this backend")

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information. Override per backend."""
        raise NotImplementedError("get_contacts not implemented by this backend")

    # Discovery / introspection

    def describe(self) -> dict[str, Any]:
        """Return a machine-readable summary of this engine's live contract.

        Agents should call this first to learn what robots exist, what cameras
        are attached, and the signatures of the methods most commonly needed --
        in a single call, instead of guessing method names.

        Returns:
            Plain dict with keys: robots, cameras, methods, note.
        """
        return {
            "robots": self.list_robots(),
            "cameras": [],  # backends override to list camera names
            "methods": {
                "get_robot_state": "(robot_name: str) -> dict",
                "get_observation": "(robot_name: str | None = None, *, skip_images: bool = False) -> dict",
                "send_action": "(action: dict, robot_name: str | None = None, n_substeps: int = 1) -> None",
                "run_policy": "(robot_name: str, policy_provider='mock', ...) -> dict",
                "start_policy": "(robot_name: str, policy_provider='mock', ...) -> dict",
                "list_robots": "() -> list[str]",
                "render": "(camera_name='default', width=None, height=None) -> dict",
                "reset": "() -> dict",
                "step": "(n_steps: int = 1) -> dict",
            },
            "note": (
                "robot_name defaults to the sole robot when only one exists "
                "for get_observation and send_action. For get_robot_state, "
                "pass the robot name explicitly (from the 'robots' list)."
            ),
        }

    def cleanup(self) -> None:
        """Release all resources. Called on __del__ / context exit."""
        pass

    def __enter__(self) -> SimEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        try:
            self.cleanup()
        except Exception as e:
            # Best-effort cleanup during GC - exceptions can't propagate
            # from __del__ (CPython ignores them), so log for visibility.
            logger.warning("Cleanup error during __del__: %s", e)
