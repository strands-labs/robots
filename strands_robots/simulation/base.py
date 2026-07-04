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
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
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
# ``TYPE_CHECKING`` - CodeQL's ``py/unsafe-cyclic-import`` rule walks
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

    def _init_ros_bridge(self, *, ros2_bridge: bool = False, ros2_domain: int = 0) -> None:
        """Initialize the optional ROS 2 telemetry bridge state.

        Backends that accept a ``ros2_bridge`` flag call this once from their
        own ``__init__``. It is intentionally a plain method rather than an ABC
        ``__init__`` override: the simulation interface imposes no base-class
        constructor contract, so lightweight subclasses and test doubles need
        not thread ``super().__init__()`` through just to satisfy the ABC.

        Args:
            ros2_bridge: When True, publish per-robot ``joint_states`` and
                camera ``image_raw`` on a ROS 2 domain every :meth:`step`, so
                external ROS 2 nodes can subscribe to the running simulation.
                Requires ``rclpy`` (system ROS 2 / the official docker image);
                an :class:`ImportError` is raised here if it is missing.
                Defaults to False - the sim never touches ROS 2.
            ros2_domain: ROS 2 domain id (``ROS_DOMAIN_ID``) to publish on.
        """
        self._ros2_bridge_enabled = bool(ros2_bridge)
        self._ros2_domain = int(ros2_domain)
        self._ros_bridge: Any = None
        if self._ros2_bridge_enabled:
            from strands_robots.simulation.ros_bridge import SimRosBridge

            self._ros_bridge = SimRosBridge(domain_id=self._ros2_domain)

    def _publish_ros_telemetry(self, *, skip_images: bool = False) -> None:
        """Publish joint_states (and camera images) for every robot once.

        No-op when the ROS 2 bridge is disabled or was never initialized.
        Called by backends from :meth:`step` after the physics tick. Per-robot
        failures (e.g. a camera that did not render) never interrupt the loop.
        """
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is None:
            return
        for robot in self.list_robots():
            # Per-robot guard: a transient render/observation failure on one
            # robot (e.g. EGL/GL context loss, a camera that produced no frame)
            # must not interrupt the loop or crash the caller's step(). Publish
            # what succeeds, log-and-continue on the rest - this is the contract
            # the docstring promises on the hot ros2_bridge=True path.
            try:
                obs = self.get_observation(robot, skip_images=skip_images)
                names = self.robot_joint_names(robot)
                positions = [obs[j] for j in names if j in obs and isinstance(obs[j], (int, float))]
                bridge.publish_joint_states(robot, names, positions)
                if skip_images:
                    continue
                for key, value in obs.items():
                    if key in names:
                        continue
                    if hasattr(value, "ndim") and getattr(value, "ndim", 0) == 3:
                        bridge.publish_image(robot, key, value)
            except Exception:
                logger.warning(
                    "ROS 2 telemetry publish failed for robot %r; skipping this robot for this step",
                    robot,
                    exc_info=True,
                )
                continue

    def _shutdown_ros_bridge(self) -> None:
        """Tear down the ROS 2 bridge if one is active. Safe to call repeatedly."""
        bridge = getattr(self, "_ros_bridge", None)
        if bridge is not None:
            bridge.shutdown()
            self._ros_bridge = None

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
        """Reset simulation to its initial state.

        Contract: on return the world must be left in a fully consistent,
        observation-ready state - derived kinematics (Cartesian body/site/geom
        poses and camera transforms) must reflect the reset pose WITHOUT
        requiring a subsequent ``step()``. ``eval_policy`` calls
        ``get_observation()`` immediately after ``reset()`` and before the
        first action, so a backend that leaves derived state stale would feed
        the policy's first inference of every episode a degenerate observation.
        The MuJoCo backend enforces this by running ``mj_forward`` after
        ``mj_resetData`` (which alone zeroes all derived quantities).
        """
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

        Used by ``Policy.set_robot_state_keys`` to name the
        ``observation.state`` vector. Action-vector binding (``send_action``
        with a numeric vector, ``PolicyRunner.replay``) uses
        :meth:`robot_action_keys` instead - a robot's actuators are not always
        its joints. Order must match the backend's joint ordering.
        """
        ...

    def robot_action_keys(self, robot_name: str) -> list[str]:
        """Return the action keys ``send_action`` resolves for ``robot_name``.

        These are the names a policy should emit as its action-dict keys: the
        robot's *actuators*, which are NOT always its joints. A robot can have
        passive/mimic joints with no driving actuator (gripper finger
        followers) and tendon-driven actuators that are not joints at all (a
        grasp tendon). Keying a policy by ``robot_joint_names`` in those cases
        emits keys that ``send_action`` cannot resolve, so the affected
        actuators never move and the robot silently no-ops.

        The default mirrors :meth:`robot_joint_names` for backends whose
        actuator set matches their joint set. Backends with a distinct
        actuator namespace (e.g. MuJoCo tendon grippers) override this to
        return the actuator short-names instead.
        """
        return self.robot_joint_names(robot_name)

    def bind_policy_sim_context(self, policy: Any, robot_name: str) -> None:
        """Give a policy the backend sim context it needs to close the loop.

        Default no-op. The MuJoCo engine overrides this to hand policies that
        opt in (e.g. ``VeraPolicy.set_sim_context``) the compiled ``MjModel`` +
        the robot's namespace, so eef/cartesian-delta policies can auto-configure
        their IK end-effector frame with zero manual wiring. Policies that don't
        expose ``set_sim_context`` are unaffected.
        """
        return None

    def _maybe_install_wbc_torque_control(self, policy: Any, robot_name: str) -> Callable[[], None] | None:
        """Hook: auto-install an action controller a policy needs to run correctly.

        Default no-op (returns ``None``). The MuJoCo engine overrides this so a
        :class:`~strands_robots.policies.wbc.WBCPolicy` driven through
        :meth:`run_policy` on a position-servo scene gets the torque shim
        (:func:`~strands_robots.policies.wbc.install_wbc_torque_control`) wired
        up automatically - otherwise WBC's position targets fight the stiff
        servo gain and the documented quickstart silently falls over.

        Returns an optional zero-arg cleanup callable that :meth:`run_policy`
        invokes in a ``finally`` block to restore the scene after the rollout.
        """
        return None

    def _preflight_policy_config(
        self,
        robot_name: str,
        policy_provider: str,
        policy_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Run a provider's pre-construction preflight before ``create_policy``.

        Resolves the provider's policy class WITHOUT instantiating it and runs
        its :meth:`~strands_robots.policies.base.Policy.preflight` hook (a
        no-op for providers that do not override it) against the runtime
        observation keys. This catches a misconfiguration - e.g. sim camera
        names that cannot be routed to a VLA's declared image inputs - BEFORE
        the expensive model-weight download, instead of crashing deep inside
        the first inference.

        Args:
            robot_name: Robot whose observation keys define the runtime inputs.
            policy_provider: Provider name / smart string passed to
                ``create_policy``.
            policy_config: Provider kwargs (the policy_config).

        Returns:
            A ``status=error`` dict (for the caller to return) when the
            provider's preflight rejects the configuration; ``None`` when the
            check passes, is a no-op, or the observation is not yet available.
        """
        from strands_robots.policies import preflight_policy

        obs = self.get_observation(robot_name)
        if not isinstance(obs, dict) or not obs:
            return None
        try:
            preflight_policy(policy_provider, set(obs.keys()), **(policy_config or {}))
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}
        return None

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
        material: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Add a primitive or mesh object to the scene.

        The ``size`` convention is backend-specific -- the default MuJoCo
        backend treats ``size`` as the **full extent in meters** per axis
        (halved internally to MuJoCo's half-extents), whereas Newton consumes
        half-extents / radii directly. See the concrete backend's
        ``add_object`` docstring for the exact per-shape semantics and an
        example. Returns an agent-tool status dict.

        ``material`` (optional): backend-specific visual material/texture
        spec. ``None`` keeps the flat ``color`` rgba (unchanged); a backend
        that supports it (MuJoCo) attaches a real material so surfaces can be
        matte or textured. Backends that do not support it should reject a
        non-``None`` ``material`` loudly rather than silently ignore it.
        """
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

    def _coerce_action(
        self, action: dict[str, Any] | Sequence[float], robot_name: str
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Normalize an action into a ``{joint/actuator name: value}`` mapping.

        Policies and the ``Robot`` ABC commonly emit an ordered action *vector*
        (a ``list`` / ``tuple`` / 1-D ``numpy`` array) rather than a name->value
        mapping. To keep :meth:`send_action` usable directly with such a vector -
        and consistent with :meth:`replay_episode`, which binds a recorded action
        vector positionally to :meth:`robot_action_keys` - a sequence is zipped
        against ``robot_action_keys(robot_name)`` in declaration order. Those are
        the robot's *actuator* keys (what ``send_action`` resolves and what the
        LeRobotDataset recorder writes the ``action`` column in); they diverge
        from ``robot_joint_names`` whenever a robot has passive/mimic joints with
        no driving actuator or a tendon-driven gripper, so binding a raw action
        vector to joint names there mis-maps or drops commanded DOFs. A mapping
        is returned unchanged. The vector length must match the robot's actuator
        count exactly; a mismatch is reported as a caller error rather than
        silently truncated (which would drop commands - e.g. a gripper axis).

        Args:
            action: A ``{name: value}`` mapping, or an ordered numeric vector
                whose entries correspond to ``robot_action_keys(robot_name)``.
            robot_name: Resolved robot whose actuator order defines the binding.

        Returns:
            An ``(action_dict, error)`` tuple. When ``error`` is non-None it is a
            structured ``{"status": "error", ...}`` dict and ``action_dict`` must
            be ignored. Otherwise ``action_dict`` is the normalized mapping.
        """
        if isinstance(action, Mapping):
            return dict(action), None

        # ``str``/``bytes`` are iterable but never a valid multi-joint action;
        # a scalar has no length. Reject both with an actionable message instead
        # of producing garbage character/positional keys downstream.
        if isinstance(action, (str, bytes)) or not hasattr(action, "__len__"):
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "send_action: 'action' must be a mapping of "
                            "{joint/actuator name: value} or an ordered numeric "
                            f"vector, got {type(action).__name__}."
                        )
                    }
                ],
            }

        try:
            values = [float(v) for v in action]
        except (TypeError, ValueError) as exc:
            return None, {
                "status": "error",
                "content": [{"text": f"send_action: action vector has a non-numeric entry: {exc}."}],
            }

        action_keys = self.robot_action_keys(robot_name)
        if len(values) != len(action_keys):
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"send_action: action vector length {len(values)} does not "
                            f"match robot '{robot_name}' action-key count {len(action_keys)}. "
                            f"Action keys (in order): {action_keys}. Pass a {{name: value}} "
                            "mapping to target a subset of actuators."
                        )
                    }
                ],
            }
        return {name: value for name, value in zip(action_keys, values)}, None

    @abstractmethod
    def send_action(
        self,
        action: dict[str, Any] | Sequence[float],
        robot_name: str | None = None,
        n_substeps: int = 1,
    ) -> dict[str, Any]:
        """Apply action and advance physics by n_substeps.

        Contract: each call writes actuator/ctrl values and then runs
        ``n_substeps`` physics steps (e.g. mj_step). PolicyRunner.run()
        relies on this - it calls send_action once per control step and
        does NOT call sim.step() separately.

        Backends are responsible for internal thread-safety (e.g.
        MuJoCo acquires self._lock here). PolicyRunner does not manage
        locks.

        Returns:
            Dict with ``status`` and ``content``. When action keys cannot
            be resolved, the ``content`` list includes a ``json`` block with
            ``unresolved_keys`` so callers can self-correct.
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

        Returns an agent-tool dict with ``status`` and a ``content`` list. On
        success the content holds an ``image`` block carrying PNG bytes
        (``{"image": {"format": "png", "source": {"bytes": ...}}}``); the raw
        RGB ``numpy`` arrays are available per-camera via :meth:`get_observation`.
        Resolution comes from the named camera's configuration (set via
        ``add_camera``) unless ``width``/``height`` are given; the free camera
        and model-only cameras fall back to the engine default.
        """
        ...

    # Policy orchestration (concrete facade, not abstract)

    @staticmethod
    def _resolve_horizon(
        n_steps: int | None,
        max_steps: int | None,
        control_frequency: float,
        duration: float,
    ) -> tuple[float, int | None, dict[str, Any] | None]:
        """Resolve a step horizon into a wall-clock duration.

        ``n_steps`` (primary) or the legacy ``max_steps`` alias specify the
        rollout length as a step count; ``duration = n_steps / control_frequency``.
        ``n_steps`` wins when both are passed. The inputs are validated before
        the division so a non-positive horizon or frequency is reported as a
        caller error rather than silently producing a no-op, a negative
        duration, or a ``ZeroDivisionError``.

        Args:
            n_steps: Primary step-count horizon, or ``None``.
            max_steps: Legacy alias, normalized to ``n_steps`` when ``n_steps``
                is ``None``.
            control_frequency: Target control-loop frequency in Hz.
            duration: Fallback wall-clock duration used when no step horizon
                is given.

        Returns:
            A ``(duration, n_steps, error)`` tuple. When ``error`` is non-None
            it is a structured ``{"status": "error", ...}`` dict and the other
            fields must be ignored. Otherwise ``duration`` is the resolved
            wall-clock duration (recomputed from the horizon when one was
            given) and ``n_steps`` is the normalized step count (or ``None``).
        """
        if n_steps is None and max_steps is not None:
            n_steps = int(max_steps)
        if n_steps is not None:
            if n_steps <= 0:
                return (
                    duration,
                    n_steps,
                    {
                        "status": "error",
                        "content": [{"text": f"run_policy: n_steps must be > 0, got {n_steps}."}],
                    },
                )
            # control_frequency is validated as a positive number at the public
            # entry points (run_policy / start_policy / eval_policy) via
            # _validate_positive_frequency before this helper runs, so the
            # division below is safe.
            duration = float(n_steps) / float(control_frequency)
        return duration, n_steps, None

    @staticmethod
    def _validate_action_horizon(action_horizon: Any, method: str) -> dict[str, Any] | None:
        """Reject a non-positive-integer ``action_horizon`` at the public API.

        ``action_horizon`` is how many actions are consumed from each policy
        chunk before re-querying. A value below 1 (or a non-int) is meaningless
        and would otherwise be silently clamped to 1 by
        :func:`~strands_robots.policies.base.resolve_chunk_length`, hiding the
        caller's mistake behind a rollout that does not run the requested
        horizon. Returns a structured ``{"status": "error", ...}`` dict to
        surface, or ``None`` when the value is valid.

        Args:
            action_horizon: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        if not isinstance(action_horizon, int) or action_horizon < 1:
            return {
                "status": "error",
                "content": [{"text": f"{method}: action_horizon must be a positive integer, got {action_horizon!r}."}],
            }
        return None

    @staticmethod
    def _validate_positive_int(value: Any, name: str, method: str) -> dict[str, Any] | None:
        """Reject a non-positive-integer count at the public API.

        Shared guard for the rollout count knobs that must be ``>= 1`` -
        ``n_episodes`` (how many reset->rollout episodes to run) and
        ``max_steps`` (the per-episode step cap). A zero/negative/non-int
        value would otherwise flow into the rollout loop and produce a
        degenerate result that still reports ``status="success"``: an eval
        over zero episodes, or episodes of zero length, that fabricate a 0%
        success rate (``Episodes: -2 | Success: 0/-2``) instead of surfacing
        the caller's mistake. Returns a structured ``{"status": "error", ...}``
        dict to surface, or ``None`` when the value is valid.

        Args:
            value: The caller-supplied value to validate.
            name: Parameter name, used in the error message.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        if not isinstance(value, int) or value < 1:
            return {
                "status": "error",
                "content": [{"text": f"{method}: {name} must be a positive integer, got {value!r}."}],
            }
        return None

    @staticmethod
    def _validate_positive_frequency(control_frequency: Any, method: str) -> dict[str, Any] | None:
        """Reject a non-positive or non-numeric ``control_frequency`` at the public API.

        ``control_frequency`` (Hz) sets the control-loop rate the rollout steps
        physics at. It is used as a divisor (the per-action period is
        ``1 / control_frequency`` and ``duration = n_steps / control_frequency``)
        and is handed to :meth:`PolicyRunner`'s per-period substep computation
        (``round(1 / control_frequency / ...)``); a value ``<= 0`` or a
        non-number otherwise reaches that arithmetic deep inside the runner and
        raises a bare ``ValueError``/``TypeError``/``ZeroDivisionError`` rather
        than the structured tool-error dict the public API contracts. ``bool`` is
        rejected explicitly: it is an ``int`` subclass, so ``True`` would slip
        through the numeric check and act as a silent 1 Hz. Returns a structured
        error dict to surface, or ``None`` when valid.

        Args:
            control_frequency: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        if (
            isinstance(control_frequency, bool)
            or not isinstance(control_frequency, (int, float))
            or control_frequency <= 0
        ):
            return {
                "status": "error",
                "content": [{"text": f"{method}: control_frequency must be > 0, got {control_frequency!r}."}],
            }
        return None

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
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        n_episodes: int = 1,
        reset_between: bool = True,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        wbc_install_torque_control: bool = True,
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
            control_frequency: Target Hz for policy queries. Must be a
                positive number; a non-positive, non-numeric, or bool value
                is reported as a structured caller error.
            action_horizon: Lower bound on actions consumed from each
                policy chunk before re-querying. The effective interval is
                ``max(action_horizon, policy.execution_horizon)`` (see
                ``strands_robots.policies.resolve_chunk_length``): a
                chunk-emitting policy always keeps its full trained chunk, so
                a value below that chunk length (e.g. a VLA whose
                ``execution_horizon`` is 50) has no effect. RTC policies own
                their own interval and ignore this entirely. Must be a
                positive integer (>= 1); a non-positive or non-int value is
                reported as a caller error.
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
            seed: Optional master RNG seed for a reproducible single rollout.
                When set, reseeds Python / NumPy / torch / cuDNN and forwards
                ``policy.reset(seed=...)`` so a stochastic policy (VLA action-
                chunk sampling, diffusion noise) produces the same trajectory
                on re-run of the same scene. ``None`` (default) leaves RNG
                state untouched. Mirrors the per-episode reseed in
                :meth:`eval_policy`.
            policy_kwargs: Optional per-call goal payload forwarded verbatim to
                every ``policy.get_actions(obs, instruction, **policy_kwargs)``
                call. Carries the well-known #300 goal keys
                (``target_pose`` / ``target_joints`` / ``target_velocity`` /
                ``world_update``) to non-VLA providers (cuRobo, MoveIt2, WBC)
                that read their goal from kwargs rather than the instruction.
                This is the local-sim analogue of the mesh ``tell()`` path,
                which already forwards these keys. VLA providers ignore unknown
                kwargs per the #300 contract, so forwarding is always safe.
            n_episodes: Number of sequential episode rollouts to run in this
                single call (default ``1`` - the historical single-rollout
                behaviour, unchanged). IMPORTANT: calling with the default
                ``n_episodes=1`` produces exactly ONE dataset episode, no matter
                how many "episodes" you intend in natural language. To record N
                DISTINCT dataset episodes pass ``n_episodes=N`` in a single call
                - do NOT loop this call N times narrating "N episodes" (that
                buffers all frames into one merged ``episode_index=0``
                mega-episode). After ``stop_recording``, confirm the count with
                :meth:`verify_dataset_episodes`. When ``> 1``, each episode runs one
                rollout for the configured horizon, then a dataset episode
                boundary is flushed via :meth:`save_episode` (only when a
                recording is active) so the dataset ends up with N correctly
                delimited episodes instead of one merged episode. This is the
                first-class multi-episode collection API; it removes the need
                for a manual ``for _ in range(n): run_policy(); save_episode();
                reset()`` loop. ``seed`` (when set) is offset per episode
                (``seed + i``) for reproducible-yet-distinct rollouts, and
                ``video`` (when set) is written per episode to a path with
                ``_ep{i}`` inserted before the extension so episodes do not
                overwrite one another.
            reset_between: When running multiple episodes, reset the sim to its
                initial state between episodes (default ``True``). The reset
                never fires after the final episode. Set ``False`` to chain
                episodes from the end state of the previous one.
            async_rtc: When ``True``, overlap policy inference with action
                execution so the next action chunk is computed in the
                background while the current chunk is still draining (latency
                masking). ``False`` keeps the synchronous chunk-then-drain loop.
                ``None`` (default) auto-resolves from ``policy.is_chunk_emitting()``
                so chunk-emitting VLA/flow-matching policies (pi0, pi0.5,
                pi0-FAST, SmolVLA, MolmoAct2) get latency masking automatically
                while single-step policies stay synchronous; an explicit
                ``True``/``False`` always wins. Forwarded verbatim to
                :meth:`PolicyRunner.run`; see its docstring for the full
                contract (provider-agnostic, RTC-policy seam blending, thread
                safety).
            rtc_inference_timeout_s: Optional hard per-chunk timeout (seconds)
                for the async-RTC prefetch. When set, a stuck inference surfaces
                as a structured ``status=error`` result (carrying the RTC
                telemetry) instead of hanging the sim. ``None`` (default) waits
                without a deadline. Forwarded verbatim to
                :meth:`PolicyRunner.run`; ignored on the synchronous path.
            wbc_install_torque_control: When ``True`` (default), a
                :class:`~strands_robots.policies.wbc.WBCPolicy` run on a
                position-servo scene (the stock ``Robot("unitree_g1")``) gets the
                torque shim auto-installed for the duration of this call, then
                uninstalled. WBC emits joint-position targets; the stock G1's
                uniform ``kp=500`` servo would override SONIC's tuned per-joint
                PD and the gait diverges, so the documented quickstart silently
                falls over without it. Set ``False`` to manage the controller
                yourself or to drive a torque-actuated scene directly. No-op for
                non-WBC policies and on backends without the hook.

        Returns:
            Standard status dict with an agent-consumable ``{"json": {...}}``
            content block alongside the human-readable ``text``. The json block
            carries the rollout facts as typed fields (``n_steps``,
            ``elapsed_s``, ``stopped_early``, ``action_errors``, ``video_path``,
            ``video_frames``, ``positional_fallback_used``,
            ``generic_state_keys_used``, ...) so callers can self-correct
            programmatically without parsing the text. The two routing-
            degradation flags are True when the driving policy could not bind
            the observation to the model's inputs by name and silently fell
            back (a camera routed to a model image slot positionally, or
            ``observation.state`` composed from the observation's own scalar
            keys because none of ``robot_state_keys`` matched). A True flag on
            an otherwise ``success`` run is the signature of the robot moving
            on meaningless inputs. Mirrors :meth:`eval_policy`.
        """
        from strands_robots.policies import create_policy

        robot_name = self._resolve_single_robot(robot_name)

        if err := self._validate_positive_frequency(control_frequency, "run_policy"):
            return err

        # accept n_steps (or legacy max_steps) as an alternate horizon
        # specification. duration = n_steps / control_frequency. If both
        # are passed, n_steps wins (primary per DoD).
        duration, n_steps, horizon_error = self._resolve_horizon(n_steps, max_steps, control_frequency, duration)
        if horizon_error is not None:
            return horizon_error

        if err := self._validate_positive_int(n_episodes, "n_episodes", "run_policy"):
            return err

        if err := self._validate_action_horizon(action_horizon, "run_policy"):
            return err

        if robot_name not in self.list_robots():
            return {
                "status": "error",
                "content": [{"text": f"Robot '{robot_name}' not found."}],
            }

        if policy_object is None:
            # Fail fast on a misconfiguration (e.g. camera names that cannot be
            # routed to the policy's declared image inputs) BEFORE the expensive
            # create_policy weight download.
            preflight_error = self._preflight_policy_config(robot_name, policy_provider, policy_config)
            if preflight_error is not None:
                return preflight_error
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - skip the expensive create_policy call.
            # Caller is responsible for policy.set_robot_state_keys(...) if needed,
            # but we set it here defensively so the semantics match the provider path.
            policy = policy_object
        policy.set_robot_state_keys(self.robot_action_keys(robot_name))
        self.bind_policy_sim_context(policy, robot_name)

        # Auto-install any action controller this policy needs to run correctly
        # on this scene (e.g. the WBC torque shim on a position-servo G1). The
        # cleanup callable restores the scene in the finally below. Opt out with
        # wbc_install_torque_control=False (e.g. when you manage the controller
        # yourself or drive a torque-actuated scene directly).
        controller_cleanup = (
            self._maybe_install_wbc_torque_control(policy, robot_name) if wbc_install_torque_control else None
        )

        try:
            runner = PolicyRunner(self)

            # Single-episode fast path: byte-for-byte the historical behaviour
            # (no reset, no episode-boundary flush). n_episodes defaults to 1 so
            # existing callers are completely unaffected.
            if n_episodes == 1:
                recording = self._is_recording()
                if recording:
                    logger.info(
                        "run_policy: n_episodes=1, will produce 1 dataset episode of ~%d frames "
                        "(frames buffer into the current episode and flush at save_episode/"
                        "stop_recording). To record N DISTINCT dataset episodes pass n_episodes=N "
                        "- do NOT loop the tool call.",
                        int(duration * control_frequency),
                    )
                on_frame = self._make_run_policy_hook(robot_name, instruction)
                result = runner.run(
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
                    policy_kwargs=policy_kwargs,
                    seed=seed,
                    async_rtc=async_rtc,
                    rtc_inference_timeout_s=rtc_inference_timeout_s,
                )
                completed = 1 if result.get("status") == "success" else 0
                contract = self._episode_contract_fields(
                    requested=1, completed=completed, saved=0, flush_deferred=recording
                )
                self._merge_json_fields(result, contract)
                return result

            # Multi-episode path: one rollout per episode, flushing a dataset
            # episode boundary (save_episode) when recording and resetting between
            # episodes. Replaces the brittle manual
            # ``for _ in range(n): run_policy(); save_episode(); reset()`` loop.
            return self._run_episodes(
                runner,
                robot_name,
                policy,
                instruction=instruction,
                duration=duration,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=video,
                max_onframe_failures=max_onframe_failures,
                control_substeps=control_substeps,
                policy_kwargs=policy_kwargs,
                seed=seed,
                n_episodes=n_episodes,
                reset_between=reset_between,
                async_rtc=async_rtc,
                rtc_inference_timeout_s=rtc_inference_timeout_s,
            )
        finally:
            if controller_cleanup is not None:
                controller_cleanup()

    def _run_episodes(
        self,
        runner: PolicyRunner,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str,
        duration: float,
        control_frequency: float,
        action_horizon: int,
        fast_mode: bool,
        video: dict[str, Any] | None,
        max_onframe_failures: int | None,
        control_substeps: int | None,
        policy_kwargs: dict[str, Any] | None,
        seed: int | None,
        n_episodes: int,
        reset_between: bool,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
    ) -> dict[str, Any]:
        """Run ``n_episodes`` sequential rollouts; shared multi-episode driver.

        Behind :meth:`run_policy` when ``n_episodes > 1``. Per episode it:
        (1) runs one rollout for the configured horizon, (2) flushes a dataset
        episode boundary via :meth:`save_episode` when a recording is active,
        and (3) resets the sim between episodes unless ``reset_between`` is
        ``False`` - so a single call yields N correctly delimited dataset
        episodes instead of one merged episode. Aborts early (returning a
        structured error with the episodes completed so far) if a rollout, an
        episode flush, or a reset fails.
        """
        episodes: list[dict[str, Any]] = []
        episodes_saved = 0
        total_steps = 0
        for ep in range(n_episodes):
            ep_seed = None if seed is None else seed + ep
            ep_video = self._episode_video_config(video, ep)
            on_frame = self._make_run_policy_hook(robot_name, instruction)
            result = runner.run(
                robot_name,
                policy,
                instruction=instruction,
                duration=duration,
                control_frequency=control_frequency,
                action_horizon=action_horizon,
                fast_mode=fast_mode,
                video=ep_video,
                on_frame=on_frame,
                max_onframe_failures=max_onframe_failures,
                control_substeps=control_substeps,
                policy_kwargs=policy_kwargs,
                seed=ep_seed,
                async_rtc=async_rtc,
                rtc_inference_timeout_s=rtc_inference_timeout_s,
            )
            ep_json = self._extract_json_payload(result)
            ep_record: dict[str, Any] = {"episode": ep, **ep_json}
            total_steps += int(ep_json.get("n_steps", 0) or 0)

            if result.get("status") == "error":
                ep_record["status"] = "error"
                episodes.append(ep_record)
                return self._episodes_result(
                    episodes,
                    episodes_saved,
                    total_steps,
                    n_episodes,
                    status="error",
                    extra=(
                        f"Episode {ep} rollout failed; aborting remaining "
                        f"{n_episodes - ep - 1} episode(s). {self._first_text(result)}"
                    ),
                )

            # Flush this rollout as its own dataset episode when recording.
            if self._is_recording():
                save = self.save_episode()
                if save.get("status") == "error":
                    ep_record["save_episode_error"] = self._first_text(save)
                    episodes.append(ep_record)
                    return self._episodes_result(
                        episodes,
                        episodes_saved,
                        total_steps,
                        n_episodes,
                        status="error",
                        extra=f"save_episode failed after episode {ep}: {self._first_text(save)}",
                    )
                episodes_saved += 1
                ep_record["saved"] = True

            episodes.append(ep_record)

            # Reset between episodes - never after the last one.
            if reset_between and ep < n_episodes - 1:
                reset_result = self.reset()
                if reset_result.get("status") == "error":
                    return self._episodes_result(
                        episodes,
                        episodes_saved,
                        total_steps,
                        n_episodes,
                        status="error",
                        extra=f"reset() failed after episode {ep}: {self._first_text(reset_result)}",
                    )

        return self._episodes_result(episodes, episodes_saved, total_steps, n_episodes, status="success")

    @staticmethod
    def _first_text(result: dict[str, Any]) -> str:
        """First human-readable ``text`` block from a status dict ("" if none)."""
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict):
                text = blk.get("text")
                if isinstance(text, str):
                    return text
        return ""

    @staticmethod
    def _extract_json_payload(result: dict[str, Any]) -> dict[str, Any]:
        """First agent-consumable ``{"json": {...}}`` block ({} if none)."""
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                return dict(blk["json"])
        return {}

    @staticmethod
    def _merge_json_fields(result: dict[str, Any], fields: dict[str, Any]) -> None:
        """Merge ``fields`` into the result's ``{"json": {...}}`` block in place.

        Augments the first existing json content block, or appends a new one if
        the result has none. Lets :meth:`run_policy` attach the episode-contract
        fields onto a ``PolicyRunner.run`` result without rebuilding it.
        """
        for blk in result.get("content", []) or []:
            if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                blk["json"].update(fields)
                return
        result.setdefault("content", []).append({"json": dict(fields)})

    @staticmethod
    def _episode_video_config(video: dict[str, Any] | None, episode: int) -> VideoConfig | None:
        """Per-episode :class:`VideoConfig` with ``_ep{i}`` in the filename.

        Multi-episode runs reuse one ``video`` config; without templating every
        episode would overwrite the same MP4. Inserts ``_ep{episode}`` before
        the extension so each episode gets a distinct file. Passes through
        unchanged when no video path is set.
        """
        if not video or not video.get("path"):
            return VideoConfig.from_dict(video)
        templated = dict(video)
        root, ext = os.path.splitext(str(video["path"]))
        templated["path"] = f"{root}_ep{episode}{ext or '.mp4'}"
        return VideoConfig.from_dict(templated)

    def _episodes_result(
        self,
        episodes: list[dict[str, Any]],
        episodes_saved: int,
        total_steps: int,
        n_episodes: int,
        *,
        status: str,
        extra: str = "",
    ) -> dict[str, Any]:
        """Aggregate per-episode records into one ``run_policy`` status dict.

        Mirrors the single-rollout result shape: a human-readable ``text``
        block plus an agent-consumable ``{"json": {...}}`` block carrying typed
        aggregate fields (``n_episodes_completed``, ``episodes_saved``,
        ``total_steps``, per-episode list, ``video_paths``).
        """
        completed = len(episodes)
        video_paths = [e["video_path"] for e in episodes if e.get("video_path")]
        text = (
            f"Multi-episode run_policy: {completed}/{n_episodes} episode(s) completed, "
            f"{episodes_saved} flushed to dataset, {total_steps} total steps."
        )
        if extra:
            text += f"\n{extra}"
        dataset_episode_indices: list[int] = []
        if self._is_recording():
            recorder = self._active_recorder()
            meta = getattr(getattr(recorder, "dataset", None), "meta", None)
            total_episodes = int(getattr(meta, "total_episodes", 0) or 0) if meta is not None else 0
            dataset_episode_indices = list(range(total_episodes))
        payload: dict[str, Any] = {
            "n_episodes_requested": n_episodes,
            "n_episodes_completed": completed,
            "episodes_saved": episodes_saved,
            "dataset_episode_indices": dataset_episode_indices,
            "total_steps": total_steps,
            "episodes": episodes,
            "video_paths": video_paths,
        }
        return {"status": status, "content": [{"text": text}, {"json": payload}]}

    def _is_recording(self) -> bool:
        """Whether a dataset-recording session is active.

        Backends that support LeRobot dataset recording override this; the base
        returns ``False`` so the multi-episode :meth:`run_policy` loop only
        flushes episode boundaries on backends that actually record.
        """
        return False

    def _active_recorder(self) -> Any:
        """Return the active dataset recorder object, or ``None``.

        Backends that support LeRobot dataset recording override this to expose
        the live recorder (see the MuJoCo ``RecordingMixin``). The base has no
        recorder, so it returns ``None``. Used by :meth:`run_policy` to read the
        in-memory episode count for the episode-contract fields.
        """
        return None

    def _active_dataset_root(self) -> str | None:
        """On-disk root of the active (or most recent) recording, or ``None``.

        Backends that record override this so :meth:`verify_dataset_episodes`
        can locate the dataset parquet AFTER ``stop_recording`` has finalized it
        (the recorder object is gone by then). The base has no recorder, so it
        returns ``None``.
        """
        return None

    def verify_dataset_episodes(self, expected: int) -> dict[str, Any]:
        """Verify the recorded dataset holds exactly ``expected`` episodes.

        Reads the LeRobot dataset parquet (the ground truth) for the active or
        most-recently-recorded session AND cross-checks it against the
        ``meta/info.json`` ``total_episodes`` header. Both must agree with
        ``expected``; a parquet that matches ``expected`` but disagrees with
        info.json (an internally inconsistent dataset) still fails. Reports the
        actual episode count.
        Call this AFTER :meth:`stop_recording` for a definitive check that a
        collection run produced N distinct episodes rather than one merged
        ``episode_index=0`` mega-episode.

        Episodes are flushed to ``meta/episodes/**/*.parquet`` only at
        ``save_episode`` / ``stop_recording`` (``finalize``) time, so this reads
        the canonical on-disk truth - it does not trust the recorder's in-memory
        bookkeeping (which is what :meth:`run_policy` reports while a session is
        still open).

        Args:
            expected: The episode count the caller intended to record.

        Returns:
            Standard status dict. ``status`` is ``"success"`` when the parquet
            holds exactly ``expected`` episodes, else ``"error"``. The
            ``{"json": {...}}`` block carries ``expected``, ``actual``,
            ``info_total_episodes``, ``sources_agree``, ``episode_indices``,
            ``total_frames``, ``total_frames_per_ep`` and ``root`` so a caller
            (or CI) can fail loudly programmatically. ``status`` is ``"error"``
            both when the parquet count differs from ``expected`` AND when the
            parquet disagrees with ``meta/info.json``'s ``total_episodes``
            (``sources_agree`` is then ``False``) - the two metadata sources
            must agree, never just one.
        """
        if not isinstance(expected, int) or expected < 0:
            return {
                "status": "error",
                "content": [
                    {"text": f"verify_dataset_episodes: expected must be a non-negative int, got {expected!r}."}
                ],
            }

        root = self._active_dataset_root()
        if not root:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "verify_dataset_episodes: no active or recently-recorded dataset to verify. "
                            "Record one first (start_recording -> run_policy -> stop_recording)."
                        )
                    }
                ],
            }

        from strands_robots.dataset_recorder import read_dataset_episode_indices

        try:
            info = read_dataset_episode_indices(root)
        except FileNotFoundError as e:
            return {
                "status": "error",
                "content": [
                    {"text": f"verify_dataset_episodes: {e}"},
                    {
                        "json": {
                            "expected": expected,
                            "actual": 0,
                            "info_total_episodes": None,
                            "sources_agree": False,
                            "episode_indices": [],
                            "total_frames": 0,
                            "total_frames_per_ep": [],
                            "root": str(root),
                        }
                    },
                ],
            }
        except ImportError as e:
            return {"status": "error", "content": [{"text": f"verify_dataset_episodes: {e}"}]}

        actual = info["total_episodes"]
        info_total = info.get("info_total_episodes")

        # Two independent truths must agree: the parquet episode count AND the
        # meta/info.json total_episodes header. A dataset can report the right
        # parquet count yet carry a stale/inconsistent info.json (interrupted
        # finalize), so a parquet-only check is not sufficient. sources_agree is
        # True when info.json is absent (parquet is then the sole truth) or when
        # the header matches the parquet.
        sources_agree = info_total is None or info_total == actual
        ok = actual == expected and sources_agree
        status = "success" if ok else "error"

        if not sources_agree:
            verdict = "MISMATCH"
            text = (
                f"verify_dataset_episodes: {verdict} - meta/info.json reports "
                f"{info_total} episode(s) but the parquet holds {actual}; the "
                f"dataset metadata is inconsistent (expected {expected}). "
                f"Root: {root}"
            )
        else:
            verdict = "matches" if ok else "MISMATCH"
            text = (
                f"verify_dataset_episodes: {verdict} - expected {expected}, "
                f"found {actual} episode(s) in parquet "
                f"({info['total_frames']} total frames). Root: {root}"
            )
        return {
            "status": status,
            "content": [
                {"text": text},
                {
                    "json": {
                        "expected": expected,
                        "actual": actual,
                        "info_total_episodes": info_total,
                        "sources_agree": sources_agree,
                        "episode_indices": info["episode_indices"],
                        "total_frames": info["total_frames"],
                        "total_frames_per_ep": info["frames_per_episode"],
                        "root": str(root),
                    }
                },
            ],
        }

    def _episode_contract_fields(
        self, *, requested: int, completed: int, saved: int, flush_deferred: bool = False
    ) -> dict[str, Any]:
        """Build the episode-count truth fields for a ``run_policy`` json block.

        Returns ``n_episodes_requested`` / ``n_episodes_completed`` /
        ``episodes_saved`` plus ``dataset_episode_indices`` - the episode indices
        the active recorder reports so far (derived from the recorder's in-memory
        ``meta.total_episodes``; ``[]`` when not recording). Episodes are flushed
        to parquet only at ``stop_recording``/``finalize``, so this reflects the
        recorder bookkeeping; call :meth:`verify_dataset_episodes` after
        ``stop_recording`` for the definitive on-disk parquet count.

        ``flush_deferred`` marks the single-episode fast path while recording:
        the rollout's frames are buffered into the CURRENT episode and become one
        dataset episode at the next ``save_episode`` / ``stop_recording`` - they
        are not yet a distinct flushed episode, so ``episodes_saved`` is ``0``.
        """
        fields: dict[str, Any] = {
            "n_episodes_requested": requested,
            "n_episodes_completed": completed,
            "episodes_saved": saved,
            "dataset_episode_indices": [],
        }
        if flush_deferred:
            fields["episode_flush_deferred"] = True
        if self._is_recording():
            recorder = self._active_recorder()
            total = getattr(getattr(recorder, "dataset", None), "meta", None)
            total_episodes = int(getattr(total, "total_episodes", 0) or 0) if total is not None else 0
            fields["dataset_episode_indices"] = list(range(total_episodes))
        return fields

    def save_episode(self) -> dict[str, Any]:
        """Flush the current recording episode and begin a fresh one.

        Backends that support dataset recording override this (see the MuJoCo
        ``RecordingMixin``). The base has no recorder, so it returns a
        structured error rather than pretending to flush.
        """
        return {
            "status": "error",
            "content": [{"text": "save_episode: this backend does not support dataset recording."}],
        }

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
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> dict[str, Any]:
        """Start policy execution in a background thread (non-blocking).

        Default implementation: synchronous passthrough to ``run_policy``.
        Backends that support true background execution (like MuJoCo via
        its ``ThreadPoolExecutor``) should override.

        accepts ``n_steps`` (primary) or legacy ``max_steps`` as an
        alternate to ``duration``. See ``run_policy`` for conversion rules.
        ``policy_kwargs`` carries the per-call #300 goal payload through to
        ``policy.get_actions`` (see :meth:`run_policy`).
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
            policy_kwargs=policy_kwargs,
            seed=seed,
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

        ``speed`` is a playback-rate multiplier (1.0 = real time) and must be a
        positive number; a non-positive or non-numeric value is rejected with a
        structured error rather than raising or silently playing back at full
        speed. ``speed`` scales only the wall-clock playback rate: each recorded
        frame always advances physics for a full control period (derived from
        the dataset fps), so a position-servo robot reproduces the recorded
        trajectory instead of under-integrating it.

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
        policy_object: Policy | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        action_horizon: int = 8,
        seed: int | None = None,
        async_rtc: bool = False,
        rtc_inference_timeout_s: float | None = None,
        on_frame: Callable[[int, dict[str, Any], dict[str, Any]], None] | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Multi-episode policy evaluation via ``PolicyRunner.evaluate``.

        ``robot_name`` resolves like :meth:`run_policy`: ``None`` (the
        default) auto-selects the sole robot in a single-robot scene and
        errors with the candidate list only when the choice is ambiguous
        (multiple robots) or impossible (empty scene). This keeps the two
        sibling entry points consistent - a policy you just ran with
        ``run_policy()`` evals the same way with ``eval_policy()``.
        ``n_episodes`` default lowered from 10 to 1 (callers opt in to
        longer evals explicitly).

        ``policy_object`` mirrors :meth:`run_policy`: pass an already-built
        ``Policy`` to skip the ``create_policy`` round-trip (e.g. a loaded
        SmolVLA checkpoint you want to evaluate without re-instantiating).
        When omitted, the policy is built from ``policy_provider`` /
        ``policy_config``.

        ``control_frequency`` / ``control_substeps`` flow through to
        :meth:`PolicyRunner.evaluate` so the eval loop steps physics for the
        full control period per action (same servo-tracking semantics as
        :meth:`run_policy`). Without these the arm under-steps and the policy
        looks like a no-op (the arm under-steps each control period).

        ``async_rtc`` (default ``False``) opts into overlapping policy
        inference with action-chunk execution, evaluating a chunk-emitting
        policy under the realistic control latency it faces in deployment.
        It is forwarded to :meth:`PolicyRunner.evaluate`; the default keeps
        the success-rate synchronous and bit-stable. ``rtc_inference_timeout_s``
        bounds each async inference (structured error instead of a hung
        rollout). For benchmark-style latency masking use
        :meth:`run_policy` (``async_rtc=...``).

        ``on_frame`` is an optional ``(step, observation, action) -> None``
        hook fired per applied control step on the eval thread, immediately
        after ``sim.send_action`` - the success-rate analogue of the
        :meth:`run_policy` / :meth:`evaluate_benchmark` hook. ``step`` is a
        monotonic index that continues across episode boundaries. Use it to
        record frames or stream telemetry synchronously on the eval thread
        (e.g. paired with ``start_cameras_recording_synchronous``) so a
        daemon-thread recorder does not race ``mjData`` mutations. A hook
        exception is logged at WARN and never aborts the eval.

        ``n_episodes`` and ``max_steps`` must be positive integers and
        ``control_frequency`` must be ``> 0``; a non-positive value is
        rejected with a structured error at the entry point (before
        ``create_policy``) rather than running a degenerate eval that
        reports a fabricated success rate over zero/negative episodes.

        ``policy_kwargs`` is the per-call goal payload forwarded verbatim to
        every ``policy.get_actions(obs, instruction, **policy_kwargs)`` call,
        exactly as on :meth:`run_policy`. Goal-conditioned providers read their
        target from these well-known keys (``target_velocity`` for WBC and other
        locomotion policies; ``target_pose`` / ``target_joints`` / ``world_update``
        for cuRobo / MoveIt2 - the issue #300 contract). Without it the eval ran
        such a policy with an empty goal and reported a meaningless success rate.

        ``success_fn`` defaults to ``None``. With no ``success_fn`` (and no
        benchmark spec) there is no criterion by which an episode can be marked
        successful, so ``success_rate`` reports a hard ``0.0`` for every episode
        regardless of what the policy does - indistinguishable from a policy that
        genuinely failed every episode. This case logs a warning and sets
        ``success_measured=false`` in the returned json; pass
        ``success_fn="contact"`` (or a callable) to measure real task success.

        ``video`` optionally records one rollout MP4 PER EPISODE so an eval can
        be watched to see WHY episodes fail, not just read as an aggregate
        success rate. Same dict schema as :meth:`run_policy` (``path`` enables
        it; ``fps`` / ``camera`` / ``width`` / ``height``); the path is
        validated and the camera probed up-front. ``_ep{i}`` is inserted into
        the filename per episode (``eval.mp4`` -> ``eval_ep0.mp4``,
        ``eval_ep1.mp4``, ...) so episodes never overwrite each other, and the
        written files are returned in the result json ``video_paths``. Recording
        is unsupported on the benchmark (``evaluate_benchmark``) path.
        """
        robots = self.list_robots()
        if not robots:
            return {"status": "error", "content": [{"text": "No robots in sim. Add one first."}]}
        try:
            resolved_robot = self._resolve_single_robot(robot_name)
        except ValueError as exc:
            return {"status": "error", "content": [{"text": str(exc)}]}
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": f"Robot '{resolved_robot}' not found."}],
            }

        if err := self._validate_action_horizon(action_horizon, "eval_policy"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "eval_policy"):
            return err
        if err := self._validate_positive_int(max_steps, "max_steps", "eval_policy"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "eval_policy"):
            return err

        if policy_object is None:
            from strands_robots.policies import create_policy

            # Fail fast on a misconfiguration BEFORE the create_policy download.
            preflight_error = self._preflight_policy_config(resolved_robot, policy_provider, policy_config)
            if preflight_error is not None:
                return preflight_error
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - mirror run_policy. Caller may have already
            # set robot_state_keys; we set defensively so semantics match the
            # provider path.
            policy = policy_object
        policy.set_robot_state_keys(self.robot_action_keys(resolved_robot))
        self.bind_policy_sim_context(policy, resolved_robot)

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            max_steps=max_steps,
            success_fn=success_fn,
            control_frequency=control_frequency,
            control_substeps=control_substeps,
            action_horizon=action_horizon,
            seed=seed,
            async_rtc=async_rtc,
            rtc_inference_timeout_s=rtc_inference_timeout_s,
            on_frame=on_frame,
            policy_kwargs=policy_kwargs,
            video=video,
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
        policy_kwargs: dict[str, Any] | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        policy_object: Policy | None = None,
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
            n_episodes: Number of episodes. Must be a positive integer;
                a zero/negative/non-int value is rejected with a structured
                error rather than fabricating a 0%-success report over an
                empty rollout loop.
            seed: Master RNG seed for per-episode reproducibility.
            action_horizon: How many actions to consume from each
                ``policy.get_actions(...)`` chunk before re-querying the
                policy. Default ``8`` matches NVIDIA's upstream
                GR00T LIBERO eval (``MultiStepWrapper`` with
                ``n_action_steps=8``) - the policy commits to 8 actions
                before re-observing, which is what GR00T-N1.7-LIBERO
                checkpoints were trained against. Set to ``1`` for
                closed-loop receding-horizon control (re-observe every
                step; matches OpenVLA-style eval) ONLY for single-action
                policies: the interval is clamped up to the policy's
                ``execution_horizon`` (``resolve_chunk_length``), so a
                chunk-emitting policy (e.g. a VLA) still consumes its full
                chunk open-loop regardless of this value. Values < 1 are
                rejected with a structured error. ``on_step`` and
                success/failure checks run after EACH applied action,
                so per-step rewards and early termination work
                correctly regardless of horizon.
            on_frame: Optional ``(step, observation, action) -> None``
                hook fired per applied control step on the eval thread,
                immediately after ``sim.send_action``. Use this for
                synchronous recording or telemetry when the eval is
                dispatched from a thread distinct from the script main
                (e.g. Strands ``Agent`` tool dispatch under asyncio) -
                the daemon-thread recorder
                (:meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording`)
                races ``mjData`` mutations on the eval thread under that
                pattern and produces 2-3% frame-capture rates with
                greenish GL clear-colour artifacts. Pair with
                :meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording_synchronous`
                for the recorder side. See #191.
            policy_kwargs: Per-call goal payload forwarded verbatim to every
                ``policy.get_actions(obs, instruction, **policy_kwargs)`` call
                (same contract as :meth:`run_policy` / :meth:`eval_policy`).
                Goal-conditioned providers read their target from these keys
                (``target_velocity`` / ``target_pose`` / ``target_joints`` /
                ``world_update``); a benchmark that drives such a policy must
                pass them or the policy runs with an empty goal.
            control_frequency: Target Hz for ``policy.get_actions`` calls, used
                to derive the physics substeps executed per action
                (``round(1 / control_frequency / physics_timestep)``) so the
                benchmark loop steps a full control period per action. Must be
                ``> 0``; a non-positive value is rejected with a structured
                error. Defaults to ``50.0`` (same default as :meth:`eval_policy`).
                Set it to the rate the policy was trained/evaluated at - a
                benchmark's ``max_steps`` maps to a wall-clock episode length
                that depends on this rate, so a mismatched frequency changes the
                effective episode horizon.
            control_substeps: Explicit physics substeps per action, overriding
                the ``control_frequency``-derived value (mirrors
                :meth:`eval_policy`). ``None`` (default) derives it from
                ``control_frequency``.
            policy_object: An already-built :class:`Policy` to evaluate,
                skipping the ``create_policy`` round-trip (mirrors
                :meth:`run_policy` / :meth:`eval_policy`). Use it to benchmark a
                checkpoint you have already loaded - e.g. a multi-GB VLA - once
                per process instead of reloading it on every benchmark call.
                When ``None`` the policy is built from ``policy_provider`` /
                ``policy_config``.

        Returns:
            Standard status dict. On success, carries per-episode cumulative
            reward + aggregate success_rate / avg_reward / avg_steps in the
            JSON payload.
        """
        from strands_robots.policies import create_policy
        from strands_robots.simulation.benchmark import get_benchmark

        if err := self._validate_action_horizon(action_horizon, "evaluate_benchmark"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "evaluate_benchmark"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "evaluate_benchmark"):
            return err

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

        if policy_object is None:
            policy = create_policy(policy_provider, **(policy_config or {}))
        else:
            # Pre-built policy path - mirror run_policy / eval_policy. Lets a
            # caller benchmark an already-loaded checkpoint (e.g. a multi-GB
            # VLA) without a create_policy round-trip / redundant reload.
            policy = policy_object
        policy.set_robot_state_keys(self.robot_action_keys(resolved_robot))
        self.bind_policy_sim_context(policy, resolved_robot)

        return PolicyRunner(self).evaluate(
            resolved_robot,
            policy,
            instruction=instruction,
            n_episodes=n_episodes,
            spec=spec,
            seed=seed,
            action_horizon=action_horizon,
            control_frequency=control_frequency,
            control_substeps=control_substeps,
            on_frame=on_frame,
            policy_kwargs=policy_kwargs,
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
                        f"Registered benchmark '{benchmark_name}' from {spec_path}\n"
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

    def set_obs_noise(self, **kwargs: Any) -> dict[str, Any]:
        """Configure additive sensor noise on observations.

        Models real-sensor measurement noise (joint encoders, camera frames)
        so policies are not trained on noise-free observations. Concrete
        backends define their own parameter signatures. Override per backend.
        """
        raise NotImplementedError("set_obs_noise not implemented by this backend")

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
                "send_action": "(action: dict, robot_name: str | None = None, n_substeps: int = 1) -> dict",
                "add_robot": (
                    "(name: str, urdf_path=None, data_config=None, position=None, "
                    "orientation=None) -> dict  # add a robot to the scene by "
                    "registry name (or urdf_path); the first scene-construction step"
                ),
                "add_object": (
                    "(name: str, shape='box', position=None, orientation=None, "
                    "size=None, color=None, mass=0.1, is_static=None, mesh_path=None, "
                    "material=None) -> dict  # add a manipulable object "
                    "(cube/sphere/.../mesh) to the scene. material is an optional "
                    "dict for matte/textured surfaces: keys reflectance|specular|"
                    "shininess (0..1), texture (abs image path) OR builtin "
                    "(checker|gradient|flat) + rgb1/rgb2/texdim, texrepeat [u,v]"
                ),
                "remove_object": "(name: str) -> dict  # remove a previously added object",
                "run_policy": "(robot_name: str, policy_provider='mock', n_episodes=1, reset_between=True, ...) -> dict",
                "start_policy": "(robot_name: str, policy_provider='mock', ...) -> dict",
                "eval_policy": (
                    "(robot_name: str, policy_provider='mock', n_episodes=1, "
                    "max_steps=300, success_fn=None, ...) -> dict  # multi-episode "
                    "success-rate evaluation (the rollout sibling of run_policy)"
                ),
                "replay_episode": (
                    "(repo_id: str, robot_name=None, episode=0, root=None, "
                    "speed=1.0, action_key_map=None) -> dict  # replay a recorded "
                    "LeRobotDataset episode through the sim"
                ),
                "list_robots": "() -> list[str]",
                "render": "(camera_name='default', width=None, height=None) -> dict",
                "reset": "() -> dict  # during recording, flushes the buffered rollout as one episode before resetting",
                "step": "(n_steps: int = 1) -> dict",
            },
            "note": (
                "robot_name defaults to the sole robot when only one exists "
                "for get_observation, send_action, get_robot_state, run_policy, "
                "and start_policy. With multiple robots, pass robot_name "
                "explicitly (from the 'robots' list above)."
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
