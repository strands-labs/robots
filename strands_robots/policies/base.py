"""Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

The :class:`Policy` ABC is intentionally agnostic about *how* actions are
produced.  Built-in providers (`mock`, `groot`, `lerobot_local`) are VLA-style,
but the same interface is the right shape for:

* **Classical motion planners** - cuRobo, MoveIt2, OMPL, RRT*: take a goal
  pose and joint state, return a collision-free trajectory.
* **Model-predictive controllers** (MPC) - solve a finite-horizon optimal
  control problem each tick.
* **Scripted / pure-IK trajectories** - analytic IK followed by interpolation;
  zero learning involved.

Non-VLA implementations typically set :attr:`Policy.requires_images` to
``False`` to skip camera rendering (~10x throughput win at 500Hz) and read
their goal from the well-known ``**kwargs`` keys documented on
:meth:`Policy.get_actions` rather than parsing the natural-language
``instruction`` string.

See :class:`~strands_robots.policies.mock.MockPolicy` for the canonical
non-VLA reference implementation.
"""

import asyncio
import concurrent.futures
from abc import ABC, abstractmethod
from typing import Any, Protocol, runtime_checkable


class Policy(ABC):
    """Abstract base class for robot policies (VLA, motion planners, MPC, scripted).

    All policies implement async :meth:`get_actions`.  For convenience, a
    synchronous wrapper :meth:`get_actions_sync` is provided.

    The interface is general enough to cover both **VLA-style** providers
    (consume images + instruction, output joint targets) and **non-VLA**
    providers such as classical motion planners (cuRobo, MoveIt2),
    model-predictive controllers, and pure-IK / scripted trajectories.
    Non-VLA providers typically set :attr:`requires_images` to ``False``
    and read their goal from the well-known ``**kwargs`` keys documented
    on :meth:`get_actions`.

    All providers MUST honour the per-tick **action value convention**
    documented on :meth:`get_actions`: each action value is a python
    ``float`` (single-DOF) or ``list[float]`` (multi-DOF group), never a
    raw ``np.ndarray``, so downstream consumers handle every provider's
    output uniformly regardless of its internal compute backend. See
    ``MockPolicy`` for the canonical reference.
    """

    #: Control rate (Hz) at which the consumer loop executes the actions this
    #: policy returns. Set by the runtime (e.g. ``PolicyRunner``) via
    #: :meth:`set_control_frequency` BEFORE the rollout loop starts, so
    #: latency-sensitive providers can convert wall-clock inference latency into
    #: a count of action steps consumed during inference (Real-Time Chunking).
    #: ``None`` means the runtime has not told the policy its clock yet -
    #: providers that need it MUST warn loudly and fall back rather than
    #: silently assuming a rate (a wrong assumed rate corrupts RTC blending at
    #: every control frequency except the assumed one).
    control_frequency: float | None = None

    def set_control_frequency(self, hz: float) -> None:
        """Tell the policy the control rate (Hz) of the executing loop.

        The runtime that drives the policy (``PolicyRunner.run`` / ``evaluate``)
        calls this once before the rollout loop so providers that estimate an
        inference delay in *action steps* (Real-Time Chunking) can convert their
        measured wall-clock latency into the correct number of steps. Without
        it, such providers fall back to a hardcoded rate and silently mis-blend
        chunks at any other control frequency.

        Args:
            hz: Positive control frequency in Hz.

        Raises:
            ValueError: If ``hz`` is not strictly positive.
        """
        if hz <= 0:
            raise ValueError(f"control_frequency must be positive, got {hz}")
        self.control_frequency = float(hz)

    #: Number of control steps the executing loop runs between issuing an
    #: inference request and applying the FIRST action it returns. The runtime
    #: (e.g. ``PolicyRunner``) sets this via :meth:`set_rtc_observed_delay`
    #: immediately before each ``get_actions`` call so latency-sensitive
    #: providers (Real-Time Chunking) can slice the leftover chunk by the EXACT
    #: number of steps that elapsed rather than estimating it from wall-clock
    #: latency. The estimate is non-reproducible (it warms up within an episode
    #: and varies run-to-run), which silently perturbs otherwise-identical
    #: seeded episodes; a counted integer is deterministic. ``None`` means the
    #: runtime did not supply a count, so providers fall back to their
    #: wall-clock estimate (appropriate for true-async hardware driven without a
    #: runner, where the robot really does move during inference).
    rtc_observed_delay_steps: int | None = None

    def set_rtc_observed_delay(self, steps: int | None) -> None:
        """Tell the policy how many control steps elapse during inference.

        The runtime that drives the policy calls this before each
        ``get_actions`` so Real-Time Chunking providers can compute the
        chunk-seam offset deterministically instead of deriving it from
        wall-clock latency. In a synchronous eval loop the world is paused
        during inference, so exactly ``0`` steps elapse; in the async overlap
        pipeline the count is the number of still-pending steps of the chunk
        being executed. Either way it is a known integer, not a measurement.

        Args:
            steps: Non-negative control-step count, or ``None`` to clear the
                override and let the provider fall back to its wall-clock
                estimate.

        Raises:
            ValueError: If ``steps`` is negative.
        """
        if steps is not None and steps < 0:
            raise ValueError(f"rtc_observed_delay_steps must be >= 0, got {steps}")
        self.rtc_observed_delay_steps = None if steps is None else int(steps)

    @abstractmethod
    async def get_actions(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).  VLA
                providers consume both ``observation.images.*`` and
                ``observation.state``.  Non-VLA providers typically
                consume ``observation.state`` only and set
                :attr:`requires_images` to ``False`` to skip camera
                rendering.
            instruction: Natural language instruction.  Required by the
                signature for VLA providers; non-VLA providers (motion
                planners, MPC, scripted) may ignore it and read the goal
                from ``**kwargs`` instead.
            **kwargs: Provider-specific parameters.  The following keys
                are **well-known** and SHOULD be honoured by non-VLA
                providers when present so callers don't have to JSON-encode
                goals into the ``instruction`` string:

                - ``target_pose: list[float]`` - Cartesian goal as
                  ``[x, y, z, qw, qx, qy, qz]`` (position in metres,
                  orientation as a unit quaternion in the robot base frame).
                - ``target_joints: dict[str, float]`` - joint-space goal
                  keyed by joint name; values are in radians (revolute) or
                  metres (prismatic).
                - ``world_update: dict | None`` - per-call world refresh
                  for collision-aware planners (e.g. point cloud / depth
                  image / mesh updates).  ``None`` means "reuse the world
                  configured at init time".

                Providers MUST ignore unknown ``**kwargs`` rather than
                raising, so callers can pass shared keys across providers.

        Returns:
            List of action dicts for robot execution.  Each dict maps a
            robot state key (joint/actuator name) to its **target value**
            for that tick.

            Values MUST be **JSON / python-native**: a python ``float`` for
            a single-DOF actuator, or a ``list[float]`` for a multi-DOF
            actuator group.  Implementations MUST NOT return raw
            ``np.ndarray`` objects -- coerce with ``.tolist()`` /
            ``float(...)`` before returning -- so downstream consumers can
            treat every provider's output uniformly (e.g. ``float(v)`` on a
            scalar, ``len(v)`` on a group) regardless of the policy's
            internal compute backend.

            The list length is the action-chunk horizon; consumers execute
            it at a fixed control rate (e.g. 50Hz).
        """
        pass

    def get_actions_sync(
        self, observation_dict: dict[str, Any], instruction: str, **kwargs: Any
    ) -> list[dict[str, Any]]:
        """Synchronous convenience wrapper around get_actions().

        Safe to call from sync code, event loops, or notebooks.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                return pool.submit(
                    asyncio.run,
                    self.get_actions(observation_dict, instruction, **kwargs),
                ).result()
        else:
            return asyncio.run(self.get_actions(observation_dict, instruction, **kwargs))

    @abstractmethod
    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Configure the policy with robot state keys."""
        pass

    def reset(self, seed: int | None = None) -> None:
        """Reset per-episode policy state.

        Default implementation is a no-op. Policies that hold per-episode
        state (e.g. diffusion sampler RNG, action chunk caches, KV-caches)
        should override to apply the reset.

        For SERVICE-mode policies (e.g. ``Gr00tPolicy(host=...)`` over
        ZMQ), the override forwards the call to the server so its
        per-episode RNG state can be re-initialised - without this,
        ``set_eval_seed`` only seeds the client-side process, leaving
        the server's diffusion sampler RNG drifting across calls and
        breaking reproducibility (#187).

        Args:
            seed: Optional master seed forwarded to the policy's
                random-number generators. When ``None``, implementations
                may apply a default seed or leave RNG state untouched.
        """
        # Default no-op. Concrete policies override to apply per-episode
        # state reset (RNG seeding, action-cache flush, server-side
        # reset endpoint call, etc.).
        return None

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        """Cheap pre-construction validation hook (no download, no instantiation).

        Called by the simulation's ``run_policy`` / ``eval_policy`` BEFORE
        :func:`~strands_robots.policies.create_policy` builds the policy - and
        therefore before any model weight download - with the set of
        observation keys the runtime will feed the policy. Override this to
        fail fast on a misconfiguration (e.g. sim camera names that cannot be
        routed to the model's declared image inputs) instead of surfacing it as
        a confusing failure deep inside inference after a multi-minute weight
        download.

        The default implementation is a no-op. Implementations MUST be cheap:
        no network access, no model instantiation - only local metadata
        (``policy_config`` plus packaged JSON such as the embodiment registry)
        and the provided ``observation_keys``.

        Args:
            observation_keys: Keys present in the runtime observation dict the
                policy will receive (joint state names + attached camera
                names), as returned by ``SimEngine.get_observation``.
            **policy_config: The same provider kwargs that will be forwarded to
                the policy constructor by ``create_policy``.

        Raises:
            ValueError: When the configuration cannot consume the runtime
                observation (e.g. a required camera source key is absent and no
                override maps an available key onto the model's image feature).
        """
        return None

    @property
    def requires_images(self) -> bool:
        """Whether this policy needs camera frames in its observation.

        Default ``True`` (most VLA policies do). Subclasses that only
        consume joint state (e.g. ``MockPolicy``, classical motion planners
        such as cuRobo / MoveIt2, MPC, pure-IK controllers, scripted
        trajectories) can return ``False`` to let the simulation skip
        expensive camera rendering - a ~10x throughput win at 500Hz when
        no cameras are needed.
        """
        return True

    @property
    def execution_horizon(self) -> int:
        """Number of actions the SIM consumes from one ``get_actions`` chunk before re-querying.

        This is the SINGLE source of truth for the re-query interval; a chunk
        consumer (the single-policy runner, the multi-episode eval loop, the
        synchronized multi-robot loop) reads it via
        :func:`resolve_chunk_length` and never inspects ``actions_per_step``
        directly. Distinguishing the re-query interval from the trained chunk
        length is what makes Real-Time Chunking (RTC) actually engage:

        * **RTC policy** -> the RTC ``execution_horizon`` (typically << the
          trained chunk). The policy is re-queried mid-chunk so it can blend
          the unexecuted tail of the previous chunk (``prev_chunk_left_over``)
          into the next one. Re-querying only after the full trained chunk
          drains leaves that tail permanently empty and silently degrades RTC
          to plain open-loop replay.
        * **chunked open-loop** (ACT, diffusion, pi0/SmolVLA without RTC) ->
          ``actions_per_step`` (the trained chunk; truncating drops its tail
          and forces an out-of-distribution re-query).
        * **single-step** (``MockPolicy``, classical planners) -> ``1``.

        The default derives from ``actions_per_step`` (``1`` when undeclared),
        so a single-step or chunked open-loop policy needs no override; only a
        policy with an inference-time budget distinct from its trained chunk
        (RTC) overrides this.
        """
        intended = getattr(self, "actions_per_step", 1)
        try:
            intended_int = int(intended)
        except (TypeError, ValueError):
            intended_int = 1
        return max(1, intended_int)

    def is_chunk_emitting(self) -> bool:
        """Whether this policy returns multi-action chunks per ``get_actions``.

        A chunk-emitting policy (ACT, diffusion, pi0, pi0.5, pi0-FAST, SmolVLA,
        MolmoAct2) returns more than one action per inference, so its inference
        latency can be hidden behind the EXECUTION of the current chunk while the
        next chunk is computed in the background. The async-RTC pipeline in
        :meth:`PolicyRunner.run` uses this signal to auto-enable latency masking
        for exactly the policies that benefit (``run_policy(async_rtc=None)``);
        single-step policies (``MockPolicy``, classical planners) gain nothing
        from overlap and stay on the synchronous loop.

        The default derives the answer from the re-query interval the consumer
        actually drives - :attr:`execution_horizon` - so ANY policy that emits a
        chunk longer than one action is detected without enumerating provider
        names: a model under RTC reports its RTC horizon (> 1), a chunked
        open-loop model reports its trained chunk length (> 1), and a single-step
        policy reports ``1``. Providers whose chunk shape is not visible through
        ``execution_horizon`` (e.g. a model that must be driven via
        ``predict_action_chunk``) override this.

        Returns:
            ``True`` when the policy emits multi-action chunks; ``False`` for
            single-step policies.
        """
        return self.execution_horizon > 1

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Get provider name for identification."""
        pass


@runtime_checkable
class ChunkedPolicy(Protocol):
    """Introspection contract for policies that emit ACTION CHUNKS.

    A *chunked* policy returns more than one action per
    :meth:`Policy.get_actions` call: a model trained for N-step open-loop replay
    (ACT, diffusion, pi0, SmolVLA, MolmoAct2) emits a length-N chunk that a
    consumer executes before re-querying. The chunk PRODUCER is the existing
    async :meth:`Policy.get_actions` - this protocol deliberately does NOT add a
    second chunk-producing method (that would split one contract across two
    code paths); it only surfaces the metadata a consumer needs to drive an
    already-produced chunk correctly.

    Every consumer of a chunk (the single-policy runner, the multi-episode eval
    loop, and the synchronized multi-robot loop) must size the chunk the same
    way - see :func:`resolve_chunk_length`. Routing all of them through one
    helper that reads this contract keeps a chunk-emitting policy from being
    truncated differently depending on which loop happens to drive it.

    The protocol is ``runtime_checkable`` so a consumer can branch on
    ``isinstance(policy, ChunkedPolicy)`` and a type checker rejects a
    non-chunked policy where a chunked one is required.

    Attributes:
        actions_per_step: Number of actions the policy intends a consumer to
            execute open-loop from one ``get_actions`` chunk before re-querying
            (the policy's trained chunk length). Truncating below this drops the
            chunk tail and forces an out-of-distribution re-query.
        supports_rtc: Whether the policy blends chunk seams internally via
            Real-Time Chunking - it carries prev-chunk state across re-queries
            so consecutive chunks join smoothly. Introspection only; a consumer
            never has to drive RTC, the policy does it inside ``get_actions``.
    """

    actions_per_step: int
    supports_rtc: bool


def resolve_chunk_length(policy: "Policy", action_horizon: int) -> int:
    """Effective number of actions to consume from one ``get_actions`` chunk.

    Centralizes the single re-query rule every consumer must apply identically.
    The number of actions consumed before re-querying is the policy's
    :attr:`Policy.execution_horizon` - the single source of truth - never
    ``actions_per_step`` read directly. How ``action_horizon`` interacts with it
    depends on whether the policy carries cross-chunk state (RTC):

    * **RTC policy** (``supports_rtc`` is true): the policy hard-decides the
      interval and is re-queried at exactly its ``execution_horizon`` so it can
      blend the unexecuted tail of the previous chunk into the next one. A
      caller-supplied ``action_horizon`` must NOT stretch (or shrink) this
      interval - doing so leaves ``prev_chunk_left_over`` empty and silently
      degrades RTC to plain open-loop replay. ``action_horizon`` is ignored.
    * **non-RTC** (open-loop chunked or single-step): consume
      ``max(action_horizon, execution_horizon)`` so a model trained for N-step
      replay (``execution_horizon == actions_per_step == N``) keeps its FULL
      chunk - clamping to a smaller ``action_horizon`` drops the chunk tail and
      forces an out-of-distribution re-query. Single-action providers
      (``MockPolicy``) have ``execution_horizon == 1`` so the result is just
      ``max(action_horizon, 1)``.

    Before this helper existed each consumer inlined the same
    ``max(action_horizon, getattr(policy, "actions_per_step", 1))`` expression
    and they drifted; worse, all of them keyed off ``actions_per_step``, so an
    RTC policy was re-queried only after its full trained chunk drained and its
    cross-chunk blending never engaged.

    Args:
        policy: Any policy. The re-query interval is read from
            :attr:`Policy.execution_horizon` (falling back to a raw
            ``actions_per_step`` attribute for duck-typed objects that are not
            ``Policy`` subclasses); a policy that declares neither is treated as
            single-action.
        action_horizon: Consumer-requested actions per chunk (clamped to >= 1).
            Ignored for RTC policies, which decide their own interval.

    Returns:
        The number of leading chunk actions to execute before re-querying.
    """
    horizon = getattr(policy, "execution_horizon", None)
    if horizon is None:
        # Duck-typed object that is not a ``Policy`` subclass (no
        # execution_horizon property): fall back to its raw chunk length.
        horizon = getattr(policy, "actions_per_step", 1)
    try:
        horizon_int = 1 if horizon is None else int(horizon)
    except (TypeError, ValueError):
        horizon_int = 1
    if horizon_int < 1:
        horizon_int = 1
    if getattr(policy, "supports_rtc", False):
        # RTC owns the interval; action_horizon cannot override it.
        return horizon_int
    return max(int(action_horizon), 1, horizon_int)
