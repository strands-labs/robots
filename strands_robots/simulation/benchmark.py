"""Benchmark-agnostic evaluation protocol for any ``SimEngine``.

Every standard benchmark (LIBERO, Meta-World, RoboSuite, ManiSkill, user-authored
tasks) has a different notion of "what a task is" - sparse-success, dense-reward,
procedural scenes, BDDL predicates, hardcoded robots, etc. The correct abstraction
is the protocol the eval loop calls into, not a benchmark-specific schema.

:class:`BenchmarkProtocol` is that protocol. Each adapter implements a handful of
lifecycle hooks (``on_episode_start``, ``on_step``, ``is_success``, ``is_failure``)
and declares the robots it is compatible with. The evaluation loop
(:meth:`~strands_robots.simulation.policy_runner.PolicyRunner.evaluate`) drives
the protocol without knowing anything about the underlying benchmark.

Adapters live in optional extras (``strands-robots[benchmark-libero]`` etc.);
the core package stays dependency-free. A reference :class:`DeclarativeBenchmark`
shipped in :mod:`strands_robots.simulation.benchmark_spec` turns a YAML/JSON
spec into a fully functional ``BenchmarkProtocol`` instance - LLMs can author
and register benchmarks at runtime without writing Python code.

Registry: a module-level ``dict[str, BenchmarkProtocol]`` keyed by name,
mirroring the shape of :func:`~strands_robots.simulation.model_registry.register_urdf`.
Registration is idempotent-by-overwrite: re-registering the same name replaces
the previous entry and logs a warning. This matches how users iterate on a
spec file during development.

Thread safety: the registry is guarded by an internal lock so concurrent
registrations from agent threads do not race. The benchmark instances
themselves are expected to be immutable after registration - adapters that
keep per-episode state MUST put it on the ``rng``-scoped call, not on ``self``.
"""

from __future__ import annotations

import logging
import random
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StepInfo:
    """Per-step feedback from a :class:`BenchmarkProtocol`.

    Returned by :meth:`BenchmarkProtocol.on_step`. The evaluation loop
    accumulates ``reward`` across steps and terminates the episode early
    when ``done`` is ``True`` *or* when :meth:`BenchmarkProtocol.is_success`
    / :meth:`BenchmarkProtocol.is_failure` fires.

    Attributes:
        reward: Dense reward for this step. Sparse-success benchmarks
            return ``0.0`` on every step that isn't a terminal success.
        done: Early-termination flag - set when the benchmark knows the
            episode is over (e.g. the object fell off the table and
            nothing further will happen).
        info: Free-form metadata propagated into the per-episode result
            under the ``info`` key. Safe for small scalars / diagnostics -
            do NOT stuff large tensors here.
    """

    reward: float = 0.0
    done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BenchmarkProtocol(ABC):
    """Protocol every benchmark (LIBERO, Meta-World, custom) implements.

    Subclass this, declare :attr:`supported_robots` + :attr:`default_robot`,
    and implement :meth:`on_step` + :meth:`is_success`. The default
    :meth:`on_episode_start` validates robot compatibility and auto-loads
    :attr:`default_robot` when the sim is empty, which is the right behaviour
    for 90% of adapters; override only if you need per-episode scene setup
    beyond what :meth:`SimEngine.reset` provides.

    Robot compatibility is first-class metadata. LIBERO's BDDL and scene
    files reference Panda body names, Meta-World hardcodes Sawyer, RoboSuite
    parameterizes over a fixed robot list - without declaring which robots a
    benchmark accepts, agents will silently evaluate with the wrong robot.
    :meth:`~strands_robots.simulation.policy_runner.PolicyRunner.evaluate`
    validates the sim's robot against :attr:`supported_robots` before episode
    1 and returns a structured error on mismatch.

    Attributes:
        max_steps: Per-episode horizon. Instance attribute (not abstract
            property) so subclasses can set it in ``__init__`` or as a class
            attribute. Defaults to ``300``.
    """

    max_steps: int = 300

    # Robot compatibility (first-class metadata)

    @property
    @abstractmethod
    def supported_robots(self) -> list[str]:
        """Registry ``data_config`` names this benchmark accepts.

        Empty list means "any robot" (unusual; dense-reward benchmarks rarely
        generalise across embodiments). LIBERO-shaped adapters should return
        a closed list of Panda variants; Meta-World should return its Sawyer
        variants; etc.
        """

    @property
    @abstractmethod
    def default_robot(self) -> str:
        """Robot :meth:`on_episode_start` loads when the sim is empty.

        Must be an element of :attr:`supported_robots` (or any compatible
        registry name when ``supported_robots`` is empty). Declared
        separately from ``supported_robots[0]`` so multi-robot benchmarks
        can be explicit about their canonical default.
        """

    @property
    def instruction(self) -> str:
        """Natural-language instruction for the policy.

        Default ``""`` (no instruction). Benchmarks that ship a per-task
        language description should override to return it (e.g. LIBERO's
        BDDL ``:language`` clause). The
        :class:`~strands_robots.simulation.policy_runner.PolicyRunner`
        eval loop falls back to this property when the user-supplied
        ``instruction=`` argument to ``evaluate_benchmark`` is empty.
        Without it, language-conditioned policies (GR00T, OpenVLA, etc.)
        receive an empty string and produce off-task actions, manifesting
        as low ``success_rate`` on the same checkpoint that succeeds when
        the instruction is provided. #187 GPU bisection isolated this
        as the dominant cause of the LIBERO-10/SCENE5 ZMQ-vs-in-process
        gap (0.40 vs 1.00 \u2014 same model, same wire format, same engine,
        but ``policy_obs['annotation.human.action.task_description']=['']``
        on the failing path vs the actual task description on the
        passing path).
        """
        return ""

    # Lifecycle hooks

    def on_episode_start(self, sim: SimEngine, rng: random.Random) -> None:
        """Per-episode init. Called after ``sim.reset()`` and before the first obs.

        Default implementation enforces robot compatibility:

        * If the sim has no robots, add :attr:`default_robot` via
          ``sim.add_robot(name="robot", data_config=default_robot)``.
        * Otherwise, validate that every loaded robot's ``data_config`` is
          in :attr:`supported_robots` (when non-empty). Mismatches raise
          :class:`BenchmarkCompatibilityError` - the eval loop catches that
          and returns a structured error with the allowed list.

        Override to layer on per-episode randomization, goal sampling, or
        procedural scene generation. Always call ``super().on_episode_start``
        first unless you deliberately want to skip compatibility checks.

        Args:
            sim: The engine being driven.
            rng: Seeded per-episode RNG. Always use this - don't create your
                own ``random.Random()`` or seeding will be non-reproducible.
        """
        robots = sim.list_robots()
        if not robots:
            sim.add_robot(name="robot", data_config=self.default_robot)
            return

        # Validate all loaded robots against supported_robots
        supported = self.supported_robots
        if not supported:  # empty list means "any robot"
            return

        # data_config lookup is backend-specific; MuJoCo stores it on SimRobot.
        # Reach into sim._world if available (cheap duck-typing); otherwise skip
        # the check rather than false-positive error. Adapters needing stricter
        # checks should override on_episode_start.
        world = getattr(sim, "_world", None)
        if world is None or not hasattr(world, "robots"):
            return
        for rname in robots:
            robot_obj = world.robots.get(rname)
            if robot_obj is None:
                continue
            data_config = getattr(robot_obj, "data_config", None)
            if data_config is None or data_config in supported:
                continue
            raise BenchmarkCompatibilityError(
                robot_name=rname,
                data_config=data_config,
                supported=supported,
            )

    @abstractmethod
    def on_step(self, sim: SimEngine, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        """Return dense reward + done flag + info dict for this step.

        Called after every ``sim.send_action(action)`` with the observation
        that produced the action and the action itself. Sparse-success
        benchmarks return ``StepInfo(reward=0.0, done=False)`` on every
        non-terminal step.
        """

    def augment_observation(
        self,
        sim: SimEngine,
        obs: dict[str, Any],
    ) -> dict[str, Any]:
        """Hook to enrich the per-step observation with benchmark-specific keys.

        Called by :meth:`PolicyRunner._evaluate_with_spec` between
        ``sim.get_observation(...)`` and ``policy.get_actions(...)``. The
        returned dict is what the policy actually sees - so this is the
        place to bridge a sim's observation schema (typically joint-space)
        to whatever shape the benchmark's policy was trained on
        (Cartesian end-effector pose, dataset-specific encodings, …).

        Default implementation is identity (no-op). Subclasses MUST return
        a dict; mutating ``obs`` in place and returning it is allowed but
        returning a new merged dict is preferred to avoid surprising
        downstream code that retained a reference to the pre-augmented
        observation.

        Side-effect contract: implementations MUST be safe to call N times
        per step (the eval loop guarantees one call per step today, but
        future replay / preview features may call it more often). Do not
        send actions, do not step physics, do not mutate the registry.

        Errors are caught by the eval loop and surfaced as a structured
        error dict; raising here aborts the episode without further policy
        calls. See :class:`~strands_robots.benchmarks.libero.LiberoAdapter`
        for an example that injects ``x`` / ``y`` / ``z`` / ``roll`` / ``pitch``
        / ``yaw`` / ``gripper`` for the LIBERO ``state.*`` schema.
        """
        return obs

    @abstractmethod
    def is_success(self, sim: SimEngine) -> bool:
        """Terminal success predicate.

        Called every step after :meth:`on_step`. Returning ``True`` ends
        the episode with ``success=True``. Must be side-effect-free: the
        evaluation loop may call this multiple times per step depending on
        how backends batch success / failure checks.
        """

    def is_failure(self, sim: SimEngine) -> bool:
        """Optional early-termination failure condition.

        Default: always ``False``. Override to end an episode early without
        marking success (e.g. the arm self-collided, the object fell off
        the table, the agent picked the wrong object). Failure ends the
        episode; it does not count as a success.
        """
        return False


class BenchmarkCompatibilityError(ValueError):
    """Raised when a benchmark's robot compatibility check fails.

    Carries enough context (robot name, loaded data_config, supported list)
    for the eval loop to produce an actionable structured error. Subclasses
    :class:`ValueError` so code that uses broad ``except ValueError`` still
    catches it cleanly.
    """

    def __init__(self, robot_name: str, data_config: str, supported: list[str]):
        self.robot_name = robot_name
        self.data_config = data_config
        self.supported = list(supported)
        super().__init__(
            f"Robot '{robot_name}' (data_config={data_config!r}) is not compatible "
            f"with this benchmark. Supported: {self.supported}"
        )


# Registry

# Module-level registry - mirrors model_registry._URDF_REGISTRY. Mutable dict
# plus an RLock for thread safety; registry ops are cheap so we do not shard.
_BENCHMARK_REGISTRY: dict[str, BenchmarkProtocol] = {}
_REGISTRY_LOCK = threading.RLock()


def register_benchmark(name: str, benchmark: BenchmarkProtocol) -> None:
    """Register a :class:`BenchmarkProtocol` under ``name``.

    Idempotent-by-overwrite: re-registering the same name replaces the
    previous entry and logs a warning. This matches how users iterate on a
    spec file during development.

    Args:
        name: String key. Must be non-empty; any other validation is up to
            the caller (lowercase / underscores / hyphens are all fine).
        benchmark: An instantiated :class:`BenchmarkProtocol` subclass.

    Raises:
        TypeError: If ``benchmark`` is not a :class:`BenchmarkProtocol`.
        ValueError: If ``name`` is empty.
    """
    if not name or not isinstance(name, str):
        raise ValueError(f"register_benchmark: name must be a non-empty string, got {name!r}")
    if not isinstance(benchmark, BenchmarkProtocol):
        raise TypeError(f"register_benchmark: expected BenchmarkProtocol instance, got {type(benchmark).__name__}")
    with _REGISTRY_LOCK:
        if name in _BENCHMARK_REGISTRY:
            logger.warning("Overwriting existing benchmark registration: %s", name)
        _BENCHMARK_REGISTRY[name] = benchmark
        logger.info("📋 Registered benchmark '%s' (%s)", name, type(benchmark).__name__)


def unregister_benchmark(name: str) -> BenchmarkProtocol | None:
    """Remove a benchmark from the registry.

    Returns the removed benchmark or ``None`` if it was not registered.
    Primarily used by tests for cleanup; user code is rarely expected to
    unregister benchmarks at runtime.
    """
    with _REGISTRY_LOCK:
        return _BENCHMARK_REGISTRY.pop(name, None)


def get_benchmark(name: str) -> BenchmarkProtocol | None:
    """Return the registered benchmark or ``None`` if not found."""
    with _REGISTRY_LOCK:
        return _BENCHMARK_REGISTRY.get(name)


def list_benchmarks() -> dict[str, dict[str, Any]]:
    """Enumerate registered benchmarks with their metadata.

    Returns a shallow-copy snapshot keyed by name. Each value is a dict
    with ``class``, ``supported_robots``, ``default_robot``, ``max_steps``
    - enough for an LLM to pick an appropriate benchmark without
    instantiating one. Reads a snapshot under the registry lock so a
    concurrent registration does not corrupt the returned dict.
    """
    with _REGISTRY_LOCK:
        snapshot = dict(_BENCHMARK_REGISTRY)
    return {
        name: {
            "class": type(bench).__name__,
            "supported_robots": list(bench.supported_robots),
            "default_robot": bench.default_robot,
            "max_steps": bench.max_steps,
        }
        for name, bench in snapshot.items()
    }


__all__ = [
    "BenchmarkCompatibilityError",
    "BenchmarkProtocol",
    "StepInfo",
    "get_benchmark",
    "list_benchmarks",
    "register_benchmark",
    "unregister_benchmark",
]
