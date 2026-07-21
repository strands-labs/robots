"""Declarative benchmark specs loaded from YAML / JSON files.

This module is the LLM-facing surface for authoring benchmarks without
writing Python. A spec file declares scene, success predicate, failure
predicate, and dense reward terms using the named-predicate DSL from
:mod:`strands_robots.simulation.predicates`. Nothing in a spec ever reaches
``eval`` / ``exec`` - predicates are looked up in a closed registry and
kwargs are forwarded as-is, so spec files are safe to load from untrusted
input.

Spec schema (top-level keys)::

    name: string                          # required
    max_steps: int                        # default 300
    supported_robots: list[str]           # default [] (any)
    default_robot: string                 # required - registry data_config
    scene: string                         # optional MJCF/URDF path for sim.load_scene()
    instruction: string                   # optional natural-language task command
                                          #   for language-conditioned policies (GR00T, ...)
    success:
      all: [<bool predicate_call>, ...]   # all must be true (reward terms rejected)
      any: [<bool predicate_call>, ...]   # at least one must be true
    failure:
      all: [<bool predicate_call>, ...]
      any: [<bool predicate_call>, ...]
    dense_reward: [<predicate_call>, ...] # reward terms summed per step (a bool
                                          #   predicate here contributes a sparse 0/1)

A ``<predicate_call>`` is a dict with a ``predicate`` key naming the
predicate and any remaining keys forwarded as kwargs::

    {predicate: body_above_z, body: cube, z: 0.2}

Example::

    name: drawer-open
    max_steps: 300
    supported_robots: [panda]
    default_robot: panda
    success:
      all:
        - {predicate: joint_above, joint: drawer_slide, value: 0.15}
    failure:
      any:
        - {predicate: body_below_z, body: gripper, z: -0.1}
    dense_reward:
      - {predicate: distance_neg, body_a: gripper, body_b: drawer_handle, weight: 1.0}
      - {predicate: joint_progress, joint: drawer_slide, target: 0.2, weight: 5.0}

Load + register via :func:`register_benchmark_from_file`; agents call this
through the ``register_benchmark_from_file`` tool action.

YAML files require ``pyyaml`` - not a core dep. JSON works out of the box.
The loader autodetects format by extension.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.benchmark import (
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.predicates import PREDICATE_REGISTRY, make_predicate, predicate_kind
from strands_robots.utils import require_optional

if TYPE_CHECKING:
    import random

    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

# Canonical top-level keys allowed in a spec. Anything else is a user error
# and produces a clear message rather than silently being ignored.
_ALLOWED_TOP_LEVEL = frozenset(
    {
        "name",
        "max_steps",
        "supported_robots",
        "default_robot",
        "scene",
        "instruction",
        "success",
        "failure",
        "dense_reward",
    }
)


def _compile_bool_group(
    clause: dict[str, Any] | None,
    *,
    default: bool,
    context: str,
) -> Callable[[SimEngine], bool]:
    """Compile an ``{"all": [...], "any": [...]}`` bool group into a single callable.

    * ``None`` / missing → returns a function always returning ``default``.
    * ``all``: every listed predicate must be true.
    * ``any``: at least one predicate must be true.
    * Both: both conditions must hold (all AND any).

    Args:
        clause: The ``success`` / ``failure`` dict from the spec.
        default: Value returned when the clause is absent (``False`` for
            success → "never succeeds", ``False`` for failure → "never
            fails"; both are reasonable).
        context: Name for error messages (``"success"`` or ``"failure"``).

    Raises:
        ValueError: If the clause shape is wrong.
    """
    if clause is None:
        return lambda _sim: default
    if not isinstance(clause, dict):
        raise ValueError(f"{context}: expected a dict with 'all' / 'any' keys, got {type(clause).__name__}")

    unknown = set(clause.keys()) - {"all", "any"}
    if unknown:
        raise ValueError(f"{context}: unknown keys {sorted(unknown)}; allowed: ['all', 'any']")

    all_calls = [_compile_call(c, context=f"{context}.all", require_kind="bool") for c in (clause.get("all") or [])]
    any_calls = [_compile_call(c, context=f"{context}.any", require_kind="bool") for c in (clause.get("any") or [])]

    if not all_calls and not any_calls:
        return lambda _sim: default

    def check(sim: SimEngine) -> bool:
        if all_calls and not all(bool(p(sim)) for p in all_calls):
            return False
        if any_calls and not any(bool(p(sim)) for p in any_calls):
            return False
        return True

    return check


def _compile_call(entry: Any, *, context: str, require_kind: str | None = None) -> Callable[[SimEngine], Any]:
    """Compile one ``{predicate: <name>, **kwargs}`` entry to a callable.

    ``require_kind="bool"`` (success / failure clauses) rejects a float-valued
    reward term up front: used as a success predicate it is read as
    ``bool(<term(sim)>)`` - a reward term returns a (usually nonzero) float, so
    that is almost always ``True``, silently making the benchmark report instant
    success. Reward terms belong in ``dense_reward``.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{context}: expected a dict like {{predicate: <name>, ...}}, got {type(entry).__name__}")
    pred_name = entry.get("predicate")
    if not isinstance(pred_name, str) or not pred_name:
        raise ValueError(f"{context}: each entry must have a non-empty 'predicate' string")
    if require_kind == "bool" and predicate_kind(pred_name) == "float":
        bool_preds = sorted(n for n in PREDICATE_REGISTRY if predicate_kind(n) == "bool")
        raise ValueError(
            f"{context}: predicate {pred_name!r} is a reward term (float-valued) but a success / "
            "failure clause requires a bool predicate - used here it is read as bool(<nonzero "
            f"float>) and silently reports instant success. Move it to dense_reward. Bool "
            f"predicates: {bool_preds}"
        )
    kwargs = {k: v for k, v in entry.items() if k != "predicate"}
    try:
        return make_predicate(pred_name, **kwargs)
    except ValueError:
        # Unknown predicate; surface verbatim (already carries the valid list).
        raise
    except TypeError as e:
        # Bad kwargs; wrap so the caller knows which predicate failed to compile.
        raise ValueError(f"{context}: predicate '{pred_name}' compilation failed: {e}") from e


def _compile_reward_terms(terms: list[Any] | None) -> list[Callable[[SimEngine], float]]:
    if terms is None:
        return []
    if not isinstance(terms, list):
        raise ValueError(f"dense_reward: expected a list, got {type(terms).__name__}")
    compiled: list[Callable[[SimEngine], float]] = []
    for i, t in enumerate(terms):
        term = _compile_call(t, context=f"dense_reward[{i}]")
        compiled.append(term)
    return compiled


class DeclarativeBenchmark(BenchmarkProtocol):
    """:class:`BenchmarkProtocol` backed by a compiled DSL spec.

    Use :func:`register_benchmark_from_file` or
    :meth:`DeclarativeBenchmark.from_dict` to construct one - direct
    instantiation is only for tests / internal use.

    Thread safety: the compiled predicate closures capture only the spec
    kwargs (ints, floats, strings, lists of floats) so instances are safe
    to share across threads. The evaluation loop still drives each episode
    sequentially; we do not batch episodes.
    """

    def __init__(
        self,
        *,
        name: str,
        supported_robots: list[str],
        default_robot: str,
        max_steps: int,
        success_fn: Callable[[SimEngine], bool],
        failure_fn: Callable[[SimEngine], bool],
        reward_terms: list[Callable[[SimEngine], float]],
        scene: str | None = None,
        instruction: str = "",
    ):
        self._name = name
        self._supported_robots = list(supported_robots)
        self._default_robot = default_robot
        self.max_steps = int(max_steps)
        self._success_fn = success_fn
        self._failure_fn = failure_fn
        self._reward_terms = list(reward_terms)
        self._scene = scene
        self._instruction = instruction

    @property
    def name(self) -> str:
        return self._name

    @property
    def supported_robots(self) -> list[str]:
        return list(self._supported_robots)

    @property
    def default_robot(self) -> str:
        return self._default_robot

    @property
    def instruction(self) -> str:
        """Natural-language task command declared in the spec (default ``""``).

        Overrides :attr:`BenchmarkProtocol.instruction` so a spec-authored
        (declarative) benchmark can carry a task description without a Python
        subclass -- the ``PolicyRunner`` eval loop falls back to this when the
        caller passes no ``instruction=`` to ``evaluate_benchmark``, so a
        language-conditioned policy (GR00T, OpenVLA, ...) receives the command
        instead of an empty string (the #187 off-task failure mode).
        """
        return self._instruction

    def on_episode_start(self, sim: SimEngine, rng: random.Random) -> None:
        """Load the declared scene (if any) before delegating to the base impl.

        The base impl adds :attr:`default_robot` when the sim is empty and
        validates robot compatibility. Scene loading happens *before* that so
        a scene-declared robot is detected by the compatibility check.
        """
        if self._scene:
            load_scene = getattr(sim, "load_scene", None)
            if load_scene is None:
                logger.warning(
                    "DeclarativeBenchmark '%s' declares scene=%r but sim has no load_scene()",
                    self._name,
                    self._scene,
                )
            else:
                result = load_scene(self._scene)
                if isinstance(result, dict) and result.get("status") == "error":
                    msg = (result.get("content") or [{}])[0].get("text", "")
                    raise RuntimeError(
                        f"DeclarativeBenchmark '{self._name}': load_scene({self._scene!r}) failed: {msg}"
                    )
        # Reset any stateful reward terms (e.g. a staged_reward phase machine)
        # so per-episode phase state does not leak across episodes. Stateless
        # function terms have no reset() and are skipped.
        for term in self._reward_terms:
            term_reset = getattr(term, "reset", None)
            if callable(term_reset):
                term_reset()
        super().on_episode_start(sim, rng)

    def on_step(self, sim: SimEngine, obs: dict[str, Any], action: dict[str, Any]) -> StepInfo:
        """Sum all registered reward terms; ``done`` is False (handled by is_success/is_failure)."""
        reward = 0.0
        for term in self._reward_terms:
            try:
                reward += float(term(sim))
            except Exception as e:  # noqa: BLE001 - defensive: one bad term shouldn't kill the episode
                logger.warning("reward term failed in '%s': %s", self._name, e)
        return StepInfo(reward=reward, done=False)

    def is_success(self, sim: SimEngine) -> bool:
        return bool(self._success_fn(sim))

    def is_failure(self, sim: SimEngine) -> bool:
        return bool(self._failure_fn(sim))

    @classmethod
    def from_dict(cls, spec: dict[str, Any]) -> DeclarativeBenchmark:
        """Compile a spec dict (already parsed from YAML/JSON) into a benchmark."""
        if not isinstance(spec, dict):
            raise ValueError(f"spec must be a dict, got {type(spec).__name__}")

        unknown = set(spec.keys()) - _ALLOWED_TOP_LEVEL
        if unknown:
            raise ValueError(
                f"Unknown top-level keys in spec: {sorted(unknown)}. Allowed: {sorted(_ALLOWED_TOP_LEVEL)}"
            )

        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("spec.name: required non-empty string")

        default_robot = spec.get("default_robot")
        if not isinstance(default_robot, str) or not default_robot:
            raise ValueError("spec.default_robot: required non-empty string")

        supported_robots = spec.get("supported_robots", [])
        if not isinstance(supported_robots, list) or not all(isinstance(r, str) for r in supported_robots):
            raise ValueError("spec.supported_robots: must be a list of strings")

        # default_robot should be in supported_robots (unless list is empty = any)
        if supported_robots and default_robot not in supported_robots:
            raise ValueError(
                f"spec.default_robot={default_robot!r} not in supported_robots={supported_robots}; "
                "either add it to supported_robots or leave supported_robots empty for any-robot benchmarks"
            )

        max_steps_raw = spec.get("max_steps", 300)
        if not isinstance(max_steps_raw, int) or isinstance(max_steps_raw, bool) or max_steps_raw <= 0:
            raise ValueError(f"spec.max_steps: must be a positive int, got {max_steps_raw!r}")

        scene = spec.get("scene")
        if scene is not None and not isinstance(scene, str):
            raise ValueError(f"spec.scene: must be a string path or omitted, got {type(scene).__name__}")

        instruction = spec.get("instruction", "")
        if not isinstance(instruction, str):
            raise ValueError(f"spec.instruction: must be a string or omitted, got {type(instruction).__name__}")

        success_fn = _compile_bool_group(spec.get("success"), default=False, context="success")
        failure_fn = _compile_bool_group(spec.get("failure"), default=False, context="failure")
        reward_terms = _compile_reward_terms(spec.get("dense_reward"))

        return cls(
            name=name,
            supported_robots=supported_robots,
            default_robot=default_robot,
            max_steps=max_steps_raw,
            success_fn=success_fn,
            failure_fn=failure_fn,
            reward_terms=reward_terms,
            scene=scene,
            instruction=instruction,
        )


def _load_spec_file(path: str | Path) -> dict[str, Any]:
    """Parse a spec file by extension. JSON via stdlib, YAML via ``pyyaml`` (optional).

    Return type is declared as ``dict[str, Any]`` but ``json.loads`` /
    ``yaml.safe_load`` may produce lists, strings, etc. Caller
    (``register_benchmark_from_file``) validates the parsed shape before
    passing it to :meth:`DeclarativeBenchmark.from_dict`; we do the
    ``isinstance`` check here so the returned value is actually a dict.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Benchmark spec file not found: {path}")
    if not p.is_file():
        raise ValueError(f"Benchmark spec path is not a file: {path}")

    suffix = p.suffix.lower()
    text = p.read_text()

    parsed: Any
    if suffix == ".json":
        parsed = json.loads(text)
    elif suffix in (".yaml", ".yml"):
        yaml = require_optional(
            "yaml",
            pip_install="pyyaml",
            purpose="YAML benchmark spec loading",
        )
        parsed = yaml.safe_load(text)  # type: ignore[attr-defined]
    else:
        raise ValueError(f"Unsupported spec file extension: {suffix!r}. Use .json, .yaml, or .yml.")

    if not isinstance(parsed, dict):
        raise ValueError(f"Benchmark spec {path} must parse to a dict, got {type(parsed).__name__}")
    return parsed


def register_benchmark_from_file(
    name: str,
    spec_path: str | Path,
) -> BenchmarkProtocol:
    """Load a declarative benchmark spec from disk and register it under ``name``.

    Convenience wrapper that:

    1. Parses ``spec_path`` (JSON or YAML, autodetected by extension).
    2. Compiles it into a :class:`DeclarativeBenchmark`.
    3. Registers it via :func:`register_benchmark`.
    4. Returns the instantiated benchmark for programmatic use.

    Args:
        name: Registry key. Overrides any ``name`` declared inside the spec
            (so the same spec file can be registered under multiple names).
        spec_path: Path to a ``.json`` / ``.yaml`` / ``.yml`` file.

    Returns:
        The registered :class:`DeclarativeBenchmark` instance.

    Raises:
        FileNotFoundError / ValueError: From :func:`_load_spec_file`.
        ValueError: From :meth:`DeclarativeBenchmark.from_dict` on bad schema.
    """
    if not isinstance(name, str) or not name:
        raise ValueError(f"register_benchmark_from_file: name must be a non-empty string, got {name!r}")
    spec_dict = _load_spec_file(spec_path)
    # The registry name always wins: unconditionally override any spec-internal
    # ``name`` so the instance's ``.name`` matches the key it is registered under
    # (the documented contract). ``setdefault`` was insufficient - a spec that
    # declared its own ``name`` kept it, so the same spec registered under two
    # keys produced two instances that both reported the spec-internal name and
    # neither matched its registry key.
    spec_dict["name"] = name
    benchmark = DeclarativeBenchmark.from_dict(spec_dict)
    register_benchmark(name, benchmark)
    return benchmark


__all__ = [
    "DeclarativeBenchmark",
    "register_benchmark_from_file",
]
