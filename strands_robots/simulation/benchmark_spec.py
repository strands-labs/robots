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
    supported_robots: list[str]           # default [] (any); must contain default_robot
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

Every value in that schema is held to one domain, applied identically whether
it arrives as a spec key or as a keyword to :meth:`DeclarativeBenchmark.from_dict`
or the constructor - so a spec file and a direct construction cannot disagree on
what they accept. ``max_steps`` is a positive whole count, ``supported_robots``
is a list of distinct non-blank names containing ``default_robot``, and the four
string fields go through :func:`_spec_string_error`: ``name`` and
``default_robot`` are non-empty, ``instruction`` is a possibly-empty string, and
``scene`` is a string or omitted.

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
from typing import TYPE_CHECKING, Any, cast

from strands_robots.simulation.benchmark import (
    BenchmarkProtocol,
    StepInfo,
    register_benchmark,
)
from strands_robots.simulation.predicates import PREDICATE_REGISTRY, make_predicate, predicate_kind
from strands_robots.utils import name_list_error, positive_count_error, require_optional

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


def _spec_string_error(
    value: Any,
    param: str,
    context: str,
    *,
    allow_empty: bool = False,
    allow_none: bool = False,
) -> str | None:
    """Return an error message if ``value`` cannot be honored as ``param``'s text.

    The string half of the spec vocabulary, shared by
    :meth:`DeclarativeBenchmark.from_dict` and
    :meth:`DeclarativeBenchmark.__init__` so the key a spec file sets and the
    keyword a direct construction passes cannot drift apart on what they accept
    - the same reason ``max_steps`` and ``supported_robots`` share their
    domains with the shared helpers in :mod:`strands_robots.utils`.

    It lives here rather than there because what it refuses is this module's own
    vocabulary and nothing else's. The comparable shared helper,
    :func:`~strands_robots.utils.entity_name_error`, scopes itself in its
    docstring to the creation sites in the simulation backends, and its reasons
    are MuJoCo's (``""`` is that engine's unnamed-entity sentinel, a NUL
    truncates in the compiled model) - neither applies to a benchmark id.

    Two axes, because the four fields differ only along them:

    * ``allow_empty`` - ``name`` and ``default_robot`` are identifiers, so the
      empty string names nothing; ``instruction`` and ``scene`` are content, and
      an empty one is the documented "not declared" spelling.
    * ``allow_none`` - ``scene`` alone is optional, so ``None`` is how a spec
      says it declares no scene.

    A ``bool`` is refused by the type test rather than separately: it is not a
    ``str`` at all, unlike the numeric domains where ``bool`` is an ``int``
    subclass and has to be turned away by name.

    Args:
        value: The value supplied for ``param``.
        param: Field name, quoted in the message.
        context: Where the value came from - ``"spec"`` for a spec dict, the
            class name for a direct construction.
        allow_empty: Accept the empty string.
        allow_none: Accept ``None``.

    Returns:
        A message naming the field, the accepted set and the type supplied, or
        ``None`` when the value can be honored.
    """
    if value is None:
        if allow_none:
            return None
        return f"{context}: {param} must be a string, got NoneType."
    if not isinstance(value, str):
        omitted = " or omitted" if allow_none else ""
        return f"{context}: {param} must be a string{omitted}, got {type(value).__name__}."
    if not value and not allow_empty:
        return f"{context}: {param} must be a non-empty string."
    return None


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


def compile_stop_when(stop_when: Any, *, context: str = "stop_when") -> Callable[[SimEngine], bool]:
    """Compile a ``stop_when`` early-return clause into a ``(sim) -> bool`` callable.

    This is the ``run_policy(stop_when=...)`` compiler: the clause uses the
    same predicate-DSL schema as a benchmark spec's ``success`` clause, so an
    agent that can author a success condition can gate a rollout with the
    identical vocabulary. Two shapes are accepted - a single predicate call::

        {"predicate": "grasped", "body": "cube", "gripper_prefix": "so100"}

    or an ``all`` / ``any`` group of predicate calls::

        {"all": [{"predicate": "body_above_z", "body": "cube", "z": 0.2},
                 {"predicate": "body_upright", "body": "cube"}]}

    Predicates resolve through the closed registry
    (:func:`~strands_robots.simulation.predicates.make_predicate`) - nothing
    here reaches ``eval`` / ``exec``, so the clause is safe to accept from an
    LLM tool call. Float-valued reward terms are rejected up front (a nonzero
    float reads as always-``True`` and would stop the rollout on step 1), and
    an empty clause is rejected rather than compiling to a check that never
    fires (which would silently run the rollout to its step budget).

    Args:
        stop_when: The clause dict (single predicate call or ``all``/``any``
            group).
        context: Label used to prefix error messages.

    Returns:
        A side-effect-free callable evaluating the clause against the sim
        (predicates only read sim state), suitable for per-step checking.

    Raises:
        ValueError: On a non-dict clause, an empty / never-firing clause, a
            clause mixing the single-call and group forms, an unknown
            predicate name (the message lists the valid set), a float-valued
            predicate, or bad predicate kwargs.
    """
    if not isinstance(stop_when, dict):
        raise ValueError(
            f"{context}: expected a predicate call like {{'predicate': 'grasped', ...}} or an "
            f"'all'/'any' group of them, got {type(stop_when).__name__}"
        )
    if "predicate" in stop_when:
        group_keys = {"all", "any"} & set(stop_when.keys())
        if group_keys:
            raise ValueError(
                f"{context}: a clause is either a single predicate call or an 'all'/'any' group, "
                f"not both (got 'predicate' plus {sorted(group_keys)})"
            )
        return _compile_call(stop_when, context=context, require_kind="bool")
    unknown = set(stop_when.keys()) - {"all", "any"}
    if unknown:
        raise ValueError(
            f"{context}: unknown keys {sorted(unknown)}; pass a single "
            "{'predicate': <name>, ...} call or an 'all'/'any' group of them"
        )
    if not (stop_when.get("all") or stop_when.get("any")):
        raise ValueError(
            f"{context}: empty clause - it would never fire and the rollout would silently run "
            "to its step budget. Pass a predicate call or a non-empty 'all'/'any' group, or "
            f"omit {context} entirely."
        )
    return _compile_bool_group(stop_when, default=False, context=context)


# Predicate kwargs that name scene entities, per kind. A stop_when clause is
# probed against the LIVE sim before the rollout starts (see
# SimEngine.run_policy): a typo'd name would otherwise compile clean, degrade
# to a constant False at evaluation time, and burn the whole step budget
# reporting stopped_reason="budget" - indistinguishable from an honest miss.
_BODY_NAME_KWARGS = frozenset({"body", "body_a", "body_b", "container"})
# Kwargs that carry a LIST of body names (the particle-proxy pour predicates).
# Collected element-wise so every particle / container in a stop_when clause is
# probed against the live sim exactly like a singular body kwarg.
_BODY_LIST_KWARGS = frozenset({"particles", "containers"})
_JOINT_NAME_KWARGS = frozenset({"joint"})


def stop_when_referenced_entities(stop_when: Any) -> tuple[list[str], list[str]]:
    """Collect the body and joint names a ``stop_when`` clause references.

    Walks the same shapes :func:`compile_stop_when` accepts (a single
    predicate call or an ``all``/``any`` group) and gathers the values of the
    entity-naming kwargs (``body`` / ``body_a`` / ``body_b`` / ``container``
    for bodies, plus each element of the list-valued ``particles`` /
    ``containers``, and ``joint`` for joints) so the caller can probe each one
    against the live sim before arming the clause. Geom names
    (``contact_between``) are not collected - there is no generic geom
    lookup on the engine ABC to probe them with.

    Args:
        stop_when: A clause dict already validated by
            :func:`compile_stop_when` (unrecognized shapes yield empty
            results rather than raising - validation is the compiler's job).

    Returns:
        ``(bodies, joints)`` - deduplicated, insertion-ordered name lists.
    """
    bodies: dict[str, None] = {}
    joints: dict[str, None] = {}

    def _collect(call: Any) -> None:
        if not isinstance(call, dict):
            return
        for key, value in call.items():
            if isinstance(value, str) and value:
                if key in _BODY_NAME_KWARGS:
                    bodies.setdefault(value)
                elif key in _JOINT_NAME_KWARGS:
                    joints.setdefault(value)
            elif key in _BODY_LIST_KWARGS and isinstance(value, list):
                for entry in value:
                    if isinstance(entry, str) and entry:
                        bodies.setdefault(entry)

    if isinstance(stop_when, dict):
        if "predicate" in stop_when:
            _collect(stop_when)
        else:
            for group_key in ("all", "any"):
                entries = stop_when.get(group_key)
                if isinstance(entries, list):
                    for entry in entries:
                        _collect(entry)
    return list(bodies), list(joints)


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
        # Mirror the four string checks ``from_dict`` runs, for the reason the two
        # mirrors below state: a directly constructed benchmark must not carry a
        # value the evaluation loop, or the policy it drives, has to deal with
        # later. Measured before this: ``instruction=42`` was handed to the
        # policy verbatim as its task command (``PolicyRunner`` falls back to
        # ``spec.instruction`` when the caller passes none), and a falsy
        # non-string ``scene`` such as ``[]`` was skipped by the truthiness test
        # in :meth:`on_episode_start`, so a declared scene was never loaded -
        # both under ``status="success"``.
        #
        # ``default_robot`` is checked here, ahead of the membership test below:
        # on ``default_robot=7`` that test would report ``7 not in [...]``, which
        # describes the symptom rather than the mistake. Same ordering reason as
        # the shape-before-membership note on ``supported_robots``.
        cls_name = type(self).__name__
        if error := _spec_string_error(name, "name", cls_name):
            raise ValueError(error)
        if error := _spec_string_error(default_robot, "default_robot", cls_name):
            raise ValueError(error)
        if error := _spec_string_error(scene, "scene", cls_name, allow_empty=True, allow_none=True):
            raise ValueError(error)
        if error := _spec_string_error(instruction, "instruction", cls_name, allow_empty=True):
            raise ValueError(error)
        self._name = name
        # Mirrors the two checks ``from_dict`` runs on this value, for the
        # reason the ``max_steps`` mirror below states: a directly constructed
        # benchmark must not carry a robot set the evaluation loop has to
        # refuse later. ``list()`` alone read a single name passed as a bare
        # string one character at a time, so ``supported_robots="panda"``
        # became five one-letter robots, ``list_benchmarks`` advertised them,
        # and the evaluation then refused the benchmark's own ``default_robot``
        # by naming them.
        #
        # Shape first: on a bare string the membership check below would report
        # ``'panda' not in ['p', 'a', 'n', 'd', 'a']``, which describes the
        # symptom rather than the mistake. Ungated, unlike the callers that
        # derive the list when it is falsy - an empty list is this parameter's
        # documented "any robot" spelling and ``name_list_error`` accepts it,
        # while ``""`` is a mistyped name that would otherwise widen the
        # benchmark to every robot silently.
        if error := name_list_error(supported_robots, "supported_robots", type(self).__name__):
            raise ValueError(error)
        if supported_robots and default_robot not in supported_robots:
            raise ValueError(
                f"{type(self).__name__}: default_robot={default_robot!r} not in "
                f"supported_robots={list(supported_robots)}; either add it to "
                "supported_robots or leave supported_robots empty for any-robot benchmarks"
            )
        self._supported_robots = list(supported_robots)
        self._default_robot = default_robot
        # Mirrors the check ``from_dict`` runs on the raw spec value, so a
        # directly constructed benchmark cannot carry a horizon the
        # evaluation loop has to refuse later. ``int()`` alone silently
        # truncated ``2.7`` to 2 and read ``True`` as 1.
        if error := positive_count_error(max_steps, "max_steps", type(self).__name__):
            raise ValueError(error)
        self.max_steps = max_steps
        self._success_fn = success_fn
        self._failure_fn = failure_fn
        self._reward_terms = list(reward_terms)
        self._scene = scene
        self._instruction = instruction

    @property
    def name(self) -> str:
        """Unique benchmark id, from the spec's required ``name`` key.

        Used as the registry key by :func:`register_benchmark_from_file`;
        registering two specs with the same ``name`` overwrites the first.

        The constructor holds this to :func:`_spec_string_error`: a non-empty
        ``str``, on both construction paths. A non-string was previously stored
        and then advertised by ``sim.list_benchmarks()`` as the benchmark's id.
        """
        return self._name

    @property
    def supported_robots(self) -> list[str]:
        """Registry ``data_config`` names this benchmark accepts (spec ``supported_robots``).

        Implements :attr:`BenchmarkProtocol.supported_robots`. Returns a fresh
        copy so callers cannot mutate the compiled spec. An empty list (the
        spec omitted the key) means "any robot".

        The constructor holds this to the shared
        :func:`~strands_robots.utils.name_list_error` domain: several distinct
        non-blank names, as a list or tuple rather than a single bare string,
        and containing :attr:`default_robot` whenever it is non-empty. So the
        names read back here are the names that were asked for, and a benchmark
        cannot declare a robot set that its own default robot is outside of.
        """
        return list(self._supported_robots)

    @property
    def default_robot(self) -> str:
        """Robot loaded when the sim is empty, from the spec's required ``default_robot``.

        Implements :attr:`BenchmarkProtocol.default_robot`;
        :meth:`on_episode_start` adds it via ``sim.add_robot`` before the first
        observation when no robot is present.

        The constructor holds this to :func:`_spec_string_error` - a non-empty
        ``str`` - before the :attr:`supported_robots` membership test, so a
        non-string is reported as the wrong type rather than as a robot missing
        from the supported set.
        """
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

        The constructor holds this to :func:`_spec_string_error`: a ``str``,
        possibly empty, on both construction paths. That fallback is why the
        mirror matters - a non-string stored here was handed to the policy
        verbatim as its task command, so a language-conditioned policy received
        a value its tokenizer cannot take while the evaluation reported success.
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
        """Evaluate the compiled ``success`` clause against ``sim``.

        Implements :meth:`BenchmarkProtocol.is_success`. Side-effect-free (the
        predicate closures only read sim state), so the eval loop may call it
        multiple times per step. Returns ``True`` once the spec's ``success``
        all/any predicates are satisfied, ending the episode with success.
        """
        return bool(self._success_fn(sim))

    def is_failure(self, sim: SimEngine) -> bool:
        """Evaluate the compiled ``failure`` clause against ``sim``.

        Implements :meth:`BenchmarkProtocol.is_failure`. Side-effect-free;
        returns ``True`` when the spec's ``failure`` all/any predicates are
        satisfied (a spec that omits ``failure`` compiles to an always-``False``
        clause). Failure ends the episode without marking success.
        """
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
        # Shared string domain, so the key a spec file sets and the keyword a
        # direct construction passes cannot drift apart on what they accept -
        # the same reason ``supported_robots`` and ``max_steps`` below share
        # theirs.
        if error := _spec_string_error(name, "name", "spec"):
            raise ValueError(error)
        # The guard above is what proves the type; the cast only carries that
        # proof forward for the type checker, which cannot see through a helper
        # returning a message. Both fields are required, so neither can be None
        # by the time they reach the constructor.
        name = cast("str", name)

        default_robot = spec.get("default_robot")
        if error := _spec_string_error(default_robot, "default_robot", "spec"):
            raise ValueError(error)
        default_robot = cast("str", default_robot)

        supported_robots = spec.get("supported_robots", [])
        # Shared name-list domain, so the key a spec file sets and the keyword a
        # direct construction passes cannot drift apart on what they accept -
        # the same reason ``max_steps`` below shares its count domain. The
        # hand-rolled check this replaces accepted a repeated name and a blank
        # one, neither of which names a robot the registry can resolve.
        if error := name_list_error(supported_robots, "supported_robots", "spec"):
            raise ValueError(error)

        # default_robot should be in supported_robots (unless list is empty = any)
        if supported_robots and default_robot not in supported_robots:
            raise ValueError(
                f"spec.default_robot={default_robot!r} not in supported_robots={supported_robots}; "
                "either add it to supported_robots or leave supported_robots empty for any-robot benchmarks"
            )

        max_steps_raw = spec.get("max_steps", 300)
        # Shared count domain, so the key a spec file sets and the keyword a
        # direct construction passes cannot drift apart on what they accept.
        if error := positive_count_error(max_steps_raw, "max_steps", "spec"):
            raise ValueError(error)

        scene = spec.get("scene")
        if error := _spec_string_error(scene, "scene", "spec", allow_empty=True, allow_none=True):
            raise ValueError(error)

        instruction = spec.get("instruction", "")
        if error := _spec_string_error(instruction, "instruction", "spec", allow_empty=True):
            raise ValueError(error)

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
    "compile_stop_when",
    "register_benchmark_from_file",
    "stop_when_referenced_entities",
]
