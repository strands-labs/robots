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

import contextlib
import difflib
import logging
import math
import numbers
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any, SupportsFloat, cast

if TYPE_CHECKING:
    import numpy as np

    from strands_robots.policies import Policy
    from strands_robots.rendering import CameraParams

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
from strands_robots.utils import (
    FREE_CAMERA_TOKENS,
    dds_domain_id_error,
    is_boolean,
    non_negative_count_error,
    positive_count_error,
    positive_finite_number_error,
    sequence_length,
)

logger = logging.getLogger(__name__)


# Robot-setup keyword arguments that identify a caller who confused a backend
# *constructor* with the robot-setup entry points. A constructor builds only an
# empty engine; a robot is added afterwards via ``add_robot`` (or in one step by
# the ``Robot(name, mode="sim")`` factory). Backend constructors accept
# ``**kwargs`` for cross-backend forward compatibility - so a single call can
# carry GPU-backend options such as ``num_envs`` / ``device`` that non-GPU
# backends simply drop - but that sink must not silently swallow an argument
# that names a *robot to set up*. Matching the "no silent swallow of unknown
# kwargs" contract already enforced for ``add_object``, these are rejected
# loudly instead of being dropped and failing far downstream with an unrelated
# "No world" error.
_SETUP_KWARGS: tuple[str, ...] = ("robot_name", "robot")


def reject_setup_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Reject robot-setup keyword arguments passed to a backend constructor.

    A backend ``__init__`` accepts ``**kwargs`` only as a forward-compatibility
    sink for backend-specific options. Passing ``robot_name`` (or ``robot``)
    there is always a mistake: the constructor creates an empty engine, so the
    argument is meaningless and would otherwise be silently dropped, leaving a
    robot-less engine that fails later with a confusing "No world" error.

    Args:
        kwargs: The residual keyword arguments a backend ``__init__`` is about
            to drop into its forward-compatibility sink.

    Raises:
        TypeError: If ``kwargs`` names a robot-setup argument. The message
            points at the ``Robot(name, mode="sim")`` factory (one-step setup)
            and the ``create_world()`` + ``add_robot(name)`` sequence.
    """
    offending = [k for k in _SETUP_KWARGS if k in kwargs]
    if not offending:
        return
    names = ", ".join(repr(k) for k in offending)
    raise TypeError(
        f"Simulation backend constructor does not accept {names}: a constructor "
        'builds an empty engine, not a robot. Use Robot("so101", mode="sim") for '
        'one-step setup, or create_world() then add_robot("so101").'
    )


def close_match_hint(requested: object, known: Sequence[str]) -> str:
    """The ``" Did you mean: a, b?"`` fragment of an unknown-entity message.

    Returns ``""`` when there is no usable suggestion, so a caller can append
    it unconditionally.

    This is the *only* part of an unknown-entity message that needs
    ``requested`` to be a :class:`str`: :func:`difflib.get_close_matches`
    compares character sequences, and a name of another type has none. What is
    registered - and the discovery action that lists it - is a property of the
    world alone, so it must not be gated on the requested name's type. Owning
    that distinction here is what keeps the two from being conflated again: one
    ``isinstance`` test guarding both suppressed the whole listing for every
    non-``str`` name, and such a name is exactly what
    :func:`strands_robots.simulation.models.registered` routes to these
    messages - it is total so an unhashable name is reported rather than
    raising, and the report it hands off to has to be usable.

    A suggestion identical to ``requested`` is dropped. ``difflib`` scores an
    exact match 1.0 and therefore ranks it first, so a caller whose ``known``
    set can contain the requested name would otherwise be told to try the name
    it just refused - a suggestion that carries no information and displaces a
    real one out of the three slots. Callers whose ``known`` set is what the
    world holds cannot reach that (the name is absent, which is why the message
    is being built), but a caller comparing against a wider catalogue can:
    :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine._unknown_model_msg`
    suggests over the whole robot registry, where a registered robot whose
    asset is missing is exactly the case whose name is already correct. Owning
    the rule here keeps it from being a precondition each caller has to know.

    Args:
        requested: Caller-supplied entity name, of any type.
        known: Entity names that are registered.

    Returns:
        A leading-space ``" Did you mean: ...?"`` fragment naming up to three
        registered names, or ``""`` when ``requested`` is not a string, nothing
        is registered, or no registered name other than ``requested`` itself is
        close enough to suggest.
    """
    if not isinstance(requested, str) or not known:
        return ""
    # Ask for one more than we render so dropping an exact self-match promotes
    # the next-best candidate instead of shortening the list. difflib returns
    # matches best-first, so the rendered three are unchanged for a caller
    # whose known set cannot contain ``requested``.
    matches = [m for m in difflib.get_close_matches(requested, list(known), n=4, cutoff=0.4) if m != requested][:3]
    if not matches:
        return ""
    return " Did you mean: " + ", ".join(matches) + "?"


def unknown_kwargs_error(method: str, kwargs: Mapping[str, Any], accepted: Sequence[str]) -> dict[str, Any] | None:
    """Return a tool-envelope error for keyword arguments a method cannot use.

    Some engine methods declare ``**kwargs`` only to keep the ``**kwargs``-typed
    :class:`SimEngine` base signature, then drop the residual keys. For a
    *forwarding* sink (``attach_teleop``, ``stream_dataset``) dropping is right -
    the keys belong to the callee. For a *discarding* sink it turns a misspelled
    or invented parameter into a successful no-op: a caller asking for
    object-position randomization or sensor noise is told the request was
    applied when nothing happened. Discarding sinks call this helper instead, so
    an unusable parameter is named rather than swallowed - the same contract the
    action dispatcher already enforces for methods without ``**kwargs``.

    Args:
        method: Method (and action) name to quote in the message.
        kwargs: The residual keyword arguments the method would otherwise drop.
        accepted: Every keyword the method honors, including any it reads out of
            its own ``**kwargs`` (Newton's ``randomize`` answers
            ``randomize_positions`` with a dedicated unsupported-axis error).
            Also listed as the "Valid:" hint.

    Returns:
        ``None`` when every key in ``kwargs`` is accepted, otherwise a
        ``status="error"`` result dict naming the unusable keys. An error dict
        rather than a raised exception because these methods are dispatched as
        agent tool actions, which must not raise past dispatch.
    """
    unexpected = sorted(k for k in kwargs if k not in accepted)
    if not unexpected:
        return None
    return {
        "status": "error",
        "content": [{"text": (f"Unknown parameter(s) {unexpected} for action '{method}'. Valid: {sorted(accepted)}")}],
    }


_BOOLEAN_WORLD_REASON = (
    "float(True) is 1.0, so a boolean survives a numeric coercion as the "
    "quantity 1 in whatever unit the parameter carries - a 1 m/s^2 "
    "acceleration, a 1-second integration step, a 1 kg body, a noise standard "
    "deviation of 1, a randomization scale of 1. Each is a usable number on "
    "its own terms, so the call reports success and the world is configured "
    "with a value the caller never chose. Pass the quantity in the "
    "parameter's own units."
)


def _boolean_world_error(method: str, param: str, value: Any) -> dict[str, Any]:
    """Structured error for a boolean supplied as a world-physics quantity.

    The world-parameter counterpart to :func:`_boolean_action_error`, sharing
    the one predicate in :func:`strands_robots.utils.is_boolean` rather than
    re-deciding what a boolean is. It carries its own reason because
    :data:`_BOOLEAN_ACTION_REASON` argues from the *ambiguity* of ``1.0`` across
    actuator drives, which does not apply here: gravity, a timestep and a mass
    each read ``1.0`` as one unambiguous quantity, so a boolean is not ambiguous
    but simply a value nobody asked for, applied under ``status="success"``.

    Args:
        method: Public method name, used to prefix the message.
        param: Parameter name to quote.
        value: The raw value, reported before coercion - ``float(True)`` is
            ``1.0``, so the boolean is unrecoverable once coerced.

    Returns:
        A structured ``{"status": "error", ...}`` dict to surface.
    """
    return {
        "status": "error",
        "content": [
            {"text": (f"{method}: '{param}' must be a number, not a bool (got {value!r}). {_BOOLEAN_WORLD_REASON}")}
        ],
    }


def randomization_range_error(value: Any, param: str, *, allow_zero: bool = True) -> str | None:
    """Return why a ``(lo, hi)`` randomization range cannot be applied.

    Domain randomization multiplies live physics constants (body mass, geom
    friction) and re-samples colours inside a caller-supplied range. A range
    that is not a pair of finite numbers with ``0 <= lo <= hi`` has no sampling
    interval a backend could draw from: the sampler either raises deep inside
    the mutation loop or, worse, succeeds and installs a physically impossible
    constant - a negative body mass falls *upward* under gravity and a negative
    friction coefficient is not a Coulomb model. Either way the randomized world
    no longer models anything, so the request is refused up front.

    Args:
        value: The candidate ``(lo, hi)`` pair.
        param: Parameter name to quote in the message (``"mass_range"``).
        allow_zero: Whether ``lo == 0`` is meaningful for this quantity. True
            for the ranges where zero is a real physical setting (a frictionless
            surface, a black colour channel); pass False for a multiplicative
            mass scale, where a zero multiplier leaves a massless body that
            ignores gravity instead of a lighter one.

    Returns:
        ``None`` when the range is usable, otherwise the reason as a string.
    """
    # The unpack and the coercion are separate steps so the boolean check can sit
    # between them; both report the same thing to the caller.
    not_a_pair = f"{param} must be a (lo, hi) pair of numbers, got {value!r}"
    try:
        lo, hi = value
    except (TypeError, ValueError):
        return not_a_pair
    # Before float(): a boolean bound is a scale factor of 1 (identity) or 0
    # (erases the quantity it multiplies), and the sampler cannot tell either
    # from a deliberate one once coerced.
    if is_boolean(lo) or is_boolean(hi):
        return f"{param} bounds must be numbers, not bools (got {value!r}). {_BOOLEAN_WORLD_REASON}"
    try:
        lo, hi = float(lo), float(hi)
    except (TypeError, ValueError):
        return not_a_pair
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return f"{param} bounds must be finite, got {value!r}"
    if lo > hi:
        return f"{param} lower bound {lo} exceeds upper bound {hi}"
    if allow_zero:
        if lo < 0:
            return f"{param} bounds must be non-negative, got {value!r}"
    elif lo <= 0:
        detail = (
            "a zero scale erases the quantity it multiplies"
            if lo == 0
            else "a negative scale flips the sign of the quantity it multiplies"
        )
        return f"{param} bounds must be positive, got {value!r} ({detail})"
    return None


def finite_non_negative_error(value: Any, param: str, context: str) -> str | None:
    """Return why a magnitude parameter cannot be used as a noise/offset scale.

    Shared by the sensor-noise standard deviations and the position-jitter
    amplitude: all of them are half-widths or standard deviations, so a
    non-numeric, non-finite or negative value describes no distribution. A
    NaN amplitude propagates into ``qpos`` and poisons the whole physics state
    on the next step, and a negative half-width inverts the sampling bounds.

    Args:
        value: The candidate magnitude. A boolean is refused: these are
            standard deviations and half-widths, so ``True`` describes a
            distribution of width 1 rather than the "noise off" a caller
            passing a flag would have meant (``0`` is how noise is disabled).
        param: Parameter name to quote in the message.
        context: Method name to prefix the message with.

    Returns:
        ``None`` when the value is a finite non-negative number, otherwise the
        reason as a string.
    """
    # Before float(): these are standard deviations and half-widths, so a
    # boolean reads as a distribution of width 1 in the sensor's own units -
    # not a flag disabling the noise, which is what a caller passing True
    # would most plausibly have meant.
    if is_boolean(value):
        return f"{context}: {param} must be a number, not a bool (got {value!r}). {_BOOLEAN_WORLD_REASON}"
    try:
        fvalue = float(value)
    except (TypeError, ValueError):
        return f"{context}: {param} must be a number, got {value!r}"
    if not math.isfinite(fvalue) or fvalue < 0:
        return f"{context}: {param} must be a finite non-negative number, got {value!r}"
    return None


# The largest seed a rollout can apply. ``set_eval_seed`` reseeds the legacy
# NumPy global RNG (``numpy.random.seed``), which refuses anything above
# 2**32 - 1 - unlike ``numpy.random.default_rng``, the destination of the
# ``randomize`` / ``set_obs_noise`` seeds, which accepts an integer of any
# width. That is why a rollout seed carries a ceiling those two do not: the
# accepted domain of a parameter is bounded by what its applier can honor, and
# ``random.seed`` / ``torch.manual_seed`` (the other two RNGs seeded there) are
# wider still.
#
# It lives here, beside the ``max_seed`` parameter it feeds, rather than in
# ``policy_runner`` with the applier it describes. The note above this module's
# ``policy_runner`` import is the reason: CodeQL's ``py/unsafe-cyclic-import``
# walks ``TYPE_CHECKING`` blocks, and ``policy_runner`` imports ``SimEngine``
# from here under one, so every name added to that module-level import line
# closes an AST-visible cycle. Adding this constant to it raised
# ``py/unsafe-cyclic-import`` on all three names that line carries. The rollout
# side reaches it through the function-local import it already uses for
# ``randomization_seed_error``, so neither module gains a module-level edge.
MAX_EVAL_SEED = 2**32 - 1


def randomization_seed_error(
    value: Any, context: str, *, max_seed: int | None = None, allow_none: bool = True
) -> str | None:
    """Return why a value cannot seed a reproducible randomization stream.

    The seed reaches ``numpy.random.default_rng``, which accepts only
    non-negative integers (and a few RNG objects the ``int | None`` annotations
    on these methods do not advertise). A float or string seed raises there -
    on the sensor-noise path not until the first observation is drawn, long
    after the configuring call reported success - so it is rejected at the call
    that supplied it.

    Two families share this domain, and they share it because the failure is
    the same: the ``seed`` of ``randomize`` / ``set_obs_noise``, which drives
    the domain-randomization streams, and the ``seed`` of a policy rollout or
    evaluation (``run_policy`` / ``eval_policy`` / ``start_policy`` /
    ``evaluate_benchmark``), which pins the client RNGs a stochastic policy
    samples from. The name reads for the first family and is accurate for both:
    a rollout seed exists precisely to make the policy's randomization
    reproducible.

    Their appliers are not equally wide, so the accepted domain is not either.
    ``randomize`` / ``set_obs_noise`` reach ``default_rng``, which takes a
    non-negative integer of any width. A rollout seed is applied through
    :func:`~strands_robots.simulation.policy_runner.set_eval_seed`, which also
    reseeds the legacy NumPy global RNG (``numpy.random.seed``) - the one most
    policies draw from - and that refuses anything above :data:`MAX_EVAL_SEED`.
    ``max_seed`` carries that ceiling, so the rollout surfaces refuse a value
    they could not apply while the randomization surfaces keep the width they
    can honor. One rule with an explicit bound per destination is what stops
    the accepted domain drifting from the applier in either direction.

    ``allow_none`` is the same idea at the other end of the domain. ``None`` is
    a legitimate *parameter* value for most callers - it selects fresh entropy
    for ``randomize`` / ``set_obs_noise`` and means "do not seed" at the rollout
    facades - but it is not a *seed*, so an applier that has to hand one to an
    RNG cannot honor it: ``random`` and NumPy would reseed from entropy while
    ``torch.manual_seed`` refuses it, leaving a process-wide RNG side effect on
    a rollout that asked for none. ``allow_none=False`` refuses it there and
    drops ``None`` from the messages, so the reason a caller is given always
    describes the domain that caller actually has.

    Args:
        value: The candidate seed (``None`` selects fresh entropy).
        context: Method name to prefix the message with.
        max_seed: Largest value the caller's applier can honor, or ``None``
            when the non-negative-integer rule is the only bound.
        allow_none: Whether ``None`` is a value this caller can honor. True for
            a parameter where it selects fresh entropy or means "do not seed";
            False for an applier that has to hand a seed to an RNG, which has
            nothing to apply. When False the messages stop advertising ``None``
            too, so a caller is never offered a value this destination refuses.

    Returns:
        ``None`` when the seed is usable, otherwise the reason as a string.
    """
    none_clause = " or None" if allow_none else ""
    entropy_hint = " (None draws fresh entropy)" if allow_none else ""
    if value is None:
        if allow_none:
            return None
        return (
            f"{context}: seed is required; None is the absence of a seed, not a seed to apply. "
            f"To leave the RNGs untouched, do not call {context} - reseeding them from entropy "
            "is a global side effect an unseeded rollout must not acquire."
        )
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return f"{context}: seed must be a non-negative integer{none_clause}, got {value!r}{entropy_hint}"
    if int(value) < 0:
        return f"{context}: seed must be a non-negative integer{none_clause}, got {value!r}{entropy_hint}"
    if max_seed is not None and int(value) > max_seed:
        return (
            f"{context}: seed must be an integer in [0, {max_seed}]{none_clause}, got {value!r} "
            "(a rollout seed is applied to the legacy NumPy global RNG, which refuses a larger value)"
        )
    return None


_NON_FINITE_ACTION_REASON = (
    "A non-finite value is not clamped into the actuator's control range - the "
    "integrator has no usable target for it. On MuJoCo the physics step is "
    "discarded and every robot in the scene is reset to its initial pose, so no "
    "robot follows the command and the reset is not scoped to the commanded "
    "actuator or robot."
)


def _non_finite_action_error(label: str, value: Any) -> dict[str, Any] | None:
    """Structured error when an already-coerced action value is not finite.

    ``nan``/``inf`` are valid ``float`` objects, so a scalar-coercion check
    admits them and they reach the actuator command unexamined. The sibling
    state writers (``set_joint_positions`` / ``set_joint_velocities``) already
    refuse a non-finite value; this is the same rule for the actuator-command
    path, applied to both accepted action shapes so a mapping value and a vector
    entry cannot diverge.

    Args:
        label: How to name the offending element in the message - the caller
            supplies it because a mapping names a key while a vector names a
            position and the actuator key it binds to.
        value: The value, already confirmed to coerce to a scalar ``float`` by
            the caller (so the coercion here cannot raise).

    Returns:
        A structured ``{"status": "error", ...}`` dict, or ``None`` when the
        value is finite and therefore usable.
    """
    if math.isfinite(float(value)):
        return None
    return {
        "status": "error",
        "content": [
            {"text": (f"send_action: {label} must be finite (no nan/inf), got {value!r}. {_NON_FINITE_ACTION_REASON}")}
        ],
    }


_BOOLEAN_ACTION_REASON = (
    "A boolean cannot express an actuator command. float(True) is 1.0, and each "
    "drive reads 1.0 in its own units: a 1-radian target on a joint-position "
    "drive, a full-travel command on a normalized or tendon drive (a [0, 255] "
    "tendon gripper reads 1.0 as fully open), and an out-of-range value that is "
    "silently clamped on a drive whose ctrlrange excludes 1. The same True "
    "therefore commands a different pose on every actuator. Pass the command in "
    "the actuator's own units - for a binary gripper, the endpoint value rather "
    "than a flag."
)


def _unwrap_single_element_action_value(value: Any) -> Any:
    """Unwrap a length-1 sequence action value to its scalar, else return it as-is.

    The ``Policy.get_actions -> list[dict]`` contract emits ``list[float]`` for
    vector-valued keys, which for a 1-DOF key yields a length-1 list (GR00T's
    service unpack emits ``{"x": [0.05], ...}`` for the LIBERO delta-EEF
    layout). Such a value carries exactly one scalar and is unambiguous, so it
    is unwrapped rather than rejected.

    A 0-d numpy array (``np.array(True)``, ``np.mean(...)``) declares
    ``__len__`` and ``__getitem__`` but raises from ``len()``, which is why the
    length is read through :func:`strands_robots.utils.sequence_length` rather
    than probed here: it reports such a value as carrying no length, so the
    value is returned unchanged for the value checks to read (it already holds
    exactly one scalar, so there is nothing to unwrap).
    """
    if isinstance(value, (str, bytes, Mapping)):
        return value
    if not hasattr(value, "__getitem__"):
        return value
    return value[0] if sequence_length(value) == 1 else value


def _boolean_action_error(label: str, value: Any) -> dict[str, Any] | None:
    """Structured error when an action value is a python or numpy boolean.

    ``bool`` is an ``int`` subclass and ``numpy.bool_`` coerces the same way, so
    a scalar-coercion check admits both and they reach the actuator command as a
    silent ``1.0``/``0.0``. The teleop wire validator
    (:func:`strands_robots.mesh.security.validate_input_frame`, which
    ``InputReceiver`` applies through ``send_action``) already refuses a boolean
    for exactly that reason, so a remote peer's frame is held to a stricter
    domain than the local call it is applied through; this is the same rule for
    the actuator-command path, applied to both accepted action shapes so a
    mapping value and a vector entry cannot diverge.

    Args:
        label: How to name the offending element in the message - the caller
            supplies it because a mapping names a key while a vector names a
            position and the actuator key it binds to.
        value: The raw value, before scalar coercion (``float(True)`` is 1.0, so
            the boolean is unrecoverable once coerced).

    Returns:
        A structured ``{"status": "error", ...}`` dict, or ``None`` when the
        value is not a boolean and is therefore usable as a command.
    """
    if not is_boolean(value):
        return None
    return {
        "status": "error",
        "content": [
            {"text": (f"send_action: {label} must be a number, not a bool (got {value!r}). {_BOOLEAN_ACTION_REASON}")}
        ],
    }


# Why a runtime *state* write refuses a boolean, kept in one place because three
# surfaces quote it (set_joint_positions, set_joint_velocities, apply_force).
#
# The actuator-command path refuses one for a stronger reason - see
# :data:`_BOOLEAN_ACTION_REASON`: 1.0 is re-read in each drive's own units, so the
# same True commands a different pose on every actuator. Here 1.0 is a single
# unambiguous quantity - 1 radian, 1 rad/s, 1 N - so a boolean is merely wrong
# rather than ambiguous.
_BOOLEAN_STATE_REASON = (
    "float(True) is 1.0, so a boolean would be written as 1 radian, 1 rad/s or "
    "1 N depending on the surface, and the call would report success. Pass the "
    "quantity in the surface's own units."
)


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

    Concrete engines must set ``self._init_complete = True`` as the final
    statement of their ``__init__``. :meth:`__del__` consults it and skips an
    instance that never finished construction, so a half-built engine is never
    reported as a cleanup failure.
    """

    # Whether ``__init__`` ran to completion, i.e. whether this instance holds
    # engine resources that need releasing. :meth:`__del__` reads it to tell
    # "never acquired anything" from "failed to release something": on a
    # ``__new__`` skeleton, or an ``__init__`` that raised part-way, the class
    # attribute below answers False and the finalizer skips ``cleanup()``
    # instead of reporting whichever attribute ``__init__`` had not reached yet
    # as a cleanup failure. Declared on the class rather than assigned in an
    # ABC ``__init__`` so the read itself can never raise, and so lightweight
    # subclasses and test doubles need not thread ``super().__init__()``
    # through (the same constraint :meth:`_init_ros_bridge` documents).
    _init_complete: bool = False

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
                Only an ``int`` in ``[0, 232]`` names a domain; a value
                outside the RTPS port map raises :class:`ValueError`.
        """
        self._ros2_bridge_enabled = bool(ros2_bridge)
        # Refuse a domain id outside the RTPS port map here, so a backend that
        # only publishes later still rejects it at construction.
        if error := dds_domain_id_error(ros2_domain, "ros2_domain", type(self).__name__):
            raise ValueError(error)
        self._ros2_domain = ros2_domain
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

    def _unknown_robot_msg(self, requested: object) -> str:
        """Actionable 'robot not found' message for the backend-agnostic facade.

        Keeps the "Robot 'X' not found." prefix (the consistent error shape the
        concrete backends also emit via their own ``_unknown_robot_msg``), then
        appends a difflib close-match, the robots currently in the world, and the
        discovery action so an agent driving the API by name can recover a typo in
        zero extra calls instead of hitting a dead-end string. Uses the abstract
        :meth:`list_robots` primitive, so every backend inherits it; the MuJoCo
        engine overrides with a ``self._world.robots``-backed variant. Mirrors the
        ``_unknown_object_msg`` / ``_unknown_camera_msg`` pattern (#1299/#1303/#1306).

        ``requested`` is typed ``object`` because a name of any type reaches here:
        :func:`~strands_robots.simulation.models.registered` is total, so a name
        that cannot be a registry key resolves to "no such entity" and is
        reported rather than raising. Only the close match needs a string (see
        :func:`close_match_hint`); what is registered is a fact about the world,
        so it is listed for every name type instead of being replaced by an
        "empty scene" claim that would be false.
        """
        known = self.list_robots()
        msg = f"Robot '{requested}' not found."
        if known:
            msg += close_match_hint(requested, known)
            msg += f" Available robots: {known}. Use action='list_robots' to see all."
        else:
            msg += " No robots in the scene; add one with action='add_robot'."
        return msg

    # World lifecycle

    @abstractmethod
    def create_world(
        self,
        timestep: float | None = None,
        gravity: list[float] | None = None,
        ground_plane: bool = True,
        terrain: str | None = None,
        difficulty: float = 1.0,
    ) -> dict[str, Any]:
        """Create a new simulation world.

        ``terrain`` (``"rough"`` = value-noise bumps, ``"stairs"`` = discrete
        step plateaus rising along +x, ``"pyramid"`` = concentric step plateaus
        rising toward the centre, ``"slope"`` = a constant-grade inclined ramp;
        see :mod:`strands_robots.simulation.terrain`) lays down a
        deterministic heightfield instead of the flat ground plane so a
        locomotion policy can be spawned/evaluated on non-flat ground; it is
        only meaningful when ``ground_plane=True`` and defaults to ``None`` (a
        flat plane). Backends without heightfield support reject a non-None
        ``terrain`` with an actionable error rather than silently ignoring it.

        ``difficulty`` scales the terrain's peak elevation (``1.0`` = full
        height, ``<1`` gentler, ``>1`` harsher) so a curriculum can ramp
        terrain magnitude across resets without changing the terrain *kind*.
        It is only meaningful with a ``terrain``; setting ``difficulty != 1.0``
        with no ``terrain`` is rejected with an actionable error rather than
        silently having no effect. Must be a finite value ``> 0``.

        A floating-base robot added to a terrain world is spawned seated on
        the local terrain surface (raised by the heightfield height beneath
        it) at ``add_robot`` and on ``reset()``, so its feet are not buried
        below the raised terrain.

        ``timestep`` (seconds) and ``gravity`` must be values the engine can
        honor, on the same terms the ``set_timestep`` / ``set_gravity`` setters
        enforce: ``timestep`` a finite number ``> 0`` (``0`` is rejected, never
        coalesced to the engine default), ``gravity`` a 3-element vector of
        finite numbers or a real scalar taken as the z-component. A value the
        backend cannot apply is rejected with a structured error rather than
        compiled into the world - a world built around a negative or ``nan``
        ``dt`` integrates backwards or to ``nan`` while every subsequent call
        still reports ``status="success"``. ``None`` means "use the engine
        default".
        """
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
        ``mj_resetData`` (which alone zeroes all derived quantities). It also
        re-applies any per-robot home pose captured from an
        ``add_robot(keyframe=...)`` spawn, so a keyframe pose survives a reset
        instead of collapsing to the zero configuration.
        """
        ...

    # Steps a ``step`` implementation may advance per lock acquisition. This
    # bounds the window in which every OTHER locked method on the engine - a
    # concurrent ``get_state``, ``get_observation``, ``stop_policy`` or the
    # ``cleanup`` world handoff - is blocked, so a long run is slow rather than
    # unresponsive.
    #
    # It is deliberately not a limit on the total work one call may request.
    # That is a per-backend resource policy (MuJoCo's ``_MAX_STEPS_PER_CALL``)
    # whose value cannot be shared: 100_000 MuJoCo ``mj_step`` calls on a small
    # arm and 100_000 RTX-rendered Isaac ``world.step`` calls are not the same
    # amount of wall time, and a Newton step is a control step of ``substeps``
    # solver steps, so one number does not express one policy. What IS shared is
    # the reason above, which is why the granularity is one constant here and
    # the ceiling is not. See #1871.
    _STEPS_PER_BATCH = 1000

    @abstractmethod
    def step(self, n_steps: int = 1) -> dict[str, Any]:
        """Advance simulation by n physics steps.

        When the backend exposes an engine lock (``self._lock``, all in-tree
        backends), implementations must not hold it for the whole count: they
        release it at least every :attr:`_STEPS_PER_BATCH` steps, and re-check
        that the world still exists on each batch boundary before advancing it,
        aborting with a structured error naming the steps completed if it does
        not. Releasing the lock is what makes a concurrent teardown reachable
        mid-call, so the two halves are one contract rather than two - the same
        pairing ``_primitive_abort_reason`` already makes for the motion-primitive
        loops, which release the lock on the same schedule.
        """
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
        keyframe: str | int | None = None,
    ) -> dict[str, Any]:
        """Add a robot to the simulation.

        ``keyframe`` optionally spawns the robot in a canonical pose declared
        by a ``<keyframe>`` in its source model (e.g. panda ``"home"``, aloha
        ``"neutral_pose"``) instead of the default all-zero configuration.
        Pass the keyframe name (``str``) or index (``int``). The pose is
        applied to the robot's joints by name and stored so :meth:`reset`
        restores it (a keyframe spawn is sticky across resets, matching how a
        benchmark restores its canonical start each episode). ``None`` (the
        default) keeps the historical zero-pose spawn. An unknown keyframe
        name/index is a hard error that names the available keyframes; it
        never silently falls back to zeros.
        """
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
        actuator set matches their joint set. Two kinds of backend override it.
        One has a distinct actuator *namespace* (MuJoCo tendon grippers) and
        returns actuator short-names instead. The other shares the namespace but
        commands a *subset* of it: the Newton engine drops a floating base's
        6-DoF free joint, which is a joint and not a commandable scalar, so its
        action keys are the joint names minus that one. An override may
        therefore rename or narrow this list, and a caller must not assume it
        has the same width as :meth:`robot_joint_names`.
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

        An unresolvable ``policy_provider`` is reported here too. That check
        runs before the observation lookup below: a robot whose observation is
        not yet available would otherwise take the early return and let the
        unresolvable name reach ``create_policy``, whose raise escapes the
        ``status=error`` envelope this method exists to produce.

        Returns:
            A ``status=error`` dict (for the caller to return) when the
            provider cannot be resolved or its preflight rejects the
            configuration; ``None`` when the check passes, is a no-op, or the
            observation is not yet available.
        """
        from strands_robots.policies import policy_provider_error, preflight_policy

        reason = policy_provider_error(policy_provider, **(policy_config or {}))
        if reason is not None:
            return {"status": "error", "content": [{"text": reason}]}

        obs = self.get_observation(robot_name)
        if not isinstance(obs, dict) or not obs:
            return None
        try:
            preflight_policy(policy_provider, set(obs.keys()), **(policy_config or {}))
        except ValueError as e:
            return {"status": "error", "content": [{"text": str(e)}]}
        return None

    def _unresolvable_policy_provider_error(
        self,
        policy_provider: str,
        policy_config: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Report an unresolvable ``policy_provider`` as a structured error.

        For surfaces that cannot use :meth:`_preflight_policy_config` because
        they build the policy elsewhere -- notably a ``start_policy`` that
        submits the rollout to a worker thread -- this gives the same verdict
        synchronously, so the caller is refused instead of being told a
        rollout started that could never build its policy.

        Args:
            policy_provider: Provider name / smart string.
            policy_config: Provider kwargs (the policy_config).

        Returns:
            A ``status=error`` dict when the provider cannot be resolved;
            ``None`` when it resolves.
        """
        from strands_robots.policies import policy_provider_error

        reason = policy_provider_error(policy_provider, **(policy_config or {}))
        if reason is None:
            return None
        return {"status": "error", "content": [{"text": reason}]}

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
        is_static: bool | None = None,
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

        A backend MUST NOT discard ``size`` components the caller did supply.
        When the vector is shorter than the shape consumes it either rejects it
        (MuJoCo: the per-shape component count is part of the contract) or pads
        only the missing *trailing* components from a documented default
        (Isaac). Replacing the whole vector with a backend default compiles a
        differently-sized object while reporting success -- and the reported
        size echoes what was asked for, not what was built.

        The same rule applies to ``color``: a backend either honors the
        component count it was given or rejects it, and may complete only
        components it documents a default for (MuJoCo completes an RGB triple
        with an opaque alpha, and rejects every other count). Falling back to
        the backend's default colour paints a surface the caller never asked
        for under a success result.

        ``mass`` must be a finite number greater than zero for a dynamic
        object. A backend MUST NOT establish a body on a mass its own
        ``set_body_properties`` would refuse: a non-finite mass makes the first
        integration step produce ``nan`` and, because the solver shares one
        state vector, poisons every other body in the world too.

        ``is_static`` is tri-state, and ``None`` is the default because it
        is the only value that means "the caller did not specify". That is what
        lets a backend derive the answer from ``shape``: MuJoCo forces a
        ``shape="plane"`` static -- a plane is infinite and cannot carry a
        dynamic mass -- and *refuses* an explicit ``is_static=False`` there
        rather than quietly overriding it. A backend with no shape-derived rule
        resolves ``None`` to ``False`` (dynamic). Declaring the default as
        ``False`` would state a value the default backend does not deliver, and
        would make restating that declared default a hard error for the one
        shape whose whole point is being static.

        ``material`` (optional): backend-specific visual material/texture
        spec. ``None`` keeps the flat ``color`` rgba (unchanged); a backend
        that supports it (MuJoCo) attaches a real material so surfaces can be
        matte or textured. Backends that do not support it should reject a
        non-``None`` ``material`` loudly rather than silently ignore it. A
        supporting backend must likewise reject material keys it cannot honor
        (a typo, or a field from another renderer) instead of dropping them --
        a dropped key renders the backend default while reporting success.
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
              Shape ``(H, W, 3)``. A key MUST carry the view of the camera it
              names; a backend that cannot render that camera MUST omit the key
              rather than substitute another view (the free/overview camera in
              particular), because every consumer of this schema - a policy
              reading ``observation.images.<name>``, a recorded dataset column -
              reads the key as a promise about which camera it is looking
              through and has no way to detect a substitution. Cameras whose
              render fails MAY be omitted; joint state MUST still be returned.
            - Floating base: a robot whose root is a 6-DoF free joint (a
              humanoid's named ``floating_base_joint`` or a mobile base's
              unnamed ``<freejoint>``) does NOT report that free joint as a
              scalar ``"<joint_name>"`` entry - its qpos is [xyz + quat], so a
              scalar would report the base x-coordinate as a joint angle and
              drop the rest. Instead it surfaces the full base pose + twist as
              ``"base_pos"`` (world x,y,z incl. height), ``"base_quat"``
              (w,x,y,z), ``"base_lin_vel"`` and ``"base_ang_vel"``, matching
              :meth:`get_robot_state`'s ``"base"`` entry. Absent for fixed-base
              arms.
            - ``"body.<name>.pos"`` / ``".quat"`` / ``".lin_vel"`` /
              ``".ang_vel"`` (list[float]): World pose + twist of a NAMED body,
              present only when the running policy declared that body in
              :attr:`~strands_robots.policies.base.Policy.required_bodies`.
              Backends do not emit these from ``get_observation`` itself - the
              runtime (:class:`~strands_robots.simulation.policy_runner.PolicyRunner`)
              merges them in from :meth:`get_body_state` for the declared bodies
              only, so the default observation is unchanged and nothing pays for
              a link nobody asked for. Motion-mimic trackers need them because
              their anchor link (``torso_link`` on a G1) is separated from
              ``base_quat`` (the pelvis) by the waist joints.

        Single-camera rendering is :meth:`render`'s job, not this method's.
        For batched multi-robot observation (future Isaac / Newton), add a
        separate ``get_observations(robot_names)`` method - do NOT extend
        this one.

        Args:
            robot_name: Which robot to observe. If ``None`` and exactly one
                robot exists, that robot is used; otherwise returns ``{}``.
            skip_images: Skip camera rendering and return joint state only.
                Rendering dominates the per-step cost, so every consumer that
                reads joint values alone passes ``True`` - the predicate /
                reward DSL (:mod:`~strands_robots.simulation.predicates`), the
                LIBERO adapter's state reads, and the ROS 2 bridge when it
                publishes ``joint_states`` without ``image_raw``. Camera keys
                are then absent from the result rather than present and empty,
                so a caller must not read a missing frame as a render failure.
                A backend overrides a ``True`` here while a dataset recording is
                active - the recorded frames must carry the camera images the
                schema declared - so this is a hint, not a guarantee that
                nothing renders. Defaults to False (render every attached
                camera).

        Returns:
            Observation dict per schema above. Returns ``{}`` if the world
            is not yet created or ``robot_name`` is unknown.
        """
        ...

    def _ground_height_at(self, x: float, y: float) -> float:
        """Terrain surface height (world z) beneath world ``(x, y)``; ``0.0`` on flat ground.

        Default ``0.0`` -- a flat ground plane, and any backend without a
        heightfield. The MuJoCo backend overrides this to sample a
        ``create_world(terrain=...)`` heightfield so that height-based locomotion
        predicates (``base_below_z``) measure a base's clearance above the
        *local* ground instead of an absolute world z -- an absolute test
        silently misses a collapse on a
        raised terrain plateau (the base still sits above a flat-ground
        threshold). Not a public tool action.
        """
        return 0.0

    def get_ground_height(self, x: SupportsFloat, y: SupportsFloat) -> dict[str, Any]:
        """Query the terrain surface height (world z) beneath world ``(x, y)``.

        Public counterpart of the internal :meth:`_ground_height_at` hook: a
        ``create_world(terrain=...)`` heightfield raises the local ground up to
        ``TERRAIN_ELEVATION * difficulty`` above ``z=0``, and there was no public
        way to ask where that surface is. Callers building a terrain scene need
        it to place an object / camera / goal *on* the surface -- an object added
        at a flat-ground ``z`` (computed as if the support were at ``z=0``) on a
        raised plateau spawns *buried* in the heightfield and sinks through
        instead of resting on it. The same local-height sampler already backs the
        terrain-relative locomotion predicates (``base_below_z``) and the
        spawn/reset base-seating; this exposes it as a facade query.

        Returns ``0.0`` for a flat ground plane, for any backend without a
        heightfield, and before ``create_world`` (a world-less engine has no
        terrain), so a non-terrain -- or not-yet-built -- world reports a flat
        surface rather than raising, unlike the world-scoped physics queries.

        Args:
            x: World x coordinate. Any object convertible to ``float``
                (``SupportsFloat``), including NumPy scalars, that is a finite
                real number.
            y: World y coordinate. Same accepted types as ``x``.

        Returns:
            Agent-tool status dict. On success ``content`` carries a
            ``{"json": {"x": ..., "y": ..., "height": ...}}`` block with the
            surface height in meters. Errors when ``x`` / ``y`` is not a finite
            real number. Accepts any real scalar, including NumPy scalar
            types (``np.float32`` / ``np.int64`` / ...), since terrain
            coordinates naturally come from ``mj_data`` / an observation
            (a NumPy array), not hand-typed Python floats.
        """
        for label, val in (("x", x), ("y", y)):
            if isinstance(val, bool) or not isinstance(val, numbers.Real) or not math.isfinite(float(val)):
                return {
                    "status": "error",
                    "content": [{"text": f"get_ground_height: {label} must be a finite number, got {val!r}."}],
                }
        fx, fy = float(x), float(y)
        height = float(self._ground_height_at(fx, fy))
        return {
            "status": "success",
            "content": [
                {"text": f"Ground height at ({fx:.4f}, {fy:.4f}) = {height:.4f}m"},
                {"json": {"x": fx, "y": fy, "height": height}},
            ],
        }

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
        silently truncated (which would drop commands - e.g. a gripper axis). A
        mapping is returned unchanged once every value is confirmed to coerce to
        a scalar float; a *single-element* sequence/array value (e.g. GR00T's
        per-key ``[0.05]`` rows - the documented ``list[float]`` shape of the
        ``Policy.get_actions`` contract for a 1-DOF key) is unwrapped to its
        scalar, while a *multi-element* value is rejected with an actionable
        error rather than raised as an unhandled ``TypeError`` deep in the
        actuator-application loop.

        Every value must additionally be **finite**. ``nan``/``inf`` are valid
        ``float`` objects, so the scalar-coercion check above admits them and they
        reach the actuator command unexamined. A non-finite command is not clamped
        into the actuator's range: MuJoCo finds the resulting non-finite ``qacc``,
        discards the step and resets the world to its initial pose - on every
        substep, and for *every* robot in the scene rather than only the commanded
        one - while ``send_action`` still reports success, so a rollout recording
        such a step writes a trajectory no robot followed. An ``inf`` is instead
        clamped into ``ctrlrange``, i.e. silently rewritten into a full-travel
        command. The sibling state writers (:meth:`set_joint_positions` /
        :meth:`set_joint_velocities`) already refuse a non-finite value, so this
        holds the actuator-command path to the same rule.

        A value must additionally not be a **boolean**. ``bool`` is an ``int``
        subclass and ``numpy.bool_`` coerces identically, so the scalar check
        admits both and they reach the actuator as ``1.0``/``0.0`` - and 1.0 is
        not one command: it is a 1-radian target on a joint-position drive, a
        full-travel command on a normalized or tendon drive, and an out-of-range
        value that is silently clamped where ``ctrlrange`` excludes 1. A boolean
        is the conventional binary-gripper action, so the value arrives at this
        surface routinely rather than as a typo. The teleop wire validator
        (:func:`strands_robots.mesh.security.validate_input_frame`) already
        refuses a boolean so it "can't masquerade as a 1.0/0.0 command", and it
        applies frames through ``send_action`` - so refusing it here holds the
        local call to the domain its own remote surface enforces.

        Otherwise the domain is finiteness: a numeric string is an accepted
        spelling of a scalar here, and a finite magnitude outside ``ctrlrange``
        is a units question already surfaced by the clamp warning.

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
            # Each value is applied downstream as ``float(value)`` per actuator
            # with no guard, so a non-scalar value (a list / tuple / multi-element
            # array - e.g. a policy emitting a vector-valued key such as a
            # ``base_velocity`` [vx, vy, omega]) would raise an unhandled
            # ``TypeError`` past send_action's structured-error contract and crash
            # the caller mid-rollout, after partially applying the earlier keys.
            # Validate every value coerces to a scalar float up front so the whole
            # action is rejected atomically with an actionable message, symmetric
            # with the vector-form non-numeric-entry validation below.
            #
            # Single-element unwrap (#1538 GR00T-LIBERO regression): the
            # ``Policy.get_actions -> list[dict]`` contract emits ``list[float]``
            # for vector-valued keys, which for a 1-DOF key yields a length-1
            # list (GR00T's service unpack emits ``{"x": [0.05], ...}`` for the
            # LIBERO delta-EEF layout). Such a value carries exactly one scalar
            # and is unambiguous, so unwrap it instead of rejecting - the
            # pre-validation code path applied it fine via the LIBERO action
            # controller's own scalar coercion, and rejecting it silently
            # no-ops an entire GR00T eval to success_rate=0. Multi-element
            # values (the actual #1179 crash class) are still rejected
            # atomically.
            normalized: dict[str, Any] = {}
            for key, value in action.items():
                value = _unwrap_single_element_action_value(value)
                # Before the scalar coercion: ``float(True)`` succeeds, so the
                # coercion below cannot see a boolean and the value would reach
                # the actuator as a silent 1.0/0.0 command.
                if error := _boolean_action_error(f"action value for key '{key}'", value):
                    return None, error
                try:
                    float(value)
                except (TypeError, ValueError):
                    return None, {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"send_action: action value for key '{key}' must be a "
                                    "scalar number (one value per actuator/joint), got "
                                    f"{type(value).__name__}."
                                )
                            }
                        ],
                    }
                if error := _non_finite_action_error(f"action value for key '{key}'", value):
                    return None, error
                normalized[key] = value
            return normalized, None

        # ``str``/``bytes`` are iterable but never a valid multi-joint action;
        # a scalar has no length. Reject both with an actionable message instead
        # of producing garbage character/positional keys downstream.
        if isinstance(action, (str, bytes)) or sequence_length(action) is None:
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
            # Keep the raw entries: ``float`` erases a boolean into 1.0/0.0, so
            # the bool gate below has to see the value the caller passed.
            raw_entries = list(action)
            values = [float(v) for v in raw_entries]
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
        bound: dict[str, Any] = {}
        for idx, (name, raw, value) in enumerate(zip(action_keys, raw_entries, values, strict=True)):
            label = f"action vector entry {idx} ('{name}')"
            if error := _boolean_action_error(label, raw):
                return None, error
            if error := _non_finite_action_error(label, value):
                return None, error
            bound[name] = value
        return bound, None

    @staticmethod
    def _coerce_joint_state_map(
        values: dict[str, Any],
        name: str,
        method: str,
    ) -> tuple[dict[str, float], dict[str, Any] | None]:
        """Coerce a ``{joint_name: value}`` state map to finite floats before any write.

        Backs the kinematic state writers on every backend
        (:meth:`set_joint_positions` / :meth:`set_joint_velocities`), so the
        accepted domain cannot differ by which engine the caller happens to be
        driving.

        Each value must be a real number (Python or NumPy scalar) and finite, and
        must not be a boolean. A non-numeric value would otherwise raise
        ``ValueError`` from ``float(value)`` past the structured-error dispatch
        contract, and ``nan`` / ``inf`` would slip straight into the engine's
        joint state - MuJoCo's ``mj_forward`` propagates the ``nan`` across the
        whole kinematic state (or an ``inf`` velocity blows up the integrator),
        and PhysX reports a non-finite articulation as an "Illegal
        BroadPhaseUpdateData - non-finite bounds" error from a later step -
        while the tool still reports ``status="success"``. A boolean is refused
        for the reason in :data:`_BOOLEAN_STATE_REASON`: it survives ``float()``
        as a silent ``1.0``, so it is the one invalid value the finiteness check
        cannot see. Validating up front keeps the write atomic: an invalid value
        leaves the joint state untouched.

        Args:
            values: The ``{joint_name: value}`` mapping to validate.
            name: Parameter name (``"positions"`` / ``"velocities"``), used in error text.
            method: Calling method name, used in error text.

        Returns:
            ``(coerced, None)`` on success, or ``({}, error_dict)`` on the first
            invalid value -- matching the structured-error tool contract so the
            caller never raises past dispatch.
        """
        out: dict[str, float] = {}
        for jnt_name, value in values.items():
            # Before float(): float(True) is 1.0, so the boolean is unrecoverable
            # once coerced and the write would report success having set 1 rad /
            # 1 rad/s. numpy.bool_ needs the .item() unwrap is_boolean applies.
            if is_boolean(value):
                return {}, {
                    "status": "error",
                    "content": [
                        {
                            "text": f"{method}: '{name}' value for joint '{jnt_name}' must be a number, not a bool (got {value!r}). {_BOOLEAN_STATE_REASON}"
                        }
                    ],
                }
            try:
                f = float(value)
            except (TypeError, ValueError):
                return {}, {
                    "status": "error",
                    "content": [
                        {"text": f"{method}: '{name}' value for joint '{jnt_name}' must be a number, got {value!r}"}
                    ],
                }
            if not math.isfinite(f):
                return {}, {
                    "status": "error",
                    "content": [
                        {
                            "text": f"{method}: '{name}' value for joint '{jnt_name}' must be finite (no nan/inf), got {value!r}"
                        }
                    ],
                }
            out[jnt_name] = f
        return out, None

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

        ``n_substeps`` is a **positive** whole number, on the shared
        :func:`~strands_robots.utils.positive_whole_number_error` domain every
        backend applies. A NumPy or float count with an integral value is
        honored and coerced; a fractional, zero, negative, non-finite, boolean
        or non-numeric count is refused as a structured error, and nothing is
        written when it is - a refusal arriving after the write would leave the
        robot commanded and the world un-advanced, which is the one state this
        surface must never report an error from. The floor is ``1`` rather than
        :meth:`step`'s ``0`` precisely because of the write: "advance nothing"
        is ``step(0)``, an accepted no-op that commands nothing, while a
        ``send_action`` advancing nothing leaves a target the world never
        integrates. It is also the floor both producers of this count already
        enforce - ``PolicyRunner._control_substeps`` returns ``>= 1`` and
        raises otherwise, and ``training.rl.env.SimEnv`` refuses an
        ``n_substeps`` below 1 - so this surface was the only member of that
        chain without the guarantee.

        Backends are responsible for internal thread-safety (e.g.
        MuJoCo acquires self._lock here). PolicyRunner does not manage
        locks.

        Returns:
            Dict with ``status`` and ``content``. When action keys cannot
            be resolved, the ``content`` list includes a ``json`` block with
            ``unresolved_keys`` so callers can self-correct. ``status`` is
            ``"error"`` when ``n_substeps`` is outside its domain.
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
        method: str = "run_policy",
    ) -> tuple[float, int | None, dict[str, Any] | None]:
        """Resolve a step horizon into a wall-clock duration.

        ``n_steps`` (primary) or the legacy ``max_steps`` alias specify the
        rollout length as a step count; ``duration = n_steps / control_frequency``.
        ``n_steps`` wins when both are passed. The effective horizon is validated
        before the division against the shared positive-count domain
        (:func:`~strands_robots.utils.positive_count_error`) - the same domain
        :meth:`_validate_positive_int` already applies to ``eval_policy``'s
        ``max_steps`` and to ``n_episodes`` / ``action_horizon`` /
        ``control_substeps`` - so a horizon that is not a whole number of steps
        is reported as a caller error rather than silently truncated. A bare
        ``<= 0`` test only saw the sign: ``n_steps=2.7`` ran two steps and
        ``n_steps=True`` ran one, both reported as a successful rollout of a
        horizon the caller never asked for, while the identically-named
        ``eval_policy`` budget refused both.

        Args:
            n_steps: Primary step-count horizon, or ``None``. Must be a
                positive integer when given.
            max_steps: Legacy alias, normalized to ``n_steps`` when ``n_steps``
                is ``None``, and validated under its own name so a caller who
                wrote ``max_steps`` is not pointed at a parameter they never
                passed. Same domain as ``n_steps``.
            control_frequency: Target control-loop frequency in Hz.
            duration: Fallback wall-clock duration used when no step horizon
                is given. Returned unchanged when no horizon is given, so the
                caller must still validate it with :meth:`_validate_duration`
                (this helper only owns the horizon-to-duration conversion).
            method: Public method name, used to prefix the error message.

        Returns:
            A ``(duration, n_steps, error)`` tuple. When ``error`` is non-None
            it is a structured ``{"status": "error", ...}`` dict and the other
            fields must be ignored. Otherwise ``duration`` is the resolved
            wall-clock duration (recomputed from the horizon when one was
            given) and ``n_steps`` is the normalized step count (or ``None``).
        """
        if n_steps is not None:
            if error := positive_count_error(n_steps, "n_steps", method):
                return duration, n_steps, {"status": "error", "content": [{"text": error}]}
        elif max_steps is not None:
            # Validate the alias under its OWN name: the caller wrote
            # ``max_steps``, so a refusal naming ``n_steps`` points them at a
            # parameter they never passed. The normalization below is then exact
            # (an ``int()`` coercion here would be a second, weaker contract
            # that silently truncated the value the domain just accepted).
            if error := positive_count_error(max_steps, "max_steps", method):
                return duration, max_steps, {"status": "error", "content": [{"text": error}]}
            n_steps = max_steps
        if n_steps is not None:
            # control_frequency is validated as a positive number at the public
            # entry points (run_policy / start_policy / eval_policy) via
            # _validate_positive_frequency before this helper runs, so the
            # division below is safe.
            duration = float(n_steps) / float(control_frequency)
        return duration, n_steps, None

    @staticmethod
    def _validate_action_horizon(
        action_horizon: Any, method: str, param: str = "action_horizon"
    ) -> dict[str, Any] | None:
        """Reject a non-positive-integer ``action_horizon`` at the public API.

        ``action_horizon`` is how many actions are consumed from each policy
        chunk before re-querying. A value below 1 (or a non-int) is meaningless
        and would otherwise be silently clamped to 1 by
        :func:`~strands_robots.policies.base.resolve_chunk_length`, hiding the
        caller's mistake behind a rollout that does not run the requested
        horizon. ``True`` is rejected for the same reason: it would act as a
        silent horizon of 1. Returns a structured ``{"status": "error", ...}``
        dict to surface, or ``None`` when the value is valid.

        The domain is delegated to :meth:`_validate_positive_int`, which the
        hardware control loop's ``action_horizon`` guard shares through
        :func:`~strands_robots.utils.positive_count_error`, so a horizon refused
        for a simulated rollout cannot be accepted for the real arm.

        Args:
            action_horizon: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.
            param: Parameter label for the error message. Multi-robot drivers
                accept a ``{robot_name: horizon}`` mapping and pass
                ``"action_horizon['alice']"`` so the message names the entry the
                caller got wrong rather than the whole mapping.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        return SimEngine._validate_positive_int(action_horizon, param, method)

    @staticmethod
    def _validate_per_robot_mapping(
        mapping: Mapping[Any, Any], driven: Iterable[str], param: str, method: str
    ) -> dict[str, Any] | None:
        """Reject a per-robot mapping key that names no robot in this call.

        Multi-robot drivers accept ``{robot_name: value}`` overrides alongside
        the ``policies`` mapping that names the robots being driven. Reading
        those overrides with ``mapping.get(robot, default)`` silently discards
        every key that does not match a driven robot, so a typo'd or stale robot
        name left the rollout running the defaults while still reporting
        ``status="success"`` - the caller's per-robot request was never applied
        and nothing said so. Keys that ARE absent from the mapping keep their
        documented default (the mapping is an override layer, so a partial map
        is legitimate); it is the unmatched key that is a caller error.

        Args:
            mapping: The caller-supplied ``{robot_name: value}`` mapping.
            driven: Robot names being driven in this call (the authoritative
                key set - usually the ``policies`` mapping's keys).
            param: Parameter name, used in the error message.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the unmatched keys, or ``None`` when every key
            names a driven robot.
        """
        known = list(driven)
        unknown = [key for key in mapping if key not in known]
        if not unknown:
            return None
        text = f"{method}: {param} names {'robots' if len(unknown) > 1 else 'a robot'} not driven by this call: "
        text += f"{unknown!r}."
        matches: list[str] = []
        for key in unknown:
            if isinstance(key, str):
                matches += [m for m in difflib.get_close_matches(key, known, n=2, cutoff=0.6) if m not in matches]
        if matches:
            text += " Did you mean: " + ", ".join(matches) + "?"
        text += f" Robots driven by this call: {known} (the keys of 'policies')."
        return {"status": "error", "content": [{"text": text}]}

    @staticmethod
    def _validate_positive_int(value: Any, name: str, method: str) -> dict[str, Any] | None:
        """Reject a non-positive-integer count at the public API.

        Shared guard for the rollout count knobs that must be ``>= 1`` -
        ``n_episodes`` (how many reset->rollout episodes to run), ``max_steps``
        (the per-episode step cap), ``control_substeps`` and
        ``action_horizon``. A zero/negative/non-int value would otherwise flow
        into the rollout loop and produce a degenerate result that still reports
        ``status="success"``: an eval over zero episodes, or episodes of zero
        length, that fabricate a 0% success rate (``Episodes: -2 | Success:
        0/-2``) instead of surfacing the caller's mistake. Returns a structured
        ``{"status": "error", ...}`` dict to surface, or ``None`` when the value
        is valid.

        Thin binding of the shared count domain
        (:func:`~strands_robots.utils.positive_count_error`) to this class's
        tool-error envelope. The domain lives in :mod:`strands_robots.utils`
        because the hardware control loop's ``action_horizon`` must enforce the
        identical rule and :mod:`strands_robots.hardware_robot` cannot import
        :mod:`strands_robots.simulation`; sharing one implementation is what
        stops the same count being refused for a digital twin and accepted for
        the arm it mirrors. ``bool`` is part of that shared domain rather than
        each caller's business: a bare ``value < 1`` test lets ``True`` through
        as a silent count of 1 while rejecting ``False``.

        Args:
            value: The caller-supplied value to validate.
            name: Parameter name, used in the error message.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        if error := positive_count_error(value, name, method):
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_seed(seed: Any, method: str) -> dict[str, Any] | None:
        """Reject an unusable RNG seed at the public API.

        A rollout seed is the caller's reproducibility contract: it reseeds the
        client RNGs (and is forwarded to ``policy.reset``) so the same scene and
        the same policy replay identically. Only a non-negative integer can do
        that - the seed ends at ``numpy.random.seed`` / ``default_rng``, which
        refuses everything else - so a value that cannot be applied is refused
        at the call that supplied it rather than at the first draw.

        Without this guard the same mistake surfaced three different ways, none
        of them naming the parameter: ``run_policy`` raised NumPy's own
        ``TypeError: Cannot cast scalar from dtype('float64') to dtype('int64')``
        straight out of a method documented to return this envelope,
        ``start_policy`` reported "started" and failed on its worker thread, and
        ``True`` was accepted everywhere as a silent seed of ``1``.

        Thin binding of :func:`randomization_seed_error` to this class's
        tool-error envelope, so a seed refused for ``randomize`` cannot be
        accepted for the rollout whose reproducibility it is supposed to pin.

        The rollout binding supplies :data:`MAX_EVAL_SEED` as the ceiling. Its
        applier reseeds the legacy NumPy global RNG as well as ``default_rng``,
        and that one refuses a larger value - so without the bound a seed in
        ``[2**32, inf)`` passed this guard and then raised NumPy's own message
        from inside the rollout, which is the failure this guard exists to
        replace. ``randomize`` / ``set_obs_noise`` reach only ``default_rng``
        and keep the unbounded domain they can honor.

        Args:
            seed: The caller-supplied value (``None`` draws fresh entropy).
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        if error := randomization_seed_error(seed, method, max_seed=MAX_EVAL_SEED):
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_control_substeps(control_substeps: Any, method: str) -> dict[str, Any] | None:
        """Reject a ``control_substeps`` override the rollout cannot honor.

        ``control_substeps`` is how many physics steps are integrated per
        applied action. ``None`` (the default) means "derive it from the
        backend's physics timestep so the arm tracks the full control period",
        which is why ``None`` is accepted here. Any explicit value must be a
        positive integer: ``0`` / a negative value was previously clamped to a
        single physics step by ``max(1, int(override))``, which is precisely the
        under-integration pathology
        :meth:`~strands_robots.simulation.policy_runner.PolicyRunner._control_substeps`
        exists to avoid - the arm integrates ~2 ms of a 20 ms control period, so
        the rollout reports ``status="success"`` while the policy looks like a
        no-op. A float (``2.7``) was silently truncated, ``True`` acted as a
        silent 1 substep (``bool`` is an ``int`` subclass), and ``nan`` / ``inf``
        reached ``int()`` deep inside the runner and surfaced as a bare
        ``ValueError``/``OverflowError`` instead of the structured tool-error
        dict the public API contracts.

        The positive-integer domain itself - including the ``bool`` rejection,
        which this guard used to repeat locally - is delegated to
        :meth:`_validate_positive_int` so this guard and the rollout count knobs
        cannot drift apart.

        Args:
            control_substeps: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid (including ``None``, which means "auto-derive").
        """
        if control_substeps is None:
            return None
        return SimEngine._validate_positive_int(control_substeps, "control_substeps", method)

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
        than the structured tool-error dict the public API contracts. Any real
        scalar is accepted (``numbers.Real``), so a NumPy-scalar frequency such
        as ``np.float32(50.0)`` or ``np.int64(50)`` passes; ``bool`` is rejected
        explicitly (an ``int`` subclass, ``True`` would slip through and act as a
        silent 1 Hz) and non-finite values (``nan``/``inf``) are rejected before
        the ``<= 0`` comparison. That domain is
        :func:`~strands_robots.utils.positive_finite_number_error`, shared with
        every other rate/duration knob (including the teleop control loop, which
        divides by its ``hz`` the same way) so they cannot diverge. Returns a
        structured error dict to surface, or ``None`` when valid.

        Args:
            control_frequency: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None``.
        """
        error = positive_finite_number_error(control_frequency, "control_frequency", method)
        if error:
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_timestep(timestep: Any, method: str, param: str = "timestep") -> dict[str, Any] | None:
        """Reject a physics timestep the integrator cannot honor.

        The timestep is the ``dt`` every physics substep advances by, so a
        non-positive or non-finite value poisons the whole world rather than
        one call: a negative ``dt`` runs the integrator backwards (sim time
        counts down, accelerations blow up) and a ``nan`` makes every state
        ``nan`` - both while the creating call still reports
        ``status="success"``. ``0`` is equally unusable, and must be rejected
        rather than coalesced to the engine default: a caller that passed
        ``0`` and was silently given ``0.002`` never learns its value was
        discarded. This is the same contract
        :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.set_timestep`
        already enforces, so the value cannot be set at world creation on
        terms the setter would refuse.

        Args:
            timestep: The caller-supplied value. Anything ``float()`` accepts
                is coerced (so a NumPy scalar passes); ``bool`` is rejected
                explicitly since ``True`` would act as a silent 1-second step.
            method: Public method name, used to prefix the error message.
            param: Parameter name to quote - ``"timestep"`` for a caller
                argument, or the name of the engine default it fell back to.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is usable.
        """
        message = f"{method}: {param} must be a finite positive number, got {timestep!r}."
        # is_boolean, not isinstance(timestep, bool): numpy.bool_ is not a bool
        # subclass, so the narrower check refused a hand-typed True and admitted
        # the np.True_ a comparison produces - a 1-second dt under success.
        if is_boolean(timestep):
            return _boolean_world_error(method, param, timestep)
        try:
            value = float(timestep)
        except (TypeError, ValueError):
            return {"status": "error", "content": [{"text": message}]}
        if not math.isfinite(value) or value <= 0:
            return {"status": "error", "content": [{"text": message}]}
        return None

    @staticmethod
    def _validate_mass(mass: Any, method: str, param: str = "mass") -> dict[str, Any] | None:
        """Reject a body mass the physics engine cannot honor.

        A dynamic body's mass is the divisor of every force applied to it, so a
        value outside ``(0, inf)`` does not merely mis-size one object - it
        poisons the whole world on the next step. ``inf`` makes the very first
        integration produce ``nan`` acceleration, and because the solver shares
        one state vector, every *other* body's ``qpos``/``qvel`` goes ``nan``
        with it. ``0`` and negatives violate MuJoCo's
        "mass and inertia of moving bodies must be larger than mjMINVAL"
        invariant, which surfaces as a compile refusal that names neither the
        parameter nor the reason. This is the same domain
        :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.set_body_properties`
        already enforces when it writes the same ``body_mass`` field, so a mass
        cannot be established at creation on terms the setter would refuse.

        Args:
            mass: The caller-supplied value. Anything ``float()`` accepts is
                coerced (so a NumPy scalar passes); ``bool`` is rejected
                explicitly since ``True`` would act as a silent 1 kg body.
            method: Public method name, used to prefix the error message.
            param: Parameter name to quote in the message.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is usable.
        """
        # is_boolean, not isinstance(mass, bool): numpy.bool_ is not a bool
        # subclass, so the narrower check refused a hand-typed True and admitted
        # the np.True_ a comparison produces - a 1 kg body under success.
        if is_boolean(mass):
            return _boolean_world_error(method, param, mass)
        try:
            value = float(mass)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "content": [{"text": f"{method}: '{param}' must be a positive number, got {mass!r}"}],
            }
        if not math.isfinite(value) or value <= 0:
            return {
                "status": "error",
                "content": [{"text": f"{method}: '{param}' must be a finite number > 0, got {value}"}],
            }
        return None

    @staticmethod
    def _normalize_gravity(
        gravity: Any, method: str, param: str = "gravity"
    ) -> tuple[list[float] | None, dict[str, Any] | None]:
        """Coerce a gravity argument to three finite floats, or explain why not.

        Gravity reaches the engine as a raw vector assignment (MuJoCo:
        ``model.opt.gravity[:] = ...``), so a mis-shaped value either raises a
        binding-level ``TypeError`` naming the physics library's internals
        instead of the parameter the caller got wrong, or - for values that
        happen to be assignable - lands in the world while the result echoes
        the caller's input as if it had been applied. Callers therefore
        normalize through this helper and store the returned components, so
        what the result reports is what the engine received.

        Exactly one element of the returned tuple is non-``None``.

        Args:
            gravity: A 3-element ``[x, y, z]`` sequence, or a real scalar taken
                as the z-component. A boolean is refused in either form -
                ``float(True)`` is ``1.0``, so ``True`` configured a +1 m/s^2
                gravity pointing *up* and reported success (``[0, 0, z]``,
                matching
                :meth:`~strands_robots.simulation.mujoco.simulation.MuJoCoSimEngine.set_gravity`).
            method: Public method name, used to prefix the error message.
            param: Parameter name to quote in the message.

        Returns:
            ``(components, None)`` with three finite floats, or
            ``(None, error_dict)`` describing what is wrong with the value.
        """
        # Refuse a boolean before either path coerces it. bool is an int
        # subclass, so it satisfies numbers.Real and float(True) is 1.0 - a
        # scalar True became a +1 m/s^2 gravity pointing *up*, reported as
        # success. numpy.bool_ is not numbers.Real, so it missed the scalar
        # branch and fell through to len(), which raised "len() of unsized
        # object" and surfaced as a component-count complaint that described
        # neither the value nor the reason.
        if is_boolean(gravity):
            return None, _boolean_world_error(method, param, gravity)
        # Accept any real scalar (numbers.Real) as a z-only gravity so a value
        # computed as a NumPy scalar (np.float32 / np.int64) is treated like a
        # plain float. A NumPy array is not numbers.Real, so it still takes the
        # vector path below.
        if isinstance(gravity, numbers.Real):
            components = [0.0, 0.0, float(gravity)]
        else:
            try:
                vector = cast("Sequence[Any]", gravity)
                if len(vector) != 3:
                    return None, {
                        "status": "error",
                        "content": [
                            {"text": f"{method}: '{param}' must be a 3-element list [x,y,z], got {len(vector)}"}
                        ],
                    }
                # Per component too: the vector path coerces with float(), so a
                # single True among three reals is otherwise a 1 m/s^2 axis.
                for component in vector:
                    if is_boolean(component):
                        return None, _boolean_world_error(method, param, gravity)
                components = [float(g) for g in vector]
            except (TypeError, ValueError) as e:
                return None, {
                    "status": "error",
                    "content": [{"text": f"{method}: '{param}' must be a 3-element list of numbers ({e})"}],
                }
        if not all(math.isfinite(g) for g in components):
            return None, {
                "status": "error",
                "content": [{"text": f"{method}: all components must be finite, got {components}"}],
            }
        return components, None

    @staticmethod
    def _validate_duration(duration: Any, method: str) -> dict[str, Any] | None:
        """Reject a rollout ``duration`` that cannot produce a single control step.

        ``duration`` is the default horizon knob: when no ``n_steps`` /
        ``max_steps`` is given, the rollout length is ``int(duration *
        control_frequency)`` control steps. A value ``<= 0`` yields zero steps,
        which used to be reported as ``status="success"`` for a rollout that
        never queried the policy and never stepped physics - and, when a
        ``video`` was requested, wrote no MP4 while still claiming success. A
        non-finite value never reached that arithmetic intact either: ``nan``
        surfaced as a bare ``ValueError`` ("cannot convert float NaN to
        integer") naming a library internal, and ``inf`` as an
        ``OverflowError``. Validating at the public entry point - before any
        policy is created or a background thread is submitted - turns all of
        these into an actionable caller error.

        The accepted domain is :func:`~strands_robots.utils.positive_finite_number_error`,
        shared with :meth:`_validate_positive_frequency` (the other knob in the
        same ``duration * control_frequency`` product) so the two cannot
        diverge: any finite positive real scalar, including a NumPy scalar such
        as ``np.float32(2.5)``; ``bool`` is rejected explicitly (an ``int``
        subclass, ``True`` would act as a silent 1 second) and ``nan``/``inf``
        are rejected before the ``<= 0`` comparison so a ``nan`` - which is
        never ``<= 0`` - cannot slip through.

        Args:
            duration: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid.
        """
        error = positive_finite_number_error(duration, "duration", method)
        if error:
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_rtc_inference_timeout(rtc_inference_timeout_s: Any, method: str) -> dict[str, Any] | None:
        """Reject an async-RTC prefetch deadline the runner cannot wait out.

        ``rtc_inference_timeout_s`` is the async-RTC chunk pipeline's hard
        per-chunk deadline: the seam swap does
        ``future.result(timeout=rtc_inference_timeout_s)`` and converts a
        ``concurrent.futures.TimeoutError`` into "policy inference is stuck.
        Raise the timeout or check the policy/server." That message is only true
        of a deadline a healthy inference could have met, and every value
        outside the accepted domain makes it false:

        * ``0`` / a negative value / ``nan`` make ``Future.result`` raise
          immediately - ``nan`` because every comparison against it is false, so
          the deadline is never considered met - and the rollout is reported as
          ``status="error"`` blaming a policy that answered on time. The value
          the caller supplied is quoted back in a sentence accusing the model.
        * ``inf`` cannot be honored either: it reaches ``time_t`` arithmetic and
          raises ``OverflowError: timestamp out of range for platform time_t``,
          which names neither the parameter nor the method. ``None`` - not
          ``inf`` - is this parameter's documented "wait without a deadline"
          spelling, so refusing ``inf`` costs no capability.
        * ``True`` is an ``int`` subclass and acts as a silent 1-second budget.
        * A string or a list reaches the same comparison and leaks
          ``TypeError: '>' not supported between instances of 'str' and 'int'``.

        The accepted domain is
        :func:`~strands_robots.utils.positive_finite_number_error` - the same
        rule as :meth:`_validate_duration` and
        :meth:`_validate_positive_frequency`, the other wall-clock knobs of a
        rollout - plus ``None``, which this parameter documents as "no deadline"
        and which is its default.

        Validated unconditionally rather than only when the async path is
        active. Whether that path runs is not knowable here: ``async_rtc=None``
        (the default) auto-resolves from the policy's own chunk-emitting shape
        one layer down, after the policy has been constructed. Gating the check
        on the flag would therefore leave the dominant path - every
        chunk-emitting VLA - unguarded, and give one value two answers depending
        on a resolution the caller cannot see. Checking here instead costs a bad
        deadline no weight download and no frame.

        Args:
            rtc_inference_timeout_s: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid (including when it is ``None``).
        """
        if rtc_inference_timeout_s is None:
            return None
        error = positive_finite_number_error(rtc_inference_timeout_s, "rtc_inference_timeout_s", method)
        if error:
            return {"status": "error", "content": [{"text": error}]}
        return None

    @staticmethod
    def _validate_onframe_failure_limit(max_onframe_failures: Any, method: str) -> dict[str, Any] | None:
        """Reject an ``on_frame`` failure tolerance the watchdog cannot count against.

        ``max_onframe_failures`` is the consecutive-failure ceiling that stops a
        broken ``on_frame`` hook from producing an empty capture behind a
        successful-looking rollout (GH #117). The runner counts failures into a
        plain ``int`` and compares ``consecutive_onframe_failures >= limit``, so a
        value outside the accepted domain does not merely mis-size the tolerance -
        it silences the mechanism whose own abort text reads "aborting episode to
        avoid silent dataset corruption":

        * ``nan`` and ``inf`` make that comparison false for every counter value,
          so the abort never fires. Measured on a 100-step rollout whose hook
          raises on every step: 100 of 100 frames lost and
          ``status="success"``. Both values also break the per-failure warning
          that would otherwise report the hook - it interpolates the limit with
          ``%d``, and ``"%d" % nan`` raises ``ValueError`` while ``"%d" % inf``
          raises ``OverflowError``, so ``logging`` emits its own error instead of
          the warning and the operator is told nothing at all.
        * ``0`` is a duplicate spelling of ``1`` carrying a false message. The
          counter is incremented before the comparison, so a limit of ``1``
          already aborts on the first failure; ``0`` aborts on the same failure
          and reports "failed 0 times in a row" when one failure occurred.
          Refusing it costs no capability, and ``-5`` is the same abort with a
          message that names a negative count.
        * ``2.7`` tolerates two failures and aborts on the third while reporting
          "failed 2.7 times in a row" - a tolerance the caller never asked for.
          ``True`` is an ``int`` subclass and reports "failed True times in a row".
        * A string or a list reaches the same comparison and leaks
          ``TypeError: '>=' not supported between instances of 'int' and 'str'``
          from inside the hook's own exception handler - and only once the hook
          first fails, so the value is accepted and inert until then.

        The accepted domain is :meth:`_validate_positive_int`
        (:func:`~strands_robots.utils.positive_count_error`) - the same rule this
        method already applies to ``n_steps``, ``max_steps`` and ``n_episodes``,
        the other step counts of the same signature - plus ``None``, which this
        parameter documents as "use the runner's own limit" and which is its
        default.

        Args:
            max_onframe_failures: The caller-supplied value to validate.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending parameter, or ``None`` when the
            value is valid (including when it is ``None``).
        """
        if max_onframe_failures is None:
            return None
        return SimEngine._validate_positive_int(max_onframe_failures, "max_onframe_failures", method)

    def _validate_recording_rate(self, control_frequency: float, method: str) -> dict[str, Any] | None:
        """Reject a rollout whose rate the active dataset recording cannot describe.

        Instance method (not a ``@staticmethod`` like the other numeric guards)
        because the value it compares against lives on the engine: the rate is
        only knowable once a recording is open, which happens one call earlier
        in ``start_recording``. Backends that cannot record inherit
        :meth:`_is_recording` returning ``False`` and are unaffected, so this
        one call site per rollout entry point covers every backend without each
        needing its own copy.

        Args:
            control_frequency: Rate the rollout will capture frames at. Validate
                it with :meth:`_validate_positive_frequency` first, so a
                non-finite value is reported as the parameter error it is
                rather than as a rate disagreement.
            method: Public method name, used to prefix the error message.

        Returns:
            A structured ``{"status": "error", ...}`` dict, or ``None`` when no
            recording is active or the rates agree. See
            :func:`~strands_robots.simulation.recording.dataset_rate_mismatch_reason`
            for the contract and why a mismatch is refused rather than warned.
        """
        if not self._is_recording():
            return None
        from strands_robots.simulation.recording import dataset_rate_mismatch_error

        return dataset_rate_mismatch_error(method, self._active_recorder(), control_frequency)

    def _validate_recording_start_rate(self, fps: Any, method: str) -> dict[str, Any] | None:
        """Reject opening a recording at a rate an in-flight rollout does not capture at.

        The inverse ordering of :meth:`_validate_recording_rate`, and the same
        disagreement: that guard runs when a rollout starts against an open
        recording, this one when a recording is opened against a rollout that is
        already running. ``start_policy`` makes the second ordering reachable by
        design - it submits the rollout and returns while it continues - and the
        two library defaults collide (``fps=30`` against
        ``control_frequency=50.0``), so the plain sequence produced a 1.667x
        mislabelled episode with every call reporting success.

        Instance method for the same reason as :meth:`_validate_recording_rate`:
        the value it compares against lives on the engine. Backends with no
        asynchronous rollout inherit :meth:`_active_rollout_rates` returning an
        empty mapping and are unaffected, so one call site per backend's
        ``start_recording`` covers every backend without each needing its own
        copy of the rule.

        Args:
            fps: Caller-supplied dataset frame rate. Validate it with
                :func:`~strands_robots.simulation.recording.dataset_recording_option_error`
                first, so a value no dataset can be written at is reported as
                the parameter error it is rather than as a rate disagreement.
            method: Public method name, used to prefix the error message.

        Returns:
            A structured ``{"status": "error", ...}`` dict, or ``None`` when no
            rollout is in flight or every one of them already captures at
            ``fps``. See
            :func:`~strands_robots.simulation.recording.rollout_rate_mismatch_reason`
            for the contract and the measured consequence.
        """
        rates = self._active_rollout_rates()
        if not rates:
            return None
        from strands_robots.simulation.recording import rollout_rate_mismatch_error

        return rollout_rate_mismatch_error(method, fps, rates)

    @staticmethod
    def _validate_video_config(video: Any, method: str) -> dict[str, Any] | None:
        """Reject a ``video`` recording config the rollout cannot honor.

        ``video`` is a free-form dict, so a mistyped key has no signature to
        bounce off and used to be dropped silently: a rollout asked to record
        at ``{"filename": ...}`` reported ``status="success"`` having written
        no MP4, and one asked for ``{"resolution": [320, 240]}`` recorded at
        the default 640x480. Checking it at the public entry point (before any
        policy is created or a background thread is submitted) turns both into
        an actionable error. Returns a structured ``{"status": "error", ...}``
        dict to surface, or ``None`` when the config is valid.

        Args:
            video: The caller-supplied ``video`` dict, or ``None``.
            method: Public method name, used to prefix the error message.

        Returns:
            An error dict naming the offending key, or ``None``.
        """
        video_error = VideoConfig.validation_error(video)
        if video_error is None:
            return None
        return {"status": "error", "content": [{"text": f"{method}: {video_error}"}]}

    @staticmethod
    def _validate_policy_mapping(value: Any, param: str, method: str) -> dict[str, Any] | None:
        """Reject a ``policy_config`` / ``policy_kwargs`` that cannot be splatted.

        Both parameters are opaque keyword bags reaching their consumer through
        ``**``: ``policy_config`` lands in ``create_policy``, ``policy_kwargs``
        in ``policy.get_actions``. A value of the wrong shape used to surface as
        a bare ``TypeError`` from CPython's call machinery, naming a library
        internal instead of the parameter the caller got wrong - and on the
        background-thread path (``start_policy``) it was raised inside the
        future, so the caller was told the policy had started when no rollout
        ever ran. Checking at the public entry point, before any policy is
        created or a thread is submitted, turns both into an actionable error.

        Args:
            value: The caller-supplied value, or ``None``.
            param: ``"policy_config"`` or ``"policy_kwargs"``.
            method: Public method name, used to prefix the error message.

        Returns:
            A structured ``{"status": "error", ...}`` dict to surface, or
            ``None`` when the value is valid.
        """
        from strands_robots.policies import policy_mapping_error

        message = policy_mapping_error(value, param)
        if message is None:
            return None
        return {"status": "error", "content": [{"text": f"{method}: {message}"}]}

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
        stop_when: dict[str, Any] | Callable[[SimEngine], bool] | None = None,
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
            duration: Wall-clock seconds to run. Used only when no ``n_steps``
                / ``max_steps`` is given (the step count wins and ``duration``
                is recomputed from it). Must be a finite positive number; a
                non-positive, non-finite, non-numeric, or bool value is
                reported as a structured caller error rather than running a
                zero-step rollout that reports success.
            control_frequency: Target Hz for policy queries. Must be a
                positive number; a non-positive, non-numeric, or bool value
                is reported as a structured caller error.
            control_substeps: Explicit physics steps to integrate per applied
                action, overriding the ``control_frequency``-derived value.
                Must be a positive integer; ``0``, a negative value, a float,
                or a bool is reported as a structured caller error rather than
                collapsing to a single physics step (which under-integrates
                each control period so the arm barely moves while the rollout
                still reports success). ``None`` (default) derives it from
                ``control_frequency`` and the backend's physics timestep.
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
                Any other key - and any ``fps``/``width``/``height`` that is
                not a positive whole number - is reported as a caller error
                naming the offending key, rather than being dropped (which
                used to turn a mistyped ``path`` into a "successful" rollout
                with no MP4, and a mistyped size into a default-resolution
                recording).
                For extension points beyond video (custom telemetry,
                dataset recording), backends plug into
                ``PolicyRunner.run``'s ``on_frame`` hook via
                :meth:`_make_run_policy_hook`.
            policy_object: Already-constructed
                :class:`~strands_robots.policies.Policy` to drive the rollout,
                bypassing ``create_policy`` entirely (``policy_provider`` /
                ``policy_config`` are then unused). Reuse one instance across
                calls so the checkpoint is not reloaded per rollout - the
                ``policy_load_cache_hit`` field below reports when a caller
                rebuilt it instead. ``None`` (default) builds the policy from
                ``policy_provider`` / ``policy_config``.
            n_steps: Exact control-step horizon. When given it REPLACES
                ``duration``, which is recomputed as
                ``n_steps / control_frequency``, and it bypasses the lossy
                ``int(duration * control_frequency)`` conversion so the rollout
                executes exactly this many control steps. Must be a positive
                integer; ``0``, a negative value, a bool, or a fractional or
                non-numeric value is reported as a structured caller error
                rather than truncated to a horizon the caller never asked for.
                ``None`` (default) falls back to ``max_steps``, then to
                ``duration``.
            max_steps: Legacy alias for ``n_steps``, kept for callers written
                against the older name. Consulted only when ``n_steps`` is
                ``None`` - ``n_steps`` wins when both are passed - and refused
                under its own name on the same domain.
            max_onframe_failures: Maximum *consecutive* ``on_frame``-hook
                exceptions tolerated before the rollout aborts the episode. That
                hook is where a backend attaches dataset recording and video
                capture, so a broken recorder otherwise fills an empty dataset
                behind a successful-looking rollout. ``None`` (default) uses the
                runner's own limit (currently ``5``); non-consecutive failures
                reset the counter. Must otherwise be a positive integer - the
                same domain as ``n_steps`` and ``n_episodes`` above, since the
                runner compares it against a plain integer counter. ``nan`` and
                ``inf`` make that comparison false forever and so disable the
                abort entirely; ``0`` aborts on the first failure exactly as
                ``1`` does while reporting a count of zero. Both are reported as
                a structured caller error rather than silencing the watchdog -
                see
                :meth:`~strands_robots.simulation.base.SimEngine._validate_onframe_failure_limit`.
                Forwarded verbatim to
                :meth:`~strands_robots.simulation.policy_runner.PolicyRunner.run`.
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
            stop_when: Optional semantic early-return condition: end the
                rollout as soon as the WORLD reaches a state, not only when
                the step budget runs out - which turns a monolithic rollout
                into a retryable primitive an agent can invoke -> inspect ->
                re-invoke. A predicate-DSL clause in the same schema as a
                benchmark spec's ``success`` clause: a single call
                ``{"predicate": "grasped", "body": "cube", "gripper_prefix":
                "so100"}`` or an ``{"all": [...]}`` / ``{"any": [...]}`` group
                of bool predicate calls. Compiled via
                :func:`~strands_robots.simulation.benchmark_spec.compile_stop_when`
                against the closed predicate registry (never ``eval`` /
                ``exec``; an unknown predicate name is rejected up front with
                the valid list), and the clause's referenced body/joint names
                are probed against the LIVE scene before the rollout starts -
                a typo'd name (or a backend without body lookups) is an
                up-front structured error instead of a clause that silently
                never fires and burns the whole budget. The compiled clause is
                evaluated against the SIM after every
                applied action - matching the benchmark semantics, not the
                observation dict - on both the synchronous and async-RTC
                paths, so the stop lands within one control step of the
                condition holding. Composes with an active recording session:
                frames are captured up to the stop, so a recorded episode's
                frame count equals the result's ``steps_used``. Programmatic
                callers may pass a callable ``(sim) -> bool`` instead of a
                dict (the tool surface accepts dicts only). ``None`` (default)
                keeps the pure step-budget horizon. The result json reports
                why the rollout ended via ``stopped_reason``.

        Returns:
            The standard agent-tool envelope
            ``{"status": "success"|"error", "content": [{"text": ...},
            {"json": {...}}]}``. The ``json`` block is the machine-readable
            rollout report; ``text`` carries the same facts for humans.

            Read the json block by SCANNING ``content`` for the first block
            with a ``"json"`` key, never by a fixed index::

                report = next(b["json"] for b in result["content"] if "json" in b)

            An early caller-error return (a rejected ``duration``, an unknown
            robot) carries a ``text`` block ONLY, so a hardcoded
            ``content[1]`` raises ``IndexError`` on exactly the results a
            caller most needs to read.

            IMPORTANT - ``status`` is not the rollout verdict. It reports
            whether the CALL was accepted and the loop ran; it does not say the
            robot did anything useful. A rollout that drove only a SUBSET of
            the robot's actuators is deliberately ``success`` (it is
            operational), so ``status`` alone cannot see it: a policy driving 1
            of a Panda's 8 actuators returns ``status="success"`` with
            ``action_errors=0`` and ``partial_action_failure_rate=0.875``. Gate
            on ``partial_action_failure_rate`` and the binding-degradation
            flags below to decide whether a rollout is worth anything. A TOTAL
            failure - no emitted key resolving to any actuator - is reported as
            ``status="error"``.

            Fields in the json block:

            Identity: ``robot_name``, ``policy`` (the driving policy's class
            name), ``instruction``.

            Horizon: ``n_steps`` (control steps executed), ``steps_used``
            (alias of ``n_steps`` under the retry-loop name), ``elapsed_s``,
            ``sim_time_s`` (when the backend reports sim time),
            ``stopped_early``, ``stopped_reason`` (``"predicate"`` - the
            ``stop_when`` condition fired; ``"budget"`` - the step/duration
            horizon was exhausted; ``"cancelled"`` - a cooperative stop, e.g.
            ``stop_policy``; ``"error"`` on error results - so an agent
            deciding whether to retry knows WHY the rollout ended).

            Action health: ``action_errors`` (steps where at least one emitted
            key did not resolve), ``action_resolution_rate`` (an
            ``{actuator_name: fraction_of_steps_driven}`` map, so a joint stuck
            at ``0.0`` names the actuator the policy never drove) and
            ``partial_action_failure_rate`` (the mean fraction of the robot's
            DOF never driven; ``0.0`` == every actuator moved every step,
            ``~0.83`` == only 1 of 6 actuators ever moved).

            Video: ``video_path`` (``None`` when no MP4 was written) and
            ``video_frames``.

            Episodes: ``n_episodes_requested``, ``n_episodes_completed``,
            ``episodes_saved`` and ``dataset_episode_indices`` (the dataset
            episode indices this call flushed, empty without a recording).

            Policy binding: ``positional_fallback_used``,
            ``generic_state_keys_used`` and ``missing_state_keys_used``. True
            means the driving policy could not bind the observation to the
            model's inputs by name and silently fell back (a camera routed to a
            model image slot positionally, or ``observation.state`` composed
            from the observation's own scalar keys because none of
            ``robot_state_keys`` matched). A True flag on an otherwise
            ``success`` run is the signature of a robot moving on meaningless
            inputs.

            Policy load: ``policy_load_time_s``, ``policy_load_cache_hit``
            (``False`` on episode 2+ of a loop is a smell that the caller
            rebuilt the policy instead of reusing ``policy_object=``) and
            ``policy_resident_rss_mb``.

            Async-RTC telemetry, so latency masking is provable from the
            payload instead of from logs: ``rtc_async_enabled``,
            ``rtc_chunks_acquired``, ``rtc_prefetch_hits``,
            ``rtc_prefetch_blocks``, ``rtc_avg_inference_ms`` and
            ``rtc_max_inference_ms``.

            Fail-fast: if EVERY action step in the opening probe window drives
            zero actuators - none of the policy's emitted keys resolve to any of
            the robot's actuators - the rollout can never move the robot, so it
            returns ``status="error"`` at the probe boundary instead of running
            the full episode (and every remaining model inference call +
            recording write). The error enumerates the unresolved keys and the
            robot's valid actuator names. A PARTIAL failure runs to completion,
            surfaced via ``partial_action_failure_rate``.
        """
        from strands_robots.policies import create_policy

        robot_name = self._resolve_single_robot(robot_name)

        if err := self._validate_positive_frequency(control_frequency, "run_policy"):
            return err
        # Coerce to a plain Python float now the value is validated: a NumPy
        # scalar (accepted above via numbers.Real) flows into 1 / control_frequency
        # and time.sleep(...) downstream, and time.sleep rejects a numpy.float32
        # with a bare "cannot be interpreted as an integer" TypeError.
        control_frequency = float(control_frequency)

        # The seed is the caller's reproducibility contract; an unusable one
        # cannot be applied at all, and reached NumPy as a bare cast TypeError
        # out of a method documented to return this envelope. Refused here,
        # before a policy is built or a frame is written.
        if err := self._validate_seed(seed, "run_policy"):
            return err

        # accept n_steps (or legacy max_steps) as an alternate horizon
        # specification. duration = n_steps / control_frequency. If both
        # are passed, n_steps wins (primary per DoD).
        duration, n_steps, horizon_error = self._resolve_horizon(n_steps, max_steps, control_frequency, duration)
        if horizon_error is not None:
            return horizon_error

        # ``duration`` only sets the horizon when no step count was given - with
        # an ``n_steps`` the resolution above recomputes it - so validate the
        # value the rollout will actually run on, and only then.
        if n_steps is None:
            if err := self._validate_duration(duration, "run_policy"):
                return err

        if err := self._validate_positive_int(n_episodes, "n_episodes", "run_policy"):
            return err

        if err := self._validate_video_config(video, "run_policy"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "run_policy"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "run_policy"):
            return err
        if err := self._validate_action_horizon(action_horizon, "run_policy"):
            return err
        if err := self._validate_control_substeps(control_substeps, "run_policy"):
            return err
        if err := self._validate_rtc_inference_timeout(rtc_inference_timeout_s, "run_policy"):
            return err
        if err := self._validate_onframe_failure_limit(max_onframe_failures, "run_policy"):
            return err
        # Both rates are known only here: the dataset rate was fixed by
        # start_recording one call earlier. Checked before any policy is
        # built so a rate disagreement costs no weight download and no frame.
        if err := self._validate_recording_rate(control_frequency, "run_policy"):
            return err

        # Compile the stop_when early-return clause BEFORE any policy is
        # created (an unknown predicate name or bad kwargs is a caller error,
        # not a mid-rollout crash after an expensive weight download). The
        # tool surface only ever passes predicate-DSL dicts, resolved through
        # the closed registry - never eval/exec; programmatic callers may pass
        # a callable directly, mirroring PolicyRunner.evaluate's success_fn.
        stop_when_fn: Callable[[SimEngine], bool] | None = None
        if stop_when is not None:
            if callable(stop_when):
                stop_when_fn = stop_when
            else:
                from strands_robots.simulation.benchmark_spec import compile_stop_when

                try:
                    stop_when_fn = compile_stop_when(stop_when)
                except ValueError as e:
                    return {
                        "status": "error",
                        "content": [
                            {"text": f"run_policy: {e}"},
                            {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}},
                        ],
                    }

        if robot_name not in self.list_robots():
            return {
                "status": "error",
                "content": [{"text": self._unknown_robot_msg(robot_name)}],
            }

        # Probe the clause's referenced bodies/joints against the LIVE scene.
        # compile_stop_when validates the predicate NAMES but cannot see the
        # scene: a typo'd body would compile clean, degrade to a constant
        # False at evaluation time (predicates never raise), and burn the
        # whole step budget reporting stopped_reason="budget" -
        # indistinguishable from an honest miss. Probing here (dict clauses
        # only - a programmatic callable is opaque) turns that silent
        # never-fires into an up-front structured error, including on
        # backends whose predicates cannot resolve bodies at all.
        if stop_when_fn is not None and isinstance(stop_when, dict):
            probe_err = self._stop_when_unresolved_error(stop_when)
            if probe_err is not None:
                return probe_err

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
        # set_robot_state_keys + sim-context binding are best-effort policy
        # configuration: a raising robot_action_keys (a backend quirk, a world
        # torn down mid-setup) must not crash the whole rollout. A genuine
        # wrong-embodiment mismatch is surfaced far more actionably downstream
        # by PolicyRunner's fail-fast probe ("the robot has not moved"). This
        # matches the guarded binding in MujocoSimulation.run_policy's
        # multi-robot path.
        try:
            policy.set_robot_state_keys(self.robot_action_keys(robot_name))
            self.bind_policy_sim_context(policy, robot_name)
        except Exception as exc:  # noqa: BLE001 - non-fatal policy configuration
            logger.debug("policy binding for %r failed: %s", robot_name, exc)

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
                    n_steps=n_steps,
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
                    stop_when=stop_when_fn,
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
                n_steps=n_steps,
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
                stop_when=stop_when_fn,
            )
        finally:
            if controller_cleanup is not None:
                controller_cleanup()

    def run_multi_policy(
        self,
        policies: dict[str, Policy],
        instructions: dict[str, str] | str = "",
        duration: float = 10.0,
        control_frequency: float = 50.0,
        action_horizon: int | dict[str, int] = 8,
        n_steps: int | None = None,
        max_steps: int | None = None,
    ) -> dict[str, Any]:
        """Drive MULTIPLE robots, each with its own policy, in ONE synchronized loop.

        The backend-agnostic contract for concurrent multi-robot rollout
        (e.g. two arms doing a handover, or a bimanual setup). A backend that
        implements it must honour every clause below - they are what
        distinguishes this driver from launching one :meth:`start_policy`
        thread per robot, which steps physics per robot and interleaves
        single-robot recording frames:

        - **Per-robot policies**: ``policies`` maps each driven robot to its
          own :class:`~strands_robots.policies.Policy`. Every key must name a
          robot in the scene; ``policies`` order defines the merged
          state/action column order.
        - **Per-robot instructions**: ``instructions`` is either one string
          applied to all robots or a ``{robot_name: instruction}`` mapping.
          A mapping key naming no driven robot is rejected rather than
          silently dropped; a robot omitted from the mapping gets an empty
          instruction (see :meth:`_normalize_multi_policy_instructions`).
        - **Per-robot action_horizon**: ``action_horizon`` is either one int
          applied to all robots or a ``{robot_name: horizon}`` mapping. Every
          horizon must be a positive integer, and the effective per-robot
          chunk length is resolved through
          :func:`~strands_robots.policies.base.resolve_chunk_length` exactly
          as :meth:`run_policy` resolves its own (see
          :meth:`_normalize_multi_policy_horizons`).
        - **Shared control_frequency**: one target Hz for every robot's
          policy queries, so the robots stay phase-aligned.
        - **Lockstep physics**: each loop iteration applies EVERY robot's
          control, then steps physics ONCE - regardless of each robot's
          individual re-query cadence.
        - **One merged recording frame per timestep**: when a dataset
          recording is active, each timestep records a single frame carrying
          ALL robots' prefixed state/action (``alice__shoulder_pan`` ...)
          plus all camera images - never one interleaved frame per robot.

        The step horizon follows :meth:`run_policy`'s resolution: ``n_steps``
        (then its legacy alias ``max_steps``) overrides ``duration``, via
        :meth:`_resolve_horizon` on the shared positive-count domain.

        This base implementation is a documented refusal, not a fallback: a
        backend that has no synchronized multi-robot loop must say so rather
        than silently driving robots one at a time (which would interleave
        frames and break the merged-frame contract above). The MuJoCo and
        Isaac backends override it with full implementations; backends that
        do not yet (Newton) inherit this structured error.

        Args:
            policies: Mapping ``{robot_name: Policy}`` of the robots to drive.
            instructions: Single instruction string for all robots, or a
                ``{robot_name: instruction}`` mapping (see contract above).
            duration: Episode length in seconds (steps = duration x freq).
                Used only when no ``n_steps`` / ``max_steps`` is given. Must
                be a finite positive number.
            control_frequency: Target Hz for policy queries / physics. Must
                be a positive number.
            action_horizon: Actions consumed from each policy's chunk before
                re-querying it, as one int or a per-robot mapping (see
                contract above).
            n_steps: Exact step horizon (overrides ``duration`` when set).
            max_steps: Legacy alias for ``n_steps``.

        Returns:
            A structured ``{"status": "error", ...}`` dict naming this
            backend class and stating that it does not implement synchronized
            multi-robot rollout. Implementing backends return the standard
            status dict with per-robot step counts.
        """
        return {
            "status": "error",
            "content": [
                {
                    "text": f"run_multi_policy: {type(self).__name__} does not implement synchronized "
                    "multi-robot rollout (per-robot policies driven in one lockstep physics loop with "
                    "one merged recording frame per timestep). Use the MuJoCo backend, or drive robots "
                    "individually with run_policy / start_policy (frames are then interleaved per robot, "
                    "not merged)."
                }
            ],
        }

    @staticmethod
    def _validate_multi_policies(policies: Mapping[str, Any], method: str) -> dict[str, Any] | None:
        """Reject an empty ``policies`` mapping at a multi-robot entry point.

        ``policies`` names the robots a synchronized multi-robot driver will
        drive, so an empty mapping is a caller error: a loop over zero robots
        would run zero steps and still report ``status="success"`` (the same
        degenerate-success shape :meth:`_validate_positive_int` exists to
        refuse). Shared by every backend's ``run_multi_policy`` so the refusal
        text is identical everywhere.

        Args:
            policies: The caller-supplied ``{robot_name: Policy}`` mapping.
            method: Public method name, used to prefix the error message.

        Returns:
            A structured ``{"status": "error", ...}`` dict, or ``None`` when
            at least one robot is named.
        """
        if not policies:
            return {"status": "error", "content": [{"text": f"{method}: 'policies' is empty."}]}
        return None

    @staticmethod
    def _normalize_multi_policy_instructions(
        policies: Mapping[str, Any],
        instructions: Mapping[str, str] | str,
        method: str,
        warn_logger: logging.Logger | None = None,
    ) -> tuple[dict[str, str] | None, dict[str, Any] | None]:
        """Normalize ``instructions`` to a complete per-robot mapping.

        A single string broadcasts to every driven robot. A mapping is an
        override layer keyed by robot name: a key that names no robot in this
        call is a caller error (read with ``.get(r, "")`` it was silently
        discarded, so a typo'd robot name ran the whole episode on an empty
        instruction and still reported success - see
        :meth:`_validate_per_robot_mapping`), while a robot omitted from the
        mapping legitimately gets an empty instruction. A value that is
        neither a string nor a mapping is refused up front rather than
        reaching ``.get()`` and surfacing as a bare ``AttributeError`` past
        the tool-envelope contract.

        LeRobot stores ONE task string per frame, so when the normalized
        mapping carries distinct non-empty instructions (e.g. ``{alice:
        'pour', bob: 'catch'}``) only the first robot's task is recorded - a
        downstream pipeline that splits by task string would mis-attribute
        the other robots' frames. That is surfaced here as a
        ``logger.warning`` (a known limitation, not lost silently).

        Args:
            policies: ``{robot_name: Policy}`` mapping naming the driven
                robots (the authoritative key set).
            instructions: Caller-supplied single string or per-robot mapping.
            method: Public method name, used to prefix error messages.
            warn_logger: Logger for the distinct-instructions warning, so the
                warning is attributed to the calling backend's module rather
                than to this one. Defaults to this module's logger.

        Returns:
            ``(instr_map, None)`` with one entry per driven robot, or
            ``(None, error_dict)``.
        """
        if isinstance(instructions, str):
            instr_map = {r: instructions for r in policies}
        elif isinstance(instructions, Mapping):
            if err := SimEngine._validate_per_robot_mapping(instructions, policies, "instructions", method):
                return None, err
            instr_map = {r: instructions.get(r, "") for r in policies}
        else:
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": f"{method}: 'instructions' must be a string applied to all robots or a "
                        f"{{robot_name: instruction}} mapping, got {type(instructions).__name__}."
                    }
                ],
            }

        distinct_tasks = {t for t in instr_map.values() if t}
        if len(distinct_tasks) > 1:
            (warn_logger or logger).warning(
                "%s: %d distinct per-robot instructions supplied (%s) but "
                "LeRobot records one task per frame; only '%s' (robot '%s') will be stored. "
                "Per-robot task columns are not yet supported.",
                method,
                len(distinct_tasks),
                sorted(distinct_tasks),
                instr_map[next(iter(policies))],
                next(iter(policies)),
            )
        return instr_map, None

    @staticmethod
    def _normalize_multi_policy_horizons(
        policies: Mapping[str, Any],
        action_horizon: int | Mapping[str, int],
        method: str,
        default_horizon: int = 8,
    ) -> tuple[dict[str, int] | None, dict[str, Any] | None]:
        """Normalize ``action_horizon`` to a complete per-robot mapping.

        A single int broadcasts to every driven robot; a ``{robot_name:
        horizon}`` mapping overrides per robot, with a key naming no driven
        robot refused (see :meth:`_validate_per_robot_mapping`) and a robot
        omitted from the mapping keeping ``default_horizon``. Every horizon
        is validated through :meth:`_validate_action_horizon` - the same
        guard ``run_policy`` / ``start_policy`` / ``eval_policy`` enforce -
        rather than coerced: a ``max(1, int(...))`` clamp used to turn ``0``
        / ``-5`` into a silent 1-action horizon, truncate ``2.7``, and let
        ``nan`` / ``None`` / ``"x"`` reach ``int()`` as a bare ``ValueError``
        / ``TypeError`` past the tool-envelope contract.

        Args:
            policies: ``{robot_name: Policy}`` mapping naming the driven
                robots (the authoritative key set).
            action_horizon: Caller-supplied single int or per-robot mapping.
            method: Public method name, used to prefix error messages.
            default_horizon: Horizon for robots omitted from a mapping.

        Returns:
            ``(horizon_map, None)`` with one entry per driven robot, or
            ``(None, error_dict)``.
        """
        if isinstance(action_horizon, Mapping):
            if err := SimEngine._validate_per_robot_mapping(action_horizon, policies, "action_horizon", method):
                return None, err
            for rname, value in action_horizon.items():
                if err := SimEngine._validate_action_horizon(value, method, f"action_horizon[{rname!r}]"):
                    return None, err
            # A robot omitted from the mapping keeps the signature default.
            return {r: int(action_horizon.get(r, default_horizon)) for r in policies}, None
        if err := SimEngine._validate_action_horizon(action_horizon, method):
            return None, err
        return {r: int(action_horizon) for r in policies}, None

    def _stop_when_unresolved_error(self, stop_when: dict[str, Any]) -> dict[str, Any] | None:
        """Structured error if a ``stop_when`` clause references unresolvable entities.

        Probes every body/joint name in the clause through the SAME lookup
        path the predicates use at evaluation time
        (:func:`~strands_robots.simulation.predicates.can_resolve_body` /
        :func:`~strands_robots.simulation.predicates.can_resolve_joint`,
        including the LIBERO ``<name>_main`` fallback), against the live
        scene, once, before the rollout starts. Returns ``None`` when every
        referenced entity resolves. Bodies added to the scene AFTER this
        check are out of contract - a rollout does not create bodies.
        """
        from strands_robots.simulation.benchmark_spec import stop_when_referenced_entities
        from strands_robots.simulation.predicates import can_resolve_body, can_resolve_joint, supports_body_lookup

        bodies, joints = stop_when_referenced_entities(stop_when)

        def _err(text: str) -> dict[str, Any]:
            return {
                "status": "error",
                "content": [
                    {"text": f"run_policy: {text}"},
                    {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}},
                ],
            }

        if bodies and not supports_body_lookup(self):
            return _err(
                f"stop_when references bodies {bodies} but this backend has no body lookup "
                "(get_body_state), so the clause could never fire and the rollout would "
                "silently run to its step budget. Use a clause without body-referencing "
                "predicates, or a backend that supports body lookups."
            )
        missing_bodies = [b for b in bodies if not can_resolve_body(self, b)]
        if missing_bodies:
            return _err(
                f"stop_when references bodies not present in the scene: {missing_bodies}. "
                "The clause would never fire and the rollout would silently run to its "
                "step budget. Check the names against the loaded scene (get_state lists "
                "objects; describe() lists actions)."
            )
        missing_joints = [j for j in joints if not can_resolve_joint(self, j)]
        if missing_joints:
            return _err(
                f"stop_when references joints not present in the observation: {missing_joints}. "
                "The clause would never fire and the rollout would silently run to its "
                "step budget. Check the names against get_observation()'s keys "
                "(joint names are namespaced '<robot>/<joint>')."
            )
        return None

    def _run_episodes(
        self,
        runner: PolicyRunner,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str,
        duration: float,
        n_steps: int | None,
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
        stop_when: Callable[[SimEngine], bool] | None = None,
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

        ``stop_when`` (already compiled to a callable by :meth:`run_policy`)
        is forwarded to every per-episode rollout, giving multi-episode
        collection a per-episode success gate: each episode ends at its own
        predicate hit (or budget), and its dataset episode is flushed with
        exactly the frames captured up to that stop.
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
                n_steps=n_steps,
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
                stop_when=stop_when,
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
        ``total_steps``, per-episode list, ``video_paths``). The payload keeps
        ONE shape across episode counts: ``stopped_reason`` / ``steps_used``
        are present here just as on the single-episode payload -
        ``stopped_reason`` is ``"error"`` on error results and otherwise the
        LAST episode's reason (why the call as a whole stopped running), with
        the per-episode attribution in ``stopped_reasons`` (aligned with
        ``episodes``); ``steps_used`` equals ``total_steps``.
        """
        completed = len(episodes)
        video_paths = [e["video_path"] for e in episodes if e.get("video_path")]
        stopped_reasons = [e.get("stopped_reason") for e in episodes]
        if status == "error":
            stopped_reason = "error"
        elif stopped_reasons and isinstance(stopped_reasons[-1], str):
            stopped_reason = stopped_reasons[-1]
        else:
            stopped_reason = "budget"
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
            "steps_used": total_steps,
            "stopped_reason": stopped_reason,
            "stopped_reasons": stopped_reasons,
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

    def _active_rollout_rates(self) -> dict[str, float]:
        """Capture rate in Hz of every rollout currently in flight, per robot.

        Backends that can run a rollout asynchronously - returning to the caller
        while it continues, as the MuJoCo ``start_policy`` does - override this
        so :meth:`_validate_recording_start_rate` can compare a recording about
        to be opened against what is already capturing frames. The base runs
        every rollout to completion before returning, so no rollout can be in
        flight when a caller reaches ``start_recording`` and the mapping is
        empty.

        Returns:
            ``{robot_name: control_frequency}`` for live rollouts only; an empty
            mapping when none is running.
        """
        return {}

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
            expected: The episode count the caller intended to record. A
                non-negative int; anything else is reported as an error dict.

        Returns:
            Standard status dict. ``status`` is ``"success"`` when the parquet
            holds exactly ``expected`` episodes, else ``"error"``. The
            ``{"json": {...}}`` block carries ``expected``, ``actual``,
            ``info_total_episodes``, ``sources_agree``, ``episode_indices``,
            ``total_frames``, ``total_frames_per_ep``, ``unreadable_files`` and
            ``root`` so a caller (or CI) can fail loudly programmatically.
            ``status`` is ``"error"`` when the parquet count differs from
            ``expected``, when the parquet disagrees with ``meta/info.json``'s
            ``total_episodes`` (``sources_agree`` is then ``False``) - the two
            metadata sources must agree, never just one - and when any episode
            parquet file could not be read (``unreadable_files`` non-empty),
            since the episodes found are then only a lower bound. An unreadable
            or corrupt parquet is reported as this same error dict, never raised.
        """
        # Shares the count domain with the programmatic
        # :func:`strands_robots.verify_dataset.verify_dataset` gate rather than
        # re-deriving it: one rule for one question, so neither surface can accept
        # an episode count the other refuses.
        if error := non_negative_count_error(expected, "expected", "verify_dataset_episodes"):
            return {"status": "error", "content": [{"text": error}]}

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
        except (ValueError, OSError) as e:
            # OSError covers the empty/unfinalized dataset (FileNotFoundError -
            # no episode parquet yet) and an unreadable file; ValueError covers a
            # corrupt / truncated / foreign parquet (pyarrow raises ArrowInvalid,
            # a ValueError subclass). This facade is agent-callable, so both must
            # surface as a structured error dict, never as an escaping traceback.
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
                            "unreadable_files": [],
                            "root": str(root),
                        }
                    },
                ],
            }
        except ImportError as e:
            return {"status": "error", "content": [{"text": f"verify_dataset_episodes: {e}"}]}

        actual = info["total_episodes"]
        info_total = info.get("info_total_episodes")
        unreadable = info.get("unreadable_files") or []

        # Two independent truths must agree: the parquet episode count AND the
        # meta/info.json total_episodes header. A dataset can report the right
        # parquet count yet carry a stale/inconsistent info.json (interrupted
        # finalize), so a parquet-only check is not sufficient. sources_agree is
        # True when info.json is absent (parquet is then the sole truth) or when
        # the header matches the parquet.
        sources_agree = info_total is None or info_total == actual
        # A dataset with unreadable episode parquet files can never be certified:
        # the readable files are a LOWER BOUND on the episode count, so a count
        # that happens to equal ``expected`` proves nothing about the whole
        # dataset. Fail loud and name the broken files.
        ok = actual == expected and sources_agree and not unreadable
        status = "success" if ok else "error"

        if unreadable:
            text = (
                f"verify_dataset_episodes: UNREADABLE - {len(unreadable)} episode "
                f"parquet file(s) could not be read, so the {actual} episode(s) found "
                f"are a lower bound (expected {expected}): {'; '.join(unreadable)}. "
                f"Root: {root}"
            )
        elif not sources_agree:
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
                        "unreadable_files": list(unreadable),
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

        ``episode`` must be a non-negative whole number - the shared domain the
        ``replay_episode`` teleop knob uses - and is rejected with a structured
        error before the dataset is downloaded. A bool is refused rather than
        read as an index: ``episode=True`` previously resolved episode 1 and
        replayed it under a ``"success"`` status.

        ``speed`` is a playback-rate multiplier (1.0 = real time) and must be a
        positive number; a non-positive or non-numeric value is rejected with a
        structured error rather than raising or silently playing back at full
        speed. ``speed`` scales only the wall-clock playback rate: each recorded
        frame always advances physics for a full control period (derived from
        the dataset fps), so a position-servo robot reproduces the recorded
        trajectory instead of under-integrating it.

        ``action_key_map`` binds recorded action-vector indices to action keys
        (default: :meth:`robot_action_keys`). It must be a non-empty list/tuple
        of unique strings whose length matches the recorded action width; a bare
        string, a non-string entry, a duplicate key or a width mismatch is
        rejected rather than truncated to fit. A ``"success"`` status therefore
        means every recorded frame actually reached the actuators - a frame that
        ``send_action`` could not apply aborts the replay with the frame index,
        the frames applied so far and the unresolved keys.

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

        ``seed`` pins the eval the way it pins a single :meth:`run_policy`
        rollout: the client RNGs are reseeded once from it and then per episode
        from a master RNG derived from it, and each per-episode seed is
        forwarded to ``policy.reset`` so a service-mode policy can reseed its
        own process. Two evals at the same seed replay identically; ``None``
        leaves RNG state untouched. Only a non-negative integer can seed those
        RNGs, so anything else is refused here rather than at the first draw.

        ``policy_object`` mirrors :meth:`run_policy`: pass an already-built
        ``Policy`` to skip the ``create_policy`` round-trip (e.g. a loaded
        SmolVLA checkpoint you want to evaluate without re-instantiating).
        When omitted, the policy is built from ``policy_provider`` /
        ``policy_config``.

        ``control_frequency`` / ``control_substeps`` flow through to
        :meth:`PolicyRunner.evaluate` so the eval loop steps physics for the
        full control period per action (same servo-tracking semantics as
        :meth:`run_policy`). Without these the arm under-steps and the policy
        looks like a no-op (the arm under-steps each control period). An explicit
        ``control_substeps`` must be a positive integer - ``0``/negative/float
        is rejected with a structured error instead of collapsing to a single
        physics step, which would reinstate that same no-op.

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
        daemon-thread recorder does not race ``mjData`` mutations. A non-
        ``CooperativeStop`` hook exception is logged at WARN and never aborts
        the eval; raising :class:`~strands_robots.simulation.policy_runner.CooperativeStop`
        stops the evaluation gracefully after the episodes completed so far
        (the result carries ``stopped_early=True`` and ``episodes_completed``),
        matching :meth:`run_policy`.

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

        Returns:
            The standard agent-tool envelope
            ``{"status": "success"|"error", "content": [{"text": ...},
            {"json": {...}}]}``, read the same way as :meth:`run_policy`'s -
            by scanning ``content`` for the first block with a ``"json"`` key,
            never by a fixed index (an early caller-error return carries a
            ``text`` block only).

            ``status`` reports whether the evaluation RAN, not whether the
            policy succeeded: an evaluation in which every episode failed is
            still ``status="success"`` with ``success_rate=0.0``. Read
            ``success_measured`` first - it is ``False`` when no
            ``success_fn`` / benchmark spec was supplied, in which case
            ``success_rate`` is ``0.0`` for every policy regardless of what it
            did and measures nothing.

            Fields in the json block:

            Outcome: ``success_rate``, ``n_success``, ``success_measured``,
            ``episodes_completed``, ``episodes`` (the per-episode records) and
            ``avg_steps``.

            Horizon: ``n_episodes``, ``max_steps`` (the values the evaluation
            ran with) and ``stopped_early``.

            Video: ``video_paths`` (one MP4 per episode, empty when no
            recording was requested).

            Policy binding: ``positional_fallback_used``,
            ``generic_state_keys_used`` and ``missing_state_keys_used`` - True
            means the policy silently fell back to positional camera routing or
            to observation-derived state keys, so the robot moved on
            meaningless inputs and the success rate measures nothing about the
            policy. See :meth:`run_policy` for the full contract.

            Policy load: ``policy_load_time_s``, ``policy_load_cache_hit`` and
            ``policy_resident_rss_mb``.

            Async-RTC telemetry: ``rtc_async_enabled``,
            ``rtc_chunks_acquired``, ``rtc_prefetch_hits``,
            ``rtc_prefetch_blocks``, ``rtc_avg_inference_ms`` and
            ``rtc_max_inference_ms``.
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
                "content": [{"text": self._unknown_robot_msg(resolved_robot)}],
            }

        if err := self._validate_video_config(video, "eval_policy"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "eval_policy"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "eval_policy"):
            return err
        if err := self._validate_action_horizon(action_horizon, "eval_policy"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "eval_policy"):
            return err
        if err := self._validate_positive_int(max_steps, "max_steps", "eval_policy"):
            return err
        if err := self._validate_seed(seed, "eval_policy"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "eval_policy"):
            return err
        if err := self._validate_control_substeps(control_substeps, "eval_policy"):
            return err
        if err := self._validate_rtc_inference_timeout(rtc_inference_timeout_s, "eval_policy"):
            return err
        # Both rates are known only here: the dataset rate was fixed by
        # start_recording one call earlier. Checked before any policy is
        # built so a rate disagreement costs no weight download and no frame.
        if err := self._validate_recording_rate(control_frequency, "eval_policy"):
            return err
        # Coerce to a plain Python float now the value is validated: a NumPy
        # scalar (accepted above via numbers.Real) flows into 1 / control_frequency
        # and time.sleep(...) downstream, and time.sleep rejects a numpy.float32
        # with a bare "cannot be interpreted as an integer" TypeError.
        control_frequency = float(control_frequency)

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
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run a registered :class:`BenchmarkProtocol` against the current sim.

        Benchmark-agnostic evaluation entry point. Looks up ``benchmark_name``
        in the global benchmark registry, validates robot compatibility, and
        forwards to :meth:`PolicyRunner.evaluate` with the spec.
        ``max_steps`` comes from the benchmark (not a parameter here), so it
        is validated where it is read rather than at this signature: a
        benchmark declaring a horizon that is not a positive integer is
        rejected with a structured error, for the same reason ``n_episodes``
        is. Both are bounds of the same nested loop, and a non-positive one
        runs episodes of zero length and then reports a 0% success rate over
        them.

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
                for the recorder side. See #191. Raising
                :class:`~strands_robots.simulation.policy_runner.CooperativeStop`
                from the hook ends the benchmark gracefully after the
                episodes completed so far - the result json carries
                ``stopped_early=True`` and ``episodes_completed`` (matching
                :meth:`run_policy` / :meth:`eval_policy`); any in-progress
                episode's partial video is closed cleanly and is NOT listed
                in ``video_paths``. A non-``CooperativeStop`` hook exception
                is logged at WARN and never aborts the eval.
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
                :meth:`eval_policy`). Must be a positive integer; ``0``,
                negative, float and bool values are rejected with a structured
                error rather than collapsing to a single under-integrated
                physics step. ``None`` (default) derives it from
                ``control_frequency``.
            policy_object: An already-built :class:`Policy` to evaluate,
                skipping the ``create_policy`` round-trip (mirrors
                :meth:`run_policy` / :meth:`eval_policy`). Use it to benchmark a
                checkpoint you have already loaded - e.g. a multi-GB VLA - once
                per process instead of reloading it on every benchmark call.
                When ``None`` the policy is built from ``policy_provider`` /
                ``policy_config``.
            video: Optional per-episode rollout MP4 config (same dict schema as
                :meth:`run_policy` / :meth:`eval_policy`: ``path`` enables it,
                plus ``fps`` / ``camera`` / ``width`` / ``height``). One file
                per episode with ``_ep{i}`` inserted into the filename so a
                benchmark eval can be WATCHED to see why episodes fail, not just
                read as an aggregate success_rate. Frames are captured
                synchronously on the eval thread (render is read-only over
                ``mjData``), so recording does not perturb the bit-stable
                benchmark rollout. Written paths are returned in the result
                json ``video_paths``. ``None`` (default) records nothing.

        Returns:
            Standard status dict. On success, carries per-episode cumulative
            reward + aggregate success_rate / avg_reward / avg_steps in the
            JSON payload, plus ``video_paths`` (the per-episode MP4s written
            when ``video`` is set).
        """
        from strands_robots.policies import create_policy
        from strands_robots.simulation.benchmark import get_benchmark

        if err := self._validate_video_config(video, "evaluate_benchmark"):
            return err
        if err := self._validate_policy_mapping(policy_config, "policy_config", "evaluate_benchmark"):
            return err
        if err := self._validate_policy_mapping(policy_kwargs, "policy_kwargs", "evaluate_benchmark"):
            return err
        if err := self._validate_action_horizon(action_horizon, "evaluate_benchmark"):
            return err
        if err := self._validate_positive_int(n_episodes, "n_episodes", "evaluate_benchmark"):
            return err
        if err := self._validate_positive_frequency(control_frequency, "evaluate_benchmark"):
            return err
        if err := self._validate_control_substeps(control_substeps, "evaluate_benchmark"):
            return err
        # Both rates are known only here: the dataset rate was fixed by
        # start_recording one call earlier. Checked before any policy is
        # built so a rate disagreement costs no weight download and no frame.
        if err := self._validate_recording_rate(control_frequency, "evaluate_benchmark"):
            return err
        if err := self._validate_seed(seed, "evaluate_benchmark"):
            return err
        # Coerce to a plain Python float now the value is validated: a NumPy
        # scalar (accepted above via numbers.Real) flows into 1 / control_frequency
        # and time.sleep(...) downstream, and time.sleep rejects a numpy.float32
        # with a bare "cannot be interpreted as an integer" TypeError.
        control_frequency = float(control_frequency)

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
                "content": [{"text": self._unknown_robot_msg(resolved_robot)}],
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
            video=video,
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

    def register_builtin_benchmarks(self) -> dict[str, Any]:
        """Register the built-in benchmark specs shipped with strands_robots.

        Wraps :func:`strands_robots.simulation.builtin_benchmarks.register_builtin_benchmarks`
        so the shipped specs become discoverable via :meth:`list_benchmarks` and
        runnable via :meth:`evaluate_benchmark` without hand-authoring a spec
        file. Ships a canonical velocity-tracking locomotion benchmark
        (``go2_walk_forward``) composed from the floating-base predicate/reward
        DSL. Opt-in and idempotent (mirrors the on-demand LIBERO suite
        registration); importing strands_robots performs no registry mutation.

        Returns:
            A status dict whose JSON payload carries the ``registered`` list of
            benchmark names now available to :meth:`evaluate_benchmark`.
        """
        from strands_robots.simulation.builtin_benchmarks import (
            register_builtin_benchmarks as _register,
        )

        names = _register()
        return {
            "status": "success",
            "content": [
                {"text": f"Registered {len(names)} built-in benchmark(s): {', '.join(names)}"},
                {"json": {"registered": names}},
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

        Concrete backends define their own parameter signatures. Because this
        base signature is ``**kwargs``-typed, an override inherits a sink that
        would swallow any keyword it does not declare; backends must reject the
        residual keys (see :func:`unknown_kwargs_error`) so a misspelled axis
        cannot report success while leaving that axis untouched.
        Override per backend.
        """
        raise NotImplementedError("randomize not implemented by this backend")

    def set_obs_noise(self, **kwargs: Any) -> dict[str, Any]:
        """Configure additive sensor noise on observations.

        Models real-sensor measurement noise (joint encoders, camera frames)
        so policies are not trained on noise-free observations. Concrete
        backends define their own parameter signatures and, as for
        :meth:`randomize`, must reject keywords they do not declare rather than
        let this ``**kwargs``-typed signature swallow them. Override per backend.
        """
        raise NotImplementedError("set_obs_noise not implemented by this backend")

    def get_contacts(self) -> dict[str, Any]:
        """Get contact information. Override per backend.

        Returns:
            The agent-tool envelope -- ``{"status": ..., "content": [...]}`` --
            whose ``json`` content block carries ``contacts``, a list of
            per-contact records. The payload lives in that block, not on the
            envelope itself, so a caller reading ``result["contacts"]``
            directly always misses. The predicate DSL's ``contact_*``
            factories (see
            :mod:`strands_robots.simulation.predicates`) are the supported
            readers; ``success_fn="contact"`` on
            :meth:`~strands_robots.simulation.policy_runner.PolicyRunner.evaluate`
            shares them.

        Raises:
            NotImplementedError: Backends that expose no contact list.
        """
        raise NotImplementedError("get_contacts not implemented by this backend")

    # Raw-frame render APIs (programmatic, not tool-envelope). Optional per
    # backend, but every in-tree backend implements them: they are the public
    # counterpart of the per-backend private render internals (issue #1537)
    # and the substrate for strands_robots.rendering.HybridCompositor.

    def get_frame(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """Render a camera to raw ``(rgb, depth)`` ndarrays.

        The numeric-array counterpart of :meth:`render` (which wraps pixels in
        the agent-tool PNG envelope). In-process consumers -- the hybrid
        compositor, dataset recorders, video writers -- use this to get pixels
        without a PNG round-trip.

        Args:
            camera_name: name of a camera previously added via ``add_camera``
                (backends supporting a free camera also accept their free-cam
                tokens for the RGB path).
            width: image width in pixels; ``None`` uses the camera's
                configured resolution.
            height: image height in pixels; ``None`` uses the camera's
                configured resolution.

        Returns:
            ``(rgb, depth)`` where ``rgb`` is ``(H, W, 3) uint8`` and
            ``depth`` is ``(H, W) float32`` metric meters, or ``None`` on
            backends with no depth path (Newton). Backends must never
            substitute silently wrong pixels -- failures raise.

        Raises:
            KeyError: unknown camera name.
            ValueError: invalid render dimensions.
            RuntimeError: no world / renderer unavailable / backend render
                failure.
            NotImplementedError: backend has no raw-frame path.
        """
        raise NotImplementedError("get_frame not implemented by this backend")

    def get_camera_params(
        self, camera_name: str = "default", width: int | None = None, height: int | None = None
    ) -> CameraParams:
        """Return pinhole intrinsics/extrinsics for a named camera.

        The returned :class:`strands_robots.rendering.CameraParams` carries
        the intrinsic matrix ``K`` (pixels), the world-from-camera SE(3) pose
        ``T_world_cam`` in the OpenGL optical convention (+X right, +Y up,
        **-Z forward**), the image size, and the clip planes. Backends whose
        native camera basis differs (e.g. Isaac's USD camera prim) apply the
        fixed basis correction here so consumers never see a backend-specific
        frame.

        Args:
            camera_name: name of a camera previously added via ``add_camera``.
                Backends with a free camera also accept their free-cam tokens
                here, reporting the same view :meth:`get_frame` renders, so the
                two APIs stay symmetric (MuJoCo: ``None`` / ``""`` /
                ``"default"`` / ``"free"``).
            width: image width to compute ``K`` for; ``None`` uses the
                camera's configured resolution.
            height: image height to compute ``K`` for; ``None`` uses the
                camera's configured resolution.

        Raises:
            KeyError: unknown camera name.
            ValueError: a camera whose projection no pinhole ``K`` can
                represent (e.g. an orthographic camera), or a resolution the
                backend cannot honor.
            RuntimeError: no world created.
            NotImplementedError: backend has no camera-params path.
        """
        raise NotImplementedError("get_camera_params not implemented by this backend")

    # Hard cap on pixels per get_world_point call: bounds the per-call work an
    # LLM can request (agents ground a handful of samples per object; a whole
    # image belongs to get_frame, not this lookup).
    _WORLD_POINT_MAX_PIXELS = 1024

    def get_world_point(
        self,
        camera_name: str = "default",
        pixels: Sequence[Sequence[SupportsFloat]] | None = None,
        width: int | None = None,
        height: int | None = None,
    ) -> dict[str, Any]:
        """Ground image pixels to metric world coordinates via the depth buffer.

        The perception half of deployment-shaped grounding (Harness VLA,
        arXiv:2607.08448, Appendix E.2): instead of reading privileged object
        poses (:meth:`get_body_state` -- sim-only oracle truth), the agent
        picks pixels on the visible surface of the target in the RGB frame and
        this call unprojects each one through the pixel-aligned metric depth
        buffer -- ``p_cam = depth * K^-1 @ [u, v, 1]`` in the OpenGL optical
        frame, then ``p_world = T_world_cam @ p_cam``. The same call shape
        works on hardware with an RGB-D camera, so grounding built on it
        transfers.

        Guidance for agents (the paper's localization rule):

        1. Render the camera first (``render`` / ``get_frame``) and pick
           pixels ON the visible surface of the target object.
        2. Avoid rims, edges, reflections, transparent surfaces, and
           background pixels -- depth there is unstable or belongs to
           something else.
        3. Sample SEVERAL pixels on the same surface (typically 3-9): the
           returned ``point`` is the median over the valid samples, which
           rejects stray outliers. The median is PER-COMPONENT, so on a
           strongly tilted surface the combined ``[x, y, z]`` may lie on no
           single sampled point - treat it as a robust surface estimate, not
           as one of ``points``.
        4. Pixels with no depth (background / far plane) are dropped, not
           zero-filled; check ``n_valid`` against the count you sent.
        5. Re-localize after any robot, camera, or object motion -- world
           points are snapshots, not tracks.

        Depth samples are treated as z-depth (distance along the optical
        axis), the convention every in-tree backend emits. Pixels are indexed
        ``[u, v]`` with ``u`` the column from the left and ``v`` the row from
        the top; the unprojection uses the pixel center ``(u + 0.5, v + 0.5)``.

        Atomicity: when the backend exposes an engine lock (``self._lock``,
        all in-tree backends), the frame render and the camera-params read
        happen under it, so a concurrent scene mutation cannot slip between
        the two. All failures return a structured error dict -- this is a
        tool-envelope method and never raises.

        Args:
            camera_name: a camera previously added via ``add_camera``
                (backends with a free camera also accept their free-cam
                tokens, as for :meth:`get_frame`).
            pixels: non-empty list of ``[u, v]`` pixel coordinates
                (integer-valued; at most ``_WORLD_POINT_MAX_PIXELS``).
            width: image width; ``None`` uses the camera's configured
                resolution.
            height: image height; ``None`` uses the camera's configured
                resolution.

        Returns:
            On success ``{"status": "success", "content": [{"text": ...},
            {"json": {"point": [x, y, z], "points": [...], "n_valid": int,
            "n_requested": int, "camera": str, "width": int, "height": int}}]}``
            where ``point`` is the per-component median over the valid
            samples and ``points`` is aligned with the input ``pixels``
            (``None`` where the pixel had no valid depth). Backends without a
            metric-depth path (Newton), all-invalid pixel sets, out-of-bounds
            pixels, malformed input, and a failed frame render or
            camera-params read all return
            ``{"status": "error", "content": [{"text": ...}]}``, with the
            two backend reads reporting distinguishable text so a caller
            knows which one failed. The camera-params read can fail on input
            this call already accepted and a frame it already rendered --
            most notably a camera whose projection no pinhole ``K`` can
            represent, such as MuJoCo's orthographic free camera, which
            renders normally but has no intrinsics. So check ``status``
            rather than inferring success from a valid pixel set.
        """
        # numpy stays TYPE_CHECKING-only at this module's top level; import at
        # use time like the backends' render paths do.
        import numpy as np

        def _err(msg: str) -> dict[str, Any]:
            return {"status": "error", "content": [{"text": msg}]}

        # -- Structural validation (before any render work) -- #
        # camera_name reaches backend name-lookup APIs (e.g. MuJoCo's
        # mj_name2id) that raise TypeError on non-string input, and the
        # dispatcher enforces no scalar types - so an LLM passing a camera
        # INDEX must be caught here to keep the never-raises envelope.
        if camera_name is not None and not isinstance(camera_name, str):
            return _err(
                f"get_world_point 'camera_name' must be a camera name string, got "
                f"{type(camera_name).__name__} ({camera_name!r}). Cameras are addressed by "
                "name, not index; see add_camera / get_frame."
            )
        n_pixels = None if pixels is None or isinstance(pixels, (str, bytes)) else sequence_length(pixels)
        # ``pixels is None`` is retested rather than folded into ``n_pixels`` so
        # the narrowing survives to the ``enumerate(pixels)`` walk below.
        if pixels is None or n_pixels is None:
            return _err(
                "get_world_point requires 'pixels': a non-empty list of [u, v] pixel coordinates, e.g. [[320, 240], [322, 238]]."
            )
        if n_pixels == 0:
            return _err("get_world_point requires at least one [u, v] pixel; got an empty list.")
        if n_pixels > self._WORLD_POINT_MAX_PIXELS:
            return _err(
                f"get_world_point accepts at most {self._WORLD_POINT_MAX_PIXELS} pixels per call, "
                f"got {n_pixels}. Sample a handful of pixels on the target surface instead."
            )
        parsed: list[tuple[int, int]] = []
        for i, px in enumerate(pixels):
            if isinstance(px, (str, bytes)) or sequence_length(px) != 2:
                return _err(f"pixels[{i}] must be a [u, v] pair, got {px!r}.")
            coords: list[int] = []
            for axis, component in zip("uv", px, strict=True):
                if not isinstance(component, numbers.Real) or isinstance(component, bool):
                    return _err(f"pixels[{i}] {axis} must be numeric, got {type(component).__name__}.")
                value = float(component)
                if not math.isfinite(value):
                    return _err(f"pixels[{i}] {axis} must be finite, got {value}.")
                if not value.is_integer():
                    return _err(
                        f"pixels[{i}] {axis} must be an integer pixel index, got {value} "
                        "(fractional pixels are rejected, never silently truncated)."
                    )
                coords.append(int(value))
            parsed.append((coords[0], coords[1]))

        # -- Render + camera params (atomic under the engine lock) -- #
        lock = getattr(self, "_lock", None)
        ctx = lock if lock is not None else contextlib.nullcontext()
        with ctx:
            try:
                _rgb, depth = self.get_frame(camera_name, width=width, height=height)
            except NotImplementedError:
                return _err(
                    "get_world_point is unavailable: this backend has no raw-frame path (get_frame is not implemented)."
                )
            except (KeyError, ValueError, RuntimeError, TypeError) as e:
                # TypeError included as defense-in-depth for backend lookup
                # APIs that reject non-string names (the type is validated
                # above, but the envelope must hold regardless).
                return _err(f"get_world_point failed to render camera frame: {e}")
            if depth is None:
                return _err(
                    f"get_world_point is unavailable on this backend: camera '{camera_name}' produced no "
                    "metric depth (get_frame returned depth=None; e.g. Newton's ray-traced camera has no "
                    "depth output). Use a depth-capable backend (MuJoCo, Isaac) or an RGB-D camera."
                )
            h, w = int(depth.shape[0]), int(depth.shape[1])
            for i, (u, v) in enumerate(parsed):
                if not (0 <= u < w and 0 <= v < h):
                    return _err(
                        f"pixels[{i}] = [{u}, {v}] is outside the rendered {w}x{h} frame "
                        f"(valid u: 0..{w - 1}, v: 0..{h - 1})."
                    )
            try:
                cam = self.get_camera_params(camera_name, width=w, height=h)
            except NotImplementedError:
                return _err(
                    "get_world_point is unavailable: this backend has no camera-params path (get_camera_params is not implemented)."
                )
            except (KeyError, ValueError, RuntimeError, TypeError) as e:
                return _err(f"get_world_point failed to read camera parameters: {e}")

        # -- Unproject (pure math; no engine state touched past this point) -- #
        fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
        cx, cy = float(cam.K[0, 2]), float(cam.K[1, 2])
        # Background convention across backends: MuJoCo pins no-geometry
        # pixels to exactly zfar; Isaac reports 0 or non-finite. A small
        # relative margin below zfar absorbs float rounding at the far plane.
        zfar_cut = float(cam.zfar) * (1.0 - 1e-6)
        points: list[list[float] | None] = []
        valid_points: list[list[float]] = []
        for u, v in parsed:
            d = float(depth[v, u])
            if not (math.isfinite(d) and 0.0 < d < zfar_cut):
                points.append(None)
                continue
            # Pixel center -> OpenGL optical frame (+X right, +Y up, -Z
            # forward): image v grows down so y flips, and z-depth lies
            # along -Z.
            x_cam = (u + 0.5 - cx) / fx * d
            y_cam = -((v + 0.5 - cy) / fy) * d
            p_world = cam.T_world_cam @ np.array([x_cam, y_cam, -d, 1.0], dtype=np.float64)
            world_xyz = [float(p_world[0]), float(p_world[1]), float(p_world[2])]
            points.append(world_xyz)
            valid_points.append(world_xyz)

        # One label per camera: every free-camera token (None / "" / "free" /
        # "default") reports as "default", so the same camera never appears
        # under two names across calls.
        camera_label = "default" if camera_name in FREE_CAMERA_TOKENS else str(camera_name)
        if not valid_points:
            return _err(
                f"get_world_point found no valid depth at any of the {len(parsed)} requested pixels via "
                f"camera '{camera_label}': every sample hit the background / far plane (zfar={cam.zfar:g} m) "
                "or had no depth. Pick pixels on the visible surface of the target object -- avoid sky, "
                "rims/edges, reflections, and background."
            )

        median = np.median(np.asarray(valid_points, dtype=np.float64), axis=0)
        point = [float(median[0]), float(median[1]), float(median[2])]
        n_valid = len(valid_points)
        text = (
            f"World point [{point[0]:.4f}, {point[1]:.4f}, {point[2]:.4f}] m "
            f"(median over {n_valid}/{len(parsed)} valid pixels) via camera '{camera_label}'."
        )
        return {
            "status": "success",
            "content": [
                {"text": text},
                {
                    "json": {
                        "point": point,
                        "points": points,
                        "n_valid": n_valid,
                        "n_requested": len(parsed),
                        "camera": camera_label,
                        "width": w,
                        "height": h,
                    }
                },
            ],
        }

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
                "send_action": (
                    "(action: dict, robot_name: str | None = None, n_substeps: int = 1) -> dict"
                    "  # n_substeps must be a positive whole number; use step() to advance without commanding"
                ),
                "add_robot": (
                    "(name: str, urdf_path=None, data_config=None, position=None, "
                    "orientation=None) -> dict  # add a robot to the scene by "
                    "registry name (or urdf_path); the first scene-construction step. "
                    "position OFFSETS the model's own authored root pose (a locomotion "
                    "model is authored standing), so it is the world position only for "
                    "a model whose root declares pos 0 0 0; the result reports the "
                    "measured placement"
                ),
                "add_object": (
                    "(name: str, shape='box', position=None, orientation=None, "
                    "size=None, color=None, mass=0.1, is_static=None, mesh_path=None, "
                    "material=None) -> dict  # add a manipulable object "
                    "(cube/sphere/.../mesh) to the scene. material is an optional "
                    "dict for matte/textured surfaces: keys reflectance|specular|"
                    "shininess (0..1), texture (abs image path) OR builtin "
                    "(checker|gradient|flat) + rgb1/rgb2/texdim, texrepeat [u,v]; "
                    "any other key (or an empty dict) is rejected, never ignored"
                ),
                "remove_object": "(name: str) -> dict  # remove a previously added object",
                "remove_robot": (
                    "(name: str) -> dict  # remove a robot (and every scene "
                    "element it introduced) from the world; the inverse of "
                    "add_robot, completing the add/remove pair alongside "
                    "remove_object"
                ),
                "run_policy": (
                    "(robot_name: str, policy_provider='mock', n_episodes=1, "
                    "reset_between=True, stop_when=None, ...) -> dict  # "
                    "stop_when: optional semantic early-return clause in the "
                    "benchmark success: predicate DSL - a single "
                    "{'predicate': <name>, ...} call or an {'all'/'any': "
                    "[...]} group - checked against the sim after every "
                    "applied action so the rollout ends as soon as the world "
                    "reaches the state; the result json reports "
                    "stopped_reason ('predicate'|'budget'|'cancelled'; "
                    "'error' on failures) + steps_used so a caller can decide "
                    "whether to retry"
                ),
                "start_policy": "(robot_name: str, policy_provider='mock', ...) -> dict",
                "eval_policy": (
                    "(robot_name: str, policy_provider='mock', n_episodes=1, "
                    "max_steps=300, success_fn=None, ...) -> dict  # multi-episode "
                    "success-rate evaluation (the rollout sibling of run_policy)"
                ),
                "evaluate_benchmark": (
                    "(benchmark_name: str, robot_name=None, policy_provider='mock', "
                    "n_episodes=1, seed=None, video=None, ...) -> dict  # score a "
                    "registered benchmark's success/failure/dense-reward DSL over a "
                    "rollout (max_steps comes from the benchmark, not a parameter); "
                    "the DSL-scored sibling of eval_policy's success_fn"
                ),
                "list_benchmarks": (
                    "() -> dict  # enumerate registered benchmarks (names, "
                    "supported robots, default robot, max_steps) - the source of the "
                    "benchmark_name evaluate_benchmark expects"
                ),
                "register_benchmark_from_file": (
                    "(benchmark_name: str, spec_path: str) -> dict  # author a "
                    "declarative benchmark (success/failure/dense_reward predicate "
                    "DSL) as YAML/JSON at runtime and register it under benchmark_name"
                ),
                "register_builtin_benchmarks": (
                    "() -> dict  # register the shipped built-in velocity-tracking "
                    "locomotion benchmarks - the go2_walk_forward quadruped task and "
                    "the g1_walk_forward / t1_walk_forward humanoid tasks - so they "
                    "appear in list_benchmarks and can be run via evaluate_benchmark"
                ),
                "replay_episode": (
                    "(repo_id: str, robot_name=None, episode=0, root=None, "
                    "speed=1.0, action_key_map=None) -> dict  # replay a recorded "
                    "LeRobotDataset episode through the sim; action_key_map needs "
                    "one unique key per recorded action index (default: "
                    "robot_action_keys) and status='success' means every frame "
                    "reached the actuators"
                ),
                "list_robots": "() -> list[str]",
                "get_features": (
                    "(robot_name: str | None = None) -> dict  # joint / "
                    "actuator / camera / robot names of the scene (scoped to "
                    "one robot when robot_name is given) - the source of truth "
                    "for the action keys a policy must emit; consult it when "
                    "run_policy reports unresolved keys"
                ),
                "render": "(camera_name='default', width=None, height=None) -> dict",
                "create_world": (
                    "(timestep=None, gravity=None, ground_plane=True, terrain=None, "
                    "difficulty=1.0) -> dict  # create a fresh simulation world - the "
                    "world-lifecycle entry point that precedes add_robot / add_object. "
                    "gravity is [gx, gy, gz]; ground_plane lays a floor; terrain lays a "
                    "deterministic locomotion heightfield instead of the flat plane "
                    "('rough' value-noise bumps, 'stairs' step plateaus rising +x, "
                    "'pyramid' concentric steps rising to the centre, 'slope' a "
                    "constant-grade ramp); difficulty (finite, > 0; 1.0 = full height) "
                    "scales the terrain peak elevation for a curriculum without changing "
                    "the terrain kind. Backends without heightfield support reject a "
                    "non-None terrain rather than ignoring it"
                ),
                "destroy": (
                    "() -> dict  # tear down the world and release all resources "
                    "(joins any running background policy first); the inverse of "
                    "create_world, called at session end"
                ),
                "reset": "() -> dict  # during recording, flushes the buffered rollout as one episode before resetting",
                "step": "(n_steps: int = 1) -> dict",
                "get_state": (
                    "() -> dict  # snapshot of the live world: sim time, step "
                    "count, timestep, gravity, and robot / object / camera / "
                    "body / joint / actuator counts (the whole-world sibling of "
                    "get_robot_state / get_observation)"
                ),
                "load_scene": (
                    "(scene_path: str) -> dict  # load a complete scene from "
                    "an MJCF/URDF file; the alternative scene-construction "
                    "entry point to building it up with add_robot / add_object"
                ),
                "randomize": (
                    "(**kwargs) -> dict  # domain randomization (colors, "
                    "lighting, physics, positions); each backend defines its "
                    "own opt-in axes - see the backend describe() for the "
                    "concrete signature"
                ),
                "set_obs_noise": (
                    "(**kwargs) -> dict  # configure additive Gaussian sensor "
                    "noise on joint observations and rendered frames so a "
                    "policy is not evaluated on noise-free observations"
                ),
                "get_contacts": (
                    "() -> dict  # active contacts at the current step - the "
                    "physics-grounding read used to verify a grasp or detect "
                    "a collision instead of trusting a rendered caption"
                ),
            },
            "note": (
                "robot_name defaults to the sole robot when only one exists "
                "for get_observation, send_action, get_robot_state, run_policy, "
                "and start_policy. With multiple robots, pass robot_name "
                "explicitly (from the 'robots' list above)."
            ),
        }

    def cleanup(self) -> None:
        """Release all resources.

        Called on context exit, and best-effort from :meth:`__del__` for an
        engine whose ``__init__`` ran to completion. Implementations are
        written against a fully-constructed instance: a caller whose
        ``__init__`` raised part-way must release whatever it acquired itself
        rather than relying on the finalizer.
        """
        pass

    def __enter__(self) -> SimEngine:
        return self

    def __exit__(self, *exc: object) -> None:
        self.cleanup()

    def __del__(self) -> None:
        if not self._init_complete:
            # ``__init__`` never finished, so this instance holds no engine
            # resources. Calling cleanup() here would raise on whichever
            # attribute ``__init__`` had not reached yet and report that name
            # as a cleanup failure - noise indistinguishable from a real leak,
            # and a red herring for whoever reads it.
            return
        try:
            self.cleanup()
        except Exception as e:
            # Best-effort cleanup during GC - exceptions can't propagate
            # from __del__ (CPython ignores them), so log for visibility.
            logger.warning("Cleanup error during __del__: %s", e)
