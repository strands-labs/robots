"""Named-predicate library for declarative :class:`BenchmarkProtocol` specs.

Each entry in :data:`PREDICATE_REGISTRY` is a factory ``(**kwargs) -> callable``
where the returned callable takes a :class:`SimEngine` and returns either
``bool`` (for success/failure predicates) or ``float`` (for reward terms).

The registry is a closed set - the YAML/JSON loader in
:mod:`strands_robots.simulation.benchmark_spec` refuses predicates whose
name is not in this registry, so spec files are safe to parse from
untrusted / LLM-authored input. **No ``eval`` is ever called.** User-defined
predicates must be registered programmatically via :func:`register_predicate`
before loading the spec.

Predicates are backend-aware but not backend-specific: they exclusively call
``SimEngine`` methods (abstract) or probe for MuJoCo-only methods via
``getattr`` and return a safe fallback (``False`` / ``0.0``) when the
backend does not support them. A predicate that silently evaluates to
``False`` because of an unimplemented backend call is a bug in the
predicate, not the benchmark - file an issue.

Available predicates (bool):

    body_above_z(body, z)
    body_below_z(body, z)
    joint_above(joint, value)
    joint_below(joint, value)
    distance_less_than(body_a, body_b, threshold)
    inside_region(body, min, max)
    contact_between(geom_a, geom_b)
    contact_any()
    body_on(body_a, body_b, z_offset=0.02, xy_tol=0.15)
    body_inside(body, container, xy_tol=0.15, z_tol=0.15)
    body_upright(body, tol=0.15)
    grasped(body, gripper_prefix)

Available reward terms (float):

    distance_neg(body_a, body_b, weight=1.0)
    joint_progress(joint, target, weight=1.0)
    constant(value)

Register custom predicates with :func:`register_predicate`.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from strands_robots.simulation.base import SimEngine

logger = logging.getLogger(__name__)

BoolPredicate = Callable[["SimEngine"], bool]
RewardTerm = Callable[["SimEngine"], float]
PredicateFactory = Callable[..., Callable[["SimEngine"], Any]]


# Helpers for digging values out of the structured ``{"status", "content"}``
# dicts that MuJoCo-backend methods return. Defensive against empty content
# lists and missing keys - predicates should never crash the eval loop.


def _extract_json(result: dict[str, Any] | None) -> dict[str, Any]:
    """Return the ``json`` content block payload, or ``{}`` if absent."""
    if not isinstance(result, dict):
        return {}
    for block in result.get("content", []) or []:
        if isinstance(block, dict):
            payload = block.get("json")
            if isinstance(payload, dict):
                # dict[str, Any] by construction of the content schema; mypy can't
                # narrow through dict.get() so we cast via a new dict to keep it typed.
                return dict(payload)
    return {}


def _body_position(sim: SimEngine, body: str) -> list[float] | None:
    """Best-effort body-position lookup. Returns ``None`` on any failure.

    Requires the backend to implement ``get_body_state`` (MuJoCo only at time
    of writing). Future backends can add the same method signature - see
    :meth:`strands_robots.simulation.mujoco.physics.PhysicsMixin.get_body_state`.

    LIBERO body-name convention: BDDL names objects without a suffix
    (``porcelain_mug_1``), but the MJCF root body is suffixed with
    ``_main`` (``porcelain_mug_1_main``). Upstream resolves this via
    ``env.objects_dict[name].root_body`` (see
    ``libero/libero/envs/bddl_base_domain.py``). We mirror that with a
    bounded fallback: try the bare name first, then ``<name>_main`` if
    the bare lookup fails. #176 (sub-task 3d) — without this
    fallback, BDDL goal predicates like ``(On porcelain_mug_1
    plate_1)`` resolve to ``None`` (body not found) → predicate
    silently False even when the mug is physically on the plate.
    """
    get_body_state = getattr(sim, "get_body_state", None)
    if get_body_state is None:
        return None

    def _try(name: str) -> list[float] | None:
        try:
            result = get_body_state(body_name=name)
        except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
            logger.debug("body_position(%r) failed: %s", name, e)
            return None
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        payload = _extract_json(result)
        pos = payload.get("position")
        if isinstance(pos, list) and len(pos) == 3 and all(isinstance(c, (int, float)) for c in pos):
            return [float(c) for c in pos]
        return None

    # 1. Bare name (works for fixtures with explicit body names matching
    # the BDDL name, e.g. ``living_room_table``).
    pos = _try(body)
    if pos is not None:
        return pos
    # 2. LIBERO ``<name>_main`` convention (the root body of
    # procedurally-generated objects). Skip if the name already has
    # the suffix to avoid double-suffixing on retries.
    if not body.endswith("_main"):
        pos = _try(f"{body}_main")
        if pos is not None:
            return pos
    return None


def _joint_position(sim: SimEngine, joint: str) -> float | None:
    """Best-effort joint-position lookup via ``get_observation``.

    ``get_observation`` is on the ABC and returns ``{<joint_name>: float}``.
    When the joint is absent from the observation dict (wrong robot, wrong
    namespace) we return ``None`` so predicates can decide between ``False``
    and an explicit error path.
    """
    try:
        obs = sim.get_observation(skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive
        logger.debug("get_observation() failed: %s", e)
        return None
    if not isinstance(obs, dict):
        return None
    val = obs.get(joint)
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        return float(val)
    return None


def _body_quaternion(sim: SimEngine, body: str) -> list[float] | None:
    """Best-effort quaternion lookup. Returns ``None`` on any failure.

    Quaternion convention: MuJoCo reports ``[w, x, y, z]``. Callers that
    need just an axis can derive it from the rotation matrix, but doing
    the arithmetic inline here keeps the predicate library numpy-free.
    """
    get_body_state = getattr(sim, "get_body_state", None)
    if get_body_state is None:
        return None
    try:
        result = get_body_state(body_name=body)
    except Exception as e:  # noqa: BLE001 - defensive
        logger.debug("body_quaternion(%r) failed: %s", body, e)
        return None
    if not isinstance(result, dict) or result.get("status") != "success":
        return None
    payload = _extract_json(result)
    quat = payload.get("quaternion")
    if isinstance(quat, list) and len(quat) == 4 and all(isinstance(c, (int, float)) for c in quat):
        return [float(c) for c in quat]
    return None


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Simple 3D Euclidean distance; no numpy so predicates stay dependency-free."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


# Predicate factories


def _body_above_z(body: str, z: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        return pos is not None and pos[2] > float(z)

    return check


def _body_below_z(body: str, z: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        return pos is not None and pos[2] < float(z)

    return check


def _joint_above(joint: str, value: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        q = _joint_position(sim, joint)
        return q is not None and q > float(value)

    return check


def _joint_below(joint: str, value: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        q = _joint_position(sim, joint)
        return q is not None and q < float(value)

    return check


def _distance_less_than(body_a: str, body_b: str, threshold: float) -> BoolPredicate:
    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return False
        return _euclidean_distance(pos_a, pos_b) < float(threshold)

    return check


def _inside_region(body: str, min: list[float], max: list[float]) -> BoolPredicate:  # noqa: A002 - DSL keyword
    if not (isinstance(min, list) and len(min) == 3 and isinstance(max, list) and len(max) == 3):
        raise ValueError("inside_region: 'min' and 'max' must each be a list of 3 numbers")
    lo = [float(c) for c in min]
    hi = [float(c) for c in max]
    if any(lo[i] > hi[i] for i in range(3)):
        raise ValueError(f"inside_region: 'min' {lo} must be component-wise <= 'max' {hi}")

    def check(sim: SimEngine) -> bool:
        pos = _body_position(sim, body)
        if pos is None:
            return False
        return all(lo[i] <= pos[i] <= hi[i] for i in range(3))

    return check


def _contact_between(geom_a: str, geom_b: str) -> BoolPredicate:
    """Pairwise contact predicate.

    Requires ``get_contacts()`` (MuJoCo). Ignores contact ordering - a contact
    reported as ``(geom_a, geom_b)`` matches the same predicate as
    ``(geom_b, geom_a)``.
    """

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("contact_between(%r,%r) failed: %s", geom_a, geom_b, e)
            return False
        payload = _extract_json(result)
        contacts = payload.get("contacts")
        if not isinstance(contacts, list):
            return False
        want = {geom_a, geom_b}
        for c in contacts:
            if not isinstance(c, dict):
                continue
            pair = {c.get("geom1"), c.get("geom2")}
            if want <= pair:
                return True
        return False

    return check


def _contact_any() -> BoolPredicate:
    """Sparse "any contact" predicate - matches the legacy ``success_fn='contact'`` path."""

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("contact_any() failed: %s", e)
            return False
        payload = _extract_json(result)
        if payload.get("n_contacts", 0) > 0:
            return True
        contacts = payload.get("contacts")
        return bool(isinstance(contacts, list) and contacts)

    return check


def _body_contact(sim: SimEngine, body_a: str, body_b: str) -> bool | None:
    """Best-effort body-contact lookup.

    Returns ``True`` / ``False`` when ``sim.get_contacts()`` is available
    AND any geom of ``body_a`` is in contact with any geom of ``body_b``.
    Returns ``None`` when ``get_contacts()`` is unavailable so the
    caller can decide whether to gracefully degrade (fall back to
    geometric-only checks) or hard-fail.

    Heuristic: matches contacts by **geom name prefix** (``<bddl_name>_g``
    for LIBERO scenes; works for any scene whose geoms follow the
    ``<body_name>_g<idx>`` convention). Mirrors how upstream LIBERO's
    ``ObjectState.check_contact`` walks the per-object geom list, but
    avoids hard-coding the body→geom map by using the naming
    convention.

    Used by the contact-aware branch of :func:`_body_on` (LIBERO's
    ``On(A, B)`` predicate semantics requires
    ``arg2.check_contact(arg1)`` per
    ``libero/libero/envs/predicates/base_predicates.py``).
    """
    get_contacts = getattr(sim, "get_contacts", None)
    if get_contacts is None:
        return None
    try:
        result = get_contacts()
    except Exception as e:  # noqa: BLE001 - defensive
        logger.debug("body_contact(%r, %r) get_contacts raised: %s", body_a, body_b, e)
        return None
    if not isinstance(result, dict) or result.get("status") != "success":
        # Engine returned an error stub or a malformed payload; treat as
        # "unknown" so the caller can degrade gracefully (False would
        # be a false negative; we want geometric-only fallback).
        return None
    payload = _extract_json(result)
    contacts = payload.get("contacts")
    if not isinstance(contacts, list):
        return None

    prefix_a = f"{body_a}_g"
    prefix_b = f"{body_b}_g"
    for c in contacts:
        if not isinstance(c, dict):
            continue
        g1 = c.get("geom1") or ""
        g2 = c.get("geom2") or ""
        # Geom-prefix matching: ``<bddl_name>_g<idx>`` is LIBERO's
        # convention. Either direction (a-then-b or b-then-a) counts.
        if (g1.startswith(prefix_a) and g2.startswith(prefix_b)) or (
            g1.startswith(prefix_b) and g2.startswith(prefix_a)
        ):
            return True
    return False


def _body_on(
    body_a: str,
    body_b: str,
    z_offset: float = 0.02,
    xy_tol: float = 0.15,
    require_contact: bool = False,
) -> BoolPredicate:
    """Approximate ``(on A B)`` predicate - A resting on top of B.

    True when ``A.z > B.z + z_offset`` AND horizontal distance ``|A.xy - B.xy|
    < xy_tol``. When ``require_contact=True``, ALSO requires physics
    contact between A and B via ``sim.get_contacts()`` — matches
    upstream LIBERO's ``ObjectState.check_ontop`` which combines a
    geometric check with ``check_contact``. The z-offset parameter
    accounts for B's half-height + a small buffer; tune per scene.
    Intended for sparse-success benchmarks (LIBERO, etc.) where exact
    geometric containment isn't required.

    Contact-check graceful degradation: when
    ``require_contact=True`` but the sim engine doesn't expose
    ``get_contacts`` (e.g. test stubs, custom engines), the contact
    check is skipped and only the geometric check fires. This
    preserves backwards compatibility — engines without contact
    support get the pre-#171 behaviour. LIBERO benchmarks running on
    ``MuJoCoSimEngine`` (which implements ``get_contacts``) get the
    strict upstream-matching semantics.

    For full fidelity (MJCF geom size lookup + narrow-phase collision), write
    a scene-specific predicate and register it via :func:`register_predicate`.
    """

    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return False
        dx = pos_a[0] - pos_b[0]
        dy = pos_a[1] - pos_b[1]
        if (dx * dx + dy * dy) ** 0.5 > float(xy_tol):
            return False
        if not (pos_a[2] > pos_b[2] + float(z_offset)):
            return False
        if require_contact:
            in_contact = _body_contact(sim, body_a, body_b)
            # ``None`` ⇒ engine doesn't support contacts; fall back to
            # geometric-only verdict (preserves pre-#171 behaviour).
            # ``False`` ⇒ engine reports no contact ⇒ predicate False.
            # ``True`` ⇒ contact confirmed ⇒ predicate True (combined
            # with the passing geometric check above).
            if in_contact is False:
                return False
        return True

    return check


def _body_inside(body: str, container: str, xy_tol: float = 0.15, z_tol: float = 0.15) -> BoolPredicate:
    """Approximate ``(inside A B)`` predicate - A contained within B's volume.

    True when A's position is within an axis-aligned box centered on B with
    half-extents (``xy_tol``, ``xy_tol``, ``z_tol``). LIBERO-typical use is
    "object inside basket / drawer / compartment" where exact bbox is
    benchmark-specific; the defaults are tuned for table-top manipulation.

    When richer geometry is available, override by registering a
    scene-specific predicate.
    """

    def check(sim: SimEngine) -> bool:
        pos_a = _body_position(sim, body)
        pos_b = _body_position(sim, container)
        if pos_a is None or pos_b is None:
            return False
        return (
            abs(pos_a[0] - pos_b[0]) <= float(xy_tol)
            and abs(pos_a[1] - pos_b[1]) <= float(xy_tol)
            and abs(pos_a[2] - pos_b[2]) <= float(z_tol)
        )

    return check


def _body_upright(body: str, tol: float = 0.15) -> BoolPredicate:
    """True when ``body``'s local +Z axis is within ``tol`` of world +Z.

    Computes the rotation-matrix element ``R[2,2]`` from the body's
    quaternion. Upright → ``R[2,2] > 1 - tol``. The math (all unit-quat
    identities, w² + x² + y² + z² = 1):

        R[2,2] = 1 - 2*(x² + y²)

    so the check is ``2*(x² + y²) < tol``. This is monotonic in "how
    tipped over" the body is, so a small tol (0.01-0.2) corresponds
    directly to the maximum allowed tilt.
    """
    t = float(tol)
    if t < 0:
        raise ValueError(f"body_upright: 'tol' must be >= 0, got {t}")

    def check(sim: SimEngine) -> bool:
        quat = _body_quaternion(sim, body)
        if quat is None:
            return False
        # MuJoCo quat layout is (w, x, y, z).
        _, x, y, _ = quat
        return 2.0 * (x * x + y * y) < t

    return check


def _grasped(body: str, gripper_prefix: str) -> BoolPredicate:
    """True when ``body`` is in contact with any geom whose name starts with ``gripper_prefix``.

    Treats the gripper as a *set* of geoms (fingers, pads, tip sites) so
    the caller only has to specify the common prefix - e.g. ``"robot0_gripper"``
    for Panda covers both fingers. A body is "grasped" as long as any one
    gripper geom is in contact with any geom matching the body name.

    Backends must implement ``get_contacts()`` returning the MuJoCo
    ``{"contacts": [{"geom1", "geom2", ...}]}`` shape. Other backends are
    treated as "cannot check" and return ``False``.
    """

    def check(sim: SimEngine) -> bool:
        get_contacts = getattr(sim, "get_contacts", None)
        if get_contacts is None:
            return False
        try:
            result = get_contacts()
        except Exception as e:  # noqa: BLE001 - defensive
            logger.debug("grasped(%r, %r) failed: %s", body, gripper_prefix, e)
            return False
        payload = _extract_json(result)
        contacts = payload.get("contacts")
        if not isinstance(contacts, list):
            return False
        for c in contacts:
            if not isinstance(c, dict):
                continue
            g1 = c.get("geom1") or ""
            g2 = c.get("geom2") or ""
            # One side must be the grasped body (bare name or "_geom" suffix);
            # the other must start with the gripper prefix.
            body_match = {g1, g2} & {body, f"{body}_geom"}
            gripper_match = any(isinstance(g, str) and g.startswith(gripper_prefix) for g in (g1, g2))
            if body_match and gripper_match:
                return True
        return False

    return check


# Reward terms (float-valued)


def _distance_neg(body_a: str, body_b: str, weight: float = 1.0) -> RewardTerm:
    """Negative Euclidean distance between two bodies, weighted.

    The canonical "reach" reward: ``weight * -dist(a, b)``. Monotonic in
    the distance, so naive policy improvement pulls the bodies together.
    """
    w = float(weight)

    def term(sim: SimEngine) -> float:
        pos_a = _body_position(sim, body_a)
        pos_b = _body_position(sim, body_b)
        if pos_a is None or pos_b is None:
            return 0.0
        return -w * _euclidean_distance(pos_a, pos_b)

    return term


def _joint_progress(joint: str, target: float, weight: float = 1.0) -> RewardTerm:
    """Negative absolute distance from a joint to its target, weighted.

    Useful for drawer/door tasks where success is "joint near target
    position" and you want dense signal during training.
    """
    w = float(weight)
    t = float(target)

    def term(sim: SimEngine) -> float:
        q = _joint_position(sim, joint)
        if q is None:
            return 0.0
        return -w * abs(q - t)

    return term


def _constant(value: float) -> RewardTerm:
    """Constant reward per step. Useful for shaping a survival bonus."""
    v = float(value)

    def term(_sim: SimEngine) -> float:
        return v

    return term


# Registry

PREDICATE_REGISTRY: dict[str, PredicateFactory] = {
    # bool-valued
    "body_above_z": _body_above_z,
    "body_below_z": _body_below_z,
    "joint_above": _joint_above,
    "joint_below": _joint_below,
    "distance_less_than": _distance_less_than,
    "inside_region": _inside_region,
    "contact_between": _contact_between,
    "contact_any": _contact_any,
    "body_on": _body_on,
    "body_inside": _body_inside,
    "body_upright": _body_upright,
    "grasped": _grasped,
    # float-valued
    "distance_neg": _distance_neg,
    "joint_progress": _joint_progress,
    "constant": _constant,
}


def register_predicate(name: str, factory: PredicateFactory) -> None:
    """Register a user-defined predicate factory.

    Must be called before loading a spec that references ``name``. Factories
    registered at runtime are NOT sandboxed - by registering, you opt into
    running the factory with kwargs parsed from the spec. Only register
    predicates from trusted code paths; anything LLM-authored should use the
    built-in DSL exclusively.

    Args:
        name: Predicate name used in spec files. Must not shadow a built-in.
        factory: Callable that takes DSL kwargs and returns a predicate
            ``(sim) -> bool`` or reward term ``(sim) -> float``.

    Raises:
        ValueError: If ``name`` shadows a built-in predicate.
        TypeError: If ``factory`` is not callable.
    """
    if name in PREDICATE_REGISTRY:
        raise ValueError(f"register_predicate: '{name}' shadows a built-in predicate; pick a different name")
    if not callable(factory):
        raise TypeError(f"register_predicate: factory must be callable, got {type(factory).__name__}")
    PREDICATE_REGISTRY[name] = factory


def make_predicate(name: str, **kwargs: Any) -> Callable[[SimEngine], Any]:
    """Instantiate a predicate from its name + kwargs.

    This is the single entry point the DSL loader uses - it never touches
    ``eval`` or ``exec``. Unknown names produce a ``ValueError`` listing
    the valid set; bad kwargs surface as whatever ``TypeError`` the factory
    raises.

    Args:
        name: Predicate name. Must be registered in :data:`PREDICATE_REGISTRY`.
        **kwargs: Forwarded verbatim to the factory.

    Returns:
        A callable ``(sim) -> bool`` or ``(sim) -> float`` depending on the
        predicate.

    Raises:
        ValueError: If ``name`` is unknown.
        TypeError: If required factory kwargs are missing.
    """
    factory = PREDICATE_REGISTRY.get(name)
    if factory is None:
        valid = sorted(PREDICATE_REGISTRY.keys())
        raise ValueError(f"Unknown predicate '{name}'. Valid: {valid}")
    return factory(**kwargs)


__all__ = [
    "PREDICATE_REGISTRY",
    "BoolPredicate",
    "PredicateFactory",
    "RewardTerm",
    "make_predicate",
    "register_predicate",
]
