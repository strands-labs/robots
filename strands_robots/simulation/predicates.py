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

When the backend *does* support a lookup but the referenced ``body`` /
``joint`` name cannot be resolved (almost always a spec typo), the term still
degrades to a constant (``False`` / ``0.0``) but the offending name is logged
once at ``WARNING`` (see :func:`_warn_unresolved`), so a broken spec surfaces
instead of silently preventing episode success or emitting a dead reward.

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
    base_velocity(vx=0.0, vy=0.0, wz=0.0, weight=1.0, robot=None)
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


# Names the DSL has already warned about, so a broken spec cannot spam the
# reward/eval hot loop. Keyed by (kind, name); process-global and deduplicated.
_RESOLUTION_WARNED: set[tuple[str, str]] = set()


def _warn_unresolved(kind: str, name: str, tried: tuple[str, ...] = ()) -> None:
    """Warn once that a spec references an entity the sim cannot resolve.

    Called from the body/joint lookup helpers only when the backend *supports*
    the lookup (``get_body_state`` / ``get_observation`` present) but the named
    ``body``/``joint`` is not found - almost always a spec typo. The offending
    term then degrades to a constant (a bool predicate to ``False``, a reward
    term to ``0.0``), which silently prevents episode success or yields a dead,
    return-inflating reward. Surfacing the name once turns that silent
    corruption into an actionable log line without changing any returned value.
    A missing lookup *method* (unsupported backend) is a capability gap, not a
    typo, and stays silent.
    """
    key = (kind, name)
    if key in _RESOLUTION_WARNED:
        return
    _RESOLUTION_WARNED.add(key)
    extra = f" (tried {list(tried)})" if len(tried) > 1 else ""
    logger.warning(
        "predicate/reward DSL: %s %r is not present in the simulation%s; the "
        "referencing term degrades to a constant (bool predicate -> False, "
        "reward -> 0.0), which silently prevents success / yields a dead reward. "
        "Check the name against the loaded scene / benchmark spec.",
        kind,
        name,
        extra,
    )


def _reset_resolution_warnings() -> None:
    """Clear the one-time-warning dedup cache (test isolation)."""
    _RESOLUTION_WARNED.clear()


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
    the bare lookup fails. #176 (sub-task 3d) - without this
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
    tried = [body]
    if not body.endswith("_main"):
        tried.append(f"{body}_main")
        pos = _try(f"{body}_main")
        if pos is not None:
            return pos
    _warn_unresolved("body", body, tuple(tried))
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
    # The backend produced an observation but this joint is not in it: almost
    # always a spec typo (an empty obs is a backend/capability gap, not a name
    # error, so stay silent there).
    if obs and joint not in obs:
        _warn_unresolved("joint", joint)
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

    def _try(name: str) -> list[float] | None:
        try:
            result = get_body_state(body_name=name)
        except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
            logger.debug("body_quaternion(%r) failed: %s", name, e)
            return None
        if not isinstance(result, dict) or result.get("status") != "success":
            return None
        payload = _extract_json(result)
        quat = payload.get("quaternion")
        if isinstance(quat, list) and len(quat) == 4 and all(isinstance(c, (int, float)) for c in quat):
            return [float(c) for c in quat]
        return None

    # Mirror _body_position's resolution: bare BDDL name first, then the LIBERO
    # ``<name>_main`` root-body convention (#176). Without the fallback,
    # body_upright(<bddl_name>) resolved to None -> silently False for every
    # procedurally-generated LIBERO object, whose MJCF root body is _main-suffixed.
    quat = _try(body)
    if quat is not None:
        return quat
    tried = [body]
    if not body.endswith("_main"):
        tried.append(f"{body}_main")
        quat = _try(f"{body}_main")
        if quat is not None:
            return quat
    _warn_unresolved("body", body, tuple(tried))
    return None


def _euclidean_distance(a: list[float], b: list[float]) -> float:
    """Simple 3D Euclidean distance; no numpy so predicates stay dependency-free."""
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return float((dx * dx + dy * dy + dz * dz) ** 0.5)


def _quat_rotate_inverse_wxyz(quat_wxyz: list[float], vec: list[float]) -> list[float]:
    """Express a WORLD-frame 3-vector in the body frame given a (w,x,y,z) quaternion.

    Computes ``R(q)^T @ vec`` - the standard "rotate by the inverse". Pure Python
    (no numpy) so predicates stay dependency-free. A near-zero-norm quaternion
    returns ``vec`` unchanged. Matches the Newton backend's
    ``_quat_rotate_inverse_wxyz`` used to body-frame the base angular velocity.
    """
    w, x, y, z = (float(c) for c in quat_wxyz)
    norm = (w * w + x * x + y * y + z * z) ** 0.5
    if norm < 1e-8:
        return [float(v) for v in vec]
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    vx, vy, vz = (float(c) for c in vec)
    two_w = 2.0 * w
    s = 2.0 * w * w - 1.0
    # b = cross(q_vec, v); term = v * s - b * 2w + q_vec * (q_vec . v) * 2
    cx = y * vz - z * vy
    cy = z * vx - x * vz
    cz = x * vy - y * vx
    d = 2.0 * (x * vx + y * vy + z * vz)
    return [
        vx * s - cx * two_w + x * d,
        vy * s - cy * two_w + y * d,
        vz * s - cz * two_w + z * d,
    ]


def _base_twist(sim: SimEngine, robot: str | None) -> tuple[float, float, float] | None:
    """Return a floating base's BODY-frame planar twist ``(vx, vy, wz)``, or None.

    Reads ``get_observation``'s floating-base signals: ``base_lin_vel`` (world
    frame) is rotated into the base frame via ``base_quat`` so ``vx``/``vy`` are
    the forward/lateral velocity in the robot's own heading; ``base_ang_vel`` is
    already body-frame (the IMU-gyro convention on both backends) so its z
    component is the yaw rate directly. This is the frame a locomotion velocity
    command is expressed against (IsaacLab / legged_gym convention). Returns None
    (and warns once) when the robot exposes no floating base - almost always a
    spec referencing ``base_velocity`` on a fixed-base arm.
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_velocity get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    lin = obs.get("base_lin_vel")
    quat = obs.get("base_quat")
    ang = obs.get("base_ang_vel")
    if not (
        isinstance(lin, list)
        and len(lin) == 3
        and isinstance(quat, list)
        and len(quat) == 4
        and isinstance(ang, list)
        and len(ang) == 3
    ):
        # A floating base surfaces all three; their absence means this robot has
        # no floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    v_body = _quat_rotate_inverse_wxyz(quat, lin)
    return float(v_body[0]), float(v_body[1]), float(ang[2])


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
    contact between A and B via ``sim.get_contacts()`` - matches
    upstream LIBERO's ``ObjectState.check_ontop`` which combines a
    geometric check with ``check_contact``. The z-offset parameter
    accounts for B's half-height + a small buffer; tune per scene.
    Intended for sparse-success benchmarks (LIBERO, etc.) where exact
    geometric containment isn't required.

    Contact-check graceful degradation: when
    ``require_contact=True`` but the sim engine doesn't expose
    ``get_contacts`` (e.g. test stubs, custom engines), the contact
    check is skipped and only the geometric check fires. This
    preserves backwards compatibility - engines without contact
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


def _geom_belongs_to_body(geom: str, body: str) -> bool:
    """True when geom name ``geom`` is one of ``body``'s geoms.

    Handles the geom-naming conventions across the supported scene sources:

    - exact ``body`` (single-geom scenes whose geom is named after the body),
    - ``<body>_geom`` (strands :meth:`add_object`), and
    - ``<body>_g<idx>`` (LIBERO / robosuite multi-geom objects).

    The ``<body>_g`` prefix subsumes both ``<body>_geom`` and ``<body>_g<idx>``;
    it mirrors the prefix :func:`_body_contact` uses so contact-based
    predicates agree on what counts as a body's geom. The ``_g`` boundary
    keeps distinct names apart (``cube_1_g`` does not match ``cube_10_g0``).
    """
    return geom == body or geom.startswith(f"{body}_g")


def _grasped(body: str, gripper_prefix: str) -> BoolPredicate:
    """True when ``body`` is in contact with any geom whose name starts with ``gripper_prefix``.

    Treats the gripper as a *set* of geoms (fingers, pads, tip sites) so
    the caller only has to specify the common prefix - e.g. ``"robot0_gripper"``
    for Panda covers both fingers. A body is "grasped" as long as any one
    gripper geom is in contact with any geom belonging to ``body``.

    Body-geom matching follows the same naming conventions as
    :func:`_body_contact`, so ``grasped`` fires on real LIBERO/robosuite
    scenes (where a BDDL object ``cube_1`` owns collision geoms
    ``cube_1_g0`` / ``cube_1_g1`` ...) as well as on strands-native
    ``add_object`` scenes (``<body>_geom``) and single-geom scenes whose
    geom is named exactly after the body. Previously only the exact
    ``body`` / ``<body>_geom`` names matched, so ``(grasped cube_1)`` BDDL
    goals silently never fired on LIBERO scenes.

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
            # One side must be a geom of the grasped body; the other must
            # start with the gripper prefix. Match the body's geoms across
            # the naming conventions in play: an exact ``body`` name, the
            # strands ``add_object`` ``<body>_geom`` name, and the
            # LIBERO/robosuite ``<body>_g<idx>`` multi-geom convention
            # (``<body>_geom`` is itself covered by the ``<body>_g`` prefix).
            # This mirrors :func:`_body_contact`'s prefix matching so a
            # LIBERO ``(grasped cube_1)`` goal fires on ``cube_1_g0`` etc.
            body_match = _geom_belongs_to_body(g1, body) or _geom_belongs_to_body(g2, body)
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


def _base_velocity(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    weight: float = 1.0,
    robot: str | None = None,
) -> RewardTerm:
    """Negative base velocity-tracking error - the canonical locomotion reward.

    Rewards a floating-base robot for matching a commanded BODY-frame velocity
    ``(vx, vy, wz)``: ``vx`` forward, ``vy`` lateral (both in the robot's own
    heading, m/s) and ``wz`` the yaw rate (rad/s). The reward is
    ``-weight * ||(v_body_x, v_body_y, w_body_z) - (vx, vy, wz)||`` so it is 0 at
    perfect tracking and grows more negative with error - a dense, monotonic
    signal for a velocity-tracking / locomotion task (G1, Go2, T1, mobile bases),
    directly composable in a :class:`DeclarativeBenchmark` spec or an RL
    ``SimEnv`` reward.

    Reads the floating-base twist from ``get_observation``: ``base_lin_vel``
    (world frame) is rotated into the base frame via ``base_quat``, and
    ``base_ang_vel`` is already body-frame, so the tracked quantity is
    heading-relative (walking "forward at vx" tracks the robot's own +x, not a
    fixed world axis). Requires a robot with a floating base; a fixed-base arm
    has no base twist, so the term degrades to ``0.0`` and the missing base is
    logged once. ``robot`` selects the robot in a multi-robot scene (default:
    the sole robot).
    """
    w = float(weight)
    tvx, tvy, twz = float(vx), float(vy), float(wz)
    rname = robot

    def term(sim: SimEngine) -> float:
        twist = _base_twist(sim, rname)
        if twist is None:
            return 0.0
        bvx, bvy, bwz = twist
        dvx, dvy, dwz = bvx - tvx, bvy - tvy, bwz - twz
        return -w * float((dvx * dvx + dvy * dvy + dwz * dwz) ** 0.5)

    return term


# Stateful reward terms (declarative phase machine)
#
# A plain RewardTerm is stateless: ``(SimEngine) -> float``. Some rewards need
# memory across steps - a pick-place curriculum advances Reach -> Grasp ->
# Transport -> Place, awards a one-time bonus on each transition, and only ever
# moves forward. Rather than hardcode any specific task, we expose ONE
# generic primitive, ``staged_reward``, that composes EXISTING registry
# predicates into a phase machine. The task itself is then authored as data
# (a spec dict / YAML) by a human or LLM - never as shipped code, and never via
# ``eval`` (sub-predicates are compiled through :func:`make_predicate`, the same
# closed-registry path as every other DSL call).


class StatefulRewardTerm:
    """A reward term that carries per-episode state and must be ``reset()``.

    Duck-typed by consumers: anything with ``__call__(sim) -> float`` AND a
    zero-arg ``reset()`` is treated as episode-stateful. ``SimEnv.reset`` and
    ``DeclarativeBenchmark.on_episode_start`` call ``reset()`` on any reward
    term that has it, so stateless plain-function terms are unaffected.
    """

    def __call__(self, sim: SimEngine) -> float:  # pragma: no cover - interface
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class _StagedReward(StatefulRewardTerm):
    """Monotonic multi-stage (phase-machine) reward built from sub-predicates.

    Each stage declares:
        - ``reward``: a float-valued registry predicate giving the dense
          shaping signal while the machine is IN that stage.
        - ``advance_when``: a bool-valued registry predicate; the FIRST step it
          returns True the machine awards ``bonus`` once and advances to the
          next stage. Phases only ever move forward (no regression), matching
          curriculum semantics and giving a stable, non-oscillating signal.
        - ``bonus``: a one-time scalar added on the transition out of the stage
          (default 0.0).

    The last stage has no ``advance_when`` gate (the task is "done" there for
    reward purposes; episode termination is a separate ``success`` predicate).
    Per step the emitted reward is ``current_stage.reward(sim) +
    (bonus if this step advanced else 0.0)``.
    """

    def __init__(
        self,
        stages: list[tuple[RewardTerm, BoolPredicate | None, float]],
    ) -> None:
        self._stages = stages
        self._phase = 0

    def reset(self) -> None:
        self._phase = 0

    @property
    def phase(self) -> int:
        """Current stage index (0-based). Exposed for logging / tests."""
        return self._phase

    def __call__(self, sim: SimEngine) -> float:
        if not self._stages:
            return 0.0
        phase = min(self._phase, len(self._stages) - 1)
        reward_fn, advance_fn, bonus = self._stages[phase]
        r = float(reward_fn(sim))
        # Advance (and award the one-time bonus) only if there IS a next stage
        # and this stage declares a gate that now fires.
        if self._phase < len(self._stages) - 1 and advance_fn is not None and bool(advance_fn(sim)):
            self._phase += 1
            return r + float(bonus)
        return r


def _staged_reward(stages: list[Any]) -> RewardTerm:
    """Factory: compile a declared stage list into a :class:`_StagedReward`.

    This is the single new primitive that turns the stateless DSL into a
    declarative phase machine. It recursively compiles each stage's ``reward``
    and ``advance_when`` through :func:`make_predicate`, so the whole thing
    stays inside the closed-registry / no-``eval`` safety contract: a spec can
    only ever reference predicates that already exist in the registry.

    Args:
        stages: Ordered list of stage dicts. Each stage::

            {
                "reward": {"predicate": <float-term name>, **kwargs},
                "advance_when": {"predicate": <bool-pred name>, **kwargs},  # omit on last stage
                "bonus": <float>,   # optional, default 0.0
            }

    Returns:
        A callable+resettable :class:`_StagedReward`.

    Raises:
        ValueError: stages is not a non-empty list, a stage is malformed, a
            non-final stage omits ``advance_when``, or ``bonus`` is non-numeric.
        TypeError: surfaced from :func:`make_predicate` for bad sub-kwargs.
    """
    if not isinstance(stages, list) or not stages:
        raise ValueError("staged_reward: 'stages' must be a non-empty list of stage dicts")

    compiled: list[tuple[RewardTerm, BoolPredicate | None, float]] = []
    n = len(stages)
    for i, stage in enumerate(stages):
        if not isinstance(stage, dict):
            raise ValueError(f"staged_reward: stage[{i}] must be a dict, got {type(stage).__name__}")
        unknown = set(stage.keys()) - {"reward", "advance_when", "bonus"}
        if unknown:
            raise ValueError(
                f"staged_reward: stage[{i}] has unknown keys {sorted(unknown)}; allowed: reward, advance_when, bonus"
            )

        reward_call = stage.get("reward")
        if not isinstance(reward_call, dict) or "predicate" not in reward_call:
            raise ValueError(
                f"staged_reward: stage[{i}].reward must be a predicate-call dict "
                "like {predicate: distance_neg, body_a: ..., body_b: ...}"
            )
        reward_name = reward_call["predicate"]
        reward_kwargs = {k: v for k, v in reward_call.items() if k != "predicate"}
        reward_fn = make_predicate(reward_name, **reward_kwargs)

        advance_call = stage.get("advance_when")
        advance_fn: BoolPredicate | None
        if advance_call is None:
            if i != n - 1:
                raise ValueError(
                    f"staged_reward: stage[{i}] is not the final stage and must declare "
                    "'advance_when' (a bool predicate gating the transition to the next stage)"
                )
            advance_fn = None
        else:
            if not isinstance(advance_call, dict) or "predicate" not in advance_call:
                raise ValueError(
                    f"staged_reward: stage[{i}].advance_when must be a predicate-call dict "
                    "like {predicate: distance_less_than, body_a: ..., body_b: ..., threshold: ...}"
                )
            advance_name = advance_call["predicate"]
            advance_kwargs = {k: v for k, v in advance_call.items() if k != "predicate"}
            advance_fn = make_predicate(advance_name, **advance_kwargs)

        bonus_raw = stage.get("bonus", 0.0)
        if isinstance(bonus_raw, bool) or not isinstance(bonus_raw, (int, float)):
            raise ValueError(f"staged_reward: stage[{i}].bonus must be a number, got {bonus_raw!r}")

        compiled.append((reward_fn, advance_fn, float(bonus_raw)))

    return _StagedReward(compiled)


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
    "base_velocity": _base_velocity,
    "constant": _constant,
    # stateful (phase machine)
    "staged_reward": _staged_reward,
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
    "StatefulRewardTerm",
    "make_predicate",
    "register_predicate",
]
