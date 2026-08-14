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

Contact predicates count a geom pair only when the physics engine reports it
as a real touch. ``get_contacts`` also lists pairs inside the detection range
that carry no force -- see :func:`contact_is_active` -- and counting those
would make ``contact_any`` / ``contact_between`` / ``grasped`` /
``body_on(require_contact=True)`` fire for bodies that are visibly apart.

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
    body_on(body_a, body_b, z_offset=0.02, xy_tol=0.15, require_contact=False)
    body_inside(body, container, xy_tol=0.15, z_tol=0.15)
    body_upright(body, tol=0.15)
    grasped(body, gripper_prefix)
    base_tipped(tol=0.15, robot=None)
    base_below_z(z, robot=None)
    base_beyond_x(x, robot=None)
    base_beyond_y(y, robot=None)
    base_yaw_beyond(yaw, robot=None)

Available reward terms (float):

    distance_neg(body_a, body_b, weight=1.0)
    joint_progress(joint, target, weight=1.0)
    base_velocity(vx=0.0, vy=0.0, wz=0.0, weight=1.0, robot=None)
    base_velocity_tracking(vx=0.0, vy=0.0, wz=0.0, lin_weight=1.0, ang_weight=0.5, tracking_sigma=0.25, robot=None)
    base_height(target, weight=1.0, robot=None)
    base_orientation(weight=1.0, robot=None)
    base_lin_vel_z(weight=1.0, robot=None)
    base_ang_vel_xy(weight=1.0, robot=None)
    staged_reward(stages)
    constant(value)

Register custom predicates with :func:`register_predicate`.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

from strands_robots.utils import finite_number_error, finite_vector_error

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
    """Return a backend result's payload mapping, or ``{}`` if there is none.

    Every in-tree backend returns the agent-tool envelope, so the payload is
    the ``json`` content block -- see
    :meth:`strands_robots.simulation.base.SimEngine.get_contacts` for the
    shape. A result that carries no ``content`` at all is not an envelope, so
    the mapping itself is the payload: a minimal engine can return a plain
    reading without wrapping it, and every predicate reads both shapes the
    same way instead of each one guessing.
    """
    if not isinstance(result, dict):
        return {}
    if "content" not in result:
        # Not an envelope - the mapping is the payload.
        return dict(result)
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


def _base_body_velocity(sim: SimEngine, robot: str | None) -> tuple[list[float], list[float]] | None:
    """Return a floating base's BODY-frame ``(linear_velocity, angular_velocity)``, or None.

    Reads ``get_observation``'s floating-base signals and expresses BOTH twists
    in the base (body) frame - the frame a legged controller regularizes: the
    world-frame ``base_lin_vel`` is rotated into the base frame via ``base_quat``
    (so its z is the vertical velocity along the base's OWN up-axis), and
    ``base_ang_vel`` is already body-frame (the IMU-gyro convention on both
    backends) so its xy are the roll/pitch rates directly. Returns None (and
    warns once) when the robot exposes no floating base - almost always a spec
    referencing a base term on a fixed-base arm. Shared by the two uncommanded-
    base-velocity regularizer terms (``base_lin_vel_z`` / ``base_ang_vel_xy``).
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base motion get_observation(%r) failed: %s", robot, e)
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
    return (
        [float(v_body[0]), float(v_body[1]), float(v_body[2])],
        [float(ang[0]), float(ang[1]), float(ang[2])],
    )


def _base_position(sim: SimEngine, robot: str | None) -> list[float] | None:
    """Return a floating base's WORLD position ``[x, y, z]`` (incl. height), or None.

    Reads ``get_observation``'s ``base_pos`` floating-base signal - the base
    body's world-frame position, whose z component is the base height a
    locomotion controller regularizes (torso/pelvis height above the ground).
    Returns None (and warns once) when the robot exposes no floating base -
    almost always a spec referencing a base term on a fixed-base arm.
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_height get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    pos = obs.get("base_pos")
    if not (isinstance(pos, list) and len(pos) == 3):
        # A floating base surfaces base_pos; its absence means this robot has no
        # floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    return [float(pos[0]), float(pos[1]), float(pos[2])]


def _base_quaternion(sim: SimEngine, robot: str | None) -> list[float] | None:
    """Return a floating base's orientation quaternion ``[w, x, y, z]``, or None.

    Reads ``get_observation``'s ``base_quat`` floating-base signal (MuJoCo and
    Newton both report ``[w, x, y, z]``). Returns None (and warns once) when the
    robot exposes no floating base (a fixed-base arm) - almost always a spec
    referencing a base term on a robot that has no base orientation.
    """
    try:
        obs = sim.get_observation(robot_name=robot, skip_images=True)
    except Exception as e:  # noqa: BLE001 - defensive: predicates never raise
        logger.debug("base_orientation get_observation(%r) failed: %s", robot, e)
        return None
    if not isinstance(obs, dict):
        return None
    quat = obs.get("base_quat")
    if not (isinstance(quat, list) and len(quat) == 4):
        # A floating base surfaces base_quat; its absence means this robot has
        # no floating base (a fixed-base arm) - almost always a spec error.
        _warn_unresolved("robot base", robot or "<sole robot>")
        return None
    return [float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3])]


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


def contact_is_active(record: Mapping[str, Any]) -> bool:
    """True when a ``get_contacts`` record is a touch rather than a near miss.

    ``get_contacts`` reports every geom pair inside the *detection* range,
    which is the pair's ``margin`` plus its ``gap``. Only the pairs inside
    ``margin`` reach the constraint solver; a pair between the two thresholds
    carries no force, so treating it as contact answers "touching" for bodies
    that are visibly apart. Assets ship a non-zero ``margin``/``gap`` -- a
    foot geom fractions of a millimetre off the floor is the usual case -- so
    this is the difference between a locomotion or placement clause firing on
    physics and firing on proximity.

    ``dist`` cannot stand in for this: a pair with a wide ``margin`` is
    load-bearing at a *positive* distance (the body hovers on a real force),
    while the near miss above is also positive, so the sign of ``dist`` splits
    neither case. The solver's own admission decision does, and
    ``get_contacts`` reports it as ``active``.

    This is the single owner of that reading, so every contact predicate
    agrees about which records count -- the same role
    :func:`_geom_belongs_to_body` plays for body-to-geom names.

    A record that does not report ``active`` is treated as a touch, so a
    payload from a backend or test stub that cannot make the distinction keeps
    its previous verdict rather than silently answering ``False`` for every
    contact.
    """
    flag = record.get("active")
    if flag is None:
        return True
    return bool(flag)


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
            if not isinstance(c, dict) or not contact_is_active(c):
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
        contacts = payload.get("contacts")
        if isinstance(contacts, list):
            # The per-record list wins over a bare count: only a record
            # carries the touch/proximity flag, which ``n_contacts`` cannot
            # express. The count is the fallback for payloads that report
            # nothing else.
            return any(isinstance(c, dict) and contact_is_active(c) for c in contacts)
        return bool(payload.get("n_contacts", 0) > 0)

    return check


def _geom_belongs_to_body(geom: str, body: str) -> bool:
    """True when geom name ``geom`` is one of ``body``'s geoms.

    This is the single owner of the body-to-geom name mapping: every
    body-level contact predicate resolves through it, so ``grasped`` and
    ``body_on(require_contact=True)`` cannot disagree about whether a
    reported contact belongs to a body.

    Handles the geom-naming conventions across the supported scene sources:

    - exact ``body`` (single-geom scenes whose geom is named after the body),
    - ``<body>_geom`` (strands :meth:`add_object`),
    - ``<body>_g<idx>`` (LIBERO / robosuite multi-geom objects), and
    - ``<body>/geom_<id>`` - the name ``get_contacts`` **synthesizes** for a
      geom the asset left unnamed, which is the dominant case in real MJCF
      (a Panda scene has 81 unnamed geoms out of 82).

    The ``<body>_g`` prefix subsumes both ``<body>_geom`` and ``<body>_g<idx>``;
    the ``_g`` boundary keeps distinct names apart (``cube_1_g`` does not match
    ``cube_10_g0``). The synthesized form is matched exactly rather than as a
    ``<body>/`` prefix, because bodies are themselves namespaced: a broad
    prefix would let body ``panda`` claim ``panda/link0/geom_1``, which belongs
    to ``panda/link0``.
    """
    if geom == body or geom.startswith(f"{body}_g"):
        return True
    # ``get_contacts`` names an unnamed geom after its parent body as
    # ``<body>/geom_<id>``; anything else after ``<body>/`` is a child body.
    suffix = geom.removeprefix(f"{body}/geom_")
    return suffix != geom and suffix.isdigit()


def _body_contact(sim: SimEngine, body_a: str, body_b: str) -> bool | None:
    """Best-effort body-contact lookup.

    Returns ``True`` / ``False`` when ``sim.get_contacts()`` is available
    AND any geom of ``body_a`` is in contact with any geom of ``body_b``.
    Returns ``None`` when ``get_contacts()`` is unavailable so the
    caller can decide whether to gracefully degrade (fall back to
    geometric-only checks) or hard-fail.

    Body-geom matching is delegated to :func:`_geom_belongs_to_body`, which
    owns every supported geom-naming convention. This mirrors how upstream
    LIBERO's ``ObjectState.check_contact`` walks the per-object geom list, but
    avoids hard-coding the body→geom map by using the naming conventions.

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

    for c in contacts:
        if not isinstance(c, dict) or not contact_is_active(c):
            continue
        g1 = c.get("geom1") or ""
        g2 = c.get("geom2") or ""
        # Resolve both sides through the shared body-to-geom mapping so this
        # agrees with ``grasped``. Either direction (a-then-b or b-then-a)
        # counts.
        if (_geom_belongs_to_body(g1, body_a) and _geom_belongs_to_body(g2, body_b)) or (
            _geom_belongs_to_body(g1, body_b) and _geom_belongs_to_body(g2, body_a)
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
    support get the geometric-only verdict. LIBERO benchmarks running on
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
            # geometric-only verdict.
            # ``False`` ⇒ engine reports no contact ⇒ predicate False.
            # ``True`` ⇒ contact confirmed ⇒ predicate True (combined
            # with the passing geometric check above).
            if in_contact is False:
                return False
        return True

    return check


def _body_inside(body: str, container: str, xy_tol: float = 0.15, z_tol: float = 0.15) -> BoolPredicate:
    """Approximate ``(in A B)`` predicate - A contained within B's volume.

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
    gripper geom is in contact with any geom belonging to ``body``.

    Body-geom matching is delegated to :func:`_geom_belongs_to_body`, the
    shared owner of that mapping, so ``grasped`` fires on real LIBERO/robosuite
    scenes (where a BDDL object ``cube_1`` owns collision geoms
    ``cube_1_g0`` / ``cube_1_g1`` ...), on strands-native ``add_object`` scenes
    (``<body>_geom``), on single-geom scenes whose geom is named exactly after
    the body, and on assets whose geoms are unnamed.

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
            if not isinstance(c, dict) or not contact_is_active(c):
                continue
            g1 = c.get("geom1") or ""
            g2 = c.get("geom2") or ""
            # One side must be a geom of the grasped body; the other must
            # start with the gripper prefix. ``_geom_belongs_to_body`` owns
            # every geom-naming convention, so a LIBERO ``(grasped cube_1)``
            # goal fires on ``cube_1_g0`` and a scene with unnamed geoms
            # fires on the synthesized ``cube_1/geom_<id>`` name.
            body_match = _geom_belongs_to_body(g1, body) or _geom_belongs_to_body(g2, body)
            gripper_match = any(isinstance(g, str) and g.startswith(gripper_prefix) for g in (g1, g2))
            if body_match and gripper_match:
                return True
        return False

    return check


def _base_tipped(tol: float = 0.15, robot: str | None = None) -> BoolPredicate:
    """True when a floating base has tilted more than ``tol`` from level.

    The failure-clause counterpart of the ``base_orientation`` reward term: while
    ``base_orientation`` *penalises* tilt in ``dense_reward`` (a dense shaping
    signal), ``base_tipped`` *terminates* the episode when the base falls over -
    the canonical legged_gym / IsaacLab locomotion fall-over termination. Put it
    in a ``failure`` clause
    (``failure: {any: [{predicate: base_tipped, tol: 0.7}]}``) so a
    velocity-tracking rollout ends the instant the robot topples instead of
    flailing on the ground to ``max_steps``.

    Reads ``get_observation``'s ``base_quat`` - the same embodiment-agnostic
    floating-base surface the ``base_*`` reward terms read, so it needs no base
    body name (it works on a mobile base whose free joint is unnamed) and is
    identical across the MuJoCo and Newton backends, unlike ``body_upright``
    which resolves a specific body by name. The tilt test is the exact
    complement of ``body_upright``'s upright check applied to the base
    quaternion (MuJoCo/Newton layout ``(w, x, y, z)``) - the fall-over
    counterpart of ``body_upright``'s ``2*(x**2+y**2) < tol`` upright test:

        R[2,2] = 1 - 2 * (x ** 2 + y ** 2)  ->  tipped when 2 * (x ** 2 + y ** 2) > tol

    monotonic in how far the base is tipped, so ``tol`` is the maximum tilt the
    base may reach before it counts as fallen. ``tol`` shares ``body_upright``'s
    scale: ``2 * (x ** 2 + y ** 2) = 1 - cos(theta)`` for a roll/pitch of
    ``theta``, so ``tol=0.15`` trips at ~32 deg (a tight "leaning" bound) and
    ``tol=1.0`` trips at 90 deg (fully on its side) - a fall-over termination
    typically uses a larger ``tol`` (~0.7-1.0) than the default.

    Requires a robot with a floating base; a fixed-base arm has no base
    orientation, so the predicate degrades to ``False`` (never tipped) and the
    missing base is logged once. ``robot`` selects the robot in a multi-robot
    scene (default: the sole robot).
    """
    t = float(tol)
    if t < 0:
        raise ValueError(f"base_tipped: 'tol' must be >= 0, got {t}")
    rname = robot

    def check(sim: SimEngine) -> bool:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return False
        # base_quat layout is (w, x, y, z) on both backends.
        _, x, y, _ = quat
        return 2.0 * (x * x + y * y) > t

    return check


def _ground_height(sim: SimEngine, x: float, y: float) -> float:
    """Local terrain surface height (world z) beneath ``(x, y)``; ``0.0`` on flat ground.

    A height/fall predicate must measure a base's clearance above the ground
    *beneath it*, not an absolute world z: on a raised-terrain heightfield
    (``create_world(terrain=...)``) a collapsed robot on a plateau still has an
    absolute base z above a flat-ground threshold, so an absolute test silently
    misses the fall. Reads the backend's ``_ground_height_at`` hook (``0.0`` for
    flat ground / a backend with no heightfield); a sim lacking the hook
    degrades to ``0.0`` so flat-ground behaviour is unchanged and the predicate
    never raises.
    """
    fn = getattr(sim, "_ground_height_at", None)
    if fn is None:
        return 0.0
    try:
        return float(fn(x, y))
    except Exception as e:  # noqa: BLE001 - predicates never raise
        logger.debug("_ground_height_at(%.3f, %.3f) failed: %s", x, y, e)
        return 0.0


def _base_below_z(z: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's height ABOVE THE LOCAL GROUND drops below ``z``.

    The height counterpart of :func:`_base_tipped`, and the second half of a
    complete floating-base fall termination: ``base_tipped`` fires when the base
    *topples* (rolls/pitches off level) and ``base_below_z`` fires when the base
    *collapses* (its torso/pelvis sinks to the ground). Put both in a ``failure``
    clause so a velocity-tracking rollout ends the instant the robot either
    falls over OR drops to the floor, instead of flailing on the ground to
    ``max_steps``::

        failure:
          any:
            - {predicate: base_tipped, tol: 0.7}
            - {predicate: base_below_z, z: 0.3}

    Reads ``get_observation``'s ``base_pos`` - the same embodiment-agnostic
    floating-base surface the ``base_*`` reward terms and ``base_tipped`` read -
    so it needs no base body name and works on a mobile base whose free joint is
    unnamed, unlike ``body_below_z`` which resolves a specific body by name (a
    name a mobile base's unnamed free joint does not expose). It is the
    base-surface, name-free analogue of ``body_below_z(<base body>, z)``.

    The height is measured ABOVE THE LOCAL GROUND beneath the base, not as an
    absolute world z: on a raised-terrain heightfield (``create_world(terrain=
    ...)``, the terrain curriculum) a collapse onto a plateau leaves the base
    above a flat-ground threshold, so an absolute test would silently miss the
    fall. The local terrain height is subtracted (``0.0`` on a flat ground plane
    / a backend with no heightfield, so flat-ground behaviour is unchanged).

    ``z`` is the collapse clearance in metres (height of the base above the
    ground beneath it); a fall termination sets it well below the standing base
    height (a G1 pelvis stands ~0.74 m, so ``z=0.3`` catches a collapse). Requires a robot with a floating base; a fixed-base arm
    has no base position, so the predicate degrades to ``False`` (never
    collapsed -> never spuriously fails an episode) and the missing base is
    logged once. ``robot`` selects the robot in a multi-robot scene (default:
    the sole robot).
    """
    zt = float(z)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        # Height ABOVE THE LOCAL GROUND, not absolute world z: on raised terrain
        # (create_world(terrain=...)) a collapse leaves the base above a
        # flat-ground threshold, so an absolute test misses the fall. On flat
        # ground _ground_height is 0.0 and this reduces to the world height.
        return (pos[2] - _ground_height(sim, pos[0], pos[1])) < zt

    return check


def _base_beyond_x(x: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's world x-position has passed forward of ``x``.

    The forward-progress SUCCESS counterpart of the base-fall termination
    predicates (:func:`_base_tipped` / :func:`_base_below_z`): those FAIL a
    velocity-tracking episode when the base topples or collapses, while
    ``base_beyond_x`` SUCCEEDS it once the base has actually walked forward past
    ``x`` metres. It is the missing success half of a walk-forward locomotion
    task - the reward terms (``base_velocity_tracking`` + the base regularizers)
    shape *how* to walk, the fall predicates end a *bad* rollout, and this
    predicate scores the *goal* ("the base reached forward distance ``x``"). Put
    it in a ``success`` clause next to the fall predicates in ``failure``::

        success:
          all:
            - {predicate: base_beyond_x, x: 2.0}
        failure:
          any:
            - {predicate: base_tipped, tol: 0.7}
            - {predicate: base_below_z, z: 0.3}

    Reads ``get_observation``'s ``base_pos`` x (world frame) - the same
    embodiment-agnostic floating-base surface the ``base_*`` reward terms,
    ``base_tipped``, and ``base_below_z`` read - so it needs no base body name
    and works on a mobile base whose free joint is unnamed, unlike
    ``inside_region`` which resolves a specific body by name (a name a mobile
    base's unnamed free joint does not expose).

    ``x`` is an ABSOLUTE world x-threshold in metres, not a displacement: the
    canonical walk-forward scene spawns the base near the origin facing +x
    (identity orientation -> world +x is forward), so ``base_beyond_x(x=D)``
    reads "walked ~D metres forward". The author sets ``x`` relative to the
    known spawn x, exactly as ``base_below_z`` sets its collapse height relative
    to the known standing height; a base that starts already past ``x`` reads
    True immediately. It is a pure x-position test - height and orientation do
    not affect it (a base that walked forward but then toppled still reads True
    on x, so pair it with the fall predicates in a ``failure`` clause to reject
    that outcome).

    Requires a robot with a floating base; a fixed-base arm has no base
    position, so the predicate degrades to ``False`` (never made forward
    progress -> never spuriously succeeds) and the missing base is logged once.
    ``robot`` selects the robot in a multi-robot scene (default: the sole
    robot).
    """
    xt = float(x)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        return pos[0] > xt

    return check


def _base_beyond_y(y: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base's world y-position has passed beyond ``y``.

    The LATERAL-progress SUCCESS counterpart of :func:`_base_beyond_x`: where
    ``base_beyond_x`` scores a walk-FORWARD goal off ``base_pos`` x,
    ``base_beyond_y`` scores a strafe-LEFT goal off ``base_pos`` y (world +y is
    the robot's left for the identity spawn orientation the locomotion scenes
    use). It is the missing lateral half of the position-success family - the
    reward terms (``base_velocity_tracking`` with a non-zero ``vy`` command +
    the base regularizers) shape *how* to strafe, the fall predicates
    (``base_tipped`` / ``base_below_z``) end a *bad* rollout, and this predicate
    scores the *goal* ("the base reached lateral distance ``y``")::

        success:
          all:
            - {predicate: base_beyond_y, y: 1.0}
        failure:
          any:
            - {predicate: base_tipped, tol: 0.7}
            - {predicate: base_below_z, z: 0.18}

    Reads ``get_observation``'s ``base_pos`` y (world frame) - the same
    embodiment-agnostic floating-base surface the ``base_*`` reward terms,
    ``base_tipped``, ``base_below_z``, and ``base_beyond_x`` read - so it needs
    no base body name and works on a mobile base whose free joint is unnamed.

    ``y`` is an ABSOLUTE world y-threshold in metres, not a displacement
    (mirroring ``base_beyond_x``): the canonical locomotion scenes spawn the
    base near the origin, so ``base_beyond_y(y=D)`` reads "strafed ~D metres to
    the left". The author sets ``y`` relative to the known spawn y. It is a pure
    y-position test - height and orientation do not affect it (a base that
    strafed but then toppled still reads True on y, so pair it with the fall
    predicates in a ``failure`` clause to reject that outcome).

    Requires a robot with a floating base; a fixed-base arm has no base
    position, so the predicate degrades to ``False`` (never made lateral
    progress -> never spuriously succeeds) and the missing base is logged once.
    ``robot`` selects the robot in a multi-robot scene (default: the sole
    robot).
    """
    yt = float(y)
    rname = robot

    def check(sim: SimEngine) -> bool:
        pos = _base_position(sim, rname)
        if pos is None:
            return False
        return pos[1] > yt

    return check


def _base_yaw_beyond(yaw: float, robot: str | None = None) -> BoolPredicate:
    """True when a floating base has turned past a heading of ``yaw`` radians.

    The YAW (turn-in-place) SUCCESS counterpart of :func:`_base_beyond_x`
    (forward) and :func:`_base_beyond_y` (lateral): those score a *position*
    goal off ``base_pos``, while ``base_yaw_beyond`` scores a *heading* goal off
    ``base_quat``. It is the missing rotational third of the omnidirectional
    velocity-tracking vocabulary - the reward term ``base_velocity_tracking``
    already accepts a yaw-rate command ``wz``, but until now no predicate could
    SCORE reaching a turn goal, so a turn-in-place benchmark could reward a
    ``wz`` command yet had no terminal for "the base actually turned". It drops
    straight into a ``success`` clause next to the fall predicates in
    ``failure``::

        success:
          all:
            - {predicate: base_yaw_beyond, yaw: 1.0}
        failure:
          any:
            - {predicate: base_tipped, tol: 0.7}
            - {predicate: base_below_z, z: 0.18}

    The heading is the base's yaw about the world vertical, extracted from the
    ``base_quat`` (``w, x, y, z``) surface the ``base_*`` reward terms,
    ``base_tipped``, ``base_beyond_x`` and ``base_beyond_y`` read - so it needs
    no base body name and works on a mobile base whose free joint is unnamed::

        yaw = atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))

    ``yaw`` is an ABSOLUTE world heading in radians, single-signed and measured
    from the spawn heading (the locomotion scenes spawn at the identity
    orientation -> yaw 0), exactly as ``base_beyond_x`` reads an absolute world x
    from a +x-facing spawn: positive is a LEFT (counter-clockwise) turn, so
    ``base_yaw_beyond(yaw=1.0)`` reads "turned ~1 rad (~57 deg) to the left". A
    turn-in-place task targets a sub-pi heading (a single turn of less than a
    half-revolution); the yaw wraps at +-pi, so a goal at or beyond pi is not a
    well-defined single-turn heading (measuring cumulative revolutions would
    need integrated-angle tracking, out of scope here just as ``base_beyond_x``
    does not track total path length). It is a pure heading test - roll and
    pitch do NOT affect the yaw, so a base that merely tips (without turning
    about the vertical) never satisfies a yaw goal; position and height do not
    affect it either. Because the yaw of a fully toppled base is ill-defined,
    pair it with ``base_tipped`` in a ``failure`` clause so a "turned then fell"
    rollout is rejected on the tilt, not scored as a valid turn.

    Requires a robot with a floating base; a fixed-base arm has no base
    orientation, so the predicate degrades to ``False`` (never turned -> never
    spuriously succeeds) and the missing base is logged once. ``robot`` selects
    the robot in a multi-robot scene (default: the sole robot).
    """
    yt = float(yaw)
    rname = robot

    def check(sim: SimEngine) -> bool:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return False
        # base_quat layout is (w, x, y, z) on both backends.
        w, x, y, z = quat
        heading = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return heading > yt

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


def _base_velocity_tracking(
    vx: float = 0.0,
    vy: float = 0.0,
    wz: float = 0.0,
    lin_weight: float = 1.0,
    ang_weight: float = 0.5,
    tracking_sigma: float = 0.25,
    robot: str | None = None,
) -> RewardTerm:
    """Bounded exponential-kernel velocity-tracking reward - the canonical legged_gym primary locomotion reward.

    The standard legged_gym / IsaacLab velocity-tracking reward: a POSITIVE,
    BOUNDED signal that peaks when the base matches a commanded BODY-frame
    velocity ``(vx, vy, wz)`` (``vx`` forward, ``vy`` lateral in m/s, ``wz`` yaw
    rate in rad/s) and decays smoothly to 0 as the error grows::

        lin_weight * exp(-((v_body_x - vx)**2 + (v_body_y - vy)**2) / tracking_sigma)
      + ang_weight * exp(-(w_body_z - wz)**2 / tracking_sigma)

    the sum of legged_gym's ``tracking_lin_vel`` (planar velocity) and
    ``tracking_ang_vel`` (yaw rate) terms with their canonical default weights
    (1.0 / 0.5) and kernel width (``tracking_sigma`` 0.25). It is bounded to
    ``[0, lin_weight + ang_weight]`` and maximal (``lin_weight + ang_weight``) at
    perfect tracking.

    This is the exp-kernel counterpart of :func:`_base_velocity`, which is an
    UNBOUNDED negative-L2 error (``-weight * ||twist - command||``). The
    difference matters for RL: an unbounded negative error is dominated by the
    large initial tracking error and can swamp the bounded regularizer terms
    (``base_height`` / ``base_orientation`` / ``base_lin_vel_z`` /
    ``base_ang_vel_xy``), whereas this bounded kernel saturates near the command
    and stays well-scaled against those regularizers - the reason legged_gym /
    IsaacLab use an exponential kernel for the primary tracking reward and why a
    faithful velocity-tracking reward for #873 pairs it with those regularizers.
    It also weights planar-velocity tracking and yaw-rate tracking separately
    (``lin_weight`` / ``ang_weight``), which the single combined ``base_velocity``
    norm cannot express.

    Reads the floating-base twist from ``get_observation`` exactly like
    ``base_velocity``: ``base_lin_vel`` (world frame) is rotated into the base
    frame via ``base_quat`` and ``base_ang_vel`` is already body-frame, so the
    tracked quantity is heading-relative. Requires a robot with a floating base;
    a fixed-base arm has no base twist, so the term degrades to ``0.0`` and the
    missing base is logged once. ``robot`` selects the robot in a multi-robot
    scene (default: the sole robot).
    """
    tvx, tvy, twz = float(vx), float(vy), float(wz)
    lw, aw = float(lin_weight), float(ang_weight)
    sigma = float(tracking_sigma)
    if sigma <= 0:
        raise ValueError(f"base_velocity_tracking: 'tracking_sigma' must be > 0, got {sigma}")
    rname = robot

    def term(sim: SimEngine) -> float:
        twist = _base_twist(sim, rname)
        if twist is None:
            return 0.0
        bvx, bvy, bwz = twist
        lin_err = (bvx - tvx) ** 2 + (bvy - tvy) ** 2
        ang_err = (bwz - twz) ** 2
        return lw * math.exp(-lin_err / sigma) + aw * math.exp(-ang_err / sigma)

    return term


def _base_height(target: float, weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared base-height error - a locomotion-regularizer reward.

    Rewards a floating-base robot for keeping its base (torso/pelvis) near a
    target height ABOVE THE LOCAL GROUND beneath it:
    ``-weight * ((base_z - ground_z) - target) ** 2`` - 0 at the target and
    growing more negative as the base deviates. Composed alongside
    ``base_velocity`` in a ``dense_reward`` list, it is the standard regularizer
    that stops a velocity-tracking policy from cheating the forward-velocity
    reward by crouching or diving (the legged_gym / IsaacLab ``base_height``
    term). ``base_velocity`` alone is degenerate for locomotion - a policy can
    dive forward to maximise it - so a viable velocity-tracking reward pairs the
    two.

    The height is measured ABOVE THE LOCAL GROUND, not as an absolute world z:
    on a raised-terrain heightfield (``create_world(terrain=...)``, the terrain
    curriculum) a robot standing at its proper posture on a plateau has an
    absolute base z above the target, so an absolute test spuriously penalises
    it - worse, it makes the reward *reward crouching* to the flat-ground
    absolute height while on the plateau, inverting the anti-crouch incentive
    this term exists to provide. The local terrain height is subtracted (``0.0``
    on a flat ground plane / a backend with no heightfield, so flat-ground
    behaviour is byte-for-byte unchanged).

    Reads ``get_observation``'s ``base_pos`` (world frame). ``target`` is the
    desired base height in metres, measured above the ground beneath the base
    (task-specific: a G1 pelvis ~0.74 m, a Go2 trunk ~0.34 m). Requires a robot
    with a floating base; a fixed-base arm has no base position, so the term
    degrades to ``0.0`` and the missing base is logged once. ``robot`` selects
    the robot in a multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    tgt = float(target)
    rname = robot

    def term(sim: SimEngine) -> float:
        pos = _base_position(sim, rname)
        if pos is None:
            return 0.0
        # Height ABOVE THE LOCAL GROUND, not absolute world z: on raised terrain
        # (create_world(terrain=...)) a robot standing at its target posture on
        # a plateau has an absolute base z above the target, so an absolute test
        # spuriously penalises it (and rewards a crouch on the plateau). On flat
        # ground _ground_height is 0.0 and this reduces to the world height.
        d = (pos[2] - _ground_height(sim, pos[0], pos[1])) - tgt
        return -w * d * d

    return term


def _base_orientation(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative flat-orientation error - the locomotion base-orientation regularizer.

    Penalises a floating-base robot for tilting (roll/pitch) away from level:
    ``-weight * (g_x ** 2 + g_y ** 2)`` where ``(g_x, g_y, g_z)`` is the world
    gravity direction expressed in the base frame (the "projected gravity" a
    legged controller reads). When the base is level the projected gravity is
    ``(0, 0, -1)`` so the penalty is 0; a roll or pitch of ``theta`` makes the xy
    magnitude ``sin(theta)`` so the penalty grows as ``-weight * sin(theta) ** 2``.
    This is the standard legged_gym / IsaacLab ``orientation`` term and the third
    piece of a minimal velocity-tracking reward: ``base_velocity`` alone is
    degenerate (a policy can crouch OR lean to cheat the forward-velocity
    reward), so a viable locomotion reward pairs it with ``base_height`` (which
    stops crouch-cheating) AND ``base_orientation`` (which stops lean/tilt-
    cheating) in one ``dense_reward`` list.

    Crucially the penalty is invariant to YAW (heading): a robot may turn freely
    while walking upright, only roll/pitch off level is penalised. Reads
    ``get_observation``'s ``base_quat`` (``[w, x, y, z]`` on both backends).
    Requires a robot with a floating base; a fixed-base arm has no base
    orientation, so the term degrades to ``0.0`` and the missing base is logged
    once. ``robot`` selects the robot in a multi-robot scene (default: the sole
    robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        quat = _base_quaternion(sim, rname)
        if quat is None:
            return 0.0
        gx, gy, _gz = _quat_rotate_inverse_wxyz(quat, [0.0, 0.0, -1.0])
        return -w * float(gx * gx + gy * gy)

    return term


def _base_lin_vel_z(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared vertical base velocity - a locomotion-regularizer reward.

    Penalises a floating-base robot for moving vertically (bouncing):
    ``-weight * v_body_z ** 2`` where ``v_body_z`` is the base's linear velocity
    along its OWN up-axis (the world-frame ``base_lin_vel`` rotated into the base
    frame via ``base_quat``). 0 when the base holds a constant height, growing
    more negative the faster it bounces. This is the standard legged_gym /
    IsaacLab ``lin_vel_z`` term, and it complements ``base_height``:
    ``base_height`` penalises a base-height OFFSET (a static crouch) while
    ``base_lin_vel_z`` directly damps vertical bouncing whose mean height error
    can be ~0 - a velocity-tracking policy that porpoises around the target
    height is caught by this term but not by ``base_height`` alone. One of the
    two uncommanded-base-velocity regularizers (with ``base_ang_vel_xy``) that a
    viable velocity-tracking reward adds to ``base_velocity`` + ``base_height`` +
    ``base_orientation`` (the default-nonzero legged_gym reward set).

    Reads ``get_observation``'s ``base_lin_vel`` + ``base_quat``. Requires a
    robot with a floating base; a fixed-base arm has no base velocity, so the
    term degrades to ``0.0`` and the missing base is logged once. ``robot``
    selects the robot in a multi-robot scene (default: the sole robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        motion = _base_body_velocity(sim, rname)
        if motion is None:
            return 0.0
        vz = motion[0][2]
        return -w * vz * vz

    return term


def _base_ang_vel_xy(weight: float = 1.0, robot: str | None = None) -> RewardTerm:
    """Negative squared roll/pitch angular velocity - a locomotion-regularizer reward.

    Penalises a floating-base robot for rolling/pitching (wobbling):
    ``-weight * (w_body_x ** 2 + w_body_y ** 2)`` where ``(w_body_x, w_body_y)``
    are the base's roll and pitch RATES (body-frame ``base_ang_vel``, the
    IMU-gyro reading). 0 when the base is not tipping, growing more negative the
    faster it wobbles. This is the standard legged_gym / IsaacLab ``ang_vel_xy``
    term, and it complements ``base_orientation``: ``base_orientation`` penalises
    a tilt OFFSET (a static lean) while ``base_ang_vel_xy`` directly damps
    oscillatory roll/pitch whose mean tilt can be ~0. Crucially it is INVARIANT
    to the yaw rate (``w_body_z``): a walking policy may turn freely, only
    roll/pitch RATE is penalised. One of the two uncommanded-base-velocity
    regularizers (with ``base_lin_vel_z``) that a viable velocity-tracking reward
    adds to ``base_velocity`` + ``base_height`` + ``base_orientation`` (the
    default-nonzero legged_gym reward set).

    Reads ``get_observation``'s ``base_ang_vel`` (already body-frame on both
    backends). Requires a robot with a floating base; a fixed-base arm has no
    base velocity, so the term degrades to ``0.0`` and the missing base is
    logged once. ``robot`` selects the robot in a multi-robot scene (default:
    the sole robot).
    """
    w = float(weight)
    rname = robot

    def term(sim: SimEngine) -> float:
        motion = _base_body_velocity(sim, rname)
        if motion is None:
            return 0.0
        wx, wy = motion[1][0], motion[1][1]
        return -w * (wx * wx + wy * wy)

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
        """Score the current sim state, returning this step's reward contribution."""
        raise NotImplementedError

    def reset(self) -> None:  # pragma: no cover - interface
        """Clear per-episode state at an episode boundary.

        Called by :meth:`SimEnv.reset` and
        :meth:`DeclarativeBenchmark.on_episode_start` before a new episode so
        accumulated progress (phase counters, best-so-far distances, latched
        successes) does not leak across episodes.
        """
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
            non-final stage omits ``advance_when``, or ``bonus`` is not a
            finite number.
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
            if isinstance(advance_name, str) and predicate_kind(advance_name) == "float":
                raise ValueError(
                    f"staged_reward: stage[{i}].advance_when predicate {advance_name!r} is a "
                    "reward term (float-valued); advance_when gates the stage transition and "
                    "must be a bool predicate. Reward terms belong in the stage's 'reward' field."
                )
            advance_kwargs = {k: v for k, v in advance_call.items() if k != "predicate"}
            advance_fn = make_predicate(advance_name, **advance_kwargs)

        bonus_raw = stage.get("bonus", 0.0)
        # Not a registry factory param, so make_predicate's kwarg-domain check
        # does not see it - the same domain is applied here directly.
        if (err := finite_number_error(bonus_raw, "bonus", f"staged_reward: stage[{i}]")) is not None:
            raise ValueError(err)

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
    "base_tipped": _base_tipped,
    "base_below_z": _base_below_z,
    "base_beyond_x": _base_beyond_x,
    "base_beyond_y": _base_beyond_y,
    "base_yaw_beyond": _base_yaw_beyond,
    # float-valued
    "distance_neg": _distance_neg,
    "joint_progress": _joint_progress,
    "base_velocity": _base_velocity,
    "base_velocity_tracking": _base_velocity_tracking,
    "base_height": _base_height,
    "base_orientation": _base_orientation,
    "base_lin_vel_z": _base_lin_vel_z,
    "base_ang_vel_xy": _base_ang_vel_xy,
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


# Parameter annotations that carry a numeric domain. Read as the strings the
# module's ``from __future__ import annotations`` leaves in ``__annotations__``,
# which is the same mechanism :func:`predicate_kind` uses on the return
# annotation - so a new predicate is covered by declaring its params, with no
# separate per-predicate table to drift out of step with the registry.
_SCALAR_NUMBER_ANNOTATIONS = frozenset({"float", "int"})
_NUMBER_SEQUENCE_ANNOTATIONS = frozenset({"list[float]", "list[int]", "tuple[float, ...]"})


def _kwarg_domain_error(name: str, factory: PredicateFactory, kwargs: dict[str, Any]) -> str | None:
    """Return an error message if a numeric kwarg for *name* is not a finite number.

    A spec kwarg is coerced with a bare ``float(...)`` inside the factory and
    then closed over, so a ``nan``/``inf`` threshold or weight compiles clean
    and only shows up in the evaluated result:

    * In a ``dense_reward`` term it makes the episode's ``cumulative_reward``
      and the eval's ``avg_reward`` ``nan``/``inf``, reported under
      ``status="success"`` - the score silently poisons whatever consumes it.
    * In a ``success`` / ``failure`` / ``stop_when`` clause every comparison
      against ``nan`` is ``False``, so the clause is unsatisfiable and the
      rollout burns its whole step budget reporting an honest miss. That is the
      exact failure mode a typo'd body name is probed against the live sim to
      prevent (see :func:`can_resolve_body`); a non-finite threshold reaches it
      by a different route and was not checked.
    * A non-numeric value (``"abc"``) otherwise escapes as a bare
      ``ValueError: could not convert string to float: 'abc'`` naming neither
      the predicate nor the clause it came from.

    Only params the factory annotates as numeric are constrained, so a ``str``
    body name, a ``bool`` flag and a ``str | None`` robot selector are untouched,
    and a predicate registered via :func:`register_predicate` without
    annotations is exempt (the caller opted in by registering it).

    Args:
        name: Predicate name, used in the message.
        factory: The registered factory whose annotations declare the domain.
        kwargs: The spec-supplied kwargs, checked by name.

    Returns:
        An error message, or ``None`` when every numeric kwarg is usable.
    """
    annotations = getattr(factory, "__annotations__", {})
    context = f"predicate '{name}'"
    for param, value in kwargs.items():
        annotation = str(annotations.get(param, ""))
        if annotation in _SCALAR_NUMBER_ANNOTATIONS:
            if (err := finite_number_error(value, param, context)) is not None:
                return err
        elif annotation in _NUMBER_SEQUENCE_ANNOTATIONS:
            if (err := finite_vector_error(context, param, value)) is not None:
                return err
    return None


def make_predicate(name: str, **kwargs: Any) -> Callable[[SimEngine], Any]:
    """Instantiate a predicate from its name + kwargs.

    This is the single entry point the DSL loader uses - it never touches
    ``eval`` or ``exec``. Unknown names produce a ``ValueError`` listing
    the valid set; bad kwargs surface as whatever ``TypeError`` the factory
    raises.

    Every numeric kwarg is held to a finite domain here rather than in the
    spec compiler, because this is the only choke point every predicate call
    passes through: ``staged_reward`` builds its per-stage ``reward`` /
    ``advance_when`` calls by calling back into this function, so a guard in
    :func:`~strands_robots.simulation.benchmark_spec._compile_call` would
    leave nested stage calls unchecked. See :func:`_kwarg_domain_error` for
    what a non-finite threshold or weight does when it compiles.

    Args:
        name: Predicate name. Must be registered in :data:`PREDICATE_REGISTRY`.
        **kwargs: Forwarded verbatim to the factory.

    Returns:
        A callable ``(sim) -> bool`` or ``(sim) -> float`` depending on the
        predicate.

    Raises:
        ValueError: If ``name`` is unknown, or a kwarg the factory annotates
            as numeric is not a finite number.
        TypeError: If required factory kwargs are missing.
    """
    factory = PREDICATE_REGISTRY.get(name)
    if factory is None:
        valid = sorted(PREDICATE_REGISTRY.keys())
        raise ValueError(f"Unknown predicate '{name}'. Valid: {valid}")
    if (err := _kwarg_domain_error(name, factory, kwargs)) is not None:
        raise ValueError(err)
    return factory(**kwargs)


def predicate_kind(name: str) -> str:
    """Classify a registered predicate as ``"bool"`` or ``"float"``.

    Success / failure clauses require a ``"bool"`` predicate; ``dense_reward``
    terms are ``"float"``. The kind is read from the factory's
    ``-> BoolPredicate`` / ``-> RewardTerm`` return annotation, so it stays in
    lock-step with the registry with no separate table to drift. A predicate
    registered via :func:`register_predicate` without a recognizable return
    annotation classifies as ``"unknown"`` and is exempt from kind validation
    (the caller opted in by registering it).

    Args:
        name: A predicate name. Must be registered in :data:`PREDICATE_REGISTRY`.

    Returns:
        ``"bool"``, ``"float"``, or ``"unknown"``.

    Raises:
        ValueError: If ``name`` is not registered (mirrors :func:`make_predicate`).
    """
    factory = PREDICATE_REGISTRY.get(name)
    if factory is None:
        valid = sorted(PREDICATE_REGISTRY.keys())
        raise ValueError(f"Unknown predicate '{name}'. Valid: {valid}")
    annotation = str(getattr(factory, "__annotations__", {}).get("return", ""))
    if "Bool" in annotation:
        return "bool"
    if "Reward" in annotation:
        return "float"
    return "unknown"


def supports_body_lookup(sim: SimEngine) -> bool:
    """Whether *sim*'s backend can resolve body names at all.

    Body-referencing predicates resolve through ``get_body_state`` (MuJoCo
    only at time of writing). On a backend without that method every
    body-referencing predicate degrades to a constant ``False``, so callers
    that arm such a predicate (e.g. ``run_policy(stop_when=...)``) should
    check this up front and reject rather than silently never fire.

    Args:
        sim: The engine to probe.

    Returns:
        ``True`` when the backend implements ``get_body_state``.
    """
    return getattr(sim, "get_body_state", None) is not None


def can_resolve_body(sim: SimEngine, body: str) -> bool:
    """Whether *body* resolves in *sim* right now, via the predicate DSL's own lookup.

    Uses the exact resolution path the body-referencing predicates use at
    evaluation time (:func:`_body_position`), including the LIBERO
    ``<name>_main`` fallback - so a ``True`` here means the predicate will
    genuinely be evaluable against the live scene, and a ``False`` means it
    would degrade to a constant ``False`` forever (a typo'd name, or a
    backend without body lookups).

    Args:
        sim: The engine to probe.
        body: The body name a predicate clause references.

    Returns:
        ``True`` when the lookup resolves to a position.
    """
    return _body_position(sim, body) is not None


def can_resolve_joint(sim: SimEngine, joint: str) -> bool:
    """Whether *joint* resolves in *sim* right now, via the predicate DSL's own lookup.

    Mirrors :func:`can_resolve_body` for joint-referencing predicates
    (``joint_above`` / ``joint_below``), using the same ``get_observation``
    path the predicates use at evaluation time.

    Args:
        sim: The engine to probe.
        joint: The joint name a predicate clause references.

    Returns:
        ``True`` when the lookup resolves to a value.
    """
    return _joint_position(sim, joint) is not None


__all__ = [
    "PREDICATE_REGISTRY",
    "BoolPredicate",
    "PredicateFactory",
    "RewardTerm",
    "StatefulRewardTerm",
    "can_resolve_body",
    "can_resolve_joint",
    "contact_is_active",
    "make_predicate",
    "predicate_kind",
    "register_predicate",
    "supports_body_lookup",
]
