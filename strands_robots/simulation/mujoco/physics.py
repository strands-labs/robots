"""Physics mixin - advanced MuJoCo physics introspection and manipulation.

Exposes the deep MuJoCo C API through clean Python methods:
- Raycasting (mj_ray)
- Jacobians (mj_jacBody, mj_jacSite, mj_jacGeom)
- Energy computation (mj_energyPos, mj_energyVel)
- External forces (mj_applyFT, xfrc_applied)
- Mass matrix (mj_fullM)
- State checkpointing (mj_getState, mj_setState)
- Inverse dynamics (mj_inverse)
- Body/joint introspection (poses, velocities, accelerations)
- Direct joint position/velocity control (qpos, qvel)
- Runtime model modification (mass, friction, color, size)
- Sensor readout (sensordata)
- Contact force analysis (mj_contactForce)
"""

import logging
import math
import numbers
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from strands_robots.simulation.base import _BOOLEAN_STATE_REASON, close_match_hint
from strands_robots.simulation.models import registered
from strands_robots.simulation.mujoco.backend import (
    _NO_WORLD_MSG,
    _ensure_mujoco,
    filter_mujoco_attach_noise,
    mj_name_to_id,
)
from strands_robots.simulation.mujoco.scene_ops import (
    fromto_fixed_size_components,
    persist_body_mass,
    persist_geom_properties,
    refresh_body_inertial_from_geometry,
)
from strands_robots.simulation.safe_output import atomic_write_bytes, validate_output_path
from strands_robots.utils import BOOLEAN_VECTOR_REASON, coerce_rgba, is_boolean

logger = logging.getLogger(__name__)


def _coerce_finite_vector(
    values: Any,
    name: str,
    method: str,
    *,
    min_value: float | None = None,
    strict_min: bool = False,
    accepted_lengths: tuple[int, ...] | None = None,
    layout: str = "",
) -> tuple[list[float] | None, dict[str, Any] | None]:
    """Coerce a numeric vector to ``float`` and validate every element.

    Each element must be a real number (Python or NumPy scalar) and finite --
    ``nan`` / ``inf`` are rejected because they slip silently into the MuJoCo
    model buffers and corrupt the solver while the tool still reports success.
    An optional lower bound enforces physics invariants (non-negative friction,
    positive geom extent).

    ``accepted_lengths`` enforces the component count the target buffer defines.
    A vector shorter than its target cannot be written without inventing the
    missing components (leaving them at their compiled value, or padding with a
    fabricated one), and a longer vector can only be written by discarding its
    tail -- both apply a value the caller never asked for, so the count is
    rejected instead.

    Args:
        values: The input sequence (list / tuple / NumPy array).
        name: Parameter name, used in error text.
        method: Calling method name, used in error text.
        min_value: If set, every element must be ``>= min_value`` (or ``>`` when
            ``strict_min`` is True).
        strict_min: Use a strict ``>`` comparison against ``min_value``.
        accepted_lengths: If set, the component counts that can be honored.
        layout: Human-readable meaning of the components, used in the
            component-count error text.

    Returns:
        ``(floats, None)`` on success, or ``(None, error_dict)`` on the first
        invalid element or an unusable component count -- matching the
        structured-error tool contract so the caller never raises past dispatch.
    """
    try:
        seq = list(values)
    except TypeError:
        return None, {
            "status": "error",
            "content": [{"text": f"{method}: '{name}' must be a sequence of numbers, got {values!r}"}],
        }
    out: list[float] = []
    for elem in seq:
        # Before float(), which cannot tell a boolean apart afterwards: bool is
        # an int subclass and numpy.bool_ coerces the same way, so both arrived
        # as a silent 1.0/0.0 component under status="success". This is the one
        # chokepoint for every vector the runtime writers take - a raycast
        # origin and direction, a geom size and friction, an rgba colour, and
        # each ray of a multi_raycast batch - so the gate belongs here rather
        # than at the seven call sites.
        if is_boolean(elem):
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": f"{method}: '{name}' elements must be numbers, not a bool (got {values!r}). {BOOLEAN_VECTOR_REASON}"
                    }
                ],
            }
        try:
            f = float(elem)
        except (TypeError, ValueError):
            return None, {
                "status": "error",
                "content": [{"text": f"{method}: '{name}' elements must be numbers, got {values!r}"}],
            }
        if not math.isfinite(f):
            return None, {
                "status": "error",
                "content": [{"text": f"{method}: '{name}' must contain finite numbers (no nan/inf), got {values!r}"}],
            }
        if min_value is not None and ((f <= min_value) if strict_min else (f < min_value)):
            rel = ">" if strict_min else ">="
            return None, {
                "status": "error",
                "content": [{"text": f"{method}: '{name}' values must be {rel} {min_value}, got {values!r}"}],
            }
        out.append(f)
    if accepted_lengths is not None and len(out) not in accepted_lengths:
        expected = " or ".join(str(n) for n in accepted_lengths)
        detail = f" ({layout})" if layout else ""
        return None, {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: '{name}' must have exactly {expected} "
                        f"component(s){detail}, got {len(out)}: {out}. Pass every "
                        f"component - a partial '{name}' cannot be applied "
                        "without inventing the missing values."
                    )
                }
            ],
        }
    return out, None


def _coerce_rgba(color: Any, method: str, name: str = "color") -> tuple[list[float] | None, dict[str, Any] | None]:
    """Coerce a caller-supplied colour to the 4 components a geom's rgba stores.

    Envelope binding over :func:`strands_robots.utils.coerce_rgba`, which is the
    single definition of the colour contract for every backend. This wrapper
    exists only to report the shared reason in the structured-error shape the
    MuJoCo scene creator (``add_object``) and runtime mutator
    (``set_geom_properties``) return, so those two agreed domains stay in step
    with Newton's and Isaac's rather than being a second copy of the rule.

    Args:
        color: The caller's colour sequence (list / tuple / NumPy array).
        method: Calling method name, used in error text.
        name: Parameter name, used in error text.

    Returns:
        ``(rgba, None)`` with exactly 4 finite floats, or ``(None, error_dict)``
        matching the structured-error tool contract.
    """
    rgba, reason = coerce_rgba(method, name, color)
    if reason is not None:
        return None, {"status": "error", "content": [{"text": reason}]}
    return rgba, None


# A ray batch is a sequence of 3-component direction vectors. Spelling it out in
# one place keeps the batch-shape rejection and the tool_spec description in
# agreement about what a castable batch looks like.
_RAY_BATCH_HINT = "Pass one [dx, dy, dz] direction per ray, e.g. [[0, 0, -1], [1, 0, 0]]."


def _coerce_ray_batch(directions: Any, method: str) -> tuple[list[Any] | None, dict[str, Any] | None]:
    """Materialize a ray batch, refusing a value that names no castable ray.

    Guards the batch parameter itself, before the per-ray direction checks. A
    ``str`` is iterable, so ``directions="abc"`` would otherwise be read as three
    rays - one per character - and a non-iterable raises ``TypeError`` past the
    structured-error tool contract. An empty batch requests no ray at all, so
    there is no cast whose result could be reported.

    Args:
        directions: The caller-supplied batch.
        method: Calling method name, used in error text.

    Returns:
        ``(rays, None)`` with the batch materialized as a list, or
        ``(None, error_dict)`` when it holds no castable ray.
    """
    if isinstance(directions, (str, bytes, bytearray)):
        return None, {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: 'directions' must be a sequence of direction vectors, "
                        f"not {type(directions).__name__} - iterating one casts a ray per "
                        f"character. {_RAY_BATCH_HINT}"
                    )
                }
            ],
        }
    try:
        rays = list(directions)
    except TypeError:
        return None, {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: 'directions' must be a sequence of direction vectors, "
                        f"got {directions!r}. {_RAY_BATCH_HINT}"
                    )
                }
            ],
        }
    if not rays:
        return None, {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: 'directions' must hold at least one direction - "
                        f"an empty batch casts no ray. {_RAY_BATCH_HINT}"
                    )
                }
            ],
        }
    return rays, None


def _coerce_excluded_body(value: Any, method: str, nbody: int) -> tuple[int | None, dict[str, Any] | None]:
    """Coerce ``exclude_body`` to a body id ``mj_ray`` can honor.

    ``mj_ray`` takes the exclusion as a C ``int`` and skips the geoms whose body
    id equals it, so only ``-1`` (the documented "exclude nothing") or an id the
    compiled model actually defines can be honored. A float / str / nan value
    raises ``TypeError`` out of the pybind11 signature - past the structured-error
    tool contract - and an id outside ``[0, nbody)`` matches no body, so the cast
    silently includes the geoms the caller asked to skip and can report a hit on
    the caller's own robot as an obstacle.

    Accepts any real scalar with an integral value (a NumPy ``np.int64`` body id
    read back from ``list_bodies`` passes) and rejects ``bool`` explicitly - an
    ``int`` subclass whose ``True`` would silently exclude body 1.

    Args:
        value: The caller-supplied exclusion.
        method: Calling method name, used in error text.
        nbody: Body count of the compiled model (``model.nbody``).

    Returns:
        ``(body_id, None)`` on success, or ``(None, error_dict)``.
    """
    error = {
        "status": "error",
        "content": [
            {
                "text": (
                    f"{method}: 'exclude_body' must be -1 (exclude nothing) or a body id "
                    f"in [0, {nbody}), got {value!r}. Read an id from list_bodies()."
                )
            }
        ],
    }
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        return None, error
    numeric = float(value)
    # ``isfinite`` first: ``int(nan)`` raises, and short-circuiting keeps it out
    # of the integrality check below.
    if not math.isfinite(numeric) or numeric != int(numeric):
        return None, error
    body_id = int(numeric)
    if body_id < -1 or body_id >= nbody:
        return None, error
    return body_id, None


def _full_mass_matrix(mj: Any, model: Any, data: Any) -> np.ndarray:
    """Return the dense ``nv x nv`` mass matrix M(q), robust to MuJoCo drift.

    MuJoCo moved this API twice inside the supported version range:

    - MuJoCo < 3.10: ``mj_fullM(model, dst, qM)``, reading the legacy
      ancestor-walk sparse buffer ``data.qM`` - accepted either as a 1D array
      or as a 2D ``[m, 1]`` column, depending on the build.
    - MuJoCo >= 3.10: ``mj_fullM(model, data, dst)`` - the sparse buffer is
      read from ``data`` internally; ``dst`` must be writeable + C-contiguous.
    - MuJoCo >= 3.11: ``data.qM`` is removed. The joint-space inertia is kept
      only as the compressed-sparse-row ``data.M``, and ``mju_sym2dense`` is
      the conversion MuJoCo's release notes prescribe for callers that used to
      pass ``qM`` to ``mj_fullM``. That helper is only exported from MuJoCo
      3.10 onwards, while the CSR buffers and their index arrays
      (``M_rownnz`` / ``M_rowadr`` / ``M_colind``) ship from 3.5, so when the
      helper is absent the stored lower triangle is expanded through those
      index arrays instead.

    Probe the modern signature first, then the legacy orders - but only on a
    build that still exposes ``qM``, because the CSR ``data.M`` that replaced
    it is a different layout and passing it to the legacy call would fill
    ``dst`` from the wrong buffer rather than fail - then the CSR conversion.
    ``dst`` is always allocated C-contiguous to satisfy the buffer contract of
    every one of those calls.

    Args:
        mj: The imported ``mujoco`` module.
        model: The ``MjModel`` whose DoF count defines the matrix size.
        data: The ``MjData`` holding the sparse inertia (after a forward pass).

    Returns:
        A C-contiguous ``(nv, nv)`` float64 array. Empty (``(0, 0)``) when the
        model has no DoFs.

    Raises:
        TypeError: If no known ``mj_fullM`` signature accepts the arguments.
        AttributeError: If MjData exposes the joint-space inertia under
            neither ``qM`` nor ``M`` (MuJoCo < 3.5 predates the CSR buffer).
    """
    nv = model.nv
    dst = np.zeros((nv, nv), dtype=np.float64, order="C")
    if nv == 0:
        return dst
    try:
        # MuJoCo >= 3.10: dst is the third positional argument.
        mj.mj_fullM(model, data, dst)
        return dst
    except TypeError:
        pass
    legacy = getattr(data, "qM", None)
    if legacy is not None:
        # Legacy signature: mj_fullM(model, dst, qM). Some builds require the
        # sparse buffer as a 2D [m, 1] column; others accept the raw 1D buffer.
        qm = np.ascontiguousarray(legacy, dtype=np.float64)
        try:
            mj.mj_fullM(model, dst, qm.reshape(-1, 1))
        except TypeError:
            mj.mj_fullM(model, dst, qm)
        return dst
    # MuJoCo >= 3.11: no legacy buffer to pass, so convert the CSR inertia
    # directly.
    csr = getattr(data, "M", None)
    if csr is None:
        raise AttributeError(
            "MjData exposes the joint-space inertia under neither name (tried "
            f"data.qM and data.M) on mujoco {getattr(mj, '__version__', 'unknown')}, "
            "so the dense mass matrix cannot be built."
        )
    values = np.ascontiguousarray(csr, dtype=np.float64)
    sym2dense = getattr(mj, "mju_sym2dense", None)
    if sym2dense is not None:
        sym2dense(dst, values, model.M_rownnz, model.M_rowadr, model.M_colind)
        return dst
    # MuJoCo 3.5 - 3.9 ship the CSR buffers and their index arrays but not the
    # conversion, so expand the stored lower triangle through those indices and
    # mirror it. Same arithmetic mju_sym2dense performs, and it keeps the CSR
    # rung working across the whole supported MuJoCo range rather than only on
    # the builds that also export the helper.
    rownnz, rowadr, colind = model.M_rownnz, model.M_rowadr, model.M_colind
    for row in range(nv):
        start = int(rowadr[row])
        stored = values[start : start + int(rownnz[row])]
        cols = np.asarray(colind[start : start + int(rownnz[row])], dtype=np.intp)
        dst[row, cols] = stored
        dst[cols, row] = stored
    return dst


def _recompute_primitive_geom_bounds(mj: Any, model: Any, gid: int) -> bool:
    """Recompute a geom's collision bounding volumes after a runtime size change.

    ``geom_rbound`` (the broadphase bounding-sphere radius) and ``geom_aabb``
    (the mid-phase axis-aligned bounding box) are derived from ``geom_size`` at
    compile time and are NOT recomputed by ``mj_forward``/``mj_step``. Writing a
    larger ``geom_size`` at runtime without refreshing them leaves the broadphase
    culling against the old, smaller radius, so contacts with the grown surface
    are silently dropped and other bodies pass straight through it.

    Recompute both from ``geom_type`` + ``geom_size`` for the primitive types
    whose extent is defined by ``geom_size`` (sphere, capsule, cylinder,
    ellipsoid, box). Mesh/plane/height-field/SDF geoms derive their extent from
    asset data, not ``geom_size``, so a size write is inert for them and their
    bounds are left untouched.

    Args:
        mj: The imported ``mujoco`` module.
        model: The live ``MjModel`` to update in place.
        gid: The geom id whose ``geom_size`` was just changed.

    Returns:
        ``True`` if the bounds were recomputed (a size-defined primitive),
        ``False`` for a type whose extent is not defined by ``geom_size``.
    """
    g = mj.mjtGeom
    gtype = int(model.geom_type[gid])
    s = [float(v) for v in model.geom_size[gid]]

    # Per-type extent: (bounding-sphere radius, aabb half-extents). geom_size
    # semantics differ by type, so both the rbound and the aabb are type-
    # specific (they match a fresh compile at the new size).
    if gtype == g.mjGEOM_SPHERE:
        rbound, half = s[0], [s[0], s[0], s[0]]
    elif gtype == g.mjGEOM_CAPSULE:
        rbound, half = s[0] + s[1], [s[0], s[0], s[0] + s[1]]
    elif gtype == g.mjGEOM_CYLINDER:
        rbound, half = float(np.hypot(s[0], s[1])), [s[0], s[0], s[1]]
    elif gtype == g.mjGEOM_ELLIPSOID:
        rbound, half = max(s[0], s[1], s[2]), [s[0], s[1], s[2]]
    elif gtype == g.mjGEOM_BOX:
        rbound, half = float(np.linalg.norm(s[:3])), [s[0], s[1], s[2]]
    else:
        # mesh / plane / hfield / sdf: extent is not defined by geom_size.
        return False

    model.geom_rbound[gid] = float(rbound)
    # geom_aabb layout is [center(3), half-extent(3)]; primitives are centered.
    model.geom_aabb[gid, 3:6] = half
    return True


# Number of ``geom_size`` components each MuJoCo geom type defines, plus the
# meaning of each one. MuJoCo stores every geom's extent in a 3-wide
# ``geom_size`` row, but only the leading components its type defines carry
# meaning - a sphere reads one, a capsule two, a box three. A type whose extent
# comes from asset data (mesh, height field, SDF) defines none.
_GEOM_SIZE_LAYOUTS: dict[str, tuple[int, str]] = {
    "plane": (3, "x half-extent, y half-extent, grid spacing"),
    "sphere": (1, "radius"),
    "capsule": (2, "radius, half-length"),
    "ellipsoid": (3, "three semi-axes"),
    "cylinder": (2, "radius, half-length"),
    "box": (3, "three half-extents"),
}


def _geom_type_name(mj: Any, geom_type: int) -> str:
    """Return a geom type's short lowercase name (``"box"``, ``"mesh"``, ...).

    Args:
        mj: The imported ``mujoco`` module.
        geom_type: A ``model.geom_type`` entry (an ``mjtGeom`` value).

    Returns:
        The ``mjtGeom`` name with its ``mjGEOM_`` prefix stripped and lowercased,
        or ``"type_<n>"`` if the value is not a known ``mjtGeom`` member.
    """
    try:
        return str(mj.mjtGeom(int(geom_type)).name).removeprefix("mjGEOM_").lower()
    except ValueError:
        return f"type_{int(geom_type)}"


class PhysicsMixin:
    """Advanced MuJoCo physics capabilities mixed into ``Simulation``.

    Lives at roughly ``self._world._data`` + ``self._world._model`` level:
    reads/writes MuJoCo arrays directly for checkpointing, raycasts,
    jacobians, joint control, sensor readout, etc.

    **Coupling** (see the :mod:`simulation` top-level docstring): mixin reaches
    into ``self._world``, ``self._lock``, and the host's
    ``_require_no_running_policy`` / ``_require_world`` / ``_prune_done_futures``
    helpers. ``TYPE_CHECKING`` stubs below exist so mypy accepts those
    lookups; they are a documentary contract, not an enforceable protocol.

    Naming: methods match action names in tool_spec.json for direct dispatch.
    """

    if TYPE_CHECKING:
        import threading

        from strands_robots.simulation.models import SimWorld

        _lock: "threading.RLock"
        _world: "SimWorld | None"

        # Bodies are one-line docstrings rather than ``...`` because an
        # ellipsis body is an expression statement with no effect.
        def _require_no_running_policy(self, action_name: str, robot_name: str | None = None) -> dict[str, Any] | None:
            """Refuse a mutation while a policy thread is stepping the world."""

        def _require_world(self) -> dict[str, Any] | None:
            """Refuse a call made before ``create_world``."""

        def _unknown_robot_msg(self, requested: object) -> str:
            """Build the "robot not found" message with close-match hints."""

        def _validate_mass(self, mass: Any, method: str, param: str = "mass") -> dict[str, Any] | None:
            """Reject a body mass the physics engine cannot honor."""

        def _coerce_joint_state_map(
            self, values: dict[str, Any], name: str, method: str
        ) -> tuple[dict[str, float], dict[str, Any] | None]:
            """Coerce a joint-state map to finite floats before any write."""

    # State Checkpointing

    def save_state(self, name: str = "default") -> dict[str, Any]:
        """Save the full integration state to a named checkpoint.

        Captures the complete ``mjSTATE_INTEGRATION`` vector - qpos, qvel,
        act, ctrl, qfrc_applied, xfrc_applied, mocap pose, eq_active, plugin
        state and time - so a subsequent ``load_state`` restores the servo
        targets (``ctrl``) and latched external forces, not just positions.
        ``mjSTATE_FULLPHYSICS`` (used previously) silently excluded ``ctrl``
        and ``qfrc_applied``, so the first step after a restore drove toward
        the pre-restore targets - contradicting this docstring and
        ``describe()``.

        The checkpoint is stamped with the model's structural fingerprint
        (state size + nq/nv/na/nu). A scene recompile that changes the model
        shape (e.g. ``add_object`` inserts a free joint) invalidates the
        stored vector; ``load_state`` detects the mismatch and returns a
        structured error instead of raising a raw ``ValueError`` or silently
        applying a misaligned vector.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            state_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
            state = np.zeros(state_size)
            mj.mj_getState(model, data, state, mj.mjtState.mjSTATE_INTEGRATION)
            fingerprint = (
                int(model.nq),
                int(model.nv),
                int(model.na),
                int(model.nu),
                self._world._recompile_generation,
            )

        if not hasattr(self._world, "_checkpoints"):
            self._world._checkpoints = {}

        self._world._checkpoints[name] = {
            "state": state.copy(),
            "state_size": int(state_size),
            "fingerprint": fingerprint,
            "sim_time": self._world.sim_time,
            "step_count": self._world.step_count,
        }

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"State '{name}' saved\n"
                        f"  t={self._world.sim_time:.4f}s, step={self._world.step_count}\n"
                        f"State vector: {state_size} floats (mjSTATE_INTEGRATION, incl. ctrl)\n"
                        f"Checkpoints: {list(self._world._checkpoints.keys())}"
                    )
                }
            ],
        }

    def load_state(self, name: str = "default") -> dict[str, Any]:
        """Restore integration state (incl. ctrl) from a named checkpoint.

        Refuses to apply a checkpoint whose structural fingerprint no longer
        matches the live model - a scene recompile since ``save_state`` (e.g.
        ``add_object`` / ``remove_robot``) resized the state vector, so
        applying it would raise a raw ``ValueError`` or silently misalign
        qpos/ctrl. In that case a structured error is returned.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # load_state during a running policy races worker thread
        if err := self._require_no_running_policy("load_state"):
            return err

        checkpoints = getattr(self._world, "_checkpoints", {})
        if not registered(checkpoints, name):
            available = list(checkpoints.keys()) if checkpoints else ["none"]
            return {
                "status": "error",
                "content": [{"text": f"Checkpoint '{name}' not found. Available: {available}"}],
            }

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data
        checkpoint = checkpoints[name]

        with self._lock:
            current_size = mj.mj_stateSize(model, mj.mjtState.mjSTATE_INTEGRATION)
            current_fp = (
                int(model.nq),
                int(model.nv),
                int(model.na),
                int(model.nu),
                self._world._recompile_generation,
            )
            saved_size = checkpoint.get("state_size", checkpoint["state"].shape[0])
            saved_fp = checkpoint.get("fingerprint")
            if current_size != saved_size or (saved_fp is not None and current_fp != saved_fp):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"Checkpoint '{name}' is stale: the scene was recompiled since it "
                                f"was saved (saved nq/nv/na/nu={saved_fp}, size={saved_size}; "
                                f"current={current_fp}, size={current_size}). Applying it would "
                                f"misalign the physics state. Save a fresh checkpoint after scene "
                                f"mutations such as add_object / remove_robot."
                            )
                        }
                    ],
                }

            mj.mj_setState(model, data, checkpoint["state"], mj.mjtState.mjSTATE_INTEGRATION)
            mj.mj_forward(model, data)

            self._world.sim_time = checkpoint["sim_time"]
            self._world.step_count = checkpoint["step_count"]

        return {
            "status": "success",
            "content": [
                {"text": f"State '{name}' restored (t={self._world.sim_time:.4f}s, step={self._world.step_count})"}
            ],
        }

    # External Forces

    def apply_force(
        self,
        body_name: str,
        force: list[float] | None = None,
        torque: list[float] | None = None,
        point: list[float] | None = None,
    ) -> dict[str, Any]:
        """Apply an external force and/or torque to a body (latched).

        The wrench is latched in the target body's own ``xfrc_applied`` row
        and applied on every subsequent ``mj_step`` until the next
        ``apply_force`` call for that body. A call replaces the wrench latched
        on its own target rather than accumulating onto it, and leaves wrenches
        latched on other bodies alone - so a wind field, two thrusters, or a
        per-object disturbance sweep can hold several bodies at once.

        MuJoCo re-maps that Cartesian wrench into joint space on every step,
        so a latched force keeps pointing the way the caller asked as the body
        moves. ``force`` and ``torque`` are world-frame; a ``point`` away from
        the body centre of mass contributes its lever-arm torque.

        To stop the force on one body: ``apply_force(body, force=[0, 0, 0])``.
        ``reset()`` clears every latched wrench in the world.

        Each vector may be a list, a tuple or a NumPy array (a computed wrench
        is an array), and every element must be a finite real number. A boolean
        element (python or ``numpy.bool_``) is refused rather than applied as
        ``1.0`` N - see :data:`_BOOLEAN_STATE_REASON`.

        Args:
            body_name: Target body name.
            force: [fx, fy, fz] in world frame (Newtons).
            torque: [tx, ty, tz] in world frame (N·m).
            point: [px, py, pz] world-frame point of force application.
                   Defaults to body CoM if not specified.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # apply_force during a running policy races worker thread
        if err := self._require_no_running_policy("apply_force"):
            return err

        # must supply at least one non-zero force or torque
        if force is None and torque is None:
            return {
                "status": "error",
                "content": [{"text": "apply_force: specify at least one of 'force' or 'torque' (non-zero vector)."}],
            }

        # Validate vector lengths before hitting numpy
        for _name, _vec in (("force", force), ("torque", torque), ("point", point)):
            if _vec is not None:
                try:
                    if len(_vec) != 3:
                        return {
                            "status": "error",
                            "content": [
                                {"text": f"apply_force: '{_name}' must be a 3-element vector [x,y,z], got {len(_vec)}"}
                            ],
                        }
                except TypeError:
                    return {
                        "status": "error",
                        "content": [{"text": f"apply_force: '{_name}' must be a list/tuple of 3 numbers"}],
                    }
                # Every element must be a finite real number. Without this,
                # non-numeric elements (e.g. ["a", "b", "c"]) raise ValueError
                # inside np.array(dtype=float64) - escaping the structured-error
                # contract - and nan/inf or nested lists slip silently into
                # mj_applyFT, injecting bad state into the physics buffer.
                # A bool element is refused rather than accepted as finite. The
                # earlier note here deferred it as "out of scope for numeric-element
                # validation"; that scope note is what #1838 revisited, because
                # float(True) is 1.0 and a True force component silently applies
                # 1 N along that axis while the call reports success.
                for _elem in _vec:
                    if is_boolean(_elem):
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": f"apply_force: '{_name}' elements must be numbers, not a bool (got {_vec!r}). {_BOOLEAN_STATE_REASON}"
                                }
                            ],
                        }
                    try:
                        _f = float(_elem)
                    except (TypeError, ValueError):
                        return {
                            "status": "error",
                            "content": [{"text": f"apply_force: '{_name}' elements must be numbers, got {_vec!r}"}],
                        }
                    if not math.isfinite(_f):
                        return {
                            "status": "error",
                            "content": [
                                {
                                    "text": f"apply_force: '{_name}' must contain finite numbers (no nan/inf), got {_vec!r}"
                                }
                            ],
                        }

        mj = _ensure_mujoco()
        data = self._world._data

        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Body", body_name)}]}

        # Membership, not truthiness: a vector is supplied when it is not None.
        # ``force or [0, 0, 0]`` raised a bare "truth value of an array with more
        # than one element is ambiguous" ValueError - through the structured
        # tool-result contract - for a NumPy force/torque/point, which is what
        # any computed wrench (``mass * accel``, a Jacobian row) actually is.
        # Note: explicit [0,0,0] is a valid "clear the latched force" command; we only
        # reject the case where the caller forgot both args (handled above).
        f = np.array([0.0, 0.0, 0.0] if force is None else force, dtype=np.float64)
        t = np.array([0.0, 0.0, 0.0] if torque is None else torque, dtype=np.float64)
        p = np.array(point, dtype=np.float64) if point is not None else data.xipos[body_id].copy()

        # Latch the wrench in this body's own row of ``xfrc_applied``.
        #
        # The buffer choice is the whole per-body contract. ``qfrc_applied`` is
        # one world-wide generalized-force vector, and a wrench on a body part
        # way down a kinematic chain writes into its ancestors' DOFs too - so
        # there is no slice of it that belongs to one body, and zeroing it to
        # make a call idempotent revoked every wrench already latched on every
        # other body. ``xfrc_applied`` is indexed by body, so replacing this
        # body's wrench cannot touch anyone else's.
        #
        # It is also the more faithful latch: MuJoCo re-maps the Cartesian
        # wrench through the current configuration on every step, whereas a
        # generalized force frozen by mj_applyFT at call time stops describing
        # the caller's world-frame wrench as soon as the body moves.
        #
        # NOTE: MuJoCo does NOT reset xfrc_applied in mj_step - the wrench
        # persists on every subsequent step until the next apply_force call
        # for this body (or a reset()).
        with self._lock:
            # xfrc_applied's torque acts about the body centre of mass, so a
            # force applied at an offset point contributes (point - com) x
            # force. A caller who named no point asked for the CoM itself,
            # where that lever arm is exactly zero.
            com_torque = t if point is None else t + np.cross(p - data.xipos[body_id], f)
            data.xfrc_applied[body_id, :3] = f
            data.xfrc_applied[body_id, 3:] = com_torque

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Force applied to '{body_name}' (body {body_id})\n"
                        f"Force: {f.tolist()} N\n"
                        f"Torque: {t.tolist()} N·m\n"
                        f"Point: {p.tolist()}"
                    )
                }
            ],
        }

    # Raycasting

    def _resolve_mj_name(self, obj_type: int, name: str) -> int:
        """Look up a MuJoCo name, tolerating robot namespacing.

        For physics/introspection methods that accept raw body/joint/site
        names (``get_body_state("gripper")`` etc.), we try the name
        verbatim first, then fall back to trying it prefixed with every
        robot's namespace. This preserves the pre-namespacing UX for
        single-robot scenes while still working in multi-robot scenes
        when the name is unambiguous.

        In multi-robot scenes where multiple robots contain a body with
        the same short name (e.g. two so101s each having ``gripper``),
        the caller MUST pass the namespaced form (``arm0/gripper``) to
        disambiguate. The fallback returns the first match it finds,
        which is non-deterministic - this is a deliberate
        "unambiguous or explicit" contract.
        """

        assert self._world is not None and self._world._model is not None
        model = self._world._model
        mid = mj_name_to_id(model, obj_type, name)
        if mid >= 0:
            return int(mid)
        if not isinstance(name, str):
            # mj_name_to_id already refused it; the namespace retry below
            # would only reach `in` / `+` on a value that supports neither.
            return -1
        if "/" in name:  # already namespaced, no point retrying
            return -1
        for robot in self._world.robots.values():
            if robot.namespace:
                mid = mj_name_to_id(model, obj_type, robot.namespace + name)
                if mid >= 0:
                    return int(mid)
        return -1

    def _unknown_mj_entity_msg(self, kind: str, requested: object) -> str:
        """Actionable "<kind> not found" message for the physics/introspection
        lookups (``get_body_state`` / ``get_jacobian`` / ``set_body_properties`` /
        ``set_geom_properties`` / ``get_sensor_data`` ...): name the entity, offer
        a difflib close-match over the model's *named* entities, list the
        available names (capped), and - for bodies - point at the ``list_bodies``
        discovery action. Consistent with ``_unknown_object_msg`` /
        ``_unknown_camera_msg`` / ``_unknown_robot_msg`` (#1299/#1303/#1306)
        rather than a dead-end "<Kind> 'X' not found." that forces an agent
        driving the API blind into guesswork on every typo.

        The ``"<Kind> 'X' not found."`` prefix is preserved so the consistent
        error shape (T15 in ``test_agenttool_contract``) is unaffected.

        ``kind`` is one of ``"Body" | "Site" | "Geom" | "Sensor" | "Joint"``.

        ``requested`` is typed ``object``: a name of any type reaches here, and
        only the close match needs a string (see
        :func:`~strands_robots.simulation.base.close_match_hint`). The
        available-entity listing is a fact about the compiled model, so it is
        emitted for every name type rather than being suppressed into a bare
        ``"<Kind> 'X' not found."`` dead end.
        """
        import mujoco as _mj

        model = self._world._model if self._world is not None else None
        msg = f"{kind} '{requested}' not found."
        if model is None:
            return msg
        obj_type, count = {
            "Body": (_mj.mjtObj.mjOBJ_BODY, model.nbody),
            "Site": (_mj.mjtObj.mjOBJ_SITE, model.nsite),
            "Geom": (_mj.mjtObj.mjOBJ_GEOM, model.ngeom),
            "Sensor": (_mj.mjtObj.mjOBJ_SENSOR, model.nsensor),
            "Joint": (_mj.mjtObj.mjOBJ_JOINT, model.njnt),
        }[kind]
        known = [nm for i in range(int(count)) if (nm := _mj.mj_id2name(model, obj_type, i)) and nm != "world"]
        if known:
            msg += close_match_hint(requested, known)
            shown = known if len(known) <= 30 else known[:30] + ["..."]
            plural = {
                "Body": "bodies",
                "Site": "sites",
                "Geom": "geoms",
                "Sensor": "sensors",
                "Joint": "joints",
            }[kind]
            msg += f" Available {plural}: {shown}."
            if kind == "Body":
                msg += " Use action='list_bodies' to see all."
            elif kind == "Joint":
                msg += " Use action='robot_joint_names' to see one robot's joints."
        return msg

    def raycast(
        self,
        origin: list[float],
        direction: list[float],
        exclude_body: int = -1,
        include_static: bool = True,
    ) -> dict[str, Any]:
        """Cast a ray and find the first geom intersection.

        Uses mj_ray for precise distance sensing / obstacle detection. Geom
        world poses are refreshed (``mj_kinematics``) under the sim lock before
        the cast, so the result reflects the current ``qpos`` and cannot be torn
        by a concurrent policy thread's ``mj_step``.

        Args:
            origin: [x, y, z] ray start point in world frame.
            direction: [dx, dy, dz] ray direction (auto-normalized).
            exclude_body: Body ID whose geoms the ray passes through (``-1`` =
                exclude nothing). Any other value must be a body id the compiled
                model defines - an id outside ``[0, model.nbody)`` matches no
                body, so the geoms the caller asked to skip would be included
                and could be reported as the obstacle.
            include_static: Whether to include static geoms.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        # validate vector shapes and reject zero-direction (mj_ray aborts the process on len=0)
        try:
            if len(origin) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"raycast: 'origin' must be 3 elements [x,y,z], got {len(origin)}"}],
                }
            if len(direction) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"raycast: 'direction' must be 3 elements [dx,dy,dz], got {len(direction)}"}],
                }
        except TypeError:
            return {
                "status": "error",
                "content": [{"text": "raycast: 'origin' and 'direction' must be lists of 3 numbers"}],
            }

        # Every element must be a finite real number. Without this, a
        # non-numeric element (e.g. ["a", ...]) raises ValueError inside
        # np.array(dtype=float64) -- escaping the structured-error contract --
        # and nan/inf slip silently into mj_ray (nan direction survives the
        # zero-length guard because ``nan < 1e-10`` is False, then poisons the
        # normalized vector fed to the C solver).
        origin_f, err = _coerce_finite_vector(origin, "origin", "raycast")
        if err is not None:
            return err
        direction_f, err = _coerce_finite_vector(direction, "direction", "raycast")
        if err is not None:
            return err

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        # ``exclude_body`` reaches mj_ray as a C int. A non-integral value raises
        # TypeError out of the pybind11 signature (past the tool contract) and an
        # out-of-range id matches no body, so the exclusion silently does nothing.
        exclusion, err = _coerce_excluded_body(exclude_body, "raycast", int(model.nbody))
        if err is not None:
            return err

        pnt = np.array(origin_f, dtype=np.float64)
        vec = np.array(direction_f, dtype=np.float64)
        # Normalize direction
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return {
                "status": "error",
                "content": [{"text": "raycast: 'direction' vector is zero-length - supply a non-zero direction."}],
            }
        vec = vec / norm

        geomid = np.array([-1], dtype=np.int32)
        # mj_ray intersects the ray against ``data.geom_xpos``/``geom_xmat``
        # (world-frame geom poses -- derived state populated by kinematics, not
        # recomputed on a bare ``qpos`` write). Refresh them under the lock so
        # the cast reflects the current pose (e.g. after a direct ``qpos`` write
        # from a planning/IK loop) and cannot be torn by a policy thread's
        # ``mj_step``. ``mj_kinematics`` is the minimal forward that populates
        # geom world poses -- cheaper than a full ``mj_forward`` and matching
        # the defensive refresh the other query methods perform.
        with self._lock:
            mj.mj_kinematics(model, data)
            dist = mj.mj_ray(
                model,
                data,
                pnt,
                vec,
                None,  # geom group filter (None = all)
                1 if include_static else 0,
                exclusion,
                geomid,
            )
            hit = dist >= 0
            geom_name = None
            if hit and geomid[0] >= 0:
                geom_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, geomid[0])

        result = {
            "hit": hit,
            "distance": float(dist) if hit else None,
            "geom_id": int(geomid[0]) if hit else None,
            "geom_name": geom_name,
            "hit_point": (pnt + vec * dist).tolist() if hit else None,
        }

        if hit:
            text = f"Ray hit '{geom_name or geomid[0]}' at dist={dist:.4f}m, point={result['hit_point']}"
        else:
            text = "Ray: no intersection"

        return {"status": "success", "content": [{"text": text}, {"json": result}]}

    # Jacobians

    def get_jacobian(
        self,
        body_name: str | None = None,
        site_name: str | None = None,
        geom_name: str | None = None,
    ) -> dict[str, Any]:
        """Compute the Jacobian (position + rotation) for a body, site, or geom.

        The Jacobian maps joint velocities to Cartesian velocities:
            v = J @ dq

        Returns both positional (3×nv) and rotational (3×nv) Jacobians,
        computed at the current ``qpos`` (the position pipeline is recomputed
        first so the result is never a stale earlier configuration).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))

        with self._lock:
            # Reflect the CURRENT configuration. mj_jac* read data.xpos/site_xpos/
            # geom_xpos, data.subtree_com and data.cdof, all of which are stale if
            # qpos changed since the last forward (e.g. a direct data.qpos write or
            # set_joint_velocities). Recompute the position pipeline first so the
            # Jacobian is not silently that of an earlier pose. Matches
            # forward_kinematics; cheaper than a full mj_forward.
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            if body_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Body", body_name)}]}
                mj.mj_jacBody(model, data, jacp, jacr, obj_id)
                label = f"body '{body_name}'"
            elif site_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_SITE, site_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Site", site_name)}]}
                mj.mj_jacSite(model, data, jacp, jacr, obj_id)
                label = f"site '{site_name}'"
            elif geom_name:
                obj_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, geom_name)
                if obj_id < 0:
                    return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Geom", geom_name)}]}
                mj.mj_jacGeom(model, data, jacp, jacr, obj_id)
                label = f"geom '{geom_name}'"
            else:
                return {"status": "error", "content": [{"text": "Specify body_name, site_name, or geom_name."}]}

        return {
            "status": "success",
            "content": [
                {"text": f"Jacobian for {label}: pos={jacp.shape}, rot={jacr.shape}, nv={model.nv}"},
                {"json": {"jacp": jacp.tolist(), "jacr": jacr.tolist(), "nv": model.nv}},
            ],
        }

    # Energy

    def get_energy(self) -> dict[str, Any]:
        """Compute potential and kinetic energy of the system.

        Reflects the CURRENT configuration. ``mj_energyPos`` reads the
        position-stage derived state (``data.xipos`` for the gravitational
        term, spring/tendon lengths) and ``mj_energyVel`` reads the
        config-dependent inertia (the sparse joint-space inertia, itself
        position-stage derived)
        against ``data.qvel``. All of that is stale after a bare ``qpos``/
        ``qvel`` write (e.g. a direct ``data.qpos`` write from a planning/IK
        loop, or ``set_joint_velocities``), so the position pipeline is
        recomputed first - otherwise the reported energy is silently that of
        an earlier pose. Matches the defensive forward in ``get_mass_matrix``
        / ``inverse_dynamics``. The explicit ``mj_energyPos``/``mj_energyVel``
        calls are kept because ``mj_forward`` only recomputes ``data.energy``
        when ``mjENBL_ENERGY`` is enabled (it is not, by default), whereas the
        explicit calls populate ``data.energy`` unconditionally.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            mj.mj_forward(model, data)
            mj.mj_energyPos(model, data)
            mj.mj_energyVel(model, data)
            potential = float(data.energy[0])
            kinetic = float(data.energy[1])
        total = potential + kinetic

        return {
            "status": "success",
            "content": [
                {"text": f"Energy: potential={potential:.4f}J, kinetic={kinetic:.4f}J, total={total:.4f}J"},
                {"json": {"potential": potential, "kinetic": kinetic, "total": total}},
            ],
        }

    # Mass Matrix

    def get_mass_matrix(self) -> dict[str, Any]:
        """Compute the full mass (inertia) matrix M(q).

        M is nv×nv where nv is the number of DoFs.
        Useful for dynamics analysis, impedance control, etc.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        # The sparse inertia is only valid after a forward pass. Serialize the
        # forward+fullM read against concurrent policy threads (GH: concurrency
        # audit) so a sibling robot's mj_step can't mutate data mid-read.
        with self._lock:
            mj.mj_forward(model, data)
            nv = model.nv
            M = _full_mass_matrix(mj, model, data)
            if nv > 0:
                rank = int(np.linalg.matrix_rank(M))
                cond = float(np.linalg.cond(M)) if rank > 0 else float("inf")
            else:
                # Empty scene (no DOFs yet) - return a well-typed zero payload
                # instead of crashing in numpy on the empty matrix.
                rank = 0
                cond = float("inf")

        return {
            "status": "success",
            "content": [
                {"text": f"Mass matrix: {nv}×{nv}, rank={rank}, cond={cond:.2e}"},
                {
                    "json": {
                        "shape": [nv, nv],
                        "rank": rank,
                        "condition_number": cond,
                        "diagonal": np.diag(M).tolist(),
                        "total_mass": float(np.sum(model.body_mass)),
                    }
                },
            ],
        }

    # Inverse Dynamics

    def inverse_dynamics(self) -> dict[str, Any]:
        """Compute the generalized forces required to hold the current state.

        Runs ``mj_inverse`` for a target acceleration of zero, so the result
        is the gravity- and velocity-bias (Coriolis/centrifugal) compensation
        torques that keep the robot at its current ``qpos``/``qvel`` with zero
        acceleration - the standard inverse-dynamics query for a manipulator
        (at rest, pure gravity compensation).

        ``mj_inverse`` reads ``data.qacc`` as the *desired* acceleration, so
        this method runs ``mj_forward`` first (so the position/velocity
        kinematics match the current ``qpos``/``qvel``, matching the defensive
        forward in ``get_mass_matrix``) and zeroes ``qacc`` for the solve,
        restoring the buffer afterwards. Without this it would use whatever
        stale forward-dynamics acceleration was left in ``data.qacc`` and ask
        ``mj_inverse`` to reproduce free-fall - returning ~0 forces regardless
        of pose, never the compensation torques the query is for.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            # Establish consistent position/velocity kinematics for the current
            # state, then solve inverse dynamics for zero desired acceleration.
            mj.mj_forward(model, data)
            saved_qacc = data.qacc.copy()
            data.qacc[:] = 0.0
            try:
                mj.mj_inverse(model, data)
                # Build named force mapping
                forces = {}
                for i in range(model.njnt):
                    name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                    if name:
                        dof_adr = model.jnt_dofadr[i]
                        forces[name] = float(data.qfrc_inverse[dof_adr])
            finally:
                data.qacc[:] = saved_qacc

        return {
            "status": "success",
            "content": [
                {"text": f"Inverse dynamics: {len(forces)} joint forces computed"},
                {"json": {"qfrc_inverse": forces}},
            ],
        }

    # Body Introspection

    def get_body_state(
        self,
        body_name: str,
    ) -> dict[str, Any]:
        """Get the full state of a body: position, orientation, velocity, acceleration.

        Returns Cartesian pose + 6D spatial velocity (linear + angular),
        computed at the current ``qpos``/``qvel`` (the forward pipeline is run
        first so pose and velocity are never a stale earlier state).
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Body", body_name)}]}

        with self._lock:
            # Reflect the CURRENT qpos/qvel. This reads data.xpos/xquat/xmat/xipos
            # (position pipeline) and data.cvel via mj_objectVelocity (velocity
            # pipeline); both are stale if state changed since the last forward
            # (e.g. set_joint_velocities writes qvel without forwarding, or a
            # direct data.qpos write). Run the full pipeline so pose AND 6D
            # velocity are consistent with the current state. Matches
            # get_mass_matrix / inverse_dynamics / get_sensor_data.
            mj.mj_forward(model, data)
            # Position and orientation
            pos = data.xpos[body_id].tolist()
            quat = data.xquat[body_id].tolist()
            rotmat = data.xmat[body_id].reshape(3, 3).tolist()

            # Velocity (6D: angular then linear in world frame)
            vel = np.zeros(6)
            mj.mj_objectVelocity(model, data, mj.mjtObj.mjOBJ_BODY, body_id, vel, 0)
            linvel = vel[3:].tolist()
            angvel = vel[:3].tolist()

            # Mass and inertia
            mass = float(model.body_mass[body_id])
            com = data.xipos[body_id].tolist()

        state = {
            "position": pos,
            "quaternion": quat,
            "rotation_matrix": rotmat,
            "linear_velocity": linvel,
            "angular_velocity": angvel,
            "mass": mass,
            "center_of_mass": com,
        }

        text = (
            f"Body '{body_name}' (id={body_id}):\n"
            f"  pos: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]\n"
            f"  quat: [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]\n"
            f"  linvel: [{linvel[0]:.4f}, {linvel[1]:.4f}, {linvel[2]:.4f}]\n"
            f"  angvel: [{angvel[0]:.4f}, {angvel[1]:.4f}, {angvel[2]:.4f}]\n"
            f"  mass: {mass:.4f}kg, com: {com}"
        )

        return {"status": "success", "content": [{"text": text}, {"json": state}]}

    # Direct Joint Control

    def _resolve_joint_write_targets(
        self,
        values: dict[str, float],
        name: str,
        method: str,
    ) -> tuple[dict[str, int], dict[str, Any] | None]:
        """Resolve every joint name to a MuJoCo joint id before any state write.

        The dict form of :meth:`set_joint_positions` / :meth:`set_joint_velocities`
        used to skip names it could not resolve and still answer
        ``status="success"``, so a typo (or a namespaced name from the wrong
        robot) wrote nothing - or worse, wrote only part of the requested pose -
        while the caller was told the pose had been applied. Resolving up front
        makes the write all-or-nothing, matching the list form (which already
        rejects a joint-count mismatch) and ``send_action`` (which already
        rejects action keys it cannot resolve).

        Args:
            values: The ``{joint_name: value}`` mapping about to be written.
            name: Parameter name (``"positions"`` / ``"velocities"``), used in error text.
            method: Calling method name, used in error text.

        Returns:
            ``({joint_name: joint_id}, None)`` when every name resolves, else
            ``({}, error_dict)`` naming the unresolved names, the joints the
            model does have, and the discovery action - the structured-error
            tool contract, so the caller never raises past dispatch.
        """
        mj = _ensure_mujoco()
        if not values:
            return {}, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{method}: '{name}' is empty, so there is nothing to write. "
                            "Pass at least one joint (dict form) or a full ordered vector (list form); "
                            "use action='robot_joint_names' to see one robot's joints."
                        )
                    }
                ],
            }

        resolved: dict[str, int] = {}
        unresolved: list[str] = []
        for jnt_name in values:
            jnt_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                resolved[jnt_name] = jnt_id
            else:
                unresolved.append(jnt_name)
        if not unresolved:
            return resolved, None

        detail = self._unknown_mj_entity_msg("Joint", unresolved[0])
        if len(unresolved) > 1:
            detail = f"Unresolved '{name}' keys: {unresolved}. {detail}"
        return {}, {
            "status": "error",
            "content": [
                {
                    "text": (
                        f"{method}: {len(unresolved)} of {len(values)} '{name}' keys are not joints "
                        f"in this model, so nothing was written (the write is all-or-nothing). {detail}"
                    )
                }
            ],
        }

    def set_joint_positions(
        self,
        positions: dict[str, float] | list[float] | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Set joint positions directly (bypassing actuators).

        Writes to qpos and runs mj_forward to update kinematics.
        Useful for teleportation, IK solutions, or keyframe setting.

        Accepts EITHER form:

        * dict: {joint_name: value, ...} - explicit per-joint, safest in multi-robot scenes.
        * list/tuple: [v0, v1, ...] - ordered positional. Must match a single robot's
          joint count (when ``robot_name`` is given, that robot's joints; otherwise the
          world must contain exactly one robot, or the call errors).

        Every value must be a finite real number (Python or NumPy scalar), and
        must not be a boolean. A
        ``nan`` / ``inf``, a boolean or a non-numeric value returns a structured
        ``status="error"`` and leaves ``qpos`` untouched, rather than corrupting
        the kinematic state (``mj_forward`` propagates a ``nan`` everywhere) or
        raising past the tool-dispatch contract.

        The write is all-or-nothing: every dict key must name a joint of the
        model (verbatim or resolvable through a robot namespace). A key that
        does not resolve returns ``status="error"`` listing the model's joints
        and leaves ``qpos`` untouched -- a typo can no longer report success
        while silently applying a partial pose (or no pose at all). An empty
        mapping is likewise rejected instead of reporting a successful no-op.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        # mutating qpos under a running policy races mj_step
        if err := self._require_no_running_policy("set_joint_positions"):
            return err

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if positions is None:
            return {
                "status": "error",
                "content": [{"text": "set_joint_positions: 'positions' is required (list or dict of joint values)."}],
            }

        # normalize list input to dict using a deterministic joint ordering
        if isinstance(positions, (list, tuple)):
            robots = list(self._world.robots.values())
            if robot_name is not None:
                robots = [r for r in robots if r.name == robot_name]
                if not robots:
                    return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
            if len(robots) == 0:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": "set_joint_positions: list form requires a robot in the world; pass a dict instead, or add a robot first."
                        }
                    ],
                }
            if len(robots) > 1 and robot_name is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"set_joint_positions: list form is ambiguous with {len(robots)} robots; pass 'robot_name=' or use a dict."
                        }
                    ],
                }
            robot = robots[0]
            joint_names = list(getattr(robot, "joint_names", []) or [])
            if not joint_names:
                # Fall back: enumerate joints that belong to this robot via namespace
                ns = getattr(robot, "namespace", "") or ""
                joint_names = []
                for jid in range(model.njnt):
                    jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                    if jn and (not ns or jn.startswith(ns)):
                        joint_names.append(jn)
            if len(positions) != len(joint_names):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_joint_positions: list length {len(positions)} does not match robot "
                                f"'{robot.name}' joint count {len(joint_names)}. Use a dict for partial updates."
                            )
                        }
                    ],
                }
            positions = dict(zip(joint_names, positions, strict=True))
        elif not isinstance(positions, dict):
            return {
                "status": "error",
                "content": [
                    {"text": f"set_joint_positions: 'positions' must be a dict or list, got {type(positions).__name__}"}
                ],
            }

        # Validate every value is a finite number before any qpos write. Without
        # this a non-numeric entry raises ValueError past the structured-error
        # contract, and a nan/inf lands in data.qpos where mj_forward propagates
        # it across the whole kinematic state while the tool still reports success.
        positions, err = self._coerce_joint_state_map(positions, "positions", "set_joint_positions")
        if err:
            return err

        joint_ids, err = self._resolve_joint_write_targets(positions, "positions", "set_joint_positions")
        if err:
            return err

        with self._lock:
            for jnt_name, value in positions.items():
                qpos_adr = model.jnt_qposadr[joint_ids[jnt_name]]
                data.qpos[qpos_adr] = float(value)

            mj.mj_forward(model, data)

        count = len(positions)
        return {
            "status": "success",
            "content": [{"text": f"Set {count}/{count} joint positions, FK updated"}],
        }

    def set_joint_velocities(
        self,
        velocities: dict[str, float] | list[float] | None = None,
        robot_name: str | None = None,
    ) -> dict[str, Any]:
        """Set joint velocities directly.

        Writes to qvel. Useful for initializing dynamics. Accepts dict or list
        (see set_joint_positions for list semantics).

        Every value must be a finite real number (Python or NumPy scalar), and
        must not be a boolean. A
        ``nan`` / ``inf``, a boolean or a non-numeric value returns a structured
        ``status="error"`` and leaves ``qvel`` untouched, rather than blowing up
        the integrator on the next step or raising past the tool-dispatch contract.

        The write is all-or-nothing on the same terms as
        :meth:`set_joint_positions`: an unresolvable joint name (or an empty
        mapping) returns ``status="error"`` and leaves ``qvel`` untouched.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_joint_velocities"):
            return err

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if velocities is None:
            return {
                "status": "error",
                "content": [{"text": "set_joint_velocities: 'velocities' is required (list or dict)."}],
            }

        if isinstance(velocities, (list, tuple)):
            robots = list(self._world.robots.values())
            if robot_name is not None:
                robots = [r for r in robots if r.name == robot_name]
                if not robots:
                    return {"status": "error", "content": [{"text": self._unknown_robot_msg(robot_name)}]}
            if len(robots) == 0:
                return {
                    "status": "error",
                    "content": [{"text": "set_joint_velocities: list form requires a robot in the world."}],
                }
            if len(robots) > 1 and robot_name is None:
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": f"set_joint_velocities: list form is ambiguous with {len(robots)} robots; pass 'robot_name=' or use a dict."
                        }
                    ],
                }
            robot = robots[0]
            joint_names = list(getattr(robot, "joint_names", []) or [])
            if not joint_names:
                ns = getattr(robot, "namespace", "") or ""
                joint_names = []
                for jid in range(model.njnt):
                    jn = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, jid)
                    if jn and (not ns or jn.startswith(ns)):
                        joint_names.append(jn)
            if len(velocities) != len(joint_names):
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_joint_velocities: list length {len(velocities)} does not match robot "
                                f"'{robot.name}' joint count {len(joint_names)}. Use a dict for partial updates."
                            )
                        }
                    ],
                }
            velocities = dict(zip(joint_names, velocities, strict=True))
        elif not isinstance(velocities, dict):
            return {
                "status": "error",
                "content": [
                    {
                        "text": f"set_joint_velocities: 'velocities' must be a dict or list, got {type(velocities).__name__}"
                    }
                ],
            }

        # Validate every value is a finite number before any qvel write (see
        # set_joint_positions): a nan/inf velocity blows up the integrator on the
        # next step and a non-numeric entry escapes the structured-error contract.
        velocities, err = self._coerce_joint_state_map(velocities, "velocities", "set_joint_velocities")
        if err:
            return err

        joint_ids, err = self._resolve_joint_write_targets(velocities, "velocities", "set_joint_velocities")
        if err:
            return err

        with self._lock:
            for jnt_name, value in velocities.items():
                dof_adr = model.jnt_dofadr[joint_ids[jnt_name]]
                data.qvel[dof_adr] = float(value)

        count = len(velocities)
        msg = f"Set {count}/{count} joint velocities"
        return {
            "status": "success",
            "content": [{"text": msg}],
        }

    # Sensor Readout

    def get_sensor_data(self, sensor_name: str | None = None) -> dict[str, Any]:
        """Read sensor values from the simulation.

        MuJoCo supports: jointpos, jointvel, accelerometer, gyro, force,
        torque, touch, rangefinder, framequat, subtreecom, clock, etc.

        Args:
            sensor_name: Specific sensor name, or None for all sensors.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        if model.nsensor == 0:
            # distinguish "no sensors at all" from "that specific sensor not found"
            if sensor_name:
                return {
                    "status": "error",
                    "content": [{"text": f"Sensor '{sensor_name}' not found. Model has no sensors."}],
                }
            return {"status": "success", "content": [{"text": "No sensors in model."}]}

        # Lock while running mj_forward + reading sensordata so a policy
        # thread's mj_step can't mutate data between our forward pass and
        # the slice read. Also snapshot sensor metadata under the lock
        # because the model could theoretically change during this call
        # (it's not gated with _require_no_running_policy).
        with self._lock:
            mj.mj_forward(model, data)
            sensordata_snapshot = np.asarray(data.sensordata).copy()

        sensors = {}
        for i in range(model.nsensor):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_SENSOR, i)
            if not name:
                name = f"sensor_{i}"

            adr = model.sensor_adr[i]
            dim = model.sensor_dim[i]
            values = sensordata_snapshot[adr : adr + dim].tolist()

            if sensor_name and name != sensor_name:
                continue

            sensors[name] = {
                "values": values if dim > 1 else values[0],
                "dim": int(dim),
                "type": int(model.sensor_type[i]),
            }

        if sensor_name and not registered(sensors, sensor_name):
            return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Sensor", sensor_name)}]}

        lines = [f"Sensors ({len(sensors)}/{model.nsensor}):"]
        for name, info in sensors.items():
            lines.append(f"{name}: {info['values']} (dim={info['dim']})")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"sensors": sensors}}],
        }

    # Runtime Model Modification

    def set_body_properties(
        self,
        body_name: str,
        mass: float | None = None,
    ) -> dict[str, Any]:
        """Modify body properties at runtime (no recompile needed).

        Currently supports setting a body's ``mass``. Because a rigid body's
        inertia tensor tracks its mass at fixed geometry (a uniform density
        change), the body's ``body_inertia`` is scaled by the same ratio so the
        translational and rotational dynamics stay physically consistent.

        Changes take effect on the next ``mj_step``.

        Changes are recorded in the scene spec as well as the compiled model, so
        they survive the next scene recompile. The model is derived state that
        every scene mutation (``add_object`` / ``add_camera`` / ``add_robot``)
        rebuilds from the spec, so a value written only there would be restored
        to whatever the scene was compiled with - after this call had already
        reported the new one.

        Args:
            body_name: Name of the body to modify.
            mass: New absolute mass (kg); must be a finite number ``> 0``. When set, the body's
                inertia is scaled by ``mass / old_mass`` to preserve consistency.

        Returns:
            A tool-result dict; ``status="error"`` if the world is missing, a
            policy is running, ``mass`` is not a finite positive number, the body is
            not found, or the body has no mass of its own to scale (the world body
            declares no inertial and owns no geom), otherwise ``status="success"``
            summarizing the change.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_body_properties"):
            return err

        # mass must be > 0 (physics invariant). Shared with add_object so a
        # mass cannot be established at creation on terms this setter refuses.
        if mass is not None:
            if err := self._validate_mass(mass, "set_body_properties"):
                return err
            mass = float(mass)

        mj = _ensure_mujoco()
        model = self._world._model
        body_id = self._resolve_mj_name(mj.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Body", body_name)}]}

        changes = []
        with self._lock:
            if mass is not None:
                old_mass = float(model.body_mass[body_id])
                if old_mass <= 0:
                    # A mass change is applied as a scale (see below), which a
                    # body with no mass of its own cannot carry: there is no
                    # inertial and no geom whose density the ratio could move.
                    # The world body is the one such body in a normal scene.
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"set_body_properties: body '{body_name}' has no mass of its own "
                                    f"({old_mass:.3f} kg), so there is nothing to scale to {mass} kg. Only a "
                                    "body that declares an <inertial> or owns geoms carries a mass."
                                )
                            }
                        ],
                    }
                mass_ratio = mass / old_mass
                # model is DERIVED from the scene spec: the next scene mutation
                # recompiles the spec over it, so a mass written only here is
                # restored to the compiled value while the caller has already
                # been told the change took effect. Record it in the spec first,
                # so a scene that cannot carry the change is refused before
                # either representation is touched.
                if reason := persist_body_mass(self._world, body_id, mass_ratio=mass_ratio):
                    return {"status": "error", "content": [{"text": f"set_body_properties: {reason}"}]}
                model.body_mass[body_id] = mass
                changes.append(f"mass: {old_mass:.3f} → {mass:.3f}")
                # Inertia tracks mass for fixed geometry: setting a rigid body's
                # mass to a new value at constant shape is a uniform density
                # change, which scales its inertia tensor by the same factor
                # (I = integral of r^2 dm). Updating body_mass alone leaves a
                # physically inconsistent body - heavy in translation but with
                # the old rotational resistance - which silently corrupts the
                # rotational dynamics (and cannot be corrected by the caller,
                # since mass is the only settable property). Scale body_inertia
                # by the same ratio (matches randomize(randomize_physics=True)
                # and the Newton backend, which scale both together).
                model.body_inertia[body_id] *= mass_ratio

        return {
            "status": "success",
            "content": [{"text": f"Body '{body_name}': {', '.join(changes)}"}],
        }

    def set_geom_properties(
        self,
        geom_name: str | None = None,
        geom_id: int | None = None,
        color: list[float] | None = None,
        friction: list[float] | None = None,
        size: list[float] | None = None,
    ) -> dict[str, Any]:
        """Modify geom properties at runtime (no recompile needed).

        Changes take effect immediately for rendering (``color``) or on the next
        step (``friction``, ``size``). When ``size`` is changed on a size-defined
        primitive (sphere/capsule/cylinder/ellipsoid/box), the geom's collision
        bounding volumes (``geom_rbound`` for broadphase, ``geom_aabb`` for
        mid-phase) are recomputed so a grown geom collides correctly instead of
        letting other bodies pass through it. A plane's bounds are type-derived,
        so only its stored ``geom_size`` changes.

        Changes are recorded in the scene spec as well as the compiled model, so
        they survive the next scene recompile. The model is derived state that
        every scene mutation (``add_object`` / ``add_camera`` / ``add_robot``)
        rebuilds from the spec, so a value written only there would be restored
        to whatever the scene was compiled with - after this call had already
        reported the new one.

        A resize also changes what the owning body's geometry weighs and how it is
        balanced, so the body's mass, center of mass and inertia tensor - all
        integrated from its geoms at compile time, and never recomputed by a step -
        are re-derived from the resized shape. They are read from a compile of the
        persisted spec, so they equal the values the next scene recompile produces
        and the resize means the same thing whether or not anything follows it. A
        body that declares its own ``<inertial>`` takes nothing from geometry and is
        left alone. The cost is one spec compile per resize; the live model is not
        swapped, so entity ids and joint state are preserved.

        Every vector must carry the exact number of components its target
        defines, because there is no meaningful value to invent for a component
        the caller omitted:

        * ``color``: 3 (RGB, alpha set to 1.0) or 4 (RGBA).
        * ``friction``: 3 (sliding, torsional, rolling).
        * ``size``: whatever the geom's compiled type defines - 1 for a sphere,
          2 for a capsule/cylinder, 3 for a box/ellipsoid/plane. Mesh, height
          field and SDF geoms take their extent from asset data and define no
          ``geom_size`` component, so ``size`` is refused for them.
          A geom declared with ``<fromto>`` has the compiler fix its extent
          along that axis (and a box's / ellipsoid's cross-section), so a
          change to one of those components is refused - pass the value the
          compiler produces to resize the components it leaves alone.

        All numeric inputs are validated before any model write: ``color``,
        ``friction`` and ``size`` must contain only finite numbers (``nan`` /
        ``inf`` are rejected), ``friction`` coefficients must be ``>= 0`` and
        ``size`` half-extents must be ``> 0``. An invalid value returns a
        structured ``status="error"`` result and leaves the model untouched,
        rather than silently corrupting the solver or broadphase bounds, or
        applying a shape/appearance the caller never asked for.

        Args:
            geom_name: Name of the geom to modify. The owning object's name is
                accepted as an alias for an ``add_object`` geom (``"<name>"`` for
                ``"<name>_geom"``).
            geom_id: Geom id, as an alternative to ``geom_name``.
            color: RGB (3) or RGBA (4) components in ``[0, 1]``.
            friction: The three MuJoCo friction coefficients (sliding,
                torsional, rolling), each ``>= 0``.
            size: The geom's half-extents, with exactly as many components as
                its type defines, each ``> 0``.

        Returns:
            A tool-result dict; ``status="error"`` if the world is missing, a
            policy is running, the geom is not found, or a vector's values or
            component count cannot be honored, otherwise ``status="success"``
            summarizing the changes applied.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}
        if err := self._require_no_running_policy("set_geom_properties"):
            return err

        mj = _ensure_mujoco()
        model = self._world._model

        gid = geom_id
        if geom_name:
            gid = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, geom_name)
            # our add_object pipeline names geoms as ``{object_name}_geom``.
            # Accept the plain object name as a convenience alias.
            if (gid is None or gid < 0) and not geom_name.endswith("_geom"):
                gid = self._resolve_mj_name(mj.mjtObj.mjOBJ_GEOM, f"{geom_name}_geom")
        if gid is None or gid < 0 or gid >= model.ngeom:
            return {
                "status": "error",
                "content": [{"text": self._unknown_mj_entity_msg("Geom", str(geom_name or geom_id))}],
            }

        # Validate numeric inputs before any model write. Without this a nan/inf
        # (or negative) value lands directly in geom_rgba / geom_friction /
        # geom_size and silently corrupts rendering, the contact solver, or the
        # broadphase bounds (geom_rbound becomes inf) while the tool still
        # reports success. Friction coefficients are non-negative and a geom's
        # size (half-extent) must be strictly positive.
        #
        # Component counts are validated in the same pass. A vector shorter than
        # its target buffer used to be written component-wise (or padded with
        # zeros / an invented alpha), so a partial value silently mixed the
        # caller's components with the compiled ones - a one-element size on a
        # box resized only x and left y/z at their old half-extents, while a
        # one-element friction zeroed the torsional and rolling coefficients the
        # caller never mentioned. A longer vector had its tail discarded. Both
        # applied a shape / appearance / contact model nobody asked for under a
        # status="success" result, so the exact count is now required.
        if color is not None:
            color, err = _coerce_rgba(color, "set_geom_properties")
            if err:
                return err
        if friction is not None:
            friction, err = _coerce_finite_vector(
                friction,
                "friction",
                "set_geom_properties",
                min_value=0.0,
                accepted_lengths=(3,),
                layout="sliding, torsional, rolling",
            )
            if err:
                return err
        if size is not None:
            gtype = _geom_type_name(mj, model.geom_type[gid])
            geom_layout = _GEOM_SIZE_LAYOUTS.get(gtype)
            if geom_layout is None:
                # mesh / hfield / sdf: the extent comes from the asset, so no
                # component of the requested size can be honored. Storing it
                # anyway would report a resize that never happens.
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_geom_properties: geom '{geom_name or gid}' has type "
                                f"'{gtype}', whose extent comes from its asset data and "
                                "defines no 'size' component - resize the asset, or use a "
                                "size-defined primitive geom "
                                f"({', '.join(sorted(_GEOM_SIZE_LAYOUTS))}) instead."
                            )
                        }
                    ],
                }
            size, err = _coerce_finite_vector(
                size,
                "size",
                "set_geom_properties",
                min_value=0.0,
                strict_min=True,
                accepted_lengths=(geom_layout[0],),
                layout=f"{gtype}: {geom_layout[1]}",
            )
            if err:
                return err

            # A geom declared with <fromto> has part of its geom_size fixed by
            # the compiler rather than read from ``size``, and re-derived on
            # every compile. Writing such a component would report a resize the
            # next scene recompile discards, and would leave the body's inertial
            # row - re-derived below from the spec, which the endpoints still
            # govern - describing the old extent while the model collided as the
            # requested one. Refused for the same reason an asset-defined extent
            # is above; the components the fromto leaves alone still apply.
            # ``_coerce_finite_vector`` returns a value whenever it reports no
            # error, so the cast carries that proof rather than re-testing it.
            # It also fixed the length at this geom type's exact component count,
            # and every component a fromto fixes falls inside that count - 1 of a
            # capsule's / cylinder's 2, 1 and 2 of a box's / ellipsoid's 3 - so
            # each index below is in range. A bounds check here would stand in
            # for that proof while reading as though a short vector could arrive.
            requested = cast("list[float]", size)
            for index, (component, follows) in sorted(fromto_fixed_size_components(self._world, gid).items()):
                expected = float(requested[follows]) if follows is not None else float(model.geom_size[gid][index])
                if float(requested[index]) == expected:
                    continue
                return {
                    "status": "error",
                    "content": [
                        {
                            "text": (
                                f"set_geom_properties: geom '{geom_name or gid}' declares a <fromto>, "
                                f"so the compiler fixes its {component} (size component {index + 1} of "
                                f"a {gtype}) rather than reading it from 'size'. A change to it cannot "
                                f"be recorded durably: the next scene recompile restores {expected}. "
                                f"Pass {expected} for that component to resize the ones the fromto "
                                f"leaves alone, edit the fromto to resize along its axis, or declare "
                                f"the geom with an explicit size, pos and quat."
                            )
                        }
                    ],
                }

        label = geom_name or f"geom_{gid}"
        changes = []

        with self._lock:
            # model is DERIVED from the scene spec, which the next scene
            # mutation recompiles over it - so a value written only here is
            # discarded by the next add_object/add_camera/add_robot call and the
            # geom silently reverts after this call reported the new value.
            # Record it in the spec first, so a scene that cannot carry the
            # change is refused before either representation is touched.
            # Kept so a refresh that cannot be honored restores the spec to the
            # size the model is still compiled with, leaving the two in step.
            prior_size = None if size is None else model.geom_size[gid, : len(size)].tolist()
            if reason := persist_geom_properties(self._world, gid, color=color, friction=friction, size=size):
                return {"status": "error", "content": [{"text": f"set_geom_properties: {reason}"}]}

            if color is not None:
                # Already coerced to 4 components (RGB got an opaque alpha).
                model.geom_rgba[gid] = color
                changes.append(f"color → {model.geom_rgba[gid].tolist()}")

            if friction is not None:
                # Validated as exactly the three MuJoCo coefficients.
                model.geom_friction[gid] = friction
                changes.append(f"friction → {friction}")

            if size is not None:
                # A resize changes the shape the owning body's inertial row was
                # integrated from. Re-derive that row from the spec, which now
                # carries the new size, BEFORE touching the model: a scene whose
                # resized geometry cannot be compiled is refused with both
                # representations restored rather than left describing different
                # shapes. The reported result is then the one the next recompile
                # reproduces, so the resize does not depend on what follows it.
                if reason := refresh_body_inertial_from_geometry(self._world, gid):
                    persist_geom_properties(self._world, gid, size=prior_size)
                    return {"status": "error", "content": [{"text": f"set_geom_properties: {reason}"}]}

                # Validated as exactly the component count this geom's type
                # defines; the unused tail of the 3-wide row stays as compiled.
                model.geom_size[gid, : len(size)] = size
                # geom_rbound (broadphase) and geom_aabb (mid-phase) are derived
                # from geom_size at compile time and are not refreshed by the
                # solver; without recomputing them a grown geom keeps its old,
                # smaller collision bounds and other bodies silently pass through
                # it. Recompute both for size-defined primitives.
                _recompute_primitive_geom_bounds(mj, model, gid)
                changes.append(f"size → {model.geom_size[gid].tolist()}")

        return {
            "status": "success",
            "content": [{"text": f"Geom '{label}': {', '.join(changes)}"}],
        }

    # Contact Force Analysis

    def get_contact_forces(self) -> dict[str, Any]:
        """Get detailed contact forces for all active contacts.

        Uses ``mj_contactForce`` for each active contact pair; returns
        normal and friction forces.

        Runs ``mj_forward`` first (under the sim lock) so the contact list
        AND the constraint forces reflect the current ``qpos``/``qvel`` --
        exactly like :meth:`get_contacts`. Without it, a manual ``qpos``
        write (planning/IK loop), a pose set immediately after ``reset`` /
        ``add_robot``, or a policy thread mid-``mj_step`` leaves
        ``data.ncon`` / ``data.contact[]`` / ``data.efc_force`` stale, and
        this method would silently report phantom contacts with fabricated
        forces while still returning ``status=success``.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        contacts = []
        with self._lock:
            # Refresh contacts + constraint forces to the current qpos/qvel
            # (mirrors get_contacts). mj_contactForce reads data.efc_force,
            # which only the forward's constraint solve populates.
            mj.mj_forward(model, data)
            for i in range(data.ncon):
                c = data.contact[i]
                g1 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
                g2 = mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"

                # Get contact force (normal + friction in contact frame)
                force = np.zeros(6)
                mj.mj_contactForce(model, data, i, force)

                contacts.append(
                    {
                        "geom1": g1,
                        "geom2": g2,
                        "distance": float(c.dist),
                        "position": c.pos.tolist(),
                        "normal_force": float(force[0]),
                        "friction_force": force[1:3].tolist(),
                        "full_wrench": force.tolist(),
                    }
                )

        if not contacts:
            return {"status": "success", "content": [{"text": "No active contacts."}]}

        lines = [f"{len(contacts)} contacts:"]
        for c in contacts[:15]:
            lines.append(f"{c['geom1']} ↔ {c['geom2']}: normal={c['normal_force']:.3f}N, dist={c['distance']:.4f}m")
        if len(contacts) > 15:
            lines.append(f"  ... and {len(contacts) - 15} more")

        return {
            "status": "success",
            "content": [{"text": "\n".join(lines)}, {"json": {"contacts": contacts}}],
        }

    # Multi-Ray (batch raycasting)

    def multi_raycast(
        self,
        origin: list[float],
        directions: list[list[float]],
        exclude_body: int = -1,
    ) -> dict[str, Any]:
        """Cast multiple rays from a single origin (e.g., for LIDAR simulation).

        Efficiently casts N rays using individual mj_ray calls. Geom world poses
        are refreshed once (``mj_kinematics``) and the whole batch is cast under
        the sim lock, so every ray samples one consistent, current snapshot of
        the scene (see ``raycast``).

        The batch is all-or-nothing: every direction is validated before any ray
        is cast, and a direction that cannot be cast (wrong component count,
        non-numeric, nan/inf, zero-length) refuses the whole call, naming every
        offending index. Casting the rest and reporting ``distance: None`` for
        the rejected ones - the previous behavior - makes a ray that was never
        cast indistinguishable from a ray that found nothing, i.e. free space,
        which is the dangerous reading for the clearance and obstacle checks this
        method exists to serve. It matches ``raycast``, which refuses the same
        directions outright, so a caller does not get two contracts for one
        malformed vector.

        Args:
            origin: [x, y, z] ray start point in world frame.
            directions: One [dx, dy, dz] direction per ray, each auto-normalized.
                Must be a non-empty sequence of 3-component vectors (a bare
                string is refused rather than read as one ray per character).
            exclude_body: Body ID whose geoms every ray passes through (``-1`` =
                exclude nothing); see :meth:`raycast`.

        Returns:
            Standard status dict. On success the ``{"json": ...}`` block carries
            ``rays`` - one ``{"distance", "geom_id"}`` entry per direction, in
            order, with ``distance`` ``None`` for a genuine miss - plus ``hits``.
            On rejection it carries ``invalid_directions``, one
            ``{"index", "error"}`` entry per direction that cannot be cast.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        # validate origin shape
        try:
            if len(origin) != 3:
                return {
                    "status": "error",
                    "content": [{"text": f"multi_raycast: 'origin' must be 3 elements [x,y,z], got {len(origin)}"}],
                }
        except TypeError:
            return {"status": "error", "content": [{"text": "multi_raycast: 'origin' must be a list of 3 numbers"}]}

        # See raycast: reject non-numeric / nan / inf origin before np.array so
        # a bad element cannot raise past the tool contract or poison mj_ray.
        origin_f, err = _coerce_finite_vector(origin, "origin", "multi_raycast")
        if err is not None:
            return err

        rays, batch_err = _coerce_ray_batch(directions, "multi_raycast")
        if rays is None:
            return cast("dict[str, Any]", batch_err)

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        exclusion, err = _coerce_excluded_body(exclude_body, "multi_raycast", int(model.nbody))
        if err is not None:
            return err

        # Validate and normalize EVERY direction before casting anything, so a
        # batch that cannot be cast in full is refused rather than half-cast. The
        # loop collects all offending indices instead of returning on the first,
        # so one round-trip is enough to fix a whole malformed fan.
        vectors: list[np.ndarray] = []
        invalid: list[dict[str, Any]] = []
        for idx, d in enumerate(rays):
            param = f"directions[{idx}]"
            floats, d_err = _coerce_finite_vector(d, param, "multi_raycast", accepted_lengths=(3,), layout="dx, dy, dz")
            if d_err is not None:
                invalid.append({"index": idx, "error": d_err["content"][0]["text"]})
                continue
            vec = np.array(floats, dtype=np.float64)
            norm = float(np.linalg.norm(vec))
            if norm < 1e-10:
                invalid.append(
                    {
                        "index": idx,
                        "error": (f"multi_raycast: '{param}' is zero-length - supply a non-zero direction."),
                    }
                )
                continue
            vectors.append(vec / norm)

        if invalid:
            detail = "; ".join(f"[{entry['index']}] {entry['error']}" for entry in invalid)
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"multi_raycast: {len(invalid)} of {len(rays)} direction(s) cannot be "
                            f"cast, so no ray was cast (a rejected ray is not a miss): {detail}"
                        )
                    },
                    {"json": {"invalid_directions": invalid}},
                ],
            }

        pnt = np.array(origin_f, dtype=np.float64)
        results: list[dict[str, Any]] = []

        # Refresh geom world poses once, then serialize every mj_ray against a
        # policy thread's mj_step (see ``raycast``). Held for the whole batch so
        # all rays sample one consistent snapshot of the scene.
        with self._lock:
            mj.mj_kinematics(model, data)
            for vec in vectors:
                geomid = np.array([-1], dtype=np.int32)
                dist = mj.mj_ray(model, data, pnt, vec, None, 1, exclusion, geomid)
                results.append(
                    {
                        "distance": float(dist) if dist >= 0 else None,
                        "geom_id": int(geomid[0]) if dist >= 0 else None,
                    }
                )

        hit_count = sum(1 for r in results if r["distance"] is not None)
        return {
            "status": "success",
            "content": [
                {"text": f"Multi-ray: {hit_count}/{len(results)} hits from {origin}"},
                {"json": {"rays": results, "hits": hit_count}},
            ],
        }

    # Forward Kinematics (explicit)

    def forward_kinematics(self, body_name: str | None = None) -> dict[str, Any]:
        """Run forward kinematics to update all body positions/orientations.

        Usually called implicitly by mj_step, but useful after manually
        setting qpos to see updated Cartesian positions.

        If ``body_name`` is given, the response is filtered to that
        single body (and errors cleanly if the body doesn't exist).
        Otherwise returns every body as before.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        with self._lock:
            mj.mj_kinematics(model, data)
            mj.mj_comPos(model, data)
            mj.mj_camlight(model, data)

            if body_name is not None:
                bid = mj_name_to_id(model, mj.mjtObj.mjOBJ_BODY, body_name)
                if bid < 0:
                    return {"status": "error", "content": [{"text": self._unknown_mj_entity_msg("Body", body_name)}]}
                body_payload = {
                    "position": data.xpos[bid].tolist(),
                    "quaternion": data.xquat[bid].tolist(),
                }
                return {
                    "status": "success",
                    "content": [
                        {"text": f"FK for '{body_name}': pos={body_payload['position']}"},
                        {"json": {"body": body_name, **body_payload}},
                    ],
                }

            bodies = {}
            for i in range(model.nbody):
                name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
                bodies[name] = {
                    "position": data.xpos[i].tolist(),
                    "quaternion": data.xquat[i].tolist(),
                }

        return {
            "status": "success",
            "content": [
                {"text": f"FK computed for {model.nbody} bodies"},
                {"json": {"bodies": bodies}},
            ],
        }

    # Total Mass

    def get_total_mass(self) -> dict[str, Any]:
        """Get total mass and per-body mass breakdown."""
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        mj = _ensure_mujoco()
        model = self._world._model

        total = float(mj.mj_getTotalmass(model))
        bodies = {}
        for i in range(model.nbody):
            name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_BODY, i) or f"body_{i}"
            m = float(model.body_mass[i])
            if m > 0:
                bodies[name] = m

        return {
            "status": "success",
            "content": [
                {"text": f"Total mass: {total:.4f}kg ({len(bodies)} bodies with mass)"},
                {"json": {"total_mass": total, "bodies": bodies}},
            ],
        }

    def _ground_height_at(self, x: float, y: float) -> float:
        """Terrain surface height (world z) beneath world ``(x, y)``.

        Samples a ``create_world(terrain=...)`` MuJoCo ``<hfield>`` so a
        height-based locomotion predicate measures a base's clearance above the
        *local* terrain rather than an absolute world z. Returns ``0.0`` when no
        heightfield is present (a flat ground plane) and the hfield's base level
        for a point outside the terrain patch. Bilinearly interpolates the grid.
        The terrain ground geom is static (world-aligned, welded to the
        worldbody), so its pose and heightfield are constant after compile.
        """
        world = self._world
        if world is None or world._model is None or world._data is None:
            return 0.0
        model, data = world._model, world._data
        if model.nhfield == 0:
            return 0.0
        mj = _ensure_mujoco()
        hgeom = -1
        for g in range(model.ngeom):
            if model.geom_type[g] == mj.mjtGeom.mjGEOM_HFIELD:
                hgeom = g
                break
        if hgeom < 0:
            return 0.0
        hid = int(model.geom_dataid[hgeom])
        if hid < 0:
            return 0.0
        gx, gy, gz = (float(v) for v in data.geom_xpos[hgeom])
        rx, ry, elev = (float(v) for v in model.hfield_size[hid][:3])
        nrow = int(model.hfield_nrow[hid])
        ncol = int(model.hfield_ncol[hid])
        if nrow < 2 or ncol < 2 or rx <= 0.0 or ry <= 0.0:
            return gz
        adr = int(model.hfield_adr[hid])
        grid = np.asarray(model.hfield_data[adr : adr + nrow * ncol], dtype=float).reshape(nrow, ncol)
        # MuJoCo <hfield> userdata is row-major, row 0 -> min y and col 0 -> min
        # x, the grid spanning +-radius about the geom origin. Map (x, y) to a
        # fractional (row, col) and bilinearly interpolate the normalized height.
        u = (x - gx + rx) / (2.0 * rx)  # 0..1 across x (columns)
        v = (y - gy + ry) / (2.0 * ry)  # 0..1 across y (rows)
        if u < 0.0 or u > 1.0 or v < 0.0 or v > 1.0:
            return gz  # off the terrain patch -> its flush (base) level
        fc = u * (ncol - 1)
        fr = v * (nrow - 1)
        c0 = int(math.floor(fc))
        r0 = int(math.floor(fr))
        c1 = min(c0 + 1, ncol - 1)
        r1 = min(r0 + 1, nrow - 1)
        tc = fc - c0
        tr = fr - r0
        h = (
            grid[r0, c0] * (1.0 - tc) * (1.0 - tr)
            + grid[r0, c1] * tc * (1.0 - tr)
            + grid[r1, c0] * (1.0 - tc) * tr
            + grid[r1, c1] * tc * tr
        )
        return gz + elev * float(h)

    # Export Model XML

    def export_xml(self, output_path: str | None = None) -> dict[str, Any]:
        """Export the current scene as canonical MJCF via ``spec.to_xml()``.

        Every code path in the MjSpec backend stashes the live ``MjSpec`` in
        ``_backend_state["spec"]`` (``create_world`` / ``load_scene`` /
        ``replace_scene_mjcf`` / ``patch_scene_mjcf`` / the ``inject_*``
        helpers all do this). The serialised XML reflects any runtime
        mutation, so no extra caching or round-tripping is needed.

        ``output_path`` is treated as untrusted (LLM-callable tool): a ``..``
        traversal segment, a symlinked target, shell metacharacters, and
        backslash separators are rejected with ``status=error``. An absolute
        destination is accepted (the historic contract for this sink). The
        write is atomic and the success text reports the RESOLVED path. A
        destination the filesystem cannot accept (a directory, an unwritable
        parent) is reported the same way; a missing parent is created.
        """
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        spec = self._world._backend_state.get("spec") if self._world._backend_state else None
        if spec is None:
            # Should never happen in the MjSpec backend. Surfacing as an
            # error is better than a C-level crash via mj_saveLastXML.
            return {
                "status": "error",
                "content": [
                    {"text": "No MjSpec tracked on this world - cannot export. This is a bug; please file an issue."}
                ],
            }

        try:
            # spec.to_xml() emits benign "Attach conflict ... keeping parent
            # value" chatter (Python UserWarning + raw fd-2 writes) for every
            # attached robot scene. Every other call site wraps it; export_xml
            # must too, or the noise leaks straight to the user's console.
            with filter_mujoco_attach_noise():
                xml = spec.to_xml()
        except Exception as e:
            return {"status": "error", "content": [{"text": f"spec.to_xml() failed: {e}"}]}

        if output_path:
            # output_path is LLM-supplied (export_xml is an agent-callable
            # action): reject traversal, a symlinked target, and shell
            # metacharacters before writing. Guards-only (no sandbox root) keeps
            # the historic contract that an absolute destination is accepted -
            # unlike render(), whose output_path is documented as a newer,
            # sandboxed-by-design feature. The write is atomic so a crash
            # mid-export cannot truncate an existing file at the destination.
            try:
                safe = validate_output_path(output_path, sandbox_root=None, allow_abs=True)
            except ValueError as e:
                return {"status": "error", "content": [{"text": f"export_xml: {e}"}]}
            try:
                atomic_write_bytes(safe, xml.encode("utf-8"))
            except OSError as e:
                # A destination the caller supplied but the filesystem cannot
                # accept (a directory, an unwritable parent) is the same class
                # of caller error as an unsafe path, so it is reported through
                # the envelope rather than raised past it. strerror keeps the
                # internal temp filename out of the message.
                return {
                    "status": "error",
                    "content": [{"text": f"export_xml: cannot write {safe}: {e.strerror or e}"}],
                }
            # Report the RESOLVED path: the raw argument can normalize to a
            # different location, so echoing it would name a file we did not write.
            return {"status": "success", "content": [{"text": f"Model exported to {safe}"}]}

        return {
            "status": "success",
            "content": [{"text": f"Model XML ({len(xml)} chars):\n{xml[:2000]}{'...' if len(xml) > 2000 else ''}"}],
        }
