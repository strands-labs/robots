"""Declarative robot/sim ↔ LeRobot-model key mapping for the local policy.

This module replaces the per-step imperative observation/action remapping
heuristics in :mod:`policy` with a **declarative, validated, build-once**
mapping that plugs straight into LeRobot's own processor pipeline.

* :class:`EmbodimentMap` is a frozen dataclass describing how a robot/sim's
  native observation keys map onto the model's declared LeRobot feature keys
  (``observation.images.*`` / ``observation.state``) and how the model's action
  tensor maps back onto named robot actuators. It mirrors the GR00T
  ``Gr00tDataConfig`` pattern that already works in this codebase.

* :class:`PackStateProcessorStep` is the ONE new registered pipeline step: it
  composes the robot's scalar joint observations into ``observation.state`` in a
  declared order, with an **explicit** dim-reconciliation policy (no silent
  truncate/pad). It runs inside LeRobot's pipeline, right after the rename step.

* The map is built and **validated against the model's declared features once at
  load time** (fail-fast), then the pipeline owns every per-step transform.

Embodiment definitions live in ``embodiments.json`` next to this module and
support ``_extends`` inheritance + ``aliases`` (same loader shape as
``groot/data_configs.json``).
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# Dim reconciliation


def reconcile_dim(values: list[float], expected_dim: int, dim_policy: str, *, label: str = "state") -> list[float]:
    """Reconcile a value vector to ``expected_dim`` per an explicit policy.

    Unlike the old hot-path heuristics, this is called ONCE per step inside a
    single registered pipeline step and the policy is **declared** by the
    embodiment, never guessed.

    Args:
        values: The collected scalar values (ordered).
        expected_dim: The dimension the model declares for this feature.
        dim_policy: One of ``"strict"`` | ``"pad"`` | ``"truncate"``.
        label: Human label for error/log messages.

    Returns:
        A list of length ``expected_dim``.

    Raises:
        ValueError: If ``dim_policy == "strict"`` and lengths differ.
    """
    n = len(values)
    if n == expected_dim:
        return values
    if dim_policy == "strict":
        raise ValueError(
            f"{label} dim {n} != model expected {expected_dim}. "
            f"Set dim_policy='pad' or 'truncate' on the embodiment to opt in to adaptation."
        )
    if dim_policy == "truncate":
        if n < expected_dim:
            raise ValueError(f"{label} dim {n} < model expected {expected_dim}; cannot truncate. Use dim_policy='pad'.")
        return values[:expected_dim]
    if dim_policy == "pad":
        if n > expected_dim:
            raise ValueError(f"{label} dim {n} > model expected {expected_dim}; cannot pad. Use dim_policy='truncate'.")
        return values + [0.0] * (expected_dim - n)
    raise ValueError(f"Unknown dim_policy {dim_policy!r}; expected 'strict'|'pad'|'truncate'.")


def _convert_joint_vector(
    values: list[float],
    *,
    to_model: bool,
    gripper_index: int = -1,
    gripper_joint_range: list[float] | None = None,
    joint_mids: list[float] | None = None,
) -> list[float]:
    """Convert an ordered joint vector between sim units (radians + gripper joint
    range) and the LeRobot SO-arm training units (arm degrees, gripper 0..100).

    Shared by :class:`EmbodimentMap` (action side) and ``PackStateProcessorStep``
    (state side) so both directions use one implementation.

    * ``to_model=True``  sim -> model: arm radians -> degrees; gripper joint
      radians -> 0..100.
    * ``to_model=False`` model -> sim: arm degrees -> radians; gripper 0..100 ->
      joint radians.

    The gripper column (``gripper_index``) maps against ``gripper_joint_range``
    because the SO-arm gripper uses ``MotorNormMode.RANGE_0_100`` (0..100), not
    degrees - see ``lerobot/robots/so_follower/so_follower.py``.

    LeRobot's ``MotorNormMode.DEGREES`` is **mid-point-centered**: the value a
    checkpoint trains on is the angular displacement from each motor's
    calibration mid-point, not the absolute joint angle (ground truth:
    ``lerobot/motors/motors_bus.py`` ``_normalize`` / ``_unnormalize`` ->
    ``mid = (range_min + range_max) / 2``; reported degrees = ``(val - mid) *
    360 / max_res``). When ``joint_mids`` is supplied (per-joint mid offsets in
    degrees, aligned to ``values``), the arm conversion subtracts the mid going
    to the model and adds it back coming from the model, so the packed
    ``observation.state`` matches the distribution the checkpoint was trained on
    rather than being offset by each joint's mid. When ``joint_mids`` is empty
    (the default), the mid is treated as zero -- i.e. the sim ``qpos = 0`` is
    assumed to coincide with the calibration mid (absolute ``deg = rad *
    180/pi``), preserving the prior behavior.

    Args:
        values: Ordered joint values.
        to_model: Conversion direction (see above).
        gripper_index: Index of the gripper column, or -1 for none.
        gripper_joint_range: ``[min, max]`` radians of the sim gripper joint;
            empty/None treats the gripper like an arm joint (deg<->rad).
        joint_mids: Per-joint calibration mid-points in DEGREES, aligned to
            ``values``. Subtracted from arm columns when ``to_model`` and added
            back otherwise, matching ``motors_bus`` DEGREES mid-centering. The
            gripper column (``gripper_index``) is exempt (RANGE_0_100 has no
            mid). Empty/None / out-of-range indices use a mid of ``0.0``.

    Returns:
        A new list of converted values (input is not mutated).
    """
    out = list(values)
    rad_per_deg = float(np.pi) / 180.0
    grange = gripper_joint_range or []
    mids = joint_mids or []
    for i, v in enumerate(out):
        if i == gripper_index and len(grange) == 2:
            lo, hi = float(grange[0]), float(grange[1])
            span = hi - lo
            if span == 0.0:
                continue
            if to_model:
                out[i] = (float(v) - lo) / span * 100.0  # joint rad -> 0..100
            else:
                out[i] = lo + (float(v) / 100.0) * span  # 0..100 -> joint rad
        else:
            mid = float(mids[i]) if i < len(mids) else 0.0
            if to_model:
                out[i] = float(v) / rad_per_deg - mid  # radians -> mid-centered degrees
            else:
                out[i] = (float(v) + mid) * rad_per_deg  # mid-centered degrees -> radians
    return out


# Action diagnostics


def diagnose_action_dim(n_action_values: int, n_action_keys: int, *, name: str = "") -> str | None:
    """Return a warning message when a model action vector mis-matches the
    embodiment's declared actuator count, else ``None``.

    The local policy maps a model's action tensor onto robot actuators by index
    (``LerobotLocalPolicy._tensor_to_action_dicts``). When the model emits FEWER
    values than the embodiment declares actuator keys, the unmatched actuators
    are zero-filled -- which silently freezes those joints and looks exactly like
    "the policy runs but the robot does not move". When it emits MORE, the extra
    trailing values are dropped. Either case is almost always an
    embodiment/checkpoint mismatch the operator wants surfaced, not swallowed.

    Args:
        n_action_values: Length of the model's per-step action vector.
        n_action_keys: Number of declared actuator keys (``robot_state_keys``).
        name: Embodiment name for the message (optional).

    Returns:
        A human-readable warning string, or ``None`` when the dims match.
    """
    if n_action_values == n_action_keys:
        return None
    label = f" '{name}'" if name else ""
    if n_action_values < n_action_keys:
        missing = n_action_keys - n_action_values
        return (
            f"Policy action dim {n_action_values} < embodiment{label} actuator count "
            f"{n_action_keys}: the {missing} unmatched actuator(s) are zero-filled and will "
            f"not move. Check the embodiment's action_keys order/count against the "
            f"checkpoint's action dimension."
        )
    extra = n_action_values - n_action_keys
    return (
        f"Policy action dim {n_action_values} > embodiment{label} actuator count "
        f"{n_action_keys}: {extra} trailing action value(s) are dropped. Check the "
        f"embodiment's action_keys against the checkpoint's action dimension."
    )


class ZeroActionMonitor:
    """Detect a policy that keeps emitting near-zero actions (no robot motion).

    Even with correct action dims and units, a misconfigured obs/rename pipeline
    (a dropped camera key, an all-zero ``observation.state``) makes a VLA emit
    effectively-zero actions every step: the robot "runs the policy" but never
    moves. This monitor watches the per-step action magnitude and emits ONE
    warning when it stays below ``threshold`` for ``patience`` consecutive steps,
    pointing the operator at the embodiment / rename config.

    Stateful but dependency-free (no torch/lerobot) so it is unit-testable in
    isolation. Call :meth:`update` once per inference step and :meth:`reset` on
    episode reset.

    Attributes:
        threshold: Max-abs action magnitude below which a step counts as
            near-zero.
        patience: Consecutive near-zero steps required before warning.
    """

    def __init__(self, threshold: float = 1e-3, patience: int = 10) -> None:
        if threshold < 0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        if patience < 1:
            raise ValueError(f"patience must be >= 1, got {patience}")
        self.threshold = threshold
        self.patience = patience
        self._streak = 0
        self._warned = False

    def update(self, max_abs_action: float) -> str | None:
        """Record one step's max-abs action magnitude.

        Args:
            max_abs_action: ``max(abs(action))`` for this inference step.

        Returns:
            A warning string exactly once -- on the step where the near-zero
            streak first reaches ``patience`` -- and ``None`` otherwise. A single
            above-threshold step clears the streak and re-arms the warning.
        """
        if max_abs_action >= self.threshold:
            self._streak = 0
            self._warned = False
            return None
        self._streak += 1
        if self._streak >= self.patience and not self._warned:
            self._warned = True
            return (
                f"Policy emitted near-zero actions (max abs < {self.threshold:g}) for "
                f"{self._streak} consecutive steps: the robot will not move. This usually "
                f"means the observation never reached the model -- check the embodiment's "
                f"obs_rename / camera keys and that observation.state is populated."
            )
        return None

    def reset(self) -> None:
        """Reset streak + warned state (call on episode reset)."""
        self._streak = 0
        self._warned = False


# Registered pipeline step: pack scalar joint obs -> observation.state


# Imported lazily so this module is importable without lerobot (e.g. for unit
# testing EmbodimentMap loading/validation in a minimal env).
# Warn-once dedup for state_keys absent from the observation (keyed by the
# missing-key tuple, which is stable across the 50Hz control loop).
_WARNED_MISSING_STATE_KEYS: set[tuple[str, ...]] = set()


def _warn_missing_state_keys(missing: list[str]) -> None:
    """Warn once that some declared ``state_keys`` were absent (zero-filled in place)."""
    sig = tuple(missing)
    if sig in _WARNED_MISSING_STATE_KEYS:
        return
    _WARNED_MISSING_STATE_KEYS.add(sig)
    logger.warning(
        "lerobot_local: state_key(s) %s absent from the observation; zero-filled IN "
        "PLACE so the present joints keep their model index (the missing dims read 0). "
        "The canonical trigger is a mimic/tendon gripper actuator (e.g. left/gripper) "
        "whose sim observation exposes finger joints instead -- the arm proprioception "
        "stays aligned while the gripper state reads 0.",
        missing,
    )


def register_pack_state_step() -> type | None:
    """Define + register :class:`PackStateProcessorStep` against lerobot.

    Returns the step class, or ``None`` if lerobot's processor framework is
    unavailable. Idempotent: returns the already-registered class on re-call.
    """
    try:
        import torch
        from lerobot.processor.pipeline import ObservationProcessorStep, ProcessorStepRegistry
    except ImportError:
        logger.debug("lerobot processor framework unavailable; PackStateProcessorStep not registered")
        return None

    # Idempotent re-registration via the PUBLIC lookup. Reading the internal
    # ``ProcessorStepRegistry._registry`` dict couples us to a private attribute
    # LeRobot can rename/restructure any release (cf. TransitionKey moving
    # between 0.5.1 and 0.5.2). ``get(name)`` is the documented lookup; it
    # raises (KeyError/ValueError) or returns None when the name is unregistered
    # depending on the LeRobot version, so treat any miss as "not yet registered"
    # and fall through to the register decorator below.
    try:
        existing = ProcessorStepRegistry.get("strands_pack_state")
    except (KeyError, ValueError, AttributeError):
        existing = None
    if existing is not None:
        return existing

    @ProcessorStepRegistry.register(name="strands_pack_state")
    @dataclass
    class PackStateProcessorStep(ObservationProcessorStep):  # type: ignore[misc]
        """Compose declared scalar joint keys into ``observation.state``.

        Runs after the rename step and before normalization. If the observation
        already carries ``observation.state`` (e.g. a benchmark adapter or a
        natively-LeRobot obs), it passes through untouched (idempotent).

        Attributes:
            state_keys: Ordered robot/sim scalar keys composing the state vector.
            expected_dim: Model's declared ``observation.state`` dimension.
            dim_policy: ``"strict"`` | ``"pad"`` | ``"truncate"``.
        """

        state_keys: list[str] = field(default_factory=list)
        expected_dim: int = 0
        dim_policy: str = "strict"
        # Sim->model unit conversion (see EmbodimentMap). "degrees" => the sim's
        # radian joints are converted to the model's training units (arm
        # degrees, gripper 0..100) before packing observation.state.
        state_units: str = "native"
        gripper_index: int = -1
        gripper_joint_range: list[float] = field(default_factory=list)
        # Per-joint calibration mid-points in DEGREES (aligned to state_keys);
        # subtracted from arm columns so observation.state is mid-centered like
        # lerobot motors_bus DEGREES mode. Empty = mid 0 (prior behavior).
        joint_mids: list[float] = field(default_factory=list)

        def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
            if "observation.state" in observation:
                return observation  # already packed -> passthrough

            vals: list[float] = []
            missing: list[str] = []
            present = 0
            for k in self.state_keys:
                if k in observation:
                    present += 1
                    v = observation[k]
                    if isinstance(v, np.ndarray):
                        if v.ndim == 0:
                            vals.append(float(v))
                        else:
                            vals.extend(float(x) for x in v.ravel())
                    elif isinstance(v, (list, tuple)):
                        vals.extend(float(x) for x in v)
                    else:
                        vals.append(float(v))
                else:
                    # Zero-fill the missing key IN PLACE so it holds its slot.
                    # Skipping it (the previous behavior) collapsed the vector and
                    # shifted every following joint out of its model index -- the
                    # embodiment-path analog of the generic-path fix in
                    # LerobotLocalPolicy._collect_state_values.
                    vals.append(0.0)
                    missing.append(k)

            if present == 0:
                # No declared state key present at all; leave obs alone so a
                # clearer downstream error (or a state-less policy) can handle
                # it, rather than emitting an all-zero state vector.
                return observation

            if missing:
                _warn_missing_state_keys(missing)

            # Convert sim units (radians + gripper joint range) to the model's
            # training units (arm degrees, gripper 0..100) BEFORE packing, so the
            # model conditions on state in the space it was trained on. No-op
            # unless state_units == "degrees". See so_follower.py MotorNormMode.
            if self.state_units == "degrees":
                vals = _convert_joint_vector(
                    vals,
                    to_model=True,
                    gripper_index=self.gripper_index,
                    gripper_joint_range=self.gripper_joint_range,
                    joint_mids=self.joint_mids,
                )

            target = self.expected_dim or len(vals)
            vals = reconcile_dim(vals, target, self.dim_policy, label="observation.state")

            out = {k: v for k, v in observation.items() if k not in self.state_keys}
            # Emit a 1-D float32 torch Tensor (not a numpy array) so the
            # pipeline's own AddBatchDimensionObservationStep batches it to
            # (1, D) exactly as it batches image tensors. A numpy state is left
            # unbatched by that step, which on the declarative path stranded
            # observation.state at (D,) while images were (1, C, H, W) -> a
            # torch.stack rank mismatch inside the model. A real LeRobot dataset
            # feeds observation.state as a tensor too, so this matches the
            # convention the downstream steps (normalizer, batcher) expect.
            out["observation.state"] = torch.as_tensor(vals, dtype=torch.float32)
            return out

        def get_config(self) -> dict[str, Any]:
            return {
                "state_keys": list(self.state_keys),
                "expected_dim": self.expected_dim,
                "dim_policy": self.dim_policy,
            }

        def transform_features(self, features):  # type: ignore[no-untyped-def]
            # State vector composition doesn't change the model's declared
            # feature set (the normalizer already knows observation.state);
            # we only reshape the runtime obs. Pass features through.
            return features

    return PackStateProcessorStep


# Embodiment map


@dataclass(frozen=True)
class EmbodimentMap:
    """Declarative robot/sim ↔ model key mapping. Built + validated once.

    Attributes:
        name: Config identifier.
        obs_rename: ``{robot_obs_key: model_feature_key}`` for cameras (and any
            other direct passthroughs), e.g.
            ``{"image": "observation.images.image"}``. Fed into LeRobot's
            ``RenameObservationsProcessorStep.rename_map``.
        state_keys: Ordered scalar robot keys composing ``observation.state``.
        action_keys: Ordered robot actuator names for the action tensor's
            index→name mapping (output side).
        dim_policy: ``"strict"`` | ``"pad"`` | ``"truncate"`` for state dim.
    """

    name: str = ""
    obs_rename: dict[str, str] = field(default_factory=dict)
    state_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    dim_policy: str = "strict"
    # Unit conventions for state/action vectors. The MuJoCo sim expresses
    # revolute joints in RADIANS, but LeRobot SO-arm checkpoints (so100/so101,
    # MolmoAct2 etc.) are trained on the driver's MotorNormMode: arm joints in
    # DEGREES and the gripper in RANGE_0_100. "native" = no conversion (the
    # default; real-hardware *_real maps already speak the driver units).
    # "degrees" = arm columns are degrees + the gripper column is 0..100; the
    # policy converts deg<->rad and 0..100<->the gripper joint range when packing
    # state (model<-sim) and emitting actions (model->sim). See so_follower.py
    # (MotorNormMode.DEGREES for the arm, RANGE_0_100 for the gripper).
    state_units: str = "native"
    action_units: str = "native"
    # Index of the gripper column in state_keys/action_keys (RANGE_0_100, not a
    # degree joint). -1 = no special gripper column. SO arms = 5 (the 6th key).
    gripper_index: int = -1
    # The sim gripper joint's [min, max] radians, used to map the model's
    # 0..100 gripper command onto the joint range (and back). Empty = treat the
    # gripper like an arm joint (deg<->rad). SO arms: [-0.175, 1.745].
    gripper_joint_range: list[float] = field(default_factory=list)
    # Per-joint calibration mid-points in DEGREES, aligned to state_keys /
    # action_keys. LeRobot's MotorNormMode.DEGREES is mid-point-centered: a
    # checkpoint conditions on (joint_angle - calibration_mid), not the absolute
    # angle (ground truth: lerobot/motors/motors_bus.py mid = (min + max) / 2).
    # The sim expresses absolute angles, so without the mid the packed
    # observation.state is offset per joint from the training distribution and
    # can fall outside the dataset MIN_MAX range after normalization -> OOD.
    # When set, the "degrees" conversion subtracts the mid (sim -> model) and
    # adds it back (model -> sim). The gripper column (gripper_index) is exempt
    # (RANGE_0_100). Empty (default) = mid 0, i.e. sim qpos=0 is assumed to be
    # the calibration mid (the prior absolute-degrees behavior).
    joint_mids: list[float] = field(default_factory=list)

    def validate(self, input_features: dict[str, Any], output_features: dict[str, Any]) -> None:
        """Fail-fast validation against the model's declared features.

        Args:
            input_features: ``config.input_features`` from the loaded policy.
            output_features: ``config.output_features`` from the loaded policy.

        Raises:
            ValueError: On any mismatch (unknown rename target, wrong state/action dim).
        """
        # 1. Every rename target must be a declared model input feature.
        for src, dst in self.obs_rename.items():
            if dst not in input_features:
                raise ValueError(
                    f"Embodiment '{self.name}': obs_rename {src!r}->{dst!r} targets a feature "
                    f"the model doesn't declare. Model input_features: {sorted(input_features)}"
                )

        # 2. State dim check (only when the model declares observation.state).
        state_feat = input_features.get("observation.state")
        if state_feat is not None and getattr(state_feat, "shape", None):
            sdim = state_feat.shape[0]
            if self.state_keys and self.dim_policy == "strict" and len(self.state_keys) != sdim:
                raise ValueError(
                    f"Embodiment '{self.name}': {len(self.state_keys)} state_keys but model "
                    f"expects observation.state dim {sdim}. Fix state_keys or set "
                    f"dim_policy='pad'/'truncate' to opt in to adaptation."
                )

        # 3. Action dim check (only when the model declares an action feature).
        action_feat = output_features.get("action")
        if action_feat is not None and getattr(action_feat, "shape", None) and self.action_keys:
            adim = action_feat.shape[0]
            if len(self.action_keys) != adim:
                raise ValueError(
                    f"Embodiment '{self.name}': {len(self.action_keys)} action_keys but model "
                    f"action dim is {adim}. Action mapping would mis-index."
                )

    def _convert_vector(self, values: list[float], *, to_model: bool) -> list[float]:
        """Convert an ordered joint vector between sim (radians / sim units) and
        the model's training units (degrees + gripper RANGE_0_100).

        Applies only when ``units == "degrees"``; otherwise returns ``values``
        unchanged. Direction:

        * ``to_model=True``  sim -> model: arm radians -> degrees; gripper joint
          radians -> 0..100.
        * ``to_model=False`` model -> sim: arm degrees -> radians; gripper 0..100
          -> joint radians.

        The gripper column (``gripper_index``) is mapped against
        ``gripper_joint_range`` because the SO-arm gripper uses
        ``MotorNormMode.RANGE_0_100`` (0..100), not degrees - see
        ``lerobot/robots/so_follower/so_follower.py``.

        Args:
            values: Ordered joint values (length matches state_keys/action_keys).
            to_model: Conversion direction (see above).

        Returns:
            A new list of converted values (input is not mutated).
        """
        return _convert_joint_vector(
            values,
            to_model=to_model,
            gripper_index=self.gripper_index,
            gripper_joint_range=self.gripper_joint_range,
            joint_mids=self.joint_mids,
        )

    def sim_state_to_model(self, values: list[float]) -> list[float]:
        """Convert a sim state vector into the model's training units.

        No-op unless ``state_units == "degrees"``.
        """
        if self.state_units != "degrees":
            return list(values)
        return self._convert_vector(values, to_model=True)

    def model_action_to_sim(self, values: list[float]) -> list[float]:
        """Convert a model action vector into sim (radian) units.

        No-op unless ``action_units == "degrees"``.
        """
        if self.action_units != "degrees":
            return list(values)
        return self._convert_vector(values, to_model=False)

    def expected_state_dim(self, input_features: dict[str, Any]) -> int:
        """Return the model's declared state dim, or len(state_keys) if absent."""
        state_feat = input_features.get("observation.state")
        if state_feat is not None and getattr(state_feat, "shape", None):
            return state_feat.shape[0]
        return len(self.state_keys)


# JSON registry loader (with _extends inheritance + aliases)

_CONFIG_FILE = Path(__file__).parent / "embodiments.json"


def _resolve(name: str, definitions: dict) -> EmbodimentMap:
    """Resolve a definition name to an :class:`EmbodimentMap`, following ``_extends``.

    Keys beginning with a double underscore (e.g. ``__note__``, ``__doc__``) are
    treated as human-facing documentation/metadata and are stripped before
    constructing the dataclass, so the JSON can carry inline provenance notes
    (ground-truth source per robot) without breaking the loader.
    """
    definition = definitions[name]
    if "_extends" in definition:
        parent = _resolve(definition["_extends"], definitions)
        merged: dict[str, Any] = {
            "obs_rename": dict(parent.obs_rename),
            "state_keys": list(parent.state_keys),
            "action_keys": list(parent.action_keys),
            "dim_policy": parent.dim_policy,
        }
        for k, v in definition.items():
            if k != "_extends" and not k.startswith("__"):
                merged[k] = v
    else:
        merged = {k: v for k, v in definition.items() if not k.startswith("__")}
    merged["name"] = name
    return EmbodimentMap(**merged)


def _load_defs() -> tuple[dict, dict]:
    if not _CONFIG_FILE.exists():
        return {}, {}
    with open(_CONFIG_FILE) as fh:
        raw = json.load(fh)
    return raw.get("configs", {}), raw.get("aliases", {})


EMBODIMENT_MAP: dict[str, EmbodimentMap] = {}
_defs, _aliases = _load_defs()
for _cfg_name in _defs:
    EMBODIMENT_MAP[_cfg_name] = _resolve(_cfg_name, _defs)
for _alias, _target in _aliases.items():
    if _target in EMBODIMENT_MAP:
        EMBODIMENT_MAP[_alias] = EMBODIMENT_MAP[_target]
del _defs, _aliases


def load_embodiment(embodiment: str | EmbodimentMap | dict) -> EmbodimentMap:
    """Load an embodiment map by name, dict, or pass through an instance.

    Args:
        embodiment: Registry name (e.g. ``"panda_libero"``), an inline dict
            (``{"obs_rename": ..., "state_keys": ...}``), or an
            :class:`EmbodimentMap`.

    Returns:
        Resolved :class:`EmbodimentMap`.

    Raises:
        ValueError: If a string name is unknown.
    """
    if isinstance(embodiment, EmbodimentMap):
        return embodiment
    if isinstance(embodiment, dict):
        data = dict(embodiment)
        data.setdefault("name", "<inline>")
        return EmbodimentMap(**data)
    if isinstance(embodiment, str):
        if embodiment in EMBODIMENT_MAP:
            return EMBODIMENT_MAP[embodiment]
        raise ValueError(f"Unknown embodiment '{embodiment}'. Available: {sorted(EMBODIMENT_MAP)}")
    raise ValueError(f"embodiment must be str | dict | EmbodimentMap, got {type(embodiment)}")


__all__ = [
    "EmbodimentMap",
    "EMBODIMENT_MAP",
    "ZeroActionMonitor",
    "diagnose_action_dim",
    "load_embodiment",
    "reconcile_dim",
    "register_pack_state_step",
]
