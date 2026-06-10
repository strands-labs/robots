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


# Registered pipeline step: pack scalar joint obs -> observation.state


# Imported lazily so this module is importable without lerobot (e.g. for unit
# testing EmbodimentMap loading/validation in a minimal env).
def register_pack_state_step() -> type | None:
    """Define + register :class:`PackStateProcessorStep` against lerobot.

    Returns the step class, or ``None`` if lerobot's processor framework is
    unavailable. Idempotent: returns the already-registered class on re-call.
    """
    try:
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

        def observation(self, observation: dict[str, Any]) -> dict[str, Any]:
            if "observation.state" in observation:
                return observation  # already packed -> passthrough

            vals: list[float] = []
            for k in self.state_keys:
                if k in observation:
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

            if not vals:
                # No declared state keys present; leave obs alone so a clearer
                # downstream error (or a state-less policy) can handle it.
                return observation

            target = self.expected_dim or len(vals)
            vals = reconcile_dim(vals, target, self.dim_policy, label="observation.state")

            out = {k: v for k, v in observation.items() if k not in self.state_keys}
            out["observation.state"] = np.asarray(vals, dtype=np.float32)
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
    "load_embodiment",
    "reconcile_dim",
    "register_pack_state_step",
]
