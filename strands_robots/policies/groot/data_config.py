"""GR00T data configuration — typed embodiment key mappings.

Provides :class:`Gr00tDataConfig` dataclasses and an ``_extends`` inheritance
mechanism so new robot configs can be defined by overriding only what differs
from a parent.

Robot configurations are stored in ``data_configs.json`` alongside this module.
"""

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for a single modality (cameras, state, actions, language)."""

    delta_indices: list[int]
    modality_keys: list[str]

    def model_dump_json(self) -> str:
        """Serialize to JSON string (used by :class:`MsgSerializer`)."""
        return json.dumps({"delta_indices": self.delta_indices, "modality_keys": self.modality_keys})


@dataclass
class Gr00tDataConfig:
    """Typed representation of a GR00T embodiment data configuration.

    Attributes:
        name: Config identifier (e.g. "so100_dualcam").
        video_keys: Camera observation keys (e.g. ["video.front", "video.wrist"]).
        state_keys: Robot state keys (e.g. ["state.single_arm", "state.gripper"]).
        action_keys: Action output keys from the model.
        language_keys: Natural-language instruction keys.
        observation_indices: Temporal indices for observations.
        action_indices: Temporal indices for actions (horizon).
    """

    name: str = ""
    video_keys: list[str] = field(default_factory=list)
    state_keys: list[str] = field(default_factory=list)
    action_keys: list[str] = field(default_factory=list)
    language_keys: list[str] = field(default_factory=list)
    observation_indices: list[int] = field(default_factory=list)
    action_indices: list[int] = field(default_factory=list)

    def modality_config(self) -> dict[str, ModalityConfig]:
        """Build per-modality config dict (used by Isaac-GR00T loaders)."""
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }


# ---------------------------------------------------------------------------
# Config resolution with _extends inheritance
# ---------------------------------------------------------------------------


def _resolve_config(name: str, definitions: dict) -> Gr00tDataConfig:
    """Resolve a config name to a :class:`Gr00tDataConfig`, following ``_extends`` chains."""
    definition = definitions[name]

    if "_extends" in definition:
        parent = _resolve_config(definition["_extends"], definitions)
        merged: dict = {
            "video_keys": list(parent.video_keys),
            "state_keys": list(parent.state_keys),
            "action_keys": list(parent.action_keys),
            "language_keys": list(parent.language_keys),
            "observation_indices": list(parent.observation_indices),
            "action_indices": list(parent.action_indices),
        }
        for field_name, field_value in definition.items():
            if field_name != "_extends":
                merged[field_name] = field_value
    else:
        merged = {field_name: field_value for field_name, field_value in definition.items()}

    merged["name"] = name
    return Gr00tDataConfig(**merged)


# ---------------------------------------------------------------------------
# Load configs from JSON
# ---------------------------------------------------------------------------

_CONFIG_FILE = Path(__file__).parent / "data_configs.json"


def _load_config_defs() -> tuple:
    """Load config definitions and aliases from the JSON file."""
    with open(_CONFIG_FILE) as fh:
        raw = json.load(fh)
    return raw["configs"], raw.get("aliases", {})


# Pre-resolve all configs at import time
DATA_CONFIG_MAP: dict[str, Gr00tDataConfig] = {}
_defs, _aliases = _load_config_defs()
for _config_name in _defs:
    DATA_CONFIG_MAP[_config_name] = _resolve_config(_config_name, _defs)
for _alias_name, _target_name in _aliases.items():
    DATA_CONFIG_MAP[_alias_name] = DATA_CONFIG_MAP[_target_name]
del _defs, _aliases


def load_data_config(data_config: str | Gr00tDataConfig) -> Gr00tDataConfig:
    """Load a data configuration by name or pass through an existing instance.

    Args:
        data_config: Config name (e.g. "so100_dualcam") or a :class:`Gr00tDataConfig`.

    Returns:
        Resolved :class:`Gr00tDataConfig`.

    Raises:
        ValueError: If *data_config* is a string that doesn't match any known config,
            or if it is not a str or Gr00tDataConfig.
    """
    if isinstance(data_config, Gr00tDataConfig):
        return data_config
    if isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            return DATA_CONFIG_MAP[data_config]
        raise ValueError(f"Unknown data_config '{data_config}'. Available: {sorted(DATA_CONFIG_MAP)}")
    raise ValueError(f"data_config must be str or Gr00tDataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    video_keys: list[str],
    state_keys: list[str],
    action_keys: list[str],
    language_keys: list[str] | None = None,
    observation_indices: list[int] | None = None,
    action_indices: list[int] | None = None,
) -> Gr00tDataConfig:
    """Create and register a custom data config at runtime.

    The config is added to :data:`DATA_CONFIG_MAP` so it can be looked up by
    name via :func:`load_data_config`.
    """
    config = Gr00tDataConfig(
        name=name,
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys or ["annotation.human.task_description"],
        observation_indices=observation_indices or [0],
        action_indices=action_indices or list(range(16)),
    )
    DATA_CONFIG_MAP[name] = config
    logger.info("Registered custom config '%s': cameras=%s state=%s", name, video_keys, state_keys)
    return config


__all__ = [
    "ModalityConfig",
    "Gr00tDataConfig",
    "DATA_CONFIG_MAP",
    "load_data_config",
    "create_custom_data_config",
]
