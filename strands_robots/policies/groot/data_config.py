"""GR00T data configuration."""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

from ._data_config_defs import _ALIASES, _DATA_CONFIG_DEFS

logger = logging.getLogger(__name__)


@dataclass
class ModalityConfig:
    """Configuration for a modality (cameras, state, actions, language)."""

    delta_indices: List[int]
    modality_keys: List[str]

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"delta_indices": self.delta_indices, "modality_keys": self.modality_keys})


@dataclass
class DataConfig:
    """Base class for GR00T data configurations."""

    video_keys: List[str]
    state_keys: List[str]
    action_keys: List[str]
    language_keys: List[str]
    observation_indices: List[int]
    action_indices: List[int]

    def modality_config(self) -> Dict[str, ModalityConfig]:
        return {
            "video": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.video_keys),
            "state": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.state_keys),
            "action": ModalityConfig(delta_indices=self.action_indices, modality_keys=self.action_keys),
            "language": ModalityConfig(delta_indices=self.observation_indices, modality_keys=self.language_keys),
        }


_RESOLVED_CACHE: Dict[str, DataConfig] = {}


def _resolve_config(name: str) -> DataConfig:
    """Resolve a config name to a DataConfig, following _extends chains."""
    if name in _RESOLVED_CACHE:
        return _RESOLVED_CACHE[name]

    defn = _DATA_CONFIG_DEFS[name]
    if "_extends" in defn:
        parent = _resolve_config(defn["_extends"])
        merged = {
            "video_keys": list(parent.video_keys),
            "state_keys": list(parent.state_keys),
            "action_keys": list(parent.action_keys),
            "language_keys": list(parent.language_keys),
            "observation_indices": list(parent.observation_indices),
            "action_indices": list(parent.action_indices),
        }
        for k, v in defn.items():
            if k != "_extends":
                merged[k] = v
    else:
        merged = {k: v for k, v in defn.items()}

    config = DataConfig(**merged)
    _RESOLVED_CACHE[name] = config
    return config


DATA_CONFIG_MAP: Dict[str, DataConfig] = {}
for _name in _DATA_CONFIG_DEFS:
    DATA_CONFIG_MAP[_name] = _resolve_config(_name)
for _alias, _target in _ALIASES.items():
    DATA_CONFIG_MAP[_alias] = DATA_CONFIG_MAP[_target]


def load_data_config(data_config: Union[str, DataConfig]) -> DataConfig:
    """Load a data configuration from string name or return the object directly."""
    if isinstance(data_config, DataConfig):
        return data_config
    if isinstance(data_config, str):
        if data_config in DATA_CONFIG_MAP:
            return DATA_CONFIG_MAP[data_config]
        raise ValueError(f"Unknown data_config '{data_config}'. Available: {list(DATA_CONFIG_MAP.keys())}")
    raise ValueError(f"data_config must be str or DataConfig, got {type(data_config)}")


def create_custom_data_config(
    name: str,
    video_keys: List[str],
    state_keys: List[str],
    action_keys: List[str],
    language_keys: Optional[List[str]] = None,
    observation_indices: Optional[List[int]] = None,
    action_indices: Optional[List[int]] = None,
) -> DataConfig:
    """Create and register a custom data config at runtime."""
    config = DataConfig(
        video_keys=video_keys,
        state_keys=state_keys,
        action_keys=action_keys,
        language_keys=language_keys or ["annotation.human.task_description"],
        observation_indices=observation_indices or [0],
        action_indices=action_indices or list(range(16)),
    )
    DATA_CONFIG_MAP[name] = config
    logger.info(f"Registered custom config '{name}': cameras={video_keys} state={state_keys}")
    return config


__all__ = [
    "ModalityConfig",
    "DataConfig",
    "DATA_CONFIG_MAP",
    "load_data_config",
    "create_custom_data_config",
]
