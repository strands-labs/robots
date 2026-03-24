"""GR00T policy — N1.5/N1.6 service and local inference.

Implements :class:`~strands_robots.policies.base.Policy` for NVIDIA GR00T models.

The Isaac-GR00T model operates on NESTED observation dicts::

    {
        "video": {"cam_name": np.ndarray(B, T, H, W, C)},
        "state": {"joint_group": np.ndarray(B, T, D)},
        "language": {"task": [["instruction"]]},
    }

and returns BARE action dicts::

    {"joint_group": np.ndarray(B, T, D)}

Our job: translate robot sensor names ↔ model modality keys via explicit
mappings.  No positional guessing.  One step in, one step out.
"""

import importlib.util
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import Gr00tInferenceClient
from .data_config import Gr00tDataConfig, load_data_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Isaac-GR00T version detection
# ---------------------------------------------------------------------------

_GROOT_VERSION: str | None = None  # "n1.5", "n1.6", or None


def _detect_groot_version(*, force: bool = False) -> str | None:
    """Auto-detect which Isaac-GR00T version (if any) is installed.

    Args:
        force: Re-detect even if a cached value exists.
    """
    global _GROOT_VERSION
    if _GROOT_VERSION is not None and not force:
        return _GROOT_VERSION

    # Reset before re-detection
    _GROOT_VERSION = None

    try:
        if importlib.util.find_spec("gr00t.policy.gr00t_policy") is not None:
            _GROOT_VERSION = "n1.6"
            logger.info("Detected Isaac-GR00T N1.6")
            return _GROOT_VERSION
    except (ModuleNotFoundError, ValueError):
        pass

    try:
        if importlib.util.find_spec("gr00t.model.policy") is not None:
            _GROOT_VERSION = "n1.5"
            logger.info("Detected Isaac-GR00T N1.5")
            return _GROOT_VERSION
    except (ModuleNotFoundError, ValueError):
        pass

    return None


# ---------------------------------------------------------------------------
# Mapping dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObservationMapping:
    """Maps robot sensor names → model modality keys.

    Attributes:
        video: ``{robot_camera: model_video_key}`` — bare, no prefix.
        state: ``{robot_state: model_state_key}`` — bare, no prefix.
        language_key: Model's language key (e.g. ``"task"``).
    """

    video: dict[str, str] = field(default_factory=dict)
    state: dict[str, str] = field(default_factory=dict)
    language_key: str = "task"

    def validate(self, modality_configs: dict) -> None:
        """Validate all mapped model keys exist in the model config."""
        model_video = set(modality_configs["video"].modality_keys)
        for robot_key, model_key in self.video.items():
            if model_key not in model_video:
                raise ValueError(
                    f"Observation mapping: robot '{robot_key}' → model video "
                    f"'{model_key}', but model only has: {sorted(model_video)}"
                )

        model_state = set(modality_configs["state"].modality_keys)
        for robot_key, model_key in self.state.items():
            if model_key not in model_state:
                raise ValueError(
                    f"Observation mapping: robot '{robot_key}' → model state "
                    f"'{model_key}', but model only has: {sorted(model_state)}"
                )

        model_lang = set(modality_configs["language"].modality_keys)
        if self.language_key not in model_lang:
            raise ValueError(
                f"Observation mapping: language_key '{self.language_key}' not in model: {sorted(model_lang)}"
            )


@dataclass(frozen=True)
class ActionMapping:
    """Maps model action keys → robot actuator names.

    Attributes:
        actions: ``{model_action_key: robot_actuator}`` — bare, no prefix.
    """

    actions: dict[str, str] = field(default_factory=dict)

    def validate(self, modality_configs: dict) -> None:
        """Validate all mapped model action keys exist in the model config."""
        model_action = set(modality_configs["action"].modality_keys)
        for model_key in self.actions:
            if model_key not in model_action:
                raise ValueError(f"Action mapping: model key '{model_key}' not in model: {sorted(model_action)}")


# ---------------------------------------------------------------------------
# Auto-inference (exact name match → positional fallback)
# ---------------------------------------------------------------------------


def _auto_infer_observation_mapping(
    data_config: Gr00tDataConfig,
    modality_configs: dict,
) -> ObservationMapping:
    """Auto-infer observation mapping from data_config + model config."""
    ours_v = [k.removeprefix("video.") for k in data_config.video_keys]
    model_v = list(modality_configs["video"].modality_keys)
    video_map = _match_keys(ours_v, model_v, "video")

    ours_s = [k.removeprefix("state.") for k in data_config.state_keys]
    model_s = list(modality_configs["state"].modality_keys)
    state_map = _match_keys(ours_s, model_s, "state")

    lang = modality_configs["language"].modality_keys[0]
    return ObservationMapping(video=video_map, state=state_map, language_key=lang)


def _auto_infer_action_mapping(
    data_config: Gr00tDataConfig,
    modality_configs: dict,
) -> ActionMapping:
    """Auto-infer action mapping from data_config + model config."""
    ours = [k.removeprefix("action.") for k in data_config.action_keys]
    model = list(modality_configs["action"].modality_keys)
    model_set = set(model)

    actions: dict[str, str] = {}
    used: set = set()
    for k in ours:
        if k in model_set:
            actions[k] = k
            used.add(k)
    remaining_ours = [k for k in ours if k not in actions.values()]
    remaining_model = [k for k in model if k not in used]
    for mdl, our in zip(remaining_model, remaining_ours):
        actions[mdl] = our
        logger.info("Auto-mapped action: model '%s' → robot '%s' (positional)", mdl, our)
    return ActionMapping(actions=actions)


def _match_keys(ours: list[str], model: list[str], label: str) -> dict[str, str]:
    """Match our keys to model keys: exact first, positional fallback."""
    model_set = set(model)
    mapping: dict[str, str] = {}
    used: set = set()
    for k in ours:
        if k in model_set:
            mapping[k] = k
            used.add(k)
    remaining_ours = [k for k in ours if k not in mapping]
    remaining_model = [k for k in model if k not in used]
    for our, mdl in zip(remaining_ours, remaining_model):
        mapping[our] = mdl
        logger.info("Auto-mapped %s: '%s' → '%s' (positional)", label, our, mdl)
    return mapping


# ---------------------------------------------------------------------------
# Parse user-provided flat mapping dicts
# ---------------------------------------------------------------------------


def _parse_observation_mapping(
    flat: dict[str, str],
    modality_configs: dict | None = None,
) -> ObservationMapping:
    """Parse ``{robot_key: "video.X" | "state.X"}`` → ObservationMapping."""
    video: dict[str, str] = {}
    state: dict[str, str] = {}

    for robot_key, model_key in flat.items():
        if model_key.startswith("video."):
            video[robot_key] = model_key.removeprefix("video.")
        elif model_key.startswith("state."):
            state[robot_key] = model_key.removeprefix("state.")
        else:
            raise ValueError(f"Mapping value must start with 'video.' or 'state.', got '{model_key}' for '{robot_key}'")

    lang = "task"
    if modality_configs is not None:
        lang = modality_configs["language"].modality_keys[0]

    return ObservationMapping(video=video, state=state, language_key=lang)


def _parse_action_mapping(flat: dict[str, str]) -> ActionMapping:
    """Parse ``{"action.X": "robot_key"}`` → ActionMapping."""
    return ActionMapping(actions={k.removeprefix("action."): v for k, v in flat.items()})


# ---------------------------------------------------------------------------
# Gr00tPolicy
# ---------------------------------------------------------------------------


class Gr00tPolicy(Policy):
    """GR00T policy — service mode and local inference (N1.5/N1.6).

    For **local mode**, loads the model directly and talks its native nested-dict
    format.  Robot↔model key translation is done by explicit mappings.

    For **service mode**, connects to a GR00T inference server via ZMQ.

    Args:
        data_config: Config name or :class:`Gr00tDataConfig`.
        host: Service host.
        port: Service port.
        model_path: HF model ID or local path (triggers local mode).
        embodiment_tag: Embodiment tag string.
        device: ``"cuda"`` or ``"cpu"``.
        groot_version: Force ``"n1.5"`` or ``"n1.6"``.
        strict: Strict input validation.
        api_token: ZMQ auth token.
        observation_mapping: ``{robot_key: "video.X" | "state.X"}``.
        action_mapping: ``{"action.X": "robot_key"}``.
        language_key: Override the model's language key.

    Examples::

        # Local N1.6 with explicit mapping
        policy = Gr00tPolicy(
            data_config="so100_dualcam",
            model_path="nvidia/GR00T-N1.6-3B",
            observation_mapping={
                "front": "video.front",
                "wrist": "video.wrist",
                "joint_position": "state.single_arm",
                "gripper_position": "state.gripper",
            },
            action_mapping={
                "action.single_arm": "joint_position",
                "action.gripper": "gripper_position",
            },
        )
    """

    def __init__(
        self,
        data_config: str | Gr00tDataConfig = "so100_dualcam",
        host: str = "localhost",
        port: int = 5555,
        model_path: str | None = None,
        embodiment_tag: str = "NEW_EMBODIMENT",
        device: str = "cuda",
        groot_version: str | None = None,
        strict: bool = False,
        api_token: str | None = None,
        observation_mapping: dict[str, str] | None = None,
        action_mapping: dict[str, str] | None = None,
        language_key: str | None = None,
        **kwargs,
    ):
        self.data_config = load_data_config(data_config)
        self.data_config_name = data_config if isinstance(data_config, str) else type(data_config).__name__

        self._local_policy: Any = None
        self._client: Gr00tInferenceClient | None = None
        self._groot_version = groot_version or _detect_groot_version()
        self._strict = strict

        # DOF per model state key — discovered from model at load time
        self._model_state_dof: dict[str, int] = {}

        # Raw user mappings (parsed after model load)
        self._raw_obs_mapping = observation_mapping
        self._raw_action_mapping = action_mapping
        self._language_key_override = language_key

        # Resolved mappings
        self._obs_mapping: ObservationMapping | None = None
        self._action_mapping: ActionMapping | None = None

        if model_path is not None:
            self._mode = "local"
            logger.info("GR00T local mode, model=%s", model_path)
            self._load_local_policy(model_path, embodiment_tag, device)
            self._init_mappings()
        else:
            self._mode = "service"
            logger.info("GR00T service mode, %s:%s", host, port)
            self._client = Gr00tInferenceClient(host=host, port=port, api_token=api_token)

        logger.info(
            "GR00T ready [mode=%s, version=%s, config=%s]",
            self._mode,
            self._groot_version or "service-only",
            self.data_config_name,
        )

    # ------------------------------------------------------------------
    # Mapping initialization
    # ------------------------------------------------------------------

    def _init_mappings(self) -> None:
        """Initialize observation/action mappings after model load."""
        if self._local_policy is None:
            return

        mmc = self._get_modality_configs()
        if mmc is None:
            logger.warning("Could not read model modality configs")
            return

        self._discover_model_state_dof(mmc)

        # Observation mapping
        if self._raw_obs_mapping is not None:
            self._obs_mapping = _parse_observation_mapping(self._raw_obs_mapping, mmc)
        else:
            self._obs_mapping = _auto_infer_observation_mapping(self.data_config, mmc)

        if self._language_key_override:
            self._obs_mapping = ObservationMapping(
                video=self._obs_mapping.video,
                state=self._obs_mapping.state,
                language_key=self._language_key_override,
            )

        self._obs_mapping.validate(mmc)

        # Action mapping
        if self._raw_action_mapping is not None:
            self._action_mapping = _parse_action_mapping(self._raw_action_mapping)
        else:
            self._action_mapping = _auto_infer_action_mapping(self.data_config, mmc)

        self._action_mapping.validate(mmc)

        logger.info(
            "Mappings: obs_video=%s, obs_state=%s, actions=%s",
            self._obs_mapping.video,
            self._obs_mapping.state,
            self._action_mapping.actions,
        )

    def _get_modality_configs(self) -> dict | None:
        """Get the model's per-embodiment modality configs."""
        try:
            if self._groot_version == "n1.6":
                # Direct N16Policy
                mmc = getattr(self._local_policy, "modality_configs", None)
                if mmc is not None:
                    return mmc
                # Wrapped via SimPolicyWrapper
                inner = getattr(self._local_policy, "policy", None)
                if inner is not None:
                    return getattr(inner, "modality_configs", None)
                return None
            elif self._groot_version == "n1.5":
                return getattr(self._local_policy, "modality_config", None)
        except (AttributeError, TypeError) as e:
            logger.debug("Could not read modality configs: %s", e)
        return None

    def _discover_model_state_dof(self, mmc: dict) -> None:
        """Discover DOF per state key from the loaded model.

        Sources (in priority order):
        1. Model normalizer stats
        2. Model processor norm_params

        If DOF cannot be discovered for a key, it is omitted and
        that key will not be zero-filled if unmapped.
        """
        self._model_state_dof = {}

        # Source 1: normalizer stats (N1.6)
        try:
            inner = getattr(self._local_policy, "policy", self._local_policy)
            normalizer = getattr(inner, "normalizer", None)
            if normalizer is not None:
                for key in mmc["state"].modality_keys:
                    stat = normalizer.get_stat(f"state.{key}")
                    if stat is not None and hasattr(stat, "shape"):
                        self._model_state_dof[key] = stat.shape[-1]
        except (AttributeError, TypeError):
            pass

        # Source 2: processor norm_params (N1.6)
        try:
            processor = getattr(self._local_policy, "processor", None)
            if processor is not None:
                sa = getattr(processor, "state_action_processor", None)
                if sa is not None and hasattr(sa, "norm_params"):
                    tag = self._local_policy.embodiment_tag.value
                    for key in mmc["state"].modality_keys:
                        if key not in self._model_state_dof:
                            params = sa.norm_params.get(tag, {}).get("state", {}).get(key, {})
                            if "dim" in params:
                                dim = params["dim"]
                                self._model_state_dof[key] = int(dim.item()) if hasattr(dim, "item") else int(dim)
        except (AttributeError, TypeError):
            pass

        discovered = set(self._model_state_dof.keys())
        all_keys = set(mmc["state"].modality_keys)
        missing = all_keys - discovered
        if missing:
            logger.warning(
                "Could not discover DOF for state keys: %s — these will not be zero-filled if unmapped",
                sorted(missing),
            )

        if self._model_state_dof:
            logger.info("Model state DOF: %s", self._model_state_dof)

    # ------------------------------------------------------------------
    # Model loading
    # ------------------------------------------------------------------

    def _load_local_policy(self, model_path: str, embodiment_tag: str, device: str):
        if self._groot_version == "n1.6":
            self._load_n16(model_path, embodiment_tag, device)
        elif self._groot_version == "n1.5":
            self._load_n15(model_path, embodiment_tag, device)
        else:
            raise ImportError("Isaac-GR00T not installed. Use service mode (host/port).")

    def _load_n15(self, model_path: str, embodiment_tag: str, device: str):
        from gr00t.experiment.data_config import DATA_CONFIG_MAP as N15_CONFIGS
        from gr00t.model.policy import Gr00tPolicy as N15Policy

        cfg_name = self.data_config_name if isinstance(self.data_config_name, str) else "so100_dualcam"
        native = N15_CONFIGS.get(cfg_name)
        mc = native.modality_config() if native else self.data_config.modality_config()
        mt = native.transform() if native else None

        kw = {
            "model_path": model_path,
            "embodiment_tag": embodiment_tag,
            "modality_config": mc,
            "modality_transform": mt,
            "device": device,
        }
        self._local_policy = N15Policy(**{k: v for k, v in kw.items() if v is not None})
        logger.info("GR00T N1.5 loaded from %s", model_path)

    def _load_n16(self, model_path: str, embodiment_tag: str, device: str):
        """Load N1.6 — uses Gr00tPolicy directly (NOT SimPolicyWrapper)."""
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as N16Policy

        tag = getattr(EmbodimentTag, embodiment_tag.upper(), EmbodimentTag.NEW_EMBODIMENT)
        self._local_policy = N16Policy(
            embodiment_tag=tag,
            model_path=model_path,
            device=device,
            strict=self._strict,
        )
        logger.info("GR00T N1.6 loaded from %s (direct)", model_path)

    # ------------------------------------------------------------------
    # Policy interface
    # ------------------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """No-op.  Mappings handle key translation."""

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs) -> list[dict[str, Any]]:
        if self._mode == "local":
            return self._local_get_actions(observation_dict, instruction)
        return self._service_get_actions(observation_dict, instruction)

    # ------------------------------------------------------------------
    # Local inference — talks model's native nested-dict format
    # ------------------------------------------------------------------

    def _local_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        """Local: prepare nested obs → infer → unpack actions."""
        nested_obs = self._prepare_observation(robot_obs, instruction)

        if self._groot_version == "n1.6":
            actions_raw, _ = self._local_policy.get_action(nested_obs)
        elif self._groot_version == "n1.5":
            actions_raw = self._local_policy.get_action(nested_obs)
        else:
            raise RuntimeError(f"Unknown GR00T version: {self._groot_version}")

        return self._unpack_actions(actions_raw)

    def _prepare_observation(self, robot_obs: dict[str, Any], instruction: str) -> dict:
        """Build the model's native nested-dict observation.

        Isaac-GR00T expects::

            {
                "video": {"key": np.ndarray(B=1, T=1, H, W, 3, uint8)},
                "state": {"key": np.ndarray(B=1, T=1, D, float32)},
                "language": {"key": [["instruction"]]},
            }
        """
        mmc = self._get_modality_configs()

        video_dict: dict[str, np.ndarray] = {}
        state_dict: dict[str, np.ndarray] = {}

        assert self._obs_mapping is not None, "Observation mapping not initialized"

        # ── Video ──
        mapped_video_keys = set(self._obs_mapping.video.keys())
        for robot_key, model_key in self._obs_mapping.video.items():
            if robot_key in robot_obs:
                video_dict[model_key] = _to_video_batch(robot_obs[robot_key])
            else:
                logger.warning("Robot key '%s' missing in obs", robot_key)

        if mmc is not None:
            for model_key in mmc["video"].modality_keys:
                if model_key not in video_dict:
                    ref = _reference_video_shape(robot_obs, mapped_video_keys)
                    video_dict[model_key] = np.zeros((1, 1, *ref), dtype=np.uint8)

        # ── State ──
        for robot_key, model_key in self._obs_mapping.state.items():
            if robot_key in robot_obs:
                state_dict[model_key] = _to_state_batch(robot_obs[robot_key])
            else:
                logger.warning("Robot key '%s' missing in obs", robot_key)

        # Zero-fill unmapped model state keys (only if DOF was discovered)
        if mmc is not None:
            for model_key in mmc["state"].modality_keys:
                if model_key not in state_dict:
                    dof = self._model_state_dof.get(model_key)
                    if dof is not None:
                        state_dict[model_key] = np.zeros((1, 1, dof), dtype=np.float32)
                    else:
                        logger.debug(
                            "Skipping zero-fill for '%s' — DOF unknown",
                            model_key,
                        )

        # ── Language ──
        lang_key = self._obs_mapping.language_key
        language_dict = {lang_key: [[instruction]]}

        return {
            "video": video_dict,
            "state": state_dict,
            "language": language_dict,
        }

    def _unpack_actions(self, raw_actions: dict) -> list[dict[str, Any]]:
        """Unpack model output → per-timestep robot actuator dicts."""
        squeezed: dict[str, np.ndarray] = {}
        for key, value in raw_actions.items():
            bare = key.removeprefix("action.")
            arr = np.asarray(value)
            while arr.ndim > 2:
                arr = arr[0]
            squeezed[bare] = arr

        if not squeezed:
            return []

        assert self._action_mapping is not None, "Action mapping not initialized"
        horizon = next(iter(squeezed.values())).shape[0]
        mapped_keys = set(self._action_mapping.actions.keys())

        actions: list[dict[str, Any]] = []
        for t in range(horizon):
            step: dict[str, Any] = {}
            for model_key, robot_key in self._action_mapping.actions.items():
                if model_key in squeezed:
                    step[robot_key] = squeezed[model_key][t]
            for model_key in squeezed:
                if model_key not in mapped_keys:
                    step[f"unmapped.{model_key}"] = squeezed[model_key][t]
            actions.append(step)

        return actions

    # ------------------------------------------------------------------
    # Service inference
    # ------------------------------------------------------------------

    def _service_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        """Service mode: build observation, call server, unpack."""
        assert self._client is not None, "Service client not initialized"
        if self._obs_mapping is not None:
            nested_obs = self._prepare_observation(robot_obs, instruction)
            action_chunk = self._client.get_action(nested_obs)
        else:
            obs = self._build_service_observation(robot_obs, instruction)
            action_chunk = self._client.get_action(obs)

        return self._unpack_service_actions(action_chunk)

    def _build_service_observation(self, robot_obs: dict[str, Any], instruction: str) -> dict:
        """Build flat-key observation for legacy service servers."""
        obs: dict = {}
        for vk in self.data_config.video_keys:
            bare = vk.removeprefix("video.")
            if bare in robot_obs:
                obs[vk] = robot_obs[bare]
        for sk in self.data_config.state_keys:
            bare = sk.removeprefix("state.")
            if bare in robot_obs:
                obs[sk] = np.asarray(robot_obs[bare], dtype=np.float32)
        if self.data_config.language_keys:
            obs[self.data_config.language_keys[0]] = instruction
        for k in obs:
            if isinstance(obs[k], np.ndarray):
                obs[k] = obs[k][np.newaxis, ...]
            else:
                obs[k] = [obs[k]]
        return obs

    def _unpack_service_actions(self, action_chunk: dict) -> list[dict[str, Any]]:
        """Unpack service response into per-timestep dicts.

        Applies ``_action_mapping`` if available (consistent with local mode),
        otherwise returns bare model keys.
        """
        normalized: dict = {}
        for key, value in action_chunk.items():
            bare = key.removeprefix("action.")
            arr = np.asarray(value)
            while arr.ndim > 2:
                arr = arr[0]
            normalized[bare] = arr

        if not normalized:
            return []

        horizon = next(iter(normalized.values())).shape[0]

        # If we have action mappings, use them for consistent key translation
        if self._action_mapping and self._action_mapping.actions:
            mapped_keys = set(self._action_mapping.actions.keys())
            actions: list[dict[str, Any]] = []
            for t in range(horizon):
                step: dict[str, Any] = {}
                for model_key, robot_key in self._action_mapping.actions.items():
                    if model_key in normalized:
                        row = normalized[model_key][t]
                        step[robot_key] = row.tolist() if hasattr(row, "tolist") else list(row)
                for model_key in normalized:
                    if model_key not in mapped_keys:
                        row = normalized[model_key][t]
                        step[f"unmapped.{model_key}"] = row.tolist() if hasattr(row, "tolist") else list(row)
                actions.append(step)
            return actions

        # No mapping — return bare model keys
        actions = []
        for t in range(horizon):
            step = {}
            for k, v in normalized.items():
                row = v[t]
                step[k] = row.tolist() if hasattr(row, "tolist") else list(row)
            actions.append(step)
        return actions


# ---------------------------------------------------------------------------
# Shape helpers — match Isaac-GR00T's expected formats exactly
# ---------------------------------------------------------------------------


def _to_video_batch(value: np.ndarray) -> np.ndarray:
    """Ensure video is (B=1, T=1, H, W, C) uint8."""
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 3:
        return arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 4:
        return arr[np.newaxis, ...]
    return arr


def _to_state_batch(value) -> np.ndarray:
    """Ensure state is (B=1, T=1, D) float32."""
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        return arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 2:
        return arr[np.newaxis, ...]
    return arr


def _reference_video_shape(
    robot_obs: dict[str, Any],
    video_keys: set | None = None,
) -> tuple:
    """Get reference video shape from mapped video observations.

    Only inspects keys listed in *video_keys* (the robot-side keys from the
    observation mapping).  Falls back to ``(256, 256, 3)`` if none match.

    Args:
        robot_obs: Robot observation dict.
        video_keys: Set of robot-side keys known to be video.  When *None*,
            falls back to heuristic scan (legacy behaviour).
    """
    if video_keys:
        for k in video_keys:
            v = robot_obs.get(k)
            if isinstance(v, np.ndarray) and v.ndim >= 3:
                return v.shape

    # Fallback: heuristic scan (only when video_keys not provided)
    if video_keys is None:
        for v in robot_obs.values():
            if isinstance(v, np.ndarray) and v.ndim >= 3 and v.shape[-1] in (1, 3, 4):
                return v.shape

    return (256, 256, 3)
