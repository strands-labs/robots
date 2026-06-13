"""GR00T policy - N1.5/N1.6 service and local inference.

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
import os
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from strands_robots.policies.base import Policy

from .client import Gr00tInferenceClient
from .data_config import Gr00tDataConfig, load_data_config

logger = logging.getLogger(__name__)

# Isaac-GR00T version detection

_GROOT_VERSION: str | None = None  # "n1.5", "n1.6", "n1.7", or None


def _detect_groot_version(*, force: bool = False) -> str | None:
    """Auto-detect which Isaac-GR00T version (if any) is installed.

    Detection order (newest first):
      * **N1.7**: ``gr00t.model.gr00t_n1d7`` module (new VLM backbone package).
      * **N1.6**: ``gr00t.policy.gr00t_policy`` module exists but N1.7 signal absent.
      * **N1.5**: only ``gr00t.model.policy`` exists (legacy layout).

    N1.6 and N1.7 share the same ``gr00t.policy.gr00t_policy`` entry point,
    so we probe for the N1.7-specific ``gr00t_n1d7`` subpackage first.

    Args:
        force: Re-detect even if a cached value exists.
    """
    global _GROOT_VERSION
    if _GROOT_VERSION is not None and not force:
        return _GROOT_VERSION

    # Reset before re-detection
    _GROOT_VERSION = None

    # N1.7 first - the new Cosmos-Reason2-2B backbone lives here.
    # Detecting by subpackage (not enum values) keeps the probe cheap.
    try:
        if importlib.util.find_spec("gr00t.model.gr00t_n1d7") is not None:
            _GROOT_VERSION = "n1.7"
            logger.info("Detected Isaac-GR00T N1.7")
            return _GROOT_VERSION
    except (ModuleNotFoundError, ValueError):
        pass

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


# Mapping dataclasses


@dataclass(frozen=True)
class ObservationMapping:
    """Maps robot sensor names → model modality keys.

    Attributes:
        video: ``{robot_camera: model_video_key}`` - bare, no prefix.
        state: ``{robot_state: model_state_key}`` - bare, no prefix.
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
        actions: ``{model_action_key: robot_actuator}`` - bare, no prefix.
    """

    actions: dict[str, str] = field(default_factory=dict)

    def validate(self, modality_configs: dict) -> None:
        """Validate all mapped model action keys exist in the model config."""
        model_action = set(modality_configs["action"].modality_keys)
        for model_key in self.actions:
            if model_key not in model_action:
                raise ValueError(f"Action mapping: model key '{model_key}' not in model: {sorted(model_action)}")


# Auto-inference (exact name match → positional fallback)


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


# Parse user-provided flat mapping dicts


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


# Gr00tPolicy


class Gr00tPolicy(Policy):
    """GR00T policy - service mode and local inference (N1.5/N1.6).

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
        api_token: ZMQ auth token. Falls back to ``GROOT_API_TOKEN`` env var if not provided.
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

        # DOF per model state key - discovered from model at load time
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
            # Resolve api_token from env var if not provided as parameter
            resolved_token = api_token or os.environ.get("GROOT_API_TOKEN")
            self._client = Gr00tInferenceClient(host=host, port=port, api_token=resolved_token)

        logger.info(
            "GR00T ready [mode=%s, version=%s, config=%s]",
            self._mode,
            self._groot_version or "service-only",
            self.data_config_name,
        )

        # #187 wire-payload diagnostic: per-instance call counter so
        # ``_maybe_dump_wire_payload`` can cap dumps at
        # ``STRANDS_GROOT_WIRE_LOG_MAX_CALLS`` (default 10) without
        # filling the disk on long rollouts. Counter is shared across
        # LOCAL and SERVICE modes; the dump filename includes the mode
        # prefix so a single dir holds both paths' payloads side-by-side
        # for offline diff.
        self._wire_log_call_count: int = 0
        # Track whether we've already logged a "diagnostic disabled"
        # warning so we don't spam the eval log with one-warning-per-step
        # if the dump dir is unwritable.
        self._wire_log_disabled: bool = False

    # Mapping initialization

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
        """Get the model's per-embodiment modality configs.

        N1.6 and N1.7 expose ``modality_configs`` directly on ``Gr00tPolicy``
        (or via an optional ``PolicyWrapper``/``SimPolicyWrapper``).  N1.5 uses
        the singular ``modality_config`` attribute.
        """
        try:
            if self._groot_version in ("n1.6", "n1.7"):
                # Direct policy object
                mmc = getattr(self._local_policy, "modality_configs", None)
                if mmc is not None:
                    return mmc
                # Wrapped via PolicyWrapper (N1.7) or SimPolicyWrapper (N1.6)
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
                "Could not discover DOF for state keys: %s - these will not be zero-filled if unmapped",
                sorted(missing),
            )

        if self._model_state_dof:
            logger.info("Model state DOF: %s", self._model_state_dof)

    # Model loading

    def _load_local_policy(self, model_path: str, embodiment_tag: str, device: str):
        if self._groot_version == "n1.7":
            self._load_n17(model_path, embodiment_tag, device)
        elif self._groot_version == "n1.6":
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
        """Load N1.6 - uses Gr00tPolicy directly (NOT SimPolicyWrapper)."""
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

    def _load_n17(self, model_path: str, embodiment_tag: str, device: str):
        """Load N1.7 - identical entry point to N1.6 (same ``Gr00tPolicy`` signature).

        The user-visible policy class is still ``gr00t.policy.gr00t_policy.Gr00tPolicy``;
        internally it pulls the new Cosmos-Reason2-2B / Qwen3-VL backbone via
        ``gr00t.model.gr00t_n1d7``. Signature is backwards-compatible with N1.6,
        so we reuse the same kwargs.
        """
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy as N17Policy

        tag = getattr(EmbodimentTag, embodiment_tag.upper(), EmbodimentTag.NEW_EMBODIMENT)
        self._local_policy = N17Policy(
            embodiment_tag=tag,
            model_path=model_path,
            device=device,
            strict=self._strict,
        )
        logger.info("GR00T N1.7 loaded from %s (direct)", model_path)

    # Policy interface

    @property
    def provider_name(self) -> str:
        return "groot"

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """No-op.  Mappings handle key translation."""

    def reset(self, seed: int | None = None) -> None:
        """Per-episode reset.

        In SERVICE mode, forwards a ``reset`` call to the GR00T inference
        server so its per-episode RNG state (diffusion sampler noise,
        cuDNN benchmark state, etc.) can be re-initialised. Without this
        the server's RNG drifts across calls and produces different
        action chunks for byte-identical inputs across re-runs of the
        same eval — the #187 success-rate gap.

        The standard ``gr00t.eval.run_gr00t_server`` registers a ``reset``
        endpoint that maps to ``policy.reset(options=...)`` (see
        ``server_client.py:94``). The default ``Gr00tPolicy.reset`` upstream
        is a no-op; deployments that need per-episode RNG control should
        either patch the server (see
        ``examples/gr00t_server_deterministic_wrapper.py`` in robots-sim)
        or use the ``Robot()`` factory which auto-mounts the wrapper.

        In LOCAL mode, applies the same client-side reseed
        ``set_eval_seed`` would (Python / NumPy / torch / cuDNN), which
        is sufficient for reproducibility because the diffusion sampler
        runs in the same process as the client.

        Best-effort: any failure (server doesn't expose ``reset``,
        endpoint raises, network timeout) is logged and swallowed —
        ``reset`` is a soft hint to the policy, not a hard requirement.
        Eval correctness is preserved even if reset is a no-op.

        Args:
            seed: Master seed for the per-episode reset. When ``None``,
                no seed is forwarded (server uses its compiled-in default).
        """
        if self._mode == "service":
            assert self._client is not None, "service mode requires a client"
            try:
                # `options` is the standard kwarg the server's `reset`
                # endpoint maps to `policy.reset(options=options)`. We
                # pass the seed there so the server-side patch can
                # apply it.
                payload: dict[str, Any] = {}
                if seed is not None:
                    payload = {"options": {"seed": int(seed)}}
                self._client.call_endpoint("reset", payload if payload else None)
                logger.debug("Gr00tPolicy.reset: forwarded to server (seed=%r)", seed)
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.info(
                    "Gr00tPolicy.reset: server did not accept reset (seed=%r): %s; "
                    "continuing without per-episode server-side reseed",
                    seed,
                    e,
                )
            return

        # LOCAL mode: same reseed set_eval_seed would do. #331: delegate to the
        # shared helper so Gr00tPolicy and Cosmos3Policy reseed identically.
        if seed is None:
            return
        from strands_robots.policies._rng import reseed_client_rngs

        reseed_client_rngs(seed)
        logger.debug("Gr00tPolicy.reset: local-mode reseed applied (seed=%r)", seed)

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs) -> list[dict[str, Any]]:
        if self._mode == "local":
            return self._local_get_actions(observation_dict, instruction)
        return self._service_get_actions(observation_dict, instruction)

    # Local inference - talks model's native nested-dict format

    def _local_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        """Local: prepare nested obs → infer → unpack actions."""
        nested_obs = self._prepare_observation(robot_obs, instruction)

        if self._groot_version in ("n1.6", "n1.7"):
            # Both return (action_dict, info_dict) from get_action().
            actions_raw, _ = self._local_policy.get_action(nested_obs)
        elif self._groot_version == "n1.5":
            actions_raw = self._local_policy.get_action(nested_obs)
        else:
            raise RuntimeError(f"Unknown GR00T version: {self._groot_version}")

        # #187 wire-payload diagnostic: capture (nested_obs, actions_raw)
        # for offline diff against the SERVICE path. Zero overhead when
        # STRANDS_GROOT_WIRE_LOG is unset.
        self._maybe_dump_wire_payload("local", nested_obs, actions_raw)

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

        # Video
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

        # State
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
                            "Skipping zero-fill for '%s' - DOF unknown",
                            model_key,
                        )

        # Language
        lang_key = self._obs_mapping.language_key
        language_dict = {lang_key: [[instruction]]}

        # Match Isaac-GR00T training preprocessing for embodiments that need
        # it (#169) - same rotation that ``_build_service_observation``
        # applies, kept consistent so LOCAL and SERVICE inference modes
        # see identical observations. Pre-#169 the rotation was service-
        # only, which silently fed local-mode users upside-down or
        # reversed-direction images relative to training. The helper
        # operates on the 5D ``(1, 1, H, W, C)`` tensor directly via
        # negative-axis flips so the rotation always lands on H/W.
        if self.data_config.image_rotation_180:
            _apply_image_rotation_180_inplace(video_dict, list(video_dict.keys()))

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

    # Service inference

    def _service_get_actions(self, robot_obs: dict[str, Any], instruction: str) -> list[dict[str, Any]]:
        """Service mode: build observation, call server, unpack."""
        assert self._client is not None, "Service client not initialized"
        if self._obs_mapping is not None:
            wire_obs = self._prepare_observation(robot_obs, instruction)
            action_chunk = self._client.get_action(wire_obs)
        else:
            wire_obs = self._build_service_observation(robot_obs, instruction)
            action_chunk = self._client.get_action(wire_obs)

        # #187 wire-payload diagnostic: capture (wire_obs, action_chunk)
        # for offline diff against the LOCAL path. Zero overhead when
        # STRANDS_GROOT_WIRE_LOG is unset. Run an eval once with each
        # mode into the same dump dir, then ``np.allclose`` matching
        # ``call0`` files to bisect the divergence.
        self._maybe_dump_wire_payload("service", wire_obs, action_chunk)

        return self._unpack_service_actions(action_chunk)

    def _build_service_observation(self, robot_obs: dict[str, Any], instruction: str) -> dict:
        """Build flat-key observation for legacy service servers.

        Wire-format dimensions differ across server versions:

        * **N1.5 / N1.6** (default): video tensors are ``(B, H, W, C)`` and
          state tensors are ``(B, D)``. Single observation step per call,
          so leading ``B=1`` is sufficient.
        * **N1.7** (``self._groot_version == "n1.7"``): the
          ``gr00t.eval.run_gr00t_server`` entrypoint adds an explicit time
          axis, so video must be ``(B, T, H, W, C)`` and state must be
          ``(B, T, D)`` with ``T=1`` for one observation step. State
          tensors must additionally be ``np.float32`` (the server rejects
          ``float64``).

        Language values stay a ``list[str]`` of length ``B`` regardless of
        protocol version - the server matches it against the batch axis,
        not a time axis.

        Versioning is opt-in via the ``groot_version=`` constructor kwarg
        (or auto-detected from the *client*-side ``gr00t`` import). Service
        mode cannot introspect the remote server's version, so users
        targeting an N1.7 server must pass ``groot_version="n1.7"``
        explicitly when constructing the policy.
        """
        obs: dict = {}
        # Track which keys are video vs. state vs. other (language) so the
        # newaxis-fanout below stays type-safe per category.
        video_keys: list[str] = []
        state_keys: list[str] = []

        for vk in self.data_config.video_keys:
            bare = vk.removeprefix("video.")
            if bare in robot_obs:
                obs[vk] = robot_obs[bare]
                video_keys.append(vk)
        # Match Isaac-GR00T training preprocessing for embodiments that need
        # it - see :func:`_apply_image_rotation_180_inplace` for the algebra.
        # #169 moved the inline implementation into the shared helper and
        # added a parallel call in :meth:`_prepare_observation` so
        # local-mode inference applies the same rotation.
        if self.data_config.image_rotation_180:
            _apply_image_rotation_180_inplace(obs, video_keys)
        for sk in self.data_config.state_keys:
            bare = sk.removeprefix("state.")
            if bare in robot_obs:
                arr = np.asarray(robot_obs[bare], dtype=np.float32)
                # Scalars (joint readings, gripper pose components, …)
                # arrive as 0-D arrays. Promote to (D=1,) so the newaxis
                # loop below produces the canonical (B, [T,] D) shape
                # rather than (B, [T,]) that breaks the n1.7 server.
                if arr.ndim == 0:
                    arr = arr[np.newaxis]
                obs[sk] = arr
                state_keys.append(sk)
        if self.data_config.language_keys:
            obs[self.data_config.language_keys[0]] = instruction

        # Add the leading batch (and time, for n1.7) axes. Language and any
        # non-ndarray values stay as B-length list[str] regardless of
        # version - the server matches them against batch, not time.
        n_lead = 2 if self._groot_version == "n1.7" else 1
        for k in list(obs.keys()):
            v = obs[k]
            if isinstance(v, np.ndarray):
                for _ in range(n_lead):
                    v = v[np.newaxis, ...]
                obs[k] = v
            else:
                obs[k] = [v]
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

        # No mapping - return bare model keys
        actions = []
        for t in range(horizon):
            step = {}
            for k, v in normalized.items():
                row = v[t]
                step[k] = row.tolist() if hasattr(row, "tolist") else list(row)
            actions.append(step)
        return actions

    # Wire-payload diagnostic

    def _maybe_dump_wire_payload(
        self,
        mode: str,
        observation: dict[str, Any],
        action_chunk: dict[str, Any],
    ) -> None:
        """Dump the pre-inference observation and post-inference action
        chunk to disk for offline diff between LOCAL and SERVICE paths.

        Gated on ``STRANDS_GROOT_WIRE_LOG=<dir>``. When unset, this is a
        zero-overhead no-op (one ``os.environ.get`` per call). When set,
        dumps a pickle file per call to ``<dir>/{mode}_call{N:04d}.pkl``
        until ``STRANDS_GROOT_WIRE_LOG_MAX_CALLS`` (default 10) is hit.

        Pickle file structure::

            {
                "mode": "local" | "service",
                "call_index": int,
                "observation": dict,    # nested for local, flat for service
                "action_chunk": dict,   # raw model / server output (pre _unpack_*)
                "groot_version": str | None,
                "data_config_name": str,
            }

        Used by the #187 bisection plan to verify whether LOCAL and
        SERVICE paths send byte-identical observations to the model.
        Run an eval twice (once with each mode) into the same dump dir,
        then ``pickle.load`` the matching ``call0`` files and ``np.allclose``
        the per-key tensors. Any divergence at step 0 is the bug
        (assuming wire format is identical, which the regression test
        in this PR pins).

        Best-effort: if the dump dir doesn't exist or isn't writable,
        log ONE warning and disable further dumps for the rest of the
        process. Diagnostic instrumentation must never crash production
        eval. The user gets a one-line warning early; subsequent calls
        are silent no-ops.

        :param mode: ``"local"`` or ``"service"``. Becomes the filename prefix.
        :param observation: The pre-inference observation dict. For LOCAL
            this is the nested-dict format; for SERVICE this is the flat
            wire format AFTER newaxis fanout (i.e. what would have been
            msgpack-packed onto the wire).
        :param action_chunk: The raw post-inference action chunk dict
            (before ``_unpack_actions`` / ``_unpack_service_actions``
            squashes axes for the per-step dispatch loop).
        """
        # Tolerate construction paths that bypass ``__init__`` (test
        # fixtures that ``Gr00tPolicy.__new__(Gr00tPolicy)`` then stuff
        # attributes manually). Diagnostic instrumentation must never
        # crash production OR test paths.
        if getattr(self, "_wire_log_disabled", False):
            return
        log_dir = _wire_log_dir()
        if log_dir is None:
            return
        call_count = getattr(self, "_wire_log_call_count", 0)
        if call_count >= _wire_log_max_calls():
            return

        import pickle

        path = os.path.join(log_dir, f"{mode}_call{call_count:04d}.pkl")
        payload = {
            "mode": mode,
            "call_index": call_count,
            "observation": observation,
            "action_chunk": action_chunk,
            "groot_version": getattr(self, "_groot_version", None),
            "data_config_name": getattr(self, "data_config_name", "unknown"),
        }
        try:
            os.makedirs(log_dir, exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        except (OSError, pickle.PicklingError) as e:
            # Disk full / dir not writable / permission denied / un-picklable
            # object in the payload (e.g. a thread-local). Log once and
            # disable further dumps so the eval doesn't spam the log
            # with one warning per step.
            logger.warning(
                "STRANDS_GROOT_WIRE_LOG=%r: failed to dump wire payload to %r (%s); "
                "disabling diagnostic for this process",
                log_dir,
                path,
                e,
            )
            self._wire_log_disabled = True
            return

        self._wire_log_call_count = call_count + 1
        if self._wire_log_call_count == 1:
            # First successful dump: log INFO so the user sees confirmation
            # the diagnostic is active. Subsequent dumps are silent.
            logger.info(
                "STRANDS_GROOT_WIRE_LOG=%r: dumping wire payloads (mode=%s, max=%d). First file: %s",
                log_dir,
                mode,
                _wire_log_max_calls(),
                path,
            )


# Wire-payload diagnostic (#187) - dump pre-inference observation +
# post-inference action chunk to disk for offline diff between LOCAL
# and SERVICE paths.


def _wire_log_dir() -> str | None:
    """Return the directory where wire payloads should be dumped, or None.

    Reads ``STRANDS_GROOT_WIRE_LOG``. Returns ``None`` when unset or empty,
    in which case the dumper is a no-op (zero overhead in production eval).

    Used by :meth:`Gr00tPolicy._maybe_dump_wire_payload` to gate the dump
    so production paths pay no cost when the diagnostic is off.
    """
    path = os.environ.get("STRANDS_GROOT_WIRE_LOG", "").strip()
    return path or None


def _wire_log_max_calls() -> int:
    """Return the cap on number of wire-payload dumps per process.

    Reads ``STRANDS_GROOT_WIRE_LOG_MAX_CALLS``. Defaults to ``10`` so a
    full LIBERO eval (5 episodes × 720 steps / 8 chunk = ~450 calls)
    doesn't fill the disk with multi-GB pickle archives.

    The user's bisection plan from #187 only needs the first few calls
    to detect a divergence between LOCAL and SERVICE wire payloads.
    """
    raw = os.environ.get("STRANDS_GROOT_WIRE_LOG_MAX_CALLS", "10").strip()
    try:
        return max(1, int(raw))
    except ValueError:
        logger.warning(
            "STRANDS_GROOT_WIRE_LOG_MAX_CALLS=%r is not an integer; defaulting to 10",
            raw,
        )
        return 10


# Shape helpers - match Isaac-GR00T's expected formats exactly


def _apply_image_rotation_180_inplace(obs: dict[str, Any], video_keys: list[str]) -> None:
    """Apply 180° H/W rotation to ``video_keys`` in ``obs``.

    Match Isaac-GR00T training preprocessing for embodiments that need it.
    The GR00T-N1.7-LIBERO checkpoint was trained on data the upstream
    pipeline rotates 180° via Isaac-GR00T's
    ``examples/Libero/eval/utils.py:get_libero_image()``. Without this
    rotation at eval time, every observation the policy sees is
    upside-down relative to its training distribution and the success
    rate collapses to 0 (#168 bug H, re-broken in service mode
    by #168, fixed by #169).

    Producers (``LiberoAdapter.augment_observation``) are expected to
    deliver images in OpenGL framebuffer convention (bottom-row-zero).
    This helper applies the second 180° to convert OpenGL → training
    convention, matching what NVIDIA's reference eval does.

    Operates IN PLACE on ``obs``: the rotated array replaces the
    original entry with an ``np.ascontiguousarray`` view (downstream
    msgpack / ``np.tobytes()`` requires C-contiguous memory; reversed
    views are not contiguous).

    Handles any-dim input where H and W are the trailing-3rd and
    trailing-2nd axes (e.g. raw 3D ``(H, W, C)`` or batched 4D
    ``(B, H, W, C)`` / 5D ``(B, T, H, W, C)``). Uses ``np.flip`` with
    a negative-axis tuple so the rotation lands on H/W regardless of
    whether the leading B/T axes have been added yet.

    Called from BOTH service-mode (``_build_service_observation``) and
    local-mode (``_prepare_observation``) paths so the rotation is
    applied consistently regardless of inference transport. Pre-#169 it
    was service-only, which made the LOCAL path silently OOD relative
    to training (engine outputs OpenGL convention, policy applies no
    rotation → upside-down input).
    """
    for vk in video_keys:
        v = obs.get(vk)
        if isinstance(v, np.ndarray) and v.ndim >= 3:
            # Negative-axis indexing handles 3D / 4D / 5D uniformly:
            # H = axis -3, W = axis -2, C = axis -1.
            obs[vk] = np.ascontiguousarray(np.flip(v, axis=(-3, -2)))


def _to_video_batch(value: np.ndarray) -> np.ndarray:
    """Ensure video is (B=1, T=1, H, W, C) uint8."""
    arr = np.asarray(value, dtype=np.uint8)
    if arr.ndim == 3:
        return arr[np.newaxis, np.newaxis, ...]
    elif arr.ndim == 4:
        return arr[np.newaxis, ...]
    return arr


def _to_state_batch(value) -> np.ndarray:
    """Ensure state is (B=1, T=1, D) float32.

    Handles every shape pre-fanout:
      * scalar / 0-D ndarray → (1, 1, 1)  (e.g. ``state.x = 0.123``)
      * 1-D ndarray (D,)     → (1, 1, D)  (e.g. ``state.gripper = [0.02, -0.02]``)
      * 2-D ndarray (T, D)   → (1, T, D)
      * 3-D and beyond       → passthrough
    """
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        # 0-D scalar: promote to (D=1,) then to (1, 1, 1) so the model
        # sees a proper (B, T, D) shape. Without this, NVIDIA's
        # _unbatch_observation crashes with `IndexError: too many indices
        # for array: array is 0-dimensional` on every scalar state key
        # (#187 LOCAL-mode regression I caught while bisecting).
        return arr[np.newaxis, np.newaxis, np.newaxis]
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
