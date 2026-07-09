"""LeRobot Local Policy - Direct HuggingFace model inference (no server needed).

Uses LeRobot's own factory for auto-detection. Any model LeRobot supports,
this policy supports.

Architecture:
    Observation (dict)
        → ProcessorBridge.preprocess (normalize, device, crop, ...)
        → LeRobot PreTrainedPolicy.select_action / predict_action_chunk (RTC)
        → ProcessorBridge.postprocess (unnormalize, delta-action, ...)
        → Action dict
"""

import logging
import threading
import time
from collections import deque
from typing import Any

import numpy as np
import torch

from .. import Policy
from .embodiment import ZeroActionMonitor, diagnose_action_dim
from .processor import ProcessorBridge
from .resolution import resolve_policy_class_by_name, resolve_policy_class_from_hub

logger = logging.getLogger(__name__)


def _declared_feature_is_image(name: str, feature: Any = None) -> bool:
    """Return True if a declared input feature is a camera/image (VISUAL) feature.

    Prefers the authoritative ``FeatureType.VISUAL`` carried by the declared
    ``PolicyFeature`` (a real enum whose ``name`` is a string); falls back to
    the ``observation.image*`` key-name convention only when no usable type
    metadata is available. A plain ``"image" in name`` substring test silently
    misclassifies VISUAL features whose key does not contain the literal
    ``"image"`` (e.g. MolmoAct2 declares image feature keys such as
    ``base``/``wrist``), which drops their camera frames from the remapped
    observation and surfaces later as a confusing "image_keys missing from
    observation" failure inside the preprocessor.

    Args:
        name: The declared feature key (e.g. ``observation.images.top``).
        feature: The declared ``PolicyFeature`` (or any object exposing a
            ``type`` whose ``name`` is ``"VISUAL"``). ``None`` forces the
            name-based fallback.

    Returns:
        True when the feature is an image/visual input feature.
    """
    type_name = getattr(getattr(feature, "type", None), "name", None)
    if isinstance(type_name, str):
        return type_name == "VISUAL"
    return "image" in name


def _merge_obs_rename(base: dict[str, str], override: dict[str, str | None] | None) -> dict[str, str]:
    """Merge an ``obs_rename_override`` over an embodiment's ``obs_rename``.

    An override entry whose value is falsy (``None`` or ``""``) DROPS the
    matching source key from the merged map instead of adding a bogus rename.
    This is how a caller adapts a multi-camera embodiment (e.g. ``so_real``
    declares both ``front`` and ``wrist``) to a single-camera checkpoint that
    does not declare ``observation.images.wrist_image``: pass
    ``obs_rename_override={"wrist": None}`` to remove the stale rename whose
    target the model never declares (the embodiment ``validate`` would
    otherwise reject it). Non-empty values add or replace a rename as before.

    Args:
        base: The embodiment's declared ``obs_rename`` map.
        override: Caller override; falsy values drop, truthy values set.

    Returns:
        The merged rename map with dropped keys removed.
    """
    merged = dict(base)
    for src, dst in (override or {}).items():
        if dst:
            merged[src] = dst
        else:
            merged.pop(src, None)
    return merged


# Fallback control rate used ONLY when RTC runs without the runtime having
# called set_control_frequency(). Used to keep a standalone (no-runner) RTC
# call functional; the runner always plumbs the real rate. A loud one-time
# warning fires whenever this fallback is used.
_RTC_FALLBACK_FPS: float = 30.0


# Process-level cache of loaded underlying models, keyed by the load-determining
# inputs. Loading a VLA checkpoint (e.g. MolmoAct2 SO-100/101 = 1295 weight
# files) reads gigabytes from disk and uploads them to the GPU - on the order of
# 100s per load. Re-instantiating ``LerobotLocalPolicy`` for the same checkpoint
# (the common multi-episode / per-rollout ``create_policy`` pattern) would pay
# that cost every time. Caching the built nn.Module here lets repeated
# instances share one resident model.
#
# Contract: the cached object is the SAME live nn.Module shared across every
# wrapper that requests the same key. LeRobot policies hold per-episode mutable
# state (action queue, temporal-ensemble buffers) which ``Policy.reset()`` (and
# thus ``PolicyRunner`` between episodes) clears - so SEQUENTIAL reuse is safe.
# Two wrappers driving the SAME checkpoint+device CONCURRENTLY would share that
# state; opt out with ``cache_model=False`` for that (rare) case. Call
# :func:`clear_model_cache` to evict and free the held GPU/CPU memory.
_MODEL_CACHE: dict[tuple[Any, ...], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


def clear_model_cache(pretrained_name_or_path: str | None = None) -> int:
    """Evict cached lerobot_local models, freeing their held memory.

    Args:
        pretrained_name_or_path: When ``None`` (default), evict every entry.
            When set, evict only the entries loaded from that checkpoint
            (matched against the second field of each cache key) - lets a caller
            free one model before loading a different one without dropping other
            resident checkpoints.

    Returns:
        Number of cache entries evicted. Best-effort releases the CUDA caching
        allocator afterwards so freed GPU memory is returned to the driver.
    """
    with _MODEL_CACHE_LOCK:
        if pretrained_name_or_path is None:
            n = len(_MODEL_CACHE)
            _MODEL_CACHE.clear()
        else:
            doomed = [k for k in _MODEL_CACHE if len(k) > 1 and k[1] == pretrained_name_or_path]
            for k in doomed:
                del _MODEL_CACHE[k]
            n = len(doomed)
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except (RuntimeError, AssertionError):
        # Best-effort GPU memory release: the entries are already evicted, so a
        # failure here (no live CUDA context, driver hiccup) must not turn a
        # successful cache clear into an error. Swallow and report the count.
        pass
    return n


def list_cached_models() -> list[dict[str, Any]]:
    """Report the underlying models currently held in the process-level cache.

    Complements :func:`clear_model_cache` with read-only introspection so a
    caller (or an LLM harness deciding whether to evict before loading a
    different checkpoint) can see what is resident without poking at the
    private ``_MODEL_CACHE`` dict. The heavy model object itself is never
    returned - only its identifying key fields and resolved class name.

    Returns:
        One dict per cached entry, each with:
            ``namespace``: load path family (``"generic"`` or ``"molmoact2"``).
            ``pretrained_name_or_path``: the checkpoint the entry was keyed on.
            ``device``: the requested device string (or ``None`` if unset).
            ``policy_class``: class name of the cached model (best-effort).
        The list is ordered as the cache was populated.
    """
    with _MODEL_CACHE_LOCK:
        items = list(_MODEL_CACHE.items())
    out: list[dict[str, Any]] = []
    for key, value in items:
        # Keys are (namespace, pretrained_name_or_path, device, *extra). The
        # generic path stores a (policy, policy_type) tuple; molmoact2 stores
        # the bare policy. Resolve the underlying module for the class name.
        model = value[0] if isinstance(value, tuple) else value
        namespace = key[0] if len(key) > 0 else None
        path = key[1] if len(key) > 1 else None
        device = key[2] if len(key) > 2 else None
        out.append(
            {
                "namespace": namespace,
                "pretrained_name_or_path": path,
                "device": device,
                "policy_class": type(model).__name__,
            }
        )
    return out


class LerobotLocalPolicy(Policy):
    """Policy that loads and runs LeRobot models directly (no server).

    Auto-detects policy type from HF config.json → delegates to LeRobot's
    own class registry.

    Optionally loads the model's processor pipeline (preprocessor.json /
    postprocessor.json) for automatic normalization, device transfer,
    observation formatting, and action unnormalization.

    Optionally supports Real-Time Chunking (RTC) for flow-matching policies,
    blending action chunks across inference calls to compensate for latency.

    Args:
        pretrained_name_or_path: HF model ID or local path. If empty, model
            is not loaded until first inference call.
        policy_type: Explicit LeRobot policy type (e.g. "act", "diffusion").
            Auto-detected from config.json if not provided.
        device: Target device (e.g. "cuda", "cpu"). Auto-detected if None.
        actions_per_step: Number of action steps to return per inference
            call. Defaults to 1 (closed-loop). When left at the default,
            it is auto-set from the loaded model's ``config.n_action_steps``
            (the model's trained open-loop chunk size) if that exceeds 1;
            pass an explicit value > 1 to override the auto-detection.
        use_processor: Whether to load the model's processor pipeline.
        processor_overrides: Dict of overrides for processor pipeline steps.
        tokenizer_max_length: Max token length for VLA language tokenization.
        tokenizer_padding_side: Padding side for VLA tokenizer ("left" or "right").
        rtc_enabled: Enable Real-Time Chunking for flow-matching policies.
            Auto-detected from model config if None.
        rtc_execution_horizon: Number of timesteps from the prefix to use for
            guidance. Defaults to model config value or 10.
        rtc_max_guidance_weight: Maximum guidance weight for RTC correction.
            Defaults to model config value or 10.0.
        camera_key_map: Optional explicit mapping of robot/sim camera name
            (e.g. "top") to the policy's declared image feature key
            (e.g. "observation.images.top"). When omitted, cameras are
            routed by exact short-name match and then by declared order
            with a warning on mismatch.
        strict_keys: When True, raise (instead of warning + positional
            fallback) if any camera name cannot be matched to a declared
            policy image key by exact name and no ``camera_key_map`` covers
            it. Defaults to False (positional fallback).
    """

    def __init__(
        self,
        pretrained_name_or_path: str = "",
        policy_type: str | None = None,
        device: str | None = None,
        actions_per_step: int = 1,
        use_processor: bool = True,
        processor_overrides: dict | None = None,
        tokenizer_max_length: int = 48,
        tokenizer_padding_side: str = "right",
        rtc_enabled: bool | None = None,
        rtc_execution_horizon: int | None = None,
        rtc_max_guidance_weight: float | None = None,
        inference_kwargs: dict | None = None,
        embodiment: str | dict | Any | None = None,
        norm_tag: str | None = None,
        image_keys: list[str] | None = None,
        inference_action_mode: str = "continuous",
        camera_key_map: dict[str, str] | None = None,
        obs_rename_override: dict[str, str | None] | None = None,
        strict_keys: bool = False,
        cache_model: bool = True,
        revision: str | None = None,
        **kwargs,
    ):
        self.pretrained_name_or_path = pretrained_name_or_path
        # Optional Hub revision (branch, tag, or commit SHA) to pin the
        # checkpoint to a reproducible version. None loads the default branch.
        self.revision = revision
        self.policy_type = policy_type
        self.requested_device = device
        self.actions_per_step = actions_per_step
        self.use_processor = use_processor
        self.processor_overrides = processor_overrides
        # Extra keyword args forwarded verbatim to the underlying LeRobot
        # policy's select_action()/predict_action_chunk() on every inference
        # call. Required by policies that demand a runtime mode selector with
        # no usable default - e.g. MolmoAct2 needs
        # inference_kwargs={"inference_action_mode": "continuous"|"discrete"}
        # (its select_action raises ValueError otherwise). RTC kwargs are
        # handled separately and take precedence on the RTC path.
        self.inference_kwargs: dict[str, Any] = dict(inference_kwargs or {})
        # Declarative robot/sim -> model key mapping (SOLUTION.md). When set,
        # observation/action remapping is configured ONCE into LeRobot's own
        # processor pipeline at load time, and the per-step heuristic remap
        # (_to_lerobot_observation / _fixup_preprocessed_batch) is bypassed.
        self._embodiment_spec = embodiment
        self._embodiment: Any | None = None
        self.robot_state_keys: list[str] = []
        # Optional explicit strands-camera-name -> policy-image-key routing.
        # When None, cameras are matched by exact short name and then by
        # declared order (see _resolve_camera_targets).
        self.camera_key_map = dict(camera_key_map) if camera_key_map else None
        # Optional override merged OVER the embodiment's declared ``obs_rename``
        # (``{runtime_obs_key: "observation.images.*"}``). Lets a caller keep
        # custom sim camera names (e.g. ``realsense_top``) and still route them
        # onto the model's declared image features without renaming cameras. A
        # falsy value (``None`` or ``""``) DROPS that source key, so a
        # multi-camera embodiment (e.g. ``so_real`` declares ``front`` +
        # ``wrist``) can be adapted to a single-camera checkpoint that does not
        # declare the second image feature. Merged in
        # :meth:`_configure_embodiment` via :func:`_merge_obs_rename`; also
        # consulted by the class-level :meth:`preflight` so the override is
        # honoured before any model download.
        self._obs_rename_override = dict(obs_rename_override) if obs_rename_override else None
        # When True, raise instead of routing cameras positionally if their
        # names cannot be matched to the policy's declared image keys (and no
        # camera_key_map covers them). Defaults to False (positional fallback
        # with a warning), preserving zero-config ergonomics.
        self.strict_keys = strict_keys
        # Routing-degradation telemetry. The heuristic (non-declarative)
        # remap path can keep a run alive while silently producing
        # meaningless inputs - a camera routed to an arbitrary model image
        # slot positionally, or a state vector ordered by the observation's
        # own keys because none of robot_state_keys matched. Both make the
        # robot wiggle with success_rate ~ 0 while the run reports success.
        # These flags are flipped True the first time each fallback fires so
        # run_policy / eval_policy can surface a machine-checkable signal
        # (positional_fallback_used / generic_state_keys_used) instead of the
        # degradation only living in a log line nobody reads.
        self.positional_fallback_used = False
        self.generic_state_keys_used = False
        # When True (default), the loaded underlying model is cached at
        # process level and shared by later instances with the same load
        # key (see _MODEL_CACHE). Set False to force a private load (e.g.
        # concurrent rollouts of the same checkpoint that must not share
        # per-episode model state).
        self.cache_model = cache_model
        # MolmoAct2-specific knobs. MolmoAct2 SO-100/101 checkpoints are
        # transformers-native (no lerobot draccus `type`), so they take a
        # dedicated load path (see lerobot_local.molmoact2). These are inert
        # for every other policy type.
        self._molmoact2_norm_tag = norm_tag
        self._molmoact2_image_keys = image_keys
        self._molmoact2_inference_action_mode = inference_action_mode

        self._policy: Any | None = None
        self._device: torch.device | None = None
        self._input_features: dict[str, Any] = {}
        self._output_features: dict[str, Any] = {}
        self._loaded = False
        # Load telemetry, observable end-to-end (PolicyRunner surfaces these
        # as ``policy_load_time_s`` / ``policy_load_cache_hit`` in its result
        # JSON). ``load_cache_hit`` True means the heavy from_pretrained weight
        # read was skipped because the process-level _MODEL_CACHE already held
        # this checkpoint - an agent driving a multi-episode loop can read a
        # False on episode 2+ as a smell that it rebuilt the policy instead of
        # reusing policy_object=. ``load_time_s`` is the wall time the load
        # actually took (near 0 on a cache hit).
        self.load_time_s: float = 0.0
        self.load_cache_hit: bool = False
        self._processor_bridge: ProcessorBridge | None = None
        # Set True when the pipeline loaded and was active but the caller's
        # embodiment / image_keys were incompatible with the model's declared
        # features, so the bridge was discarded (see _load_processor_bridge).
        self._embodiment_config_failed = False
        self._tokenizer: Any = None
        self._tokenizer_max_length: int = tokenizer_max_length
        self._tokenizer_padding_side: str = tokenizer_padding_side

        # RTC state
        self._rtc_requested = rtc_enabled
        self._rtc_enabled = False
        self._rtc_execution_horizon = rtc_execution_horizon
        self._rtc_max_guidance_weight = rtc_max_guidance_weight
        self._rtc_prev_chunk: torch.Tensor | None = None
        # Absolute-coordinate copy of the leftover tail, populated ONLY for
        # relative-action flow policies so the next chunk can re-express it
        # against the current robot state (see _predict_with_rtc). Stays None
        # for absolute-action policies, whose frame does not move.
        self._rtc_prev_chunk_abs: torch.Tensor | None = None
        # Lazily-resolved (once) preprocessor steps + helper used to re-anchor a
        # relative-action RTC prefix to LeRobot parity. All stay None unless an
        # enabled RelativeActionsProcessorStep is present in the pipeline.
        self._rtc_relative_step: Any = None
        self._rtc_normalizer_step: Any = None
        self._rtc_reanchor_fn: Any = None
        self._rtc_rebase_resolved: bool = False
        self._rtc_action_queue: deque = deque()
        self._rtc_latency_history: deque = deque(maxlen=100)
        self._rtc_last_inference_time: float = 0.0
        self._rtc_last_log_time: float = 0.0
        # Warn at most once per policy when RTC runs without a known
        # control_frequency (set_control_frequency never called) so the
        # 30Hz fallback is loud, not silent.
        self._rtc_freq_warned: bool = False
        # Warn at most once per policy when the configured robot_state_keys
        # do not match ANY key in the observation (the generic joint_0..N
        # vs named-joint mismatch). Keeps the state-key fallback loud, not
        # silent (see _resolve_state_order).
        self._state_key_mismatch_warned: bool = False
        # Warn at most once per policy when SOME (but not all) configured
        # robot_state_keys are absent from the observation - the
        # partial-mismatch case. Previously the absent keys were silently
        # dropped from the state vector, shifting every following joint
        # value into the wrong index before a trailing zero-pad (the
        # canonical trigger is a mimic/tendon gripper whose actuator name
        # differs from the observation's finger-joint names, e.g. aloha).
        # Missing dims are now zero-filled IN PLACE so present dims keep
        # their model index, and the degradation is surfaced rather than
        # silent (mirrors _resolve_state_order's all-missing guard).
        self._state_missing_keys_warned: bool = False
        # Telemetry parallel to generic_state_keys_used: flipped True the
        # first time a resolved state key is missing from the observation
        # so run_policy / eval_policy can surface a machine-checkable signal.
        self.missing_state_keys_used = False

        # Action diagnostics: surface a model<->embodiment action-dim mismatch
        # (zero-filled actuators) and a persistent near-zero action stream
        # (robot "runs the policy" but never moves) instead of swallowing them.
        self._zero_action_monitor = ZeroActionMonitor()
        self._action_dim_warned = False

        if pretrained_name_or_path:
            self._load_model()

    @property
    def provider_name(self) -> str:
        return "lerobot_local"

    @property
    def supports_rtc(self) -> bool:
        """Whether Real-Time Chunking is active for the loaded policy.

        Surfaces the internal RTC state as the public ``ChunkedPolicy`` contract
        attribute (see ``strands_robots.policies.base.ChunkedPolicy``). ``True``
        only after a flow-matching policy with an enabled ``rtc_config`` has been
        loaded - it carries prev-chunk state across re-queries to blend chunk
        seams internally, so a consumer never has to drive RTC itself. Returns
        ``False`` before a model is loaded or for non-flow-matching policies
        (ACT, diffusion), which still emit chunks (``actions_per_step``) but do
        not blend seams.
        """
        return self._rtc_enabled

    @property
    def execution_horizon(self) -> int:
        """Actions the SIM executes from one chunk before re-querying.

        Overrides :attr:`Policy.execution_horizon` to separate the inference-time
        re-query budget from the trained chunk length (``actions_per_step``):

        * RTC active -> the RTC execution horizon (``rtc_execution_horizon``,
          default 10). The policy is re-queried mid-chunk so it blends the
          unexecuted tail of the previous chunk (``prev_chunk_left_over``) into
          the next - the whole point of Real-Time Chunking. Re-querying only
          after the full trained chunk drains keeps that tail empty and silently
          collapses RTC to open-loop replay.
        * otherwise -> ``actions_per_step`` (the trained chunk, consumed whole).

        Falls back to ``actions_per_step`` when RTC is enabled but the horizon
        was never resolved (defensive; ``_init_rtc`` always sets it).
        """
        if self._rtc_enabled and self._rtc_execution_horizon:
            return max(1, int(self._rtc_execution_horizon))
        return max(1, int(self.actions_per_step))

    def is_chunk_emitting(self) -> bool:
        """Whether this LeRobot policy returns multi-action chunks per inference.

        Extends :meth:`Policy.is_chunk_emitting` so the async-RTC pipeline
        auto-enables latency masking for every chunk-emitting LeRobot model, not
        only those whose chunk shape is visible through ``execution_horizon``:

        * ``execution_horizon > 1`` covers ACT, diffusion, pi0, pi0.5, pi0-FAST
          and SmolVLA, whose trained chunk (or RTC horizon) is more than one
          action (the base-class check).
        * :attr:`supports_rtc` covers a flow-matching model that blends chunk
          seams internally - it is chunk-emitting by construction.
        * :meth:`_requires_action_chunk` covers MolmoAct2, which MUST be driven
          via ``predict_action_chunk`` (its ``select_action`` raises under an
          enabled ``rtc_config``); its trained chunk is not always reflected in
          ``actions_per_step``, so detect it through the same path that already
          routes it to chunked inference.

        Returns:
            ``True`` when the loaded policy emits multi-action chunks; ``False``
            for single-step checkpoints or before a model is loaded.
        """
        return super().is_chunk_emitting() or self.supports_rtc or self._requires_action_chunk()

    def reset(self, seed: int | None = None) -> None:
        """Reset policy state between episodes.

        **MUST** be called whenever the environment or task episode resets.
        LeRobot policies cache internal state such as
        action queues and temporal ensemble buffers. Without resetting, stale
        actions from the previous episode leak into the next one.

        Also clears RTC state (previous chunk leftover, action queue, latency
        history) to prevent cross-episode contamination.

        Args:
            seed: Per-episode master seed (added in #187 for the
                ``Policy.reset(seed=...)`` contract). Currently
                unused - LeRobot policies don't expose RNG state via a
                seed kwarg, and reproducibility is handled by
                ``set_eval_seed`` upstream of the call. Reserved for
                future per-policy RNG plumbing.
        """
        del seed  # explicit no-op, not silently ignored
        if self._policy is not None and hasattr(self._policy, "reset"):
            self._policy.reset()
            logger.debug("Policy internal state reset")
        if self._processor_bridge is not None:
            self._processor_bridge.reset()
        # Clear RTC state
        self._rtc_prev_chunk = None
        self._rtc_prev_chunk_abs = None
        self._rtc_action_queue.clear()
        self._rtc_latency_history.clear()
        self._rtc_last_inference_time = 0.0
        self._rtc_last_log_time = 0.0
        self.rtc_observed_delay_steps = None
        # Re-arm action diagnostics for the next episode.
        self._zero_action_monitor.reset()
        self._action_dim_warned = False

    def set_robot_state_keys(self, robot_state_keys: list[str]) -> None:
        """Set robot state keys for observation→tensor mapping.

        Args:
            robot_state_keys: List of joint/motor names. If empty, auto-detects
                from model output_features (action dim) or input_features (state dim).
                Auto-detected keys are generic (joint_0, joint_1, ...).

        Raises:
            ValueError: If keys are empty and no model features available for
                auto-detection.
        """
        if robot_state_keys:
            self.robot_state_keys = robot_state_keys
            logger.info(
                "LeRobot local state keys set: %d keys = %s%s",
                len(self.robot_state_keys),
                self.robot_state_keys[:5],
                "..." if len(self.robot_state_keys) > 5 else "",
            )
            return

        # Auto-detect from model's action output dimension
        if self._loaded and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-detected %d state keys from output_features.action.shape=%s. "
                    "For meaningful names, pass the robot's actual joint names.",
                    action_dim,
                    action_feat.shape,
                )
                return

        # Fallback: try input state dimension
        if self._loaded and self._input_features:
            state_feat = self._input_features.get("observation.state")
            if state_feat and hasattr(state_feat, "shape") and state_feat.shape:
                state_dim = state_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(state_dim)]
                logger.info(
                    "Auto-detected %d state keys from input_features.observation.state.shape=%s.",
                    state_dim,
                    state_feat.shape,
                )
                return

        raise ValueError(
            "robot_state_keys is empty and no model features available for auto-detection. "
            "Call set_robot_state_keys() with the robot's actual joint/motor names."
        )

    # Tokenizer resolution (VLA language token injection)

    def _resolve_tokenizer(self) -> Any | None:
        """Resolve and cache the tokenizer for VLA language token injection.

        Resolution order:
            1. Explicit ``tokenizer_name`` on policy config (e.g. xvla)
            2. ``vlm_model_name`` on policy config (maps to the VLM's tokenizer)
            3. Policy's own ``.processor.tokenizer`` (e.g. Paligemma-based)

        Returns:
            The tokenizer instance, or None if not available.
        """
        if self._tokenizer is not None:
            return self._tokenizer

        if not self._loaded or not self._policy:
            return None

        config = getattr(self._policy, "config", None)
        if config is None:
            return None

        # Override defaults with model config if present
        self._tokenizer_max_length = getattr(config, "tokenizer_max_length", self._tokenizer_max_length)
        self._tokenizer_padding_side = getattr(config, "tokenizer_padding_side", self._tokenizer_padding_side)

        # 1. tokenizer_name (explicit config field)
        tokenizer_id = getattr(config, "tokenizer_name", None)

        # 2. vlm_model_name (VLA models)
        if not tokenizer_id:
            tokenizer_id = getattr(config, "vlm_model_name", None)

        if tokenizer_id:
            try:
                from transformers import AutoTokenizer

                self._tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
                self._tokenizer.padding_side = self._tokenizer_padding_side  # type: ignore[attr-defined]
                logger.info("Auto-resolved tokenizer from '%s' (%s)", tokenizer_id, type(self._tokenizer).__name__)
                return self._tokenizer
            except (ImportError, OSError, ValueError) as exc:
                logger.debug("Tokenizer '%s' unavailable (%s), trying next strategy...", tokenizer_id, exc)

        # 3. policy.processor.tokenizer (built-in)
        processor = getattr(self._policy, "processor", None)
        if processor and hasattr(processor, "tokenizer"):
            self._tokenizer = processor.tokenizer
            self._tokenizer.padding_side = self._tokenizer_padding_side  # type: ignore[attr-defined]
            logger.info("Using policy's built-in processor tokenizer (%s)", type(self._tokenizer).__name__)
            return self._tokenizer

        return None

    def _tokenize_instruction(self, instruction: str) -> tuple[torch.Tensor, torch.Tensor | None] | None:
        """Tokenize an instruction into (input_ids, attention_mask) tensors.

        Args:
            instruction: Natural language instruction string.

        Returns:
            Tuple of (input_ids, attention_mask) tensors, or None if no tokenizer.
        """
        tokenizer = self._resolve_tokenizer()
        if tokenizer is None or not instruction:
            return None

        encoded = tokenizer(
            instruction,
            return_tensors="pt",
            padding="max_length",
            max_length=self._tokenizer_max_length,
            truncation=True,
        )
        tokens = encoded["input_ids"].to(self._device)
        mask = encoded.get("attention_mask")
        if mask is not None:
            mask = mask.bool().to(self._device)
        return tokens, mask

    def _needs_language_tokens(self) -> bool:
        """Check whether this policy requires observation.language.tokens.

        Returns True if the model config indicates VLA language input is needed
        (tokenizer_name, vlm_model_name, or language-related input features).
        """
        config = getattr(self._policy, "config", None)
        if config is None:
            return False

        if getattr(config, "tokenizer_name", None):
            return True
        if getattr(config, "vlm_model_name", None):
            return True
        if any("language" in key for key in self._input_features):
            return True

        return False

    # Model loading

    def _model_cache_key(self, namespace: str, *extra: Any) -> tuple[Any, ...] | None:
        """Build the process-cache key for the underlying model load.

        Returns ``None`` when caching is disabled or there is no checkpoint
        path to key on (a from-scratch / parameterless policy), which makes the
        cache a transparent no-op for those cases.
        """
        if not self.cache_model or not self.pretrained_name_or_path:
            return None
        return (namespace, self.pretrained_name_or_path, self.requested_device, *extra)

    def _cache_get(self, key: tuple[Any, ...] | None) -> Any:
        if key is None:
            return None
        with _MODEL_CACHE_LOCK:
            return _MODEL_CACHE.get(key)

    def _cache_put(self, key: tuple[Any, ...] | None, value: Any) -> None:
        if key is None:
            return
        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[key] = value

    def _load_model(self) -> None:
        """Load the LeRobot model from pretrained path.

        Raises:
            ImportError: If required dependencies are missing.
            ValueError: If model path is invalid or config cannot be parsed.
            RuntimeError: If model loading fails.
        """
        # MolmoAct2 SO-100/101 checkpoints are transformers-native (config.json
        # has model_type=molmoact2 and NO lerobot draccus `type`). The standard
        # resolve→from_pretrained path raises ParsingError on them, so route to
        # the dedicated wrapper that builds MolmoAct2Config(checkpoint_path=...)
        # and the molmoact2 pre/post processor pipeline programmatically.
        from . import molmoact2 as _molmoact2

        if _molmoact2.is_molmoact2(self.pretrained_name_or_path, self.policy_type):
            if self.revision:
                raise ValueError(
                    "revision pinning is not supported for transformers-native "
                    "MolmoAct2 checkpoints (loaded via checkpoint_path, not "
                    "PreTrainedPolicy.from_pretrained). Pin the version by passing "
                    "a commit-SHA-qualified pretrained_name_or_path, or download the "
                    "revision locally and point at the directory."
                )
            self._load_molmoact2_model()
            return

        # XVLA compat: Florence2LanguageConfig.forced_bos_token_id missing
        # in transformers 5.x. Florence2 was originally built with an older
        # transformers that had this attribute. Without this patch, XVLA
        # models fail to load with AttributeError.
        try:
            from transformers.models.florence2.configuration_florence2 import (  # type: ignore[attr-defined,import-not-found]  # noqa: E501
                Florence2LanguageConfig,
            )

            if not hasattr(Florence2LanguageConfig, "forced_bos_token_id"):
                Florence2LanguageConfig.forced_bos_token_id = None
                logger.debug("Patched Florence2LanguageConfig.forced_bos_token_id for XVLA compat")
        except ImportError:
            pass

        logger.info("Loading %s...", self.pretrained_name_or_path)
        start = time.time()

        # Reuse a process-cached model when one was already built for this
        # (path, policy_type, device) - skips the expensive from_pretrained
        # weight read + GPU upload on repeat instantiation. The resolved
        # policy_type is cached alongside the module so a hit needs no hub call.
        cache_key = self._model_cache_key("generic", self.policy_type, self.revision)
        cached = self._cache_get(cache_key)
        self.load_cache_hit = cached is not None
        if cached is not None:
            self._policy, self.policy_type = cached
            logger.info(
                "Reusing cached %s for %s (skipped from_pretrained)",
                type(self._policy).__name__,
                self.pretrained_name_or_path,
            )
        else:
            # Resolve the correct policy class
            if self.policy_type:
                PolicyClass = resolve_policy_class_by_name(self.policy_type)
            else:
                PolicyClass, self.policy_type = resolve_policy_class_from_hub(
                    self.pretrained_name_or_path, revision=self.revision
                )

            # Pass revision only when set so the call matches lerobot's
            # default (revision=None) and stays compatible with policy
            # classes whose from_pretrained does not accept the kwarg.
            from_pretrained_kwargs = {"revision": self.revision} if self.revision else {}
            self._policy = PolicyClass.from_pretrained(self.pretrained_name_or_path, **from_pretrained_kwargs)
            assert self._policy is not None

            self._policy.eval()
            self._cache_put(cache_key, (self._policy, self.policy_type))

        # Resolve device: prefer user-requested, then config.device, fallback to first param
        if self.requested_device:
            self._device = torch.device(self.requested_device)
        elif hasattr(self._policy, "config") and hasattr(self._policy.config, "device"):
            self._device = torch.device(self._policy.config.device)
        else:
            self._device = next(self._policy.parameters()).device

        # Move the model onto the resolved device. LeRobot's from_pretrained
        # places weights on config.device (e.g. 'mps'/'cuda' baked into the
        # checkpoint config), which may differ from the user's requested
        # device. get_actions() moves every input tensor onto self._device,
        # so without this the model weights and inputs land on different
        # devices and the first conv2d raises "input and weight must be on
        # the same device". Keep them in lockstep.
        try:
            current = next(self._policy.parameters()).device
            if current != self._device:
                self._policy.to(self._device)
        except StopIteration:
            pass  # parameterless policy - nothing to move

        if hasattr(self._policy, "config"):
            config = self._policy.config
            if hasattr(config, "input_features"):
                self._input_features = config.input_features
            if hasattr(config, "output_features"):
                self._output_features = config.output_features

        elapsed = time.time() - start
        self.load_time_s = elapsed
        logger.info(
            "Loaded %s (type='%s') in %.1fs on %s",
            type(self._policy).__name__,
            self.policy_type,
            elapsed,
            self._device,
        )
        self._loaded = True
        self._auto_detect_actions_per_step()

        # Auto-detect robot_state_keys from model config if not set
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat and hasattr(action_feat, "shape") and action_feat.shape:
                action_dim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(action_dim)]
                logger.info(
                    "Auto-generated %d generic state keys (joint_0..joint_%d). "
                    "Set explicit keys with set_robot_state_keys() for meaningful joint names.",
                    action_dim,
                    action_dim - 1,
                )

        # Load processor pipeline (preprocessor + postprocessor), configure the
        # declarative embodiment map, and emit load-time diagnostics.
        self._load_processor_bridge()

        # Initialize RTC if supported by this policy
        self._init_rtc()

    def _load_processor_bridge(self) -> None:
        """Load the processor pipeline, configure the embodiment, and emit load-time diagnostics.

        Two distinct failure modes were previously conflated into one silent,
        debug-level discard of the whole pipeline:

        * ``ProcessorBridge.from_pretrained`` raises (no processor configs are
          shipped, or an optional import is missing): the checkpoint legitimately
          has no pipeline, so fall back to the raw obs/action flow. This is a
          common, benign shape - logged at debug, and the missing-postprocessor
          warning below still fires so a raw-action checkpoint is not mistaken for
          a frozen policy.
        * ``_configure_embodiment`` raises ``ValueError``: the pipeline loaded and
          was ACTIVE, but the caller's embodiment / ``image_keys`` are incompatible
          with the model's declared features. Discarding it silently degrades a
          WORKING normalization pipeline to the raw flow AND misdirects the
          downstream "no policy_postprocessor.json" diagnostic (the checkpoint
          shipped one - it was discarded here). Surface the real cause as a
          warning, or raise when ``processor_overrides`` were given (mirroring the
          from_pretrained path).

        A malformed embodiment *spec* (bad name / dict) raises ``RuntimeError``
        from ``load_embodiment`` and is intentionally NOT caught here - that is a
        caller error that should abort the load loudly.
        """
        self._embodiment_config_failed = False
        if not (self.use_processor and self.pretrained_name_or_path):
            return

        try:
            self._processor_bridge = ProcessorBridge.from_pretrained(
                self.pretrained_name_or_path,
                device=str(self._device) if self._device else None,
                overrides=self.processor_overrides or {},
                policy_type=self.policy_type,
                policy_config=getattr(self._policy, "config", None),
            )
        except (FileNotFoundError, ValueError, ImportError) as exc:
            # Processor bridge is optional - models work without it via the raw
            # obs/action flow. Fail-fast only if the caller requested overrides.
            if self.processor_overrides:
                raise RuntimeError(
                    f"Processor bridge failed to load but processor_overrides were specified: {exc}"
                ) from exc
            logger.debug("Processor bridge not loaded: %s", exc)
            self._processor_bridge = None
        else:
            if self._processor_bridge.is_active:
                logger.info("Processor bridge loaded: %s", self._processor_bridge)
                # SOLUTION.md: configure the declarative embodiment map into the
                # pipeline ONCE (rename_map + pack-state step), validated fail-fast
                # against the model's declared features. After this, the hot path
                # feeds RAW obs straight to preprocess().
                try:
                    self._configure_embodiment()
                except ValueError as exc:
                    # The pipeline loaded and was active, but the embodiment /
                    # image_keys do not match the model's declared features. Do NOT
                    # swallow this at debug: discarding an otherwise-working
                    # normalization pipeline is a silent behaviour change, and the
                    # missing-postprocessor warning below would misattribute it to a
                    # checkpoint lacking a policy_postprocessor.json.
                    if self.processor_overrides:
                        raise RuntimeError(
                            f"Embodiment configuration failed but processor_overrides were specified: {exc}"
                        ) from exc
                    logger.warning(
                        "lerobot_local: %s loaded an ACTIVE processor pipeline but its "
                        "embodiment could not be configured (%s). The pipeline (including "
                        "normalization) was discarded and the policy falls back to the raw "
                        "obs/action flow -- align the embodiment / image_keys with the "
                        "model's declared input/output features.",
                        self.pretrained_name_or_path or "<model>",
                        exc,
                    )
                    self._processor_bridge = None
                    self._embodiment_config_failed = True
            else:
                self._processor_bridge = None
                logger.debug("No processor configs found, using raw obs/action flow")

        # Action unnormalization only happens when a postprocessor pipeline is
        # present (get_actions: ``if self._processor_bridge.has_postprocessor``).
        # A checkpoint shipped without a ``policy_postprocessor.json`` emits raw
        # normalized actions (~[-1, 1] or z-scored) straight to the robot. Fed
        # to a radian-joint sim those are micro-motions and the arm barely
        # moves. Warn once at load so this isn't debugged as a frozen policy.
        # Skipped when the pipeline was discarded by an embodiment-config failure
        # above (already warned with the accurate cause), so we do not emit the
        # misleading "no policy_postprocessor.json" message for that case.
        if self.use_processor and not self._embodiment_config_failed:
            bridge = self._processor_bridge
            if bridge is None or not bridge.has_postprocessor:
                logger.warning(
                    "lerobot_local: %s loaded WITHOUT an action postprocessor "
                    "(no policy_postprocessor.json). Actions are emitted in the "
                    "model's RAW/normalized space and are NOT unnormalized to robot "
                    "units -- if the arm barely moves, this is why. Provide the "
                    "checkpoint's postprocessor, or drive it through a provider with "
                    "explicit unit handling (e.g. the 'transformers' provider's "
                    "state_units/action_units/stats knobs).",
                    self.pretrained_name_or_path or "<model>",
                )
            else:
                # Subtler failure than a missing postprocessor: the pipeline IS
                # present and active, but its normalizer stats do not cover the
                # declared action/state keys, so LeRobot's normalizer silently
                # passes those tensors through (normalize_processor.py returns the
                # tensor unchanged when the key is absent from its stats).
                # Pretraining base checkpoints (e.g. lerobot/smolvla_base) ship
                # stats keyed by the training dataset ('so100.buffer.action') with
                # no 'observation.state' stats and no bare 'action' key -- so state
                # reaches the model raw and actions reach the robot un-unnormalized
                # while has_postprocessor stays True. Detect and warn, matching the
                # no-silent-passthrough intent of the missing-postprocessor case.
                inert = bridge.inert_normalization_features()
                if inert:
                    logger.warning(
                        "lerobot_local: %s has an ACTIVE normalization pipeline "
                        "but its stats do not cover %s -- those features are passed "
                        "through UN-normalized (observation.state reaches the model "
                        "raw; predicted actions reach the robot without "
                        "unnormalization). Pretraining base checkpoints often ship "
                        "dataset-prefixed stats (e.g. 'so100.buffer.action') instead "
                        "of the canonical 'action'/'observation.state' keys. "
                        "Fine-tune the checkpoint (which writes proper stats) or pass "
                        "processor_overrides={'normalizer_processor': {'stats': "
                        "<dataset stats>}} -- if the arm reaches an out-of-distribution "
                        "pose or ignores proprioception, this is why.",
                        self.pretrained_name_or_path or "<model>",
                        inert,
                    )

    def _auto_detect_actions_per_step(self) -> None:
        """Select the inference regime the loaded checkpoint was trained for.

        Two mutually exclusive regimes exist, and the wrong one degrades motion:

        Temporal ensembling (``config.temporal_ensemble_coeff is not None``):
            LeRobot applies the ensembler inside ``select_action()`` on a fresh
            chunk every step (a receding-horizon, per-step operation - see
            ``ACTPolicy.select_action``). ``predict_action_chunk()`` returns the
            RAW chunk and BYPASSES the ensembler entirely. So an ensembling
            checkpoint MUST be driven one step at a time via ``select_action()``;
            replaying its chunk open-loop silently discards the smoothing the
            checkpoint was trained to produce (jerky motion). We therefore keep
            ``actions_per_step=1`` for these checkpoints, and if a caller pinned
            ``actions_per_step > 1`` we override it with a loud warning rather
            than quietly throwing the ensembling away.

        Open-loop chunk replay (``temporal_ensemble_coeff is None``):
            Many policies declare ``config.n_action_steps`` - the number of
            actions emitted per inference that the model was trained to replay
            open-loop before requerying observation (e.g. MolmoAct2 SO-100/101 =
            30, ACT = 100, Diffusion = 32). The default ``actions_per_step=1`` is
            a closed-loop convention that re-queries every step; for a chunk-
            trained model that is out of distribution and the shift compounds
            every chunk. When left at the default ``1`` we adopt
            ``n_action_steps`` so the chunk is consumed as trained. An explicit
            ``actions_per_step > 1`` from the caller is respected here.

        Logged at INFO so the active regime is always visible.
        """
        config = getattr(self._policy, "config", None)
        ensemble_coeff = getattr(config, "temporal_ensemble_coeff", None)
        if ensemble_coeff is not None:
            # Ensembling regime: must run per-step through select_action().
            if self.actions_per_step != 1:
                logger.warning(
                    "lerobot_local: %s checkpoint enables temporal ensembling "
                    "(temporal_ensemble_coeff=%s) but actions_per_step=%d was "
                    "set explicitly. Open-loop chunk replay and temporal "
                    "ensembling are mutually exclusive - chunk replay bypasses "
                    "the ensembler and discards the motion smoothing the "
                    "checkpoint was trained for. Overriding to actions_per_step="
                    "1 so ensembling is honored via select_action().",
                    type(self._policy).__name__,
                    ensemble_coeff,
                    self.actions_per_step,
                )
                self.actions_per_step = 1
            logger.info(
                "lerobot_local: %s temporal ensembling ON "
                "(temporal_ensemble_coeff=%s) - driving per-step via "
                "select_action().",
                type(self._policy).__name__,
                ensemble_coeff,
            )
            return
        if self.actions_per_step != 1:
            return  # caller pinned an explicit horizon - never override it
        n_action_steps = getattr(config, "n_action_steps", None)
        if isinstance(n_action_steps, int) and n_action_steps > 1:
            self.actions_per_step = n_action_steps
            logger.info(
                "lerobot_local: open-loop chunk replay - auto-set "
                "actions_per_step=%d from %s.config.n_action_steps (model's "
                "trained open-loop chunk size). Pass actions_per_step "
                "explicitly to override.",
                n_action_steps,
                type(self._policy).__name__,
            )

    def _load_molmoact2_model(self) -> None:
        """Load a transformers-native MolmoAct2 checkpoint via the lerobot wrapper.

        Unlike the generic path, MolmoAct2 needs ``MolmoAct2Config(checkpoint_path=...)``
        built explicitly and its pre/post processors created programmatically
        (the repo ships no ``policy_preprocessor.json``). The resulting
        ``ProcessorBridge`` is wrapped around those pipelines so the normal
        ``get_actions`` flow (preprocess → select_action → postprocess) works
        unchanged. The embodiment map (e.g. ``so_real``) still configures camera
        renames + state packing on the preprocessor via ``_configure_embodiment``.
        """
        from . import molmoact2 as _molmoact2
        from .processor import ProcessorBridge

        self.policy_type = _molmoact2.MOLMOACT2_TYPE

        # State/action dims come from the embodiment when known (SO arms = 6),
        # else default to 6 (the SO-100/101 convention this checkpoint targets).
        state_dim = action_dim = 6
        if self.robot_state_keys:
            state_dim = action_dim = len(self.robot_state_keys)

        logger.info("Loading MolmoAct2 (transformers-native) from %s...", self.pretrained_name_or_path)
        start = time.time()

        # Reuse a process-cached MolmoAct2 model when one was already built for
        # this checkpoint+device+load-knobs. The model weights (1295 files for
        # the SO-100/101 checkpoint) are the expensive part; the config and
        # pre/post processors are rebuilt cheaply so the returned tuple is
        # self-consistent. norm_tag / inference_action_mode / image_keys /
        # embodiment / dims are part of the key because they shape the processors
        # (and thus must match the cached weights' intended pipeline).
        model_cache_key = self._model_cache_key(
            "molmoact2",
            self._molmoact2_norm_tag,
            self._molmoact2_inference_action_mode,
            tuple(self._molmoact2_image_keys or ()),
            repr(self._embodiment_spec),
            state_dim,
            action_dim,
        )
        cached_model = self._cache_get(model_cache_key)
        self.load_cache_hit = cached_model is not None
        policy, preprocessor, postprocessor, cfg = _molmoact2.build_policy(
            self.pretrained_name_or_path,
            device=self.requested_device,
            norm_tag=self._molmoact2_norm_tag,
            inference_action_mode=self._molmoact2_inference_action_mode,
            image_keys=self._molmoact2_image_keys,
            embodiment_spec=self._embodiment_spec,
            state_dim=state_dim,
            action_dim=action_dim,
            prebuilt_policy=cached_model,
        )
        if cached_model is None:
            self._cache_put(model_cache_key, policy)

        self._policy = policy
        self._device = next(policy.parameters()).device
        self._input_features = dict(getattr(cfg, "input_features", {}) or {})
        self._output_features = dict(getattr(cfg, "output_features", {}) or {})

        # MolmoAct2.select_action requires inference_action_mode every call.
        self.inference_kwargs.setdefault("inference_action_mode", self._molmoact2_inference_action_mode)

        elapsed = time.time() - start
        self.load_time_s = elapsed
        logger.info(
            "Loaded MolmoAct2Policy (type='molmoact2') in %.1fs on %s",
            elapsed,
            self._device,
        )
        self._loaded = True
        self._auto_detect_actions_per_step()

        # Auto-detect generic state keys from action dim if not set.
        if not self.robot_state_keys and self._output_features:
            action_feat = self._output_features.get("action")
            if action_feat is not None and getattr(action_feat, "shape", None):
                adim = action_feat.shape[0]
                self.robot_state_keys = [f"joint_{i}" for i in range(adim)]

        # Wrap the programmatic processors in a ProcessorBridge so the standard
        # preprocess/postprocess flow applies, then configure the embodiment
        # (camera rename + state packing) onto the preprocessor pipeline.
        if self.use_processor:
            self._processor_bridge = ProcessorBridge(
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                device=str(self._device) if self._device else None,
            )
            if self._processor_bridge.is_active:
                logger.info("MolmoAct2 processor bridge ready: %s", self._processor_bridge)
                self._configure_embodiment()
            else:
                self._processor_bridge = None

        # RTC never applies to MolmoAct2 (no rtc_config); init is a safe no-op.
        self._init_rtc()

    # Embodiment configuration (declarative obs/action mapping)

    @classmethod
    def preflight(cls, observation_keys: set[str], **policy_config: Any) -> None:
        """Validate camera routing against the embodiment BEFORE any download.

        The crash this prevents: a VLA checkpoint (e.g. MolmoAct2 SO-100/101)
        declares image input features that the embodiment's ``obs_rename`` feeds
        from named runtime keys (e.g. ``front`` -> ``observation.images.image``).
        If the sim's attached camera names do not match any source key for a
        model image feature, the mismatch only surfaces deep in the preprocessor
        after the multi-minute weight download, as a confusing "image_keys
        missing from observation" failure. This hook catches it up front.

        Resolution: the model needs every declared image feature populated, so
        for each image rename TARGET (``observation.images.*``) at least one of
        its source keys must be present in ``observation_keys``. A caller-
        supplied ``obs_rename_override`` is merged over the embodiment's
        ``obs_rename`` first, so an explicit override that maps a present camera
        onto the feature satisfies the check.

        No-op when no ``embodiment`` is configured (the policy then uses the
        legacy heuristic camera routing, which this hook cannot reason about),
        or when the embodiment name/spec cannot be resolved (``create_policy``
        surfaces that error authoritatively).

        Args:
            observation_keys: Runtime observation keys (joint + camera names).
            **policy_config: Provider kwargs (``embodiment``,
                ``obs_rename_override``, ...).

        Raises:
            ValueError: When a model image feature has no satisfiable source
                camera key in ``observation_keys``.
        """
        spec = policy_config.get("embodiment")
        if spec is None:
            return
        from .embodiment import load_embodiment

        try:
            embodiment = load_embodiment(spec)
        except Exception:  # noqa: BLE001 - unknown/odd spec; create_policy reports it
            return

        obs_rename = _merge_obs_rename(embodiment.obs_rename, policy_config.get("obs_rename_override"))

        # Group source keys by the image feature TARGET they feed. A target is
        # satisfied when ANY of its sources is present in the observation, so an
        # override that maps a present camera onto the feature counts even if
        # the embodiment's default source key is absent.
        targets: dict[str, list[str]] = {}
        for src, dst in obs_rename.items():
            if "image" in dst:
                targets.setdefault(dst, []).append(src)
        if not targets:
            return

        obs = set(observation_keys)
        unsatisfied = {dst: srcs for dst, srcs in targets.items() if not any(s in obs for s in srcs)}
        if not unsatisfied:
            return

        expected = sorted({s for srcs in unsatisfied.values() for s in srcs})
        missing_features = sorted(unsatisfied)
        raise ValueError(
            f"Embodiment {embodiment.name!r} cannot route cameras to the model's image "
            f"feature(s) {missing_features}: none of the expected source key(s) {expected} "
            f"are in the runtime observation, which provides {sorted(obs)}. Either:\n"
            f"  (a) rename your sim cameras to one of {expected} "
            f"(e.g. sim.add_camera(name={expected[0]!r}, ...)), or\n"
            f"  (b) pass policy_config={{'obs_rename_override': "
            f"{{'<your_camera_name>': '{missing_features[0]}'}}}} to map an existing "
            f"camera onto the model's image feature without renaming it."
        )

    def _synthesized_camera_renames(self) -> dict[str, str]:
        """Derive camera ``obs_rename`` entries for a state-only embodiment.

        When a caller declares only joint names via :meth:`set_robot_state_keys`
        (no explicit ``embodiment``), :meth:`_configure_embodiment` synthesizes a
        state-only embodiment so the declarative pipeline composes
        ``observation.state``. That synthesized map previously had an empty
        ``obs_rename``, so the pipeline never renamed the robot's bare camera key
        (e.g. ``front``) onto the model's declared image feature
        (``observation.images.front``); the model then raised a ``KeyError`` for
        its missing image input, and :meth:`_embodiment_image_source_keys`
        returned nothing so the bare frame was never canonicalized either. This
        routes each declared image feature from a runtime source key so a
        single-camera (or multi-camera) checkpoint works on the declarative path
        without an explicit embodiment.

        Routing precedence per declared image feature:
          1. An explicit ``camera_key_map`` entry (runtime cam -> feature) wins.
          2. Otherwise the feature is fed from its short name
             (``observation.images.front`` <- ``front``); a bare-named feature
             (e.g. MolmoAct2 ``base``) maps from itself (a harmless no-op rename).

        Returns:
            A ``{runtime_source_key: model_image_feature}`` map (possibly empty
            when the model declares no image features).
        """
        declared = {f for f, feat in self._input_features.items() if _declared_feature_is_image(f, feat)}
        renames: dict[str, str] = {}
        if self.camera_key_map:
            for cam, feat in self.camera_key_map.items():
                if feat in declared:
                    renames[cam] = feat
        mapped = set(renames.values())
        for feat in declared:
            if feat in mapped:
                continue
            short = feat.rsplit(".", 1)[-1] if "." in feat else feat
            renames.setdefault(short, feat)
        return renames

    def _configure_embodiment(self) -> None:
        """Build, validate, and inject the declarative embodiment map (SOLUTION.md).

        Called once at load time after the processor bridge is ready. It:

        1. Resolves ``self._embodiment_spec`` (name / dict / EmbodimentMap), or
           synthesises a trivial map from ``robot_state_keys`` for back-compat.
        2. Validates the map against the model's declared input/output features
           (fail-fast on dim or key mismatch).
        3. Injects ``rename_map`` + a ``strands_pack_state`` step into the
           preprocessor pipeline via :meth:`ProcessorBridge.apply_embodiment`.

        If no embodiment is declared AND no ``robot_state_keys`` are set, this is
        a no-op and the policy uses the legacy heuristic remap path.
        """
        from .embodiment import EmbodimentMap, load_embodiment

        spec = self._embodiment_spec
        if spec is None:
            # Back-compat: synthesise a trivial embodiment from robot_state_keys
            # so existing callers that only set joint names still get the clean
            # pipeline path (state composition + action naming), without any
            # camera renames (none are known in this case).
            if self.robot_state_keys and not any(k.startswith("joint_") for k in self.robot_state_keys):
                spec = EmbodimentMap(
                    name="<from robot_state_keys>",
                    obs_rename=self._synthesized_camera_renames(),
                    state_keys=list(self.robot_state_keys),
                    action_keys=list(self.robot_state_keys),
                    dim_policy="pad",
                )
            else:
                # No usable declarative info - keep legacy heuristic path.
                self._embodiment = None
                return

        try:
            embodiment = load_embodiment(spec)
        except ValueError as exc:
            raise RuntimeError(f"Failed to load embodiment {spec!r}: {exc}") from exc

        # Merge any caller-supplied obs_rename override OVER the embodiment's
        # declared renames so custom sim camera names route onto the model's
        # image features without renaming cameras. Override entries win.
        if self._obs_rename_override:
            from dataclasses import replace

            merged = _merge_obs_rename(embodiment.obs_rename, self._obs_rename_override)
            embodiment = replace(embodiment, obs_rename=merged)

        # Fail-fast validation against the model's declared features.
        embodiment.validate(self._input_features, self._output_features)

        # Inject into the pipeline (rename_map + pack-state step).
        assert self._processor_bridge is not None
        self._processor_bridge.apply_embodiment(embodiment, input_features=self._input_features)

        self._embodiment = embodiment
        # Action-side mapping: prefer the embodiment's declared action_keys so
        # _tensor_to_action_dicts indexes by real actuator names, not generic
        # joint_0..N. Keep robot_state_keys as a fallback.
        if embodiment.action_keys:
            self.robot_state_keys = list(embodiment.action_keys)
        logger.info("Embodiment '%s' configured for %s", embodiment.name, type(self._policy).__name__)

    # Real-Time Chunking (RTC) support

    def _init_rtc(self) -> None:
        """Initialize RTC if the loaded policy supports it.

        RTC is supported by flow-matching policies that implement
        ``predict_action_chunk(**kwargs)``. It requires the policy to have
        an ``rtc_config`` on its config.

        Auto-detection: if ``rtc_enabled=None`` (default), RTC is enabled
        when the model's config has ``rtc_config.enabled=True``.
        """
        if not self._loaded or self._policy is None:
            return

        # Check if policy supports predict_action_chunk (required for RTC)
        has_predict_chunk = hasattr(self._policy, "predict_action_chunk")
        if not has_predict_chunk:
            if self._rtc_requested is True:
                logger.warning(
                    "RTC requested but policy '%s' does not implement predict_action_chunk.",
                    type(self._policy).__name__,
                )
            self._rtc_enabled = False
            return

        # Auto-detect from model config.
        # RTC requires rtc_config on the model - not just predict_action_chunk().
        # In LeRobot 0.5+, predict_action_chunk() is a base class method that ALL
        # policies inherit (ACT, Diffusion, etc.), but only flow-matching policies
        # (Pi0, SmolVLA) have an rtc_config that parameterizes the denoiser for
        # cross-chunk temporal blending.  Without rtc_config, calling
        # predict_action_chunk() with RTC kwargs would either be ignored or crash.
        config = getattr(self._policy, "config", None)
        rtc_config = getattr(config, "rtc_config", None) if config else None

        if self._rtc_requested is None:
            # Auto-detect: use model's rtc_config.enabled
            self._rtc_enabled = rtc_config is not None and getattr(rtc_config, "enabled", False)
        elif self._rtc_requested is True:
            if rtc_config is None:
                # User explicitly asked for RTC, but this policy has no rtc_config.
                # This means it's not a flow-matching policy - warn and disable.
                logger.warning(
                    "RTC requested but policy '%s' has no rtc_config. "
                    "RTC is only supported by flow-matching policies (Pi0, SmolVLA). "
                    "Falling back to select_action().",
                    type(self._policy).__name__,
                )
                self._rtc_enabled = False
            else:
                self._rtc_enabled = True
        else:
            self._rtc_enabled = False

        if not self._rtc_enabled:
            logger.debug("RTC disabled for policy '%s'", type(self._policy).__name__)
            return

        # Read RTC parameters from config, with user overrides
        if rtc_config is not None:
            if self._rtc_execution_horizon is None:
                self._rtc_execution_horizon = getattr(rtc_config, "execution_horizon", 10)
            if self._rtc_max_guidance_weight is None:
                self._rtc_max_guidance_weight = getattr(rtc_config, "max_guidance_weight", 10.0)
        else:
            if self._rtc_execution_horizon is None:
                self._rtc_execution_horizon = 10
            if self._rtc_max_guidance_weight is None:
                self._rtc_max_guidance_weight = 10.0

        logger.info(
            "RTC enabled for '%s': execution_horizon=%d, max_guidance_weight=%.1f",
            type(self._policy).__name__,
            self._rtc_execution_horizon,
            self._rtc_max_guidance_weight,
        )

    def _estimate_inference_delay(self, fps: float) -> int:
        """Estimate the number of action steps consumed during inference.

        Uses the p95 latency from recent inference calls to estimate how many
        action steps the robot executed while waiting for the new chunk. The
        result is ``p95_latency * fps``, so ``fps`` MUST be the real control
        rate of the executing loop - a wrong rate scales the delay wrong and
        corrupts the RTC chunk-seam blend. The caller resolves ``fps`` from
        :attr:`Policy.control_frequency` (set by the runtime).

        Args:
            fps: Robot control frequency in Hz (the loop's real control rate).

        Returns:
            Estimated delay in action steps (minimum 0).
        """
        if not self._rtc_latency_history:
            return 0

        # Use p95 of recent latencies for robust delay estimation
        latencies = list(self._rtc_latency_history)
        latencies.sort()
        p95_idx = int(len(latencies) * 0.95)
        p95_latency = latencies[min(p95_idx, len(latencies) - 1)]

        delay = int(p95_latency * fps)
        return max(0, delay)

    def _resolve_rtc_rebase_steps(self) -> None:
        """Resolve the preprocessor steps needed to re-anchor a relative-action RTC prefix.

        Relative-action flow policies (pi0 / pi0.5 / pi0-FAST with an enabled
        ``RelativeActionsProcessorStep``) train on actions expressed as offsets
        from the current robot state. The unexecuted tail of the previous chunk
        (``prev_chunk_left_over``) is therefore only valid in the coordinate frame
        of the observation that produced it; feeding it back verbatim after the
        state has moved injects a STALE-frame prefix and corrupts the chunk-seam
        blend. LeRobot solves this by keeping the leftover in ABSOLUTE coordinates
        and re-expressing it against the live state every call via
        :func:`reanchor_relative_rtc_prefix` (reading the cached state from the
        paired ``RelativeActionsProcessorStep.get_cached_state``).

        This resolves, once, the enabled ``RelativeActionsProcessorStep``, the
        paired ``NormalizerProcessorStep`` (if any), and the LeRobot helper from
        the loaded preprocessor pipeline. All stay ``None`` for absolute-action
        policies, which keep the model-space leftover verbatim (correct - their
        frame does not move). The deterministic step-count delay is untouched.
        """
        self._rtc_rebase_resolved = True
        bridge = self._processor_bridge
        if bridge is None or not bridge.has_preprocessor:
            return
        try:
            from lerobot.processor import NormalizerProcessorStep, RelativeActionsProcessorStep
        except ImportError:
            return

        relative_step = next(
            (
                step
                for step in bridge.preprocessor_steps
                if isinstance(step, RelativeActionsProcessorStep) and getattr(step, "enabled", False)
            ),
            None,
        )
        if relative_step is None:
            return

        try:
            from lerobot.policies.rtc import reanchor_relative_rtc_prefix
        except (ImportError, TypeError):
            # ImportError: lerobot predates the helper (<= 0.5.1 ships no
            # lerobot.policies.rtc.reanchor_relative_rtc_prefix). TypeError:
            # importing lerobot.policies.rtc on some releases (e.g. 0.5.1)
            # executes a module whose dataclass fails to build, so the import
            # raises at load time rather than cleanly missing the symbol.
            # Either way the helper is unavailable - degrade gracefully.
            logger.warning(
                "Relative-action RTC policy '%s' detected but the installed lerobot lacks "
                "a usable reanchor_relative_rtc_prefix; the chunk-seam prefix will be carried "
                "in a STALE coordinate frame. Upgrade lerobot to re-anchor the leftover against "
                "the current state.",
                type(self._policy).__name__,
            )
            return

        normalizer_step = next(
            (step for step in bridge.preprocessor_steps if isinstance(step, NormalizerProcessorStep)),
            None,
        )
        # Backfill the action layout the same way LeRobot's RTCInferenceEngine does:
        # a relative step that never learned its action names cannot build the
        # joint mask, so the re-anchor would convert the wrong dimensions.
        if relative_step.action_names is None:
            config = getattr(self._policy, "config", None)
            cfg_names = getattr(config, "action_feature_names", None) if config else None
            if cfg_names:
                relative_step.action_names = list(cfg_names)

        self._rtc_relative_step = relative_step
        self._rtc_normalizer_step = normalizer_step
        self._rtc_reanchor_fn = reanchor_relative_rtc_prefix
        logger.info(
            "RTC relative-action re-anchoring enabled for '%s' (LeRobot reanchor_relative_rtc_prefix)",
            type(self._policy).__name__,
        )

    def _absolute_rtc_leftover(self, leftover_model: torch.Tensor) -> torch.Tensor | None:
        """Convert a model-space leftover tail to absolute robot coordinates for re-anchoring.

        The leftover comes out of ``predict_action_chunk`` in the model's
        normalized relative space (anchored to the current observation). Running
        it through the postprocessor unnormalizes it and adds the cached state
        (``AbsoluteActionsProcessorStep``), yielding the same absolute coordinates
        LeRobot stores as the processed leftover. The conversion is element-wise
        per action, so postprocessing only the tail equals postprocessing the
        full chunk then slicing.

        Returns ``None`` (disabling re-anchoring this step, falling back to the
        model-space leftover) when the policy is not relative-action or the
        postprocessor does not yield a plain action tensor.
        """
        if self._rtc_relative_step is None:
            return None
        bridge = self._processor_bridge
        if bridge is None or not bridge.has_postprocessor:
            return None
        absolute = bridge.postprocess(leftover_model.clone())
        if not isinstance(absolute, torch.Tensor):
            return None
        return absolute.detach()

    def _predict_with_rtc(self, batch: dict[str, Any]) -> torch.Tensor:
        """Run inference using predict_action_chunk with RTC kwargs.

        This replaces select_action() for RTC-enabled policies. It:
        1. Resolves the inference delay (RTC ``d``) BEFORE inference so it can be
           forwarded to the denoiser, not just used to slice the result.
        2. Calls predict_action_chunk with inference_delay + prev_chunk_left_over
           + execution_horizon (lerobot's RTC denoiser requires inference_delay
           as the get_prefix_weights ``start`` arg; omitting it sends None and
           raises TypeError once a previous-chunk prefix exists).
        3. Tracks inference latency for the wall-clock delay-estimate fallback.
        4. Stores the new chunk's leftover for the next call.

        Args:
            batch: Observation batch tensors ready for the policy.

        Returns:
            Action tensor - first action(s) from the chunk, accounting for
            inference delay.
        """
        # Re-anchor the relative-action chunk-seam leftover against the CURRENT
        # robot state BEFORE inference (resolve the pipeline steps once).
        if not self._rtc_rebase_resolved:
            self._resolve_rtc_rebase_steps()

        # Resolve how many control steps the executor commits while THIS
        # inference is in flight - the RTC paper's `d`. It must be known BEFORE
        # inference because lerobot's RTC denoiser consumes it as a kwarg: it is
        # the `start` argument of RTCProcessor.get_prefix_weights, which freezes
        # the first `d` actions of the new chunk to the already-committed prefix
        # and linearly blends steps d..execution_horizon. It is ALSO the offset
        # the chunk-seam slice below skips. Prefer the DETERMINISTIC count the
        # runtime supplied (set_rtc_observed_delay): in a synchronous eval loop
        # the world is paused during inference so exactly 0 steps elapse, and in
        # the async overlap pipeline the count is a known integer. Deriving it
        # from wall-clock p95 latency instead is non-reproducible - it warms up
        # within an episode and varies run-to-run, so two otherwise-identical
        # seeded episodes drift apart at the seam. Fall back to the wall-clock
        # estimate (over PRIOR calls; this inference's own latency is not yet
        # known) only when no count was supplied (true-async hardware driven
        # without a runner, where the arm really does keep moving during
        # inference).
        observed = self.rtc_observed_delay_steps
        if observed is not None:
            inference_delay = max(0, int(observed))
        else:
            # The delay is (p95 latency * control_frequency), so it MUST use the
            # loop's real control rate; assuming a fixed rate makes the
            # chunk-seam blend wrong at every other frequency. The runtime
            # (PolicyRunner) calls set_control_frequency() before the loop.
            fps = self.control_frequency
            if fps is None:
                if not self._rtc_freq_warned:
                    logger.warning(
                        "RTC: control_frequency unknown for '%s', falling back to "
                        "%.0fHz - inference-delay estimation will be off at any other "
                        "control rate. Call set_control_frequency(hz) before running "
                        "(PolicyRunner does this automatically).",
                        type(self._policy).__name__,
                        _RTC_FALLBACK_FPS,
                    )
                    self._rtc_freq_warned = True
                fps = _RTC_FALLBACK_FPS
            inference_delay = self._estimate_inference_delay(fps=fps)

        # Build RTC kwargs for the flow-matching denoiser. inference_delay is
        # passed on EVERY call: lerobot's RTCProcessor.denoise_step computes
        # get_prefix_weights(inference_delay, execution_horizon, T), and
        # `min(inference_delay, execution_horizon)` raises TypeError the moment a
        # prev_chunk_left_over prefix exists (2nd chunk onward) if it is None.
        # On the first chunk prev_chunk_left_over is None and lerobot returns
        # early, so the value is harmless there.
        rtc_kwargs: dict[str, Any] = {"inference_delay": inference_delay}
        # Relative-action policies: re-express the leftover tail against the
        # CURRENT robot state instead of carrying a stale-frame prefix. The
        # leftover is kept in absolute coordinates (_rtc_prev_chunk_abs);
        # LeRobot's reanchor helper subtracts the live cached state and
        # re-normalizes so the model receives a correctly anchored prefix.
        # Absolute-action policies fall through to the verbatim leftover.
        prev_chunk = self._rtc_prev_chunk
        if (
            self._rtc_relative_step is not None
            and self._rtc_reanchor_fn is not None
            and self._rtc_prev_chunk_abs is not None
            and self._rtc_prev_chunk_abs.numel() > 0
        ):
            current_state = self._rtc_relative_step.get_cached_state()
            if current_state is not None:
                prev_chunk = self._rtc_reanchor_fn(
                    prev_actions_absolute=self._rtc_prev_chunk_abs,
                    current_state=current_state,
                    relative_step=self._rtc_relative_step,
                    normalizer_step=self._rtc_normalizer_step,
                    policy_device=self._device or "cpu",
                )
        if prev_chunk is not None:
            rtc_kwargs["prev_chunk_left_over"] = prev_chunk
        if self._rtc_execution_horizon is not None:
            rtc_kwargs["execution_horizon"] = self._rtc_execution_horizon

        # predict_action_chunk returns (batch, chunk_size, action_dim)
        assert self._policy is not None, "Policy not loaded"
        inference_start = time.time()
        action_chunk = self._policy.predict_action_chunk(batch, **rtc_kwargs)

        inference_elapsed = time.time() - inference_start
        self._rtc_latency_history.append(inference_elapsed)

        # Remove batch dim if present: (1, T, A) → (T, A)
        if action_chunk.dim() == 3 and action_chunk.shape[0] == 1:
            action_chunk = action_chunk.squeeze(0)

        # Store leftover for next RTC call (unconsumed portion of this chunk).
        # The consumer executes ``execution_horizon`` actions before re-querying
        # (see Policy.execution_horizon / resolve_chunk_length), so the tail past
        # that point - shifted by the steps already burned during inference - is
        # what carries into the next chunk as ``prev_chunk_left_over``. Keying
        # this on the full trained chunk (actions_per_step) emptied the tail
        # whenever the chunk was consumed whole, so cross-chunk blending never
        # engaged.
        exec_horizon = self._rtc_execution_horizon or self.actions_per_step
        steps_to_consume = min(inference_delay + max(1, int(exec_horizon)), action_chunk.shape[0])
        if steps_to_consume < action_chunk.shape[0]:
            leftover_model = action_chunk[steps_to_consume:].detach()
            self._rtc_prev_chunk = leftover_model
            # For relative-action policies, also stash the leftover in absolute
            # robot coordinates so the NEXT call can re-anchor it against the new
            # state. None for absolute-action policies (no frame shift to undo).
            self._rtc_prev_chunk_abs = self._absolute_rtc_leftover(leftover_model)
        else:
            self._rtc_prev_chunk = None
            self._rtc_prev_chunk_abs = None

        # Skip delay steps - they correspond to time spent during inference
        usable_start = min(inference_delay, action_chunk.shape[0] - 1)
        usable_actions = action_chunk[usable_start:]

        # Log RTC details at debug level - throttled to once every 2s regardless of Hz
        _now = time.monotonic()
        if _now - self._rtc_last_log_time >= 2.0:
            self._rtc_last_log_time = _now
            logger.debug(
                "RTC: chunk=%s, delay=%d, usable_start=%d, leftover=%s, avg_latency=%.3fs",
                action_chunk.shape,
                inference_delay,
                usable_start,
                self._rtc_prev_chunk.shape if self._rtc_prev_chunk is not None else None,
                sum(self._rtc_latency_history) / len(self._rtc_latency_history),
            )

        return usable_actions

    # Inference

    async def get_actions(self, observation_dict: dict[str, Any], instruction: str, **kwargs) -> list[dict[str, Any]]:
        """Get actions from policy given observation and instruction.

        Args:
            observation_dict: Robot observation (cameras + state).
            instruction: Natural language instruction.

        Returns:
            List of action dicts for robot execution.

        Raises:
            RuntimeError: If model is not loaded and no path is set.
        """
        if not self._loaded:
            if self.pretrained_name_or_path:
                self._load_model()
            else:
                raise RuntimeError(
                    "No model loaded and no pretrained_name_or_path set. Create the policy with a model path."
                )

        observation = dict(observation_dict)
        if instruction and "task" not in observation:
            observation["task"] = instruction

        # When the processor bridge has a preprocessor, delegate normalization
        # and tokenization to it, then fix up any remaining raw arrays/tensors
        # that the pipeline did not convert (e.g. images left as HWC uint8
        # numpy, state tensors missing a batch dimension).
        if self._processor_bridge and self._processor_bridge.has_preprocessor:
            if self._embodiment is not None:
                # SOLUTION.md declarative path: the embodiment map was injected
                # into the pipeline at load time (rename_map + strands_pack_state
                # step), so the pipeline itself renames cameras and composes
                # observation.state. Feed RAW obs straight in - ZERO per-step
                # strands-side remapping.
                observation = self._canonicalize_obs_images(
                    observation, image_source_keys=self._embodiment_image_source_keys()
                )
                batch = self._processor_bridge.preprocess(observation, instruction=instruction)
                if not isinstance(batch, dict):
                    batch = {"observation.state": batch}
                # A standard lerobot preprocessor pipeline ends in
                # AddBatchDimension + Device steps that batch every tensor and
                # move it to the policy device, so _fixup is a no-op there. But a
                # MINIMAL pipeline does not: the norm_stats.json fallback builds
                # ``DataProcessorPipeline(steps=[normalizer])`` only (no batch /
                # device step), so without this the model receives an unbatched
                # ``observation.state`` (D,) alongside a (1, C, H, W) image -> a
                # torch.stack rank mismatch inside select_action. _fixup is
                # idempotent (it leaves already-batched, already-CHW, already-on-
                # device tensors untouched), so run it on BOTH paths to make the
                # batched-tensor contract independent of which steps the
                # checkpoint's pipeline happens to ship.
                batch = self._fixup_preprocessed_batch(batch)
            else:
                # Legacy heuristic path (no embodiment declared). B12: remap
                # strands-native obs (bare camera names + per-joint scalars) to
                # the model's LeRobot feature names BEFORE preprocess, then fix up
                # any arrays/tensors the pipeline left unconverted.
                lerobot_obs = self._to_lerobot_observation(observation)
                lerobot_obs = self._canonicalize_obs_images(lerobot_obs)
                batch = self._processor_bridge.preprocess(lerobot_obs, instruction=instruction)
                if not isinstance(batch, dict):
                    batch = {"observation.state": batch}
                batch = self._fixup_preprocessed_batch(batch)
        else:
            batch = self._build_observation_batch(observation, instruction)

        with torch.inference_mode():
            assert self._policy is not None
            self._policy.eval()
            # RTC uses predict_action_chunk() directly with cross-chunk guidance;
            # non-RTC uses select_action() which manages temporal ensemble + action queue.
            if self._rtc_enabled:
                # RTC (Real-Time Chunking) path: calls predict_action_chunk() directly
                # with prev_chunk for temporal blending. Used only for flow-matching
                # policies that support rtc_config.
                action_tensor = self._predict_with_rtc(batch)
            elif self.actions_per_step > 1 or self._requires_action_chunk():
                # Multi-step path: call predict_action_chunk() directly to get the
                # full action horizon, then slice in _tensor_to_action_dicts().
                # select_action() uses an internal queue and returns only 1 action
                # at a time, so it can't return multiple steps per call.
                #
                # Some policies (e.g. MolmoAct2) must ALWAYS take this path even at
                # actions_per_step=1: their select_action() raises AssertionError
                # when the checkpoint's rtc_config is enabled, and otherwise still
                # serves only a single action per call. _requires_action_chunk()
                # detects them so they never hit the crashing select_action() below.
                action_tensor = self._policy.predict_action_chunk(batch, **self.inference_kwargs)
            else:
                # Default single-step path: delegates to LeRobot's select_action()
                # which handles temporal ensemble smoothing
                # (config.temporal_ensemble_coeff) and action queue management
                # (n_action_steps > 1) internally.
                # We intentionally use select_action() rather than predict_action_chunk()
                # here to preserve all upstream action scheduling logic.
                # inference_kwargs forwards policy-specific runtime selectors
                # (e.g. MolmoAct2's required inference_action_mode).
                action_tensor = self._policy.select_action(batch, **self.inference_kwargs)

        if self._processor_bridge and self._processor_bridge.has_postprocessor:
            action_tensor = self._processor_bridge.postprocess(action_tensor)

        return self._tensor_to_action_dicts(action_tensor)

    # Observation batch building

    def _canonicalize_obs_images(
        self, observation: dict[str, Any], image_source_keys: set[str] | None = None
    ) -> dict[str, Any]:
        """Normalize image-array layout BEFORE the preprocessor runs.

        LeRobot's normalizer step inside ``preprocess()`` expects images as
        channel-first ``(C, H, W)`` (or batched ``(B, C, H, W)``) tensors so its
        per-channel mean/std broadcast correctly. Direct ``get_actions`` callers
        commonly pass camera frames as HWC numpy arrays (the natural format from
        OpenCV / a renderer), and uint8 frames in [0, 255]. Feeding those raw
        makes the normalizer either broadcast a 3-vector against the 480 height
        (``size of tensor a (480) must match b (3)``) or overflow uint8.

        This converts every ``observation.images.*`` entry to CHW ``float32`` in
        [0, 1] up front, so HWC-uint8, HWC-float, CHW-float, and torch/np inputs
        all work. Non-image entries pass through untouched. Runs before BOTH the
        embodiment and legacy preprocess paths; ``_fixup_preprocessed_batch``
        still adds the batch dimension afterwards.

        In the declarative embodiment path the camera rename
        (``front`` -> ``observation.images.front``) happens INSIDE the pipeline,
        so the observation handed here still carries the bare source key, which
        lacks the ``"image"`` substring and would be skipped -- leaving the frame
        as raw HWC uint8 for the normalizer to overflow on. ``image_source_keys``
        (the embodiment ``obs_rename`` sources that target a declared image
        feature; see :meth:`_embodiment_image_source_keys`) names those bare keys
        so they are canonicalized too.

        Args:
            observation: Raw observation dict (bare or LeRobot-keyed).
            image_source_keys: Extra non-``image`` keys to treat as camera
                frames (the embodiment's image rename sources). ``None`` keeps
                the legacy ``"image"``-substring-only behavior.

        Returns:
            A new dict with every recognized camera frame as CHW ``float32``.
        """
        out = dict(observation)
        for key, val in observation.items():
            is_image = "image" in key or (image_source_keys is not None and key in image_source_keys)
            if not is_image or val is None:
                continue
            if isinstance(val, np.ndarray):
                img = torch.from_numpy(val)
            elif isinstance(val, torch.Tensor):
                img = val
            else:
                continue  # leave exotic types for the pipeline to handle
            # uint8 [0,255] -> float32 [0,1]; other ints/floats just cast.
            if img.dtype == torch.uint8:
                img = img.float() / 255.0
            else:
                img = img.float()
            # HWC -> CHW (a trailing channel dim of 1/3/4 is the giveaway).
            if img.ndim == 3 and img.shape[-1] in (1, 3, 4) and img.shape[0] not in (1, 3, 4):
                img = img.permute(2, 0, 1)
            out[key] = img
        return out

    def _embodiment_image_source_keys(self) -> set[str]:
        """Bare observation keys the embodiment renames onto a declared image feature.

        In the declarative path the camera rename runs INSIDE the preprocessor
        pipeline, so the observation reaching :meth:`_canonicalize_obs_images`
        still uses the robot-native source key (e.g. ``front``), not the model
        feature name (``observation.images.front``). Returning those source
        keys lets canonicalization recognize them as camera frames even though
        they lack the ``"image"`` substring.

        Returns:
            The set of ``obs_rename`` sources whose target is a declared image
            (VISUAL) input feature. Empty when no embodiment is configured.
        """
        emb = self._embodiment
        if emb is None:
            return set()
        return {
            src for src, dst in emb.obs_rename.items() if _declared_feature_is_image(dst, self._input_features.get(dst))
        }

    def _fixup_preprocessed_batch(self, batch: dict[str, Any]) -> dict[str, Any]:
        """Fix up a preprocessor-produced batch so every value is a proper batched tensor.

        The LeRobot DataProcessorPipeline may leave some entries in their
        original format (e.g. images as HWC uint8 numpy arrays, state tensors
        without a leading batch dimension).  This method ensures every value
        is a ``torch.Tensor`` on the correct device with the shapes that
        ``policy.select_action()`` expects:

        - Images: ``(B, C, H, W)`` float32
        - State: ``(B, D)`` float32
        - Language tokens/mask: already batched by the tokenizer step

        Args:
            batch: Dict from ``ProcessorBridge.preprocess()``.

        Returns:
            Dict with all values as properly shaped device tensors.
        """
        import torch

        device = self._device or "cpu"
        fixed: dict[str, Any] = {}

        for key, val in batch.items():
            # numpy arrays → torch tensors
            if isinstance(val, np.ndarray):
                if "image" in key:
                    # HWC uint8 → CHW float32 → (1,C,H,W)
                    img = torch.from_numpy(val).float()
                    if img.ndim == 3 and img.shape[-1] in (1, 3, 4):
                        img = img.permute(2, 0, 1)  # HWC → CHW
                    if img.ndim == 3:
                        img = img.unsqueeze(0)  # CHW → (1,C,H,W)
                    fixed[key] = img.to(device)
                else:
                    t = torch.from_numpy(val).float()
                    if t.ndim == 1:
                        t = t.unsqueeze(0)  # (D,) → (1,D)
                    fixed[key] = t.to(device)

            # torch tensors: ensure batch dim + device
            elif isinstance(val, torch.Tensor):
                # Auto-cast float64 → float32: ROS/dynamixel drivers often produce float64
                t = val.float() if val.dtype == torch.float64 else val
                if "image" in key:
                    if t.ndim == 3 and t.shape[-1] in (1, 3, 4):
                        t = t.permute(2, 0, 1)  # HWC → CHW
                    if t.ndim == 3:
                        t = t.unsqueeze(0)  # CHW → (1,C,H,W)
                elif t.ndim == 1:
                    t = t.unsqueeze(0)  # (D,) → (1,D)
                fixed[key] = t.to(device)

            # pass through anything else (strings, etc.)
            else:
                fixed[key] = val

        return fixed

    def _resolve_state_order(self, observation_dict: dict[str, Any], scalar_keys: list[str]) -> list[str]:
        """Resolve which observation keys hold this step's joint-state vector.

        Honors ``robot_state_keys`` when at least one of them is present in the
        observation. When ``robot_state_keys`` is non-empty but NONE of its
        keys appear in the observation, the configured keys cannot describe
        this robot/sim - the canonical case being auto-generated generic names
        (``joint_0..joint_N``, derived from the model action dim) paired with a
        sim/robot that reports named joints (``shoulder_pan`` ...). That
        mismatch previously yielded a zero/empty ``observation.state`` silently,
        so the policy ran open-loop while reporting no error. It is now LOUD:

          * ``strict_keys=True``  -> raise :class:`ValueError` naming the actual
            observation keys and pointing at ``embodiment=`` /
            ``set_robot_state_keys()``.
          * ``strict_keys=False`` -> warn once with the same guidance, then
            fall back to the observation's own scalar keys so the state is
            populated rather than silently dropped.

        Args:
            observation_dict: Raw strands/sim observation for this step.
            scalar_keys: Non-image, non-``task`` scalar keys present in the
                observation, used as the fallback ordering.

        Returns:
            The ordered keys to pull scalar joint values from.

        Raises:
            ValueError: When ``strict_keys`` is set and no configured key
                matches the observation.
        """
        if not self.robot_state_keys:
            return scalar_keys
        if any(k in observation_dict for k in self.robot_state_keys):
            return self.robot_state_keys
        shown = self.robot_state_keys[:8]
        ellipsis = "..." if len(self.robot_state_keys) > 8 else ""
        msg = (
            f"None of the configured robot_state_keys {shown}{ellipsis} are present "
            f"in the observation. Observed joint/state keys: {scalar_keys}. This "
            "usually means generic auto-generated keys (joint_0..joint_N) were "
            "paired with a robot/sim that reports named joints. Pass "
            "embodiment='<name>' (e.g. embodiment='so101') or call "
            "set_robot_state_keys([...]) with the robot's actual joint names."
        )
        if self.strict_keys:
            raise ValueError("strict_keys=True: " + msg)
        # Surface the degraded binding as machine-detectable telemetry
        # (run_policy / eval_policy read generic_state_keys_used) and warn at
        # most once per policy so a long rollout is not spammed.
        self.generic_state_keys_used = True
        if not self._state_key_mismatch_warned:
            logger.warning(msg)
            self._state_key_mismatch_warned = True
        return scalar_keys

    def _collect_state_values(self, observation_dict: dict[str, Any], order: list[str]) -> list[float]:
        """Pull the joint-state vector from ``observation_dict`` in ``order``.

        Every key in ``order`` contributes exactly one slot, in order, so the
        emitted vector is index-aligned with the resolved key list. A key that
        is present as a scalar (Python/NumPy number or 0-d array) contributes
        its value; a key that is ABSENT from the observation (or present but
        non-scalar) contributes ``0.0`` IN PLACE - it is never dropped.

        Dropping an absent key instead (the prior behaviour) shifted every
        following joint value up by one index before the caller's trailing
        zero-pad, silently mis-aligning ``observation.state``. The canonical
        trigger is a mimic/tendon gripper whose actuator name (e.g. aloha's
        ``left/gripper``) is not among the observation's finger-joint names
        while the arm joints are: the arm values slid into the gripper slots.
        ``so101`` and any robot whose keys are all present are unaffected.

        When ``order`` is ``robot_state_keys`` and some of them are missing
        (a partial mismatch), the degradation is surfaced - ``strict_keys``
        raises, otherwise it warns once and sets ``missing_state_keys_used`` -
        mirroring :meth:`_resolve_state_order`'s all-missing guard. The
        observation's own scalar-key fallback (``order`` != ``robot_state_keys``)
        has no missing keys by construction, so this never double-fires with
        the all-missing path.

        Args:
            observation_dict: Raw strands/sim observation for this step.
            order: Resolved key ordering from :meth:`_resolve_state_order`.

        Returns:
            One float per key in ``order`` (missing keys zero-filled in place).

        Raises:
            ValueError: When ``strict_keys`` is set and a resolved
                ``robot_state_keys`` entry is missing from the observation.
        """
        values: list[float] = []
        missing: list[str] = []
        for key in order:
            v = observation_dict.get(key)
            if isinstance(v, bool):
                # bool is an int subclass but never a joint reading.
                missing.append(key)
                values.append(0.0)
            elif isinstance(v, (int, float, np.floating, np.integer)):
                values.append(float(v))
            elif isinstance(v, np.ndarray) and v.ndim == 0:
                values.append(float(v))
            else:
                missing.append(key)
                values.append(0.0)
        # Only a robot_state_keys ordering can contain a key absent from the
        # observation; the scalar-key fallback is built from present keys.
        if missing and order is self.robot_state_keys:
            shown = missing[:8]
            ellipsis = "..." if len(missing) > 8 else ""
            msg = (
                f"{len(missing)} configured robot_state_keys are not present in the "
                f"observation: {shown}{ellipsis}. Present joints keep their index and the "
                "missing dims are zero-filled in place, but the sim/robot does not report "
                "those joints - commonly a mimic/tendon gripper whose actuator name differs "
                "from the observation's finger-joint names. Pass embodiment='<name>' or call "
                "set_robot_state_keys([...]) with names the observation actually contains."
            )
            if self.strict_keys:
                raise ValueError("strict_keys=True: " + msg)
            self.missing_state_keys_used = True
            if not self._state_missing_keys_warned:
                logger.warning(msg)
                self._state_missing_keys_warned = True
        return values

    def _to_lerobot_observation(self, observation_dict: dict[str, Any]) -> dict[str, Any]:
        """Remap a strands-native observation to LeRobot feature keys.

        The processor pipeline expects the model's declared feature names:
        ``observation.images.<cam>`` for cameras and ``observation.state`` for
        the joint vector. Robot/sim observations instead use bare camera names
        (``image``, ``wrist_image``) and per-joint scalar keys. This bridges the
        two so a preprocessor-backed VLA (MolmoAct2, etc.) gets the keys it needs.

        Mapping rules:
          * Keys already starting with ``observation.`` (and ``task``) pass through.
          * ndarray values with ndim>=2 (images) are matched to the model's
            declared image features by exact name first - either the prefixed
            form (``image`` → ``observation.images.image``) or a BARE declared
            key (``base`` → ``base``, as VLAs like MolmoAct2 declare them);
            otherwise images fill remaining declared image slots in order.
          * Remaining scalar joint values are collected (in ``robot_state_keys``
            order when available, else insertion order) into ``observation.state``.

        Idempotent: a fully LeRobot-formatted observation is returned unchanged.
        """
        # Already LeRobot-formatted? (any observation.* key) → pass through.
        if any(k.startswith("observation.") for k in observation_dict):
            return dict(observation_dict)

        out: dict[str, Any] = {}

        declared_img_feats = [f for f, feat in self._input_features.items() if _declared_feature_is_image(f, feat)]

        # 1) Map images. An explicit camera_key_map wins (mirroring the routing
        #    precedence in _resolve_camera_targets), then an exact short-name
        #    match against declared features, then positional fill. Honoring the
        #    map here - not only on the batch path - means a camera-name mismatch
        #    bound via camera_key_map is resolved on this preprocessor/VLA path
        #    (MolmoAct2 etc.) too, so the strict_keys remedy advertised below
        #    ("Provide an explicit mapping (camera_key_map)") actually fixes the
        #    failure it points at instead of being silently ignored.
        image_items = [(k, v) for k, v in observation_dict.items() if isinstance(v, np.ndarray) and v.ndim >= 2]
        used_feats: set[str] = set()
        unmatched_imgs = []
        for k, v in image_items:
            mapped = self.camera_key_map.get(k) if self.camera_key_map else None
            if mapped is not None:
                if mapped not in declared_img_feats:
                    raise ValueError(
                        f"camera_key_map routes camera '{k}' to image key '{mapped}', "
                        "but the policy does not declare it. Declared image keys: "
                        f"{sorted(declared_img_feats)}."
                    )
                if mapped not in used_feats:
                    out[mapped] = v
                    used_feats.add(mapped)
                continue
            prefixed = f"observation.images.{k}"
            if prefixed in self._input_features and prefixed not in used_feats:
                out[prefixed] = v
                used_feats.add(prefixed)
            elif k in declared_img_feats and k not in used_feats:
                # VLAs such as MolmoAct2 declare bare image keys (``base`` /
                # ``wrist``) rather than the ``observation.images.<cam>`` form.
                # Mirror the exact-name precedence already in
                # ``_resolve_camera_targets`` so a camera whose name equals a
                # bare declared key binds BY NAME, not positionally - otherwise
                # every camera falls through to positional fill, which shuffles
                # views (and drops a real one when an extra free camera such as
                # the sim ``default`` is present) and emits a misleading
                # "does not match" warning even on an exact-name hit.
                out[k] = v
                used_feats.add(k)
            else:
                unmatched_imgs.append((k, v))
        # Fill any remaining declared image slots, in declaration order.
        free_feats = [f for f in declared_img_feats if f not in used_feats]
        if self.strict_keys and unmatched_imgs and free_feats:
            raise ValueError(
                "strict_keys=True: cannot resolve camera keys by exact name. "
                f"Unmatched robot keys: {sorted(k for k, _ in unmatched_imgs)}; "
                f"available model keys: {sorted(free_feats)}. "
                "Provide an explicit mapping (camera_key_map) "
                "or set strict_keys=False to allow positional fallback."
            )
        for (k, v), feat in zip(unmatched_imgs, free_feats):
            self.positional_fallback_used = True
            logger.warning(
                "Camera '%s' does not match any declared policy image key by name; "
                "routing positionally to '%s'. The frame may feed the wrong model "
                "input - pass camera_key_map to bind cameras explicitly (or "
                "strict_keys=True to make this an error).",
                k,
                feat,
            )
            out[feat] = v
            used_feats.add(feat)

        # 2) Collect scalar joint values into observation.state.
        scalar_keys = [
            k for k, v in observation_dict.items() if k != "task" and not (isinstance(v, np.ndarray) and v.ndim >= 2)
        ]
        # Resolve the joint-state ordering, raising/warning loudly when the
        # configured robot_state_keys cannot describe this observation (the
        # generic joint_0..N vs named-joint mismatch) instead of silently
        # producing a zero/empty state (see _resolve_state_order).
        order = self._resolve_state_order(observation_dict, scalar_keys)
        # Build the vector index-aligned with ``order`` - a resolved key absent
        # from the observation is zero-filled IN PLACE (not dropped, which would
        # shift following joints into the wrong slots) and surfaced loudly.
        state_vals = self._collect_state_values(observation_dict, order)
        if state_vals:
            # Adapt state dim to the model's declared observation.state shape.
            # The preprocessor's normalizer does element-wise ops against fixed
            # N-dim stats (e.g. LIBERO Franka = 8), so a 6-dof SO arm must be
            # zero-padded (or truncated) to N BEFORE preprocessing or the
            # pipeline raises a shape-mismatch. Mirrors the adaptation in
            # _build_batch_from_strands_format.
            state_feat = self._input_features.get("observation.state")
            expected_dim = (
                state_feat.shape[0]
                if state_feat is not None and getattr(state_feat, "shape", None)
                else len(state_vals)
            )
            if len(state_vals) > expected_dim:
                logger.warning(
                    "State dim %d > model expects %d - truncating (preprocess path).",
                    len(state_vals),
                    expected_dim,
                )
                state_vals = state_vals[:expected_dim]
            elif len(state_vals) < expected_dim:
                logger.warning(
                    "State dim %d < model expects %d - zero-padding (preprocess path).",
                    len(state_vals),
                    expected_dim,
                )
                state_vals = state_vals + [0.0] * (expected_dim - len(state_vals))
            out["observation.state"] = np.asarray(state_vals, dtype=np.float32)

        # 3) Preserve task/instruction passthrough.
        if "task" in observation_dict:
            out["task"] = observation_dict["task"]

        return out

    def _build_observation_batch(self, observation_dict: dict[str, Any], instruction: str) -> dict[str, Any]:
        """Convert observation dict to LeRobot-compatible batch tensors.

        Handles two observation formats:
        1. LeRobot native: keys prefixed with "observation." (e.g. "observation.state")
        2. strands-robots native: individual joint keys (e.g. "shoulder", "elbow")

        For VLA models, injects tokenized language instructions into the batch
        as "observation.language.tokens" and "observation.language.attention_mask".

        Args:
            observation_dict: Raw observation dict from robot/sim.
            instruction: Natural language instruction for VLA models.

        Returns:
            Dict of tensors ready for LeRobot policy.select_action().
        """
        batch: dict[str, Any] = {}

        has_lerobot_keys = any(key.startswith("observation.") for key in observation_dict)
        if has_lerobot_keys:
            batch = self._build_batch_from_lerobot_format(observation_dict, batch)
        else:
            batch = self._build_batch_from_strands_format(observation_dict, batch)

        # Inject tokenized language instruction for VLA models.
        # VLA models that use language tokenization expect language tokens as part
        # of the observation batch. We only inject if the model declares
        # language-related input features (tokenizer_name, vlm_model_name).
        if instruction and "observation.language.tokens" not in batch and self._needs_language_tokens():
            result = self._tokenize_instruction(instruction)
            if result is not None:
                tokens, mask = result
                batch["observation.language.tokens"] = tokens
                if mask is not None:
                    batch["observation.language.attention_mask"] = mask
                logger.debug("VLA tokenized instruction: '%s...' -> %s tokens", instruction[:50], tokens.shape)

        # Fill task key for models that read it directly from the batch
        # (e.g. some VLA models read "task" or "observation.task" from the
        # input dict rather than using tokenized language tokens)
        if instruction and has_lerobot_keys:
            needs_task = any("task" in key for key in self._input_features) and "task" not in batch
            if needs_task:
                for feat_name in self._input_features:
                    if "task" in feat_name and feat_name not in batch:
                        batch[feat_name] = instruction

        # Validate required image features are present. Missing images would
        # cause the model to produce garbage outputs silently.
        for feat_name in self._input_features:
            if feat_name not in batch and "image" in feat_name:
                raise ValueError(
                    f"Missing required image feature '{feat_name}' in observation. "
                    f"The model expects this camera input. Provide it in the observation dict "
                    f"or check your camera configuration."
                )

        return batch

    def _build_batch_from_lerobot_format(
        self, observation_dict: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Build batch from observation dict already in LeRobot format (observation.* keys).

        Converts each value to the appropriate tensor format:
        - Images (HWC uint8) → CHW float32 [0, 1] with batch dim
        - State vectors → float32 with batch dim
        - Scalars → float32 tensor with batch dim

        Non-numeric types (strings, pre-batched int64 tokens) are passed through
        unchanged - LeRobot expects these as-is for task descriptions and
        pre-tokenized inputs.

        Args:
            observation_dict: Dict with "observation.*" prefixed keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with tensors on target device.
        """
        for key, value in observation_dict.items():
            # Determine if this key represents an image based on key name
            # or shape metadata from the model's input_features config
            is_image = "image" in key or (
                key in self._input_features
                and hasattr(self._input_features.get(key), "shape")
                and len(getattr(self._input_features.get(key), "shape", ())) >= 2
            )

            if isinstance(value, torch.Tensor):
                # Auto-cast float64 → float32: common from ROS/dynamixel drivers
                tensor = value.float() if value.dtype == torch.float64 else value
                # Detect unlabeled images by shape: 3D tensor with channel-last layout
                if not is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    is_image = True
                # HWC → CHW: LeRobot expects channel-first image layout
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # Add batch dimension (required by policy.select_action)
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif tensor.dim() < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, np.ndarray):
                tensor = torch.from_numpy(value.copy()).float()
                if not is_image and value.ndim == 3 and value.shape[-1] in (1, 3, 4):
                    is_image = True
                if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                    tensor = tensor.permute(2, 0, 1)
                # uint8 images are [0, 255] - normalize to [0, 1] for model input
                if is_image and value.dtype == np.uint8:
                    tensor = tensor / 255.0
                if is_image and tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)
                elif value.ndim < 2 and not is_image:
                    tensor = tensor.unsqueeze(0)
                batch[key] = tensor.to(self._device)

            elif isinstance(value, (int, float)):
                batch[key] = torch.tensor([value], dtype=torch.float32).unsqueeze(0).to(self._device)

            elif isinstance(value, (list, tuple)):
                try:
                    array = np.array(value, dtype=np.float32)
                except (ValueError, TypeError):
                    # Non-numeric lists (e.g. string lists) - skip silently, they aren't tensor data
                    logger.debug("Skipping non-numeric list/tuple for key in observation batch")
                    continue
                tensor = torch.from_numpy(array).float()
                if array.ndim >= 2:
                    if is_image and tensor.dim() == 3 and tensor.shape[-1] in (1, 3, 4):
                        tensor = tensor.permute(2, 0, 1)
                    if is_image and array.dtype == np.uint8:
                        tensor = tensor / 255.0
                    if is_image and tensor.dim() == 3:
                        tensor = tensor.unsqueeze(0)
                    batch[key] = tensor.to(self._device)
                else:
                    batch[key] = tensor.unsqueeze(0).to(self._device)

        return batch

    def _policy_image_keys(self) -> list[str]:
        """Return the policy's ordered image feature keys.

        Prefers the model config's declared ``image_keys`` - an explicit ordered
        list used by VLAs such as MolmoAct2 to bind each camera to a fixed model
        slot. Falls back to the image entries of ``_input_features`` in
        declaration order when no such list is declared.

        Returns:
            Ordered list of policy image feature keys (e.g.
            ``["observation.images.top", "observation.images.wrist"]``).
        """
        cfg = getattr(self._policy, "config", None)
        declared = getattr(cfg, "image_keys", None) if cfg is not None else None
        if isinstance(declared, (list, tuple)) and declared:
            return [str(k) for k in declared]
        return [feat for feat, feature in self._input_features.items() if _declared_feature_is_image(feat, feature)]

    def _resolve_camera_targets(self, cam_names: list[str]) -> dict[str, str]:
        """Map robot/sim camera names to the policy's declared image feature keys.

        Routing precedence:
          1. Explicit ``camera_key_map`` ctor param wins for any name it lists.
          2. Exact name match: ``top`` -> ``observation.images.top`` (or a
             directly-declared ``top``) when the policy declares that key.
          3. Positional fallback: remaining cameras fill the remaining declared
             image slots in declaration order, with a WARN that the names did
             not match (so a wrong wiring is loud, not silent).

        Args:
            cam_names: Camera names present in the observation.

        Returns:
            Mapping of camera name -> policy image feature key. Cameras beyond
            what the policy consumes are omitted.

        Raises:
            ValueError: If an explicit mapping targets an undeclared image key,
                or the robot supplies fewer cameras than the policy requires.
        """
        targets = self._policy_image_keys()
        result: dict[str, str] = {}
        used: set[str] = set()

        # 1) Explicit camera_key_map wins.
        if self.camera_key_map:
            for cam, feat in self.camera_key_map.items():
                if targets and feat not in targets:
                    raise ValueError(
                        f"camera_key_map routes camera '{cam}' to image key '{feat}', "
                        f"but the policy does not declare it. Declared image keys: {targets}."
                    )
                if cam in cam_names:
                    result[cam] = feat
                    used.add(feat)

        # 2) Exact name match for the cameras still needing a target.
        unmatched: list[str] = []
        for cam in cam_names:
            if cam in result:
                continue
            short = f"observation.images.{cam}"
            if short in targets and short not in used:
                result[cam] = short
                used.add(short)
            elif cam in targets and cam not in used:
                result[cam] = cam
                used.add(cam)
            else:
                unmatched.append(cam)

        # 3) Positional fallback into the remaining declared slots (loud).
        free = [feat for feat in targets if feat not in used]
        if self.strict_keys and unmatched and free:
            raise ValueError(
                "strict_keys=True: cannot resolve camera keys by exact name. "
                f"Unmatched robot keys: {sorted(unmatched)}; "
                f"available model keys: {sorted(free)}. "
                "Provide an explicit mapping (camera_key_map) "
                "or set strict_keys=False to allow positional fallback."
            )
        for cam, feat in zip(unmatched, free):
            self.positional_fallback_used = True
            logger.warning(
                "Camera '%s' does not match any declared policy image key by name; "
                "routing positionally to '%s'. Pass camera_key_map to bind cameras "
                "explicitly and silence this warning.",
                cam,
                feat,
            )
            result[cam] = feat
            used.add(feat)

        # 4) Hard error if the policy still has image slots the robot cannot fill.
        unfilled = [feat for feat in targets if feat not in used]
        if unfilled:
            raise ValueError(
                f"Robot supplies {len(cam_names)} camera(s) {cam_names} but the policy "
                f"requires image input(s) {targets}; unmatched policy keys: {unfilled}. "
                f"Add the missing camera(s) to the observation or pass camera_key_map."
            )

        return result

    def _build_batch_from_strands_format(
        self, observation_dict: dict[str, Any], batch: dict[str, Any]
    ) -> dict[str, Any]:
        """Build batch from strands-robots native observation format.

        Maps individual joint keys (e.g. {"shoulder": 0.5, "elbow": -0.3}) to
        LeRobot's "observation.state" tensor using robot_state_keys ordering.
        Camera images are matched to the model's image input features by
        assigning each ndarray with ndim >= 2 to the first unoccupied image slot.

        Args:
            observation_dict: Dict with individual joint/image keys.
            batch: Existing batch dict to extend.

        Returns:
            Updated batch dict with "observation.state" and image tensors.

        Raises:
            ValueError: If robot_state_keys is empty (cannot map joints).
        """
        if not self.robot_state_keys:
            raise ValueError(
                "robot_state_keys is empty - cannot map observation to state tensor. "
                "Call set_robot_state_keys() with the robot's motor names."
            )

        # Resolve the joint-state ordering. When the configured keys match
        # nothing in the observation (generic joint_0..N vs named joints), this
        # raises (strict_keys) or warns + falls back to the observation's own
        # scalar keys, instead of silently dropping observation.state and
        # running the policy open-loop (see _resolve_state_order).
        scalar_keys = [
            k for k, v in observation_dict.items() if k != "task" and not (isinstance(v, np.ndarray) and v.ndim >= 2)
        ]
        order = self._resolve_state_order(observation_dict, scalar_keys)

        # Collect state values index-aligned with the resolved order. A key in
        # ``order`` absent from the observation is zero-filled IN PLACE (not
        # dropped, which would shift following joints into the wrong slots) and
        # surfaced loudly. Each key maps to one joint/motor position.
        state_values = self._collect_state_values(observation_dict, order)

        if state_values:
            # Auto-adapt state dimension to match what the model expects.
            # Robots may expose more joints than the policy was trained on
            # (e.g. aloha has 16 joints but ACT expects 14). Truncate excess
            # or zero-pad if fewer, rather than raising an error.
            state_feature = self._input_features.get("observation.state")
            if state_feature:
                expected_dim = state_feature.shape[0] if hasattr(state_feature, "shape") else len(state_values)
                if len(state_values) > expected_dim:
                    logger.warning(
                        "State dim %d > model expects %d - truncating to first %d values. "
                        "Check that robot_state_keys matches your robot's actual joint count.",
                        len(state_values),
                        expected_dim,
                        expected_dim,
                    )
                    state_values = state_values[:expected_dim]
                elif len(state_values) < expected_dim:
                    logger.warning(
                        "State dim %d < model expects %d - zero-padding with %d zeros. "
                        "Check that robot_state_keys matches your robot's actual joint count.",
                        len(state_values),
                        expected_dim,
                        expected_dim - len(state_values),
                    )
                    state_values.extend([0.0] * (expected_dim - len(state_values)))
            batch["observation.state"] = torch.tensor(state_values, dtype=torch.float32).unsqueeze(0).to(self._device)

        # Map camera images to the model's declared image input features.
        # Non-state ndarray values with ndim >= 2 are treated as images. Their
        # routing to policy image keys respects config.image_keys / the explicit
        # camera_key_map rather than blind positional assignment, so a "side"
        # camera is never silently fed into the slot a model reserves for a
        # "wrist" view (see _resolve_camera_targets).
        cam_items = [
            (key, value)
            for key, value in observation_dict.items()
            if key not in self.robot_state_keys and isinstance(value, np.ndarray) and value.ndim >= 2
        ]
        if cam_items:
            targets = self._resolve_camera_targets([key for key, _ in cam_items])
            for key, value in cam_items:
                feat_name = targets.get(key)
                if feat_name is None:
                    # Robot supplied more cameras than the policy consumes; the
                    # extras are intentionally dropped (resolve already errored
                    # if the policy was instead under-supplied).
                    continue
                image_tensor = torch.from_numpy(value.copy()).float()
                # HWC → CHW: convert from camera output format to model input format
                if image_tensor.dim() == 3 and image_tensor.shape[-1] in (1, 3, 4):
                    image_tensor = image_tensor.permute(2, 0, 1)
                # uint8 [0, 255] → float32 [0, 1]
                if value.dtype == np.uint8:
                    image_tensor = image_tensor / 255.0
                batch[feat_name] = image_tensor.unsqueeze(0).to(self._device)

        return batch

    # Action conversion

    def _requires_action_chunk(self) -> bool:
        """Whether the loaded policy must be driven via ``predict_action_chunk``.

        Some LeRobot policies cannot serve single actions through
        ``select_action``. MolmoAct2 is the canonical case: its
        ``select_action`` raises ``AssertionError`` whenever the checkpoint's
        ``rtc_config`` is enabled (it only supports RTC via
        ``predict_action_chunk``), and even with RTC off it returns just one
        action per call. Routing such policies through ``predict_action_chunk``
        avoids the crash and lets ``actions_per_step`` slice the full chunk.

        Detection is by policy name -- LeRobot sets
        ``PreTrainedPolicy.name = "molmoact2"`` -- with a class-name fallback for
        stubbed/mocked policies that do not set ``name``.

        Returns:
            True if the policy must use ``predict_action_chunk`` instead of
            ``select_action``; False otherwise (including when no policy is
            loaded).
        """
        policy = self._policy
        if policy is None:
            return False
        name = getattr(policy, "name", None)
        if isinstance(name, str) and name.lower() == "molmoact2":
            return True
        return type(policy).__name__ == "MolmoAct2Policy"

    def _tensor_to_action_dicts(self, action_tensor: torch.Tensor) -> list[dict[str, Any]]:
        """Convert action tensor to list of robot action dicts.

        Maps tensor values to robot_state_keys by index. Handles:
        - 1D tensor: single action step (shape [action_dim])
        - 2D tensor: action sequence (shape [horizon, action_dim])
        - 3D tensor: batched sequence (shape [batch, horizon, action_dim])

        Args:
            action_tensor: Raw action tensor from policy.select_action().

        Returns:
            List of action dicts, length capped by execution_horizon
            (== actions_per_step except under RTC, where it is the RTC
            execution horizon - the consumer re-queries at that interval).

        Raises:
            RuntimeError: If robot_state_keys is empty.
        """
        action_array = action_tensor.cpu().numpy()

        # Normalize tensor shape to a list of 1D action arrays. Cap the chunk
        # at execution_horizon (the re-query interval the consumer drives, via
        # resolve_chunk_length) rather than the raw trained chunk length: under
        # RTC these differ (execution_horizon << actions_per_step) and emitting
        # more than the consumer will execute is wasted work past the seam.
        _cap = self.execution_horizon
        if action_array.ndim == 1:
            actions_list = [action_array]
        elif action_array.ndim == 2:
            actions_list = [action_array[i] for i in range(min(len(action_array), _cap))]
        elif action_array.ndim == 3:
            # Batched: take first batch element, then slice horizon
            actions_list = [action_array[0, i] for i in range(min(action_array.shape[1], _cap))]
        else:
            actions_list = [action_array.flatten()]

        if not self.robot_state_keys:
            raise RuntimeError(
                "Cannot convert action tensor to dicts: robot_state_keys is empty. "
                "Call set_robot_state_keys() before inference."
            )

        # Diagnostics (issue: MolmoAct2 "runs but does not move in MuJoCo").
        # Surface two silent failure modes instead of swallowing them:
        #   1. action-dim mismatch -> unmatched actuators get zero-filled below
        #      (the for-loop's `else 0.0`), freezing those joints.
        #   2. a persistent near-zero action stream -> the robot never moves even
        #      though the policy "runs" every step.
        emb_name = self._embodiment.name if self._embodiment is not None else ""
        n_values = len(actions_list[0]) if actions_list else 0
        if not self._action_dim_warned:
            dim_msg = diagnose_action_dim(n_values, len(self.robot_state_keys), name=emb_name)
            if dim_msg:
                logger.warning("lerobot_local: %s", dim_msg)
                self._action_dim_warned = True
        max_abs = float(np.abs(action_array).max()) if action_array.size else 0.0
        zero_msg = self._zero_action_monitor.update(max_abs)
        if zero_msg:
            logger.warning("lerobot_local: %s", zero_msg)

        # Convert the model's action units to sim units when the embodiment
        # declares them. SO-arm checkpoints (so100/so101, MolmoAct2) emit joint
        # targets in the LeRobot driver convention (arm DEGREES + gripper
        # RANGE_0_100), but the MuJoCo sim joints are RADIANS -- feeding the raw
        # degree values straight in saturates the radian joint limits and the
        # arm freezes. EmbodimentMap.model_action_to_sim is a no-op when
        # action_units == "native" (the default / real-hardware path).
        emb = self._embodiment
        convert = emb is not None and getattr(emb, "action_units", "native") != "native"

        result = []
        for action_values in actions_list:
            vals = [
                float(action_values[i]) if i < len(action_values) else 0.0 for i in range(len(self.robot_state_keys))
            ]
            # `convert` already implies `emb is not None`; the explicit guard lets
            # the type checker narrow `emb` from `EmbodimentMap | None` at the call.
            if convert and emb is not None:
                vals = emb.model_action_to_sim(vals)
            action_dict = {key: vals[index] for index, key in enumerate(self.robot_state_keys)}
            result.append(action_dict)

        return result
