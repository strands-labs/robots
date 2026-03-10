"""Robometer VLM reward model integration for strands-robots.

Bridges the strands-robots environment/training ecosystem with the Robometer
VLM-based reward model (https://github.com/robometer/robometer).

Architecture
------------
* **Lazy imports** – heavy dependencies (``torch``, ``robometer``,
  ``transformers``) are imported only at the point of use.  Module-level
  ``HAS_*`` flags allow callers to gate functionality cheaply.
* **Gymnasium Wrapper** – ``StrandsRobometerRewardWrapper`` uses
  composition (not inheritance) so the module can be imported even
  without gymnasium installed.
* **Reward function factory** – ``robometer_reward_fn`` returns a
  ``Callable[[dict, np.ndarray], float]`` compatible with the
  ``reward_fn`` parameter in ``envs.StrandsSimEnv`` and
  ``rl_trainer.SB3Trainer``.
* **Trainer plugin** – ``RobometerTrainer`` implements the project's
  ``training.Trainer`` ABC and is registered as ``"robometer"`` via
  ``training.create_trainer()``.
* **Data loader** – ``strands_robots_loader`` converts
  ``RecordSession`` / LeRobot v3 datasets into Robometer trajectory
  dicts for fine-tuning.
* **Benchmarking** – ``RobometerBenchmark`` evaluates multiple policies
  using Robometer as the reward oracle and reports Spearman/Kendall
  rank-correlation against ground-truth rankings.

Example
-------
>>> from strands_robots.robometer import StrandsRobometerRewardWrapper
>>> env = StrandsSimEnv(robot_name="so100", render_mode="rgb_array")
>>> env = StrandsRobometerRewardWrapper(
...     env,
...     model_path="aliangdw/Robometer-4B",
...     task_instruction="pick up the red cube",
...     device="cuda",
... )
>>> obs, info = env.reset()
>>> obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
"""

from __future__ import annotations

import collections
import logging
import time
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import numpy as np

# ---------------------------------------------------------------------------
# Lazy-import flags
# ---------------------------------------------------------------------------

HAS_GYMNASIUM: bool = False
HAS_TORCH: bool = False
HAS_ROBOMETER: bool = False
HAS_SCIPY: bool = False
HAS_TRANSFORMERS: bool = False

try:
    import gymnasium as gym  # noqa: F401

    HAS_GYMNASIUM = True
except ImportError:
    gym = None  # type: ignore[assignment]

try:
    import torch  # noqa: F401

    HAS_TORCH = True
except ImportError:
    torch = None  # type: ignore[assignment]

try:
    import robometer  # noqa: F401

    HAS_ROBOMETER = True
except ImportError:
    robometer = None  # type: ignore[assignment]

try:
    import scipy.stats  # noqa: F401

    HAS_SCIPY = True
except ImportError:
    scipy = None  # type: ignore[assignment]

try:
    import transformers  # noqa: F401

    HAS_TRANSFORMERS = True
except ImportError:
    transformers = None  # type: ignore[assignment]

if TYPE_CHECKING:
    import pathlib

    import gymnasium as gym
    import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _require(flag: bool, name: str) -> None:
    """Raise ``ImportError`` if *flag* is ``False``."""
    if not flag:
        raise ImportError(f"{name} is required but not installed.  " f"Install it with:  pip install {name}")


# ---------------------------------------------------------------------------
# Lazy model loading cache
# ---------------------------------------------------------------------------

_MODEL_CACHE: Dict[str, Any] = {}


def _load_robometer_model(
    model_path: str,
    device: str = "cuda",
) -> Tuple[Any, Any, Any, Any]:
    """Load a Robometer model, caching by ``(model_path, device)``.

    Returns
    -------
    config, tokenizer, processor, model
        The four objects returned by ``robometer.utils.save.load_model_from_hf``.
    """
    _require(HAS_ROBOMETER, "robometer")

    cache_key = f"{model_path}@{device}"
    if cache_key in _MODEL_CACHE:
        logger.debug("Robometer model cache hit for %s", cache_key)
        return _MODEL_CACHE[cache_key]

    from robometer.utils.save import load_model_from_hf  # lazy

    logger.info("Loading Robometer model from %r on device=%r …", model_path, device)
    t0 = time.monotonic()
    config, tokenizer, processor, model = load_model_from_hf(model_path, device=device)
    dt = time.monotonic() - t0
    logger.info("Robometer model loaded in %.1f s", dt)

    _MODEL_CACHE[cache_key] = (config, tokenizer, processor, model)
    return config, tokenizer, processor, model


def _setup_batch_collator(processor: Any, tokenizer: Any, config: Any, *, is_eval: bool = True) -> Any:
    """Create batch collator via ``robometer.utils.setup_utils``."""
    from robometer.utils.setup_utils import setup_batch_collator  # lazy

    return setup_batch_collator(processor, tokenizer, config, is_eval=is_eval)


def _infer_progress(
    config: Any,
    model: Any,
    tokenizer: Any,
    batch_collator: Any,
    device: str,
    trajectory_dict: Dict[str, Any],
    max_frames: int,
) -> Tuple[float, Optional[float]]:
    """Run a single Robometer forward pass and extract progress + success.

    Parameters
    ----------
    trajectory_dict:
        ``{"frames": list[np.ndarray], "task_instruction": str}``

    Returns
    -------
    progress : float
        Estimated task completion ∈ [0, 1].
    success_prob : float | None
        Success probability if available, else ``None``.
    """
    from robometer.evals.eval_server import process_batch_helper  # lazy
    from robometer.evals.eval_utils import (  # lazy
        extract_rewards_from_output,
        extract_success_probs_from_output,
        raw_dict_to_sample,
    )

    # Build frames array
    frames = trajectory_dict["frames"]
    if isinstance(frames, list):
        frames_arr = np.stack(frames)
    else:
        frames_arr = np.asarray(frames)
    task = trajectory_dict.get("task_instruction", "complete the task")

    raw_data = {"frames": frames_arr, "task": task}
    sample = raw_dict_to_sample(raw_data, max_frames=max_frames, sample_type="progress")

    outputs = process_batch_helper(
        model_type=config.model_type,
        model=model,
        tokenizer=tokenizer,
        batch_collator=batch_collator,
        device=device,
        batch_data=[sample],
    )

    rewards = extract_rewards_from_output(outputs)
    progress = float(rewards[0]) if rewards.size > 0 else 0.0

    success_prob: Optional[float] = None
    try:
        probs = extract_success_probs_from_output(outputs)
        success_prob = float(probs[0]) if probs.size > 0 else None
    except (ValueError, KeyError):
        pass  # success head not always present

    return progress, success_prob


# ═══════════════════════════════════════════════════════════════════════════
# 1. StrandsRobometerRewardWrapper
# ═══════════════════════════════════════════════════════════════════════════


class StrandsRobometerRewardWrapper:
    """Gymnasium wrapper that replaces/augments rewards with Robometer VLM
    progress estimates.

    Uses **composition** instead of inheriting ``gym.Wrapper`` so the
    class body parses even when gymnasium is not installed.

    Parameters
    ----------
    env:
        A strands-robots ``gymnasium.Env`` whose observations contain a
        ``"pixels"`` key (uint8, H×W×3).
    model_path:
        HuggingFace model identifier (e.g. ``"aliangdw/Robometer-4B"``).
    device:
        PyTorch device string.
    max_frames:
        Sliding-window size for the frame buffer sent to the VLM.
    use_relative_rewards:
        Emit delta-progress (change between steps) instead of absolute.
    use_success_detection:
        Enable sliding-window voting for episode termination.
    success_vote_window:
        Number of recent success probabilities for voting.
    success_threshold:
        Per-frame probability threshold for a positive vote.
    success_vote_fraction:
        Fraction of positive votes required to declare success.
    add_estimated_reward:
        If ``True``, Robometer reward is *added* to env reward.
        If ``False`` (default), env reward is *replaced*.
    task_instruction:
        Explicit task string.  Falls back to ``env.task``.
    pixels_key:
        Observation-dict key holding the camera frame.
    reward_scale:
        Scalar multiplier applied to the Robometer reward.
    """

    def __init__(
        self,
        env: "gym.Env",
        model_path: str,
        device: str = "cuda",
        max_frames: int = 16,
        use_relative_rewards: bool = True,
        use_success_detection: bool = False,
        success_vote_window: int = 5,
        success_threshold: float = 0.5,
        success_vote_fraction: float = 0.6,
        add_estimated_reward: bool = False,
        task_instruction: Optional[str] = None,
        pixels_key: str = "pixels",
        reward_scale: float = 1.0,
    ) -> None:
        _require(HAS_GYMNASIUM, "gymnasium")
        _require(HAS_TORCH, "torch")
        _require(HAS_ROBOMETER, "robometer")

        self.env = env

        # Proxy standard gymnasium.Wrapper attributes
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.metadata = getattr(env, "metadata", {})
        self.render_mode = getattr(env, "render_mode", None)
        self.spec = getattr(env, "spec", None)
        self.reward_range = getattr(env, "reward_range", (-float("inf"), float("inf")))

        # Validate observation space
        if hasattr(self.observation_space, "spaces"):
            if pixels_key not in self.observation_space.spaces:
                raise ValueError(
                    f"Observation space does not contain key {pixels_key!r}.  "
                    f"Available keys: {list(self.observation_space.spaces.keys())}"
                )
        else:
            logger.warning(
                "Cannot verify presence of %r in observation space " "(space is not a Dict). Proceeding anyway.",
                pixels_key,
            )

        self._model_path = model_path
        self._device = device
        self._max_frames = max_frames
        self._use_relative_rewards = use_relative_rewards
        self._use_success_detection = use_success_detection
        self._success_vote_window = success_vote_window
        self._success_threshold = success_threshold
        self._success_vote_fraction = success_vote_fraction
        self._add_estimated_reward = add_estimated_reward
        self._pixels_key = pixels_key
        self._reward_scale = reward_scale

        # Resolve task instruction
        self._task_instruction = task_instruction or self._resolve_task_instruction()

        # State
        self._frame_buffer: collections.deque = collections.deque(maxlen=self._max_frames)
        self._prev_progress: Optional[float] = None
        self._success_votes: collections.deque = collections.deque(maxlen=self._success_vote_window)

        # Model artefacts (loaded lazily)
        self._config: Any = None
        self._tokenizer: Any = None
        self._processor: Any = None
        self._model: Any = None
        self._batch_collator: Any = None
        self._model_loaded: bool = False

    # ── Internals ─────────────────────────────────────────────────────

    def _resolve_task_instruction(self) -> str:
        for attr_path in ("task", "unwrapped.task"):
            obj = self.env
            try:
                for part in attr_path.split("."):
                    obj = getattr(obj, part)
                if isinstance(obj, str) and obj:
                    return obj
            except AttributeError:
                continue
        return "complete the task"

    def _ensure_model_loaded(self) -> None:
        if self._model_loaded:
            return
        self._config, self._tokenizer, self._processor, self._model = _load_robometer_model(
            self._model_path, device=self._device
        )
        self._batch_collator = _setup_batch_collator(self._processor, self._tokenizer, self._config, is_eval=True)
        self._model_loaded = True

    def _infer_reward(self) -> Tuple[float, Optional[float]]:
        """Run Robometer inference on the current frame buffer."""
        self._ensure_model_loaded()

        trajectory_dict = {
            "frames": list(self._frame_buffer),
            "task_instruction": self._task_instruction,
        }

        progress, success_prob = _infer_progress(
            self._config,
            self._model,
            self._tokenizer,
            self._batch_collator,
            self._device,
            trajectory_dict,
            self._max_frames,
        )

        if self._use_relative_rewards:
            prev = self._prev_progress if self._prev_progress is not None else 0.0
            reward = progress - prev
        else:
            reward = progress
        self._prev_progress = progress

        return reward, success_prob

    def _vote_success(self, success_prob: float) -> bool:
        self._success_votes.append(success_prob)
        if len(self._success_votes) < self._success_vote_window:
            return False
        n_positive = sum(1 for p in self._success_votes if p >= self._success_threshold)
        return (n_positive / len(self._success_votes)) >= self._success_vote_fraction

    # ── gymnasium.Wrapper interface ───────────────────────────────────

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self._frame_buffer.clear()
        self._prev_progress = None
        self._success_votes.clear()

        frame = obs.get(self._pixels_key) if isinstance(obs, dict) else None
        if frame is not None:
            self._frame_buffer.append(np.asarray(frame, dtype=np.uint8))

        info["robometer_progress"] = 0.0
        info["robometer_success"] = False
        return obs, info

    def step(self, action: Any) -> Tuple[Dict[str, Any], float, bool, bool, Dict[str, Any]]:
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        frame = obs.get(self._pixels_key) if isinstance(obs, dict) else None
        if frame is not None:
            self._frame_buffer.append(np.asarray(frame, dtype=np.uint8))

        try:
            robometer_reward, success_prob = self._infer_reward()
        except Exception:
            logger.exception("Robometer inference failed; falling back to env reward")
            robometer_reward = 0.0
            success_prob = None

        scaled_reward = robometer_reward * self._reward_scale
        reward = (env_reward + scaled_reward) if self._add_estimated_reward else scaled_reward

        is_success = False
        if self._use_success_detection and success_prob is not None:
            is_success = self._vote_success(success_prob)
            if is_success:
                terminated = True

        info["robometer_reward_raw"] = robometer_reward
        info["robometer_reward_scaled"] = scaled_reward
        info["robometer_progress"] = self._prev_progress if self._prev_progress is not None else 0.0
        info["robometer_success"] = is_success
        if success_prob is not None:
            info["robometer_success_prob"] = success_prob
        info["env_reward"] = env_reward

        return obs, reward, terminated, truncated, info

    def render(self) -> Any:
        return self.env.render()

    def close(self) -> None:
        return self.env.close()

    @property
    def unwrapped(self) -> "gym.Env":
        return self.env.unwrapped

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def __str__(self) -> str:
        return f"<StrandsRobometerRewardWrapper({self.env})>"

    def __repr__(self) -> str:
        return (
            f"StrandsRobometerRewardWrapper("
            f"env={self.env!r}, "
            f"model_path={self._model_path!r}, "
            f"device={self._device!r}, "
            f"max_frames={self._max_frames})"
        )


# ═══════════════════════════════════════════════════════════════════════════
# 2. robometer_reward_fn — closure factory for rl_trainer compatibility
# ═══════════════════════════════════════════════════════════════════════════


def robometer_reward_fn(
    model_path: str,
    device: str = "cuda",
    max_frames: int = 16,
    task_instruction: str = "complete the task",
    use_relative_rewards: bool = True,
    reward_scale: float = 1.0,
    pixels_key: str = "pixels",
) -> Callable[[Dict[str, Any], np.ndarray], float]:
    """Create a reward function compatible with ``envs.StrandsSimEnv(reward_fn=...)``
    and ``rl_trainer``.

    Returns a callable ``(obs, action) -> float`` with a ``.reset()`` method
    to flush state between episodes.

    Parameters
    ----------
    model_path:
        HuggingFace model identifier for Robometer.
    device:
        PyTorch device string.
    max_frames:
        Sliding-window size.
    task_instruction:
        Natural-language task instruction for the VLM.
    use_relative_rewards:
        Emit delta-progress.
    reward_scale:
        Scalar multiplier.
    pixels_key:
        Observation-dict key for the camera frame.

    Returns
    -------
    Callable[[dict, np.ndarray], float]
        Stateful reward function with ``.reset()`` method.
    """
    _require(HAS_ROBOMETER, "robometer")
    _require(HAS_TORCH, "torch")

    frame_buffer: collections.deque = collections.deque(maxlen=max_frames)
    prev_progress: List[Optional[float]] = [None]

    # Lazy model state
    loaded: List[bool] = [False]
    artefacts: Dict[str, Any] = {}

    def _ensure_loaded() -> None:
        if loaded[0]:
            return
        config, tokenizer, processor, model = _load_robometer_model(model_path, device=device)
        artefacts["config"] = config
        artefacts["tokenizer"] = tokenizer
        artefacts["processor"] = processor
        artefacts["model"] = model
        artefacts["collator"] = _setup_batch_collator(processor, tokenizer, config, is_eval=True)
        loaded[0] = True

    def _reward(obs: Dict[str, Any], action: np.ndarray) -> float:
        _ensure_loaded()

        frame = obs.get(pixels_key)
        if frame is None:
            logger.warning("robometer_reward_fn: obs missing %r key", pixels_key)
            return 0.0
        frame_buffer.append(np.asarray(frame, dtype=np.uint8))

        try:
            traj = {"frames": list(frame_buffer), "task_instruction": task_instruction}
            progress, _ = _infer_progress(
                artefacts["config"],
                artefacts["model"],
                artefacts["tokenizer"],
                artefacts["collator"],
                device,
                traj,
                max_frames,
            )
        except Exception:
            logger.exception("robometer_reward_fn: inference failed; returning 0.0")
            return 0.0

        if use_relative_rewards:
            prev = prev_progress[0] if prev_progress[0] is not None else 0.0
            reward = progress - prev
        else:
            reward = progress
        prev_progress[0] = progress

        return reward * reward_scale

    def _reset() -> None:
        frame_buffer.clear()
        prev_progress[0] = None

    _reward.reset = _reset  # type: ignore[attr-defined]
    return _reward  # type: ignore[return-value]


# ═══════════════════════════════════════════════════════════════════════════
# 3. RobometerTrainer — conforms to training.Trainer ABC
# ═══════════════════════════════════════════════════════════════════════════


class RobometerTrainer:
    """Trainer wrapping Robometer's ``RBMHeadsTrainer``.

    Conforms to the ``training.Trainer`` ABC interface
    (``train()``, ``evaluate()``, ``provider_name`` property).

    Supports HuggingFace model checkpoints, LoRA fine-tuning via
    ``peft``, and multi-GPU training via FSDP.

    Parameters
    ----------
    model_path:
        HuggingFace model path or local checkpoint.
    output_dir:
        Training artefact output directory.
    device:
        Primary PyTorch device.
    learning_rate:
        Peak learning rate.
    num_epochs:
        Number of training epochs.
    batch_size:
        Per-device training batch size.
    max_frames:
        Maximum trajectory length per sample.
    use_lora:
        Enable LoRA fine-tuning.
    lora_rank, lora_alpha, lora_dropout:
        LoRA hyperparameters.
    use_fsdp:
        Enable FSDP multi-GPU.
    fsdp_config:
        Optional FSDP configuration dict.
    """

    def __init__(
        self,
        model_path: str,
        output_dir: str = "./robometer_output",
        device: str = "cuda",
        learning_rate: float = 2e-5,
        num_epochs: int = 3,
        batch_size: int = 4,
        max_frames: int = 16,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        use_fsdp: bool = False,
        fsdp_config: Optional[Dict[str, Any]] = None,
        **extra_trainer_kwargs: Any,
    ) -> None:
        _require(HAS_ROBOMETER, "robometer")
        _require(HAS_TORCH, "torch")
        _require(HAS_TRANSFORMERS, "transformers")

        self._model_path = model_path
        self._output_dir = output_dir
        self._device = device
        self._learning_rate = learning_rate
        self._num_epochs = num_epochs
        self._batch_size = batch_size
        self._max_frames = max_frames
        self._use_lora = use_lora
        self._lora_rank = lora_rank
        self._lora_alpha = lora_alpha
        self._lora_dropout = lora_dropout
        self._use_fsdp = use_fsdp
        self._fsdp_config = fsdp_config or {}
        self._extra_trainer_kwargs = extra_trainer_kwargs

        self._trainer: Any = None
        self._model_artefacts: Optional[Tuple[Any, ...]] = None

    @property
    def provider_name(self) -> str:
        """Trainer provider name for the registry."""
        return "robometer"

    def _build_trainer(self, train_dataset: Any, eval_dataset: Any = None) -> Any:
        """Construct the underlying ``RBMHeadsTrainer``."""
        from robometer.trainers.rbm_heads_trainer import RBMHeadsTrainer  # lazy
        from robometer.utils.save import load_model_from_hf  # lazy
        from robometer.utils.setup_utils import setup_batch_collator  # lazy

        config, tokenizer, processor, model = load_model_from_hf(self._model_path, device=self._device)
        self._model_artefacts = (config, tokenizer, processor, model)

        # Optional LoRA
        if self._use_lora:
            try:
                from peft import LoraConfig, get_peft_model  # lazy
            except ImportError as exc:
                raise ImportError(
                    "LoRA fine-tuning requires the `peft` package.  " "Install with: pip install peft"
                ) from exc

            lora_config = LoraConfig(
                r=self._lora_rank,
                lora_alpha=self._lora_alpha,
                lora_dropout=self._lora_dropout,
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            logger.info(
                "LoRA enabled – trainable params: %s",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )

        from transformers import TrainingArguments  # lazy

        training_args_kwargs: Dict[str, Any] = {
            "output_dir": self._output_dir,
            "learning_rate": self._learning_rate,
            "num_train_epochs": self._num_epochs,
            "per_device_train_batch_size": self._batch_size,
            "per_device_eval_batch_size": self._batch_size,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "remove_unused_columns": False,
            "bf16": HAS_TORCH and torch.cuda.is_available(),
        }
        if self._use_fsdp:
            training_args_kwargs["fsdp"] = "full_shard"
            training_args_kwargs["fsdp_config"] = self._fsdp_config

        training_args = TrainingArguments(**training_args_kwargs)
        collator = setup_batch_collator(processor, tokenizer, config, is_eval=False)

        self._trainer = RBMHeadsTrainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=collator,
            tokenizer=tokenizer,
            **self._extra_trainer_kwargs,
        )
        return self._trainer

    def train(self, train_dataset: Any = None, eval_dataset: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Run Robometer reward-head training.

        Parameters
        ----------
        train_dataset:
            ``torch.utils.data.Dataset`` of Robometer-formatted samples.
        eval_dataset:
            Optional evaluation dataset.

        Returns
        -------
        dict
            Training metrics.
        """
        trainer = self._build_trainer(train_dataset, eval_dataset)
        logger.info("Starting Robometer training for %d epochs …", self._num_epochs)
        result = trainer.train(**kwargs)
        logger.info("Training complete.  Metrics: %s", result.metrics)
        return result.metrics

    def evaluate(self, eval_dataset: Any = None, **kwargs: Any) -> Dict[str, Any]:
        """Evaluate the trained model.

        Returns
        -------
        dict
            Evaluation metrics.
        """
        if self._trainer is None:
            raise RuntimeError("No trainer initialised.  Call train() first.")
        metrics = self._trainer.evaluate(eval_dataset=eval_dataset, **kwargs)
        logger.info("Evaluation metrics: %s", metrics)
        return metrics

    def save_model(self, path: Optional[str] = None) -> None:
        """Persist the trained model to disk."""
        if self._trainer is None:
            raise RuntimeError("No trainer initialised.  Call train() first.")
        save_path = path or self._output_dir
        self._trainer.save_model(save_path)
        logger.info("Model saved to %s", save_path)


# ═══════════════════════════════════════════════════════════════════════════
# 4. strands_robots_loader — data format bridge
# ═══════════════════════════════════════════════════════════════════════════


def strands_robots_loader(
    data_source: Any,
    max_frames: int = 16,
    task_instruction: Optional[str] = None,
    pixels_key: str = "pixels",
) -> List[Dict[str, Any]]:
    """Convert strands-robots data to Robometer trajectory dicts.

    Accepts:

    * A ``list`` of episode dicts already in memory.
    * A ``RecordSession`` object (from ``strands_robots.record``).
    * A path (``str`` / ``pathlib.Path``) to a LeRobot v3 dataset.

    Parameters
    ----------
    data_source:
        One of the formats above.
    max_frames:
        Maximum frames per trajectory (uniform sub-sampling).
    task_instruction:
        Explicit task instruction; otherwise read from metadata.
    pixels_key:
        Key within each observation holding the camera frame.

    Returns
    -------
    list[dict]
        Each element::

            {
                "frames": list[np.ndarray],        # uint8, H×W×3
                "task_instruction": str,
                "actions": list[np.ndarray] | None,
                "states": list[np.ndarray] | None,
            }
    """
    import pathlib

    trajectories: List[Dict[str, Any]] = []

    if isinstance(data_source, list):
        episodes = data_source
    elif hasattr(data_source, "episodes"):
        episodes = []
        for ep in data_source.episodes:
            episodes.append(
                {
                    "observations": getattr(ep, "observations", []),
                    "actions": getattr(ep, "actions", []),
                    "task": getattr(ep, "task", None) or getattr(data_source, "task", None),
                }
            )
    elif isinstance(data_source, (str, pathlib.Path)):
        episodes = _load_lerobot_v3_episodes(pathlib.Path(data_source), pixels_key=pixels_key)
    else:
        raise TypeError(
            f"Unsupported data_source type: {type(data_source).__name__}.  "
            "Expected list, RecordSession, or path-like."
        )

    for ep in episodes:
        obs_list = ep.get("observations", [])
        actions_list = ep.get("actions", None)
        ep_task = task_instruction or ep.get("task") or "complete the task"

        frames: List[np.ndarray] = []
        states: List[np.ndarray] = []

        for obs in obs_list:
            if isinstance(obs, dict):
                pixel = obs.get(pixels_key)
                state = obs.get("state")
            elif isinstance(obs, np.ndarray):
                pixel = obs
                state = None
            else:
                continue

            if pixel is not None:
                frames.append(np.asarray(pixel, dtype=np.uint8))
            if state is not None:
                states.append(np.asarray(state, dtype=np.float32))

        # Uniform sub-sampling
        if len(frames) > max_frames:
            indices = np.linspace(0, len(frames) - 1, max_frames, dtype=int)
            frames = [frames[i] for i in indices]
            if states:
                states = [states[min(i, len(states) - 1)] for i in indices]
            if actions_list is not None:
                actions_list = [actions_list[min(i, len(actions_list) - 1)] for i in indices]

        trajectories.append(
            {
                "frames": frames,
                "task_instruction": ep_task,
                "actions": actions_list,
                "states": states if states else None,
            }
        )

    logger.info("Loaded %d trajectories from %s", len(trajectories), type(data_source).__name__)
    return trajectories


def _load_lerobot_v3_episodes(
    dataset_path: "pathlib.Path",
    pixels_key: str = "pixels",
) -> List[Dict[str, Any]]:
    """Load episodes from a LeRobot v3 on-disk dataset."""
    import json

    meta_path = dataset_path / "meta" / "info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"LeRobot v3 metadata not found at {meta_path}")

    with open(meta_path, "r") as fh:
        meta = json.load(fh)

    task_instruction = meta.get("task", {}).get("instruction", "complete the task")
    num_episodes = meta.get("num_episodes", 0)
    episodes: List[Dict[str, Any]] = []

    for ep_idx in range(num_episodes):
        parquet_path = dataset_path / "data" / f"episode_{ep_idx:06d}.parquet"
        if parquet_path.exists():
            try:
                import pyarrow.parquet as pq

                table = pq.read_table(str(parquet_path))
                df = table.to_pandas()
            except ImportError:
                logger.warning("pyarrow not installed; cannot read %s", parquet_path)
                continue

            observations: List[Dict[str, Any]] = []
            actions: List[np.ndarray] = []

            for _, row in df.iterrows():
                obs: Dict[str, Any] = {}
                if pixels_key in row and row[pixels_key] is not None:
                    obs[pixels_key] = np.asarray(row[pixels_key], dtype=np.uint8)
                if "state" in row and row["state"] is not None:
                    obs["state"] = np.asarray(row["state"], dtype=np.float32)
                observations.append(obs)
                if "action" in row and row["action"] is not None:
                    actions.append(np.asarray(row["action"], dtype=np.float32))

            episodes.append(
                {
                    "observations": observations,
                    "actions": actions if actions else None,
                    "task": task_instruction,
                }
            )

    return episodes


# ═══════════════════════════════════════════════════════════════════════════
# 5. RobometerBenchmark — multi-policy evaluation with rank correlation
# ═══════════════════════════════════════════════════════════════════════════


class RobometerBenchmark:
    """Evaluate multiple policies using Robometer as the reward oracle.

    Rolls out each policy for N episodes, scores via VLM, ranks by
    mean progress, and computes Spearman ρ / Kendall τ against
    ground-truth rankings.

    Parameters
    ----------
    model_path:
        HuggingFace model path.
    device:
        PyTorch device.
    max_frames:
        Maximum trajectory length per episode.
    task_instruction:
        Task instruction for the VLM.
    num_episodes:
        Evaluation episodes per policy.
    pixels_key:
        Observation-dict key for camera frames.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        max_frames: int = 16,
        task_instruction: str = "complete the task",
        num_episodes: int = 10,
        pixels_key: str = "pixels",
    ) -> None:
        _require(HAS_ROBOMETER, "robometer")
        _require(HAS_TORCH, "torch")

        self._model_path = model_path
        self._device = device
        self._max_frames = max_frames
        self._task_instruction = task_instruction
        self._num_episodes = num_episodes
        self._pixels_key = pixels_key

    def _rollout_policy(
        self,
        policy: Callable[[Dict[str, Any]], np.ndarray],
        env: "gym.Env",
    ) -> List[Dict[str, Any]]:
        trajectories: List[Dict[str, Any]] = []

        for _ in range(self._num_episodes):
            obs, _ = env.reset()
            frames: List[np.ndarray] = []
            done = False

            frame = obs.get(self._pixels_key) if isinstance(obs, dict) else None
            if frame is not None:
                frames.append(np.asarray(frame, dtype=np.uint8))

            while not done:
                action = policy(obs)
                obs, _, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                frame = obs.get(self._pixels_key) if isinstance(obs, dict) else None
                if frame is not None:
                    frames.append(np.asarray(frame, dtype=np.uint8))

            if len(frames) > self._max_frames:
                indices = np.linspace(0, len(frames) - 1, self._max_frames, dtype=int)
                frames = [frames[i] for i in indices]

            trajectories.append({"frames": frames, "task_instruction": self._task_instruction})
        return trajectories

    def _score_trajectories(self, trajectories: List[Dict[str, Any]]) -> Tuple[List[float], List[float]]:
        config, tokenizer, processor, model = _load_robometer_model(self._model_path, device=self._device)
        collator = _setup_batch_collator(processor, tokenizer, config, is_eval=True)

        progress_scores: List[float] = []
        success_probs: List[float] = []

        for traj in trajectories:
            progress, success_prob = _infer_progress(
                config, model, tokenizer, collator, self._device, traj, self._max_frames
            )
            progress_scores.append(progress)
            success_probs.append(success_prob if success_prob is not None else 0.0)

        return progress_scores, success_probs

    def evaluate(
        self,
        policies: Dict[str, Callable[[Dict[str, Any]], np.ndarray]],
        env_factory: Callable[[], "gym.Env"],
        ground_truth_ranking: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Run the benchmark.

        Parameters
        ----------
        policies:
            Mapping policy name → callable policy function.
        env_factory:
            Zero-arg callable returning a fresh ``gym.Env``.
        ground_truth_ranking:
            Policy names ordered best → worst.

        Returns
        -------
        dict
            ``"policy_scores"``, ``"ranking"``, ``"spearman_rho"``,
            ``"kendall_tau"``.
        """
        _require(HAS_GYMNASIUM, "gymnasium")

        policy_scores: Dict[str, Dict[str, float]] = {}

        for name, policy in policies.items():
            logger.info("Evaluating policy %r (%d episodes) …", name, self._num_episodes)
            env = env_factory()
            try:
                trajectories = self._rollout_policy(policy, env)
            finally:
                env.close()

            progress, success = self._score_trajectories(trajectories)
            policy_scores[name] = {
                "mean_progress": float(np.mean(progress)),
                "mean_success": float(np.mean(success)),
                "progress_std": float(np.std(progress)),
                "success_std": float(np.std(success)),
                "num_episodes": self._num_episodes,
            }
            logger.info(
                "  %s – progress=%.4f  success=%.4f",
                name,
                policy_scores[name]["mean_progress"],
                policy_scores[name]["mean_success"],
            )

        ranking = sorted(policy_scores, key=lambda n: policy_scores[n]["mean_progress"], reverse=True)

        result: Dict[str, Any] = {
            "policy_scores": policy_scores,
            "ranking": ranking,
            "spearman_rho": None,
            "spearman_pvalue": None,
            "kendall_tau": None,
            "kendall_pvalue": None,
        }

        if ground_truth_ranking is not None and len(ground_truth_ranking) >= 2:
            _require(HAS_SCIPY, "scipy")
            import scipy.stats  # lazy

            gt_ranks = {n: r for r, n in enumerate(ground_truth_ranking)}
            pred_ranks = {n: r for r, n in enumerate(ranking)}
            common = [n for n in ground_truth_ranking if n in pred_ranks]

            if len(common) >= 2:
                gt_vec = [gt_ranks[n] for n in common]
                pred_vec = [pred_ranks[n] for n in common]

                sp = scipy.stats.spearmanr(gt_vec, pred_vec)
                kt = scipy.stats.kendalltau(gt_vec, pred_vec)

                result["spearman_rho"] = float(sp.statistic)
                result["spearman_pvalue"] = float(sp.pvalue)
                result["kendall_tau"] = float(kt.statistic)
                result["kendall_pvalue"] = float(kt.pvalue)

        return result


# ═══════════════════════════════════════════════════════════════════════════
# __all__
# ═══════════════════════════════════════════════════════════════════════════

__all__ = [
    # Flags
    "HAS_GYMNASIUM",
    "HAS_TORCH",
    "HAS_ROBOMETER",
    "HAS_SCIPY",
    "HAS_TRANSFORMERS",
    # Wrapper
    "StrandsRobometerRewardWrapper",
    # Reward function factory
    "robometer_reward_fn",
    # Trainer
    "RobometerTrainer",
    # Data loading
    "strands_robots_loader",
    # Benchmarking
    "RobometerBenchmark",
]
