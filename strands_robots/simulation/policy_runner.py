"""Backend-agnostic policy execution against any ``SimEngine``.

Runs the canonical obs → act → step loop using only the public ``SimEngine``
interface. Zero knowledge of the underlying physics engine - MuJoCo, Isaac,
Newton and any future backend get ``run_policy`` / ``replay`` / ``evaluate``
for free by implementing the ``SimEngine`` primitives.

Three entry points:

* :meth:`PolicyRunner.run` - blocking policy execution with optional video.
* :meth:`PolicyRunner.replay` - replay a recorded LeRobotDataset episode.
* :meth:`PolicyRunner.evaluate` - multi-episode evaluation with success metrics.

All three call only these public ``SimEngine`` methods:

* ``get_observation(robot_name)``
* ``send_action(action, robot_name, n_substeps)``
* ``step(n_steps)``
* ``reset()``
* ``render(camera_name, width, height)``

And two public helpers for robot discovery:

* ``list_robots()`` - ordered robot names in the world
* ``robot_joint_names(robot_name)`` - ordered joint names for a robot

Thread safety: ``PolicyRunner`` itself is stateless per invocation. The
underlying ``SimEngine`` is responsible for thread-safety inside its own
methods (e.g. MuJoCo acquires a lock inside ``send_action`` / ``step``).
"""

from __future__ import annotations

import contextlib
import difflib
import logging
import math
import numbers
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.dataset_recorder import RecordingFrameError
from strands_robots.policies.base import resolve_chunk_length
from strands_robots.utils import (
    non_negative_whole_number_error,
    positive_count_error,
    positive_finite_number_error,
    positive_whole_number_error,
    process_rss_mb,
    require_optional,
)

if TYPE_CHECKING:
    from strands_robots.mesh.pacing import Ticker
    from strands_robots.policies.base import Policy
    from strands_robots.simulation.base import SimEngine
    from strands_robots.simulation.benchmark import BenchmarkProtocol

from strands_robots.simulation.models import TrajectoryStep
from strands_robots.simulation.safe_output import validate_output_path, video_sandbox_args

logger = logging.getLogger(__name__)


def set_eval_seed(seed: int) -> None:
    """Seed Python / NumPy / torch RNGs for reproducible eval rollouts.

    Mirrors NVIDIA's ``set_seed`` from
    ``Isaac-GR00T/scripts/deployment/standalone_inference_script.py:81``,
    minus two global side effects that would persist after the eval and
    affect unrelated callers in the same process:

    * ``os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"`` - leaks into
      every subsequent torch op in the process.
    * ``torch.use_deterministic_algorithms(True, warn_only=True)`` -
      can break callers downstream that rely on non-deterministic CUDA
      kernels (e.g. some loss functions).

    Users who want NVIDIA's exact strict-determinism mode can set those
    themselves before calling :meth:`evaluate_benchmark`. The defaults
    here cover the common case: reproducible rollouts of the SAME
    policy + seed combination, without forcing the rest of the process
    into deterministic-only mode.

    Seeds applied:

    * Python ``random.seed``.
    * NumPy ``np.random.seed`` (the legacy global RNG; matches what
      most policies use under the hood). This is the narrowest of the
      three: it refuses a seed above
      :data:`~strands_robots.simulation.base.MAX_EVAL_SEED`, so that is
      the ceiling every rollout surface accepts.
    * PyTorch CPU (``torch.manual_seed``) - if torch is importable.
    * PyTorch CUDA all devices (``torch.cuda.manual_seed_all``) - if
      torch is importable AND CUDA is available.
    * cuDNN ``deterministic=True`` / ``benchmark=False`` - if torch
      is importable. These are the standard reproducibility knobs and
      are scoped to torch (not the broader environment) so the side
      effect surface is acceptable.

    Public API - the single supported RNG-seeding entry point, exported
    via ``__all__``. :meth:`evaluate_benchmark` calls it once before an
    eval, but standalone callers that drive a policy rollout without
    going through ``evaluate_benchmark`` can call it directly to get
    reproducible rollouts.

    ``seed`` is required. ``None`` is the absence of a seed, and the three RNGs
    disagree about it: ``random`` and NumPy reseed from entropy while
    ``torch.manual_seed`` refuses it outright, so passing it through would leave
    a process-wide RNG side effect on a rollout that asked for none - and none at
    all on an install without torch, where the same call silently succeeds. The
    shared domain accepts ``None`` because ``randomize(seed=None)`` legitimately
    means "draw fresh entropy"; this applier opts out with ``allow_none=False``.
    To leave the RNGs untouched, do not call it - which is what every caller in
    this module already does (``if seed is not None``).

    NumPy / torch are imported lazily so this helper works on minimal
    installs that don't have torch (e.g. ``policy_provider="mock"``
    smoke tests).
    """
    # Local import: base.py imports this module at module level, so reaching the
    # shared domain from here has to stay deferred - the same convention this
    # module already uses for simulation.benchmark / .recording / .predicates.
    from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

    # This is public API (exported via ``__all__``) and documented for standalone
    # callers, so the bound is enforced where it is owned rather than only at the
    # facades one layer up: NumPy's own "Seed must be between 0 and 2**32 - 1"
    # names neither the parameter nor the method that accepted it.
    #
    # ``allow_none=False``: this is the applier, and there is no seed to apply for
    # ``None``. Passing it through would reseed ``random`` / NumPy from entropy and
    # then raise out of ``torch.manual_seed`` (or, on an install without torch,
    # silently succeed) - a process-wide side effect on a rollout that asked for no
    # seed, which is the opposite of the rule ``evaluate`` states below: "a ``None``
    # seed leaves the master RNG unbuilt rather than seeding it from entropy". The
    # refusal sits ahead of every RNG, so it has no side effect either.
    if seed_error := randomization_seed_error(seed, "set_eval_seed", max_seed=MAX_EVAL_SEED, allow_none=False):
        raise ValueError(seed_error)
    random.seed(seed)
    try:
        import numpy as _np

        _np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch as _torch

        _torch.manual_seed(seed)
        if _torch.cuda.is_available():
            _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# Hook signature: called every control step after send_action.
# on_frame(step_idx, observation, action) -> None
OnFrame = Callable[[int, dict[str, Any], dict[str, Any]], None]

# Success function: called after each step during evaluate().
# success_fn(observation) -> bool
SuccessFn = Callable[[dict[str, Any]], bool]


def _criterion_verdict(
    check: Callable[[Any], bool],
    subject: Any,
    *,
    label: str,
    episode: int,
    step: int,
) -> bool:
    """Evaluate an episode-outcome criterion, making a raise actionable.

    The eval loops call a caller-supplied outcome criterion after every applied
    action: :meth:`PolicyRunner.evaluate`'s ``success_fn`` and a benchmark
    spec's ``is_success`` / ``is_failure``. Every other per-step hook on those
    paths already states what a raise means - ``on_frame`` is best-effort
    telemetry (warn and continue), ``spec.on_step`` returns ``status="error"``
    naming the hook, and :meth:`PolicyRunner.run`'s ``stop_when`` is fatal
    because the caller asked for an early-return semantics the runner can no
    longer honor. The outcome criterion decides the evaluation's headline
    number, so a raise is fatal for the same reason: a ``success_rate``
    averaged over episodes whose outcome was never determined is not a
    measurement, and reporting one would misreport the evaluation. What this
    adds is the message - which criterion, which episode, which step - not a
    change of posture.

    ``bool()`` mirrors ``stop_when``'s coercion, so a NumPy scalar verdict -
    what ``observation["x"] > 0.5`` returns, and not an instance of ``bool`` -
    keeps working unchanged.

    Args:
        check: The criterion. Receives ``subject``, returns a truthy verdict.
        subject: What the criterion is evaluated against - the post-action
            observation for ``success_fn``, the live sim for a spec predicate.
        label: Criterion name for the message (e.g. ``"success_fn"``).
        episode: Zero-based episode index, so the message locates the failure.
        step: Control step within that episode, likewise.

    Returns:
        The criterion's verdict, coerced with ``bool()``.

    Raises:
        RuntimeError: If ``check`` raises. Chains the original and names the
            criterion, the episode and the step, so the failure is locatable
            instead of arriving as a bare ``KeyError`` from inside the loop.
            Whether that surfaces as a raise or an error envelope stays each
            eval method's own posture: ``run`` converts it via its terminal
            handler, while ``evaluate`` propagates rollout failures by design
            (a raising ``get_actions`` and a lost recording frame reach the
            caller the same way).
    """
    try:
        return bool(check(subject))
    except Exception as e:
        raise RuntimeError(
            f"{label} raised at episode {episode}, step {step}: {e!r}. The episode-outcome "
            "criterion cannot be evaluated, so the evaluation is aborted rather than reporting "
            "a success_rate over episodes whose outcome was never determined."
        ) from e


def _extract_frame_ndarray(render_result: dict) -> np.ndarray | None:
    """Decode the PNG bytes emitted by ``SimEngine.render`` into an ndarray.

    ``render()`` returns the image nested inside a content block as
    ``{"image": {"format": "png", "source": {"bytes": <str|bytes>}}}``.
    The ``bytes`` field may contain raw bytes (legacy) or a base64-encoded
    string (current). This helper walks that structure, decodes the PNG,
    and returns a contiguous (H, W, 3) RGB numpy array - the source is
    always run through ``PIL.Image.convert("RGB")`` so any alpha channel
    is dropped and the shape is a fixed 3-channel array. Returns ``None``
    if no decodable image is found (missing/malformed content blocks, an
    empty ``bytes`` field, or a PNG that fails to decode) - the recorder
    then skips the frame rather than aborting the rollout.
    """
    if not isinstance(render_result, dict):
        return None
    for block in render_result.get("content", []) or []:
        if not isinstance(block, dict):
            continue
        image = block.get("image")
        if not isinstance(image, dict):
            continue
        source = image.get("source") or {}
        png_bytes = source.get("bytes")
        if png_bytes is None and source.get("data") is not None:
            import base64

            png_bytes = base64.b64decode(source["data"])
        if not png_bytes:
            continue
        # Handle base64-encoded strings (current render() output)
        if isinstance(png_bytes, str):
            import base64

            png_bytes = base64.b64decode(png_bytes)
        try:
            import io

            from PIL import Image

            return np.asarray(Image.open(io.BytesIO(png_bytes)).convert("RGB"))
        except Exception:
            return None
    return None


# Canonical :class:`VideoConfig` field -> the dict keys accepted for it, canonical
# key first followed by the legacy/tool_spec aliases. Single source of truth for
# both the schema check (``VideoConfig.validation_error``) and the value lookup
# (``VideoConfig.from_dict``), so the accepted set cannot drift between the two.
_VIDEO_KEY_ALIASES: dict[str, tuple[str, ...]] = {
    "path": ("path", "record_video", "output_path"),
    "fps": ("fps", "video_fps"),
    "camera": ("camera", "video_camera", "camera_name"),
    "width": ("width", "video_width"),
    "height": ("height", "video_height"),
}

_VIDEO_ACCEPTED_KEYS: tuple[str, ...] = tuple(sorted(key for aliases in _VIDEO_KEY_ALIASES.values() for key in aliases))


@dataclass(frozen=True)
class VideoConfig:
    """Configuration for optional MP4 recording during :meth:`PolicyRunner.run`.

    Consolidates the five formerly-flat video parameters on
    :meth:`SimEngine.run_policy` into one typed object. Recording is an
    opt-in feature - if ``path`` is falsy, no recording occurs and the
    other fields are ignored.

    Attributes:
        path: Output MP4 path. ``None``/empty string → recording disabled.
            LLM-supplied, so it is validated (no ``..`` traversal, backslash
            separators, shell metacharacters, or symlinked target) before a
            writer is opened; set ``STRANDS_ROBOTS_VIDEO_ROOT`` to confine it
            to a sandbox.
        fps: Frames per second to write. Capped at ``control_frequency``
            when it would exceed it, so the rollout always plays back at
            real time (a rollout renders at most one frame per control
            step and cannot be up-sampled).
        camera: Camera name to render from. ``None`` → backend default.
        width: Render width in pixels.
        height: Render height in pixels.
    """

    path: str | None = None
    fps: int = 30
    camera: str | None = None
    width: int = 640
    height: int = 480

    @property
    def enabled(self) -> bool:
        """Whether recording is on: ``True`` iff an output ``path`` was set.

        The other fields (``fps``, ``camera``, ``width``, ``height``) are
        ignored when this is ``False`` -- a falsy ``path`` opts the whole
        rollout out of MP4 capture.
        """
        return bool(self.path)

    @staticmethod
    def _pick(d: dict[str, Any], field: str, default: Any = None) -> Any:
        """First present, non-``None`` value among ``field``'s accepted keys.

        Looks the canonical key up first, then the legacy aliases, so
        ``{"path": ..., "output_path": ...}`` resolves to the canonical one.
        Membership - not truthiness - decides: a caller-supplied ``0`` is
        returned as ``0`` (and rejected by :meth:`validation_error`) instead of
        collapsing into ``default`` the way an ``or`` chain would.

        Args:
            d: The caller's video-config dict.
            field: Canonical field name (a key of the alias map).
            default: Returned when no accepted key carries a value.

        Returns:
            The caller's value, or ``default``.
        """
        for key in _VIDEO_KEY_ALIASES[field]:
            value = d.get(key)
            if value is not None:
                return value
        return default

    @staticmethod
    def _positive_int_error(value: Any, key: str) -> str | None:
        """Error text when a ``video`` dict value is not a positive whole number.

        Thin binding of the shared frame/pixel-count domain
        (:func:`positive_whole_number_error`) to the ``video:`` message prefix,
        so this dict schema and the plain-MP4 recorder's keyword parameters
        cannot drift apart on what counts as a usable ``fps`` / ``width`` /
        ``height``.

        Args:
            value: The caller-supplied value.
            key: The dict key it came from, used in the message.

        Returns:
            An error message, or ``None`` when the value is usable.
        """
        return positive_whole_number_error(value, key, "video")

    @classmethod
    def validation_error(cls, d: Any) -> str | None:
        """Error text when ``d`` is not a video config this class can honor.

        Recording options arrive as a free-form dict (LLM tool call or direct
        API), so a mistyped key has no signature to bounce off. Silently
        ignoring one is the worst outcome: ``{"filename": "/tmp/a.mp4"}``
        leaves ``path`` unset and the rollout reports ``status="success"``
        with no MP4 anywhere, and ``{"path": p, "resolution": [320, 240]}``
        records at the default 640x480 while the caller believes otherwise.
        This rejects any key outside the accepted set (with a closest-match
        hint) and any known key whose value cannot be honored.

        Args:
            d: The caller's ``video`` argument. ``None`` (recording off) and an
                empty dict are valid.

        Returns:
            An error message describing the first problem found, or ``None``
            when the config is usable.
        """
        if d is None:
            return None
        if not isinstance(d, dict):
            return f"video must be a dict of recording options, got {type(d).__name__}."
        accepted = ", ".join(_VIDEO_ACCEPTED_KEYS)
        for key in d:
            if key in _VIDEO_ACCEPTED_KEYS:
                continue
            # Match case-insensitively so "FPS"/"Path" suggest their canonical
            # spelling; the cutoff is deliberately tight so an unrelated key
            # ("filename", "resolution") gets the accepted list rather than a
            # misleading nearest-neighbour.
            close = difflib.get_close_matches(str(key).lower(), _VIDEO_ACCEPTED_KEYS, n=1, cutoff=0.7)
            hint = f" Did you mean {close[0]!r}?" if close else ""
            return f"video: unknown key {key!r}.{hint} Accepted keys: {accepted}."
        for field in ("path", "camera"):
            value = cls._pick(d, field)
            if value is not None and not isinstance(value, str):
                return f"video: {field} must be a string, got {value!r}."
        for field in ("fps", "width", "height"):
            value = cls._pick(d, field)
            if value is None:
                continue
            if error := cls._positive_int_error(value, field):
                return error
        return None

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VideoConfig | None:
        """Build from a plain dict (tool_spec dispatcher path). ``None`` passthrough.

        Accepts both canonical keys and the legacy/tool_spec aliases listed in
        :meth:`validation_error`.

        Args:
            d: Video-config dict, or ``None``/empty for "no recording".

        Returns:
            The config, or ``None`` when ``d`` is empty.

        Raises:
            ValueError: When ``d`` carries a key or value that cannot be
                honored (see :meth:`validation_error`). Public entry points
                (``run_policy`` / ``eval_policy`` / ``evaluate_benchmark`` /
                ``start_policy``) check first and return a structured tool
                error, so this raise is the guard for direct construction.
        """
        if not d:
            return None
        if error := cls.validation_error(d):
            raise ValueError(error)
        return cls(
            path=cls._pick(d, "path"),
            fps=int(cls._pick(d, "fps", 30)),
            camera=cls._pick(d, "camera"),
            width=int(cls._pick(d, "width", 640)),
            height=int(cls._pick(d, "height", 480)),
        )


class _RolloutVideoWriter:
    """Optional per-rollout MP4 writer: validate the (LLM-supplied) output path,
    probe the camera, open an imageio writer, append frames at the requested fps
    cadence, and finalize on close.

    Extracted from :meth:`PolicyRunner.run` so the multi-episode evaluation loop
    (:meth:`PolicyRunner.evaluate`) records rollout video identically - one
    source of truth for the security-sensitive path validation
    and the frame-capture cadence, rather than two copies that can drift.
    """

    def __init__(
        self,
        sim: Any,
        video: VideoConfig,
        writer: Any,
        resolved_path: str,
        control_frequency: float,
    ) -> None:
        self.sim = sim
        self.video = video
        self.path = resolved_path
        self._writer = writer
        self.frame_count = 0
        self._frame_interval = control_frequency / max(video.fps, 1)
        self._next_frame_step = 0.0

    @classmethod
    def open(
        cls, sim: Any, video: VideoConfig | None, control_frequency: float
    ) -> tuple[_RolloutVideoWriter | None, dict[str, Any] | None]:
        """Return ``(writer, error)``.

        ``(None, None)``   -> recording disabled (``video`` is falsy); proceed.
        ``(None, error)``  -> setup failed; the caller returns ``error`` verbatim.
        ``(writer, None)`` -> writer ready.
        """
        if video is None or not video.enabled:
            return None, None
        # video.enabled guarantees video.path is a non-empty str; narrow for mypy.
        assert video.path is not None
        # video.path is LLM-supplied: reject shell metacharacters, backslash
        # separators, ".." traversal, and a symlinked target before we makedirs +
        # open a writer on it. Absolute paths stay allowed (the historic
        # contract); set STRANDS_ROBOTS_VIDEO_ROOT to sandbox them.
        _sb_root, _allow_abs, _allow_abs_env = video_sandbox_args()
        try:
            resolved = str(
                validate_output_path(
                    video.path,
                    sandbox_root=_sb_root,
                    allow_abs=_allow_abs,
                    label="video path",
                    allow_abs_env=_allow_abs_env,
                )
            )
        except ValueError as _e:
            return None, {"status": "error", "content": [{"text": f"video recording: {_e}"}]}

        # Pre-validate the camera name ONCE before the step loop. This surfaces
        # "camera not found" as a clean up-front error rather than silently
        # writing a 0-byte MP4 (sim.render() returns status=error, the rollout
        # runs to completion, and the user gets an empty file with no hint).
        probe_cam = video.camera or "default"
        try:
            _probe = sim.render(camera_name=probe_cam, width=video.width, height=video.height)
        except Exception as e:
            return None, {
                "status": "error",
                "content": [{"text": f"Video recording requested but render probe crashed: {e}"}],
            }
        if _probe.get("status") != "success":
            probe_text = (_probe.get("content") or [{}])[0].get("text", "")
            return None, {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"Video recording requested but camera "
                            f"'{probe_cam}' is not renderable.\n"
                            f"{probe_text}\n"
                            "Hint: robot cameras are namespaced, e.g. a "
                            "camera named 'side' inside robot 'arm1' compiles "
                            "as 'arm1/side'. Pass video={'camera': 'arm1/side', ...}."
                        )
                    }
                ],
            }

        imageio = require_optional(
            "imageio",
            pip_install="imageio imageio-ffmpeg",
            extra="sim-mujoco",
            purpose="video recording",
        )
        os.makedirs(os.path.dirname(os.path.abspath(resolved)), exist_ok=True)
        # A rollout renders at most one frame per applied control step, so the
        # video cannot carry more than ``control_frequency`` unique frames per
        # second of sim time. When the requested ``fps`` exceeds
        # ``control_frequency`` the capture cadence still grabs every step (it
        # cannot up-sample), so writing the MP4 at the requested ``fps`` would
        # play the rollout back FASTER than real time (by ``fps /
        # control_frequency``). Cap the writer fps at ``control_frequency`` so
        # the rollout always plays back at real time. When ``fps <=
        # control_frequency`` the capture cadence down-samples and the video is
        # already real time at the requested ``fps`` (unchanged).
        write_fps = video.fps
        if control_frequency > 0 and video.fps > control_frequency:
            write_fps = max(1, round(control_frequency))
            logger.warning(
                "Video fps=%d exceeds control_frequency=%.1f Hz; a rollout can "
                "render at most one frame per control step, so the MP4 is "
                "written at %d fps to play back at real time (requesting a "
                "higher fps would only speed the video up).",
                video.fps,
                control_frequency,
                write_fps,
            )
        writer = imageio.get_writer(  # type: ignore[attr-defined]
            resolved, fps=write_fps, quality=8, macro_block_size=1
        )
        return cls(sim, video, writer, resolved, control_frequency), None

    def capture(self, step_count: int) -> None:
        """Append one frame if the fps cadence is due at ``step_count``.

        Call once per applied control step; the cadence (``control_frequency /
        fps``) decides which steps actually render. A render/decode failure is
        skipped silently rather than aborting the rollout (a renderer hiccup
        must not kill training).
        """
        if step_count < self._next_frame_step:
            return
        frame = self.sim.render(
            camera_name=self.video.camera or "default",
            width=self.video.width,
            height=self.video.height,
        )
        img_arr = _extract_frame_ndarray(frame)
        if img_arr is not None:
            self._writer.append_data(img_arr)
            self.frame_count += 1
        self._next_frame_step += self._frame_interval

    def close(self) -> None:
        self._writer.close()


# on_frame hooks that raise are logged at WARN - user-provided telemetry is
# not allowed to kill the rollout. BUT if the hook raises on every single step
# (e.g. a recording hook with a typo'd observation key), we'd complete a 500-step
# episode with zero frames written and silently corrupt the dataset. After this
# many *consecutive* failures, the runner raises and fails the episode loudly.
#
# The counter resets on every success, so this bounds an ALWAYS-failing hook and
# nothing else: a hook failing every other step never reaches the limit. That is
# the right trade for caller telemetry, which is why
# :class:`~strands_robots.dataset_recorder.RecordingFrameError` is excluded from
# the tolerance entirely - a lost dataset frame is data loss, not telemetry, and
# tolerating it writes a short, re-timestamped episode under a successful
# rollout.
#
# Overridable via the ``max_onframe_failures`` kwarg on ``PolicyRunner.run``.
# See GH #117.
_MAX_CONSECUTIVE_ONFRAME_FAILURES = 5

# Fail-fast probe window for 100%-unresolved action keys. If EVERY action step
# in the opening ``_FAIL_FAST_PROBE_STEPS`` drives zero actuators (the policy's
# output keys cannot resolve to any of the robot's actuators/joints), the
# rollout can never move the robot, so :meth:`PolicyRunner.run` raises at the
# probe boundary instead of burning the whole episode (and every remaining model
# inference call + recording write) on a rollout that is structurally dead. Once
# any step resolves a single key the probe is permanently disarmed. See the
# end-of-episode all-unresolved guard for the (already-handled) terminal case.
_FAIL_FAST_PROBE_STEPS = 3


def _extract_result_json(result: object) -> dict[str, Any] | None:
    """Return the ``{"json": {...}}`` payload from a backend status dict.

    ``send_action`` reports unresolved action keys via a ``json`` content block
    (``{"unresolved_keys": [...], "applied": [...]}``). Returns that mapping, or
    ``None`` when the result carries no structured block (e.g. a non-MuJoCo
    backend or a coarse error without a per-key breakdown).
    """
    if not isinstance(result, dict):
        return None
    for block in result.get("content", []) or []:
        if isinstance(block, dict):
            payload = block.get("json")
            if isinstance(payload, dict):
                return payload
    return None


def _validate_action_key_map(action_key_map: Any) -> dict[str, Any] | None:
    """Reject a ``replay`` ``action_key_map`` no backend could honor.

    ``action_key_map`` binds recorded action-vector indices to the action keys
    ``send_action`` resolves, so it must be an ordered collection of unique
    strings. Three shapes are silently unusable and are rejected here:

    * a bare ``str`` - ``list("gripper")`` yields one key per character;
    * a non-string entry - it cannot name an actuator or joint;
    * a duplicate key - two recorded indices would write the same actuator,
      the later index silently overwriting the earlier one.

    An empty collection is rejected too: it maps no recorded value at all.

    Args:
        action_key_map: The caller-supplied map (``None`` selects the default
            ``robot_action_keys`` ordering and is always accepted).

    Returns:
        An agent-tool error dict describing the problem, or ``None`` when the
        map is usable.
    """
    if action_key_map is None:
        return None

    def _error(text: str) -> dict[str, Any]:
        return {"status": "error", "content": [{"text": f"replay: {text}"}]}

    if isinstance(action_key_map, str | bytes):
        return _error(
            f"action_key_map must be a list of action keys, not a bare string (got {action_key_map!r}); "
            "a string is consumed one character per action index."
        )
    if not isinstance(action_key_map, list | tuple):
        return _error(f"action_key_map must be a list or tuple of action keys (got {type(action_key_map).__name__}).")
    if not action_key_map:
        return _error("action_key_map is empty; pass one action key per recorded action-vector index.")
    bad = [key for key in action_key_map if not isinstance(key, str)]
    if bad:
        return _error(f"action_key_map entries must be action-key strings; got non-string entries {bad!r}.")
    duplicates = sorted({key for key in action_key_map if action_key_map.count(key) > 1})
    if duplicates:
        return _error(
            f"action_key_map has duplicate keys {duplicates}; each recorded action index needs its own key "
            "(a repeated key silently overwrites the earlier index's value)."
        )
    return None


class CooperativeStop(BaseException):
    """Raised by an ``on_frame`` hook to cooperatively stop a run.

    Inherits ``BaseException`` (not ``Exception``) so hook authors don't
    accidentally swallow it with a broad ``except Exception``. Honored by
    ``PolicyRunner.run`` and by the ``evaluate``/``evaluate_benchmark``
    paths: it is caught at the episode loop to return a normal
    stopped-early success result (``stopped_early=True``) rather than
    propagating as an uncaught exception.
    """


class _ChunkPipeline:
    """Yield ``(observation, action)`` pairs for a policy rollout.

    Two acquisition strategies behind one iterator:

    * **synchronous** (``async_rtc=False``): query the policy, fully drain the
      returned chunk, then re-query - inference never overlaps execution.
    * **async-RTC** (``async_rtc=True``): while the current chunk drains, fire
      the next ``get_actions`` on a single background worker once the chunk is
      ~50% consumed, then atomically swap it in at the seam. A chunk-emitting
      policy whose inference latency is <= the chunk's execution time pays
      (almost) zero visible stall - exactly how an async real-time controller
      hides inference latency on real hardware.

    Backend-agnostic and free of sim data races: the worker only ever calls the
    supplied ``query_chunk`` (pure policy inference). The sim observation for a
    prefetch is captured on the CONSUMING thread via ``observation_fn`` before
    the worker is submitted, and the sim is only ever stepped by the consumer,
    so no MuJoCo/Warp array is touched from two threads at once.

    The pipeline is an unbounded iterator - the consumer controls termination
    (success / failure / max-steps) by breaking out of the loop. Use it as a
    context manager so the inference worker is always joined on exit, even when
    the consumer breaks mid-chunk::

        with _ChunkPipeline(query_chunk, obs_fn, async_rtc=True,
                            rtc_inference_timeout_s=None) as chunks:
            for observation, action in chunks:
                sim.send_action(action, ...)
                if done:
                    break

    ``chunks_acquired`` / ``prefetch_hits`` / ``prefetch_blocks`` and the
    inference timings collected by ``query_chunk`` make latency masking provable
    from the result payload without grepping logs.
    """

    def __init__(
        self,
        query_chunk: Callable[[dict[str, Any], int], list[dict[str, Any]]],
        observation_fn: Callable[[], dict[str, Any]],
        *,
        async_rtc: bool,
        rtc_inference_timeout_s: float | None,
    ) -> None:
        self._query_chunk = query_chunk
        self._observation_fn = observation_fn
        self._async_rtc = async_rtc
        self._timeout = rtc_inference_timeout_s
        self.chunks_acquired = 0
        self.prefetch_hits = 0
        self.prefetch_blocks = 0
        self._executor: Any = None

    def __enter__(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        return self._iter_async() if self._async_rtc else self._iter_sync()

    def __exit__(self, *exc: object) -> None:
        # Join any in-flight inference so no background thread touches the
        # policy/sim after the rollout returns (the caller may immediately
        # reset() or destroy() the world). Returns None so an exception raised
        # inside the ``with`` block (e.g. a prefetch timeout) propagates.
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def _iter_sync(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        while True:
            observation = self._observation_fn()
            # The world is paused during inference on the synchronous path, so
            # the policy observed exactly 0 control steps of delay.
            chunk = self._query_chunk(observation, 0)
            self.chunks_acquired += 1
            if not chunk:
                raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
            for action in chunk:
                yield observation, action

    def _iter_async(self) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
        from concurrent.futures import Future, ThreadPoolExecutor
        from concurrent.futures import TimeoutError as FuturesTimeout

        def _swap_in(fut: Future[list[dict[str, Any]]]) -> list[dict[str, Any]]:
            # A prefetch HIT means inference already finished (the seam is
            # invisible); a BLOCK means inference ran slower than the chunk's
            # execution - the actionable "shorten the chunk / earlier trigger"
            # signal, so log it. A hard timeout turns a stuck model into a
            # structured error instead of an unbounded sim hang.
            if fut.done():
                self.prefetch_hits += 1
            else:
                self.prefetch_blocks += 1
                logger.warning(
                    "async-RTC seam starvation: prefetched chunk was not ready at the swap "
                    "point (inference slower than chunk execution). Blocking on it; consider a "
                    "shorter chunk or an earlier prefetch trigger."
                )
            try:
                return fut.result(timeout=self._timeout)
            except FuturesTimeout as e:
                raise RuntimeError(
                    f"async-RTC prefetch exceeded rtc_inference_timeout_s={self._timeout}s; "
                    f"policy inference is stuck. Raise the timeout or check the policy/server."
                ) from e

        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-prefetch-eval")
        cur_obs = self._observation_fn()
        cur_chunk = self._query_chunk(cur_obs, 0)
        self.chunks_acquired += 1
        if not cur_chunk:
            raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
        idx = 0
        prefetch_trigger = max(1, len(cur_chunk) // 2)
        prefetch: Future[list[dict[str, Any]]] | None = None
        prefetch_obs: dict[str, Any] | None = None

        while True:
            if idx >= len(cur_chunk):
                if prefetch is not None:
                    cur_chunk = _swap_in(prefetch)
                    if prefetch_obs is not None:
                        cur_obs = prefetch_obs
                    prefetch = None
                    prefetch_obs = None
                    self.chunks_acquired += 1
                else:
                    # Chunk too short to have triggered a prefetch -> one
                    # synchronous re-query.
                    cur_obs = self._observation_fn()
                    cur_chunk = self._query_chunk(cur_obs, 0)
                    self.chunks_acquired += 1
                if not cur_chunk:
                    # Drop-and-requery: a prefetched chunk arriving empty (a
                    # transient policy hiccup) degrades to ONE synchronous
                    # re-query before erroring, rather than killing an
                    # otherwise-healthy rollout on a single empty result.
                    logger.warning("async-RTC chunk arrived empty; falling back to one synchronous re-query.")
                    cur_obs = self._observation_fn()
                    cur_chunk = self._query_chunk(cur_obs, 0)
                    self.chunks_acquired += 1
                    if not cur_chunk:
                        raise RuntimeError(
                            "policy returned an empty action chunk twice (prefetch + synchronous "
                            "re-query); cannot continue rollout"
                        )
                idx = 0
                prefetch_trigger = max(1, len(cur_chunk) // 2)
                continue

            if prefetch is None and idx >= prefetch_trigger:
                prefetch_obs = self._observation_fn()
                # The prefetched chunk first applies after the remaining steps of
                # the current chunk drain - a known integer independent of how
                # long inference actually takes in wall-clock time.
                observed_delay = max(0, len(cur_chunk) - prefetch_trigger)
                prefetch = self._executor.submit(self._query_chunk, prefetch_obs, observed_delay)

            yield cur_obs, cur_chunk[idx]
            idx += 1


class PolicyRunner:
    """Backend-agnostic policy execution against a ``SimEngine``.

    Construct with any ``SimEngine`` and call :meth:`run`, :meth:`replay`, or
    :meth:`evaluate`. The runner is stateless across calls - safe to reuse.

    Args:
        sim: Any ``SimEngine`` implementation.
    """

    def __init__(self, sim: SimEngine):
        self.sim = sim

    #: Observation key prefix for a named body's world pose, merged in by
    #: :meth:`_observe` for every body a policy declares in
    #: :attr:`~strands_robots.policies.base.Policy.required_bodies`.
    _BODY_KEY_PREFIX = "body."

    #: ``get_body_state`` json field -> observation key suffix. The backend
    #: reports a body's pose under its own field names; the observation spells
    #: them like the floating-base signals already in the schema
    #: (``base_pos`` / ``base_quat`` / ``base_lin_vel`` / ``base_ang_vel``) so a
    #: policy reads one convention for the base and for any other link.
    _BODY_POSE_FIELDS: tuple[tuple[str, str], ...] = (
        ("position", "pos"),
        ("quaternion", "quat"),
        ("linear_velocity", "lin_vel"),
        ("angular_velocity", "ang_vel"),
    )

    def _resolve_required_bodies(self, policy: Policy | None) -> tuple[str, ...]:
        """Validate a policy's declared ``required_bodies`` once, before the rollout.

        Resolving up front is the whole point: a mimic tracker reads its anchor
        link on EVERY tick, so a name the scene does not contain has to fail
        here - with the available body names - rather than 300 steps of a
        silently absent key that the policy reads as a zero pose.

        Args:
            policy: The policy about to be rolled out. ``None`` (replay) and a
                policy that declares nothing both resolve to ``()``.

        Returns:
            Ordered, de-duplicated body names to merge into each observation.

        Raises:
            TypeError: If ``required_bodies`` is not a sequence of non-empty
                strings. A bare ``str`` is refused explicitly rather than
                iterated into one entry per character.
            RuntimeError: If the backend exposes no ``get_body_state``, or a
                declared body does not resolve in the current scene. Raised
                rather than returned for the reason this layer raises
                everywhere else: ``PolicyRunner`` is drivable directly and a
                direct caller has no envelope to read a refusal from.
        """
        declared = getattr(policy, "required_bodies", ()) or ()
        if not declared:
            return ()
        if isinstance(declared, str):
            raise TypeError(
                f"{type(policy).__name__}.required_bodies must be a sequence of body names, "
                f"not a bare str ({declared!r}) - a str iterates into one entry per character. "
                f"Use a tuple: ('{declared}',)."
            )
        bodies: list[str] = []
        for name in declared:
            if not isinstance(name, str) or not name.strip():
                raise TypeError(
                    f"{type(policy).__name__}.required_bodies entries must be non-empty "
                    f"body-name strings, got {name!r}."
                )
            if name not in bodies:
                bodies.append(name)

        probe = getattr(self.sim, "get_body_state", None)
        if not callable(probe):
            raise RuntimeError(
                f"{type(policy).__name__} declares required_bodies={tuple(bodies)}, but backend "
                f"{type(self.sim).__name__} exposes no get_body_state() to read a named body's "
                f"pose from. Run this policy on a backend that implements it (MuJoCo, Isaac)."
            )
        for name in bodies:
            result = probe(name)
            if not isinstance(result, dict) or result.get("status") != "success":
                detail = ""
                if isinstance(result, dict):
                    detail = " ".join(
                        str(block.get("text", "")) for block in result.get("content", []) if isinstance(block, dict)
                    ).strip()
                raise RuntimeError(
                    f"{type(policy).__name__} declares required_bodies entry {name!r}, which does "
                    f"not resolve to a body in the current scene. {detail}".rstrip()
                )
        return tuple(bodies)

    def _observe(
        self,
        robot_name: str | None,
        *,
        skip_images: bool,
        bodies: tuple[str, ...] = (),
    ) -> dict[str, Any]:
        """Fetch one observation, merging the world pose of each declared body.

        The single point where every rollout loop in this module reads state, so
        the ``required_bodies`` contract holds identically on the synchronous,
        chunked and RTC paths instead of only whichever one was patched.

        Args:
            robot_name: Robot to observe, forwarded verbatim to the backend.
            skip_images: Backend's camera-render hint (from ``requires_images``).
            bodies: Pre-resolved names from :meth:`_resolve_required_bodies`.
                Empty (the default) returns the backend's dict untouched, so a
                policy that declares nothing pays no extra backend call.

        Returns:
            The observation dict, plus ``body.<name>.{pos,quat,lin_vel,ang_vel}``
            for each requested body.
        """
        obs = self.sim.get_observation(robot_name=robot_name, skip_images=skip_images)
        if not bodies:
            return obs
        for name in bodies:
            state = self._body_pose(name)
            for field, suffix in self._BODY_POSE_FIELDS:
                value = state.get(field)
                if value is not None:
                    obs[f"{self._BODY_KEY_PREFIX}{name}.{suffix}"] = [float(v) for v in value]
        return obs

    def _body_pose(self, body_name: str) -> dict[str, Any]:
        """Read one body's pose json out of the backend's ``get_body_state`` envelope.

        Args:
            body_name: Body name, already validated by
                :meth:`_resolve_required_bodies`.

        Returns:
            The backend's pose json block.

        Raises:
            RuntimeError: If the body stopped resolving mid-rollout (a scene
                replace can remove it) or the envelope carries no json block.
                The pose feeds the policy's network input, so an absent one is
                fatal - not a key to quietly omit from this tick's observation.
        """
        result = self.sim.get_body_state(body_name)  # type: ignore[attr-defined]
        if isinstance(result, dict) and result.get("status") == "success":
            for block in result.get("content", []):
                if not isinstance(block, dict):
                    continue
                pose = block.get("json")
                if isinstance(pose, dict):
                    return pose
        raise RuntimeError(
            f"required body {body_name!r} no longer resolves to a pose in the scene; "
            f"a policy declaring it via required_bodies cannot be stepped."
        )

    def _control_substeps(self, control_frequency: float, override: int | None = None) -> int:
        """Physics steps per applied action so a position-servo arm tracks the
        full control period (1/control_frequency), not a single physics dt.

        Identical derivation to :meth:`run` - extracted so the eval paths
        (:meth:`evaluate` / :meth:`_evaluate_with_spec`) step physics for the
        SAME wall-clock period per action. Without this, eval called
        ``send_action`` with the default ``n_substeps=1`` (a single ~2 ms
        ``mj_step``), so the arm integrated ~10% of the way toward each target
        before the next action overwrote ``ctrl`` - rollouts looked like the
        policy was a no-op even when commanding valid targets.

        Args:
            control_frequency: Control-loop rate in Hz, used with the backend's
                physics timestep to derive the substep count.
            override: Explicit substeps per action, or ``None`` to derive.

        Returns:
            Physics steps to advance per applied action (always ``>= 1``).

        Raises:
            ValueError: If ``override`` is not a positive integer. It used to be
                clamped with ``max(1, int(override))``, so ``0``/``-5`` silently
                collapsed to a single physics step - reinstating the exact
                under-integration this helper exists to prevent - and a float
                was truncated. The public entry points reject such a value with
                a structured error before reaching the runner; this raise is the
                guarantee for callers driving ``PolicyRunner`` directly.
        """
        if override is not None:
            if isinstance(override, bool) or not isinstance(override, int) or override < 1:
                raise ValueError(f"control_substeps must be a positive integer, got {override!r}.")
            return override
        dt = None
        try:
            dt = self.sim.physics_timestep()
        except Exception:  # noqa: BLE001 - never fail a run on a probe
            dt = None
        if dt and dt > 0 and control_frequency > 0:
            return max(1, round((1.0 / control_frequency) / dt))
        return 1

    def _reject_recording_rate_mismatch(self, control_frequency: float, method: str) -> None:
        """Refuse a rollout the engine's open dataset recording cannot describe.

        The dataset recorder is driven once per control step with no
        decimation, and LeRobot timestamps each frame positionally from the
        rate ``start_recording(fps=...)`` wrote into the metadata. A rollout
        capturing at a different ``control_frequency`` therefore cannot be
        labelled correctly, only mislabelled - the episode declares a
        duration it was not captured over, which is the control period a
        policy trains on and the budget ``replay_episode`` replays it at.

        The engine's rollout entry points already refuse this before a frame
        is written. ``run`` and ``evaluate`` are also driven directly, with
        the engine's guard off the path, so the check is repeated here for
        the same reason :meth:`_control_substeps` raises for a bad
        ``control_substeps``: this layer owes a direct caller its own
        guarantee. A backend that cannot record has no ``_is_recording``, and
        a duck-typed test double may have neither hook, so both are probed
        rather than assumed.

        Args:
            control_frequency: Rate this rollout captures frames at.
            method: Public method name, used to prefix the message.

        Raises:
            ValueError: If a recording is open at a rate other than
                ``control_frequency``. Raised rather than returned as an error
                dict so it cannot be absorbed, and reported before any frame
                is written so the caller loses nothing.
        """
        is_recording = getattr(self.sim, "_is_recording", None)
        if not callable(is_recording) or not is_recording():
            return
        active_recorder = getattr(self.sim, "_active_recorder", None)
        if not callable(active_recorder):
            return
        recorder = active_recorder()
        if recorder is None:
            return
        # Imported here, like the engine's own call site, so the recording
        # module stays out of this module's import graph.
        from strands_robots.simulation.recording import dataset_rate_mismatch_reason

        reason = dataset_rate_mismatch_reason(method, recorder, control_frequency)
        if reason is not None:
            raise ValueError(reason)

    # ------------------------------------------------------------------
    # Recorder per-episode boundary (issue #708)
    # ------------------------------------------------------------------
    #
    # The dataset_recorder attached to ``_world._backend_state`` keeps a single
    # open LeRobot episode buffer. ``add_frame`` appends to that buffer; only
    # ``save_episode`` rolls over to a new episode and bumps
    # ``episode_count``. ``stop_recording`` flushes the last open episode but
    # has no idea how many episodes the caller intended.
    #
    # Without this helper, every ``for ep in range(n_episodes):`` loop in
    # ``evaluate`` / ``_evaluate_with_spec`` records ONE giant episode of
    # ``n_episodes * max_steps`` frames into the dataset. The agent sees
    # ``total_episodes=1`` in the parquet meta but a status=OK summary
    # because the recorder did receive frames. (#708 - silent collapse.)
    #
    # Calling this at the end of each policy-runner episode forces a per-
    # episode boundary in the recorded dataset. Skipped silently when no
    # recorder is attached (eval without recording is the common case).
    def _finalize_recorder_episode(self) -> str | None:
        """Roll the attached dataset_recorder over to a new episode.

        Called at end of each rollout iteration in ``evaluate`` and
        ``_evaluate_with_spec``. No-op when no recorder is attached or when
        the episode buffer is empty (e.g. degenerate policy returned no
        actions and ``add_frame`` was never called).

        Returns:
            ``None`` when the episode was flushed, or when there was nothing
            to flush. A reason string when the flush FAILED, for the caller to
            stop on and report.

            A failed flush is not telemetry. The recorder marks itself closed
            because the LeRobot episode buffer is in an undefined state, and
            :meth:`~strands_robots.dataset_recorder.DatasetRecorder.add_frame`
            then returns on a closed recorder without writing a frame or
            counting a drop - so a later episode's frames reach no dataset and
            leave no trace in the recorder's own accounting either. An
            evaluation that carried on would report a ``success_rate`` over
            episodes whose data does not exist. Every sibling flush already
            refuses the same way: ``stop_recording`` and
            :meth:`~strands_robots.simulation.base.SimEngine.save_episode`
            drop the poisoned recorder and return an error,
            :meth:`~strands_robots.simulation.base.SimEngine.run_policy` with
            ``n_episodes`` aborts its remaining episodes, and the MuJoCo
            backend's ``reset``
            surfaces the failure rather than resetting into an undefined
            state.
        """
        try:
            world = getattr(self.sim, "_world", None)
            if world is None:
                return None
            recorder = world._backend_state.get("dataset_recorder")
        except AttributeError:
            return None
        if recorder is None:
            return None
        # Don't flush an empty buffer - LeRobot raises on save_episode with
        # zero frames, and a degenerate rollout still counts as "no data" for
        # this episode rather than an error.
        pending = getattr(recorder, "episode_frame_count", 0)
        if pending <= 0:
            return None
        try:
            result = recorder.save_episode()
        except Exception as e:  # noqa: BLE001 - reported to the caller, not raised
            return f"save_episode raised: {e}"
        if isinstance(result, dict) and result.get("status") != "success":
            return f"save_episode: {result.get('message', result)}"
        return None

    # run(): blocking policy execution
    def run(
        self,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str = "",
        duration: float = 10.0,
        n_steps: int | None = None,
        control_frequency: float = 50.0,
        action_horizon: int = 8,
        fast_mode: bool = False,
        video: VideoConfig | None = None,
        on_frame: OnFrame | None = None,
        max_onframe_failures: int | None = None,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        seed: int | None = None,
        async_rtc: bool | None = None,
        rtc_inference_timeout_s: float | None = None,
        stop_when: Callable[[SimEngine], bool] | None = None,
    ) -> dict[str, Any]:
        """Run ``policy`` on ``robot_name`` for ``duration`` seconds.

        Args:
            robot_name: Name of robot in the sim.
            policy: Already-constructed ``Policy`` instance. Callers (typically
                ``SimEngine.run_policy``) are responsible for policy
                construction so tests can inject mocks trivially.
            instruction: Natural-language instruction forwarded to the policy.
            duration: Wall-clock seconds to run (interpreted as control steps
                via ``control_frequency``). Used only when ``n_steps`` is None,
                and validated only then: a finite positive number, the domain
                :meth:`~strands_robots.simulation.base.SimEngine._validate_duration`
                applies at the facade. Raises ``ValueError`` otherwise.
            n_steps: Explicit integer step-count horizon resolved by the caller
                from ``n_steps`` / the legacy ``max_steps`` alias. When given it
                is the exact number of control steps executed, bypassing the
                lossy ``int(duration * control_frequency)`` recomputation. Must
                be a positive integer - the domain
                :meth:`~strands_robots.simulation.base.SimEngine._resolve_horizon`
                applies at the facade - and a value outside it raises
                ``ValueError`` rather than deferring the horizon to ``duration``.
            control_frequency: Target Hz for ``policy.get_actions`` calls.
            action_horizon: Max actions consumed per policy call before
                requerying observation. Clamped up to the policy's own
                ``actions_per_step`` so a model trained for N-step
                open-loop chunk replay never has its chunk truncated
                below N (the effective horizon is
                ``max(action_horizon, policy.actions_per_step)``).
                Must be a positive integer: a value outside that domain is
                either clamped to 1 - running a re-query interval the caller
                never asked for - or leaks a bare conversion error out of the
                first inference, so it is refused here exactly as
                ``run_policy`` refuses it.
            fast_mode: If True, run unpaced - as fast as inference and physics
                allow. If False (default) the loop is paced on a DEADLINE at
                ``control_frequency``: the wall clock a step spends on
                inference, physics substeps, rendering and recording is
                subtracted from the period rather than added to it, so a
                rollout of ``n`` steps takes ``n / control_frequency`` seconds
                whatever a step costs, as long as it fits inside one period. A
                step that overruns its period drops the missed deadlines
                instead of chasing them, so a slow step is followed by a gap
                rather than by a burst of back-to-back actions.
            video: Optional :class:`VideoConfig` - set ``video.path`` to enable
                MP4 recording via :meth:`SimEngine.render`.
            on_frame: Optional hook ``(step_idx, obs, action) -> None`` called
                after every ``send_action``. Public extension point - backends
                layer in recording / telemetry / graceful-stop via this hook
                without subclassing the runner.
            policy_kwargs: Optional per-call goal payload forwarded verbatim to
                every ``policy.get_actions(obs, instruction, **policy_kwargs)``
                call. This is the local-sim analogue of the mesh ``tell()``
                #300 path: it carries the well-known goal keys
                (``target_pose`` / ``target_joints`` / ``target_velocity`` /
                ``world_update``) to non-VLA providers that read their goal
                from kwargs rather than the instruction (cuRobo, MoveIt2, WBC).
                VLA providers ignore unknown kwargs per the #300 contract, so
                this is safe to forward unconditionally. ``None`` forwards no
                extra kwargs (identical to the historical behaviour).
            max_onframe_failures: Maximum *consecutive* exceptions from the
                ``on_frame`` hook before the runner aborts the episode.
                ``CooperativeStop`` and
                :class:`~strands_robots.dataset_recorder.RecordingFrameError` are
                exempt from the count rather than tolerated by it: the first is
                the documented graceful stop and the second is data loss, so a
                lost dataset frame aborts on the FIRST occurrence whatever this
                is set to. ``None`` (default) uses
                ``_MAX_CONSECUTIVE_ONFRAME_FAILURES`` (currently ``5``). A
                broken recording hook otherwise silently produces empty
                datasets - see GH #117. Non-consecutive failures reset the
                counter. Must otherwise be a positive integer
                (:func:`~strands_robots.utils.positive_count_error`), raised
                rather than returned because raising is this layer's contract:
                a value the counter cannot be compared against would disable
                the abort above instead of resizing it.
            control_substeps: Physics steps advanced per applied action. ``None``
                (default) derives the count from ``control_frequency`` and the
                backend's physics timestep, so a position-servo arm integrates
                over the full control period instead of a single physics ``dt``.
                An explicit value must be a positive integer: ``0``, a negative
                value, a bool or a float raises :class:`ValueError` rather than
                being clamped, because a clamped ``0`` reinstates the exact
                under-integration the derivation exists to prevent - the arm
                then covers a fraction of the way to each target before the next
                action overwrites it, and the rollout looks like a no-op. The
                public entry points refuse such a value with a structured error
                before it reaches the runner.
            seed: Optional master RNG seed for a reproducible single rollout.
                When set, ``set_eval_seed`` reseeds Python / NumPy / torch /
                cuDNN and ``policy.reset(seed=...)`` is forwarded so the
                policy's stochastic ops (VLA action-chunk sampling, diffusion
                noise, attention dropout) draw from a deterministic state.
                Without this, a single ``run`` draws from the unmanaged global
                RNG, so the same scene + policy can grasp on one run and miss
                on the next. ``None`` (default) leaves RNG state untouched,
                preserving historical behaviour. Multi-episode reproducibility
                already flows through :meth:`evaluate`'s per-episode reseed.
            async_rtc: When ``True``, overlap policy inference with action
                execution via a single background worker (latency masking).
                While the current action chunk drains, the next
                ``get_actions()`` is fired once the chunk is ~50% consumed,
                using a fresh mid-execution observation, and atomically swapped
                in when the current chunk runs out. A policy whose inference
                latency is at most the chunk's execution time then pays
                (almost) zero visible stall at the chunk seam - the same way an
                async real-time controller hides inference latency on real
                hardware. Whether the chunk SEAM is additionally blended is a
                separate, checkpoint-level property (``supports_rtc``): a policy
                loaded from a checkpoint with an enabled ``rtc_config`` (e.g.
                pi0 / pi0.5, or a SmolVLA checkpoint configured for RTC) carries
                prev-chunk state across the seam and joins consecutive chunks
                smoothly, whereas a chunk-emitting policy WITHOUT an
                ``rtc_config`` - MolmoAct2, ACT, diffusion, and the public
                ``lerobot/smolvla_base`` checkpoint - gets the overlap (latency
                masking) but a plain chunk swap at the seam. This flag only
                schedules the overlap; it never enables or touches the policy's
                RTC machinery, so it is provider-agnostic. ``False`` keeps the
                historical
                synchronous chunk-then-drain loop, which is correct for
                single-step policies and any policy whose ``get_actions`` reads
                live sim state. ``None`` (default) auto-resolves the flag from
                ``policy.is_chunk_emitting()``: chunk-emitting VLAs (pi0, pi0.5,
                pi0-FAST, SmolVLA, MolmoAct2) enable the overlap and single-step
                policies stay synchronous, so the latency-masking default is
                correct without the caller having to know the policy's shape. An
                explicit ``True``/``False`` always wins over the auto-resolution.
                The policy object is only ever invoked from the
                single background worker (never concurrently), and the runner
                blocks on any in-flight inference before returning so no thread
                touches the policy or sim after :meth:`run` exits.
            rtc_inference_timeout_s: Hard per-chunk timeout (seconds) for the
                async-RTC prefetch. When set and a prefetched inference has not
                returned by the time its chunk must be swapped in, the swap
                raises and :meth:`run` returns a structured ``status=error``
                result (with the RTC telemetry block) rather than waiting for
                every remaining chunk of a slow model. The runner still joins the
                single in-flight worker on shutdown (Python cannot forcibly kill a
                running thread, and a leaked worker would touch the policy after
                :meth:`run` returns), so the abort is bounded by ONE inference,
                not the whole rollout. ``None`` (default) waits without a deadline
                (historical behaviour). Ignored on the synchronous path.
            stop_when: Optional semantic early-return condition - a callable
                ``(sim) -> bool`` evaluated against the LIVE sim after every
                applied action, on BOTH the synchronous and async-RTC paths.
                The first ``True`` ends the rollout cleanly with
                ``stopped_reason="predicate"``; the remaining actions of an
                in-flight chunk are dropped, so the early-return latency bound
                is ONE control step regardless of the policy's chunk length
                (on the async-RTC path any in-flight prefetch is still joined
                before :meth:`run` returns). Callers driving the runner
                through :meth:`SimEngine.run_policy` pass a predicate-DSL dict
                compiled via
                :func:`~strands_robots.simulation.benchmark_spec.compile_stop_when`;
                programmatic callers may pass any callable (mirroring
                :meth:`evaluate`'s ``success_fn``). A raising ``stop_when`` is
                fatal (``status="error"``): the caller asked for an
                early-return semantics the runner can no longer honor, and
                silently running to the step budget would misreport the
                rollout. ``None`` (default) preserves the pure step-budget
                horizon.

        Returns:
            ``{"status": "success"|"error", "content": [{"text": ...},
            {"json": {...}}]}``. The ``json`` block is agent-consumable and
            carries the rollout facts as typed fields - ``robot_name``,
            ``policy``, ``instruction``, ``n_steps``, ``steps_used`` (the
            control steps actually executed, equal to ``n_steps``),
            ``elapsed_s``, ``stopped_early``, ``stopped_reason``
            (``"predicate"`` - the ``stop_when`` condition fired; ``"budget"``
            - the step/duration horizon was exhausted; ``"cancelled"`` - a
            cooperative stop, e.g. ``stop_policy``; on ``status="error"``
            results the field is ``"error"``), ``action_errors``,
            ``video_path`` (``None`` when
            no MP4 was written), ``video_frames`` and ``sim_time_s`` (when the
            backend reports sim time) - so callers can self-correct without
            regex-parsing the human-readable ``text``. The block also carries the
            async-RTC telemetry (``rtc_async_enabled``, ``rtc_chunks_acquired``,
            ``rtc_prefetch_hits``, ``rtc_prefetch_blocks``, ``rtc_avg_inference_ms``,
            ``rtc_max_inference_ms``) so latency masking is provable from the
            payload instead of from logs. It also carries the per-actuator
            resolution stats - ``action_resolution_rate`` (a
            ``{actuator_name: fraction_of_steps_driven}`` map) and
            ``partial_action_failure_rate`` (the mean fraction of the robot's
            DOF never driven; ``0.0`` == every actuator moved every step,
            ``~0.83`` == only 1 of 6 actuators ever moved) - so a rollout that
            silently drives only a subset of the robot's joints is visible
            instead of looking like a clean ``success`` with a zero success-rate.

            Fail-fast: if EVERY action step in the opening probe window
            (``_FAIL_FAST_PROBE_STEPS``, currently 3) drives zero actuators -
            i.e. none of the policy's emitted keys resolve to any of the robot's
            actuators - the rollout can never move the robot, so this raises
            (returned as ``status=error``) at the probe boundary instead of
            running the full episode (and every remaining model inference call +
            recording write). The error enumerates the unresolved keys and the
            robot's valid actuator names. A PARTIAL failure (some keys resolve)
            is operational and runs to completion, surfaced via
            ``partial_action_failure_rate``.
        """
        # A single rollout draws the policy's stochastic ops (VLA action-
        # chunk sampling, diffusion noise) from the unmanaged global RNG, so the
        # same scene + policy grasps on one run and misses on the next. When a
        # seed is given, reseed the client RNGs once and forward it to the policy
        # (mirrors the per-episode reseed in evaluate()). Default None leaves RNG
        # state untouched, preserving historical behaviour.
        # Refuse before any frame reaches the engine's open recording.
        self._reject_recording_rate_mismatch(control_frequency, "PolicyRunner.run")
        # The domain is enforced here as well as at the facades one layer up:
        # PolicyRunner is drivable directly, and a direct caller has no
        # structured envelope to read a refusal from. Same shared rule as
        # SimEngine._validate_seed, raised rather than returned because raising
        # is this layer's contract.
        # Local import: base.py imports PolicyRunner at module level, so
        # reaching the shared domain from here has to stay deferred - the
        # same convention this module already uses for
        # simulation.benchmark / simulation.recording / simulation.predicates.
        from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

        if seed_error := randomization_seed_error(seed, "PolicyRunner.run", max_seed=MAX_EVAL_SEED):
            raise ValueError(seed_error)
        # Same shared domain the facade one layer up enforces, raised rather
        # than returned because raising is this layer's contract: PolicyRunner is
        # drivable directly and a direct caller has no envelope to read a refusal
        # from. A deadline outside the domain makes the seam swap's own
        # "policy inference is stuck" diagnosis false - see
        # SimEngine._validate_rtc_inference_timeout for the measured failure modes.
        if rtc_inference_timeout_s is not None and (
            timeout_error := positive_finite_number_error(
                rtc_inference_timeout_s, "rtc_inference_timeout_s", "PolicyRunner.run"
            )
        ):
            raise ValueError(timeout_error)
        # Same shared domain, raised for the same reason: a limit outside it
        # silences the consecutive-failure abort this runner owns, and the
        # warning that would report the hook - see
        # SimEngine._validate_onframe_failure_limit for the measured failure modes.
        if max_onframe_failures is not None and (
            limit_error := positive_count_error(max_onframe_failures, "max_onframe_failures", "PolicyRunner.run")
        ):
            raise ValueError(limit_error)
        # Same shared domain the facade one layer up enforces, raised for the
        # same reason: PolicyRunner is drivable directly and a direct caller has
        # no envelope to read a refusal from. A horizon outside the domain is
        # silently clamped to 1 by resolve_chunk_length - so the rollout runs a
        # re-query interval the caller never asked for - or, when int() cannot
        # convert it, leaks a bare conversion error out of the FIRST inference
        # naming neither the parameter nor this method. Unconditional, exactly as
        # the entry point is: whether the policy carries cross-chunk RTC state
        # (and so ignores the horizon) is a property of the policy rather than of
        # the caller's request. See SimEngine._validate_action_horizon.
        if horizon_error := positive_count_error(action_horizon, "action_horizon", "PolicyRunner.run"):
            raise ValueError(horizon_error)
        # The horizon is a PAIR, and it is resolved here, so both halves are
        # validated here. ``n_steps`` whenever it is given: that is the exact
        # condition SimEngine._resolve_horizon refuses it on, so a step count
        # refused for a rollout through the facade cannot be accepted for the
        # same rollout driven directly. Without it, a value outside the domain
        # did not fail - the ``> 0`` test below handed the horizon to the OTHER
        # knob, so ``n_steps=0`` ran ``duration``'s 10.0s default (500 steps and
        # 500 applied actions nobody asked for), while ``2.7``/``True``
        # truncated to a horizon nobody typed.
        if n_steps is not None:
            if steps_error := positive_count_error(n_steps, "n_steps", "PolicyRunner.run"):
                raise ValueError(steps_error)
        # ``duration`` only when no step count was given, because that is the
        # only case in which it sets the horizon - the same ``if n_steps is
        # None`` gate SimEngine.run_policy applies around _validate_duration,
        # so neither layer reports on a parameter the rollout will not read.
        # Unvalidated, ``0``/``-5`` returned status=success with zero steps and
        # ``stopped_reason="budget"`` - the field an agent reads to decide
        # whether to retry, asserting a horizon was exhausted when there was
        # none - and ``nan``/``inf``/a string leaked a bare conversion or
        # operand error out of the arithmetic below, naming neither the
        # parameter nor this method.
        elif duration_error := positive_finite_number_error(duration, "duration", "PolicyRunner.run"):
            raise ValueError(duration_error)
        if seed is not None:
            set_eval_seed(seed)
            try:
                policy.reset(seed=seed)
            except Exception as e:  # noqa: BLE001 - reset is best-effort
                logger.warning(
                    "policy.reset(seed=%d) raised %s; continuing without policy-side reseed",
                    seed,
                    e,
                )

        # Auto-resolve the async-RTC overlap from the policy's own shape when the
        # caller did not pin it. Chunk-emitting VLAs (pi0/pi0.5/pi0-FAST/SmolVLA/
        # MolmoAct2) benefit from hiding inference behind chunk execution, while a
        # single-step policy gains nothing - so the latency-masking default is
        # correct without the caller knowing the policy's internals. An explicit
        # True/False always wins. Use getattr so a duck-typed policy_object that
        # predates is_chunk_emitting() simply stays on the synchronous path.
        if async_rtc is None:
            _emit = getattr(policy, "is_chunk_emitting", None)
            async_rtc = bool(_emit()) if callable(_emit) else False
            logger.info(
                "async_rtc auto-resolved to %s from %s.is_chunk_emitting()",
                async_rtc,
                type(policy).__name__,
            )

        # RTC telemetry, reported in the result json so latency masking is
        # provable without grepping logs. inference_ms collects every
        # get_actions wall-time (both paths); the prefetch hit/block counters and
        # chunks_acquired are async-only (0 on the synchronous path). list.append
        # is atomic under the GIL, so the worker thread appending an inference
        # time never races the main thread reading the list after shutdown(wait).
        inference_ms: list[float] = []
        rtc_chunks_acquired = 0
        rtc_prefetch_hits = 0
        rtc_prefetch_blocks = 0

        def _rtc_telemetry() -> dict[str, Any]:
            # The async-RTC telemetry block, merged into every result json
            # (success and error) so latency masking is provable from the
            # structured payload without grepping logs. On the synchronous path
            # the prefetch counters stay 0 and only the inference timings carry
            # information.
            _n = len(inference_ms)
            return {
                "rtc_async_enabled": bool(async_rtc),
                "rtc_chunks_acquired": rtc_chunks_acquired,
                "rtc_prefetch_hits": rtc_prefetch_hits,
                "rtc_prefetch_blocks": rtc_prefetch_blocks,
                "rtc_avg_inference_ms": round(sum(inference_ms) / _n, 3) if _n else 0.0,
                "rtc_max_inference_ms": round(max(inference_ms), 3) if _n else 0.0,
            }

        # Video recording lifecycle (path validation + camera probe + writer)
        # lives in _RolloutVideoWriter so run() and evaluate() record identically.
        vwriter, _video_err = _RolloutVideoWriter.open(self.sim, video, control_frequency)
        if _video_err is not None:
            # Every error result carries the stopped_reason="error" json block
            # (the "recorded on ALL exit paths" contract); the writer's error
            # dict is text-only because evaluate() shares it, so tag it here.
            _video_err.setdefault("content", []).append(
                {"json": {"stopped_reason": "error", "steps_used": 0, "n_steps": 0}}
            )
            return _video_err

        stopped_early = False
        # Why the rollout ended, reported in the result json so an agent
        # deciding whether to retry can distinguish "the world reached the
        # goal state" from "the step budget ran out" from "the user cancelled"
        # (stopped_early alone conflates the last two). "budget" is the
        # default (the loop ran its full horizon); the CooperativeStop handler
        # re-tags it "cancelled", a fired stop_when re-tags it "predicate",
        # and every error return reports "error".
        stopped_reason = "budget"
        stop_predicate_fired = False
        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Named-body poses the policy declared it needs (mimic trackers read an
        # anchor link). Resolved once here so an unknown name fails before the
        # loop instead of being read as a zero pose on every tick; () for the
        # overwhelming majority of policies, which adds no backend call.
        _bodies = self._resolve_required_bodies(policy)
        # Open-loop chunk replay consumes H actions from ONE observation. That
        # observation is the correct PRE-action state for the FIRST action only;
        # the sim advances as the chunk drains. When a dataset recording is
        # active the on_frame hook writes (observation, action) per step, so
        # re-using the chunk-start observation for every action records H
        # identical (frozen image + frozen proprioceptive state) frames paired
        # with H DIFFERENT actions - a temporally-misaligned behavioural-cloning
        # dataset (the recorded image never matches the action taken from it).
        # Detect an active recording via the engine's own contract and, when
        # set, refresh the observation handed to on_frame per step so each
        # recorded frame pairs the action with the state it actually acts on.
        # Inference still consumes the chunk-start observation (correct
        # open-loop replay); only the RECORDED frame is refreshed. The default
        # (no recording / duck-typed sim without the hook) keeps the historical
        # single-fetch-per-chunk behaviour, so eval/inference are unaffected.
        _is_rec = getattr(self.sim, "_is_recording", None)
        _record_per_step_obs = bool(_is_rec()) if callable(_is_rec) else False
        # Normalise the per-call goal payload once. Forwarded verbatim to every
        # get_actions() call; an empty dict is the historical (no-kwargs) path.
        _policy_kwargs = policy_kwargs or {}

        # Tell the policy the loop's control rate BEFORE the rollout so
        # latency-sensitive providers (RTC) convert wall-clock inference
        # latency into the correct number of consumed action steps. Without
        # this they fall back to a hardcoded rate and mis-blend the chunk
        # seam at any other frequency.
        policy.set_control_frequency(control_frequency)
        # Initialize BEFORE try so CooperativeStop never sees unbound names.
        # ``time.monotonic()``: the only thing derived from this base is how
        # long the rollout ran, and a duration is measured rather than
        # recorded. On ``time.time()`` a wall-clock step - an NTP correction, a
        # ``date -s``, a resume from suspend - moved the reported ``elapsed_s``
        # by the size of the step while the rollout itself ran exactly as long
        # as it did, so a 30s step turned a 2s episode into a 32s record and a
        # backward step reported it negative. Named for the clock it holds so a
        # future reader does not reach for ``time.time()`` again.
        start_mono = time.monotonic()
        step_count = 0
        # Bound before the rollout so the ``except CooperativeStop`` handler and
        # the ``_apply`` closure never see an unbound name, the same reason
        # ``start_mono`` is bound above.
        ticker: Ticker | None = None
        # The Ticker owns a selector and a socketpair, so an unclosed one leaks
        # two descriptors per rollout - which an eval loop repeats once per
        # episode. Released by an ExitStack rather than by a ``with Ticker(...)``
        # item because the pacer is acquired CONDITIONALLY - a ``fast_mode``
        # rollout must not construct one, nor pay the mesh import below - while
        # its live region is the whole rollout body, which reaches it through the
        # ``_apply`` closure rather than from a single loop. The stack expresses
        # both, and hands the release to the language instead of to a ``finally``
        # that every future edit to this body has to remember.
        with contextlib.ExitStack() as pacing_resources:
            try:
                # Prefer an explicit integer step count when the caller resolved one
                # from ``n_steps`` (or the legacy ``max_steps`` alias). Recomputing
                # ``int(duration * control_frequency)`` from the float ``duration =
                # n_steps / control_frequency`` truncates on any frequency that does
                # not divide evenly (e.g. n_steps=1 @ 49 Hz -> 0 steps reported as
                # success). Forwarding the count verbatim keeps the horizon exact.
                if n_steps is not None:
                    total_steps = n_steps
                else:
                    total_steps = int(duration * control_frequency)
                # Pace the loop on a DEADLINE, not a delay. ``time.sleep(1 /
                # control_frequency)`` after each step ADDS that step's work -
                # inference, the physics substeps, a render for the video, the
                # recorder's frame write - to the period instead of subtracting it,
                # so the loop runs at ``1 / (period + work)``. Measured on a MuJoCo
                # so101 rollout asking for 2.0s: 2.15s with a free policy, 3.15s
                # with 10ms of inference at 50Hz, and 3.90s at 30Hz with 30ms of
                # inference - 15.4Hz achieved where 30Hz was asked for. ``duration``
                # is documented as wall-clock seconds and ``fast_mode=False`` as
                # real-time pacing, so both claims were false by the cost of a step.
                # Ticker also DROPS missed deadlines rather than chasing them, so one
                # slow step does not fire a burst of back-to-back actions at the arm.
                #
                # Imported here rather than at module scope: importing any
                # ``strands_robots.mesh`` submodule executes the mesh package
                # ``__init__``, which pulls the fleet stack (measured: 14 mesh
                # modules plus boto3) - a cost a local rollout must not pay for a
                # pure-timing helper. Same reason as the local imports in the MuJoCo
                # backend and ``TeleopMixin._teleop_apply_loop``.
                if not fast_mode:
                    from strands_robots.mesh.pacing import Ticker as _Ticker

                    ticker = pacing_resources.enter_context(_Ticker(1.0 / control_frequency))

                # Control-rate substepping: a position-servo robot needs the physics
                # to advance for the FULL control period (1/control_frequency) after
                # each action so the joints actually track the commanded target
                # before the next action overwrites ``ctrl``. With the default
                # 1 substep/action, the arm only integrates one physics dt (~2 ms)
                # per action and barely moves - the policy looks like a no-op even
                # though it is sending valid targets. Derive substeps from the
                # backend's physics timestep; fall back to 1 when unknown.
                # Single source of truth for the derivation AND for the
                # positive-integer contract on an explicit override: an inline copy
                # here drifted from the shared helper the eval paths use.
                n_substeps = self._control_substeps(control_frequency, control_substeps)
                logger.info(
                    "PolicyRunner: control_frequency=%.1f Hz, physics substeps/action=%d",
                    control_frequency,
                    n_substeps,
                )
                _action_errors = 0  # count send_action failures (unresolved keys)
                # Per-actuator resolution tracking (issue #165). Init a counter to 0
                # for EVERY robot actuator so a never-driven joint surfaces as
                # resolution_rate 0.0 in the result rather than being absent from the
                # map. ``_total_failure_steps`` counts steps where the policy emitted
                # keys but NONE resolved (100% failure) -- the fail-fast trigger.
                try:
                    _robot_actuators = list(self.sim.robot_action_keys(robot_name))
                except Exception:  # noqa: BLE001 - stats are best-effort, never fatal
                    _robot_actuators = []
                _actuator_resolved: dict[str, int] = dict.fromkeys(_robot_actuators, 0)
                _total_failure_steps = 0
                _last_unresolved: list[str] = []

                onframe_failure_limit = (
                    max_onframe_failures if max_onframe_failures is not None else _MAX_CONSECUTIVE_ONFRAME_FAILURES
                )
                consecutive_onframe_failures = 0

                # Per-action execution body shared by BOTH the synchronous loop and
                # the async-RTC pipeline so they send, record, count and pace
                # identically - only the chunk-ACQUISITION strategy differs between
                # the two paths.
                def _apply(observation: dict[str, Any], action_dict: dict[str, Any]) -> None:
                    nonlocal step_count, _action_errors, consecutive_onframe_failures
                    nonlocal _total_failure_steps, _last_unresolved

                    _send_result = self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                    _is_error = isinstance(_send_result, dict) and _send_result.get("status") == "error"
                    # Resolve which of the robot's actuators this step actually drove
                    # and which emitted keys no actuator could absorb. On the success
                    # path send_action returns no json block, so every emitted key
                    # resolved; on the error path the block enumerates applied /
                    # unresolved keys.
                    _unresolved: list[str] = []
                    if _is_error:
                        _action_errors += 1
                        _json = _extract_result_json(_send_result)
                        if _json is not None:
                            _unresolved = list(_json.get("unresolved_keys", []))
                            _applied = list(_json.get("applied", []))
                        else:
                            # Error without a per-key breakdown (e.g. missing world,
                            # vector length mismatch): treat the whole step as a
                            # 100% failure so it counts toward the fail-fast probe.
                            _applied = []
                            if isinstance(action_dict, dict):
                                _unresolved = list(action_dict.keys())
                    elif isinstance(action_dict, dict):
                        _applied = list(action_dict)
                    else:
                        # A numeric vector binds positionally to every joint.
                        _applied = list(_robot_actuators)
                    for _name in _applied:
                        if _name in _actuator_resolved:
                            _actuator_resolved[_name] += 1
                    # A step is a 100%-failure when the policy emitted keys but NONE
                    # resolved to an actuator (the robot did not move at all). A
                    # PARTIAL failure (some keys resolve) is operational and runs to
                    # completion -- reported via partial_action_failure_rate.
                    if _is_error and not _applied:
                        _total_failure_steps += 1
                        if _unresolved:
                            _last_unresolved = _unresolved

                    if on_frame is not None:
                        try:
                            on_frame(step_count, observation, action_dict)
                            consecutive_onframe_failures = 0
                        except CooperativeStop:
                            # Backend (e.g. MuJoCo) signalled a graceful stop.
                            raise
                        except RecordingFrameError:
                            # A frame the dataset recorder could not write is data
                            # loss, not telemetry: the episode on disk is already
                            # shorter than this rollout. Never counted against the
                            # telemetry tolerance below, which resets on every
                            # success and so would let an intermittent recorder
                            # failure truncate the dataset without ever tripping.
                            raise
                        except Exception as e:
                            # on_frame is user-provided telemetry - never fatal
                            # *per call*. But if it fails on every step, a 500-
                            # step episode completes "successfully" with zero
                            # frames recorded and the dataset is silently empty.
                            # Count consecutive failures and fail the episode
                            # after ``onframe_failure_limit`` in a row. See GH #117.
                            consecutive_onframe_failures += 1
                            logger.warning(
                                "on_frame hook failed (%d/%d consecutive): %s",
                                consecutive_onframe_failures,
                                onframe_failure_limit,
                                e,
                            )
                            if consecutive_onframe_failures >= onframe_failure_limit:
                                raise RuntimeError(
                                    f"on_frame hook failed {onframe_failure_limit} times in a row; "
                                    f"aborting episode to avoid silent dataset corruption. "
                                    f"Last error: {e!r}"
                                ) from e

                    step_count += 1

                    # Fail fast: if EVERY step of the opening probe window drove zero
                    # actuators, the policy's output keys cannot match this robot, so
                    # the rollout is structurally dead -- raise now instead of running
                    # the remaining steps (and inference / recording I/O). Once any
                    # step resolves a key, _total_failure_steps < step_count forever
                    # and this never fires.
                    if step_count >= _FAIL_FAST_PROBE_STEPS and _total_failure_steps == step_count:
                        try:
                            _valid = self.sim.robot_action_keys(robot_name)
                        except Exception:  # noqa: BLE001
                            _valid = _robot_actuators
                        raise RuntimeError(
                            f"All of the first {step_count} action steps had 100% "
                            f"unresolved keys on '{robot_name}' -- the robot has not "
                            f"moved. Unresolved keys: {_last_unresolved}. Valid "
                            f"actuator/joint names: {_valid}. The policy is almost "
                            f"certainly running the wrong embodiment; inspect the "
                            f"expected keys via sim.get_features(robot_name="
                            f"'{robot_name}')."
                        )

                    if vwriter is not None:
                        vwriter.capture(step_count)

                    if ticker is not None:
                        # ``wait()`` returns the stop verdict of the event a Ticker
                        # was given; this one has none - the runner's cooperative
                        # stop arrives as an exception out of the on_frame hook
                        # above - so there is no verdict here to read.
                        ticker.wait()

                def _stop_when_fired() -> bool:
                    """Evaluate the caller's ``stop_when`` clause against the live sim.

                    Called after every applied action on BOTH the synchronous and
                    async-RTC paths, so the early-return latency bound is ONE
                    control step regardless of chunk length: the check fires
                    within the current chunk-slice and the remaining actions of
                    the chunk are dropped. Call sites guard on ``stop_when is not
                    None`` so the no-clause hot path pays no per-step call. A
                    raising clause is fatal - the caller asked for early-return
                    semantics the runner can no longer honor, and silently
                    running to the step budget would misreport the rollout - so
                    it surfaces as ``status="error"`` via the outer handler
                    rather than being warn-and-continued.
                    """
                    nonlocal stop_predicate_fired
                    assert stop_when is not None  # call sites hoist the None guard
                    try:
                        fired = bool(stop_when(self.sim))
                    except Exception as e:
                        raise RuntimeError(
                            f"stop_when predicate raised at step {step_count}: {e!r}. The early-return "
                            "condition cannot be evaluated, so the rollout is aborted rather than "
                            "silently running to its step budget."
                        ) from e
                    if fired:
                        stop_predicate_fired = True
                        logger.info("stop_when fired at step %d; ending rollout early", step_count)
                    return fired

                def _query_chunk(observation: dict[str, Any], observed_delay: int = 0) -> list[dict[str, Any]]:
                    # Resolve ONE action chunk from the policy. Never truncate below
                    # the policy's own intended chunk size: a model trained for
                    # N-step open-loop replay (policy.actions_per_step == N) must
                    # have its full chunk consumed; clamping to a smaller
                    # action_horizon drops the tail of every chunk and forces an
                    # out-of-distribution re-query (see LerobotLocalPolicy
                    # auto-detect of config.n_action_steps).
                    #
                    # Tell the policy how many control steps elapse between this
                    # observation and the first application of the returned chunk so
                    # latency-sensitive providers (RTC) slice the chunk-seam by an
                    # EXACT integer instead of a non-reproducible wall-clock
                    # estimate. The synchronous loop pauses the world during
                    # inference (delay 0); the async pipeline supplies the count of
                    # still-pending steps of the chunk currently executing. The set
                    # and the get_actions call happen on the SAME thread (the worker
                    # for a prefetch, the main thread otherwise), and at most one
                    # inference is ever in flight, so this never races.
                    policy.set_rtc_observed_delay(observed_delay)
                    _t_infer = time.perf_counter()
                    coro_or_result = policy.get_actions(observation, instruction, **_policy_kwargs)
                    actions = _resolve_coroutine(coro_or_result)
                    # Record inference wall-time (ms) for both the sync and async
                    # paths. Under async this runs on the prefetch worker; list
                    # append is atomic under the GIL so the read after
                    # shutdown(wait=True) sees every entry.
                    inference_ms.append((time.perf_counter() - _t_infer) * 1000.0)
                    _chunk = resolve_chunk_length(policy, action_horizon)
                    return list(actions[:_chunk])

                if async_rtc:
                    # Async chunk pipeline: overlap inference for chunk N+1 with the
                    # EXECUTION of chunk N. While the current chunk drains we fire
                    # the next get_actions() on a single background worker using a
                    # mid-execution ("horizon-shifted") observation, then atomically
                    # swap it in when the current chunk runs out. A policy whose
                    # inference latency is <= the chunk's execution time pays
                    # (almost) zero visible stall at the seam - exactly how an async
                    # real-time controller hides latency on real hardware. RTC
                    # policies blend the seam internally via their own prev-chunk
                    # state, so the runner only schedules the overlap (it never
                    # touches the policy's RTC machinery). The policy is invoked from
                    # AT MOST one thread at a time (a new prefetch is only submitted
                    # after the previous one has been consumed), and the sim is only
                    # ever touched from THIS thread, so there is no MuJoCo data race.
                    from concurrent.futures import Future, ThreadPoolExecutor
                    from concurrent.futures import TimeoutError as FuturesTimeout

                    def _swap_in(fut: Future[list[dict[str, Any]]]) -> list[dict[str, Any]]:
                        # Block on the prefetched chunk at the seam. A prefetch HIT
                        # means inference already finished (the seam is invisible); a
                        # BLOCK means we still have to wait because inference ran
                        # slower than the chunk's execution - the seam was starved,
                        # which is the actionable "tune prefetch_trigger / shorten
                        # the chunk" signal, so log it. A hard timeout turns a stuck
                        # model into a structured error instead of an unbounded sim
                        # hang.
                        nonlocal rtc_prefetch_hits, rtc_prefetch_blocks
                        if fut.done():
                            rtc_prefetch_hits += 1
                        else:
                            rtc_prefetch_blocks += 1
                            logger.warning(
                                "async-RTC seam starvation: prefetched chunk was not ready at the "
                                "swap point (inference slower than chunk execution). Blocking on it; "
                                "consider a shorter chunk or an earlier prefetch_trigger."
                            )
                        try:
                            return fut.result(timeout=rtc_inference_timeout_s)
                        except FuturesTimeout as e:
                            raise RuntimeError(
                                f"async-RTC prefetch exceeded rtc_inference_timeout_s="
                                f"{rtc_inference_timeout_s}s; policy inference is stuck. Raise the "
                                f"timeout or check the policy/server."
                            ) from e

                    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="rtc-prefetch")
                    try:
                        cur_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                        cur_chunk = _query_chunk(cur_obs)
                        rtc_chunks_acquired += 1
                        if not cur_chunk:
                            raise RuntimeError("policy returned an empty action chunk; cannot run rollout")
                        idx = 0
                        prefetch_trigger = max(1, len(cur_chunk) // 2)
                        prefetch: Future[list[dict[str, Any]]] | None = None
                        prefetch_obs: dict[str, Any] | None = None

                        while step_count < total_steps:
                            if idx >= len(cur_chunk):
                                # Current chunk drained -> swap in the next chunk.
                                if prefetch is not None:
                                    cur_chunk = _swap_in(prefetch)
                                    if prefetch_obs is not None:
                                        cur_obs = prefetch_obs
                                    prefetch = None
                                    prefetch_obs = None
                                else:
                                    # Chunk was too short to trigger a prefetch;
                                    # fall back to a synchronous re-query.
                                    cur_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                                    cur_chunk = _query_chunk(cur_obs)
                                rtc_chunks_acquired += 1
                                if not cur_chunk:
                                    # Drop-and-requery: a prefetched chunk arriving
                                    # empty (a transient policy hiccup) degrades to
                                    # ONE synchronous re-query before we give up,
                                    # rather than killing an otherwise-healthy
                                    # rollout on a single empty result.
                                    logger.warning(
                                        "async-RTC chunk arrived empty; falling back to one "
                                        "synchronous re-query before erroring."
                                    )
                                    cur_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                                    cur_chunk = _query_chunk(cur_obs)
                                    rtc_chunks_acquired += 1
                                    if not cur_chunk:
                                        raise RuntimeError(
                                            "policy returned an empty action chunk twice (prefetch + "
                                            "synchronous re-query); cannot continue rollout"
                                        )
                                idx = 0
                                prefetch_trigger = max(1, len(cur_chunk) // 2)
                                continue

                            # Fire the next inference once we are ~50% through the
                            # current chunk, on a fresh mid-chunk observation.
                            if prefetch is None and idx >= prefetch_trigger:
                                prefetch_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                                # The prefetched chunk first applies after the
                                # remaining steps of the current chunk drain - a
                                # known integer, independent of how long inference
                                # actually takes in wall-clock time (a slow inference
                                # just stalls the loop; the robot does not advance
                                # past the chunk end while waiting).
                                observed_delay = max(0, len(cur_chunk) - prefetch_trigger)
                                prefetch = executor.submit(_query_chunk, prefetch_obs, observed_delay)

                            # When recording, the chunk observation (the initial
                            # query obs, or a horizon-shifted prefetch obs after a
                            # swap) is stale for the step being applied; refresh it
                            # so the recorded frame is time-aligned (see the
                            # _record_per_step_obs note above). Inference is
                            # unaffected - it already consumed cur_obs to produce
                            # this chunk.
                            if _record_per_step_obs:
                                step_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                            else:
                                step_obs = cur_obs
                            _apply(step_obs, cur_chunk[idx])
                            idx += 1
                            # Semantic early return: checked after EVERY applied
                            # action, so the stop lands within one control step of
                            # the world reaching the condition - the rest of the
                            # in-flight chunk (and any prefetched chunk) is
                            # dropped; the executor shutdown below joins the
                            # in-flight prefetch worker. The None guard is hoisted
                            # so the no-clause hot path pays no per-step call.
                            if stop_when is not None and _stop_when_fired():
                                break
                    finally:
                        # Wait for any in-flight inference so no background thread
                        # touches the policy/sim after run() returns (the caller may
                        # immediately reset() or destroy() the world).
                        executor.shutdown(wait=True)
                else:
                    while step_count < total_steps:
                        observation = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                        chunk = _query_chunk(observation)
                        for chunk_idx, action_dict in enumerate(chunk):
                            if step_count >= total_steps:
                                break
                            # The chunk-start observation is the correct pre-action
                            # state for the first action only. When recording,
                            # refresh it before each SUBSEQUENT action so the
                            # recorded frame is time-aligned (see the
                            # _record_per_step_obs note above). chunk_idx == 0 reuses
                            # the freshly-queried observation (no re-render, sim has
                            # not stepped yet). Inference is unaffected.
                            if _record_per_step_obs and chunk_idx > 0:
                                step_obs = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                            else:
                                step_obs = observation
                            _apply(step_obs, action_dict)
                            # Semantic early return: checked after EVERY applied
                            # action (same cadence as the benchmark eval loop), so
                            # the remaining actions of the chunk are dropped as
                            # soon as the condition holds. The None guard is
                            # hoisted so the no-clause hot path pays no per-step
                            # call.
                            if stop_when is not None and _stop_when_fired():
                                break
                        if stop_predicate_fired:
                            break

            except CooperativeStop:
                stopped_early = True
                stopped_reason = "cancelled"
            except Exception as e:
                if vwriter is not None:
                    vwriter.close()
                logger.exception("PolicyRunner.run failed")
                return {
                    "status": "error",
                    "content": [
                        {"text": f"Policy failed: {e}"},
                        {"json": {**_rtc_telemetry(), "stopped_reason": "error", "steps_used": step_count}},
                    ],
                }

        # Either finished all steps, hit the stop_when condition, or was
        # cooperatively stopped.
        if stop_predicate_fired:
            stopped_early = True
            stopped_reason = "predicate"
        elapsed = time.monotonic() - start_mono
        sim_time = self._maybe_sim_time()
        if not stopped_early:
            prefix = "Policy complete"
        elif stopped_reason == "predicate":
            prefix = "Policy stopped early (stop_when condition met)"
        else:
            prefix = "Policy stopped"
        text = (
            f"{prefix} on '{robot_name}'\n{type(policy).__name__} | {instruction}\n{elapsed:.1f}s | {step_count} steps"
        )
        if sim_time is not None:
            text += f" | sim_t={sim_time:.3f}s"
        if vwriter is not None:
            assert video is not None
            video_path = vwriter.path
            frame_count = vwriter.frame_count
            vwriter.close()
            if frame_count > 0 and os.path.exists(video_path):
                file_kb = os.path.getsize(video_path) / 1024
                text += (
                    f"\nVideo: {video_path}\n"
                    f"{frame_count} frames, {video.fps}fps, "
                    f"{video.width}x{video.height} | {file_kb:.0f} KB"
                )
            else:
                # Log a loud warning so the user isn't blindsided by a silent
                # 0-byte MP4. We already pre-validate the camera name up-front,
                # so hitting this branch means frames failed DURING the rollout
                # (e.g. the camera was removed mid-episode).
                logger.warning(
                    "video recording requested but wrote 0 frames to %s - "
                    "MP4 file will be empty or absent. Check that the camera "
                    "remained valid throughout the rollout.",
                    video_path,
                )
                text += f"\nVideo requested but 0 frames captured ({video_path})"
        # Agent-consumable structured payload mirroring eval_policy()'s
        # ``{"json": {...}}`` block. Without this an agent driving run_policy has
        # to regex-parse the human-readable text to learn how many steps ran,
        # whether a video was written, or whether the rollout actually moved the
        # robot -- brittle and a documented AX friction point. The text block is
        # retained verbatim for humans; the json block carries the same facts as
        # typed fields for programmatic self-correction (deploy -> observe ->
        # re-tune loops). Keys are stable: callers can rely on them.
        payload: dict[str, Any] = {
            "robot_name": robot_name,
            "policy": type(policy).__name__,
            "instruction": instruction,
            "n_steps": step_count,
            # Alias of n_steps under the retry-loop name: the control steps
            # actually executed before the rollout ended. Paired with
            # stopped_reason it makes "the predicate fired after 37 of 200
            # steps" queryable without arithmetic on the caller side.
            "steps_used": step_count,
            "elapsed_s": round(elapsed, 3),
            "stopped_early": stopped_early,
            "stopped_reason": stopped_reason,
            "action_errors": _action_errors,
            "video_path": None,
            "video_frames": 0,
            # Load telemetry of the policy that drove this rollout. For
            # LerobotLocalPolicy these reflect the process-level model cache:
            # policy_load_cache_hit=False on episode 2+ of a loop is a smell
            # that the caller rebuilt the policy instead of reusing
            # policy_object=. Defaults (0.0 / False) cover policies that expose
            # no load telemetry (e.g. MockPolicy).
            "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
            "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
            # Routing-degradation telemetry from the driving policy. True means
            # the heuristic remap silently degraded the run: a camera routed to
            # a model image slot positionally (no name/camera_key_map match), or
            # observation.state composed from the observation's own scalar keys
            # because none of robot_state_keys matched (generic joint_0..N). The
            # robot then moves on meaningless inputs while status stays "success",
            # so the signal must be machine-readable, not only a log line.
            # Defaults False for policies without the attribute (e.g. MockPolicy).
            "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
            "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
            "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
            # Process RSS (MB) at result time: confirms a heavy model is resident
            # and, across a loop, that it stays resident instead of oscillating
            # as it would on a per-episode reload. None when unmeasurable.
            "policy_resident_rss_mb": process_rss_mb(),
        }
        if sim_time is not None:
            payload["sim_time_s"] = round(sim_time, 3)
        if vwriter is not None and video is not None:
            _vp = vwriter.path
            wrote_video = vwriter.frame_count > 0 and os.path.exists(_vp)
            payload["video_path"] = _vp if wrote_video else None
            payload["video_frames"] = vwriter.frame_count
        payload.update(_rtc_telemetry())

        # Per-actuator resolution stats (issue #165): the fraction of steps each
        # of the robot's actuators was actually driven. A joint stuck at 0.0
        # means the policy never produced a key that resolved to it (wrong name /
        # missing DOF), so a caller can see exactly which actuators the policy is
        # and is not driving -- not just a single aggregate error count.
        if step_count > 0 and _robot_actuators:
            action_resolution_rate = {
                name: round(_actuator_resolved.get(name, 0) / step_count, 4) for name in _robot_actuators
            }
            # Aggregate: the mean fraction of the robot's DOF NOT driven across
            # the rollout. 0.0 == every actuator driven every step; ~0.83 == only
            # 1 of 6 actuators ever moved. This is per-actuator coverage, distinct
            # from action_errors (a step-level status count): a policy that drives
            # 1 of 6 joints every step returns status=success with action_errors=0
            # yet a partial_action_failure_rate of ~0.83.
            _driven = sum(_actuator_resolved.get(n, 0) for n in _robot_actuators)
            partial_action_failure_rate = round(1.0 - _driven / (len(_robot_actuators) * step_count), 4)
            payload["action_resolution_rate"] = action_resolution_rate
            payload["partial_action_failure_rate"] = partial_action_failure_rate
            # Promote a high-but-not-total under-actuation to the human text so a
            # silently-crippled rollout (success_rate 0 because only 1 joint
            # moved) is not invisible. A total failure already errors above.
            if 0.5 < partial_action_failure_rate < 1.0:
                text += (
                    f"\n\nPartial action coverage: {partial_action_failure_rate:.0%} of this robot's "
                    f"actuators were never driven. Per-actuator resolution: {action_resolution_rate}."
                )
        else:
            payload["action_resolution_rate"] = {}
            payload["partial_action_failure_rate"] = 0.0

        # If EVERY step was a TOTAL failure (the policy emitted keys but none
        # resolved to an actuator), the robot never moved -- report this as an
        # error rather than a false success. This mirrors the fail-fast probe
        # and must key off ``_total_failure_steps``, NOT ``_action_errors``:
        # ``_action_errors`` also counts PARTIAL steps (some keys resolve, the
        # robot moves), so a policy that drives valid keys plus one extra
        # unresolved key every step (e.g. a 7-DOF-trained policy on a 6-DOF arm)
        # would otherwise be misreported as "the robot did not move". A partial
        # rollout is operational -- surfaced via partial_action_failure_rate.
        if _total_failure_steps >= step_count and step_count > 0:
            text += (
                f"\n\nALL {step_count} action steps had 100% unresolved keys "
                f"-- the robot did not move. Check that the policy's output keys "
                f"match the robot's actuator names."
            )
            # An error result always reports stopped_reason="error": the
            # rollout may have run its full budget, but the outcome is not a
            # retryable "budget" completion.
            payload["stopped_reason"] = "error"
            return {"status": "error", "content": [{"text": text}, {"json": payload}]}
        if _action_errors > 0:
            text += f"\n\n{_action_errors}/{step_count} action steps had unresolved keys."
        return {"status": "success", "content": [{"text": text}, {"json": payload}]}

    # replay(): replay a LeRobotDataset episode

    def replay(
        self,
        repo_id: str,
        robot_name: str | None = None,
        *,
        episode: int = 0,
        root: str | None = None,
        speed: float = 1.0,
        action_key_map: list[str] | None = None,
    ) -> dict[str, Any]:
        """Replay a recorded LeRobotDataset episode through ``send_action``.

        Each recorded frame is one control step taken at the dataset's fps, so
        replay advances physics for a full control period per frame (derived
        from the dataset fps and physics timestep, the same integration the
        recording used) rather than a single physics dt. This lets a
        position-servo robot track the recorded targets and reproduce the
        recorded trajectory; ``speed`` scales only the wall-clock playback
        rate, not the physics per frame.

        Args:
            repo_id: HuggingFace dataset id (e.g. ``lerobot/pusht``).
            robot_name: Target robot. Defaults to the first robot in the sim
                when omitted; an explicit name not present in the sim is
                rejected with a structured error (no silent replay onto a
                non-existent robot).
            episode: Episode index in the dataset. Must be a non-negative
                whole number; any real scalar with an integral value is
                accepted (including a NumPy scalar such as ``np.int64(2)``),
                and a bool, a non-integral value, a non-finite value or a
                non-numeric one is rejected with a structured error before the
                dataset is downloaded. Refused rather than coerced because the
                index selects which trajectory reaches the actuators.
            root: Optional local dataset root override.
            speed: Playback speed multiplier (1.0 = real time). Must be a
                positive, finite number (any real scalar, including a NumPy
                scalar such as ``np.float32(2.0)``); a non-positive,
                non-finite or non-numeric value is rejected with a structured
                error.
            action_key_map: Optional list of action keys, one per action
                vector index. Required when dataset action ordering differs
                from ``robot_action_keys(robot_name)``. If ``None``, positional
                mapping to ``robot_action_keys`` is used - the robot's
                *actuator* keys, which is the ordering the LeRobotDataset
                recorder writes the ``action`` column in (a robot's actuators
                are not always its joints; see :meth:`SimEngine.robot_action_keys`).
                Must be a non-empty list/tuple of unique strings; a bare
                string, a non-string entry or a duplicate key is rejected with
                a structured error. Its length must equal the recorded action
                vector's width - a mismatch is rejected rather than
                positionally truncated.

        Returns:
            Standard status dict with per-frame stats. Replay aborts with an
            ``"error"`` status when a recorded frame cannot actually be applied
            (unresolvable action keys, or a recorded vector whose width does not
            match the action-key map), reporting how many frames were applied
            before the abort. A successful status therefore means every frame
            reached the actuators.
        """
        # ``speed`` is a playback-rate multiplier used as the divisor in
        # ``frame_interval = 1 / (dataset_fps * speed)`` and, once computed,
        # flows into ``time.sleep(frame_interval - elapsed)`` on the real-time
        # playback path. A value of 0 raised a bare ZeroDivisionError (breaking
        # the documented "returns a status dict" contract) and a negative value
        # silently played the episode forward at full speed while reporting
        # success with a meaningless "Speed: -1.0x". Reject a non-positive or
        # non-numeric speed up front, before the (potentially multi-minute)
        # dataset download. Accept any real scalar (``numbers.Real``) so a
        # NumPy-scalar speed such as ``np.float32(2.0)`` or ``np.int64(2)``
        # (e.g. read from a config array) is not rejected:
        # ``isinstance(x, (int, float))`` is ``False`` for every NumPy scalar
        # except ``np.float64``. ``bool`` is still rejected explicitly (an
        # ``int`` subclass, so ``True`` would act as a silent 1.0x), and
        # non-finite values (``nan``/``inf``) are rejected via ``math.isfinite``
        # before the ``<= 0`` comparison so a ``nan`` -- which is never
        # ``<= 0`` -- cannot slip through into the ``1 / (fps * speed)``
        # arithmetic and reach ``time.sleep(nan)``. Mirrors the ``numbers.Real``
        # + finiteness contract applied to ``control_frequency`` and
        # ``add_camera(fov=...)``.
        if (
            isinstance(speed, bool)
            or not isinstance(speed, numbers.Real)
            or not math.isfinite(float(speed))
            or float(speed) <= 0
        ):
            return {
                "status": "error",
                "content": [{"text": f"replay: speed must be a positive number (got {speed!r})."}],
            }
        # Coerce to a plain Python float: a validated NumPy scalar still flows
        # into ``time.sleep(frame_interval - ...)`` and the returned
        # ``"speed": speed`` status field, where a ``numpy.float32`` raises a
        # bare "object cannot be interpreted as an integer" TypeError in
        # ``time.sleep`` and is not natively JSON-serialisable.
        speed = float(speed)

        # ``action_key_map`` binds recorded action-vector indices to action keys
        # ``send_action`` must resolve. A malformed map cannot be honored by any
        # backend, and pre-fix nothing checked its shape: a bare string was
        # ``list()``-ed into one key PER CHARACTER, a duplicate key made two
        # recorded indices write the same actuator (the later one silently
        # winning), and neither showed up in the result. Reject the shape here,
        # before the (potentially multi-minute) dataset download.
        key_map_error = _validate_action_key_map(action_key_map)
        if key_map_error is not None:
            return key_map_error

        # ``episode`` selects WHICH recorded episode is replayed, so an
        # unusable index is not a slow replay - it is the wrong trajectory
        # sent to a real position-servo robot. It reached
        # ``load_lerobot_episode``'s guard as a bare ``< 0`` test, which a
        # bool passes: ``replay(episode=True)`` resolved episode 1 and
        # replayed it under ``status="success"``. Validated here, with
        # ``speed`` and ``action_key_map``, for the two reasons those are:
        # the refusal arrives through this method's documented envelope
        # rather than as a raise, and it lands before the (potentially
        # multi-minute) dataset download. The rule is the shared one the
        # neighbouring ``replay_episode`` teleop knob already applies, so
        # the refusal is identical across both spellings of the parameter
        # rather than merely equivalent in verdict.
        if msg := non_negative_whole_number_error(episode, "episode", "replay"):
            return {"status": "error", "content": [{"text": msg}]}
        # Coerced for the same reason ``speed`` is: the accepted value flows
        # into ``load_lerobot_episode`` and the returned status field, and a
        # NumPy scalar is not natively JSON-serialisable.
        episode = int(episode)

        try:
            from strands_robots.dataset_recorder import load_lerobot_episode
        except ImportError:
            return {"status": "error", "content": [{"text": "lerobot not installed"}]}

        try:
            resolved_robot = robot_name or self._require_default_robot()
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # Validate the target robot is actually in the sim before applying any
        # actions. Without this an explicit ``robot_name`` that does not exist
        # silently "replays" onto a phantom robot (send_action no-ops), mirroring
        # neither run_policy nor eval_policy, both of which reject unknown robots.
        robots = self.sim.list_robots()
        if resolved_robot not in robots:
            return {
                "status": "error",
                "content": [{"text": f"Robot '{resolved_robot}' not found in sim. Available robots: {robots}"}],
            }

        try:
            ds, episode_start, episode_length = load_lerobot_episode(repo_id, episode, root)
        except Exception as e:  # noqa: BLE001 - library errors are opaque
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # Resolve the action-key ordering for action-vector index -> action
        # dict. The recorded ``action`` column is written in the robot's
        # *actuator* order (SimEngine.robot_action_keys), which diverges from
        # robot_joint_names whenever a robot has passive/mimic joints with no
        # driving actuator or a tendon-driven gripper. Mapping the recorded
        # vector back onto joint names there shifts/drops the recorded values
        # (send_action cannot resolve passive-joint names) while replay still
        # reports success - a silent round-trip corruption. Bind to the same
        # actuator keys the recorder used so record -> replay round-trips.
        action_keys = list(action_key_map) if action_key_map else self.sim.robot_action_keys(resolved_robot)

        dataset_fps = getattr(ds, "fps", 30)
        frame_interval = 1.0 / (dataset_fps * speed)
        # Step a FULL control period per recorded frame, not a single physics
        # dt. The recorded control frequency IS the dataset fps, so derive the
        # physics substeps from it (same convention as run() and evaluate()).
        # Without this, replay fell through to ``send_action``'s default
        # ``n_substeps=1`` - a single ~2 ms physics step per recorded frame -
        # while the recording integrated a full ~1/fps control period per
        # frame. A position-servo robot could not track the recorded targets in
        # ~2 ms, so replay produced a heavily under-integrated, attenuated
        # trajectory that did NOT reproduce the recording (the arm barely moved)
        # while still reporting ``Frames: N/N`` and ``status="success"`` - a
        # silent record -> replay fidelity gap. ``speed`` scales only the
        # wall-clock playback rate (frame_interval), never the physics per
        # frame, so it is deliberately excluded here.
        n_substeps = self._control_substeps(dataset_fps)
        frames_applied = 0
        # The replayed episode's own duration, on the same clock as the pacer
        # below for the same reason: it is measured, not recorded.
        start_mono = time.monotonic()

        # Replay only consumes the recorded action vector, which lives in the
        # dataset's parquet column store. A real LeRobotDataset's __getitem__
        # decodes every camera's video for the frame - wasted work here (the
        # decoded frames are discarded), and it raises a raw exception when the
        # video decoder (torchcodec / pyav) is unavailable or an MP4 is
        # unreadable, breaking replay()'s documented "returns a status dict"
        # contract for a dataset whose actions are perfectly readable. Read
        # from ``ds.hf_dataset`` (columns only, no video decode) when present;
        # fall back to ``ds[idx]`` for dataset objects without a column store.
        frame_source: Any = ds
        hf_dataset = getattr(ds, "hf_dataset", None)
        if hf_dataset is not None:
            frame_source = hf_dataset

        for frame_idx in range(episode_length):
            # The frame pacer's base. This one decides rather than reports: the
            # sleep below is computed from it, so on ``time.time()`` a
            # wall-clock step mid-episode either stalled the replay for the size
            # of the step (backward: the subtraction goes negative and the sleep
            # becomes ``frame_interval + step``) or dropped the pacing entirely
            # and ran the remaining frames unthrottled (forward: the sleep goes
            # negative). Both left a real robot tracking recorded targets at the
            # wrong rate under ``status="success"``.
            step_start_mono = time.monotonic()
            try:
                frame = frame_source[episode_start + frame_idx]
            except Exception as e:  # noqa: BLE001 - decoder/library errors are opaque
                return {
                    "status": "error",
                    "content": [{"text": (f"Failed to read frame {episode_start + frame_idx} from '{repo_id}': {e}")}],
                }

            action_vals = frame.get("action") if isinstance(frame, dict) else None
            if action_vals is None:
                # No action at this index - advance physics one full control
                # period so the frame still occupies its recorded time slice.
                self.sim.step(n_steps=n_substeps)
                frames_applied += 1
            else:
                if hasattr(action_vals, "numpy"):
                    action_vals = action_vals.numpy()
                if hasattr(action_vals, "tolist"):
                    action_vals = action_vals.tolist()

                # A recorded vector whose width differs from the action-key
                # map cannot be replayed faithfully: the surplus values have no
                # key (silently DROPPED pre-fix, e.g. a 2-key map swallowing a
                # 6-DOF recording's last four joints) or the surplus keys never
                # receive a value. Reject it with the recorded-vs-expected
                # widths, mirroring how ``send_action`` rejects a raw action
                # vector whose length does not match the actuator count instead
                # of truncating it.
                if len(action_vals) != len(action_keys):
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Replay aborted at frame {frame_idx}: recorded action vector has "
                                    f"{len(action_vals)} values but {len(action_keys)} action keys are "
                                    f"mapped ({action_keys}). Applied {frames_applied}/{episode_length} "
                                    "frames. Pass an action_key_map with one key per recorded action "
                                    f"value, or replay onto a robot whose actuators match the recording."
                                )
                            },
                            {
                                "json": {
                                    "episode": episode,
                                    "robot_name": resolved_robot,
                                    "frame": frame_idx,
                                    "recorded_action_width": len(action_vals),
                                    "action_keys": action_keys,
                                    "frames_applied": frames_applied,
                                    "total_frames": episode_length,
                                }
                            },
                        ],
                    }

                action_dict: dict[str, Any] = {action_keys[i]: float(val) for i, val in enumerate(action_vals)}

                # ``send_action`` reports unresolvable keys as an "error" status
                # (with an ``unresolved_keys`` json block). Pre-fix that result
                # was DISCARDED, so a typo'd action_key_map dropped every value
                # at the actuator boundary while replay still reported
                # ``status="success"`` and ``Frames: N/N`` - the recorded
                # trajectory never reached the robot. Abort on the first
                # unapplied frame instead of finishing a replay that is not
                # happening.
                send_result = self.sim.send_action(action_dict, robot_name=resolved_robot, n_substeps=n_substeps)
                if isinstance(send_result, dict) and send_result.get("status") == "error":
                    detail = next(
                        (
                            str(block["text"])
                            for block in send_result.get("content", []) or []
                            if isinstance(block, dict) and "text" in block
                        ),
                        "",
                    )
                    payload: dict[str, Any] = {
                        "episode": episode,
                        "robot_name": resolved_robot,
                        "frame": frame_idx,
                        "action_keys": action_keys,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                    }
                    send_json = _extract_result_json(send_result)
                    if send_json is not None:
                        payload.update(send_json)
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Replay aborted at frame {frame_idx}: the recorded action could not be "
                                    f"applied to '{resolved_robot}'. Applied {frames_applied}/{episode_length} "
                                    f"frames. {detail}"
                                )
                            },
                            {"json": payload},
                        ],
                    }
                frames_applied += 1

            sleep_time = frame_interval - (time.monotonic() - step_start_mono)
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.monotonic() - start_mono
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Replayed episode {episode} from {repo_id} on '{resolved_robot}'\n"
                        f"Frames: {frames_applied}/{episode_length} | "
                        f"Duration: {duration:.1f}s | Speed: {speed}x"
                    )
                },
                {
                    "json": {
                        "episode": episode,
                        "robot_name": resolved_robot,
                        "frames_applied": frames_applied,
                        "total_frames": episode_length,
                        "duration_s": round(duration, 2),
                        "speed": speed,
                    }
                },
            ],
        }

    # evaluate(): multi-episode success metrics

    def evaluate(
        self,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str = "",
        n_episodes: int = 10,
        max_steps: int = 300,
        success_fn: SuccessFn | str | None = None,
        spec: BenchmarkProtocol | None = None,
        seed: int | None = None,
        action_horizon: int = 8,
        on_frame: OnFrame | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        async_rtc: bool = False,
        rtc_inference_timeout_s: float | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Evaluate ``policy`` for ``n_episodes`` episodes.

        Two evaluation paths:

        * **``spec=``** (preferred): drive a full :class:`BenchmarkProtocol`.
          Per-episode seeded RNG, ``on_episode_start`` / ``on_step`` /
          ``is_success`` / ``is_failure`` hooks, cumulative dense reward,
          robot-compatibility validation. ``max_steps`` from the spec wins.
        * **``success_fn=``**: legacy sparse-success path kept for
          backwards compatibility. Equivalent to a
          ``BenchmarkProtocol`` whose ``on_step`` always returns
          ``StepInfo(reward=0.0, done=False)``.

        Passing both ``spec`` and ``success_fn`` is an error - benchmarks
        define their own success predicate.

        Args:
            robot_name: Robot to evaluate.
            policy: Already-constructed ``Policy`` instance.
            instruction: Instruction forwarded to the policy.
            n_episodes: Number of reset → rollout episodes. Must be a
                positive integer, refused as in :meth:`run`: a bound outside
                that domain does not shorten the evaluation, it removes it
                while still reporting a ``success_rate`` over it.
            max_steps: Cap per episode. Ignored when ``spec`` is provided
                (``spec.max_steps`` wins), and validated on the same domain as
                ``n_episodes`` only when it is the horizon actually read - so a
                ``spec=`` call is not refused for a value it never reads.
            success_fn: Legacy success predicate (see above).
            spec: :class:`BenchmarkProtocol` to drive the eval. When
                provided, overrides the ``success_fn`` path.
            seed: Master RNG seed. Each episode derives a child RNG from it,
                so evaluations are reproducible within a process. Only used
                when ``spec`` is provided.
            action_horizon: Max actions consumed per policy call before
                requerying the observation, as in :meth:`run`. Clamped up to the
                policy's own chunk length when it emits more.
                Must be a positive integer, refused as in :meth:`run`.
            on_frame: Optional ``(step, observation, action) -> None`` hook
                fired per applied control step on the eval thread, after
                ``sim.send_action``. Forwarded on BOTH the ``spec=`` and the
                legacy ``success_fn`` paths; ``step`` is a monotonic index
                that continues across episode boundaries. A hook exception
                other than ``CooperativeStop`` or
                :class:`~strands_robots.dataset_recorder.RecordingFrameError` is
                logged at WARN and never aborts the eval; a
                ``RecordingFrameError`` is data loss rather than telemetry and
                propagates on the first occurrence. Raising
                :class:`CooperativeStop` stops the
                evaluation gracefully after the episodes completed so far
                (the result carries ``stopped_early=True`` and
                ``episodes_completed``), matching :meth:`run`. Use this for
                synchronous recording when the eval runs on a thread distinct
                from the script main (e.g. Strands ``Agent`` tool dispatch
                under asyncio) - see #191 and
                :meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording_synchronous`.
            control_frequency: Target Hz for ``policy.get_actions`` calls, as in
                :meth:`run`. Also sets the wall-clock period each applied action
                is integrated over, via the ``control_substeps`` derivation
                below, so an eval steps physics for the same period per action
                that :meth:`run` does.
            control_substeps: Physics steps advanced per applied action, with
                the same contract as :meth:`run`: ``None`` (default) derives the
                count from ``control_frequency`` and the backend's physics
                timestep, and an explicit value must be a positive integer or
                :class:`ValueError` is raised rather than the value being
                clamped. Passing ``1`` here is what made eval rollouts look
                like a policy no-op before the derivation existed.
            async_rtc: Opt-in overlap of policy inference with action-chunk
                execution on the legacy ``success_fn`` path, mirroring
                :meth:`run`. Defaults to ``False`` (synchronous): the world is
                paused during inference, so the success-rate is bit-stable and
                reproducible. Set ``True`` to evaluate a chunk-emitting policy
                under the realistic control latency it faces in deployment - a
                background worker computes chunk N+1 while chunk N drains, which
                feeds the policy a slightly staler (mid-chunk) observation at
                the seam and therefore can shift the measured success-rate (that
                is the point: it measures robustness to inference latency).
                ``True`` is rejected on the ``spec=`` benchmark path, which
                stays synchronous for bit-stable reproducibility; use
                :meth:`run` (``run_policy``) for benchmark-style latency masking.
            rtc_inference_timeout_s: Hard per-chunk timeout (seconds) for the
                async prefetch. When inference does not finish within the
                timeout the eval fails with a structured error instead of
                hanging the rollout. ``None`` waits indefinitely.
            policy_kwargs: Per-call goal payload forwarded verbatim to every
                ``policy.get_actions(obs, instruction, **policy_kwargs)`` call on
                both eval paths (``success_fn`` and ``spec``). Empty/``None`` is
                the historical no-kwargs behaviour. Goal-conditioned providers
                (WBC ``target_velocity``; cuRobo/MoveIt2 ``target_pose`` /
                ``target_joints`` / ``world_update`` - the issue #300 keys) need
                this to be evaluated against a goal at all.
            video: Optional per-episode MP4 recording config (same dict schema
                as :meth:`run` / ``run_policy``: ``path`` enables it, plus
                ``fps`` / ``camera`` / ``width`` / ``height``). One file per
                episode with ``_ep{i}`` inserted into the filename; the written
                paths are returned in the result json ``video_paths``. Recorded
                on BOTH eval routes: the ``success_fn`` path and the
                ``spec``/benchmark path (:meth:`_evaluate_with_spec`). Frames are
                captured synchronously on the eval thread at the ``on_frame``
                point (after ``send_action``), so recording never perturbs the
                bit-stable spec-path rollout.

        Returns:
            Standard status dict. The JSON payload carries an RTC telemetry
            block (``rtc_async_enabled``, ``rtc_chunks_acquired``,
            ``rtc_prefetch_hits``, ``rtc_prefetch_blocks``,
            ``rtc_avg_inference_ms``, ``rtc_max_inference_ms``) so inference
            cost and latency masking are provable from the payload. When
            ``spec`` is used, it also contains ``cumulative_reward`` and
            ``avg_reward`` fields per episode and aggregate.

            Every payload carries ``success_measured`` (bool): ``True`` when a
            success criterion was in force (a ``spec`` or a non-``None``
            ``success_fn``), ``False`` when neither was given. When ``False`` the
            reported ``success_rate`` is a hard ``0.0`` that measures nothing
            (no episode can be marked successful without a criterion), and a
            warning is logged - check this flag before trusting ``success_rate``.

            The payload also carries ``episodes_completed`` (episodes that ran
            to completion) and ``stopped_early`` (bool). When an ``on_frame``
            hook raises :class:`CooperativeStop`, the eval ends gracefully after
            the completed episodes: ``stopped_early=True`` and the aggregate
            metrics are computed over ``episodes_completed`` (which may be less
            than the requested ``n_episodes``).

            ``recording_save_error`` is the third way an evaluation can cover
            fewer episodes than asked for, and the only one that makes
            ``status`` ``"error"``: it is ``None`` on every healthy run, and
            the reason string when a per-episode dataset flush failed. See
            :meth:`_finalize_recorder_episode` for why a lost episode stops
            the evaluation instead of being averaged over.
        """
        # Refuse before any frame reaches the engine's open recording.
        self._reject_recording_rate_mismatch(control_frequency, "PolicyRunner.evaluate")
        # Local import: base.py imports PolicyRunner at module level, so
        # reaching the shared domain from here has to stay deferred - the
        # same convention this module already uses for
        # simulation.benchmark / simulation.recording / simulation.predicates.
        from strands_robots.simulation.base import MAX_EVAL_SEED, randomization_seed_error

        if seed_error := randomization_seed_error(seed, "PolicyRunner.evaluate", max_seed=MAX_EVAL_SEED):
            raise ValueError(seed_error)
        # Same shared domain the facade one layer up enforces, raised rather
        # than returned because raising is this layer's contract: PolicyRunner is
        # drivable directly and a direct caller has no envelope to read a refusal
        # from. A deadline outside the domain makes the seam swap's own
        # "policy inference is stuck" diagnosis false - see
        # SimEngine._validate_rtc_inference_timeout for the measured failure modes.
        if rtc_inference_timeout_s is not None and (
            timeout_error := positive_finite_number_error(
                rtc_inference_timeout_s, "rtc_inference_timeout_s", "PolicyRunner.evaluate"
            )
        ):
            raise ValueError(timeout_error)
        # Same shared domain, raised for the same reason as in run(): a horizon
        # outside it is clamped to 1 or leaks a bare conversion error from the
        # first inference. Checked before the spec delegation below so the
        # benchmark path cannot reach _evaluate_with_spec with a value the
        # entry point would have refused.
        if horizon_error := positive_count_error(action_horizon, "action_horizon", "PolicyRunner.evaluate"):
            raise ValueError(horizon_error)
        # The two bounds of this method's own episode loop, on the same shared
        # domain and raised for the same reason. A horizon outside the domain
        # degrades a rollout; a LOOP BOUND outside it removes the evaluation
        # while still reporting one. ``n_episodes=0`` returns status="success"
        # over zero episodes and ``max_steps=0`` over two episodes of zero
        # length - both with ``success_rate: 0.0`` and ``success_measured:
        # True``, the flag that exists so a 0.0 cannot be read as a
        # measurement, and with no action ever applied. ``max_steps=inf`` is
        # worse than degenerate: ``while steps < max_steps`` has no false case,
        # so the episode never ends. Refused here, before ``sim.reset()``,
        # ``set_eval_seed`` (which reseeds the process-global RNG) and the first
        # inference, so a rejected eval costs nothing and leaves no global side
        # effect. This is also what makes ``_evaluate_with_spec``'s claim that
        # "every other bound of this nested loop is checked by the public entry
        # point before it gets here" true for a direct caller of this method.
        if episodes_error := positive_count_error(n_episodes, "n_episodes", "PolicyRunner.evaluate"):
            raise ValueError(episodes_error)
        # ``max_steps`` is only read on the legacy ``success_fn`` path: the
        # ``spec=`` path takes its horizon off the benchmark
        # (``spec.max_steps``, checked at its read below) and this parameter is
        # not forwarded there at all, so refusing it for a ``spec=`` call would
        # reject a value that call never reads. Effectiveness is a property of
        # the request here - ``spec`` is a parameter of this signature - so the
        # check can be gated on it without guessing.
        if spec is None and (steps_error := positive_count_error(max_steps, "max_steps", "PolicyRunner.evaluate")):
            raise ValueError(steps_error)
        if spec is not None and success_fn is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "evaluate() accepts either 'spec' or 'success_fn', not both. "
                            "'spec' defines its own success predicate."
                        )
                    }
                ],
            }

        # Per-call goal payload forwarded verbatim to every get_actions() call
        # on both eval paths (success_fn + spec). An empty dict is the historical
        # (no-kwargs) behaviour. Goal-conditioned providers (WBC target_velocity,
        # cuRobo/MoveIt2 target_pose/target_joints, the issue #300 keys) need this
        # to be evaluated against a goal at all; without it eval ran them with an
        # empty goal and reported a meaningless success rate.
        _policy_kwargs = policy_kwargs or {}

        if async_rtc and spec is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "async_rtc is only supported on the success_fn eval path. "
                            "The spec/benchmark path stays synchronous for bit-stable "
                            "reproducibility; use run_policy(async_rtc=...) for "
                            "benchmark-style latency masking."
                        )
                    }
                ],
            }

        if spec is not None:
            return self._evaluate_with_spec(
                robot_name,
                policy,
                spec,
                instruction=instruction,
                n_episodes=n_episodes,
                seed=seed,
                action_horizon=action_horizon,
                on_frame=on_frame,
                control_frequency=control_frequency,
                control_substeps=control_substeps,
                policy_kwargs=_policy_kwargs,
                video=video,
            )

        try:
            resolved_check = self._resolve_success_fn(success_fn)
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # A None check means no success criterion was supplied (the
        # success_fn=None default with no spec). The loop then never sets
        # success=True, so success_rate reports a hard 0.0 for every
        # episode - indistinguishable from a policy that genuinely failed
        # every episode. Warn loudly and flag the payload so 0.0 is not
        # mistaken for a measurement (mirrors the entry-point guards that
        # reject other degenerate/fabricated success-rate configs).
        success_measured = resolved_check is not None
        if not success_measured:
            logger.warning(
                "evaluate()/eval_policy called without a success criterion "
                "(success_fn=None and no spec): success_rate will be 0.0 for "
                "every episode regardless of what the policy does and does "
                "NOT measure task success. Pass success_fn (e.g. 'contact' "
                "or a callable) or a benchmark spec to measure success; the "
                "returned json flags this as success_measured=false."
            )

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Named-body poses the policy declared it needs (mimic trackers read an
        # anchor link). Resolved once here so an unknown name fails before the
        # loop instead of being read as a zero pose on every tick; () for the
        # overwhelming majority of policies, which adds no backend call.
        _bodies = self._resolve_required_bodies(policy)
        # Step physics for the full control period per action, same derivation
        # as run(). The default n_substeps=1 made eval rollouts under-step.
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        policy.set_control_frequency(control_frequency)

        # Reproducibility for this path. ``seed`` reached exactly one statement
        # in this method - the ``_evaluate_with_spec`` delegation above - so the
        # loop below ran unseeded while reporting success, and every
        # ``eval_policy`` call lands here because that facade exposes no
        # ``spec``. A caller who asked for a reproducible eval got a different
        # rollout on every run, with ``policy.reset(seed=...)`` never forwarded -
        # the half a service-mode policy needs to reseed its own process.
        #
        # A ``None`` seed leaves the master RNG unbuilt rather than seeding it
        # from entropy: an unseeded eval must not acquire a global RNG side
        # effect it never had.
        master_rng = random.Random(seed) if seed is not None else None
        if seed is not None:
            set_eval_seed(seed)

        # RTC telemetry, reported in the result json so inference cost (and,
        # under async_rtc, latency masking) is provable without grepping logs.
        # inference_ms collects every get_actions wall-time on both paths; the
        # prefetch hit/block counters are async-only (0 on the synchronous path).
        inference_ms: list[float] = []
        rtc_chunks_acquired = 0
        rtc_prefetch_hits = 0
        rtc_prefetch_blocks = 0

        def _observation_fn() -> dict[str, Any]:
            return self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)

        def _query_chunk(observation: dict[str, Any], observed_delay: int = 0) -> list[dict[str, Any]]:
            # Tell latency-sensitive (RTC) policies how many control steps
            # elapse between this observation and the first application of the
            # returned chunk so they slice the chunk-seam by an EXACT integer
            # instead of a wall-clock estimate. The synchronous path pauses the
            # world during inference (delay 0); the async pipeline supplies the
            # count of still-pending steps of the chunk currently executing.
            policy.set_rtc_observed_delay(observed_delay)
            _t_infer = time.perf_counter()
            actions = _resolve_coroutine(policy.get_actions(observation, instruction, **_policy_kwargs))
            inference_ms.append((time.perf_counter() - _t_infer) * 1000.0)
            # resolve_chunk_length is the single source of truth for the
            # re-query interval (respects RTC + execution_horizon). Consuming the
            # FULL chunk before re-querying matches run() and _evaluate_with_spec
            # (#168); truncating to a smaller horizon would force an
            # out-of-distribution re-query of chunk-predicting VLAs.
            return list(actions[: resolve_chunk_length(policy, action_horizon)])

        results: list[dict[str, Any]] = []
        # #191 - monotonic global step index handed to ``on_frame`` so a
        # synchronous recorder/telemetry hook sees a continuous count across
        # episode boundaries, exactly like the spec eval path and ``run()``.
        global_step = 0

        # Optional per-episode rollout video (eval_policy video=). One MP4 per
        # episode via the _ep{i} filename templating that run_policy's
        # multi-episode loop already uses, so an eval can be watched to see WHY
        # episodes fail - not just read as an aggregate success_rate. The writer
        # is opened per episode below; _fire_on_frame appends a frame at the fps
        # cadence on the same synchronous eval-thread point as the on_frame hook.
        video_paths: list[str] = []
        current_vwriter: _RolloutVideoWriter | None = None

        def _fire_on_frame(obs: dict[str, Any], action: dict[str, Any], ep_step: int) -> None:
            # Fire AFTER ``send_action`` (post-action obs unavailable yet, so
            # pass the pre-action obs the chunk was queried with - matches
            # ``_evaluate_with_spec``). The hook is best-effort telemetry: a
            # GENERIC failure is logged at WARN and never aborts the eval. The
            # two classes handled below are not telemetry and are exempt from
            # that posture.
            nonlocal global_step
            if current_vwriter is not None:
                current_vwriter.capture(ep_step)
            if on_frame is not None:
                try:
                    on_frame(global_step, obs, action)
                except CooperativeStop:
                    # Documented graceful early-stop (the same signal run()
                    # honors). Propagate to the episode loop; never swallow
                    # it as a best-effort telemetry failure.
                    raise
                except RecordingFrameError:
                    # Data loss, not telemetry - see the note on the tolerance
                    # constant. Propagate so the caller learns the episode is
                    # incomplete instead of reading a successful eval.
                    raise
                except Exception as e:  # noqa: BLE001 - hook is best-effort telemetry
                    logger.warning("on_frame hook failed at global_step=%d: %s", global_step, e)
            global_step += 1

        stopped_early = False
        recording_save_error: str | None = None
        try:
            for ep in range(n_episodes):
                self.sim.reset()
                success = False
                steps = 0

                # Per-episode MP4 (foo_ep{i}.mp4). Validation + camera probe happen
                # here; a bad path/camera fails the eval up-front (on ep 0) instead
                # of running N episodes and writing nothing.
                ep_vcfg = self.sim._episode_video_config(video, ep)
                current_vwriter, _video_err = _RolloutVideoWriter.open(self.sim, ep_vcfg, control_frequency)
                if _video_err is not None:
                    return _video_err

                # Per-episode reseed, mirroring ``_evaluate_with_spec``: episode
                # N starts from the same RNG state regardless of what episodes
                # 0..N-1 drew, so a stochastic policy's sampling is stable across
                # re-runs at the same master seed. Forwarded to ``policy.reset``
                # too, because a service-mode policy samples in another process
                # that ``set_eval_seed`` cannot reach. Best-effort, like every
                # other ``reset`` call site.
                if master_rng is not None:
                    episode_seed = master_rng.randint(0, 2**31 - 1)
                    set_eval_seed(episode_seed)
                    try:
                        policy.reset(seed=episode_seed)
                    except Exception as e:  # noqa: BLE001 - reset is best-effort
                        logger.warning(
                            "policy.reset(seed=%d) raised %s; continuing without per-episode reset",
                            episode_seed,
                            e,
                        )

                if async_rtc:
                    # Opt-in async overlap: a single background worker computes the
                    # next chunk while the current one drains, so a chunk-emitting
                    # policy is evaluated under the realistic inference latency it
                    # faces in deployment. The pipeline only ever calls the policy
                    # off-thread; the sim is stepped solely here, so there is no
                    # data race. The context manager joins the worker on exit even
                    # when we break mid-chunk on success.
                    pipeline = _ChunkPipeline(
                        _query_chunk,
                        _observation_fn,
                        async_rtc=True,
                        rtc_inference_timeout_s=rtc_inference_timeout_s,
                    )
                    with pipeline as chunks:
                        for _observation, action_dict in chunks:
                            if steps >= max_steps:
                                break
                            self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                            _fire_on_frame(_observation, action_dict, steps)
                            steps += 1
                            # Check success against the LIVE post-action observation
                            # (mirrors the synchronous path / _evaluate_with_spec).
                            if resolved_check is not None and _criterion_verdict(
                                resolved_check, _observation_fn(), label="success_fn", episode=ep, step=steps
                            ):
                                success = True
                                break
                    rtc_chunks_acquired += pipeline.chunks_acquired
                    rtc_prefetch_hits += pipeline.prefetch_hits
                    rtc_prefetch_blocks += pipeline.prefetch_blocks
                else:
                    while steps < max_steps:
                        observation = _observation_fn()
                        chunk = _query_chunk(observation, 0)
                        rtc_chunks_acquired += 1

                        if not chunk:
                            # Policy returned nothing - still advance one physics
                            # step so episodes don't hang on degenerate policies,
                            # then check the post-step observation (same post-action
                            # semantics as the chunk branch below).
                            self.sim.step(n_steps=1)
                            steps += 1
                            if resolved_check is not None and _criterion_verdict(
                                resolved_check, _observation_fn(), label="success_fn", episode=ep, step=steps
                            ):
                                success = True
                                break
                            continue

                        for action_dict in chunk:
                            if steps >= max_steps:
                                break
                            self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                            _fire_on_frame(observation, action_dict, steps)
                            steps += 1
                            # Check success against the LIVE post-action observation,
                            # not the stale pre-action obs. Checking the pre-action
                            # obs detects success one step late and never records a
                            # task that completes on the final step -> under-reported
                            # success_rate / inflated avg_steps. Mirrors
                            # _evaluate_with_spec's post-send is_success.
                            if resolved_check is not None and _criterion_verdict(
                                resolved_check, _observation_fn(), label="success_fn", episode=ep, step=steps
                            ):
                                success = True
                                break
                        if success:
                            break

                results.append({"episode": ep, "steps": steps, "success": success})
                # #708 - roll the attached recorder over to a new episode so the
                # dataset records per-episode boundaries rather than collapsing
                # every rollout into one mega-episode.
                recording_save_error = self._finalize_recorder_episode()

                if current_vwriter is not None:
                    current_vwriter.close()
                    if current_vwriter.frame_count > 0 and os.path.exists(current_vwriter.path):
                        video_paths.append(current_vwriter.path)
                    else:
                        logger.warning(
                            "eval_policy episode %d: video requested but wrote 0 frames to %s",
                            ep,
                            current_vwriter.path,
                        )
                    current_vwriter = None

                if recording_save_error is not None:
                    # This episode's frames did not reach the dataset and the
                    # recorder is now closed, so every later episode would run
                    # into a recorder that drops frames without counting them.
                    # Stop here and report, rather than measure a success_rate
                    # over episodes whose data is gone. This episode's video is
                    # already closed and collected above, so it is kept.
                    recording_save_error = f"episode {ep}: {recording_save_error}"
                    break

        except CooperativeStop:
            # A user/backend on_frame hook requested a graceful stop (the
            # same signal run() honors). End the evaluation over the episodes
            # completed so far instead of crashing with an uncaught
            # BaseException. Close any in-progress episode video cleanly.
            stopped_early = True
            logger.info(
                "on_frame requested a cooperative stop; ending evaluation after %d completed episode(s)",
                len(results),
            )
            if current_vwriter is not None:
                current_vwriter.close()
                current_vwriter = None
        n_completed = len(results)
        n_success = sum(1 for r in results if r["success"])
        success_rate = n_success / max(n_completed, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_completed, 1)
        _n_infer = len(inference_ms)
        rtc_telemetry = {
            "rtc_async_enabled": bool(async_rtc),
            "rtc_chunks_acquired": rtc_chunks_acquired,
            "rtc_prefetch_hits": rtc_prefetch_hits,
            "rtc_prefetch_blocks": rtc_prefetch_blocks,
            "rtc_avg_inference_ms": round(sum(inference_ms) / _n_infer, 3) if _n_infer else 0.0,
            "rtc_max_inference_ms": round(max(inference_ms), 3) if _n_infer else 0.0,
        }

        return {
            "status": "error" if recording_save_error is not None else "success",
            "content": [
                {
                    "text": (
                        f"Evaluation: {type(policy).__name__} on '{robot_name}'\n"
                        + (
                            f"Stopped after a lost recording episode - {recording_save_error}\n"
                            if recording_save_error is not None
                            else ""
                        )
                        + f"Episodes: {n_completed}"
                        + (f" of {n_episodes} (stopped early)" if stopped_early else "")
                        + f" | Success: {n_success}/{n_completed} ({success_rate:.1%})"
                        + ("" if success_measured else " [no success criterion - not measured]")
                        + "\n"
                        f"Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "success_measured": success_measured,
                        "n_episodes": n_episodes,
                        "episodes_completed": n_completed,
                        "stopped_early": stopped_early,
                        "recording_save_error": recording_save_error,
                        "n_success": n_success,
                        "avg_steps": round(avg_steps, 1),
                        "max_steps": max_steps,
                        "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
                        "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
                        "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
                        "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
                        "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
                        **rtc_telemetry,
                        "policy_resident_rss_mb": process_rss_mb(),
                        "episodes": results,
                        "video_paths": video_paths,
                    }
                },
            ],
        }

    def _evaluate_with_spec(
        self,
        robot_name: str,
        policy: Policy,
        spec: BenchmarkProtocol,
        *,
        instruction: str,
        n_episodes: int,
        seed: int | None,
        action_horizon: int = 8,
        on_frame: OnFrame | None = None,
        control_frequency: float = 50.0,
        control_substeps: int | None = None,
        policy_kwargs: dict[str, Any] | None = None,
        video: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Drive a :class:`BenchmarkProtocol` for ``n_episodes`` episodes.

        Split out from :meth:`evaluate` to keep the legacy-path body small;
        both routes share the same return-dict schema plus the spec route
        layers on cumulative-reward accounting.

        Robot compatibility is validated before episode 1: if the sim's
        loaded robot declares a ``data_config`` not in
        ``spec.supported_robots`` (non-empty), we return a structured error
        with the allowed list instead of silently running a mismatched
        evaluation.

        ``on_frame`` (#191) fires per applied control step on the eval
        thread, after ``sim.send_action`` and after the spec's per-step
        bookkeeping (``on_step`` / success / failure checks). Use this
        for synchronous recording or telemetry that needs to read sim
        state on the eval thread to avoid the cross-thread ``mjData``
        race the daemon-thread recorder hits under multi-threaded
        eval (Strands ``Agent`` tool dispatch under asyncio). Failures
        are logged WARNING; the rollout continues. The hook receives a
        global step counter (across episodes), so callers that need
        per-episode buckets should track episode boundaries themselves.

        ``video`` (optional) records one rollout MP4 per episode (``_ep{i}``
        filename templating), captured synchronously on the eval thread at the
        same point as ``on_frame`` - render is read-only over ``mjData`` so it
        does not perturb the bit-stable spec-path rollout. Written paths are
        returned in the result json ``video_paths``.
        """
        # Lazy import to avoid circular reference (benchmark module imports
        # `SimEngine` from base which imports this module under TYPE_CHECKING).
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        # The per-episode horizon is read off the benchmark, so it is the one
        # rollout count with no parameter of its own to validate: every other
        # bound of this nested loop (``n_episodes``, ``action_horizon``,
        # ``control_substeps``) is checked by the public entry point before it
        # gets here. Check it at the read instead. That covers every way a
        # benchmark can come by its horizon - ``DeclarativeBenchmark.from_dict``,
        # direct construction, a plain ``BenchmarkProtocol`` subclass setting the
        # documented ``max_steps`` attribute, or an assignment to it after
        # construction - none of which this method can see. Refuse before
        # ``set_control_frequency`` and ``set_eval_seed``: the latter reseeds the
        # process-global RNG, so a rejected eval must not reach it.
        if error := positive_count_error(spec.max_steps, "max_steps", "evaluate_benchmark"):
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"{error} Benchmark {type(spec).__name__!r} declares that horizon; a "
                            "non-positive one runs episodes of zero length and reports a 0% success "
                            "rate over them instead of surfacing the mistake."
                        )
                    }
                ],
            }

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Named-body poses the policy declared it needs (mimic trackers read an
        # anchor link). Resolved once here so an unknown name fails before the
        # loop instead of being read as a zero pose on every tick; () for the
        # overwhelming majority of policies, which adds no backend call.
        _bodies = self._resolve_required_bodies(policy)
        # Full control-period substeps per action (see run() / evaluate()).
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        policy.set_control_frequency(control_frequency)
        # #168: seed Python / NumPy / torch / cuDNN once before
        # the episode loop so policy stochastic ops (e.g. attention
        # dropout, sampling temperature) are reproducible across re-runs
        # at the same ``seed``. Mirrors NVIDIA's upstream ``set_seed`` in
        # ``Isaac-GR00T/scripts/deployment/standalone_inference_script.py``.
        # Per-episode reproducibility still flows through ``episode_rng``
        # below for the spec's per-episode RNG-driven init / jitter.
        if seed is not None:
            set_eval_seed(seed)
        master_rng = random.Random(seed)
        spec_name = type(spec).__name__
        max_steps = spec.max_steps
        results: list[dict[str, Any]] = []

        # #191 - global step counter passed to ``on_frame``. Crosses
        # episode boundaries so consumers that don't track ep ↔ step
        # mappings still get a monotonic index. Callers that need
        # per-episode buckets can read ``info["steps"]`` from the
        # returned per-episode results.
        global_step = 0

        # #187 - fall back to ``spec.instruction`` (default ``""``) when
        # the user didn't pass an explicit instruction. Language-
        # conditioned policies (GR00T, OpenVLA) need the task description
        # or they produce off-task actions; LIBERO/Meta-World/etc. ship
        # the per-task language with the benchmark, so the spec is the
        # right source of truth. User-provided ``instruction`` still
        # wins when non-empty, preserving back-compat.
        spec_instruction = ""
        try:
            spec_instruction = spec.instruction or ""
        except Exception as e:  # noqa: BLE001 - back-compat for specs without the property
            logger.debug("spec.instruction lookup raised %s; defaulting to empty", e)
        effective_instruction = instruction or spec_instruction
        if not effective_instruction:
            logger.warning(
                "evaluate_benchmark: instruction is empty (user passed %r, spec.instruction=%r). "
                "Language-conditioned policies (GR00T, OpenVLA, etc.) will receive an empty "
                "string and may produce off-task actions. Pass instruction=... explicitly or "
                "override BenchmarkProtocol.instruction on your spec.",
                instruction,
                spec_instruction,
            )

        # Optional per-episode rollout video (evaluate_benchmark video=). One
        # MP4 per episode via the _ep{i} filename templating, so a benchmark
        # eval can be watched to see WHY episodes fail - not just read as an
        # aggregate success_rate (parity with eval_policy). The writer is opened
        # per episode below and frames are captured synchronously on the eval
        # thread at the on_frame point, so recording never perturbs the
        # bit-stable spec-path rollout (render is read-only over mjData).
        video_paths: list[str] = []
        current_vwriter: _RolloutVideoWriter | None = None

        stopped_early = False
        recording_save_error: str | None = None
        try:
            for ep in range(n_episodes):
                self.sim.reset()
                # Per-episode MP4 (foo_ep{i}.mp4). Path/camera validation +
                # probe render happen here; a bad path/camera fails the eval
                # up-front (on ep 0) instead of running N episodes and writing
                # nothing. No-op (returns None) when video is unset.
                ep_vcfg = self.sim._episode_video_config(video, ep)
                current_vwriter, _video_err = _RolloutVideoWriter.open(self.sim, ep_vcfg, control_frequency)
                if _video_err is not None:
                    return _video_err
                # Per-episode seeded RNG - deterministic given the master seed
                # and the episode index.
                episode_seed = master_rng.randint(0, 2**31 - 1)
                episode_rng = random.Random(episode_seed)

                # #179 - re-seed Python / NumPy / torch / cuDNN at the start
                # of EACH episode (not just once before the loop). Without
                # the per-episode reseed, every torch op draws from a global
                # RNG state that mutates across episodes, so the diffusion
                # sampler in policies like ``nvidia/GR00T-N1.7-LIBERO`` produces
                # different action chunks per re-run even at the same
                # ``seed=42``. With the per-episode reseed, episode N always
                # starts from the same RNG state regardless of what happened
                # in episodes 0..N-1.
                #
                # Validated on libero-10/SCENE5: without the per-episode
                # reseed a 5-episode eval ranged 0.40-1.00 across runs; with
                # it the same eval is bit-stable (same successes every run).
                set_eval_seed(episode_seed)

                # #187 - for SERVICE-mode policies (e.g. Gr00tPolicy over
                # ZMQ), set_eval_seed only seeds the client process. The
                # remote inference server has its own torch/CUDA RNG that
                # drifts across calls. Forward the per-episode seed via
                # policy.reset(seed=...) so server-side state can be
                # re-initialised. Default Policy.reset is a no-op; concrete
                # policies override (Gr00tPolicy forwards to the server's
                # `reset` endpoint).
                try:
                    policy.reset(seed=episode_seed)
                except Exception as e:  # noqa: BLE001 - reset is best-effort
                    logger.warning(
                        "policy.reset(seed=%d) raised %s; continuing without per-episode reset",
                        episode_seed,
                        e,
                    )

                try:
                    spec.on_episode_start(self.sim, episode_rng)
                except BenchmarkCompatibilityError as e:
                    # Surface the structured error with the supported list -
                    # agents can fix this without retrying.
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"Benchmark compatibility error: robot '{e.robot_name}' "
                                    f"has data_config={e.data_config!r}, but benchmark "
                                    f"{spec_name} supports {e.supported}."
                                )
                            }
                        ],
                    }
                except Exception as e:  # noqa: BLE001 - surface as structured error
                    logger.exception("on_episode_start failed")
                    return {
                        "status": "error",
                        "content": [{"text": f"on_episode_start failed in {spec_name}: {e}"}],
                    }

                success = False
                failure = False
                steps = 0
                cumulative_reward = 0.0
                last_info: dict[str, Any] = {}

                for _ in range(max_steps):
                    observation = self._observe(robot_name, skip_images=_skip_images, bodies=_bodies)
                    # Hook: benchmarks may bridge the sim's observation schema
                    # (typically joint-space) to whatever the policy was trained
                    # on (e.g. LIBERO's Cartesian state.x/y/z/roll/pitch/yaw/gripper).
                    # Default impl on BenchmarkProtocol is identity. Failures
                    # surface as structured errors rather than silent fall-through
                    # since "policy got the wrong obs schema" is a common bug
                    # source.
                    try:
                        observation = spec.augment_observation(self.sim, observation)
                    except Exception as e:  # noqa: BLE001
                        logger.exception("augment_observation failed in %s", spec_name)
                        return {
                            "status": "error",
                            "content": [{"text": f"augment_observation failed in {spec_name}: {e}"}],
                        }
                    coro_or_result = policy.get_actions(observation, effective_instruction, **(policy_kwargs or {}))
                    actions = _resolve_coroutine(coro_or_result)

                    # #168: consume up to ``action_horizon`` actions
                    # per inference. Default ``action_horizon=8`` matches NVIDIA's
                    # upstream GR00T LIBERO eval (``MultiStepWrapper`` with
                    # ``n_action_steps=8``) - the GR00T-N1.7-LIBERO checkpoints
                    # were trained against an 8-step open-loop chunk replay.
                    # The earlier ``=1`` default (closed-loop OpenVLA
                    # convention) put eval out-of-distribution from training
                    # and was a contributing factor to ``success_rate=0``.
                    # Set to ``1`` for closed-loop receding-horizon control.
                    # ``on_step`` and success/failure checks run after EACH
                    # applied action so per-step rewards / early termination
                    # work whether action_horizon is 1 or 8.
                    action_applied: dict[str, Any] = {}
                    stop_episode = False
                    if not actions:
                        # Degenerate policy - advance physics so loop terminates.
                        self.sim.step(n_steps=1)
                    else:
                        _chunk = resolve_chunk_length(policy, action_horizon)
                        for action_in_chunk in actions[:_chunk]:
                            if steps >= max_steps:
                                break
                            action_applied = dict(action_in_chunk)
                            self.sim.send_action(action_applied, robot_name=robot_name, n_substeps=n_substeps)
                            # #191 - synchronous on_frame hook fires on the
                            # eval thread, after send_action + before
                            # on_step's reward bookkeeping. Use this for
                            # synchronous frame recording when the eval is
                            # dispatched from a thread distinct from the
                            # script main (e.g. Strands Agent worker thread
                            # under asyncio); the daemon-thread recorder
                            # races mjData mutations on the eval thread and
                            # produces 2-3% frame-capture rates with greenish
                            # GL clear-colour artifacts. See
                            # ``Simulation.start_cameras_recording_synchronous``
                            # for the recorder side of this contract.
                            # Capture the rollout video frame synchronously on
                            # the eval thread (same point + ep-local cadence as
                            # eval_policy's _fire_on_frame). Independent of the
                            # user on_frame hook so video records with or without
                            # one. ``steps`` is the pre-increment ep-local index.
                            if current_vwriter is not None:
                                current_vwriter.capture(steps)
                            if on_frame is not None:
                                try:
                                    on_frame(global_step, observation, action_applied)
                                except CooperativeStop:
                                    # Documented graceful early-stop; propagate
                                    # to the episode loop instead of swallowing.
                                    raise
                                except RecordingFrameError:
                                    # Data loss, not telemetry - see the note on
                                    # the tolerance constant.
                                    raise
                                except Exception as e:  # noqa: BLE001 - hook is best-effort
                                    logger.warning(
                                        "on_frame hook failed at global_step=%d (ep=%d, ep_step=%d): %s",
                                        global_step,
                                        ep,
                                        steps,
                                        e,
                                    )
                            steps += 1
                            global_step += 1
                            try:
                                info = spec.on_step(self.sim, observation, action_applied)
                            except Exception as e:  # noqa: BLE001
                                logger.exception("on_step failed in %s", spec_name)
                                return {
                                    "status": "error",
                                    "content": [{"text": f"on_step failed in {spec_name}: {e}"}],
                                }
                            cumulative_reward += float(info.reward)
                            last_info = dict(info.info) if info.info else {}
                            if info.done:
                                stop_episode = True
                                break
                            if _criterion_verdict(
                                spec.is_failure, self.sim, label=f"{spec_name}.is_failure", episode=ep, step=steps
                            ):
                                failure = True
                                stop_episode = True
                                break
                            if _criterion_verdict(
                                spec.is_success, self.sim, label=f"{spec_name}.is_success", episode=ep, step=steps
                            ):
                                success = True
                                stop_episode = True
                                break
                    if stop_episode:
                        break
                    if not actions:
                        # Degenerate-policy branch already advanced steps via
                        # sim.step(n_steps=1); count it like an applied step
                        # so the outer loop terminates.
                        steps += 1
                        global_step += 1
                        try:
                            info = spec.on_step(self.sim, observation, action_applied)
                        except Exception as e:  # noqa: BLE001
                            logger.exception("on_step failed in %s", spec_name)
                            return {
                                "status": "error",
                                "content": [{"text": f"on_step failed in {spec_name}: {e}"}],
                            }
                        cumulative_reward += float(info.reward)
                        last_info = dict(info.info) if info.info else {}
                        if info.done:
                            break
                        if _criterion_verdict(
                            spec.is_failure, self.sim, label=f"{spec_name}.is_failure", episode=ep, step=steps
                        ):
                            failure = True
                            break
                        if _criterion_verdict(
                            spec.is_success, self.sim, label=f"{spec_name}.is_success", episode=ep, step=steps
                        ):
                            success = True
                            break

                results.append(
                    {
                        "episode": ep,
                        "steps": steps,
                        "success": success,
                        "failure": failure,
                        "cumulative_reward": round(cumulative_reward, 4),
                        "seed": episode_seed,
                        "info": last_info,
                    }
                )
                # #708 - same per-episode recorder boundary as evaluate().
                recording_save_error = self._finalize_recorder_episode()

                if current_vwriter is not None:
                    current_vwriter.close()
                    if current_vwriter.frame_count > 0 and os.path.exists(current_vwriter.path):
                        video_paths.append(current_vwriter.path)
                    else:
                        logger.warning(
                            "evaluate_benchmark episode %d: video requested but wrote 0 frames to %s",
                            ep,
                            current_vwriter.path,
                        )
                    current_vwriter = None

                if recording_save_error is not None:
                    # Same rule as evaluate(): a lost episode stops the
                    # benchmark instead of being averaged over.
                    recording_save_error = f"episode {ep}: {recording_save_error}"
                    break

        except CooperativeStop:
            # A user/backend on_frame hook requested a graceful stop (the
            # same signal run() honors). End the benchmark over the episodes
            # completed so far instead of crashing with an uncaught
            # BaseException. Close any in-progress episode video cleanly.
            if current_vwriter is not None:
                current_vwriter.close()
            stopped_early = True
            logger.info(
                "on_frame requested a cooperative stop; ending benchmark after %d completed episode(s)",
                len(results),
            )
        n_completed = len(results)
        n_success = sum(1 for r in results if r["success"])
        n_failure = sum(1 for r in results if r["failure"])
        success_rate = n_success / max(n_completed, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_completed, 1)
        avg_reward = sum(r["cumulative_reward"] for r in results) / max(n_completed, 1)

        return {
            "status": "error" if recording_save_error is not None else "success",
            "content": [
                {
                    "text": (
                        f"Benchmark: {spec_name} | policy {type(policy).__name__} on '{robot_name}'\n"
                        + (
                            f"Stopped after a lost recording episode - {recording_save_error}\n"
                            if recording_save_error is not None
                            else ""
                        )
                        + f"Episodes: {n_completed}"
                        + (f" of {n_episodes} (stopped early)" if stopped_early else "")
                        + f" | Success: {n_success} | Failure: {n_failure} ({success_rate:.1%} success)\n"
                        f"Avg reward: {avg_reward:.2f} | Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "success_measured": True,
                        "n_episodes": n_episodes,
                        "episodes_completed": n_completed,
                        "stopped_early": stopped_early,
                        "recording_save_error": recording_save_error,
                        "n_success": n_success,
                        "n_failure": n_failure,
                        "avg_steps": round(avg_steps, 1),
                        "avg_reward": round(avg_reward, 4),
                        "max_steps": max_steps,
                        "seed": seed,
                        "benchmark_class": spec_name,
                        "policy_load_time_s": round(float(getattr(policy, "load_time_s", 0.0)), 3),
                        "policy_load_cache_hit": bool(getattr(policy, "load_cache_hit", False)),
                        "positional_fallback_used": bool(getattr(policy, "positional_fallback_used", False)),
                        "generic_state_keys_used": bool(getattr(policy, "generic_state_keys_used", False)),
                        "missing_state_keys_used": bool(getattr(policy, "missing_state_keys_used", False)),
                        "policy_resident_rss_mb": process_rss_mb(),
                        "episodes": results,
                        "video_paths": video_paths,
                    }
                },
            ],
        }

    # Helpers

    def _maybe_sim_time(self) -> float | None:
        """Best-effort read of sim time from any backend that exposes it.

        Tries two paths:
          1. ``sim._world.sim_time`` - fast path for backends that keep a
             structured world object (MuJoCo, and any other backend using
             ``strands_robots.simulation.models.SimWorld``).
          2. ``sim.get_state()`` fallback for backends that only expose the
             status-dict shape. If the dict's ``json`` block (or top level)
             has a ``sim_time`` key, we return it.
        """
        world = getattr(self.sim, "_world", None)
        if world is not None:
            t = getattr(world, "sim_time", None)
            if isinstance(t, (int, float)):
                return float(t)

        get_state = getattr(self.sim, "get_state", None)
        if get_state is None:
            return None
        try:
            state = get_state()
        except Exception:
            return None
        if isinstance(state, dict):
            if "sim_time" in state:
                return float(state["sim_time"])
            for blk in state.get("content", []):
                if isinstance(blk, dict) and isinstance(blk.get("json"), dict):
                    t = blk["json"].get("sim_time")
                    if isinstance(t, (int, float)):
                        return float(t)
        return None

    def _require_default_robot(self) -> str:
        robots = self.sim.list_robots()
        if not robots:
            raise ValueError("No robots in sim. Add one first.")
        return robots[0]

    def _resolve_success_fn(self, success_fn: SuccessFn | str | None) -> SuccessFn | None:
        if success_fn is None:
            return None
        if callable(success_fn):
            return success_fn
        if success_fn == "contact":
            sim = self.sim
            # Share the DSL's reader instead of keeping a second one. The
            # inline copy this replaces indexed the engine result as if it
            # were the payload, so it never saw a real backend's envelope and
            # scored every episode a failure - while still working against a
            # test double that returns the bare mapping.
            #
            # Imported inside the method, not at module level: base.py imports
            # this module at import time and predicates.py imports base under
            # TYPE_CHECKING, so a module-level edge from here to predicates
            # closes a loop that CodeQL's py/unsafe-cyclic-import walks - it
            # does not honour the guard (see the #191 note on base.py's import
            # of this module). No runtime cycle exists either way, and base.py
            # reaches into predicates the same way from
            # ``_stop_when_unresolved_error``.
            from strands_robots.simulation.predicates import make_predicate

            contact_any = make_predicate("contact_any")

            def _contact_check(_obs: dict[str, Any]) -> bool:
                return bool(contact_any(sim))

            return _contact_check
        raise ValueError(f"Unknown success_fn string: {success_fn!r}")


__all__ = [
    "PolicyRunner",
    "OnFrame",
    "SuccessFn",
    "CooperativeStop",
    "TrajectoryStep",
    "set_eval_seed",
]
