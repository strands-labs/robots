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

import logging
import os
import random
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.policies.base import resolve_chunk_length
from strands_robots.utils import process_rss_mb, require_optional

if TYPE_CHECKING:
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
      most policies use under the hood).
    * PyTorch CPU (``torch.manual_seed``) - if torch is importable.
    * PyTorch CUDA all devices (``torch.cuda.manual_seed_all``) - if
      torch is importable AND CUDA is available.
    * cuDNN ``deterministic=True`` / ``benchmark=False`` - if torch
      is importable. These are the standard reproducibility knobs and
      are scoped to torch (not the broader environment) so the side
      effect surface is acceptable.

    Public since #179: standalone integration tests
    (``tests_integ/.../test_libero_10_scene5_mujoco_engine_success_rate``)
    bypass :meth:`evaluate_benchmark` and need to call this directly to
    get reproducible policy rollouts. The leading ``_`` was an oversight
    from #168 round 38; the function is the supported way to seed an
    eval and is part of the public API.

    NumPy / torch are imported lazily so this helper works on minimal
    installs that don't have torch (e.g. ``policy_provider="mock"``
    smoke tests).
    """
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
        return bool(self.path)

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> VideoConfig | None:
        """Build from a plain dict (tool_spec dispatcher path). ``None`` passthrough."""
        if not d:
            return None
        # Accept both canonical keys and legacy/tool_spec aliases.
        return cls(
            path=d.get("path") or d.get("record_video") or d.get("output_path"),
            fps=int(d.get("fps") or d.get("video_fps") or 30),
            camera=d.get("camera") or d.get("video_camera") or d.get("camera_name"),
            width=int(d.get("width") or d.get("video_width") or 640),
            height=int(d.get("height") or d.get("video_height") or 480),
        )


class _RolloutVideoWriter:
    """Optional per-rollout MP4 writer: validate the (LLM-supplied) output path,
    probe the camera, open an imageio writer, append frames at the requested fps
    cadence, and finalize on close.

    Extracted from :meth:`PolicyRunner.run` so the multi-episode evaluation loop
    (:meth:`PolicyRunner.evaluate`) records rollout video identically - one
    source of truth for the security-sensitive path validation (see PR #930)
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
        _sb_root, _allow_abs = video_sandbox_args()
        try:
            resolved = str(
                validate_output_path(video.path, sandbox_root=_sb_root, allow_abs=_allow_abs, label="video path")
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
        """
        if override is not None:
            return max(1, int(override))
        dt = None
        try:
            dt = self.sim.physics_timestep()
        except Exception:  # noqa: BLE001 - never fail a run on a probe
            dt = None
        if dt and dt > 0 and control_frequency > 0:
            return max(1, round((1.0 / control_frequency) / dt))
        return 1

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
    def _finalize_recorder_episode(self) -> None:
        """Roll the attached dataset_recorder over to a new episode.

        Called at end of each rollout iteration in ``evaluate`` and
        ``_evaluate_with_spec``. No-op when no recorder is attached or when
        the episode buffer is empty (e.g. degenerate policy returned no
        actions and ``add_frame`` was never called).
        """
        try:
            world = getattr(self.sim, "_world", None)
            if world is None:
                return
            recorder = world._backend_state.get("dataset_recorder")
        except AttributeError:
            return
        if recorder is None:
            return
        # Don't flush an empty buffer - LeRobot raises on save_episode with
        # zero frames, and a degenerate rollout still counts as "no data" for
        # this episode rather than an error.
        pending = getattr(recorder, "episode_frame_count", 0)
        if pending <= 0:
            return
        try:
            result = recorder.save_episode()
            if isinstance(result, dict) and result.get("status") != "success":
                logger.warning(
                    "Per-episode save_episode returned non-success: %s",
                    result.get("message", result),
                )
        except Exception as e:  # noqa: BLE001 - recorder errors must not abort eval
            logger.warning("Per-episode save_episode raised: %s", e)

    # run(): blocking policy execution
    def run(
        self,
        robot_name: str,
        policy: Policy,
        *,
        instruction: str = "",
        duration: float = 10.0,
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
    ) -> dict[str, Any]:
        """Run ``policy`` on ``robot_name`` for ``duration`` seconds.

        Args:
            robot_name: Name of robot in the sim.
            policy: Already-constructed ``Policy`` instance. Callers (typically
                ``SimEngine.run_policy``) are responsible for policy
                construction so tests can inject mocks trivially.
            instruction: Natural-language instruction forwarded to the policy.
            duration: Wall-clock seconds to run (interpreted as control steps
                via ``control_frequency``).
            control_frequency: Target Hz for ``policy.get_actions`` calls.
            action_horizon: Max actions consumed per policy call before
                requerying observation. Clamped up to the policy's own
                ``actions_per_step`` so a model trained for N-step
                open-loop chunk replay never has its chunk truncated
                below N (the effective horizon is
                ``max(action_horizon, policy.actions_per_step)``).
            fast_mode: If True, skip real-time ``time.sleep`` between steps.
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
            max_onframe_failures: Maximum *consecutive* non-``CooperativeStop``
                exceptions from the ``on_frame`` hook before the runner aborts
                the episode. ``None`` (default) uses
                ``_MAX_CONSECUTIVE_ONFRAME_FAILURES`` (currently ``5``). A
                broken recording hook otherwise silently produces empty
                datasets - see GH #117. Non-consecutive failures reset the
                counter.
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

        Returns:
            ``{"status": "success"|"error", "content": [{"text": ...},
            {"json": {...}}]}``. The ``json`` block is agent-consumable and
            carries the rollout facts as typed fields - ``robot_name``,
            ``policy``, ``instruction``, ``n_steps``, ``elapsed_s``,
            ``stopped_early``, ``action_errors``, ``video_path`` (``None`` when
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
            return _video_err

        stopped_early = False
        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
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
        start_time = time.time()
        step_count = 0
        try:
            total_steps = int(duration * control_frequency)
            action_sleep = 1.0 / control_frequency

            # Control-rate substepping: a position-servo robot needs the physics
            # to advance for the FULL control period (1/control_frequency) after
            # each action so the joints actually track the commanded target
            # before the next action overwrites ``ctrl``. With the default
            # 1 substep/action, the arm only integrates one physics dt (~2 ms)
            # per action and barely moves - the policy looks like a no-op even
            # though it is sending valid targets. Derive substeps from the
            # backend's physics timestep; fall back to 1 when unknown.
            if control_substeps is not None:
                n_substeps = max(1, int(control_substeps))
            else:
                _dt = None
                try:
                    _dt = self.sim.physics_timestep()
                except Exception:  # noqa: BLE001 - never fail the run on a probe
                    _dt = None
                if _dt and _dt > 0 and control_frequency > 0:
                    n_substeps = max(1, round((1.0 / control_frequency) / _dt))
                else:
                    n_substeps = 1
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

                if not fast_mode:
                    time.sleep(action_sleep)

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
                    cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                                cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                                cur_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                            prefetch_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                            step_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                        else:
                            step_obs = cur_obs
                        _apply(step_obs, cur_chunk[idx])
                        idx += 1
                finally:
                    # Wait for any in-flight inference so no background thread
                    # touches the policy/sim after run() returns (the caller may
                    # immediately reset() or destroy() the world).
                    executor.shutdown(wait=True)
            else:
                while step_count < total_steps:
                    observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                            step_obs = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                        else:
                            step_obs = observation
                        _apply(step_obs, action_dict)

        except CooperativeStop:
            stopped_early = True
        except Exception as e:
            if vwriter is not None:
                vwriter.close()
            logger.exception("PolicyRunner.run failed")
            return {
                "status": "error",
                "content": [{"text": f"Policy failed: {e}"}, {"json": _rtc_telemetry()}],
            }

        # Either finished all steps or was cooperatively stopped
        elapsed = time.time() - start_time
        sim_time = self._maybe_sim_time()
        prefix = "Policy stopped" if stopped_early else "Policy complete"
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
            "elapsed_s": round(elapsed, 3),
            "stopped_early": stopped_early,
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

        # If every send_action call failed (all keys unresolved), the robot
        # never moved -- report this as an error rather than a false success.
        if _action_errors > 0 and _action_errors >= step_count and step_count > 0:
            text += (
                f"\n\nALL {_action_errors} action steps had unresolved keys "
                f"-- the robot did not move. Check that the policy's output keys "
                f"match the robot's actuator names."
            )
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
            episode: Episode index in the dataset (non-negative).
            root: Optional local dataset root override.
            speed: Playback speed multiplier (1.0 = real time). Must be a
                positive number; a non-positive or non-numeric value is
                rejected with a structured error.
            action_key_map: Optional list of action keys, one per action
                vector index. Required when dataset action ordering differs
                from ``robot_action_keys(robot_name)``. If ``None``, positional
                mapping to ``robot_action_keys`` is used - the robot's
                *actuator* keys, which is the ordering the LeRobotDataset
                recorder writes the ``action`` column in (a robot's actuators
                are not always its joints; see :meth:`SimEngine.robot_action_keys`).

        Returns:
            Standard status dict with per-frame stats.
        """
        # ``speed`` is a playback-rate multiplier used as the divisor in
        # ``frame_interval = 1 / (dataset_fps * speed)``. A value of 0 raised a
        # bare ZeroDivisionError (breaking the documented "returns a status
        # dict" contract) and a negative value silently played the episode
        # forward at full speed while reporting success with a meaningless
        # "Speed: -1.0x". Reject a non-positive or non-numeric speed up front,
        # before the (potentially multi-minute) dataset download. ``bool`` is
        # an ``int`` subclass, so ``True`` is rejected explicitly rather than
        # acting as a silent 1.0x.
        if isinstance(speed, bool) or not isinstance(speed, (int, float)) or speed <= 0:
            return {
                "status": "error",
                "content": [{"text": f"replay: speed must be a positive number (got {speed!r})."}],
            }

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
        start_time = time.time()

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
            step_start = time.time()
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

                action_dict: dict[str, Any] = {}
                for i, val in enumerate(action_vals):
                    if i >= len(action_keys):
                        break
                    action_dict[action_keys[i]] = float(val)

                self.sim.send_action(action_dict, robot_name=resolved_robot, n_substeps=n_substeps)
                frames_applied += 1

            sleep_time = frame_interval - (time.time() - step_start)
            if sleep_time > 0:
                time.sleep(sleep_time)

        duration = time.time() - start_time
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
          backwards compatibility with PR #85. Equivalent to a
          ``BenchmarkProtocol`` whose ``on_step`` always returns
          ``StepInfo(reward=0.0, done=False)``.

        Passing both ``spec`` and ``success_fn`` is an error - benchmarks
        define their own success predicate.

        Args:
            robot_name: Robot to evaluate.
            policy: Already-constructed ``Policy`` instance.
            instruction: Instruction forwarded to the policy.
            n_episodes: Number of reset → rollout episodes.
            max_steps: Cap per episode. Ignored when ``spec`` is provided
                (``spec.max_steps`` wins).
            success_fn: Legacy success predicate (see above).
            spec: :class:`BenchmarkProtocol` to drive the eval. When
                provided, overrides the ``success_fn`` path.
            seed: Master RNG seed. Each episode derives a child RNG from it,
                so evaluations are reproducible within a process. Only used
                when ``spec`` is provided.
            on_frame: Optional ``(step, observation, action) -> None`` hook
                fired per applied control step on the eval thread, after
                ``sim.send_action``. Forwarded on BOTH the ``spec=`` and the
                legacy ``success_fn`` paths; ``step`` is a monotonic index
                that continues across episode boundaries. A non-
                ``CooperativeStop`` hook exception is logged at WARN and never
                aborts the eval; raising :class:`CooperativeStop` stops the
                evaluation gracefully after the episodes completed so far
                (the result carries ``stopped_early=True`` and
                ``episodes_completed``), matching :meth:`run`. Use this for
                synchronous recording when the eval runs on a thread distinct
                from the script main (e.g. Strands ``Agent`` tool dispatch
                under asyncio) - see #191 and
                :meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording_synchronous`.
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
        """
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
        # Step physics for the full control period per action, same derivation
        # as run(). The default n_substeps=1 made eval rollouts under-step.
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        policy.set_control_frequency(control_frequency)

        # RTC telemetry, reported in the result json so inference cost (and,
        # under async_rtc, latency masking) is provable without grepping logs.
        # inference_ms collects every get_actions wall-time on both paths; the
        # prefetch hit/block counters are async-only (0 on the synchronous path).
        inference_ms: list[float] = []
        rtc_chunks_acquired = 0
        rtc_prefetch_hits = 0
        rtc_prefetch_blocks = 0

        def _observation_fn() -> dict[str, Any]:
            return self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)

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
            # failure is logged at WARN and never aborts the eval.
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
                except Exception as e:  # noqa: BLE001 - hook is best-effort telemetry
                    logger.warning("on_frame hook failed at global_step=%d: %s", global_step, e)
            global_step += 1

        stopped_early = False
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
                            if resolved_check is not None and resolved_check(_observation_fn()):
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
                            if resolved_check is not None and resolved_check(_observation_fn()):
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
                            if resolved_check is not None and resolved_check(_observation_fn()):
                                success = True
                                break
                        if success:
                            break

                results.append({"episode": ep, "steps": steps, "success": success})
                # #708 - roll the attached recorder over to a new episode so the
                # dataset records per-episode boundaries rather than collapsing
                # every rollout into one mega-episode.
                self._finalize_recorder_episode()

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
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Evaluation: {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_completed}"
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

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
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
                # Validated on libero-10/SCENE5: pre-#179 5-ep eval ranged
                # 0.40-1.00 across runs; post-#179 the same eval is bit-stable
                # (same successes list every run).
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
                    observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
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
                            if spec.is_failure(self.sim):
                                failure = True
                                stop_episode = True
                                break
                            if spec.is_success(self.sim):
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
                        if spec.is_failure(self.sim):
                            failure = True
                            break
                        if spec.is_success(self.sim):
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
                self._finalize_recorder_episode()

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
            "status": "success",
            "content": [
                {
                    "text": (
                        f"Benchmark: {spec_name} | policy {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_completed}"
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

            def _contact_check(_obs: dict[str, Any]) -> bool:
                get_contacts = getattr(sim, "get_contacts", None)
                if get_contacts is None:
                    return False
                try:
                    result = get_contacts()
                except NotImplementedError:
                    return False
                except Exception:
                    return False
                # Accept either {"contacts": [...]} or {"n_contacts": int}
                if isinstance(result, dict):
                    if result.get("n_contacts", 0) > 0:
                        return True
                    contacts = result.get("contacts")
                    if isinstance(contacts, list) and contacts:
                        return True
                return False

            return _contact_check
        raise ValueError(f"Unknown success_fn string: {success_fn!r}")


__all__ = ["PolicyRunner", "OnFrame", "SuccessFn", "CooperativeStop", "TrajectoryStep", "set_eval_seed"]
