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
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots._async_utils import _resolve_coroutine
from strands_robots.utils import require_optional

if TYPE_CHECKING:
    from strands_robots.policies.base import Policy
    from strands_robots.simulation.base import SimEngine
    from strands_robots.simulation.benchmark import BenchmarkProtocol

from strands_robots.simulation.models import TrajectoryStep

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


# Backward-compatibility alias for the pre-#179 private name. Internal
# callers (this module's :class:`PolicyRunner`) still use it; the public
# :func:`set_eval_seed` is the supported entry point.
_set_eval_seed = set_eval_seed


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
    and returns a (H, W, 3|4) numpy array. Returns ``None`` if no image is
    found - the recorder then skips the frame rather than aborting the rollout.
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
        fps: Frames per second to write.
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


# on_frame hooks that raise are logged at WARN - user-provided telemetry is
# not allowed to kill the rollout. BUT if the hook raises on every single step
# (e.g. a recording hook with a typo'd observation key), we'd complete a 500-step
# episode with zero frames written and silently corrupt the dataset. After this
# many *consecutive* failures, the runner raises and fails the episode loudly.
#
# Overridable via the ``max_onframe_failures`` kwarg on ``PolicyRunner.run``.
# See GH #117.
_MAX_CONSECUTIVE_ONFRAME_FAILURES = 5


class CooperativeStop(BaseException):
    """Raised by an ``on_frame`` hook to cooperatively stop a run.

    Inherits ``BaseException`` (not ``Exception``) so hook authors don't
    accidentally swallow it with a broad ``except Exception``. Re-raised
    by ``PolicyRunner.run`` and caught at the top of the loop to return
    a normal stopped-early success result.
    """


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
                requerying observation.
            fast_mode: If True, skip real-time ``time.sleep`` between steps.
            video: Optional :class:`VideoConfig` - set ``video.path`` to enable
                MP4 recording via :meth:`SimEngine.render`.
            on_frame: Optional hook ``(step_idx, obs, action) -> None`` called
                after every ``send_action``. Public extension point - backends
                layer in recording / telemetry / graceful-stop via this hook
                without subclassing the runner.
            max_onframe_failures: Maximum *consecutive* non-``CooperativeStop``
                exceptions from the ``on_frame`` hook before the runner aborts
                the episode. ``None`` (default) uses
                ``_MAX_CONSECUTIVE_ONFRAME_FAILURES`` (currently ``5``). A
                broken recording hook otherwise silently produces empty
                datasets - see GH #117. Non-consecutive failures reset the
                counter.

        Returns:
            ``{"status": "success"|"error", "content": [{"text": ...}]}``.
        """
        # Lazy optional import - only imageio is optional.
        writer = None
        frame_count = 0
        frame_interval = 0.0
        next_frame_step = 0.0
        video_path: str | None = None
        if video is not None and video.enabled:
            # video.enabled guarantees video.path is a non-empty str; narrow for mypy.
            assert video.path is not None
            video_path = video.path

            # Pre-validate the camera name ONCE before the step loop. This
            # surfaces "camera not found" as a clean up-front error rather
            # than silently writing a 0-byte MP4 (sim.render() returns
            # status=error, _extract_frame_ndarray() returns None, the
            # rollout runs to completion, writer.close() produces an empty
            # file, and the user gets no hint in the result text).
            probe_cam = video.camera or "default"
            try:
                _probe = self.sim.render(
                    camera_name=probe_cam,
                    width=video.width,
                    height=video.height,
                )
            except Exception as e:
                return {
                    "status": "error",
                    "content": [{"text": f"Video recording requested but render probe crashed: {e}"}],
                }
            if _probe.get("status") != "success":
                probe_text = (_probe.get("content") or [{}])[0].get("text", "")
                return {
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
            os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
            writer = imageio.get_writer(  # type: ignore[attr-defined]
                video_path, fps=video.fps, quality=8, macro_block_size=1
            )
            frame_interval = control_frequency / video.fps

        stopped_early = False
        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
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

            onframe_failure_limit = (
                max_onframe_failures if max_onframe_failures is not None else _MAX_CONSECUTIVE_ONFRAME_FAILURES
            )
            consecutive_onframe_failures = 0
            while step_count < total_steps:
                observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)

                coro_or_result = policy.get_actions(observation, instruction)
                actions = _resolve_coroutine(coro_or_result)

                for action_dict in actions[:action_horizon]:
                    if step_count >= total_steps:
                        break

                    _send_result = self.sim.send_action(action_dict, robot_name=robot_name, n_substeps=n_substeps)
                    if isinstance(_send_result, dict) and _send_result.get("status") == "error":
                        _action_errors += 1

                    if on_frame is not None:
                        try:
                            on_frame(step_count, observation, action_dict)
                            consecutive_onframe_failures = 0
                        except CooperativeStop:
                            # Backend (e.g. MuJoCo) signalled a graceful stop.
                            # Break both loops and return a normal success result.
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

                    if writer is not None and step_count >= next_frame_step:
                        assert video is not None  # for mypy: writer only set when video.enabled
                        frame = self.sim.render(
                            camera_name=video.camera or "default",
                            width=video.width,
                            height=video.height,
                        )
                        # sim.render() returns {status, content:[{text},{image:{source:{bytes}}}]}
                        # Decode the PNG bytes from the content block and hand an ndarray
                        # to imageio. Silently skips when the PNG decode fails rather than
                        # aborting the whole rollout (renderer errors shouldn't kill training).
                        img_arr = _extract_frame_ndarray(frame)
                        if img_arr is not None:
                            writer.append_data(img_arr)
                            frame_count += 1
                        next_frame_step += frame_interval

                    if not fast_mode:
                        time.sleep(action_sleep)

        except CooperativeStop:
            stopped_early = True
        except Exception as e:
            if writer is not None:
                writer.close()
            logger.exception("PolicyRunner.run failed")
            return {"status": "error", "content": [{"text": f"Policy failed: {e}"}]}

        # Either finished all steps or was cooperatively stopped
        elapsed = time.time() - start_time
        sim_time = self._maybe_sim_time()
        prefix = "Policy stopped" if stopped_early else "Policy complete"
        text = (
            f"{prefix} on '{robot_name}'\n"
            f"🧠 {type(policy).__name__} | 🎯 {instruction}\n"
            f"⏱️ {elapsed:.1f}s | 📊 {step_count} steps"
        )
        if sim_time is not None:
            text += f" | 🕐 sim_t={sim_time:.3f}s"
        if writer is not None:
            assert video is not None and video_path is not None
            writer.close()
            if frame_count > 0 and os.path.exists(video_path):
                file_kb = os.path.getsize(video_path) / 1024
                text += (
                    f"\n🎬 Video: {video_path}\n"
                    f"📹 {frame_count} frames, {video.fps}fps, "
                    f"{video.width}x{video.height} | 💾 {file_kb:.0f} KB"
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
                text += f"\n⚠️ Video requested but 0 frames captured ({video_path})"
        # If every send_action call failed (all keys unresolved), the robot
        # never moved -- report this as an error rather than a false success.
        if _action_errors > 0 and _action_errors >= step_count and step_count > 0:
            text += (
                f"\n\n⚠️ ALL {_action_errors} action steps had unresolved keys "
                f"-- the robot did not move. Check that the policy's output keys "
                f"match the robot's actuator names."
            )
            return {"status": "error", "content": [{"text": text}]}
        if _action_errors > 0:
            text += f"\n\n⚠️ {_action_errors}/{step_count} action steps had unresolved keys."
        return {"status": "success", "content": [{"text": text}]}

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

        Args:
            repo_id: HuggingFace dataset id (e.g. ``lerobot/pusht``).
            robot_name: Target robot. Defaults to first robot in the sim.
            episode: Episode index in the dataset.
            root: Optional local dataset root override.
            speed: Playback speed multiplier (1.0 = real time).
            action_key_map: Optional list of joint names, one per action
                vector index. Required when dataset joint ordering differs
                from ``robot_joint_names(robot_name)``. If ``None``, positional
                mapping to ``robot_joint_names`` is used.

        Returns:
            Standard status dict with per-frame stats.
        """
        try:
            from strands_robots.dataset_recorder import load_lerobot_episode
        except ImportError:
            return {"status": "error", "content": [{"text": "lerobot not installed"}]}

        try:
            resolved_robot = robot_name or self._require_default_robot()
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        try:
            ds, episode_start, episode_length = load_lerobot_episode(repo_id, episode, root)
        except Exception as e:  # noqa: BLE001 - library errors are opaque
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # Resolve joint name ordering for action vector index → action dict.
        joint_names = list(action_key_map) if action_key_map else self.sim.robot_joint_names(resolved_robot)

        dataset_fps = getattr(ds, "fps", 30)
        frame_interval = 1.0 / (dataset_fps * speed)
        frames_applied = 0
        start_time = time.time()

        for frame_idx in range(episode_length):
            step_start = time.time()
            frame = ds[episode_start + frame_idx]

            action_vals = frame.get("action") if isinstance(frame, dict) else None
            if action_vals is None:
                # No action at this index - just advance physics one step.
                self.sim.step(n_steps=1)
                frames_applied += 1
            else:
                if hasattr(action_vals, "numpy"):
                    action_vals = action_vals.numpy()
                if hasattr(action_vals, "tolist"):
                    action_vals = action_vals.tolist()

                action_dict: dict[str, Any] = {}
                for i, val in enumerate(action_vals):
                    if i >= len(joint_names):
                        break
                    action_dict[joint_names[i]] = float(val)

                self.sim.send_action(action_dict, robot_name=resolved_robot)
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
                        f"▶️ Replayed episode {episode} from {repo_id} on '{resolved_robot}'\n"
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
                ``sim.send_action``. Currently only forwarded on the
                ``spec=`` path (the legacy ``success_fn`` path doesn't
                expose telemetry hooks). Use this for synchronous
                recording when the eval runs on a thread distinct from
                the script main (e.g. Strands ``Agent`` tool dispatch
                under asyncio) - see #191 and
                :meth:`~strands_robots.simulation.mujoco.simulation.Simulation.start_cameras_recording_synchronous`.

        Returns:
            Standard status dict. When ``spec`` is used, the JSON payload
            also contains ``cumulative_reward`` and ``avg_reward`` fields
            per episode and aggregate.
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
            )

        try:
            resolved_check = self._resolve_success_fn(success_fn)
        except ValueError as e:
            return {"status": "error", "content": [{"text": f"{e}"}]}

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Step physics for the full control period per action, same derivation
        # as run(). The default n_substeps=1 made eval rollouts under-step.
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        results: list[dict[str, Any]] = []
        for ep in range(n_episodes):
            self.sim.reset()
            success = False
            steps = 0

            for _ in range(max_steps):
                observation = self.sim.get_observation(robot_name=robot_name, skip_images=_skip_images)
                coro_or_result = policy.get_actions(observation, instruction)
                actions = _resolve_coroutine(coro_or_result)

                if actions:
                    self.sim.send_action(actions[0], robot_name=robot_name, n_substeps=n_substeps)
                else:
                    # Policy returned nothing - still advance one physics step
                    # so episodes don't hang on degenerate policies.
                    self.sim.step(n_steps=1)

                steps += 1

                if resolved_check is not None and resolved_check(observation):
                    success = True
                    break

            results.append({"episode": ep, "steps": steps, "success": success})

        n_success = sum(1 for r in results if r["success"])
        success_rate = n_success / max(n_episodes, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_episodes, 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📊 Evaluation: {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_episodes} | Success: {n_success}/{n_episodes} "
                        f"({success_rate:.1%})\n"
                        f"Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "n_episodes": n_episodes,
                        "n_success": n_success,
                        "avg_steps": round(avg_steps, 1),
                        "max_steps": max_steps,
                        "episodes": results,
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
        """
        # Lazy import to avoid circular reference (benchmark module imports
        # `SimEngine` from base which imports this module under TYPE_CHECKING).
        from strands_robots.simulation.benchmark import BenchmarkCompatibilityError

        # T26: skip camera rendering when the policy does not need images.
        _skip_images = not getattr(policy, "requires_images", True)
        # Full control-period substeps per action (see run() / evaluate()).
        n_substeps = self._control_substeps(control_frequency, control_substeps)
        # #168: seed Python / NumPy / torch / cuDNN once before
        # the episode loop so policy stochastic ops (e.g. attention
        # dropout, sampling temperature) are reproducible across re-runs
        # at the same ``seed``. Mirrors NVIDIA's upstream ``set_seed`` in
        # ``Isaac-GR00T/scripts/deployment/standalone_inference_script.py``.
        # Per-episode reproducibility still flows through ``episode_rng``
        # below for the spec's per-episode RNG-driven init / jitter.
        if seed is not None:
            _set_eval_seed(seed)
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

        for ep in range(n_episodes):
            self.sim.reset()
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
                coro_or_result = policy.get_actions(observation, effective_instruction)
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
                    for action_in_chunk in actions[:action_horizon]:
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
                        if on_frame is not None:
                            try:
                                on_frame(global_step, observation, action_applied)
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

        n_success = sum(1 for r in results if r["success"])
        n_failure = sum(1 for r in results if r["failure"])
        success_rate = n_success / max(n_episodes, 1)
        avg_steps = sum(r["steps"] for r in results) / max(n_episodes, 1)
        avg_reward = sum(r["cumulative_reward"] for r in results) / max(n_episodes, 1)

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📊 Benchmark: {spec_name} | policy {type(policy).__name__} on '{robot_name}'\n"
                        f"Episodes: {n_episodes} | Success: {n_success} | Failure: {n_failure} "
                        f"({success_rate:.1%} success)\n"
                        f"Avg reward: {avg_reward:.2f} | Avg steps: {avg_steps:.0f}/{max_steps}"
                    )
                },
                {
                    "json": {
                        "success_rate": round(success_rate, 4),
                        "n_episodes": n_episodes,
                        "n_success": n_success,
                        "n_failure": n_failure,
                        "avg_steps": round(avg_steps, 1),
                        "avg_reward": round(avg_reward, 4),
                        "max_steps": max_steps,
                        "seed": seed,
                        "benchmark_class": spec_name,
                        "episodes": results,
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
