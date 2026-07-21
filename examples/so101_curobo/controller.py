# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orchestration for the SO-101 cuRobo synthetic-data demo (issue #67).

``SO101CuroboDemo`` ties the four pieces together — simulation backend
(:func:`make_sim`), scene (:func:`build_pick_place_scene`), motion planner
(:func:`make_planner`), and the LeRobot collector — behind a small API used by
both the Strands agent (:mod:`agent`) and the Gradio app (:mod:`app`):

    plan_and_execute(task) -> plan a trajectory, execute it, record one episode
    record_dataset(n)      -> generate N episodes into one LeRobot dataset
    render(camera)         -> an RGB frame for the UI
    describe()             -> a human-readable status (backend, planner, deps)

All simulation access is serialized with a lock so the Gradio worker threads
don't race on the sim.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from .collector import LeRobotDataCollector, lerobot_available
from .planner import CUROBO_AVAILABLE, make_planner
from .scene import build_pick_place_scene, make_sim

logger = logging.getLogger("so101_curobo.controller")


class _FallbackPlanner:
    """Wrap a primary planner; fall back to ScriptedPlanner if it raises.

    Keeps the demo robust when cuRobo is installed but a specific pick-place
    pose is infeasible for the 5-DOF SO-101 (a known calibration gap -- see
    README #67 T5): the arm still moves and an episode is still recorded.
    """

    def __init__(self, primary):
        self.primary = primary
        self.name = getattr(primary, "name", "planner")
        self._scripted = None

    def plan_pick_place(self, **kwargs):
        try:
            return self.primary.plan_pick_place(**kwargs)
        except Exception as exc:  # noqa: BLE001 - infeasible/unreachable -> fallback
            if self._scripted is None:
                from .planner import ScriptedPlanner

                self._scripted = ScriptedPlanner()
            logger.warning(
                "%s planning failed (%s); using scripted fallback.", getattr(self.primary, "name", "?"), str(exc)[:140]
            )
            self.name = f"scripted(fallback from {getattr(self.primary, 'name', '?')})"
            scripted_keys = (
                "joint_names",
                "start_q",
                "gripper_joint",
                "cube_xy",
                "place_xy",
                "steps_per_phase",
                "base_sign",
            )
            return self._scripted.plan_pick_place(**{k: v for k, v in kwargs.items() if k in scripted_keys})


class SO101CuroboDemo:
    """Backend-agnostic SO-101 pick-and-place + synthetic-data controller."""

    def __init__(
        self,
        backend: str = "mujoco",
        repo_id: str = "local/so101_curobo_pickplace",
        root: str | None = None,
        prefer_planner: str = "auto",
        # Recorded-video playback rate. cuRobo interpolates its plan at a fixed
        # 0.025 s per waypoint, and the executor records one frame per waypoint,
        # so 1/0.025 = 40 fps makes the saved video play the planned motion at
        # REAL TIME (a 435-waypoint plan = ~10.9 s of motion = ~10.9 s of video).
        # (An earlier attempt derived fps from the MuJoCo sim timestep *
        # n_substeps; that is unrelated to the planned trajectory timing and made
        # playback ~2.5x too fast.) The scripted fallback has no time basis (coarse
        # keyframes), so its short clips look fast regardless -- that's a fallback
        # artifact, not the playback rate.
        fps: int = 40,
        n_substeps: int = 5,
        camera_size: tuple = (640, 480),
        record_images: bool = True,
        planner_kwargs: dict | None = None,
    ):
        self.backend = backend
        self.repo_id = repo_id
        self.root = root
        self.prefer_planner = prefer_planner
        self.fps = fps
        self.n_substeps = n_substeps
        self.camera_size = camera_size
        self.record_images = record_images
        # Extra kwargs forwarded to make_planner (e.g. cuRobo urdf_path/asset_path).
        self.planner_kwargs = dict(planner_kwargs or {})

        self._lock = threading.RLock()
        self.sim = None
        self.scene = None
        self.planner = None
        self.collector: LeRobotDataCollector | None = None
        self.current_camera = "front"
        self._built = False
        self._backend_note = ""
        self._home_q = None
        self._home_snapshot = None

    # --- lifecycle ----------------------------------------------------------

    def build(self) -> SO101CuroboDemo:
        """Create the sim + scene + planner + collector. Falls back MuJoCo<-Isaac."""
        with self._lock:
            backend = self.backend
            try:
                self.sim = make_sim(backend=backend)
            except Exception as exc:  # noqa: BLE001 - Isaac runtime missing, etc.
                if backend != "mujoco":
                    logger.warning("Backend %r unavailable (%s); falling back to MuJoCo.", backend, exc)
                    self._backend_note = f"{backend} unavailable ({exc}); using MuJoCo."
                    backend = "mujoco"
                    self.sim = make_sim(backend="mujoco")
                else:
                    raise
            self.backend = backend
            # Build the planner first so we know whether cuRobo will drive the
            # arm; if so, load the sim arm from the SAME URDF cuRobo plans with
            # (identical joint conventions + EE frame -> plans execute correctly).
            self.planner = _FallbackPlanner(
                make_planner(prefer=self.prefer_planner, robot_cfg="so101", **self.planner_kwargs)
            )
            # Resolve a URDF for the sim arm when needed: the cuRobo planner
            # needs the sim to load the SAME URDF it plans with (identical joint
            # conventions + EE frame), and the Isaac backend has no data_config
            # path at all -- it requires a URDF. In both cases load the arm from
            # that URDF so plans execute correctly / the backend can build it.
            needs_urdf = getattr(self.planner.primary, "name", "") == "curobo" or self.backend in (
                "isaac",
                "isaacsim",
                "isaac_sim",
            )
            robot_urdf = None
            if needs_urdf:
                # Same precedence as the cuRobo planner (explicit kwarg ->
                # SO101_URDF -> the auto-downloaded strands-robots cache URDF) so
                # the sim arm loads the EXACT URDF cuRobo plans with, and the
                # cuRobo/URDF path works out-of-the-box once the MuJoCo demo has
                # populated the SO-101 asset cache.
                from .planner import resolve_so101_urdf

                robot_urdf = resolve_so101_urdf(self.planner_kwargs.get("urdf_path"))
            self.scene = build_pick_place_scene(
                self.sim, camera_size=self.camera_size, backend=backend, robot_urdf=robot_urdf
            )
            # Physical grasp path: when the arm has real position drives (the
            # MuJoCo-actuated path OR the Isaac PhysX articulation), drive it via
            # send_action and let the gripper physically clamp the cube -- no
            # kinematic teleport or attach-carry. Only the actuator-less MuJoCo
            # URDF path falls back to the kinematic carry.
            actuated = bool(getattr(self.scene, "actuated", False))
            is_isaac = self.backend in ("isaac", "isaacsim", "isaac_sim", "nvidia")
            physical = actuated or is_isaac
            # The force-controlled arm needs more physics substeps per waypoint
            # than the kinematic teleport: enough for the position drive to track
            # each setpoint and for the gripper to fully clamp the cube. 12 * dt
            # (0.002) = 0.024 s ~ cuRobo's 0.025 s/waypoint, so playback stays
            # ~real-time too. The kinematic path keeps the lighter 5.
            if physical and self.n_substeps < 12:
                self.n_substeps = 12
            self.collector = LeRobotDataCollector(
                self.sim,
                self.scene,
                repo_id=self.repo_id,
                fps=self.fps,
                root=self.root,
                cameras=self.scene.cameras,
                record_images=self.record_images,
                kinematic=(bool(robot_urdf) and not physical),
                grasp_attach=(bool(robot_urdf) and not physical),
                base_sign=(-1.0 if is_isaac else 1.0),
                # Isaac hybrid: physically grasp+lift the cube (real PhysX), then
                # kinematically carry it through the place traverse to the bin
                # (the 5-DOF friction grip can't reliably hold the cube through
                # the sideways move, so the transport is scripted while the
                # pick+lift stays real). Success requires a genuine physical lift.
                hybrid_carry=is_isaac,
            )
            if self.scene.cameras:
                self.current_camera = self.scene.cameras[0]
            # Capture the clean rest pose + physics state now so every
            # single-shot plan_and_execute can reset to a known, collision-free
            # start (matching the per-episode reset in record_dataset). Without
            # this the 2nd "Plan & execute" click starts from the previous
            # episode's final arm pose / moved cube, which cuRobo rejects as
            # "start state in collision" -> scripted fallback (success=False).
            self._home_q = self.collector.home_q()
            self._home_snapshot = self.collector._snapshot_state()
            self._built = True
            logger.info("Demo built: %s | planner=%s", self.scene.pretty(), self.planner.name)
            return self

    def _require(self):
        if not self._built:
            raise RuntimeError("Demo not built yet — call build() first.")

    # --- actions ------------------------------------------------------------

    def _plan(self):
        start_q = [
            float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in self.scene.joint_names
        ]
        # The SO-101 URDF (Isaac backend) has an inverted shoulder_pan sign vs
        # world Y: commanding +pan swings the arm toward -Y, so a +Y target
        # (the cube/bin) needs a negated base angle. MuJoCo's model uses the
        # default (+1) convention. Pass base_sign=-1 for Isaac so the scripted
        # planner actually aims the gripper AT the cube (otherwise it sweeps to
        # the opposite side and never reaches -> success_rate stays 0).
        base_sign = -1.0 if self.backend in ("isaac", "isaacsim", "isaac_sim", "nvidia") else 1.0
        return self.planner.plan_pick_place(
            joint_names=self.scene.joint_names,
            start_q=start_q,
            gripper_joint=self.scene.gripper_joint,
            cube_xy=self.scene.cube_position[:2],
            place_xy=self.scene.place_position[:2],
            base_sign=base_sign,
        )

    def plan_and_execute(
        self, task: str = "pick up the red cube and place it in the bin", n_substeps: int | None = None
    ) -> str:
        """Plan a pick-and-place, execute it, and record one LeRobot episode."""
        with self._lock:
            self._require()
            n_substeps = self.n_substeps if n_substeps is None else n_substeps

            def _work() -> str:
                # Reset to the clean rest pose + cube start before planning so
                # each click starts from a known, collision-free state (else the
                # previous episode's final arm pose / moved cube makes cuRobo
                # report "start state in collision" -> scripted fallback).
                try:
                    self.collector.reset_world(home_q=self._home_q, snapshot=self._home_snapshot)
                except Exception:  # noqa: BLE001 - reset is best-effort
                    logger.debug("pre-plan reset failed (non-fatal)", exc_info=True)
                try:
                    traj = self._plan()
                except RuntimeError as exc:  # cuRobo not wired -> actionable message
                    return f"Planner unavailable: {exc}"
                if not self.collector.available():
                    # Still execute (move the arm) even if we can't record a dataset.
                    robot = self.scene.robot_name
                    for wp in traj.waypoints:
                        self.sim.send_action(wp, robot_name=robot, n_substeps=n_substeps)
                    return (
                        f"Executed a {self.planner.name} pick-and-place ({len(traj)} waypoints). "
                        f"Dataset NOT recorded: {LeRobotDataCollector.__module__}: lerobot missing."
                    )
                res = self.collector.record_episode(traj, task=task, n_substeps=n_substeps)
                return (
                    f"Planned ({self.planner.name}) + executed + recorded 1 episode: "
                    f"{res.frames} frames, success={res.success} (cube moved={res.cube_moved}, "
                    f"displacement={res.displacement:.3f} m). Dataset: {self.repo_id}."
                )

            # On the Isaac backend the UI calls this from a Gradio worker thread,
            # but the sim can only be driven from the main (pump) thread. Submit
            # the WHOLE episode to the main thread so it runs inline there (like
            # the headless smoke path) instead of round-tripping every frame
            # through the action queue (slow + deadlock-prone for long plans).
            run_on_main = getattr(self.sim, "run_on_main", None)
            if callable(run_on_main):
                return run_on_main(_work)
            return _work()

    def record_dataset(
        self,
        n_episodes: int = 5,
        task: str = "pick up the red cube and place it in the bin",
        randomize: bool = True,
        n_substeps: int | None = None,
        on_episode=None,
    ) -> dict[str, Any]:
        with self._lock:
            self._require()
            n_substeps = self.n_substeps if n_substeps is None else n_substeps

            def _work() -> dict[str, Any]:
                return self.collector.record_dataset(
                    self.planner,
                    n_episodes=n_episodes,
                    task=task,
                    randomize=randomize,
                    n_substeps=n_substeps,
                    on_episode=on_episode,
                    home_q=self._home_q,
                    snapshot=self._home_snapshot,
                )

            # Run the whole multi-episode job on the main (pump) thread for the
            # Isaac backend (see plan_and_execute for the rationale).
            run_on_main = getattr(self.sim, "run_on_main", None)
            if callable(run_on_main):
                return run_on_main(_work)
            return _work()

    def render(self, camera: str | None = None) -> np.ndarray | None:
        """Return an RGB frame from ``camera`` (defaults to current)."""
        with self._lock:
            self._require()
            cam = camera or self.current_camera
            try:
                obs = self.sim.get_observation(self.scene.robot_name)
                img = obs.get(cam)
                if img is not None and hasattr(img, "shape"):
                    return self._crop_black_band(np.asarray(img)[:, :, :3])
            except Exception:  # noqa: BLE001 - rendering needs GL/EGL
                logger.debug("render failed for %s", cam, exc_info=True)
            return None

    @staticmethod
    def _crop_black_band(img: np.ndarray) -> np.ndarray:
        """Crop a hard black bottom band off a frame (Isaac headless RTX leaves
        the lower part of the camera buffer unrendered). Keeps the rendered
        region; returns the input unchanged if there's no significant band."""
        try:
            if img.ndim != 3 or img.shape[0] < 8:
                return img
            h = img.shape[0]
            rowmean = img.reshape(h, -1).mean(axis=1)
            content = np.where(rowmean > 6.0)[0]
            if content.size == 0:
                return img
            last = int(content.max()) + 1
            if last >= h - max(2, int(0.01 * h)) or last < h // 2:
                return img
            return img[:last]
        except Exception:  # noqa: BLE001
            return img

    def set_camera(self, camera: str) -> str:
        with self._lock:
            if self.scene and camera in self.scene.cameras:
                self.current_camera = camera
                return f"Camera set to {camera}."
            return f"Unknown camera {camera!r}. Options: {self.scene.cameras if self.scene else []}."

    def latest_video(self, camera: str | None = None) -> str | None:
        """Path to a browser-playable MP4 of the most recent recording for ``camera``.

        The collector writes a LeRobot v2.1 dataset under ``root`` with one video
        per camera at ``videos/observation.images.<camera>/chunk-*/file-*.mp4``.
        LeRobot encodes those as **AV1**, which browsers/``gr.Video`` generally
        cannot play, so we transcode the newest match to **H.264** in a temp file
        (cached by source mtime). Returns the H.264 path, or ``None`` if nothing
        has been recorded yet.
        """
        import glob
        import os

        cam = camera or self.current_camera
        root = getattr(self.collector, "root", None) or self.root
        if not root or not cam:
            return None
        pattern = os.path.join(root, "videos", f"observation.images.{cam}", "**", "*.mp4")
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            return None
        src = max(matches, key=os.path.getmtime)
        return self._h264(src, cam)

    def _content_crop_filter(self, src: str) -> str | None:
        """Return an ffmpeg ``-vf`` that crops a black band off ``src``, or None.

        The Isaac headless RTX camera fills only the top portion of the frame
        height; the rest is a hard black band. Read one frame, find the last row
        with real content, and build a ``crop=...,scale=...`` filter that keeps
        only the rendered region (scaled back to the original size for a stable
        player). Returns None (no crop) if no significant band is found.
        """
        try:
            import imageio.v3 as iio
            import numpy as np

            frames = iio.imread(src, plugin="pyav")
            if frames is None or len(frames) == 0:
                return None
            f = np.asarray(frames[len(frames) // 2])[..., :3]
            h, w = f.shape[:2]
            rowmean = f.reshape(h, -1).mean(axis=1)
            content = np.where(rowmean > 6.0)[0]
            if content.size == 0:
                return None
            last = int(content.max()) + 1
            # Only crop a meaningful band (> 8% of height) while keeping > half.
            if last >= h - max(2, int(0.01 * h)) or last < h // 2:
                return None
            ch = last - (last % 2)  # even height for yuv420p
            return f"crop={w}:{ch}:0:0,scale={w}:{h}"
        except Exception:  # noqa: BLE001 - cropping is best-effort
            logger.debug("content-crop analysis failed", exc_info=True)
            return None

    def _h264(self, src: str, cam: str) -> str | None:
        """Transcode ``src`` (LeRobot AV1) to a cached H.264 mp4 for browser playback."""
        import os
        import subprocess
        import tempfile

        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # noqa: BLE001 - no ffmpeg -> hand back the source as-is
            return src
        mtime = int(os.path.getmtime(src))
        out = os.path.join(tempfile.gettempdir(), f"so101_{cam}_{mtime}.mp4")
        if os.path.exists(out) and os.path.getmtime(out) >= os.path.getmtime(src):
            return out
        # The Isaac headless RTX render product fills only part of the camera
        # frame height, leaving a hard black band (rows below the rendered
        # region). Auto-detect the content height and crop the band so the video
        # shows the scene full-bleed instead of a black-padded image. Best-effort.
        vf = self._content_crop_filter(src)
        try:
            cmd = [ffmpeg, "-y", "-i", src]
            if vf:
                cmd += ["-vf", vf]
            cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-movflags", "+faststart", out]
            subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                timeout=120,
            )
            return out
        except Exception:  # noqa: BLE001 - transcode failed -> source path (may not play)
            logger.debug("h264 transcode failed for %s", src, exc_info=True)
            return src

    def describe(self) -> str:
        bits: list[str] = []
        if self.scene:
            bits.append(self.scene.pretty())
        bits.append(f"planner={self.planner.name if self.planner else '?'}")
        bits.append(f"cuRobo={'available' if CUROBO_AVAILABLE else 'NOT installed (scripted fallback)'}")
        bits.append(f"lerobot={'available' if lerobot_available() else 'NOT installed'}")
        if self._backend_note:
            bits.append(self._backend_note)
        return " | ".join(bits)

    def close(self):
        with self._lock:
            try:
                if self.sim is not None and hasattr(self.sim, "destroy"):
                    self.sim.destroy()
            except Exception:  # noqa: BLE001
                pass
