"""Newton recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset`` and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`,
which is backend-agnostic. This subclass adds the two Newton-specific halves:

* :meth:`start_recording` declares the dataset schema from the live Newton
  scene - joint names from every robot (namespaced for multi-robot scenes) and
  the named cameras registered on the world.
* :meth:`_make_run_policy_hook` returns the ``on_frame`` closure the shared
  :class:`~strands_robots.simulation.base.SimEngine` run-policy loop calls every
  control step. It feeds joint state + action + rendered camera frames to the
  active :class:`~strands_robots.dataset_recorder.DatasetRecorder`.

The recorder, episode-boundary flushing (``save_episode``), and the canonical
parquet-correctness contract are identical to the MuJoCo backend - the
``DatasetRecorder`` is engine-independent, so a Newton recording produces the
same LeRobot v3 dataset layout (``meta/info.json`` + per-episode parquet +
per-camera MP4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.recording import DatasetRecordingMixin

if TYPE_CHECKING:
    from strands_robots.simulation.models import SimWorld

logger = logging.getLogger(__name__)


class NewtonRecordingMixin(DatasetRecordingMixin):
    """Newton dataset recording mixed into :class:`NewtonSimEngine`.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and supplies the Newton-specific schema
    declaration (:meth:`start_recording`) and per-step capture hook
    (:meth:`_make_run_policy_hook`).
    """

    if TYPE_CHECKING:
        _world: SimWorld | None
        _model: Any
        _robot_free_base_joint: dict[str, str]
        default_width: int
        default_height: int

        def render(self, camera_name: str = ..., width: int | None = ..., height: int | None = ...) -> dict[str, Any]:
            """Type-only stub for the engine-provided render method."""

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "h264",
        overwrite: bool = False,
        cameras: list[str] | None = None,
    ) -> dict[str, Any]:
        """Start recording the Newton scene to LeRobotDataset format.

        Declares the dataset schema from the live scene - joint names from every
        robot (namespaced ``robot__joint`` when more than one robot is present,
        matching the MuJoCo backend) and the named cameras registered on
        ``world.cameras`` (with their real render resolutions). Per-step frames
        are then captured by the ``on_frame`` hook
        (:meth:`_make_run_policy_hook`) during ``run_policy``.

        When no named cameras are registered the dataset records joint state and
        action only (a valid proprio-only LeRobot dataset); camera columns are
        added automatically once cameras are registered on the world.

        Requires the ``lerobot`` extra for the dataset schema.

        Args:
            repo_id: HuggingFace dataset id (``owner/name``) or a local path.
            task: Default task description recorded with every frame.
            fps: Recording frame rate.
            root: Explicit on-disk dataset directory (overrides the repo_id
                cache-path resolution).
            push_to_hub: Publish to the Hub at ``stop_recording``.
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), universally decodable including by OpenCV's
                VideoCapture (used by many downstream VLM video readers). Use
                "libsvtav1" (AV1) for smaller files in storage-constrained
                training pipelines; LeRobot read-back handles AV1 but OpenCV
                wheels commonly cannot decode it and silently yield 0 frames.
            overwrite: Wipe and recreate an existing dataset dir instead of
                appending to it.
            cameras: Camera names to record into the dataset. When ``None``
                (default) every named scene camera is recorded. Pass a subset
                (e.g. ``cameras=["camera1", "camera2"]``) to scope the dataset
                to exactly those views - matching the MuJoCo backend so
                ``run_policy(dataset_cameras=...)`` behaves identically on both
                engines. Names may be given in either the raw camera name or the
                schema-safe form (``/`` collapsed to ``__``); an unknown name
                fails loudly, listing the available cameras.

        Returns:
            Standard status dict. ``status="error"`` when no world exists, the
            ``lerobot`` extra is missing, or recorder init fails.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        _DatasetRecorder: Any = None
        _has_lerobot = False
        try:
            from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
            from strands_robots.dataset_recorder import has_lerobot_dataset as _check_lerobot

            _has_lerobot = _check_lerobot()
        except ImportError:
            # lerobot extra not installed; handled by the _has_lerobot guard below.
            pass

        if not _has_lerobot or _DatasetRecorder is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "start_recording produces a LeRobotDataset (parquet + video) and "
                            "requires the lerobot extra: pip install 'strands-robots[lerobot]'.\n"
                            "For plain MP4 video, pass video={'path': ...} to run_policy instead."
                        )
                    }
                ],
            }

        world = self._world
        world._backend_state["recording"] = True
        world._backend_state["trajectory"] = []
        world._backend_state["push_to_hub"] = push_to_hub

        # Resolve the on-disk dataset dir (shared by overwrite + resume logic).
        if root:
            dataset_dir = Path(root)
        elif "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
            dataset_dir = Path(repo_id)
        else:
            dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        world._backend_state["last_dataset_root"] = str(dataset_dir)

        try:
            # Resolve create-vs-resume and make the target safe for create():
            # resume an existing dataset, clear a pre-existing EMPTY root (e.g.
            # tempfile.mkdtemp()) so create() does not dead-end on FileExistsError,
            # and wipe on overwrite. See DatasetRecordingMixin._prepare_dataset_target.
            resume_existing = self._prepare_dataset_target(dataset_dir, overwrite)

            joint_names, camera_keys, camera_dims, robot_type, recording_cameras = self._collect_recording_schema()

            # Backend parity with the MuJoCo recorder: a floating-base robot
            # (humanoid / mobile) exposes full base kinematics via
            # get_observation - position (base_pos, world x,y,z incl. height),
            # orientation (base_quat, w,x,y,z), linear velocity (base_lin_vel,
            # m/s) and angular velocity (base_ang_vel, rad/s) - but the
            # observation.state schema above is derived from scalar joint names,
            # so those base signals would be dropped and a locomotion /
            # velocity-tracking / whole-body-control policy trained on the
            # dataset would be base-blind. Preserve them as per-component scalar
            # columns (base_pos.x .. base_ang_vel.z); the DatasetRecorder
            # extra_state_specs / _state_source_keys machinery flattens the
            # vector observation keys into observation.state each frame with no
            # recorder changes. Multi-robot base columns are prefixed like the
            # joint ids (``alice__base_quat.w``) to match the prefixed
            # observation keys the recording hook emits. A fixed-base arm has no
            # free base joint -> no base columns (schema unchanged).
            free_base_joints = getattr(self, "_robot_free_base_joint", {})
            multi_robot = len(world.robots) > 1
            base_state_specs: list[tuple[str, list[str]]] = []
            for rname in world.robots:
                if free_base_joints.get(rname):
                    prefix = f"{rname}__" if multi_robot else ""
                    base_state_specs.append((f"{prefix}base_pos", ["x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_quat", ["w", "x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_lin_vel", ["x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_ang_vel", ["x", "y", "z"]))
            # Full observation.state schema (scalar joints + expanded base
            # components) - used to validate a resumed dataset's on-disk schema.
            state_names_full = list(joint_names) + [f"{src}.{c}" for src, comps in base_state_specs for c in comps]

            # Optional camera scoping (parity with the MuJoCo backend). By
            # default every named scene camera is recorded; when ``cameras`` is
            # given, record exactly that subset. Names may be the raw camera
            # name (``arm0/wrist_cam``) or the schema-safe form
            # (``arm0__wrist_cam``); an unknown name fails loudly (no silent
            # drop), listing what exists. Scoping filters the ``recording_cameras``
            # tuples so the on_frame hook renders only the selected views.
            if cameras is not None:
                raw_to_safe = {src: safe for src, safe, _w, _h in recording_cameras}
                safe_to_raw = {safe: src for src, safe in raw_to_safe.items()}
                selected_safe: list[str] = []
                selected_raw: set[str] = set()
                unknown: list[str] = []
                for requested in cameras:
                    if requested in raw_to_safe:  # raw camera name
                        raw, safe = requested, raw_to_safe[requested]
                    elif requested in safe_to_raw:  # already schema-safe
                        raw, safe = safe_to_raw[requested], requested
                    else:
                        unknown.append(requested)
                        continue
                    if safe not in selected_safe:
                        selected_safe.append(safe)
                        selected_raw.add(raw)
                if unknown:
                    world._backend_state["recording"] = False
                    available = sorted(raw_to_safe)
                    return {
                        "status": "error",
                        "content": [
                            {
                                "text": (
                                    f"start_recording: unknown camera(s) {unknown} in cameras=. "
                                    f"Available scene cameras: {available}. Add them with "
                                    "add_camera(...) before recording, or omit cameras= to "
                                    "record all of them."
                                )
                            }
                        ],
                    }
                camera_keys = selected_safe
                camera_dims = {safe: camera_dims[safe] for safe in selected_safe}
                recording_cameras = [tpl for tpl in recording_cameras if tpl[0] in selected_raw]

            world._backend_state["recording_cameras"] = recording_cameras

            if resume_existing:
                logger.info("Resuming existing dataset for append: %s", dataset_dir)
                resumed = _DatasetRecorder.resume(repo_id=repo_id, root=root, task=task, vcodec=vcodec)
                self._verify_resume_schema(resumed, state_names_full, camera_keys, camera_dims)
                world._backend_state["dataset_recorder"] = resumed
            else:
                world._backend_state["dataset_recorder"] = _DatasetRecorder.create(
                    repo_id=repo_id,
                    fps=fps,
                    robot_type=robot_type,
                    joint_names=joint_names,
                    extra_state_specs=base_state_specs,
                    camera_keys=camera_keys,
                    camera_dims=camera_dims,
                    task=task,
                    root=root,
                    vcodec=vcodec,
                    video_width=self.default_width,
                    video_height=self.default_height,
                )
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Recording Newton scene to LeRobotDataset: {repo_id}\n"
                            f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                            f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                            f"Run policies to capture frames, then stop_recording to save the episode"
                        )
                    }
                ],
            }
        except Exception as e:
            world._backend_state["recording"] = False
            logger.error("Dataset recorder init failed: %s", e)
            return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}

    def _collect_recording_schema(
        self,
    ) -> tuple[list[str], list[str], dict[str, tuple[int, int]], str, list[tuple[str, str, int, int]]]:
        """Build the dataset schema from the live Newton scene.

        Returns:
            A 5-tuple of:
              * ``joint_names``: ordered state/action joint ids (namespaced
                ``robot__joint`` when more than one robot exists).
              * ``camera_keys``: sanitized camera feature names (``/`` -> ``__``).
              * ``camera_dims``: map of camera feature name -> ``(height, width)``.
              * ``robot_type``: the dataset ``robot_type`` string.
              * ``recording_cameras``: per-camera ``(source_name, safe_name,
                width, height)`` tuples the on_frame hook renders each step.
        """
        world = self._world
        assert world is not None  # guarded by start_recording
        joint_names: list[str] = []
        robot_type = "unknown"
        multi_robot = len(world.robots) > 1
        free_base = getattr(self, "_robot_free_base_joint", {})
        for rname, robot in world.robots.items():
            # Exclude the floating base's free joint from the scalar joint
            # schema: its 6-DoF state is recorded as the structured base_*
            # columns below and get_observation no longer emits it as a scalar,
            # so a floating_base_joint scalar column would be dead/degenerate.
            # Mirrors get_observation / get_robot_state.
            free_short = free_base.get(rname)
            scalar_jn = [jn for jn in robot.joint_names if jn != free_short]
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in scalar_jn)
            else:
                joint_names.extend(scalar_jn)
            robot_type = robot.data_config or rname

        camera_keys: list[str] = []
        camera_dims: dict[str, tuple[int, int]] = {}
        recording_cameras: list[tuple[str, str, int, int]] = []
        for cam_name, cam in world.cameras.items():
            safe_name = cam_name.replace("/", "__")
            width = int(getattr(cam, "width", self.default_width))
            height = int(getattr(cam, "height", self.default_height))
            camera_keys.append(safe_name)
            camera_dims[safe_name] = (height, width)
            recording_cameras.append((cam_name, safe_name, width, height))
        return joint_names, camera_keys, camera_dims, robot_type, recording_cameras

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Build the per-step ``on_frame`` recording hook for Newton.

        Returns an ``on_frame(step, observation, action)`` closure that, while a
        recording session is active, augments the joint-state observation with a
        rendered frame for each declared camera and forwards the frame to the
        active :class:`DatasetRecorder`. In multi-robot scenes scalar
        observation/action keys are namespaced (``robot__joint``) to match the
        schema declared in :meth:`start_recording`; camera ndarrays keep their
        sanitized names.

        Returns ``None`` when there is no world or the robot is unknown, so the
        base run-policy loop runs without recording.
        """
        from strands_robots.simulation.policy_runner import _extract_frame_ndarray

        world = self._world
        if world is None or robot_name not in world.robots:
            return None

        robot = world.robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0
        multi_robot = len(world.robots) > 1

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            robot.policy_steps = step + 1
            if not world._backend_state.get("recording", False):
                return
            rec = world._backend_state.get("dataset_recorder")
            if rec is None:
                return

            obs: dict[str, Any] = dict(observation)
            for source_name, safe_name, width, height in world._backend_state.get("recording_cameras", []):
                render_result = self.render(camera_name=source_name, width=width, height=height)
                img = _extract_frame_ndarray(render_result)
                if img is not None:
                    obs[safe_name] = img

            if multi_robot:
                import numpy as np

                obs = {(k if isinstance(v, np.ndarray) else f"{robot_name}__{k}"): v for k, v in obs.items()}
                act = {f"{robot_name}__{k}": v for k, v in action.items()}
                rec.add_frame(observation=obs, action=act, task=instruction)
            else:
                rec.add_frame(observation=obs, action=action, task=instruction)

        return _hook
