"""MuJoCo recording mixin - schema declaration + raw camera MP4 capture.

The engine-independent recording lifecycle (stop/save/status/stream and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`.
This subclass adds the MuJoCo-specific ``start_recording`` (enumerates joints
and cameras from the live ``MjModel`` to declare the dataset schema) and its
resume-schema guard.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _NO_WORLD_MSG, _ensure_mujoco
from strands_robots.simulation.recording import DatasetRecordingMixin

logger = logging.getLogger(__name__)


class RecordingMixin(DatasetRecordingMixin):
    """MuJoCo trajectory recording mixed into ``Simulation``.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and adds the MuJoCo schema declaration:
    ``start_recording`` reads the live ``MjModel`` to enumerate joints and
    cameras (with their real render resolutions) before creating/resuming the
    LeRobotDataset. Per-step frames are fed by the ``on_frame`` hook built in
    :mod:`simulation`. Separately, ``start_cameras_recording`` dumps raw
    per-camera MP4s.

    **Coupling** (see the :mod:`simulation` top-level docstring): mixin reaches
    into ``self._world`` (trajectory buffer + dataset_recorder live in
    ``_world._backend_state``). ``TYPE_CHECKING`` stub below exists so mypy
    accepts the ``_world`` lookup; it is a documentary contract, not an
    enforceable protocol.
    """

    if TYPE_CHECKING:
        from strands_robots.simulation.models import SimWorld

        _world: "SimWorld | None"
        default_width: int
        default_height: int

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Actuator-ordered action keys for ``robot_name`` (concrete on SimEngine)."""

        def _robot_free_base_joint_id(self, model: Any, robot: Any) -> int:
            """Free-base joint id for ``robot`` or -1 (concrete on RenderingMixin)."""

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
        """Start recording to LeRobotDataset format (parquet + per-camera MP4).

        Requires the ``lerobot`` extra for the dataset schema. If you only
        need plain MP4 video (no dataset schema, no policy-training metadata),
        use :meth:`start_cameras_recording` - it runs under the
        ``[sim-mujoco]`` extra alone (imageio-ffmpeg backend).

        Frame/action alignment: while this session is open, :meth:`run_policy`
        records one ``(observation, action)`` frame per control step with the
        observation re-sampled at that step, so each recorded image + state
        pairs with the action actually taken from it. This holds even for
        chunk-emitting policies (ACT, diffusion, pi0/pi0.5, SmolVLA, MolmoAct2),
        whose open-loop action chunk drains across many steps from a single
        ``get_actions`` call - the recorded frames advance with the robot
        rather than freezing on the chunk-start observation.

        Args:
            root: On-disk dataset directory (defaults to the LeRobot cache under
                ``repo_id``). An existing EMPTY directory (e.g. from
                ``tempfile.mkdtemp()``) is accepted and recorded into; an
                existing dataset is resumed/appended (see ``overwrite``).
            overwrite: When True, wipe any existing dataset at ``root`` and
                record from scratch. When False (default) an existing dataset is
                appended to; a non-empty, non-dataset ``root`` is reported as an
                error rather than clobbered.
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), which decodes everywhere - including OpenCV's
                VideoCapture, used by many downstream VLM video readers - so a
                recorded episode can be replayed/reasoned about without
                transcoding. Use "libsvtav1" (AV1) for smaller files in
                storage-constrained training pipelines; LeRobot read-back
                (torchcodec/pyav) handles AV1, but OpenCV wheels commonly cannot
                decode it and silently yield 0 frames.
            cameras: Camera names to record into the dataset. When ``None``
                (default) every scene camera is recorded - which includes the
                implicit ``default`` overview camera (a one-time warning is
                logged when it is swept in alongside real sensor cameras, since
                no policy declares ``observation.images.default``). Pass an
                explicit subset to
                record exactly the views a policy declares (e.g.
                ``cameras=["camera1", "camera2", "camera3"]`` for a 3-camera
                SmolVLA dataset) and keep the stray ``default`` view out of the
                schema. Names may be raw (``arm0/wrist_cam``) or schema-safe
                (``arm0__wrist_cam``); an unknown name fails loudly.

        Raises:
            Friendly error when ``lerobot`` is not installed, directing the
            caller to :meth:`start_cameras_recording` or to install the
            optional extra.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": _NO_WORLD_MSG}]}

        _DatasetRecorder: Any = None
        _has_lerobot = False
        try:
            from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
            from strands_robots.dataset_recorder import has_lerobot_dataset as _check_lerobot

            _has_lerobot = _check_lerobot()
        except ImportError:
            pass

        if not _has_lerobot or _DatasetRecorder is None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "start_recording produces a LeRobotDataset (parquet + video) and "
                            "requires the lerobot extra. For plain MP4 video under the "
                            "[sim-mujoco] extra alone, use start_cameras_recording instead.\n"
                            "\n"
                            "  - Dataset + policy training data:  pip install 'strands-robots[lerobot]'\n"
                            "  - Plain MP4 only:                  start_cameras_recording(cameras=..., output_dir=...)"
                        )
                    }
                ],
            }

        self._world._backend_state["recording"] = True
        self._world._backend_state["trajectory"] = []
        self._world._backend_state["push_to_hub"] = push_to_hub

        # Resolve the on-disk dataset dir (shared by overwrite + resume logic).
        if root:
            dataset_dir = Path(root)
        elif "/" not in repo_id or repo_id.startswith("/") or repo_id.startswith("./"):
            dataset_dir = Path(repo_id)
        else:
            dataset_dir = Path.home() / ".cache" / "huggingface" / "lerobot" / repo_id
        # Stash the resolved root so verify_dataset_episodes can read the parquet
        # after stop_recording has finalized the dataset and dropped the recorder.
        self._world._backend_state["last_dataset_root"] = str(dataset_dir)

        # Multi-episode append: when NOT overwriting and a dataset already
        # exists on disk, resume it (append new episodes) instead of calling
        # create() - which hard-fails with FileExistsError. resume() is the
        # only correct append path in LeRobot 0.5.2+ (the plain constructor is
        # read-only). A pre-existing EMPTY root (e.g. tempfile.mkdtemp()) is
        # cleared so create() does not dead-end on its own existence guard;
        # overwrite=True wipes and recreates from scratch. See
        # DatasetRecordingMixin._prepare_dataset_target.
        try:
            resume_existing = self._prepare_dataset_target(dataset_dir, overwrite)

            # Collect joint names from every robot. When the scene contains
            # more than one robot (e.g. multi-agent dual-task recording), prefix
            # each joint with the robot's instance name (``alice__shoulder_pan``)
            # so the dataset schema has unique joint ids per agent. Single-robot
            # scenes keep the clean ``shoulder_pan`` names for backwards compat.
            joint_names: list[str] = []
            # Action columns are keyed by the ACTUATORS the rollout loops emit
            # (SimEngine.robot_action_keys), not the joint names. These diverge
            # whenever a robot has passive/mimic joints with no driving actuator
            # or a tendon-driven gripper (an actuator with no matching joint);
            # declaring the action schema from joint names there records all-zero
            # action columns because add_frame can't match the actuator keys.
            action_names: list[str] = []
            camera_keys: list[str] = []
            robot_type = "unknown"
            multi_robot = len(self._world.robots) > 1
            for rname, robot in self._world.robots.items():
                if multi_robot:
                    joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                    action_names.extend(f"{rname}__{ak}" for ak in self.robot_action_keys(rname))
                else:
                    joint_names.extend(robot.joint_names)
                    action_names.extend(self.robot_action_keys(rname))
                robot_type = robot.data_config or rname

            mj = _ensure_mujoco()
            # A floating-base robot (humanoid / mobile) exposes full base
            # kinematics via get_observation - position (base_pos, world x,y,z
            # incl. height), orientation (base_quat, w,x,y,z), linear velocity
            # (base_lin_vel, m/s) and angular velocity (base_ang_vel, rad/s) -
            # but the observation.state schema above is derived from scalar joint
            # names, so those base signals would be dropped. Preserve them as
            # per-component scalar columns so a locomotion / velocity-tracking /
            # whole-body-control policy trained on the dataset is not base-blind.
            # Detected via the shared free-base joint finder; multi-robot base
            # columns are prefixed like joint ids (``alice__base_quat.w``) to
            # match the prefixed observation keys the recording hook emits.
            base_state_specs: list[tuple[str, list[str]]] = []
            for rname, robot in self._world.robots.items():
                if self._robot_free_base_joint_id(self._world._model, robot) >= 0:
                    prefix = f"{rname}__" if multi_robot else ""
                    base_state_specs.append((f"{prefix}base_pos", ["x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_quat", ["w", "x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_lin_vel", ["x", "y", "z"]))
                    base_state_specs.append((f"{prefix}base_ang_vel", ["x", "y", "z"]))
            # Full observation.state schema names (scalar joints + expanded base
            # components) - used to validate a resumed dataset's on-disk schema.
            state_names_full = list(joint_names) + [f"{src}.{c}" for src, comps in base_state_specs for c in comps]

            # Declare each camera in the dataset schema at the SAME
            # resolution it actually renders at. Cameras added via add_camera
            # carry their own width/height (e.g. 256x256 for a LIBERO VLA),
            # which can differ from the sim's default render size. Declaring
            # everything at default_width/height made add_frame reject frames
            # ("shape (256,256,3) != expected (3,480,640)") and, with strict
            # recording, abort the whole episode. We map each safe camera name
            # to its real (height, width) so _build_features sizes it correctly.
            camera_dims: dict[str, tuple[int, int]] = {}
            # Raw MuJoCo camera name -> schema-safe name. Kept so the run_policy
            # frame hook can map a caller-requested ``cameras`` subset (which may
            # use either form) back to the RAW observation key it must keep.
            raw_to_safe: dict[str, str] = {}
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name:
                    continue
                # LeRobot feature names can't contain '/' (reserved for
                # nested-feature addressing). When a robot injects a
                # namespaced camera (e.g. ``arm0/wrist_cam``), collapse
                # the separator to ``__`` for the dataset schema.
                safe_name = cam_name.replace("/", "__")
                raw_to_safe[cam_name] = safe_name
                camera_keys.append(safe_name)
                cam_info = self._world.cameras.get(cam_name) or self._world.cameras.get(safe_name)
                if cam_info is not None:
                    camera_dims[safe_name] = (int(cam_info.height), int(cam_info.width))
                else:
                    camera_dims[safe_name] = (int(self.default_height), int(self.default_width))

            # Optional camera scoping. By default EVERY scene camera is recorded,
            # which sweeps in the implicit ``default`` overview camera and any
            # view the trained policy never declared - bloating the dataset and
            # producing image features that do not match the policy's
            # ``input_features``. When ``cameras`` is given, record exactly that
            # subset. Names may be given in either the raw MuJoCo form
            # (``arm0/wrist_cam``) or the schema-safe form (``arm0__wrist_cam``);
            # an unknown name fails loudly (no silent drop) listing what exists.
            record_raw_cameras: set[str] | None = None
            if cameras is not None:
                safe_to_raw = {safe: raw for raw, safe in raw_to_safe.items()}
                selected_safe: list[str] = []
                record_raw_cameras = set()
                unknown: list[str] = []
                for requested in cameras:
                    if requested in raw_to_safe:  # raw name
                        raw, safe = requested, raw_to_safe[requested]
                    elif requested in safe_to_raw:  # already schema-safe
                        raw, safe = safe_to_raw[requested], requested
                    else:
                        unknown.append(requested)
                        continue
                    if safe not in selected_safe:
                        selected_safe.append(safe)
                        record_raw_cameras.add(raw)
                if unknown:
                    self._world._backend_state["recording"] = False
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
            # Stash the scoped RAW camera names so the run_policy frame hook drops
            # un-recorded camera arrays before add_frame (None -> record all).
            self._world._backend_state["recording_cameras"] = record_raw_cameras

            # Doctrine: warn loudly, never silently surprise. On the record-all
            # path (``cameras is None``) the implicit ``default`` overview camera
            # that ``create_world`` auto-adds is swept into the dataset alongside
            # the real, user-added sensor cameras. That produces an
            # ``observation.images.default`` view no trained policy declares -
            # bloating the dataset and breaking feature-matching against a
            # policy's ``input_features``. Keep recording it (back-compat), but
            # surface it instead of including it silently, and point at the
            # ``cameras=`` escape hatch.
            if cameras is None and "default" in camera_keys and len(camera_keys) > 1:
                sensor_cams = [c for c in camera_keys if c != "default"]
                logger.warning(
                    "start_recording: recording the implicit 'default' overview "
                    "camera into observation.images.default alongside %d sensor "
                    "camera(s) %s. The 'default' view is not a sensor any policy "
                    "declares; it bloats the dataset and will not match a policy's "
                    "input_features. Pass cameras=%r to record only your sensors.",
                    len(sensor_cams),
                    sensor_cams,
                    sensor_cams,
                )

            assert _DatasetRecorder is not None  # checked above
            if resume_existing:
                # Append to the existing dataset (schema inherited from disk).
                logger.info("Resuming existing dataset for append: %s", dataset_dir)
                resumed = _DatasetRecorder.resume(
                    repo_id=repo_id,
                    root=root,
                    task=task,
                    vcodec=vcodec,
                )
                # resume() inherits the feature schema from disk; it does NOT
                # check it against the CURRENT scene. Adding a robot or swapping
                # a camera resolution between episodes would otherwise yield a
                # cryptic per-feature shape error on the next add_frame. Compare
                # up front and raise a clear schema-diff instead.
                self._verify_resume_schema(resumed, state_names_full, camera_keys, camera_dims, action_names)
                self._world._backend_state["dataset_recorder"] = resumed
            else:
                self._world._backend_state["dataset_recorder"] = _DatasetRecorder.create(
                    repo_id=repo_id,
                    fps=fps,
                    robot_type=robot_type,
                    joint_names=joint_names,
                    action_names=action_names,
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
                            f"Recording to LeRobotDataset: {repo_id}\n"
                            f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                            f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                            f"Run policies to capture frames, then stop_recording to save episode"
                        )
                    }
                ],
            }
        except Exception as e:
            self._world._backend_state["recording"] = False
            logger.error("Dataset recorder init failed: %s", e)
            return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}
