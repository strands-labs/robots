"""Recording mixin - start/stop trajectory recording to LeRobotDataset."""

import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.mujoco.backend import _ensure_mujoco

logger = logging.getLogger(__name__)


class RecordingMixin:
    """Trajectory recording mixed into ``Simulation``.

    Writes per-step observations + actions + instruction to a LeRobotDataset
    via ``start_recording`` / ``stop_recording`` and the ``on_frame`` hook
    in ``PolicyRunner``. Separately from that, ``start_cameras_recording``
    dumps raw per-camera MP4s.

    **Coupling** (see simulation.py top-level docstring): mixin reaches
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

    def start_recording(
        self,
        repo_id: str = "local/sim_recording",
        task: str = "",
        fps: int = 30,
        root: str | None = None,
        push_to_hub: bool = False,
        vcodec: str = "libsvtav1",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Start recording to LeRobotDataset format (parquet + per-camera MP4).

        Requires the ``lerobot`` extra for the dataset schema. If you only
        need plain MP4 video (no dataset schema, no policy-training metadata),
        use :meth:`start_cameras_recording` - it runs under the
        ``[sim-mujoco]`` extra alone (imageio-ffmpeg backend).

        Raises:
            Friendly error when ``lerobot`` is not installed, directing the
            caller to :meth:`start_cameras_recording` or to install the
            optional extra.
        """
        if self._world is None or self._world._model is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world (or load_scene) first."}]}

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

        # Multi-episode append: when NOT overwriting and a dataset already
        # exists on disk, resume it (append new episodes) instead of calling
        # create() - which hard-fails with FileExistsError (B12). resume() is
        # the only correct append path in LeRobot 0.5.2+ (the plain constructor
        # is read-only). When overwrite=True, wipe and recreate from scratch.
        resume_existing = (
            not overwrite and dataset_dir.exists() and dataset_dir.is_dir() and (dataset_dir / "meta").exists()
        )

        try:
            if overwrite:
                if dataset_dir.exists() and dataset_dir.is_dir():
                    shutil.rmtree(dataset_dir)
                    logger.info("Removed existing dataset dir: %s", dataset_dir)

            # Collect joint names from every robot. When the scene contains
            # more than one robot (e.g. multi-agent dual-task recording), prefix
            # each joint with the robot's instance name (``alice__shoulder_pan``)
            # so the dataset schema has unique joint ids per agent. Single-robot
            # scenes keep the clean ``shoulder_pan`` names for backwards compat.
            joint_names: list[str] = []
            camera_keys: list[str] = []
            robot_type = "unknown"
            multi_robot = len(self._world.robots) > 1
            for rname, robot in self._world.robots.items():
                if multi_robot:
                    joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                else:
                    joint_names.extend(robot.joint_names)
                robot_type = robot.data_config or rname

            mj = _ensure_mujoco()
            # Declare each camera in the dataset schema at the SAME
            # resolution it actually renders at. Cameras added via add_camera
            # carry their own width/height (e.g. 256x256 for a LIBERO VLA),
            # which can differ from the sim's default render size. Declaring
            # everything at default_width/height made add_frame reject frames
            # ("shape (256,256,3) != expected (3,480,640)") and, with strict
            # recording, abort the whole episode. We map each safe camera name
            # to its real (height, width) so _build_features sizes it correctly.
            camera_dims: dict[str, tuple[int, int]] = {}
            for i in range(self._world._model.ncam):
                cam_name = mj.mj_id2name(self._world._model, mj.mjtObj.mjOBJ_CAMERA, i)
                if not cam_name:
                    continue
                # LeRobot feature names can't contain '/' (reserved for
                # nested-feature addressing). When a robot injects a
                # namespaced camera (e.g. ``arm0/wrist_cam``), collapse
                # the separator to ``__`` for the dataset schema.
                safe_name = cam_name.replace("/", "__")
                camera_keys.append(safe_name)
                cam_info = self._world.cameras.get(cam_name) or self._world.cameras.get(safe_name)
                if cam_info is not None:
                    camera_dims[safe_name] = (int(cam_info.height), int(cam_info.width))
                else:
                    camera_dims[safe_name] = (int(self.default_height), int(self.default_width))

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
                self._verify_resume_schema(resumed, joint_names, camera_keys, camera_dims)
                self._world._backend_state["dataset_recorder"] = resumed
            else:
                self._world._backend_state["dataset_recorder"] = _DatasetRecorder.create(
                    repo_id=repo_id,
                    fps=fps,
                    robot_type=robot_type,
                    joint_names=joint_names,
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

    def _verify_resume_schema(
        self,
        recorder: Any,
        joint_names: list[str],
        camera_keys: list[str],
        camera_dims: dict[str, tuple[int, int]],
    ) -> None:
        """Verify the live scene matches the resumed dataset's on-disk schema.

        ``DatasetRecorder.resume`` inherits the feature schema from disk; it does
        not validate it against the current scene. If the caller added a robot,
        renamed a joint, or changed a camera resolution between episodes, the
        mismatch would only surface as a cryptic per-feature shape error on the
        next ``add_frame``. Compare here and raise a clear schema diff instead.

        Compares the expected ``observation.state`` joint names and each
        ``observation.images.*`` camera (presence + height/width). Best-effort:
        if the dataset does not expose ``features`` we skip silently rather than
        block a valid resume on an unexpected LeRobot layout.

        Args:
            recorder: The resumed DatasetRecorder.
            joint_names: Joint names the current scene will emit (namespaced for
                multi-robot scenes).
            camera_keys: Sanitized camera feature names the current scene emits.
            camera_dims: Map of camera feature name -> (height, width).

        Raises:
            ValueError: If the live scene schema diverges from the on-disk one.
        """
        features = getattr(getattr(recorder, "dataset", None), "features", None)
        if not isinstance(features, dict):
            return

        diffs: list[str] = []

        state = features.get("observation.state")
        if isinstance(state, dict):
            disk_joints = list(state.get("names") or [])
            if disk_joints and disk_joints != list(joint_names):
                diffs.append(f"observation.state joints differ: on-disk={disk_joints} vs scene={list(joint_names)}")

        for cam in camera_keys:
            key = f"observation.images.{cam}"
            disk_cam = features.get(key)
            if not isinstance(disk_cam, dict):
                diffs.append(f"camera '{cam}' is in the scene but not in the on-disk schema")
                continue
            shape = disk_cam.get("shape")
            scene_dim = camera_dims.get(cam)
            if shape and len(shape) == 3 and scene_dim is not None:
                _, disk_h, disk_w = shape
                scene_h, scene_w = scene_dim
                if (int(disk_h), int(disk_w)) != (int(scene_h), int(scene_w)):
                    diffs.append(
                        f"camera '{cam}' resolution differs: on-disk={(disk_h, disk_w)} vs scene={(scene_h, scene_w)}"
                    )

        disk_cams = {k[len("observation.images.") :] for k in features if k.startswith("observation.images.")}
        for cam in disk_cams - set(camera_keys):
            diffs.append(f"camera '{cam}' is in the on-disk schema but not in the current scene")

        if diffs:
            raise ValueError(
                "Cannot resume recording: the current scene does not match the existing dataset schema. "
                "Use overwrite=True for a fresh dataset, or restore the original scene. Differences:\n  - "
                + "\n  - ".join(diffs)
            )

    def stop_recording(self, output_path: str | None = None) -> dict[str, Any]:
        """Stop recording and save episode to LeRobotDataset.

        idempotent - calling when not recording succeeds with a
        'Was not recording' message so callers can safely call it unconditionally.
        """
        if self._world is None or not self._world._backend_state.get("recording", False):
            return {"status": "success", "content": [{"text": "Was not recording."}]}

        self._world._backend_state["recording"] = False
        recorder = self._world._backend_state.get("dataset_recorder", None)

        if recorder is None:
            return {"status": "error", "content": [{"text": "No dataset recorder active."}]}

        recorder.save_episode()
        push_result = None
        if self._world._backend_state.get("push_to_hub", False):
            push_result = recorder.push_to_hub(tags=["strands-robots", "sim"])

        repo_id = recorder.repo_id
        frame_count = recorder.frame_count
        episode_count = recorder.episode_count
        root = recorder.root

        recorder.finalize()
        self._world._backend_state["dataset_recorder"] = None
        self._world._backend_state["trajectory"] = []

        text = (
            f"Episode saved to LeRobotDataset\n"
            f"{repo_id} -- {frame_count} frames, {episode_count} episode(s)\n"
            f"Local: {root}"
        )
        if push_result and push_result.get("status") == "success":
            text += "\nPushed to HuggingFace Hub"

        return {"status": "success", "content": [{"text": text}]}

    def get_recording_status(self) -> dict[str, Any]:
        """Returns success in every lifecycle state (no world / not
        recording / recording) with a distinguishing message so callers can
        poll it unconditionally without try/except."""
        if self._world is None:
            return {
                "status": "success",
                "content": [{"text": "No world. Call create_world to start recording."}],
            }

        recording = self._world._backend_state.get("recording", False)
        steps = len(self._world._backend_state.get("trajectory", []))

        if recording:
            text = f"[recording] {steps} steps captured"
        else:
            text = f"[idle] Not recording (last episode: {steps} steps)"

        return {
            "status": "success",
            "content": [{"text": text}],
        }
