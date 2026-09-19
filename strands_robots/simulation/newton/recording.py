"""Newton recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset`` and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`,
which is backend-agnostic. This subclass adds the Newton-specific parts:

* :meth:`start_recording` declares the dataset schema from the live Newton
  scene - joint names from every robot (namespaced for multi-robot scenes) and
  the named cameras registered on the world.
* :meth:`_make_run_policy_hook` returns the ``on_frame`` closure the shared
  :class:`~strands_robots.simulation.base.SimEngine` run-policy loop calls every
  control step. It feeds joint state + action + rendered camera frames to the
  active :class:`~strands_robots.dataset_recorder.DatasetRecorder`.
* :meth:`_release_run_policy_hook` lowers the ``policy_running`` flag the hook
  builder raised, when the rollout that hook served ends. The claim and its
  release are one contract, and only the shared facade knows where the rollout
  ends, so it calls this in a ``finally`` around every rollout it drives.

The recorder, episode-boundary flushing (``save_episode``), and the canonical
parquet-correctness contract are identical to the MuJoCo backend - the
``DatasetRecorder`` is engine-independent, so a Newton recording produces the
same LeRobot v3 dataset layout (``meta/info.json`` + per-episode parquet +
per-camera MP4).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from strands_robots.simulation.models import registered
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    camera_schema_key_collision_error,
    dataset_recording_option_error,
    dataset_recording_posture_error,
    recorded_cameras_line,
    undriven_robot_state,
)
from strands_robots.utils import camera_schema_key, name_list_error

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

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Type-only stub for the engine-provided action-key accessor."""
            return []

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
        ``world.cameras`` (with their real render resolutions). In a multi-robot scene every robot's state columns are
        declared, and a single-policy rollout fills the ones it does not
        drive from the engine at each step, so a declared
        ``observation.state`` column is a measurement rather than a zero
        the robot is not at
        (:func:`~strands_robots.simulation.recording.undriven_robot_state`).
        The matching *action* columns are a separate question - no command
        was issued to a robot this rollout does not drive - and are
        unchanged. Per-step frames
        are then captured by the ``on_frame`` hook
        (:meth:`_make_run_policy_hook`) during ``run_policy``.

        When no named cameras are registered the dataset records joint state and
        action only (a valid proprio-only LeRobot dataset); camera columns are
        added automatically once cameras are registered on the world.

        Requires the ``lerobot`` extra for the dataset schema.

        Args:
            repo_id: HuggingFace dataset id (``owner/name``) or a local path. The
                directory it records into is resolved by
                :func:`~strands_robots.dataset_recorder.resolve_dataset_dir` -
                the same resolver ``DatasetRecorder.create`` uses - so an
                ``owner/name`` id lands in ``$HF_LEROBOT_HOME/{repo_id}`` while a
                value that is itself a path is taken as the directory. That home
                is read from LeRobot's own ``HF_LEROBOT_HOME`` constant, so
                relocating it moves both this recording and where
                ``LeRobotDataset`` later reads the dataset back from.
            task: Task description for frames that do not carry their own. It
                is the middle of a three-level chain owned by
                :meth:`~strands_robots.dataset_recorder.DatasetRecorder.add_frame`:
                the task passed with a frame wins, then this value, then the
                literal ``"untitled"``. Every rollout hook passes
                ``run_policy(instruction=...)`` as the frame task, so a non-empty
                instruction overrides this value; supply neither and each frame is
                annotated ``"untitled"``, which conditions a
                language-conditioned policy on a constant instruction.
            fps: Recording frame rate. Must be a positive whole number;
                a rate no dataset can be written at is rejected up front. When
                an existing dataset is RESUMED (``overwrite=False``) it must
                equal that dataset's on-disk rate, which a resume cannot change.
            root: Explicit on-disk dataset directory, used verbatim - it replaces
                the ``repo_id`` resolution above rather than being joined to it.
                See :func:`~strands_robots.dataset_recorder.resolve_dataset_dir`
                for the full precedence.
            push_to_hub: Publish to the Hub at ``stop_recording``. Must be a
                boolean - a publication posture is not read by truthiness
                (:func:`~strands_robots.simulation.recording.dataset_recording_posture_error`).
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), universally decodable including by OpenCV's
                VideoCapture (used by many downstream VLM video readers). Use
                "libsvtav1" (AV1) for smaller files in storage-constrained
                training pipelines; LeRobot read-back handles AV1 but OpenCV
                wheels commonly cannot decode it and silently yield 0 frames.
            overwrite: When True, wipe any existing dataset at the resolved
                directory and record from scratch. When False (default) an
                existing dataset is RESUMED (episodes appended), a pre-existing
                EMPTY directory (e.g. from ``tempfile.mkdtemp()``) is cleared and
                recorded into, and a non-empty non-dataset directory is reported
                as an error rather than clobbered - the four outcomes of
                :meth:`~strands_robots.simulation.recording.DatasetRecordingMixin._prepare_dataset_target`.
                Must be a boolean: a truthy non-boolean opt-out reached the
                wipe branch and deleted the dataset it was meant to append
                to (:func:`~strands_robots.simulation.recording.dataset_recording_posture_error`).
            cameras: Camera names to record into the dataset. When ``None``
                (default) every named scene camera is recorded. Pass a subset
                (e.g. ``cameras=["camera1", "camera2"]``) to scope the dataset
                to exactly those views - matching the MuJoCo backend so
                ``run_policy(dataset_cameras=...)`` behaves identically on both
                engines. Names may be given in either the raw camera name or the
                schema-safe form (``/`` collapsed to ``__``); an unknown name
                fails loudly, listing the available cameras, and is refused before any dataset is
                created, resumed or wiped, so a typo costs nothing even under
                ``overwrite=True``. Two scene cameras whose
                names collapse onto one dataset column (``arm0/wrist`` and
                ``arm0__wrist``) are refused before any dataset is created,
                because the column would be named after whichever of them lost
                it.

        Returns:
            Standard status dict. ``status="error"`` when no world exists, the
            ``lerobot`` extra is missing, or recorder init fails.
        """
        if self._world is None or self._model is None:
            return {"status": "error", "content": [{"text": "No world. Call create_world first."}]}

        # Reject an fps no dataset can be written at before creating or
        # resuming the recorder: an unusable rate was reported as success and
        # then cost the caller the whole episode (see
        # dataset_recording_option_error). Checked ahead of the lerobot-extra
        # probe so the same caller mistake reports the same way regardless of
        # which optional extras this install has.
        if error := dataset_recording_option_error("start_recording", fps):
            return error
        # ``push_to_hub`` and ``overwrite`` select postures, not quantities, so
        # each is checked on the shared boolean-flag domain before any dataset is
        # created, resumed or wiped - and before the lerobot-extra probe, so the
        # same caller mistake reports the same way on every install. Read by
        # truthiness both failed toward the branch the caller was opting out of:
        # ``overwrite="false"`` deleted the dataset it was meant to append to,
        # and ``push_to_hub="false"`` published it (see
        # dataset_recording_posture_error).
        for _flag, _value in (("push_to_hub", push_to_hub), ("overwrite", overwrite)):
            if error := dataset_recording_posture_error("start_recording", _flag, _value):
                return error
        # ``cameras`` names an ordered list of DISTINCT camera names, so it is
        # refused on the shared name-list domain before any dataset is created. Neither
        # mistake this catches could be honored as written: a single name passed
        # as a bare string is iterable per character, so it was read as one
        # camera per letter, and a repeated name collapsed in the feature dict, declaring
        # fewer camera columns than the caller asked for.
        if cameras and (text := name_list_error(cameras, "cameras", "start_recording")):
            return {"status": "error", "content": [{"text": text}]}

        # Reject a rate a rollout already in flight is not capturing at. The
        # rollout entry points cover the record-then-rollout ordering; this is
        # the same disagreement with the calls the other way round, refused
        # before any dataset is created so a refusal leaves nothing on disk.
        if error := self._validate_recording_start_rate(fps, "start_recording"):
            return error

        _DatasetRecorder: Any = None
        unavailable: str | None = None
        try:
            from strands_robots.dataset_recorder import DatasetRecorder as _DatasetRecorder
            from strands_robots.dataset_recorder import lerobot_dataset_import_error

            unavailable = lerobot_dataset_import_error()
        except ImportError as exc:
            # strands_robots.dataset_recorder itself did not import (a partial or
            # drifted install); report that rather than blaming the lerobot extra.
            unavailable = f"strands_robots.dataset_recorder is unavailable ({exc})."
        if unavailable is None and _DatasetRecorder is None:
            unavailable = "strands_robots.dataset_recorder did not provide DatasetRecorder."

        if unavailable is not None:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            "start_recording produces a LeRobotDataset (parquet + video), which "
                            "needs lerobot's dataset stack:\n"
                            "\n"
                            f"  {unavailable}\n"
                            "\n"
                            "For plain MP4 video, pass video={'path': ...} to run_policy instead."
                        )
                    }
                ],
            }

        # A dataset column is named by camera_schema_key, which collapses a
        # camera's "/" namespace separator to "__" because a LeRobot feature name
        # cannot contain "/". That mapping is not injective, so two scene cameras
        # can name one column - and the three ways of asking then disagree: every
        # camera is refused downstream as a repeated camera_keys entry, both
        # spellings requested silently drops one, and one spelling succeeds with
        # the column named after whichever camera lost. The ambiguity belongs to
        # the scene rather than to one way of recording it, so it is refused once
        # here. Reading the scene's cameras is an engine call, so this sits after
        # the dataset-stack probe (whose block is reachable on an install with no
        # engine at all, and diagnoses the missing extra without one) and ahead of
        # any session state or dataset target - a refusal leaves nothing set and
        # nothing on disk.
        if error := camera_schema_key_collision_error("start_recording", list(self._world.cameras)):
            return error

        # A second start while one recording is live used to fall through here
        # too: it replaced the recorder object (the frames buffered since the
        # last save_episode went with it - never saved, never mentioned) and,
        # when the new dataset then refused, left ``recording`` False with the
        # first session's frames gone as well. Refused on the shared domain, so
        # the three backends answer a second start identically.
        if error := self._already_recording_error("start_recording", repo_id):
            return error

        world = self._world
        world._backend_state["recording"] = True
        world._backend_state["trajectory"] = []
        world._backend_state["push_to_hub"] = push_to_hub

        # Resolve the on-disk dataset dir (shared by overwrite + resume logic)
        # through the same resolver ``DatasetRecorder.create`` uses, as the MuJoCo
        # and Isaac backends' ``start_recording`` already do. The three-branch
        # copy this replaces matched it on the first two branches and hard-coded
        # ``~/.cache/huggingface/lerobot`` on the third, so it ignored
        # ``$HF_LEROBOT_HOME`` - the override LeRobot itself honours and the only
        # way to move the dataset home. Every consumer of the value then read a
        # directory this session never writes to: ``overwrite=True`` removed a
        # dataset OUTSIDE the configured home while leaving the addressed one for
        # ``create()`` to remove, the resume probe missed an existing dataset so
        # appending dead-ended in ``FileExistsError`` advising the caller to
        # bypass this method, and ``last_dataset_root`` - which
        # ``stop_recording(bucket=...)`` syncs and ``verify_dataset_episodes``
        # reads once the recorder is dropped - named the stale path.
        dataset_dir = self._stash_dataset_target(repo_id, root)

        try:
            (
                joint_names,
                action_names,
                camera_keys,
                camera_dims,
                robot_type,
                recording_cameras,
            ) = self._collect_recording_schema()

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
            # Scene camera name -> dataset column key, in dataset column order, and
            # the scene's full camera list. start_recording's reply names the
            # cameras by their SCENE name (the spelling every camera surface
            # answers for) and reads "no camera recorded" off the scene rather
            # than assuming a cause.
            scene_cameras = [src for src, _safe, _w, _h in recording_cameras]
            recorded_cameras = {src: safe for src, safe, _w, _h in recording_cameras}

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
                recorded_cameras = {safe_to_raw[safe]: safe for safe in selected_safe}

            world._backend_state["recording_cameras"] = recording_cameras

            # Create-vs-resume, and the wipe it can perform, are deferred to here
            # rather than opened with. ``overwrite=True`` deletes the dataset being
            # replaced, so every refusal this method can still make is made above it:
            # the camera scoping just above used to sit behind this line, and a single
            # unknown name in ``cameras=`` refused the call after the existing dataset
            # had already been removed - the refusal's own remedy ("Add them with
            # add_camera(...) ... or omit cameras=") asks for a retry against the data
            # that same call destroyed. Nothing between the target resolution above and
            # this line reads or writes the dataset directory (the schema is read from
            # the scene), so the success path is unchanged. Resume an existing dataset,
            # clear a pre-existing EMPTY root (e.g. tempfile.mkdtemp()) so create() does
            # not dead-end on FileExistsError, and wipe on overwrite - the four outcomes
            # of DatasetRecordingMixin._prepare_dataset_target. This is the ordering
            # camera_schema_key_collision_error already establishes for the scene-level
            # collision, applied to the last refusal that still followed the wipe.
            resume_existing = self._prepare_dataset_target(dataset_dir, overwrite)

            if resume_existing:
                logger.info("Resuming existing dataset for append: %s", dataset_dir)
                resumed = _DatasetRecorder.resume(
                    repo_id=repo_id,
                    root=root,
                    task=task,
                    vcodec=vcodec,
                    joint_names=joint_names,
                    extra_state_specs=base_state_specs,
                )
                self._verify_resume_schema(resumed, state_names_full, camera_keys, camera_dims, fps=fps)
                recorder = resumed
            else:
                recorder = _DatasetRecorder.create(
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
            resumed_line = self._arm_dataset_recorder(world._backend_state, recorder, resumed=resume_existing)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"Recording Newton scene to LeRobotDataset: {repo_id}\n"
                            f"{resumed_line}"
                            f"{recorded_cameras_line(joint_names, recorded_cameras, scene_cameras, cameras, fps)}"
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
    ) -> tuple[list[str], list[str], list[str], dict[str, tuple[int, int]], str, list[tuple[str, str, int, int]]]:
        """Build the dataset schema from the live Newton scene.

        Returns:
            A 6-tuple of:
              * ``joint_names``: ordered scalar state joint ids (namespaced
                ``robot__joint`` when more than one robot exists).
              * ``action_names``: ordered action column ids, taken from
                :meth:`robot_action_keys` - the same authority ``send_action``
                and the recording hook's ``required_action_keys`` resolve, so a
                declared column is always a key ``add_frame`` can receive.
              * ``camera_keys``: sanitized camera feature names (``/`` -> ``__``).
              * ``camera_dims``: map of camera feature name -> ``(height, width)``.
              * ``robot_type``: the dataset ``robot_type`` string.
              * ``recording_cameras``: per-camera ``(source_name, safe_name,
                width, height)`` tuples the on_frame hook renders each step.
        """
        world = self._world
        assert world is not None  # guarded by start_recording
        joint_names: list[str] = []
        action_names: list[str] = []
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
            # Action columns come from ``robot_action_keys``, not from the
            # scalar joint list computed above, even though the two agree for
            # every robot this backend can build today. They agree only because
            # both apply the free-base exclusion, and each applied it from its
            # own copy of the rule; declaring the action schema from the joint
            # list makes that agreement load-bearing. A column declared under a
            # name the hook never emits is not a mismatch the recorder can
            # report - ``add_frame`` reads the action dict by declared name, so
            # an unmatched column takes the ``0.0`` fill and the episode records
            # a command nobody issued under a success result (#1715).
            act_keys = self.robot_action_keys(rname)
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in scalar_jn)
                action_names.extend(f"{rname}__{ak}" for ak in act_keys)
            else:
                joint_names.extend(scalar_jn)
                action_names.extend(act_keys)
            robot_type = robot.data_config or rname

        camera_keys: list[str] = []
        camera_dims: dict[str, tuple[int, int]] = {}
        recording_cameras: list[tuple[str, str, int, int]] = []
        for cam_name, cam in world.cameras.items():
            safe_name = camera_schema_key(cam_name)
            width = int(getattr(cam, "width", self.default_width))
            height = int(getattr(cam, "height", self.default_height))
            camera_keys.append(safe_name)
            camera_dims[safe_name] = (height, width)
            recording_cameras.append((cam_name, safe_name, width, height))
        return joint_names, action_names, camera_keys, camera_dims, robot_type, recording_cameras

    def _make_recording_on_frame(self, robot_name: str, instruction: str) -> Any:
        """The recording half of the per-step ``on_frame`` hook for Newton.

        Returns an ``on_frame(step, observation, action)`` closure that, while a
        recording session is active, augments the joint-state observation with a
        rendered frame for each declared camera and forwards the frame to the
        active :class:`DatasetRecorder`. In multi-robot scenes scalar
        observation/action keys are namespaced (``robot__joint``) to match the
        schema declared in :meth:`start_recording`; camera ndarrays keep their
        sanitized names.

        Returns ``None`` when there is no world or the robot is unknown. No
        rollout claim is made here: :meth:`_make_run_policy_hook` layers that
        on top, and the evaluation facades (``eval_policy``,
        ``evaluate_benchmark``) install this hook alone when a recording is
        open and the caller passed no ``on_frame``.
        """
        from strands_robots.simulation.policy_runner import _extract_frame_ndarray

        world = self._world
        if world is None or not registered(world.robots, robot_name):
            return None

        multi_robot = len(world.robots) > 1

        # Action columns this rollout is responsible for: the driven robot's own
        # actuators. A declared column the policy never produced cannot be written
        # as a placeholder without persisting a command nobody issued, so
        # ``add_frame`` refuses it.
        #
        # Resolved on the first recorded frame and cached, rather than up front:
        # ``robot_action_keys`` is explicitly best-effort for the runner's
        # fail-fast probe (a backend quirk or a mid-rollout teardown may make it
        # raise, and that must not mask the primary "robot has not moved" signal),
        # so the hook must not call it for a rollout that is not recording. Where a
        # recording IS attached the keys are load-bearing - without them the frame
        # cannot be checked - so a raise there correctly fails the recording.
        action_key_cache: dict[bool, list[str]] = {}

        def _required_action_keys(prefixed: bool) -> list[str]:
            """Action columns this frame owes the recorder, resolved once."""
            cached = action_key_cache.get(prefixed)
            if cached is None:
                keys = self.robot_action_keys(robot_name)
                cached = [f"{robot_name}__{key}" for key in keys] if prefixed else list(keys)
                action_key_cache[prefixed] = cached
            return cached

        def _record(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
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

                # The schema declares a state column for every robot in the
                # scene, and this frame carries only the driven robot's. An
                # undriven robot's columns are a readable measurement, so they
                # are filled from the engine at this step rather than left to
                # add_frame's 0.0 fill, which records them as a zero pose the
                # robot is not in. Driven keys win any collision.
                driven = {(k if isinstance(v, np.ndarray) else f"{robot_name}__{k}"): v for k, v in obs.items()}
                obs = undriven_robot_state(self, (robot_name,), world.robots)
                obs.update(driven)
                act = {f"{robot_name}__{k}": v for k, v in action.items()}
                rec.add_frame(
                    observation=obs,
                    action=act,
                    task=instruction,
                    required_action_keys=_required_action_keys(True),
                )
            else:
                rec.add_frame(
                    observation=obs,
                    action=action,
                    task=instruction,
                    required_action_keys=_required_action_keys(False),
                )

        return _record

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Build the per-step ``on_frame`` hook for a rollout: claim + recording.

        Marks the robot as driven (``policy_running`` / ``policy_instruction`` /
        ``policy_steps``, released by :meth:`_release_run_policy_hook`) and
        forwards every frame to :meth:`_make_recording_on_frame`. ``None``
        when there is no world or the robot is unknown, so the base
        run-policy loop runs without recording.
        """
        world = self._world
        if world is None or not registered(world.robots, robot_name):
            return None
        robot = world.robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0
        record_frame = self._make_recording_on_frame(robot_name, instruction)

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            robot.policy_steps = step + 1
            if record_frame is not None:
                record_frame(step, observation, action)

        return _hook

    def _release_run_policy_hook(self, robot_name: str) -> None:
        """Lower the ``policy_running`` flag :meth:`_make_run_policy_hook` raised.

        Nothing lowered it, so a recorded rollout left the robot marked as
        driven for the rest of the session. On this backend the flag is what
        :meth:`~strands_robots.simulation.models.SimRobot.request_policy_stop`
        reports as ``was_running``, and that answer is the whole verdict every
        stop path here reports - :meth:`_request_policy_stop` hands it to
        :meth:`~strands_robots.simulation.base.SimEngine.stop_policy`, and the
        Device Connect ``stop`` RPC reads that - so a flag left raised made an
        idle simulation report a halted rollout that had finished on its own.

        Args:
            robot_name: The robot whose rollout has ended. A robot removed
                mid-rollout, or a world torn down under it, has nothing to
                release.
        """
        world = self._world
        if world is not None and registered(world.robots, robot_name):
            world.robots[robot_name].policy_running = False
