"""Isaac recording mixin - LeRobotDataset schema declaration + per-step capture.

The engine-independent recording lifecycle (``stop_recording`` /
``save_episode`` / ``get_recording_status`` / ``stream_dataset`` and the
``_is_recording`` / ``_active_recorder`` / ``_active_dataset_root`` overrides)
lives in :class:`~strands_robots.simulation.recording.DatasetRecordingMixin`,
which is backend-agnostic. This subclass adds the Isaac-specific parts:

* :meth:`start_recording` declares the dataset schema from the live Isaac
  scene - joint names from every robot (namespaced for multi-robot scenes) and
  the RTX cameras registered via ``add_camera``, each at the resolution its
  render product actually produces (probed from one ``get_observation`` call,
  because RTX cameras render at a DLSS-safe native size that can differ from
  the requested output size).
* :meth:`_make_run_policy_hook` returns the ``on_frame`` closure the shared
  :class:`~strands_robots.simulation.base.SimEngine` run-policy loop calls
  every control step. Unlike the MuJoCo/Newton hooks it does not render:
  Isaac's ``get_observation`` already carries a fresh RGB frame per camera
  (refreshing every RTX render product first when more than one camera exists,
  so multi-cam recordings never capture a stale secondary product), and
  ``IsaacSimulation.get_observation`` forces images on while a recording is
  active even when the driving policy sets ``requires_images = False``.
* :meth:`_release_run_policy_hook` lowers the ``policy_running`` flag the hook
  builder raised, when the rollout that hook served ends. The claim and its
  release are one contract, and only the shared facade knows where the rollout
  ends, so it calls this in a ``finally`` around every rollout it drives.

**State seam**: Isaac's ``self._world`` is the Isaac Sim ``World`` handle, not
the :class:`~strands_robots.simulation.models.SimWorld` the shared mixin's
default accessor expects, so :meth:`_recording_state` overrides the seam to
return the engine-owned ``self._recording_state_dict`` instead. Everything
else in the shared lifecycle runs unchanged.

**Pacing**: the recorded ``fps`` is dataset metadata; the actual frame cadence
is ``run_policy(control_frequency=...)`` (one recorded frame per control
step). The RTX renderer produces new frames at ``IsaacConfig.rendering_dt``
(default 1/30 s), so a control frequency above ``1 / rendering_dt`` records
duplicate frames from the same render product. For distinct per-step images
keep ``control_frequency <= 1 / rendering_dt`` and set ``fps`` to match the
control frequency.

**Threading**: schema declaration probes ``get_observation`` once. When the
main-thread pump (:meth:`IsaacSimulation.run_pump_forever`) owns the renderer
and ``start_recording`` is called from a worker thread (a Gradio-style host),
that probe is routed through :meth:`IsaacSimulation.run_on_main` so the RTX
render-product refresh never runs off the owning thread. Per-step capture
rides the ``run_policy`` thread's own ``get_observation`` calls, which hosts
already route via ``run_on_main`` when driving rollouts from worker threads.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

from strands_robots.simulation.models import registered, registry_entry
from strands_robots.simulation.recording import (
    DatasetRecordingMixin,
    camera_schema_key_collision_error,
    dataset_recording_option_error,
    dataset_recording_posture_error,
    undriven_robot_state,
)
from strands_robots.utils import camera_schema_key, name_list_error

if TYPE_CHECKING:
    import threading

    from strands_robots.simulation.isaac.config import IsaacConfig

logger = logging.getLogger(__name__)


class IsaacRecordingMixin(DatasetRecordingMixin):
    """Isaac dataset recording mixed into :class:`IsaacSimulation`.

    Inherits the engine-independent lifecycle from
    :class:`DatasetRecordingMixin` and supplies the Isaac-specific state seam
    (:meth:`_recording_state`), schema declaration (:meth:`start_recording`)
    and per-step capture hook (:meth:`_make_run_policy_hook`).
    """

    if TYPE_CHECKING:
        _world_created: bool
        _robots: dict[str, Any]
        _cameras: dict[str, Any]
        _config: IsaacConfig
        _lock: threading.RLock
        _sim_time: float
        _pump_running: bool
        _recording_state_dict: dict[str, Any]

        def robot_action_keys(self, robot_name: str) -> list[str]:
            """Actuator-ordered action keys (concrete on SimEngine)."""

        def get_observation(self, robot_name: str | None = None, *, skip_images: bool = False) -> dict[str, Any]:
            """Type-only stub for the engine-provided observation method."""

        def run_on_main(self, fn: Any, timeout: float | None = None) -> Any:
            """Type-only stub for the engine-provided main-thread executor."""

        def _on_main_thread(self) -> bool:
            """Type-only stub for the engine-provided thread check."""

    def _recording_state(self) -> dict[str, Any] | None:
        """Engine-owned recording-state dict (the Isaac state seam).

        Overrides :meth:`DatasetRecordingMixin._recording_state`: Isaac's
        ``self._world`` is the Isaac Sim ``World`` handle (no
        ``_backend_state``), so the recording flag / trajectory mirror /
        recorder handle live in ``self._recording_state_dict`` instead
        (initialised in ``IsaacSimulation.__init__`` and reset by
        ``destroy``). Returns ``None`` before ``create_world`` so the shared
        lifecycle reports the documented no-world responses.
        """
        if not getattr(self, "_world_created", False):
            return None
        return self._recording_state_dict

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
        """Start recording the Isaac scene to LeRobotDataset format.

        Declares the dataset schema from the live scene - joint names from
        every robot (namespaced ``robot__joint`` when more than one robot is
        present, matching the MuJoCo/Newton backends), action columns from
        :meth:`robot_action_keys`, and the RTX cameras registered via
        ``add_camera``. In a multi-robot scene every robot's state columns are
        declared, and a single-policy rollout fills the ones it does not
        drive from the engine at each step, so a declared
        ``observation.state`` column is a measurement rather than a zero
        the robot is not at
        (:func:`~strands_robots.simulation.recording.undriven_robot_state`).
        The matching *action* columns are a separate question - no command
        was issued to a robot this rollout does not drive - and are
        unchanged. Camera resolutions are probed from one
        ``get_observation`` call because RTX cameras render at a DLSS-safe
        native size that can exceed the requested output size; declaring the
        probed shape keeps ``add_frame`` from rejecting every image. Per-step
        frames are then captured by the ``on_frame`` hook
        (:meth:`_make_run_policy_hook`) during ``run_policy``.

        In ``render_mode="headless"`` no RTX frames exist, so the dataset
        records joint state and action only (a valid proprio-only LeRobot
        dataset); a warning is logged when cameras are registered but cannot
        be recorded. Use ``render_mode="rtx_realtime"`` (or pathtracing) for
        camera columns.

        Pacing: ``fps`` is dataset metadata - the actual cadence is one frame
        per ``run_policy`` control step, and the renderer produces new frames
        at ``IsaacConfig.rendering_dt`` (default 1/30 s). Keep
        ``control_frequency <= 1 / rendering_dt`` and ``fps`` equal to the
        control frequency for temporally-faithful video.

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
            fps: Recording frame rate (metadata; see pacing note above).
                Must be a positive whole number; a rate no dataset can be
                written at is rejected up front. When an existing dataset is
                RESUMED (``overwrite=False``) it must equal that dataset's
                on-disk rate, which a resume cannot change.
            root: Explicit on-disk dataset directory, used verbatim - it replaces
                the ``repo_id`` resolution above rather than being joined to it.
                See :func:`~strands_robots.dataset_recorder.resolve_dataset_dir`
                for the full precedence.
            push_to_hub: Publish to the Hub at ``stop_recording``. Must be a
                boolean - a publication posture is not read by truthiness
                (:func:`~strands_robots.simulation.recording.dataset_recording_posture_error`).
            vcodec: Video codec for the per-camera MP4 streams. Defaults to
                "h264" (H.264), universally decodable including by OpenCV's
                VideoCapture. Use "libsvtav1" (AV1) for smaller files;
                LeRobot read-back handles AV1 but OpenCV wheels commonly
                cannot decode it and silently yield 0 frames.
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
                (default) every registered RTX camera is recorded. Pass a
                subset to scope the dataset to exactly those views - matching
                the MuJoCo/Newton backends so ``run_policy(dataset_cameras=...)``
                behaves identically across engines. Names may be given in
                either the raw camera name or the schema-safe form (``/``
                collapsed to ``__``); an unknown name fails loudly, listing
                the available cameras, and is refused before any dataset is
                created, resumed or wiped, so a typo costs nothing even under
                ``overwrite=True``. Two scene cameras whose
                names collapse onto one dataset column (``arm0/wrist`` and
                ``arm0__wrist``) are refused before any dataset is created,
                because the column would be named after whichever of them lost
                it.

        Returns:
            Standard status dict. ``status="error"`` when no world exists, no
            robots exist, the ``lerobot`` extra is missing, or recorder init
            fails.
        """
        state = self._recording_state()
        if state is None:
            return {"status": "error", "content": [{"text": "No world created. Call create_world() first."}]}
        if not self._robots:
            return {
                "status": "error",
                "content": [{"text": "No robots in the world. Call add_robot() before start_recording()."}],
            }

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
                            "For plain MP4 video, use start_cameras_recording instead."
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
        if error := camera_schema_key_collision_error("start_recording", list(self._cameras)):
            return error

        # Probe one observation to learn each camera's real produced
        # resolution BEFORE taking the lock: when the main-thread pump owns
        # the renderer and this runs on a worker thread, the probe must run
        # on the pump thread (run_on_main), and holding self._lock across
        # that handoff would deadlock against the probe re-acquiring it.
        probe_obs = self._probe_recording_observation()

        with self._lock:
            state["recording"] = True
            state["trajectory"] = []
            state["push_to_hub"] = push_to_hub

            # Resolve the on-disk dataset dir with the same resolver
            # DatasetRecorder.create() uses (honours $HF_LEROBOT_HOME) and
            # stash it so verify_dataset_episodes can find the parquet after
            # stop_recording drops the recorder.
            from strands_robots.dataset_recorder import resolve_dataset_dir

            dataset_dir = resolve_dataset_dir(repo_id, root)
            state["last_dataset_root"] = str(dataset_dir)

            try:
                (
                    joint_names,
                    action_names,
                    camera_keys,
                    camera_dims,
                    robot_type,
                    recording_cameras,
                    base_state_specs,
                ) = self._collect_recording_schema(probe_obs)

                # Optional camera scoping (parity with MuJoCo/Newton). Names
                # may be raw (``arm0/wrist``) or schema-safe (``arm0__wrist``);
                # an unknown name fails loudly listing what exists.
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
                        state["recording"] = False
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

                state["recording_cameras"] = recording_cameras

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
                    resumed = _DatasetRecorder.resume(repo_id=repo_id, root=root, task=task, vcodec=vcodec)
                    # The expanded names, not the bare joint list: a resumed
                    # dataset's on-disk observation.state includes the base columns,
                    # so validating against joint_names alone would report a
                    # mismatch on every floating-base append. Same construction as
                    # the MuJoCo backend's ``state_names_full``.
                    state_names_full = list(joint_names) + [
                        f"{src}.{comp}" for src, comps in base_state_specs for comp in comps
                    ]
                    self._verify_resume_schema(
                        resumed, state_names_full, camera_keys, camera_dims, action_names, fps=fps
                    )
                    state["dataset_recorder"] = resumed
                else:
                    state["dataset_recorder"] = _DatasetRecorder.create(
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
                        video_width=int(self._config.camera_width),
                        video_height=int(self._config.camera_height),
                    )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"Recording Isaac scene to LeRobotDataset: {repo_id}\n"
                                f"{len(joint_names)} joints, {len(camera_keys)} cameras @ {fps}fps\n"
                                f"Codec: {vcodec} | Task: {task or '(set per policy)'}\n"
                                f"Run policies to capture frames, then stop_recording to save the episode"
                            )
                        }
                    ],
                }
            except Exception as e:
                state["recording"] = False
                logger.error("Dataset recorder init failed: %s", e)
                return {"status": "error", "content": [{"text": f"Dataset init failed: {e}"}]}

    def _probe_recording_observation(self) -> dict[str, Any]:
        """One ``get_observation`` probe used to size the camera schema.

        RTX cameras render at a DLSS-safe native resolution that can exceed
        the requested output size (see ``add_camera``), and ``get_observation``
        returns frames at that native size - so the schema must declare the
        shape the observation stream actually produces, not the requested one.
        Routed through :meth:`run_on_main` when the main-thread pump owns the
        renderer and the caller is a worker thread (Gradio-style hosts), since
        the multi-camera render-product refresh may only run on the owning
        thread. In headless render mode the probe returns no images and the
        schema falls back to proprio-only.
        """
        first_robot = next(iter(self._robots))
        if getattr(self, "_pump_running", False) and not self._on_main_thread():
            return self.run_on_main(lambda: self.get_observation(robot_name=first_robot))
        return self.get_observation(robot_name=first_robot)

    def _collect_recording_schema(
        self, probe_obs: dict[str, Any]
    ) -> tuple[
        list[str],
        list[str],
        list[str],
        dict[str, tuple[int, int]],
        str,
        list[tuple[str, str, int, int]],
        list[tuple[str, list[str]]],
    ]:
        """Build the dataset schema from the live Isaac scene.

        Args:
            probe_obs: One observation from :meth:`_probe_recording_observation`,
                used to size each camera at the resolution its render product
                actually emits.

        Returns:
            A 6-tuple of:
              * ``joint_names``: ordered state joint ids (namespaced
                ``robot__joint`` when more than one robot exists).
              * ``action_names``: actuator-ordered action column names
                (namespaced like the joints).
              * ``camera_keys``: sanitized camera feature names (``/`` -> ``__``).
              * ``camera_dims``: map of camera feature name -> ``(height, width)``.
              * ``robot_type``: the dataset ``robot_type`` string.
              * ``recording_cameras``: per-camera ``(source_name, safe_name,
                width, height)`` tuples the on_frame hook maps observation
                keys through each step.
              * ``base_state_specs``: ``(source_key, [components])`` pairs for a
                floating-base robot's ``base_pos`` / ``base_quat`` /
                ``base_lin_vel`` / ``base_ang_vel``, appended to the
                ``observation.state`` schema so a locomotion policy trained on
                the dataset is not base-blind. Empty for a fixed-base arm, which
                leaves its schema unchanged.
        """
        joint_names: list[str] = []
        action_names: list[str] = []
        robot_type = "unknown"
        multi_robot = len(self._robots) > 1
        base_state_specs: list[tuple[str, list[str]]] = []
        for rname, robot in self._robots.items():
            if multi_robot:
                joint_names.extend(f"{rname}__{jn}" for jn in robot.joint_names)
                action_names.extend(f"{rname}__{ak}" for ak in self.robot_action_keys(rname))
            else:
                joint_names.extend(robot.joint_names)
                action_names.extend(self.robot_action_keys(rname))
            robot_type = getattr(robot, "data_config", None) or rname
            # A floating-base robot exposes its full base kinematics through
            # get_observation - base_pos (world x,y,z incl. height), base_quat
            # (w,x,y,z), base_lin_vel (m/s) and base_ang_vel (rad/s, BODY frame) -
            # but the observation.state schema above is derived from scalar joint
            # names, so those signals would be DROPPED and a locomotion /
            # velocity-tracking / whole-body-control policy trained on the dataset
            # would be base-blind.
            #
            # Both sibling backends already pass these (mujoco/recording.py,
            # newton/recording.py, with this same reasoning); Isaac was the only one
            # that did not, so adding base_* to get_observation without this left an
            # Isaac humanoid dataset missing 13 columns a MuJoCo one has.
            #
            # Detected from ``fixed_base`` - the field the MJCF free-joint read
            # records - rather than from a joint id, because this backend has no
            # compiled model to ask. A fixed-base arm adds no columns, so its schema
            # is unchanged.
            if not getattr(robot, "fixed_base", True):
                prefix = f"{rname}__" if multi_robot else ""
                base_state_specs.append((f"{prefix}base_pos", ["x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_quat", ["w", "x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_lin_vel", ["x", "y", "z"]))
                base_state_specs.append((f"{prefix}base_ang_vel", ["x", "y", "z"]))

        camera_keys: list[str] = []
        camera_dims: dict[str, tuple[int, int]] = {}
        recording_cameras: list[tuple[str, str, int, int]] = []
        if self._config.render_mode == "headless" and self._cameras:
            # get_observation never emits frames in headless render mode, so a
            # camera column would record nothing. Warn loudly and record a
            # proprio-only dataset rather than silently declaring dead columns.
            logger.warning(
                "start_recording: %d camera(s) %s are registered but render_mode='headless' "
                "produces no RTX frames - recording a proprio-only dataset. Construct the "
                "sim with render_mode='rtx_realtime' to record camera columns.",
                len(self._cameras),
                sorted(self._cameras),
            )
            return joint_names, action_names, camera_keys, camera_dims, robot_type, recording_cameras, base_state_specs

        for cam_name, cam in self._cameras.items():
            safe_name = camera_schema_key(cam_name)
            frame = probe_obs.get(cam_name)
            if isinstance(frame, np.ndarray) and frame.ndim == 3:
                height, width = int(frame.shape[0]), int(frame.shape[1])
            else:
                # Probe frame unavailable (render product not warmed yet).
                # Fall back to the camera's native render size, which is what
                # get_observation emits once the product accumulates a frame.
                width, height = int(cam.width), int(cam.height)
                logger.warning(
                    "start_recording: probe observation carried no frame for camera %r; "
                    "declaring its native render size %dx%d. If recorded frames arrive at "
                    "a different resolution, add_frame will reject them - step the sim a "
                    "few times before start_recording to warm the render product.",
                    cam_name,
                    width,
                    height,
                )
            camera_keys.append(safe_name)
            camera_dims[safe_name] = (height, width)
            recording_cameras.append((cam_name, safe_name, width, height))
        return joint_names, action_names, camera_keys, camera_dims, robot_type, recording_cameras, base_state_specs

    def _make_run_policy_hook(self, robot_name: str, instruction: str) -> Any:
        """Build the per-step ``on_frame`` recording hook for Isaac.

        Returns an ``on_frame(step, observation, action)`` closure that, while
        a recording session is active, appends a step to the trajectory mirror
        and forwards the frame to the active
        :class:`~strands_robots.dataset_recorder.DatasetRecorder`. Camera
        frames are consumed from the observation itself (Isaac's
        ``get_observation`` carries a fresh RGB frame per camera and refreshes
        every RTX render product when more than one camera exists, so
        multi-cam recordings never duplicate a stale secondary product) and
        renamed to their schema-safe names; cameras outside the
        ``start_recording(cameras=...)`` scope are dropped. In multi-robot
        scenes scalar observation/action keys are namespaced
        (``robot__joint``) to match the declared schema.

        Returns ``None`` when there is no world or the robot is unknown, so
        the base run-policy loop runs without recording.
        """
        import time

        from strands_robots.simulation.models import TrajectoryStep

        state = self._recording_state()
        if state is None or not registered(self._robots, robot_name):
            return None

        robot = self._robots[robot_name]
        robot.policy_running = True
        robot.policy_instruction = instruction
        robot.policy_steps = 0
        multi_robot = len(self._robots) > 1

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

        def _hook(step: int, observation: dict[str, Any], action: dict[str, Any]) -> None:
            robot.policy_steps = step + 1
            if not state.get("recording", False):
                return
            rec = state.get("dataset_recorder")
            if rec is None:
                return

            # Split the observation: camera ndarrays are renamed raw -> safe
            # and scoped to the declared recording cameras; scalars feed
            # observation.state. Resolved per step (not captured at hook build
            # time) so a start_recording issued after run_policy launched
            # still scopes correctly.
            raw_to_safe = {src: safe for src, safe, _w, _h in state.get("recording_cameras", [])}
            scalars: dict[str, Any] = {}
            images: dict[str, Any] = {}
            for k, v in observation.items():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    safe = raw_to_safe.get(k)
                    if safe is not None:
                        images[safe] = v
                else:
                    scalars[k] = v

            state["trajectory"].append(
                TrajectoryStep(
                    timestamp=time.time(),
                    sim_time=self._sim_time,
                    robot_name=robot_name,
                    observation=scalars,
                    action=action,
                    instruction=instruction,
                )
            )

            if multi_robot:
                # The schema declares a state column for every robot in the
                # scene, and this frame carries only the driven robot's. An
                # undriven robot's columns are a readable measurement, so they
                # are filled from the engine at this step rather than left to
                # add_frame's 0.0 fill, which records them as a zero pose the
                # robot is not in. Driven keys win any collision.
                obs = undriven_robot_state(self, (robot_name,), self._robots)
                obs.update({f"{robot_name}__{k}": v for k, v in scalars.items()})
                obs.update(images)
                act = {f"{robot_name}__{k}": v for k, v in action.items()}
                rec.add_frame(
                    observation=obs,
                    action=act,
                    task=instruction,
                    required_action_keys=_required_action_keys(True),
                )
            else:
                obs = dict(scalars)
                obs.update(images)
                rec.add_frame(
                    observation=obs,
                    action=action,
                    task=instruction,
                    required_action_keys=_required_action_keys(False),
                )

        return _hook

    def _release_run_policy_hook(self, robot_name: str) -> None:
        """Lower the ``policy_running`` flag :meth:`_make_run_policy_hook` raised.

        The hook builder marks the robot as driven so a motion primitive and
        the rollout cannot race on the same articulation's PD targets, and
        :meth:`~strands_robots.simulation.isaac.motion_primitives.IsaacMotionPrimitivesMixin._primitive_resolve_robot`
        refuses on exactly that flag. Nothing lowered it, so a recorded rollout
        left the robot marked driven for the rest of the session: every
        primitive answered ``Cannot 'set_gripper' on 'so100' while its policy
        is running ... wait for the rollout to finish``, and
        :meth:`~strands_robots.simulation.isaac.simulation.IsaacSimulation.run_multi_policy`
        answered ``policy already running ... Stop it first`` - two remedies
        for a rollout that had already ended, and at the time neither was even
        reachable, because Isaac exposed no ``stop_policy``. It inherits one now
        (:meth:`~strands_robots.simulation.base.SimEngine.stop_policy`), which
        on this backend states why it cannot help - the per-robot record here
        carries a bare ``policy_running`` flag and not the durable claim
        ``SimRobot.request_policy_stop`` writes - rather than raising
        ``AttributeError``. The release below is what makes the remedy
        unnecessary in the first place.

        ``run_multi_policy`` lowers the flag in its own ``finally`` for the
        loop it owns; this is the same release for the rollout the shared
        facade owns, so the flag means "a loop is driving this robot" on both
        paths. ``policy_instruction`` / ``policy_steps`` stay as the last
        rollout's record, as they do on the MuJoCo backend.

        Args:
            robot_name: The robot whose rollout has ended. A robot removed
                mid-rollout has nothing to release.
        """
        robot = registry_entry(self._robots, robot_name)
        if robot is not None:
            robot.policy_running = False
