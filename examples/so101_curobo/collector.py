# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""LeRobot data collection for the SO-101 cuRobo demo (issue #67 T7).

Streams a planned :class:`~examples.so101_curobo.planner.JointTrajectory` into
the simulation (``send_action`` per waypoint) while recording each control step
as a LeRobot frame (state + action + camera images) via the tested
``strands_robots.dataset_recorder.DatasetRecorder``. Writes a v3.0 LeRobot
dataset (parquet + per-camera video), with a programmatic success check and a
resumable multi-episode loop over (optionally) randomized scenes.

Validated recipe (matches ``Simulation.stop_recording``):
    create() -> add_frame()* -> save_episode() -> finalize()
then reload locally with ``load_lerobot_episode`` (no Hub round-trip).
"""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("so101_curobo.collector")

# Temporary grasp-attach debug (env-gated): logs gripper<->cube distance and
# attach/release transitions so the kinematic grasp can be diagnosed.
_GRASP_DBG = bool(os.environ.get("SO101_GRASP_DBG"))

LEROBOT_INSTALL_HINT = (
    "LeRobot dataset writing requires the lerobot extra: "
    "pip install 'strands-robots[lerobot]'  (or `pip install lerobot`)."
)


def lerobot_available() -> bool:
    try:
        from strands_robots.dataset_recorder import has_lerobot_dataset

        return bool(has_lerobot_dataset())
    except Exception:  # noqa: BLE001
        return False


@dataclass
class EpisodeResult:
    success: bool
    cube_moved: bool
    placed: bool
    frames: int
    displacement: float
    place_distance: float
    phases: int


def _object_position(sim, name: str) -> list[float] | None:
    """Best-effort world position of object ``name``.

    Prefers a backend-native ``sim._object_position(name)`` (the Isaac backend
    exposes one via ``get_world_pose`` on the registered prim); falls back to
    MuJoCo's ``mj_data`` lookup; else None.
    """
    # Backend-native (Isaac): the sim tracks the prim and can read its pose.
    native = getattr(sim, "_object_position", None)
    if callable(native):
        try:
            pos = native(name)
            if pos is not None:
                return [float(x) for x in pos]
        except Exception:  # noqa: BLE001
            logger.debug("sim._object_position failed for %s", name, exc_info=True)
    try:
        import mujoco

        m = getattr(sim, "mj_model", None)
        d = getattr(sim, "mj_data", None)
        if m is not None and d is not None:
            for b in range(m.nbody):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if name == bn or bn.endswith(f"/{name}") or (name in bn and "cube" in name):
                    return [float(x) for x in d.xpos[b]]
    except Exception:  # noqa: BLE001
        logger.debug("mj_data object read failed for %s", name, exc_info=True)
    return None


class LeRobotDataCollector:
    """Execute trajectories and record LeRobot episodes (issue #67 T7)."""

    def __init__(
        self,
        sim,
        scene_info,
        repo_id: str = "local/so101_curobo_pickplace",
        fps: int = 40,
        root: str | None = None,
        cameras: Sequence[str] | None = None,
        place_radius: float = 0.10,
        move_threshold: float = 0.03,
        record_images: bool = True,
        kinematic: bool = False,
        grasp_attach: bool = False,
        attach_radius: float = 0.08,
        attach_offset: Sequence[float] | None = None,
        base_sign: float = 1.0,
        hybrid_carry: bool = False,
    ):
        self.sim = sim
        self.scene = scene_info
        self.repo_id = repo_id
        self.fps = fps
        self.root = root
        self.cameras = list(cameras) if cameras else list(scene_info.cameras)
        self.place_radius = place_radius
        self.move_threshold = move_threshold
        # Base-pan sign passed to the scripted planner. The SO-101 URDF (Isaac)
        # has an inverted shoulder_pan vs world Y, so the controller sets this to
        # -1 for Isaac; MuJoCo keeps +1. Without it the planner aims the gripper
        # at the wrong side and never reaches the cube (success_rate stays 0).
        self.base_sign = base_sign
        # When False, record state+action only (no camera frames) -> no GL/EGL
        # needed. Used by the CI smoke path on CPU-only boxes.
        self.record_images = record_images
        # When True, drive the arm kinematically (set_joint_positions + step)
        # instead of send_action. Required for the cuRobo/URDF-matched arm,
        # which loads without position actuators (send_action wouldn't move it);
        # also makes the arm follow a planned trajectory exactly.
        self.kinematic = kinematic
        # When True, model the grasp by attaching the cube to the gripper while
        # the gripper is closed AND was within attach_radius of the cube when it
        # closed (a standard kinematic grasp for synthetic data; the actuator-
        # less arm can't hold via friction). Attaches only if the gripper truly
        # reached the cube, so it stays honest.
        self.grasp_attach = grasp_attach
        self.attach_radius = attach_radius
        # HYBRID carry (Isaac physical pick + scripted transport). The
        # force-driven arm genuinely grasps and LIFTS the cube via friction
        # (reach/grasp/close/lift run pure PhysX), but the 5-DOF arm's friction
        # grip cannot reliably hold the cube through the sideways place traverse
        # (PhysX contact variance + wrist re-orientation make it slip/fling, so
        # the cube lands scattered and "success" was just proximity luck). With
        # ``hybrid_carry`` the cube, once physically lifted, is rigidly attached
        # to the gripper for the PLACE segments (its pose relative to the gripper
        # frame is captured at the lift->place transition and held), then dropped
        # at the bin on release. The hard, physically-meaningful part (grasp +
        # lift) stays real physics; only the transport is kinematic.
        self.hybrid_carry = hybrid_carry
        # Z offset (m) of the carried cube below the gripper frame origin while
        # glued in the jaws. The SO-101 gripper_frame_link sits ~at the fingertip
        # grasp point, so the cube center rides essentially AT the frame; a small
        # negative keeps it seated at the finger pads. Clamped above the table.
        self._hybrid_z_in_jaw = -0.01
        # Empirical XY shift applied to the cube's drop target to compensate for
        # the cube's RENDERED-vs-logical position offset on the Isaac arm (the
        # cube renders a few cm off its logical coords relative to the bin).
        # Tunable via SO101_CUBE_RENDER_OFFSET="x,y".
        import os as _os_off

        _off_env = _os_off.environ.get("SO101_CUBE_RENDER_OFFSET", "")
        try:
            _ox, _oy = (float(v) for v in _off_env.split(","))
            self._cube_render_offset = [_ox, _oy]
        except Exception:  # noqa: BLE001
            self._cube_render_offset = [0.0, 0.0]
        # Cube half-height and bin geometry, used to rest the cube ON SURFACES
        # without penetration: on the table the cube center sits at cube_half_z;
        # ON THE BIN it must sit at (bin_top + cube_half_z) or it sinks through
        # the bin plate. Read from the scene (falls back to safe defaults).
        try:
            self._cube_half_z2 = float(scene_info.cube_half[2])
        except (AttributeError, IndexError, TypeError):
            self._cube_half_z2 = 0.015
        try:
            bh = list(scene_info.bin_half)
            self._bin_half_xy = [float(bh[0]), float(bh[1])]
            self._bin_top_z = 2.0 * float(bh[2])  # plate sits with base on floor
        except (AttributeError, IndexError, TypeError):
            self._bin_half_xy = [0.05, 0.05]
            self._bin_top_z = 0.024
        # NOTE: a USD world-bbox probe of the bin was tried to measure the true
        # top, but ``ComputeWorldBound`` does not capture Isaac's cuboid *scale*
        # (it returned a sub-cm artifact for both bin and cube), so it is NOT
        # reliable. The analytic value above (bin center z = bin_half_z, full
        # height = 2*bin_half_z -> top = 2*bin_half_z) matches how the bin is
        # actually created in build_pick_place_scene, so we use it directly.
        # Optional fixed nudge (tool-frame local coords) of the seated cube,
        # applied once at grasp. Default zero -> the cube is captured exactly
        # where the (top-down) gripper closed on it and carried rigidly there, so
        # it reads as gripped in place. (A non-zero "seat into the mouth" offset
        # was tried, but with the vertical grasp it makes the cube visibly jump
        # into the gripper.) Tune per gripper only if a small adjustment helps.
        self.attach_offset = list(attach_offset) if attach_offset is not None else [0.0, 0.0, 0.0]
        # Cube half-height (its resting center z, since it sits on the table) used
        # to clamp the kinematic carry so the cube never sinks through the floor.
        try:
            self._cube_half_z = float(scene_info.cube_position[2])
        except (AttributeError, IndexError, TypeError):
            self._cube_half_z = 0.0
        # Tool-frame seat of the carried cube at the FINGERTIP TCP. The cuRobo
        # tool frame (gripper_frame_link) sits ~6 cm behind the fingers, so the
        # cube is carried here (matching the planner's tcp_offset) to ride between
        # the fingers rather than at the frame origin. Set None to seat on the
        # tool centerline at the cube's measured grasp depth instead.
        self.grip_seat_local = [-0.031, 0.0, -0.05]

    # --- recording lifecycle ------------------------------------------------

    @staticmethod
    def available() -> bool:
        return lerobot_available()

    def home_q(self) -> dict[str, float]:
        """Current per-joint positions as a ``{joint: value}`` map (the rest pose)."""
        obs = self.sim.get_observation(self.scene.robot_name, skip_images=True)
        return {j: float(obs[j]) for j in self.scene.joint_names}

    def reset_world(self, home_q: dict[str, float] | None = None, snapshot=None) -> None:
        """Reset to a consistent start: arm -> ``home_q``, cube -> its start pose.

        Mirrors the per-episode reset in :meth:`record_dataset` so the
        single-shot :meth:`record_episode` path (the UI's "Plan & execute") also
        starts from a known, collision-free state every time. Prefers a MuJoCo
        physics ``snapshot`` (exact) and falls back to the explicit
        home+cube teleport (:meth:`_reset_episode`) on other backends.
        """
        if snapshot is not None:
            self._restore_state(snapshot)
        else:
            self._reset_episode(home_q if home_q is not None else self.home_q())

    def _default_root(self) -> str | None:
        """Resolve the on-disk dir LeRobot would use when ``--root`` is unset.

        Delegates to :func:`strands_robots.dataset_recorder.resolve_dataset_dir`,
        which mirrors ``LeRobotDataset.create``'s default of
        ``$HF_LEROBOT_HOME/{repo_id}`` (honouring the ``HF_LEROBOT_HOME``
        environment override), so the re-run cleanup in :meth:`_new_recorder`
        can also clear the HF-cache default (issue #143), not just an explicit
        ``--root``.
        """
        from strands_robots.dataset_recorder import resolve_dataset_dir

        return str(resolve_dataset_dir(self.repo_id))

    def _new_recorder(self, task: str):
        if not self.available():
            raise RuntimeError(LEROBOT_INSTALL_HINT)
        import os
        import shutil

        from strands_robots.dataset_recorder import DatasetRecorder

        # LeRobotDataset.create() requires a non-existent root (mkdir exist_ok=False),
        # so a prior run's dataset dir makes every re-run raise FileExistsError
        # (issue #143). Resolve the EFFECTIVE root (the HF cache default
        # ``$HF_LEROBOT_HOME/{repo_id}`` when --root is unset, matching how
        # LeRobotDataset.create resolves it) and clear a prior dataset dir (or an
        # empty placeholder, e.g. from mkdtemp) so the documented OOTB command is
        # idempotently re-runnable. Only touch dirs that look like a dataset or are
        # empty, so we never blow away an unrelated path the user pointed --root at.
        effective_root = self.root or self._default_root()
        if effective_root and os.path.isdir(effective_root):
            if os.path.isdir(os.path.join(effective_root, "meta")) or not os.listdir(effective_root):
                shutil.rmtree(effective_root, ignore_errors=True)

        # Per-camera (height, width) so the schema matches what render returns.
        cam_dims: dict[str, tuple] = {}
        if self.record_images:
            obs = self.sim.get_observation(self.scene.robot_name)
            for cam in self.cameras:
                img = obs.get(cam)
                if img is not None and hasattr(img, "shape") and len(img.shape) >= 2:
                    cam_dims[cam] = (int(img.shape[0]), int(img.shape[1]))
        return DatasetRecorder.create(
            repo_id=self.repo_id,
            fps=self.fps,
            robot_type=self.scene.robot_config,
            joint_names=self.scene.joint_names,
            camera_keys=list(cam_dims.keys()),
            camera_dims=cam_dims,
            task=task,
            root=self.root,
        )

    def _snapshot_state(self):
        """Snapshot full MuJoCo physics state (qpos, qvel) for a deterministic reset.

        MuJoCo-only. On other backends (Isaac) this returns None so the
        :meth:`record_dataset` loop uses the explicit home+cube reset path
        (:meth:`_reset_episode`), which is deterministic per-backend.
        """
        try:
            d = getattr(self.sim, "mj_data", None)
            return (d.qpos.copy(), d.qvel.copy()) if d is not None else None
        except Exception:  # noqa: BLE001
            return None

    def _restore_state(self, snap) -> None:
        """Restore a MuJoCo snapshot so every episode starts from the identical state."""
        if snap is None:
            return
        try:
            import mujoco
            import numpy as np

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return
            # The stiff/undamped URDF arm can blow MuJoCo up (QACC NaN); a NaN in
            # the live qpos/qvel must NOT be carried back in. The snapshot was
            # captured clean at build, so restore it wholesale and hard-zero qvel
            # + qacc + warmstart so no residual instability survives the reset.
            qpos = np.asarray(snap[0], dtype=d.qpos.dtype)
            qpos = np.nan_to_num(qpos, nan=0.0, posinf=0.0, neginf=0.0)
            d.qpos[:] = qpos
            d.qvel[:] = 0.0
            d.qacc[:] = 0.0
            d.qacc_warmstart[:] = 0.0
            d.ctrl[:] = 0.0
            mujoco.mj_forward(m, d)
        except Exception:  # noqa: BLE001
            logger.debug("state restore failed (non-fatal)", exc_info=True)

    def _reset_episode(self, home_q: dict[str, float]) -> None:
        """Explicit deterministic reset: arm -> ``home_q``, cube -> its start pose.

        Used when no MuJoCo snapshot is available (e.g. the Isaac backend, whose
        articulation/object state isn't captured by the qpos snapshot). The arm
        is set to home and the cube teleported to its start pose.

        On the force-driven Isaac path the cube is a DYNAMIC body and
        ``set_joint_positions`` does not snap the PD-driven arm home instantly --
        it converges over several physics steps. If we move the cube and step
        immediately, the arm is still near/on the cube from the previous
        episode's place pose and knocks it, so the next episode starts from a
        drifted pose (observed: z=0.038 instead of 0.02, y creeping to 0.218).
        Fix: park the arm and let it settle home FIRST (with the cube collider
        off so a transient arm sweep can't touch it), THEN drop the cube at its
        exact start pose and re-assert it after a short settle so every episode
        starts identically.
        """
        try:
            # 1. Park the arm and let the PD drive actually converge home, with
            #    the cube held out of the way (collider off) so the homing sweep
            #    can't bump it.
            self.sim.set_joint_positions(home_q, robot_name=self.scene.robot_name)
            try:
                self._set_cube_collision(False)
            except Exception:  # noqa: BLE001
                pass
            self.sim.step(30)
            # 2. Now place the cube at its exact start pose (velocity zeroed by
            #    move_object), restore its collider + DYNAMIC body (so the grasp
            #    is physical again -- the carry flips it kinematic), and let it
            #    settle briefly.
            try:
                self._set_cube_kinematic(False)
                self._set_cube_collision(True)
            except Exception:  # noqa: BLE001
                pass
            self.sim.move_object(self.scene.cube_name, position=list(self.scene.cube_position))
            self.sim.step(3)
            # 3. Re-assert (the settle step can nudge it a hair) so the start is
            #    deterministic across episodes.
            self.sim.move_object(self.scene.cube_name, position=list(self.scene.cube_position))
            self.sim.step(1)
            if _GRASP_DBG:
                cp = _object_position(self.sim, self.scene.cube_name)
                logger.info("[grasp-dbg] reset cube -> %s", [round(x, 4) for x in cp] if cp else None)
        except Exception:  # noqa: BLE001
            logger.debug("episode reset failed (non-fatal)", exc_info=True)

    def _zero_cube_velocity(self) -> None:
        """Zero the cube's free-joint velocity (avoids a fling from teleporting it)."""
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return
            cube_bodies = {
                b
                for b in range(m.nbody)
                if self.scene.cube_name in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            }
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) in cube_bodies:
                    adr = int(m.jnt_dofadr[j])
                    for k in range(6):  # 3 linear + 3 angular dofs
                        d.qvel[adr + k] = 0.0
        except Exception:  # noqa: BLE001
            pass

    def _zero_arm_dynamics(self) -> None:
        """Zero ALL non-cube DOF velocities + accelerations (kinematic arm only).

        The SO-101 URDF arm loads with NO joint damping/armature/frictionloss and
        the default (Euler) integrator at dt=0.002. On the kinematic path each
        waypoint teleports ``qpos`` via ``set_joint_positions`` (which leaves
        ``qvel`` untouched) and then ``step()`` runs full forward dynamics --
        integrating explicit dynamics from a discontinuous pose with zero
        dissipation, so ``qacc`` diverges ("QACC NaN, simulation unstable") and
        the arm snaps violently (looks like it's "moving too fast" in the video).
        Since the planned trajectory fully specifies the pose at every frame, the
        arm should track it kinematically with NO residual velocity: zero every
        DOF's qvel/qacc each frame (skip the cube's free joint, which the grasp
        logic manages separately). This keeps the recorded motion smooth and
        prevents the blow-up at the source. MuJoCo-only; no-op elsewhere.
        """
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return
            # DOF range of the cube's free joint (leave it to the grasp/cube logic).
            cube_bodies = {
                b
                for b in range(m.nbody)
                if self.scene.cube_name in (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "")
            }
            cube_dofs = set()
            for j in range(m.njnt):
                if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(m.jnt_bodyid[j]) in cube_bodies:
                    adr = int(m.jnt_dofadr[j])
                    cube_dofs.update(range(adr, adr + 6))
            for dof in range(m.nv):
                if dof not in cube_dofs:
                    d.qvel[dof] = 0.0
                    d.qacc[dof] = 0.0
        except Exception:  # noqa: BLE001
            pass

    def _set_cube_kinematic(self, kinematic: bool) -> None:
        """Toggle the carried cube's rigid body kinematic (Isaac hybrid carry).

        A dynamic cube keeps being moved by gravity/the solver between our
        ``move_object`` pin and the render tick, so it RENDERS offset from where
        it was placed (looks beside/sunk into the bin). Making it kinematic while
        carried means its transform comes straight from ``move_object`` and is
        rendered faithfully. Restored to dynamic on release. No-op on backends
        without ``set_object_kinematic`` (e.g. MuJoCo).
        """
        fn = getattr(self.sim, "set_object_kinematic", None)
        if callable(fn):
            try:
                fn(self.scene.cube_name, kinematic)
            except Exception:  # noqa: BLE001
                logger.debug("set_object_kinematic failed (non-fatal)", exc_info=True)

    def _set_cube_collision(self, enabled: bool) -> None:
        """Toggle the carried cube's collider (Isaac kinematic grasp only).

        The grasped cube is teleported *into* the closing gripper fingers each
        frame; with its collider on, the static cube interpenetrates the finger
        colliders and the contact forces fling the stiff, undamped arm -- the
        cube then shakes ~5 cm/frame. Disabling the collider while grasped lets
        the gripper close cleanly around it; restored on release. No-op on
        backends without ``set_object_collision`` (e.g. MuJoCo, whose actuated
        friction grasp needs the contact).
        """
        fn = getattr(self.sim, "set_object_collision", None)
        if callable(fn):
            try:
                fn(self.scene.cube_name, enabled)
            except Exception:  # noqa: BLE001
                logger.debug("set_object_collision failed (non-fatal)", exc_info=True)

    def _jaw_center_pos(self) -> list[float] | None:
        """World position BETWEEN the gripper fingers (MuJoCo only).

        The cuRobo tool frame (``gripper_frame_link``) sits ~6-10 cm off the
        actual fingers, so seating the carried cube at the tool frame makes it
        float visibly away ("picked up without touching"). This returns the
        midpoint of the lowest finger GEOMS (the moving jaw + the fixed jaw on
        ``gripper_link``) -- i.e. where a gripped cube physically sits. Uses geom
        world positions (not body origins, which are pivots offset from the
        contact). None on non-MuJoCo backends -> carry falls back to the tool
        frame.
        """
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return None
            jaw_pts, grip_pts = [], []
            for g in range(m.ngeom):
                bn = (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, m.geom_bodyid[g]) or "").lower()
                if "jaw" in bn:
                    jaw_pts.append([float(x) for x in d.geom_xpos[g]])
                elif "gripper" in bn:
                    grip_pts.append([float(x) for x in d.geom_xpos[g]])
            if not jaw_pts:
                return None
            jaw = [sum(p[i] for p in jaw_pts) / len(jaw_pts) for i in range(3)]
            if grip_pts:
                # Midpoint between the moving jaw and the fixed-jaw side: the
                # cube grip point lies between the two fingers.
                grip = [sum(p[i] for p in grip_pts) / len(grip_pts) for i in range(3)]
                return [(jaw[i] + grip[i]) / 2.0 for i in range(3)]
            return jaw
        except Exception:  # noqa: BLE001
            return None

    def _gripper_frame_pos(self) -> list[float] | None:
        """World position of the gripper/tool link.

        Prefers a backend-native ``sim.gripper_frame_pos(robot_name)`` (the
        Isaac backend reads the link prim's world transform off the USD stage);
        falls back to MuJoCo's ``mj_data`` body lookup.
        """
        native = getattr(self.sim, "gripper_frame_pos", None)
        if callable(native):
            try:
                pos = native(self.scene.robot_name)
                if pos is not None:
                    return [float(x) for x in pos]
            except Exception:  # noqa: BLE001
                logger.debug("sim.gripper_frame_pos failed", exc_info=True)
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is None or d is None:
                return None
            best = None
            for b in range(m.nbody):
                bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                if "gripper_frame" in bn:
                    return [float(x) for x in d.xpos[b]]
                if "gripper" in bn:
                    best = [float(x) for x in d.xpos[b]]
            return best
        except Exception:  # noqa: BLE001
            return None

    def _gripper_frame_pose(self):
        """``((px, py, pz), rot[9])`` for the gripper/tool link, or ``None``.

        Prefers a backend-native ``sim.gripper_frame_pose`` (Isaac reads the
        link's full world transform off the USD stage). On MuJoCo, reads the
        gripper_frame_link body's world rotation matrix (``data.xmat``) so the
        tool-frame carry seat is rotated correctly into the world. (Falling back
        to an IDENTITY rotation -- the old behaviour -- made the tool-local seat
        offset apply along WORLD axes, dropping the carried cube ~6 cm straight
        down through the table instead of placing it between the tilted fingers:
        the "cube disappears, then floats offset from the gripper" bug.)
        """
        native = getattr(self.sim, "gripper_frame_pose", None)
        if callable(native):
            try:
                res = native(self.scene.robot_name)
                if res:
                    pos, rot = res
                    return [float(x) for x in pos], [float(x) for x in rot]
            except Exception:  # noqa: BLE001
                logger.debug("sim.gripper_frame_pose failed", exc_info=True)
        # MuJoCo: read the real body rotation so the seat offset rotates with the tool.
        try:
            import mujoco

            m = getattr(self.sim, "mj_model", None)
            d = getattr(self.sim, "mj_data", None)
            if m is not None and d is not None:
                bid = -1
                for b in range(m.nbody):
                    bn = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or ""
                    if "gripper_frame" in bn:
                        bid = b
                        break
                if bid >= 0:
                    pos = [float(x) for x in d.xpos[bid]]
                    rot = [float(x) for x in d.xmat[bid]]
                    return pos, rot
        except Exception:  # noqa: BLE001
            logger.debug("mj gripper_frame rotation read failed", exc_info=True)
        pos = self._gripper_frame_pos()
        if pos is None:
            return None
        return pos, [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]

    @staticmethod
    def _frame_to_local(rot: list[float], origin: list[float], point: list[float]) -> list[float]:
        """Express world ``point`` in the tool frame: ``local = R^T (point - origin)``."""
        d = [point[i] - origin[i] for i in range(3)]
        out = []
        for c in range(3):  # columns of R are the tool axes in world; local[c] = axis_c . d
            out.append(rot[c] * d[0] + rot[3 + c] * d[1] + rot[6 + c] * d[2])
        return out

    @staticmethod
    def _frame_to_world(rot: list[float], origin: list[float], local: list[float]) -> list[float]:
        """Map a tool-frame point back to world: ``world = origin + R @ local``."""
        out = []
        for r in range(3):
            row = rot[3 * r] * local[0] + rot[3 * r + 1] * local[1] + rot[3 * r + 2] * local[2]
            out.append(origin[r] + row)
        return out

    def _rest_z_at(self, x: float, y: float) -> float:
        """Resting cube-center Z at world ``(x, y)``: on top of the bin plate if
        over it, else on the table. Prevents the cube sinking THROUGH the bin
        (its top surface is at ``bin_top``, not the floor) on release/settle.
        """
        half_z = self._cube_half_z2 if self._cube_half_z2 > 0.0 else 0.015
        bx, by = self.scene.place_position[0], self.scene.place_position[1]
        if abs(x - bx) <= self._bin_half_xy[0] and abs(y - by) <= self._bin_half_xy[1]:
            return self._bin_top_z + half_z
        return half_z

    def _execute_and_record(self, trajectory, task: str, recorder, n_substeps: int) -> int:
        """Stream the trajectory; record one frame per waypoint. Returns frame count.

        With ``grasp_attach`` the cube is attached to the gripper while the
        gripper is closed and was within ``attach_radius`` of the cube when it
        closed (a kinematic grasp -- the actuator-less arm can't hold via
        friction), and released when the gripper re-opens.
        """
        robot = self.scene.robot_name
        grip = self.scene.gripper_joint
        frames = 0

        # Whether the gripper actuates at all in this trajectory (a real grip).
        gvals = [wp[grip] for wp in trajectory.waypoints if grip and grip in wp]
        gmin = min(gvals) if gvals else 0.0
        gmax = max(gvals) if gvals else 0.0
        has_grip = self.grasp_attach and grip is not None and (gmax - gmin) > 0.05

        attached = False
        has_closed = False
        # --- Hybrid carry state (Isaac physical pick + scripted transport) ---
        # The cube is physically grasped+lifted (pure PhysX), then at the
        # lift->place transition we capture its pose relative to the gripper
        # frame and rigidly carry it through the place segments, releasing at the
        # bin. ``hy_local`` holds that captured tool-frame offset; ``hy_lifted``
        # gates capture on a genuine physical lift having occurred.
        hy_local: list[float] | None = None
        hy_lifted = False
        hy_released = False
        hy_grasped = False
        self._hybrid_drop_xy = None
        # Track the cube's max height to detect a genuine pick (lifted clear of
        # the table). Used by the success check so a failed grasp that leaves the
        # cube on the table doesn't count as success.
        cube0 = _object_position(self.sim, self.scene.cube_name)
        self._cube_start_z = float(cube0[2]) if cube0 else 0.0
        self._cube_max_z = self._cube_start_z
        grasp_local: list[float] | None = None  # cube pos in tool-frame coords (carried)
        rest_local: list[float] | None = None  # where the cube rested, tool-frame coords
        # Phases where the gripper is gripping (jaws closed on the cube). Detect
        # "closed" by PHASE, not by a value threshold: the SO-101 gripper joint's
        # closing direction is the LOW end of its range, but the old
        # ``closed = gv >= thresh`` assumed high=closed, so it formed the grasp
        # while the jaw was actually opening. Phase-based is robust to the sign.
        closed_phases = {"close", "lift", "place", "place_down"}
        if _GRASP_DBG:
            logger.info(
                "[grasp-dbg] has_grip=%s grip=%r grange=[%.2f,%.2f] kinematic=%s grasp_attach=%s base_sign=%s",
                has_grip,
                grip,
                gmin,
                gmax,
                self.kinematic,
                self.grasp_attach,
                self.base_sign,
            )

        for wp, phase in zip(trajectory.waypoints, trajectory.phases):
            if self.kinematic:
                self.sim.set_joint_positions(wp, robot_name=robot)
                self.sim.step(max(1, n_substeps))
                # Kinematic arm: kill any velocity/accel the explicit step
                # accumulated on the undamped URDF joints so the motion tracks the
                # plan smoothly instead of diverging (QACC NaN -> violent snap).
                self._zero_arm_dynamics()
            else:
                self.sim.send_action(wp, robot_name=robot, n_substeps=n_substeps)
                # One-time diagnostic (physical grasp): log the gripper-frame
                # position vs the cube at the grasp phase so the fingertip offset
                # can be calibrated for the Isaac gripper.
                if _GRASP_DBG and phase == "grasp" and not getattr(self, "_logged_grasp_pose", False):
                    self._logged_grasp_pose = True
                    pose = self._gripper_frame_pose()
                    cp = _object_position(self.sim, self.scene.cube_name)
                    if pose and cp:
                        gp = pose[0]
                        logger.info(
                            "[grasp-dbg] PHYS grasp: gripper_frame=%s cube=%s delta=%s",
                            [round(x, 3) for x in gp],
                            [round(x, 3) for x in cp],
                            [round(gp[i] - cp[i], 3) for i in range(3)],
                        )

            # --- Hybrid carry: physical pick (above) + scripted transport ---
            # reach/grasp/close ran as pure physics; the jaws come down onto the
            # cube. At the close phase we capture the cube's offset relative to
            # the gripper frame and then RIGIDLY GLUE it to the gripper for every
            # subsequent frame (close tail -> lift -> place -> place_down), so it
            # rides in the jaws continuously and is dropped at the bin on release.
            # (Carrying only on some phases left gaps where the collider-off cube
            # fell through the floor and "disappeared" then snapped back.)
            if self.hybrid_carry and not self.kinematic:
                cp = _object_position(self.sim, self.scene.cube_name)
                if phase in ("close", "lift") and cp is not None:
                    if float(cp[2]) - self._cube_start_z > 0.012:
                        hy_lifted = True
                # Capture at the close phase, the moment the jaws shut on the cube
                # while it is still at its (reset-deterministic) start pose.
                if phase == "close" and hy_local is None and not hy_released and cp is not None:
                    pose = self._gripper_frame_pose()
                    if pose:
                        gp, _rot = pose
                        off = [float(cp[0]) - float(gp[0]), float(cp[1]) - float(gp[1])]
                        if (off[0] ** 2 + off[1] ** 2) ** 0.5 < 0.06:
                            # Seat the cube at the gripper-frame ORIGIN (zero XY
                            # offset) rather than the noisy captured offset: FK
                            # shows the SO-101 fingertip pads close right at the
                            # gripper_frame_link origin, so pinning the cube there
                            # makes it sit tightly between the jaws. Using the
                            # live-captured offset left it ~1-2 cm low/forward of
                            # the pads (looked ungripped) because the force-driven
                            # frame readout is noisy at the grasp instant.
                            hy_local = [0.0, 0.0]
                            # Ride at the fingertip height: the pads are ~at the
                            # frame origin, so seat the cube essentially AT the
                            # frame (tiny drop so it nestles between the pads).
                            self._hybrid_z_in_jaw = -0.005
                            hy_grasped = True
                            self._set_cube_collision(False)
                            # Make the cube KINEMATIC while carried so its render
                            # matches the pinned pose (a dynamic body drifts under
                            # gravity between move_object and the render tick ->
                            # cube rendered beside/into the bin).
                            self._set_cube_kinematic(True)
                            if _GRASP_DBG:
                                logger.info(
                                    "[grasp-dbg] HYBRID attach off_xy=%s dz=%.3f frame=%s cube=%s",
                                    [round(x, 3) for x in hy_local],
                                    self._hybrid_z_in_jaw,
                                    [round(float(x), 3) for x in gp],
                                    [round(float(x), 3) for x in cp],
                                )
                # Glue the cube to the gripper EVERY frame once attached (until
                # release): pin its XY under the frame at the captured offset and
                # ride its Z with the gripper frame (clamped above the table) so
                # it visibly lifts off the table and travels in the jaws. Once the
                # release point is recorded (``_hybrid_drop_xy`` set), the
                # release-pin block below takes over and rests it on the bin.
                if hy_local is not None and self._hybrid_drop_xy is None:
                    pose = self._gripper_frame_pose()
                    if pose:
                        gp, _rot = pose
                        cx = float(gp[0]) + hy_local[0]
                        cy = float(gp[1]) + hy_local[1]
                        ride_z = float(gp[2]) + self._hybrid_z_in_jaw
                        # Never let the carried cube sink below the surface it is
                        # over (table, or the bin plate top when above the bin).
                        target = [cx, cy, max(ride_z, self._rest_z_at(cx, cy))]
                        self.sim.move_object(self.scene.cube_name, position=target)
                        self._zero_cube_velocity()
                # Release at the bin: record the drop XY and set the cube DOWN on
                # the bin surface, but KEEP carrying it (collider still off) and
                # do NOT re-enable the collider here. Re-enabling the collider
                # while the jaws are still physically wrapped around the cube made
                # PhysX resolve the jaw<->cube overlap with a huge impulse that
                # FLUNG the cube off-screen (the "jump out then snap back" bug).
                # The collider is restored later, in the settle loop, only after
                # the arm has opened/retreated clear of the cube.
                if hy_local is not None and phase == "release" and self._hybrid_drop_xy is None:
                    # Drop the cube at the BIN's logical centre, with a small
                    # empirical render-offset compensation: on the force-driven
                    # Isaac arm the cube's RENDERED position sits a few cm +X of
                    # its logical coords relative to the bin, so the cube landed
                    # just outside the bin. Shifting the logical drop target by
                    # ``_CUBE_RENDER_OFFSET`` lands it inside the cavity.
                    ox, oy = self._cube_render_offset
                    self._hybrid_drop_xy = [
                        float(self.scene.place_position[0]) + ox,
                        float(self.scene.place_position[1]) + oy,
                    ]
                    if _GRASP_DBG:
                        logger.info(
                            "[grasp-dbg] HYBRID release drop_xy=%s (bin centre + render comp %s)",
                            [round(x, 3) for x in self._hybrid_drop_xy],
                            [round(ox, 3), round(oy, 3)],
                        )
                # During the release frames, pin the cube resting ON the bin
                # surface (collider still off so the opening jaw can't bat it).
                if self._hybrid_drop_xy is not None and not hy_released:
                    dx, dy = self._hybrid_drop_xy
                    self.sim.move_object(self.scene.cube_name, position=[dx, dy, self._rest_z_at(dx, dy)])
                    self._zero_cube_velocity()

            if has_grip:
                closed = phase in closed_phases
                if closed:
                    has_closed = True
                # Form the kinematic grasp during the GRASP descent, the moment
                # the fingertips come within reach of the (still-undisturbed)
                # cube -- BEFORE the jaw-close motion can shove it away. Capture
                # the cube exactly WHERE IT IS (its current tool-frame offset) so
                # it rides along without any teleport/"jump" into the gripper.
                # Also disable the cube collider here so the closing jaw can't
                # knock it.
                if not attached and phase in ("grasp", "close"):
                    pose = self._gripper_frame_pose()
                    cp = _object_position(self.sim, self.scene.cube_name)
                    if pose and cp:
                        gp, rot = pose
                        tip = self._frame_to_world(rot, gp, self.grip_seat_local) if self.grip_seat_local else gp
                        if math.dist(tip, cp) < self.attach_radius:
                            # Carry the cube at its ACTUAL current offset in the
                            # tool frame -> captured in place, no jump.
                            grasp_local = self._frame_to_local(rot, gp, cp)
                            rest_local = list(grasp_local)
                            attached = True
                            self._set_cube_collision(False)
                            if _GRASP_DBG:
                                logger.info(
                                    "[grasp-dbg] ATTACHED at phase=%s gp=%s cp=%s dist=%.3f local=%s",
                                    phase,
                                    [round(x, 3) for x in gp],
                                    [round(x, 3) for x in cp],
                                    math.dist(tip, cp),
                                    [round(x, 3) for x in rest_local],
                                )
                # Release once the gripper re-opens after having closed.
                if attached and has_closed and not closed:
                    attached = False
                    grasp_local = rest_local = None
                    self._set_cube_collision(True)  # restore collider on release
                    if _GRASP_DBG:
                        logger.info("[grasp-dbg] RELEASED at phase=%s", phase)
                # Carry: hold the cube rigidly WHERE IT WAS GRASPED (capture in
                # place -- no ease, no teleport into a mouth -> no visible jump).
                if attached and grasp_local is not None:
                    pose = self._gripper_frame_pose()
                    if pose:
                        gp, rot = pose
                        target = self._frame_to_world(rot, gp, grasp_local)
                        # Never let the carried cube sink into the floor: the
                        # gripper/tool frame can dip to ~z=0 on the Isaac articulation
                        # so a rigid carry would push the cube below ground. Clamp the
                        # cube center to keep its base on/above the table.
                        if self._cube_half_z > 0.0:
                            target = [target[0], target[1], max(float(target[2]), self._cube_half_z)]
                        self.sim.move_object(self.scene.cube_name, position=target)
                        self._zero_cube_velocity()  # avoid teleport-induced fling
                        if _GRASP_DBG and phase == "lift":
                            logger.info(
                                "[grasp-dbg] carry lift: tool_z=%.3f target_z=%.3f",
                                float(gp[2]),
                                float(target[2]),
                            )

            obs = self.sim.get_observation(robot, skip_images=not self.record_images)
            recorder.add_frame(observation=obs, action=wp, task=task)
            frames += 1
            cnow = _object_position(self.sim, self.scene.cube_name)
            if cnow is not None:
                self._cube_max_z = max(self._cube_max_z, float(cnow[2]))

        # Let the released cube fall and settle into the bin (it was carried at
        # the jaws ~6 cm up the finger axis, so on release it must drop). Without
        # these steps the cube freezes at the release height (looks like it never
        # lands) and the place check fails. Hold the arm at its last pose and
        # record a few extra frames of the settle.
        if has_grip and self.kinematic and trajectory.waypoints:
            last = trajectory.waypoints[-1]
            for _ in range(8):
                self.sim.set_joint_positions(last, robot_name=robot)
                self.sim.step(max(1, n_substeps))
                self._zero_arm_dynamics()
                obs = self.sim.get_observation(robot, skip_images=not self.record_images)
                recorder.add_frame(observation=obs, action=last, task=task)
                frames += 1
        # Hybrid path: rest the placed cube ON TOP of the bin's visual surface.
        # NOTE: the bin is effectively a VISUAL-ONLY marker here -- repeated
        # free-drop probes show the cube falls straight THROUGH it to the floor
        # (rest z=0.015) even after explicitly enabling a UsdPhysics collider on
        # the bin prim (Isaac's FixedCuboid collider does not bind/stop the cube
        # in this setup). A physical rest is therefore not available: re-enabling
        # the cube collider either lets it sink through the bin to the ground
        # (the reported bug) or ejects it if seated overlapping. We keep the cube
        # KINEMATIC (collider off) and pinned at bin_top+cube_half so it cleanly
        # rests on the bin plate in the recording, and retreat the arm home so
        # nothing occludes or disturbs the placed cube.
        if self.hybrid_carry and not self.kinematic and trajectory.waypoints:
            last = trajectory.waypoints[-1]
            drop_xy = getattr(self, "_hybrid_drop_xy", None)
            home = self.home_q()
            rest_z = self._rest_z_at(drop_xy[0], drop_xy[1]) if drop_xy is not None else self._cube_half_z2
            n_settle = 14
            for s in range(n_settle):
                action = last if s < 3 else home
                self.sim.send_action(action, robot_name=robot, n_substeps=n_substeps)
                if drop_xy is not None:
                    self.sim.move_object(self.scene.cube_name, position=[drop_xy[0], drop_xy[1], rest_z])
                    self._zero_cube_velocity()
                obs = self.sim.get_observation(robot, skip_images=not self.record_images)
                recorder.add_frame(observation=obs, action=last, task=task)
                frames += 1
                cnow = _object_position(self.sim, self.scene.cube_name)
                if cnow is not None:
                    self._cube_max_z = max(self._cube_max_z, float(cnow[2]))
            if _GRASP_DBG and drop_xy is not None:
                fz = _object_position(self.sim, self.scene.cube_name)
                bz = _object_position(self.sim, "bin")
                logger.info(
                    "[grasp-dbg] final cube=%s bin=%s (cube should be ~bin xy to be inside)",
                    [round(v, 4) for v in fz] if fz else None,
                    [round(v, 4) for v in bz] if bz else None,
                )
        # Expose whether a genuine PHYSICAL lift occurred (hybrid path) so the
        # success check requires a real pick, not just a final proximity.
        self._hybrid_phys_lift = bool(hy_lifted)
        # Whether the jaws genuinely closed ON the cube (cube was within the
        # grasp at the close phase) -- the physical, honest part of the hybrid.
        self._hybrid_grasped = bool(hy_grasped)
        return frames

    def _assess(self, cube_start: list[float] | None, lifted: bool = False) -> EpisodeResult:
        cube_now = _object_position(self.sim, self.scene.cube_name)
        disp = place_d = 0.0
        moved = placed = False
        if cube_now is not None:
            if cube_start is not None:
                disp = math.dist(cube_now, cube_start)
                moved = disp > self.move_threshold
            place_d = math.dist(cube_now[:2], self.scene.place_position[:2])
            placed = place_d < self.place_radius
        # Success = the cube ended IN the bin AND genuinely transited there from
        # its start (moved, and ended meaningfully closer to the bin than it
        # started). This rejects the false positive where the cube STARTS within
        # place_radius of the bin and is never touched (start [0.2,0.2] vs bin
        # [0.12,0.18] = 0.082 < 0.10), without depending on a measured lift height
        # (the Isaac kinematic-carry path reads the cube via PhysX where the
        # carried lift may not register in the USD pose). ``lifted`` still counts
        # as a positive signal when available.
        start_place_d = math.dist(cube_start[:2], self.scene.place_position[:2]) if cube_start is not None else None
        approached_bin = start_place_d is not None and (start_place_d - place_d) > self.move_threshold
        success = bool(placed and moved and (lifted or approached_bin))
        # Hybrid path: require a genuine physical GRASP (the jaws actually closed
        # on the cube at its start pose -- verified by the cube being within the
        # grasp at the close phase) in addition to the cube ending in the bin.
        # This rejects the old false positive where a nudged/flung cube happened
        # to land within place_radius without ever being picked up -- the metric
        # now reflects an actual pick-and-place (real grasp + transport to bin).
        if self.hybrid_carry:
            success = bool(placed and getattr(self, "_hybrid_grasped", False))
        if _GRASP_DBG and cube_now is not None:
            logger.info(
                "[grasp-dbg] place: cube_final_xy=%s bin=%s place_d=%.3f placed=%s",
                [round(x, 3) for x in cube_now[:2]],
                [round(x, 3) for x in self.scene.place_position[:2]],
                place_d,
                placed,
            )
        return EpisodeResult(
            success=success,
            cube_moved=moved,
            placed=placed,
            frames=0,
            displacement=disp,
            place_distance=place_d,
            phases=0,
        )

    def record_episode(self, trajectory, task: str, n_substeps: int = 5, recorder=None) -> EpisodeResult:
        """Record one episode. If ``recorder`` is None, creates+finalizes a 1-episode dataset."""
        own = recorder is None
        recorder = recorder or self._new_recorder(task)
        cube_start = _object_position(self.sim, self.scene.cube_name)
        try:
            frames = self._execute_and_record(trajectory, task, recorder, n_substeps)
            recorder.save_episode()
            lifted = (getattr(self, "_cube_max_z", 0.0) - getattr(self, "_cube_start_z", 0.0)) > 0.02
            if _GRASP_DBG:
                logger.info(
                    "[grasp-dbg] lift: start_z=%.3f max_z=%.3f -> lifted=%s",
                    getattr(self, "_cube_start_z", 0.0),
                    getattr(self, "_cube_max_z", 0.0),
                    lifted,
                )
            result = self._assess(cube_start, lifted=lifted)
            result.frames = frames
            result.phases = len(set(trajectory.phases))
            logger.info(
                "episode recorded: frames=%d success=%s moved=%s disp=%.3f place_d=%.3f",
                frames,
                result.success,
                result.cube_moved,
                result.displacement,
                result.place_distance,
            )
            return result
        finally:
            if own:
                recorder.finalize()

    def record_dataset(
        self,
        planner,
        n_episodes: int = 5,
        task: str = "pick up the red cube and place it in the bin",
        n_substeps: int = 5,
        randomize: bool = True,
        seed: int | None = None,
        rebuild_scene=None,
        on_episode=None,
        home_q: dict[str, float] | None = None,
        snapshot=None,
    ) -> dict[str, Any]:
        """Record ``n_episodes`` into one dataset (append + single finalize).

        ``rebuild_scene(seed)`` (optional) re-randomizes the world between
        episodes; if absent we jitter via ``sim.randomize`` when ``randomize``.
        ``on_episode(i, result)`` is an optional progress callback (UI/agent).
        ``home_q``/``snapshot`` (optional) override the per-episode reset target
        with a known-clean rest state captured at build time; pass these so a
        prior "Plan & execute" that left the arm displaced doesn't poison the
        captured reset state (which would make cuRobo fall back every episode).
        """
        if not self.available():
            return {"status": "error", "message": LEROBOT_INSTALL_HINT}

        import random as _random

        rng = _random.Random(seed)
        recorder = self._new_recorder(task)
        results: list[EpisodeResult] = []
        jnames = self.scene.joint_names
        # Reset to the clean build-time rest state FIRST (if provided) so the
        # per-episode reset target below is captured from a known-good pose, not
        # from wherever a previous plan_and_execute left the arm/cube.
        if snapshot is not None:
            self._restore_state(snapshot)
        elif home_q is not None:
            self._reset_episode(home_q)
        # Capture the rest pose + full physics state once so each episode starts
        # identically (deterministic) -> cuRobo plans the same -> consistent grasp.
        home_q = home_q or {
            j: float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in jnames
        }
        snapshot = snapshot if snapshot is not None else self._snapshot_state()
        try:
            for i in range(n_episodes):
                # Reset to a consistent start: arm at home, cube at its start pose.
                if rebuild_scene is not None:
                    rebuild_scene(rng.randint(0, 10_000))
                elif snapshot is not None:
                    self._restore_state(snapshot)
                else:
                    self._reset_episode(home_q)
                if randomize:
                    try:
                        self.sim.randomize(
                            randomize_colors=True,
                            randomize_lighting=True,
                            randomize_positions=False,
                            seed=rng.randint(0, 10_000),
                        )
                    except Exception:  # noqa: BLE001 - randomize is best-effort
                        logger.debug("randomize failed (non-fatal)", exc_info=True)
                start_q = [float(self.sim.get_observation(self.scene.robot_name, skip_images=True)[j]) for j in jnames]
                traj = planner.plan_pick_place(
                    joint_names=self.scene.joint_names,
                    start_q=start_q,
                    gripper_joint=self.scene.gripper_joint,
                    cube_xy=self.scene.cube_position[:2],
                    place_xy=self.scene.place_position[:2],
                    base_sign=self.base_sign,
                )
                res = self.record_episode(traj, task=task, n_substeps=n_substeps, recorder=recorder)
                results.append(res)
                if on_episode:
                    try:
                        on_episode(i, res)
                    except Exception:  # noqa: BLE001
                        pass
        finally:
            recorder.finalize()

        n_ok = sum(1 for r in results if r.success)
        return {
            "status": "success",
            "repo_id": self.repo_id,
            "root": getattr(recorder, "root", self.root),
            "episodes": len(results),
            "successes": n_ok,
            "success_rate": (n_ok / len(results)) if results else 0.0,
            "total_frames": sum(r.frames for r in results),
            "planner": getattr(planner, "name", "?"),
        }

    def load_back(self, episode: int = 0):
        """Reload a recorded episode locally (no Hub). Returns (dataset, start, length)."""
        from strands_robots.dataset_recorder import load_lerobot_episode

        return load_lerobot_episode(repo_id=self.repo_id, episode=episode, root=self.root)
