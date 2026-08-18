"""Bridge from Kimodo qpos output to a ProtoMotions MotionPlayer cache.

Kimodo emits a full-body qpos trajectory of shape ``[T, 36]``: three root xyz
translations, four ``wxyz`` root quaternion elements, then twenty-nine G1 joint
positions in the canonical WBC joint order (which is the same as
:data:`~strands_robots.policies.protomotions.config.GTP_G1_JOINT_NAMES`, so no
per-joint reordering is needed).

The ProtoMotions Generalist Tracking Policy consumes a
:class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer` - a
dict of per-frame joint states AND per-body rigid-body states (position,
rotation, linear velocity, angular velocity). This module builds that dict by
running MuJoCo forward-kinematics on the same G1 MJCF the tracker was trained
on, then finite-differencing to fill the velocity channels.

Runs entirely in numpy + mujoco. Reused at policy build time (a one-off cost
in the tens of milliseconds per motion) - never on the hot path.
"""

from __future__ import annotations

import logging
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from strands_robots.utils import require_optional

from .config import GTP_G1_BODY_NAMES, GTP_G1_JOINT_NAMES, GTP_G1_ROOT_BODY_INDEX
from .motion_utils import lerp, slerp
from .state_utils import mujoco_wxyz_to_xyzw

logger = logging.getLogger(__name__)

__all__ = ["qpos_to_motion_data"]

#: qpos entries a free root joint occupies: xyz translation + wxyz quaternion.
_ROOT_QPOS_WIDTH = 7

#: Total qpos width the tracker's embodiment has: the free root plus one
#: entry per :data:`GTP_G1_JOINT_NAMES` hinge.
_EXPECTED_NQ = _ROOT_QPOS_WIDTH + len(GTP_G1_JOINT_NAMES)


# ---------------------------------------------------------------------------
# MJCF patching
# ---------------------------------------------------------------------------


def _declares_a_ground_geom(mujoco, mjcf_path: Path) -> bool:
    """Report whether ``mjcf_path`` already declares a ground geom.

    The question is about the model MuJoCo will build from the file, not about
    one section of the file's XML: MuJoCo merges *every* ``<worldbody>`` a file
    declares, splices ``<include>``d content in, and compiles geoms wherever
    they are nested inside bodies. Reading the first ``<worldbody>``'s direct
    ``<geom>`` children answers a narrower question and misses all three, so
    this asks MuJoCo's own parser for the flat geom list instead - the same rule
    :mod:`strands_robots.simulation.mujoco.spec_builder` applies when it decides
    whether a world already owns a ground plane.

    Parsing a spec resolves includes without compiling meshes (single-digit
    milliseconds on a G1, against ~170ms for a full compile), so this stays well
    inside this module's one-off build-time cost.

    Args:
        mujoco: The resolved ``mujoco`` module.
        mjcf_path: Path to the MJCF to inspect.

    Returns:
        ``True`` when the parsed model declares a plane geom, or a geom whose
        name reads as a floor or ground, and a patched-in floor would therefore
        be a second ground plane sharing the ``floor`` name.
    """
    spec = mujoco.MjSpec.from_file(str(mjcf_path))  # type: ignore[attr-defined]
    return any(
        "floor" in (geom.name or "").lower()
        or "ground" in (geom.name or "").lower()
        or geom.type == mujoco.mjtGeom.mjGEOM_PLANE  # type: ignore[attr-defined]
        for geom in spec.geoms
    )


def _patch_and_load_mjcf(mjcf_path: Path):
    """Load MJCF with a floor geom + no sensors - required for FK."""
    mujoco = require_optional(
        "mujoco",
        extra="sim-mujoco",
        purpose="forward kinematics for the reference-motion bridge",
    )

    tree = ET.parse(str(mjcf_path))
    root = tree.getroot()

    # Sensors add DOFs - strip so qpos indexing stays canonical.
    for sensor_elem in list(root.findall("sensor")):
        root.remove(sensor_elem)

    worldbody = root.find("worldbody")
    if worldbody is not None and not _declares_a_ground_geom(mujoco, mjcf_path):
        floor = ET.SubElement(worldbody, "geom")
        floor.set("name", "floor")
        floor.set("type", "plane")
        floor.set("size", "0 0 0.05")
        floor.set("rgba", "0.7 0.7 0.7 1")

    xml_str = ET.tostring(root, encoding="unicode")

    # Write the patched XML back into the SAME directory so MJCF asset paths
    # (mesh files, textures) resolve relative to the original location.
    asset_dir = str(mjcf_path.parent)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xml", dir=asset_dir, delete=False) as f:
        f.write(xml_str)
        tmp_path = f.name

    try:
        model = mujoco.MjModel.from_xml_path(tmp_path)  # type: ignore[attr-defined]
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:  # pragma: no cover - best-effort cleanup
            pass

    data = mujoco.MjData(model)  # type: ignore[attr-defined]
    return mujoco, model, data


# ---------------------------------------------------------------------------
# Embodiment check
# ---------------------------------------------------------------------------


def _proto_body_rows(mujoco, model, mjcf_path: Path) -> np.ndarray:
    """Resolve the MuJoCo body ids the cache's body rows must hold.

    The tracker reads a body out of the cache by ROW INDEX, not by name:
    :attr:`~strands_robots.policies.protomotions.config.ProtoMotionsConfig.anchor_body_index`
    and ``root_body_index`` are offsets into
    :data:`~strands_robots.policies.protomotions.config.GTP_G1_BODY_NAMES`. So the
    cache's rows have to be that list, in that order - MuJoCo's own body order is
    only the same thing when the supplied MJCF is the tracker's own embodiment.
    Resolving each row by name keeps the two orders from being paired positionally.

    Args:
        mujoco: The imported ``mujoco`` module.
        model: A compiled ``MjModel``.
        mjcf_path: Path the model was compiled from, quoted in refusals.

    Returns:
        MuJoCo body ids, one per :data:`GTP_G1_BODY_NAMES` entry, in that order.

    Raises:
        ValueError: If the model's qpos layout is not a free root followed by the
            tracker's joints, or if it does not carry every tracker body.
    """
    has_free_root = model.njnt > 0 and model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
    if not has_free_root or model.nq != _EXPECTED_NQ:
        root = "a free root joint" if has_free_root else "no free root joint"
        raise ValueError(
            f"ProtoMotions G1 MJCF {mjcf_path} does not have the tracker's qpos "
            f"layout: it has {root} and nq={model.nq}, but the cache is built for "
            f"nq={_EXPECTED_NQ} (a free root's {_ROOT_QPOS_WIDTH} entries plus "
            f"{len(GTP_G1_JOINT_NAMES)} joints). This is the MJCF's layout, not the "
            f"qpos argument's - supply the MJCF the tracker checkpoint was exported "
            f"against."
        )

    rows: list[int] = []
    missing: list[str] = []
    for name in GTP_G1_BODY_NAMES:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            missing.append(name)
        else:
            rows.append(body_id)
    if missing:
        raise ValueError(
            f"ProtoMotions G1 MJCF {mjcf_path} is missing {len(missing)} of the "
            f"{len(GTP_G1_BODY_NAMES)} bodies the tracker reads by row index: "
            f"{missing!r}. The tracker indexes cache rows against "
            f"GTP_G1_BODY_NAMES (row 0 is the root, row "
            f"{GTP_G1_BODY_NAMES.index('torso_link')} is the anchor), so a model "
            f"with a different body set shifts every row after the gap and the "
            f"tracker reads a different link than it asked for. Supply the MJCF the "
            f"tracker checkpoint was exported against."
        )
    return np.asarray(rows, dtype=np.intp)


# ---------------------------------------------------------------------------
# Angular velocity via quaternion finite diff
# ---------------------------------------------------------------------------


def _quat_finite_diff_ang_vel(quats_xyzw: np.ndarray, dt: float) -> np.ndarray:
    """Approximate body angular velocities from an ``xyzw`` quaternion trajectory.

    Uses ``omega ~ 2 * (q_{t+1} - q_t) * q_t^-1 / dt`` (small-angle diff),
    then keeps the vector part. Output copies the last frame from the
    second-to-last so shapes match the input trajectory.

    The delta is left-multiplied (``q_{t+1} * q_t^-1``), so the result is a
    WORLD-frame angular velocity, matching the frame every other surface here
    uses for a per-body angular-velocity array: the ``rigid_body_ang_vel``
    argument of
    :func:`~strands_robots.policies.protomotions.state_utils.compute_root_local_ang_vel`,
    which exists to rotate one body's row of it into that body's local frame.
    Right-multiplying instead (``q_t^-1 * q_{t+1}``) would give the local-frame
    quantity; on a walking G1 the two differ by whole rad/s, so a caller that
    treats one as the other feeds the tracker a wrong reference.

    Args:
        quats_xyzw: Shape ``[T, num_bodies, 4]`` xyzw quaternions.
        dt: Source period, seconds.

    Returns:
        Shape ``[T, num_bodies, 3]`` world-frame angular velocities.
    """
    T = quats_xyzw.shape[0]
    if T < 2:
        return np.zeros(quats_xyzw.shape[:-1] + (3,), dtype=np.float32)

    q0 = quats_xyzw[:-1]
    q1 = quats_xyzw[1:]
    # Conjugate q0.
    q0_conj = q0.copy()
    q0_conj[..., :3] *= -1.0

    # Hamilton product q1 * q0^-1 -> element-wise (broadcast-safe).
    ax, ay, az, aw = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    bx, by, bz, bw = q0_conj[..., 0], q0_conj[..., 1], q0_conj[..., 2], q0_conj[..., 3]
    dq = np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )
    ang_vel = 2.0 * dq[..., :3] / max(dt, 1e-8)
    # Repeat last frame so output has same T.
    out = np.zeros(quats_xyzw.shape[:-1] + (3,), dtype=np.float32)
    out[:-1] = ang_vel.astype(np.float32)
    out[-1] = out[-2]
    return out


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def qpos_to_motion_data(
    qpos: np.ndarray,
    fps: float,
    proto_mjcf_path: str | Path,
    control_dt: float = 0.02,
) -> dict:
    """Convert a Kimodo-style G1 ``qpos`` trajectory to a MotionPlayer cache.

    Steps:
    1. Load the ProtoMotions G1 MJCF and its FK data buffer.
    2. For each source frame: set ``qpos``, call ``mj_forward``, read body
       xpos + xquat.
    3. Finite-difference for joint + body linear + body angular velocities.
    4. Resample onto ``control_dt`` with SLERP + LERP.

    Args:
        qpos: Shape ``[T, 36]`` - ``root_xyz(3) + root_quat_wxyz(4) +
            joints(29)``. Anything wider than 36 is truncated on the joint end
            with a warning (some Kimodo variants emit trailing padding).
        fps: Source frame rate in Hz (Kimodo default is 30).
        proto_mjcf_path: Path to the ProtoMotions G1 MJCF (the caller supplies
            it; this module ships no asset bundle). It must be the tracker's own
            embodiment: a free root plus the
            :data:`~strands_robots.policies.protomotions.config.GTP_G1_JOINT_NAMES`
            joints, carrying every
            :data:`~strands_robots.policies.protomotions.config.GTP_G1_BODY_NAMES`
            body. A G1 variant with a different body set is refused rather than
            bridged into rows the tracker would misread.
        control_dt: Target control period, seconds (default 0.02 = 50Hz).

    Returns:
        A dict with the keys
        :class:`~strands_robots.policies.protomotions.motion_utils.MotionPlayer`
        accepts. ``body_pos``/``body_rot`` are world poses read off MuJoCo's
        ``xpos``/``xquat``, one row per
        :data:`~strands_robots.policies.protomotions.config.GTP_G1_BODY_NAMES`
        entry IN THAT ORDER - the order the tracker's ``anchor_body_index`` and
        ``root_body_index`` are offsets into, resolved by name rather than by
        MuJoCo's own body order. ``body_vel``/``body_ang_vel`` are WORLD-frame
        velocities derived from them - not body-local ones.

    Raises:
        FileNotFoundError: If ``proto_mjcf_path`` does not exist.
        ValueError: If ``qpos.shape[1]`` is not exactly 36 (after truncation), or
            if the MJCF is not the tracker's embodiment - a distinct message that
            names the MJCF, so a model-side mismatch is not read as a bad ``qpos``.
        RuntimeError: If MuJoCo is not installed.
    """
    proto_mjcf_path = Path(proto_mjcf_path)
    if not proto_mjcf_path.exists():
        raise FileNotFoundError(
            f"ProtoMotions G1 MJCF not found: {proto_mjcf_path}. Supply the "
            f"path via `proto_mjcf_path=...` - no asset is bundled."
        )

    qpos = np.asarray(qpos, dtype=np.float64)
    if qpos.ndim != 2:
        raise ValueError(f"qpos must be 2-D [T, 36], got shape {qpos.shape}.")
    if qpos.shape[1] < 36:
        raise ValueError(f"qpos must have at least 36 columns (root_xyz + root_quat + 29 joints), got {qpos.shape[1]}.")
    if qpos.shape[1] > 36:
        logger.warning(
            "qpos has %d columns (>36) - truncating trailing padding.",
            qpos.shape[1],
        )
        qpos = qpos[:, :36]

    T = qpos.shape[0]
    mujoco, model, data = _patch_and_load_mjcf(proto_mjcf_path)

    body_rows = _proto_body_rows(mujoco, model, proto_mjcf_path)
    num_bodies = len(body_rows)
    num_dofs = model.nq - _ROOT_QPOS_WIDTH

    logger.info(
        "qpos_to_motion_data: MJCF has %d bodies, %d dofs. Processing %d frames @ %.1f Hz.",
        num_bodies,
        num_dofs,
        T,
        fps,
    )

    body_pos = np.zeros((T, num_bodies, 3), dtype=np.float32)
    body_rot_xyzw = np.zeros((T, num_bodies, 4), dtype=np.float32)
    dof_pos = np.zeros((T, num_dofs), dtype=np.float32)

    for t in range(T):
        data.qpos[:] = qpos[t]
        data.qvel[:] = 0.0
        mujoco.mj_forward(model, data)  # type: ignore[attr-defined]
        body_pos[t] = data.xpos[body_rows].astype(np.float32)
        body_rot_xyzw[t] = mujoco_wxyz_to_xyzw(data.xquat[body_rows]).astype(np.float32)
        # Root body's rot is the canonical freejoint quaternion from qpos.
        body_rot_xyzw[t, GTP_G1_ROOT_BODY_INDEX] = mujoco_wxyz_to_xyzw(data.qpos[3:_ROOT_QPOS_WIDTH].astype(np.float32))
        dof_pos[t] = data.qpos[_ROOT_QPOS_WIDTH:].astype(np.float32)

    dt_src = 1.0 / max(fps, 1e-6)
    dof_vel = np.zeros_like(dof_pos)
    body_vel = np.zeros_like(body_pos)
    body_ang_vel = _quat_finite_diff_ang_vel(body_rot_xyzw, dt_src)
    if T > 1:
        dof_vel[:-1] = (dof_pos[1:] - dof_pos[:-1]) / dt_src
        dof_vel[-1] = dof_vel[-2]
        body_vel[:-1] = (body_pos[1:] - body_pos[:-1]) / dt_src
        body_vel[-1] = body_vel[-2]

    target_fps = 1.0 / control_dt
    if abs(fps - target_fps) < 0.5:
        return {
            "dof_pos": dof_pos,
            "dof_vel": dof_vel,
            "body_rot": body_rot_xyzw,
            "body_pos": body_pos,
            "body_vel": body_vel,
            "body_ang_vel": body_ang_vel,
            "control_dt": control_dt,
            "num_frames": T,
        }

    # SLERP / LERP resample onto control_dt.
    motion_length = dt_src * (T - 1)
    num_ctrl_frames = max(1, int(round(motion_length / control_dt)) + 1)

    body_pos_ctrl = np.zeros((num_ctrl_frames, num_bodies, 3), dtype=np.float32)
    body_rot_ctrl = np.zeros((num_ctrl_frames, num_bodies, 4), dtype=np.float32)
    body_vel_ctrl = np.zeros((num_ctrl_frames, num_bodies, 3), dtype=np.float32)
    body_ang_vel_ctrl = np.zeros((num_ctrl_frames, num_bodies, 3), dtype=np.float32)
    dof_pos_ctrl = np.zeros((num_ctrl_frames, num_dofs), dtype=np.float32)
    dof_vel_ctrl = np.zeros((num_ctrl_frames, num_dofs), dtype=np.float32)

    for i in range(num_ctrl_frames):
        time_s = i * control_dt
        phase = min(max(time_s / max(motion_length, 1e-8), 0.0), 1.0)
        frame_f = phase * (T - 1)
        f0 = int(frame_f)
        f1 = min(f0 + 1, T - 1)
        blend = np.float32(frame_f - f0)
        body_pos_ctrl[i] = lerp(body_pos[f0], body_pos[f1], blend)
        body_rot_ctrl[i] = slerp(body_rot_xyzw[f0], body_rot_xyzw[f1], blend)
        body_vel_ctrl[i] = lerp(body_vel[f0], body_vel[f1], blend)
        body_ang_vel_ctrl[i] = lerp(body_ang_vel[f0], body_ang_vel[f1], blend)
        dof_pos_ctrl[i] = lerp(dof_pos[f0], dof_pos[f1], blend)
        dof_vel_ctrl[i] = lerp(dof_vel[f0], dof_vel[f1], blend)

    logger.info(
        "qpos_to_motion_data: resampled %d frames @ %.1f Hz -> %d frames @ %.0f Hz.",
        T,
        fps,
        num_ctrl_frames,
        target_fps,
    )

    return {
        "dof_pos": dof_pos_ctrl,
        "dof_vel": dof_vel_ctrl,
        "body_rot": body_rot_ctrl,
        "body_pos": body_pos_ctrl,
        "body_vel": body_vel_ctrl,
        "body_ang_vel": body_ang_vel_ctrl,
        "control_dt": control_dt,
        "num_frames": num_ctrl_frames,
    }
