"""Carry joint positions, velocities and actuator targets across a recompile, by name; and set_gains' PD gains."""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


@dataclass
class Snapshot:
    """One world's state keyed by name, so a recompiled world keeps what the old one measured."""

    joints: dict[str, tuple[np.ndarray, np.ndarray]]  # joint name -> (qpos slice, qvel slice)
    ctrl: dict[str, float]  # actuator name -> control value


def snapshot(model: mujoco.MjModel | None, data: mujoco.MjData | None) -> Snapshot | None:
    if model is None or data is None:
        return None
    m, d = model, data
    joints: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for j in range(m.njnt):
        jt = m.jnt_type[j]
        nq = 7 if jt == mujoco.mjtJoint.mjJNT_FREE else (4 if jt == mujoco.mjtJoint.mjJNT_BALL else 1)
        nv = 6 if nq == 7 else (3 if nq == 4 else 1)
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        joints[m.joint(j).name] = (d.qpos[qa : qa + nq].copy(), d.qvel[va : va + nv].copy())
    ctrl = {m.actuator(i).name: float(d.ctrl[i]) for i in range(m.nu)}
    return Snapshot(joints, ctrl)


def restore(model: mujoco.MjModel, data: mujoco.MjData, snap: Snapshot | None) -> None:
    if snap is None:
        return
    m, d = model, data
    for name, (q, v) in snap.joints.items():
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
        if j < 0:
            continue
        qa, va = m.jnt_qposadr[j], m.jnt_dofadr[j]
        d.qpos[qa : qa + len(q)] = q
        d.qvel[va : va + len(v)] = v
    for name, c in snap.ctrl.items():
        a = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if a >= 0:
            d.ctrl[a] = c


def place(model: mujoco.MjModel, data: mujoco.MjData, qadr: list[int], dadr: list[int], positions: np.ndarray) -> None:
    """Joints at ``positions`` (a ``-1`` address is a tendon actuator: skipped), at rest, kinematics recomputed."""
    for qa, da, q in zip(qadr, dadr, positions, strict=True):
        if qa >= 0:
            data.qpos[qa], data.qvel[da] = q, 0.0
    mujoco.mj_forward(model, data)


Gains = dict[str, tuple[np.ndarray, np.ndarray]]  # robot id -> the (kp, kd) set_gains wrote


def write_gains(model: mujoco.MjModel, ids: list[int], kp: np.ndarray, kd: np.ndarray) -> None:
    """Actuators ``ids`` as the PD law ``tau = kp·(target − q) − kd·q̇`` (no recompile)."""
    for i, p, dgain in zip(ids, kp, kd, strict=True):
        model.actuator_gainprm[i, 0] = p
        model.actuator_biasprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = -p
        model.actuator_biasprm[i, 2] = -dgain


def rewrite_gains(model: mujoco.MjModel, kept: Gains, ids: dict[str, list[int]]) -> Gains:
    """A compile starts from the MJCF's gains: write what ``set_gains`` wrote again, for each robot still in ``ids``.

    A removed robot's gains go with it, so a new robot given its id starts from its own MJCF.
    """
    kept = {body: g for body, g in kept.items() if body in ids}
    for body, (kp, kd) in kept.items():
        write_gains(model, ids[body], kp, kd)
    return kept
