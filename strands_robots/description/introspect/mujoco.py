"""Read a model's joints, actuators, limits and keyframes through MuJoCo."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def introspect_mujoco(path: Path) -> dict[str, Any]:
    import mujoco

    m = mujoco.MjModel.from_xml_path(str(path))
    joints, limits, floating = [], {}, False
    for i in range(m.njnt):
        jt = int(m.jnt_type[i])
        if jt == mujoco.mjtJoint.mjJNT_FREE:
            floating = True
            continue
        if jt == mujoco.mjtJoint.mjJNT_BALL:
            continue
        name = m.joint(i).name or f"joint{i}"
        joints.append(name)
        if m.jnt_limited[i]:
            limits[name] = (float(m.jnt_range[i, 0]), float(m.jnt_range[i, 1]))
    actuators, action_joints = [], []
    for i in range(m.nu):
        actuators.append(m.actuator(i).name or f"actuator{i}")
        if int(m.actuator_trntype[i]) == mujoco.mjtTrn.mjTRN_JOINT:
            action_joints.append(m.joint(int(m.actuator_trnid[i, 0])).name)
        else:
            action_joints.append(actuators[-1])
    return {
        "joints": joints,
        "actuators": actuators,
        "action_joints": action_joints,
        "limits": limits,
        "keyframes": [m.key(i).name or f"key{i}" for i in range(m.nkey)],
        "floating": floating,
    }
