"""Camera management — add, inject, remove."""

from __future__ import annotations

import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List

from ._registry import _ensure_mujoco
from ._types import SimCamera

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def add_camera(
    sim: MujocoBackend,
    name: str,
    position: List[float] = None,
    target: List[float] = None,
    fov: float = 60.0,
    width: int = 640,
    height: int = 480,
) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}

    cam = SimCamera(
        name=name,
        position=position or [1.0, 1.0, 1.0],
        target=target or [0.0, 0.0, 0.0],
        fov=fov,
        width=width,
        height=height,
    )
    sim._world.cameras[name] = cam

    # If robots are loaded, inject camera via XML round-trip to preserve
    # actuators/joints (build_objects_only would destroy them).
    if sim._world.robots and sim._world._model is not None:
        try:
            _inject_camera_into_scene(sim, cam)
        except Exception as e:
            logger.warning(
                "Camera injection failed: %s. Camera tracked as metadata only.", e
            )
    else:
        from ._scene import recompile_world
        recompile_world(sim)

    return {
        "status": "success",
        "content": [{"text": f"📷 Camera '{name}' added at {cam.position}"}],
    }


def _inject_camera_into_scene(sim: MujocoBackend, cam: SimCamera) -> bool:
    """Inject a camera into a running simulation via XML round-trip.

    Same approach as _inject_object_into_scene: saves the current model
    to XML, injects a <camera> element, and reloads. Preserves all
    actuators/joints because they are serialized in the saved XML.
    """
    mj = _ensure_mujoco()
    if sim._world._model is None:
        return False

    robot_base_dir = None
    if sim._world._robot_base_xml:
        robot_base_dir = os.path.dirname(
            os.path.abspath(sim._world._robot_base_xml)
        )

    if robot_base_dir and os.path.isdir(robot_base_dir):
        scene_path = os.path.join(robot_base_dir, "_strands_scene_with_cameras.xml")
    else:
        tmpdir = tempfile.mkdtemp(prefix="strands_cam_")
        scene_path = os.path.join(tmpdir, "scene_with_cameras.xml")

    mj.mj_saveLastXML(scene_path, sim._world._model)

    with open(scene_path) as f:
        xml_content = f.read()

    px, py, pz = cam.position
    cam_xml = f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>'
    xml_content = xml_content.replace("</worldbody>", f"{cam_xml}\n</worldbody>")

    with open(scene_path, "w") as f:
        f.write(xml_content)

    try:
        new_model = mj.MjModel.from_xml_path(str(scene_path))
        new_data = mj.MjData(new_model)

        old_nq = min(sim._world._data.qpos.shape[0], new_data.qpos.shape[0])
        old_nv = min(sim._world._data.qvel.shape[0], new_data.qvel.shape[0])
        new_data.qpos[:old_nq] = sim._world._data.qpos[:old_nq]
        new_data.qvel[:old_nv] = sim._world._data.qvel[:old_nv]

        old_nu = min(sim._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
        new_data.ctrl[:old_nu] = sim._world._data.ctrl[:old_nu]

        mj.mj_forward(new_model, new_data)

        sim._world._model = new_model
        sim._world._data = new_data

        for robot_name, robot in sim._world.robots.items():
            robot.joint_ids = []
            robot.actuator_ids = []
            for jnt_name in robot.joint_names:
                jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
                if jid >= 0:
                    robot.joint_ids.append(jid)
            for i in range(new_model.nu):
                robot.actuator_ids.append(i)

        try:
            if scene_path.endswith("_strands_scene_with_cameras.xml"):
                os.remove(scene_path)
        except OSError:
            pass

        return True
    except Exception as e:
        logger.error("Camera injection reload failed: %s", e)
        try:
            if scene_path.endswith("_strands_scene_with_cameras.xml"):
                os.remove(scene_path)
        except OSError:
            pass
        return False


def remove_camera(sim: MujocoBackend, name: str) -> Dict[str, Any]:
    if sim._world is None or name not in sim._world.cameras:
        return {
            "status": "error",
            "content": [{"text": f"❌ Camera '{name}' not found."}],
        }
    del sim._world.cameras[name]
    return {
        "status": "success",
        "content": [{"text": f"🗑️ Camera '{name}' removed."}],
    }
