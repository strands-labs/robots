"""XML round-trip injection/ejection for scene modification.

Shared helper `_reload_scene_from_xml` handles the common pattern:
save XML → patch paths → modify → reload → copy state → re-discover joints.
"""

import logging
import os
import re
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import Optional

from strands_robots.simulation._backend import _ensure_mujoco
from strands_robots.simulation._mjcf_builder import MJCFBuilder
from strands_robots.simulation._models import SimCamera, SimObject, SimWorld

logger = logging.getLogger(__name__)


def _patch_xml_paths(xml_content: str, robot_base_dir: str) -> str:
    """Patch meshdir/texturedir in XML to absolute paths for tmpdir loading."""
    meshdir_match = re.search(r'meshdir="([^"]*)"', xml_content)
    existing_meshdir = meshdir_match.group(1) if meshdir_match else ""
    abs_meshdir = os.path.normpath(os.path.join(robot_base_dir, existing_meshdir))

    texdir_match = re.search(r'texturedir="([^"]*)"', xml_content)
    existing_texdir = texdir_match.group(1) if texdir_match else ""
    abs_texdir = os.path.normpath(os.path.join(robot_base_dir, existing_texdir))

    if meshdir_match:
        xml_content = re.sub(r'meshdir="[^"]*"', f'meshdir="{abs_meshdir}"', xml_content)
    elif "<compiler" in xml_content:
        xml_content = xml_content.replace("<compiler", f'<compiler meshdir="{robot_base_dir}"', 1)

    if texdir_match:
        xml_content = re.sub(r'texturedir="[^"]*"', f'texturedir="{abs_texdir}"', xml_content)
    elif "<compiler" in xml_content and "texturedir" not in xml_content:
        xml_content = xml_content.replace("<compiler", f'<compiler texturedir="{robot_base_dir}"', 1)

    return xml_content


def _reload_scene_from_xml(world: SimWorld, scene_path: str) -> bool:
    """Reload MuJoCo model from modified XML, preserving state.

    Copies qpos, qvel, ctrl from old model and re-discovers robot joint/actuator IDs.
    """
    mj = _ensure_mujoco()
    new_model = mj.MjModel.from_xml_path(str(scene_path))
    new_data = mj.MjData(new_model)

    # Copy state from old model
    old_nq = min(world._data.qpos.shape[0], new_data.qpos.shape[0])
    old_nv = min(world._data.qvel.shape[0], new_data.qvel.shape[0])
    new_data.qpos[:old_nq] = world._data.qpos[:old_nq]
    new_data.qvel[:old_nv] = world._data.qvel[:old_nv]
    old_nu = min(world._data.ctrl.shape[0], new_data.ctrl.shape[0])
    new_data.ctrl[:old_nu] = world._data.ctrl[:old_nu]

    mj.mj_forward(new_model, new_data)

    world._model = new_model
    world._data = new_data

    # Re-discover robot joints/actuators (IDs may shift)
    for robot in world.robots.values():
        robot.joint_ids = []
        robot.actuator_ids = []
        for jnt_name in robot.joint_names:
            jid = mj.mj_name2id(new_model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jid >= 0:
                robot.joint_ids.append(jid)
        for i in range(new_model.nu):
            jnt_id = new_model.actuator_trnid[i, 0]
            if jnt_id in robot.joint_ids:
                robot.actuator_ids.append(i)
        if not robot.actuator_ids:
            for i in range(new_model.nu):
                robot.actuator_ids.append(i)

    return True


def _get_robot_base_dir(world: SimWorld) -> Optional[str]:
    """Get the directory of the original robot model file."""
    if world._robot_base_xml:
        return os.path.dirname(os.path.abspath(world._robot_base_xml))
    return None


def _save_and_patch_xml(world: SimWorld, tmpdir: str, filename: str) -> str:
    """Save current model to XML in tmpdir and patch asset paths."""
    mj = _ensure_mujoco()
    scene_path = os.path.join(tmpdir, filename)
    mj.mj_saveLastXML(scene_path, world._model)

    robot_base_dir = _get_robot_base_dir(world)
    if robot_base_dir and os.path.isdir(robot_base_dir):
        with open(scene_path) as f:
            xml_content = f.read()
        xml_content = _patch_xml_paths(xml_content, robot_base_dir)
        with open(scene_path, "w") as f:
            f.write(xml_content)

    return scene_path


def inject_object_into_scene(world: SimWorld, obj: SimObject) -> bool:
    """Inject object into a running simulation via XML round-trip."""
    _ensure_mujoco()
    if world._model is None:
        return False

    tmpdir = tempfile.mkdtemp(prefix="strands_sim_")
    try:
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_with_objects.xml")

        with open(scene_path) as f:
            xml_content = f.read()

        obj_xml = MJCFBuilder._object_xml(obj, indent=4)
        xml_content = xml_content.replace("</worldbody>", f"{obj_xml}\n</worldbody>")

        # Remove keyframes — adding a freejoint changes qpos size
        xml_content = re.sub(r"<keyframe>.*?</keyframe>", "", xml_content, flags=re.DOTALL)

        with open(scene_path, "w") as f:
            f.write(xml_content)

        return _reload_scene_from_xml(world, scene_path)
    except Exception as e:
        logger.error("Object injection reload failed: %s", e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def eject_body_from_scene(world: SimWorld, body_name: str) -> bool:
    """Remove a named body from the scene via XML round-trip."""
    mj = _ensure_mujoco()

    tmpdir = tempfile.mkdtemp(prefix="strands_eject_")
    try:
        scene_path = os.path.join(tmpdir, "scene_ejected.xml")
        mj.mj_saveLastXML(scene_path, world._model)

        tree = ET.parse(scene_path)
        root = tree.getroot()

        # Patch paths
        robot_base_dir = _get_robot_base_dir(world)
        if robot_base_dir:
            compiler = root.find("compiler")
            if compiler is not None:
                existing_meshdir = compiler.get("meshdir", "")
                compiler.set("meshdir", os.path.normpath(os.path.join(robot_base_dir, existing_meshdir)))
                existing_texdir = compiler.get("texturedir", "")
                compiler.set("texturedir", os.path.normpath(os.path.join(robot_base_dir, existing_texdir)))

        # Remove target body
        removed = False
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "body" and child.get("name") == body_name:
                    parent.remove(child)
                    removed = True

        if not removed:
            logger.warning(f"Body '{body_name}' not found in MJCF XML — skipping ejection.")

        # Remove keyframes
        for keyframe_elem in root.findall("keyframe"):
            root.remove(keyframe_elem)

        tree.write(scene_path, xml_declaration=True)

        return _reload_scene_from_xml(world, scene_path)
    except Exception as e:
        logger.error("Body ejection failed for '%s': %s", body_name, e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def inject_camera_into_scene(world: SimWorld, cam: SimCamera) -> bool:
    """Inject a camera into a running simulation via XML round-trip."""
    _ensure_mujoco()
    if world._model is None:
        return False

    tmpdir = tempfile.mkdtemp(prefix="strands_cam_")
    try:
        scene_path = _save_and_patch_xml(world, tmpdir, "scene_with_cameras.xml")

        with open(scene_path) as f:
            xml_content = f.read()

        px, py, pz = cam.position
        cam_xml = f'    <camera name="{cam.name}" pos="{px} {py} {pz}" fovy="{cam.fov}" mode="fixed"/>'
        xml_content = xml_content.replace("</worldbody>", f"{cam_xml}\n</worldbody>")

        with open(scene_path, "w") as f:
            f.write(xml_content)

        return _reload_scene_from_xml(world, scene_path)
    except Exception as e:
        logger.error("Camera injection reload failed: %s", e)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
