"""Object management — add, inject, remove, eject, move, list."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import xml.etree.ElementTree as ET
from typing import TYPE_CHECKING, Any, Dict, List

from ._builder import MJCFBuilder
from ._registry import _ensure_mujoco
from ._types import SimObject

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def add_object(
    sim: MujocoBackend,
    name: str,
    shape: str = "box",
    position: List[float] = None,
    orientation: List[float] = None,
    size: List[float] = None,
    color: List[float] = None,
    mass: float = 0.1,
    is_static: bool = False,
    mesh_path: str = None,
) -> Dict[str, Any]:
    """Add an object to the simulation.

    If a robot has been loaded, objects cannot be added via XML recompilation
    (it would destroy robot actuators). In that case, objects are injected
    via XML round-trip to preserve actuators.

    For robot-free worlds, objects are compiled into the MJCF XML directly.
    """
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}
    if name in sim._world.objects:
        return {
            "status": "error",
            "content": [{"text": f"❌ Object '{name}' exists."}],
        }

    obj = SimObject(
        name=name,
        shape=shape,
        position=position or [0.0, 0.0, 0.0],
        orientation=orientation or [1.0, 0.0, 0.0, 0.0],
        size=size or [0.05, 0.05, 0.05],
        color=color or [0.5, 0.5, 0.5, 1.0],
        mass=mass,
        mesh_path=mesh_path,
        is_static=is_static,
    )
    sim._world.objects[name] = obj

    # If a robot is loaded, inject the object via XML round-trip
    # (preserves actuators since they're serialized in the XML)
    if sim._world.robots:
        try:
            result = _inject_object_into_scene(sim, obj)
            if result:
                return {
                    "status": "success",
                    "content": [
                        {"text": f"📦 '{name}' spawned: {shape} at {obj.position}"}
                    ],
                }
            else:
                # Fallback: metadata only
                logger.info(
                    "Object '%s' registered (injection returned False — metadata only)",
                    name,
                )
                return {
                    "status": "success",
                    "content": [
                        {
                            "text": (
                                f"📦 '{name}' registered: {shape} at {obj.position}\n"
                                f"⚠️ Robot scene loaded — object is tracked but not physically spawned.\n"
                                f"💡 For objects + robots: use load_scene with a pre-built scene,\n"
                                f"   or use MuJoCo's interactive viewer to position objects."
                            )
                        }
                    ],
                }
        except Exception as e:
            logger.warning("Object injection failed, tracking metadata only: %s", e)
            return {
                "status": "success",
                "content": [
                    {
                        "text": (
                            f"📦 '{name}' registered: {shape} at {obj.position}\n"
                            f"⚠️ Injection failed ({e}) — object tracked as metadata only."
                        )
                    }
                ],
            }

    # No robots — safe to recompile XML
    from ._scene import recompile_world
    result = recompile_world(sim)
    if result["status"] == "error":
        del sim._world.objects[name]
        return result

    return {
        "status": "success",
        "content": [
            {
                "text": (
                    f"📦 '{name}' added: {shape} at {obj.position}, "
                    f"size={obj.size}, {'static' if is_static else f'{mass}kg'}"
                )
            }
        ],
    }


def _inject_object_into_scene(sim: MujocoBackend, obj: SimObject) -> bool:
    """Inject object into a running simulation via XML round-trip.

    Saves current model to XML, injects the object body XML before
    </worldbody>, then reloads. Preserves actuators because they are
    serialized in the saved XML.
    """
    mj = _ensure_mujoco()
    if sim._world._model is None:
        return False

    # Determine the original model directory for mesh resolution
    robot_base_dir = None
    if sim._world._robot_base_xml:
        robot_base_dir = os.path.dirname(
            os.path.abspath(sim._world._robot_base_xml)
        )

    # Always use tempdir to avoid polluting shared asset directories.
    tmpdir = tempfile.mkdtemp(prefix="strands_sim_")
    scene_path = os.path.join(tmpdir, "scene_with_objects.xml")

    mj.mj_saveLastXML(scene_path, sim._world._model)

    # Patch meshdir/texturedir to point to original asset dir for mesh resolution
    if robot_base_dir and os.path.isdir(robot_base_dir):
        with open(scene_path) as f:
            xml_content = f.read()
        if "<compiler" in xml_content and "meshdir=" not in xml_content:
            xml_content = xml_content.replace(
                "<compiler",
                f'<compiler meshdir="{robot_base_dir}" texturedir="{robot_base_dir}"',
                1,
            )
        with open(scene_path, "w") as f:
            f.write(xml_content)

    # Read XML, inject object before </worldbody>
    with open(scene_path) as f:
        xml_content = f.read()

    obj_xml = MJCFBuilder._object_xml(obj, indent=4)
    xml_content = xml_content.replace("</worldbody>", f"{obj_xml}\n</worldbody>")

    with open(scene_path, "w") as f:
        f.write(xml_content)

    # Reload model (preserves actuators since they're in the XML)
    try:
        new_model = mj.MjModel.from_xml_path(str(scene_path))
        new_data = mj.MjData(new_model)

        # Copy state from old model
        old_nq = min(sim._world._data.qpos.shape[0], new_data.qpos.shape[0])
        old_nv = min(sim._world._data.qvel.shape[0], new_data.qvel.shape[0])
        new_data.qpos[:old_nq] = sim._world._data.qpos[:old_nq]
        new_data.qvel[:old_nv] = sim._world._data.qvel[:old_nv]

        # Copy ctrl
        old_nu = min(sim._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
        new_data.ctrl[:old_nu] = sim._world._data.ctrl[:old_nu]

        mj.mj_forward(new_model, new_data)

        # Update world
        sim._world._model = new_model
        sim._world._data = new_data

        # Re-discover robot joints/actuators (IDs may shift)
        for robot_name, robot in sim._world.robots.items():
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
            # Fallback: if no actuators matched by joint, assign all (single-robot scene)
            if not robot.actuator_ids:
                for i in range(new_model.nu):
                    robot.actuator_ids.append(i)

        # Clean up temp directory
        shutil.rmtree(tmpdir, ignore_errors=True)
        return True
    except Exception as e:
        logger.error("Object injection reload failed: %s", e)
        shutil.rmtree(tmpdir, ignore_errors=True)
        return False


def remove_object(sim: MujocoBackend, name: str) -> Dict[str, Any]:
    if sim._world is None or name not in sim._world.objects:
        return {
            "status": "error",
            "content": [{"text": f"❌ Object '{name}' not found."}],
        }
    del sim._world.objects[name]

    # Use XML round-trip when robots are present to preserve their state.
    if sim._world.robots:
        _eject_body_from_scene(sim, name)
    else:
        from ._scene import recompile_world
        recompile_world(sim)

    return {
        "status": "success",
        "content": [{"text": f"🗑️ Object '{name}' removed."}],
    }


def _eject_body_from_scene(sim: MujocoBackend, body_name: str) -> bool:
    """Remove a named body from the scene via XML round-trip, preserving robot state."""
    mj = _ensure_mujoco()

    tmpdir = tempfile.mkdtemp(prefix="strands_eject_")
    scene_path = os.path.join(tmpdir, "scene_ejected.xml")

    try:
        mj.mj_saveLastXML(scene_path, sim._world._model)

        tree = ET.parse(scene_path)
        root = tree.getroot()

        # Patch meshdir / texturedir if needed
        robot_base_dir = None
        if sim._world._robot_base_xml:
            robot_base_dir = os.path.dirname(
                os.path.abspath(sim._world._robot_base_xml)
            )
        if robot_base_dir:
            compiler = root.find("compiler")
            if compiler is not None:
                if compiler.get("meshdir") is None:
                    compiler.set("meshdir", robot_base_dir)
                if compiler.get("texturedir") is None:
                    compiler.set("texturedir", robot_base_dir)

        # Walk the tree and remove the target body element.
        removed = False
        for parent in root.iter():
            for child in list(parent):
                if child.tag == "body" and child.get("name") == body_name:
                    parent.remove(child)
                    removed = True

        if not removed:
            logger.warning(
                "Body '%s' not found in MJCF XML — skipping ejection.", body_name
            )

        tree.write(scene_path, xml_declaration=True)

        new_model = mj.MjModel.from_xml_path(scene_path)
        new_data = mj.MjData(new_model)

        # Copy state
        old_nq = min(sim._world._data.qpos.shape[0], new_data.qpos.shape[0])
        new_data.qpos[:old_nq] = sim._world._data.qpos[:old_nq]
        old_nv = min(sim._world._data.qvel.shape[0], new_data.qvel.shape[0])
        new_data.qvel[:old_nv] = sim._world._data.qvel[:old_nv]
        old_nu = min(sim._world._data.ctrl.shape[0], new_data.ctrl.shape[0])
        new_data.ctrl[:old_nu] = sim._world._data.ctrl[:old_nu]

        mj.mj_forward(new_model, new_data)
        sim._world._model = new_model
        sim._world._data = new_data
        return True
    except Exception as e:
        logger.error("Body ejection failed for '%s': %s", body_name, e)
        from ._scene import recompile_world
        recompile_world(sim)
        return False
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


def move_object(
    sim: MujocoBackend,
    name: str,
    position: List[float] = None,
    orientation: List[float] = None,
) -> Dict[str, Any]:
    if sim._world is None or sim._world._data is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}
    if name not in sim._world.objects:
        return {"status": "error", "content": [{"text": f"❌ '{name}' not found."}]}

    mj = _ensure_mujoco()
    model, data = sim._world._model, sim._world._data

    jnt_name = f"{name}_joint"
    jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
    if jnt_id >= 0:
        qpos_addr = model.jnt_qposadr[jnt_id]
        if position:
            data.qpos[qpos_addr : qpos_addr + 3] = position
            sim._world.objects[name].position = position
        if orientation:
            data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
            sim._world.objects[name].orientation = orientation
        mj.mj_forward(model, data)

    return {
        "status": "success",
        "content": [{"text": f"📍 '{name}' moved to {position or 'same'}"}],
    }


def list_objects(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}
    if not sim._world.objects:
        return {"status": "success", "content": [{"text": "No objects."}]}

    lines = ["📦 Objects:\n"]
    for name, obj in sim._world.objects.items():
        lines.append(
            f"  • {name}: {obj.shape} at {obj.position}, "
            f"{'static' if obj.is_static else f'{obj.mass}kg'}"
        )
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}
