"""Robot management — add, remove, list, get state."""

from __future__ import annotations

import logging
import os
import re
from typing import TYPE_CHECKING, Any, Dict, List

from ._registry import _ensure_mujoco, resolve_model
from ._types import SimCamera, SimRobot

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def ensure_meshes(model_path: str, robot_name: str):
    """Check if mesh files referenced by a model XML exist; auto-download if missing.

    Inspects the MJCF XML (and any <include>d files) for mesh file
    references and, if any are missing, triggers an automatic download
    from MuJoCo Menagerie via the asset download system.
    This is a no-op when all meshes are already present.
    """
    model_dir = os.path.dirname(os.path.abspath(model_path))

    # Collect XML files to inspect (model itself + any <include>d files)
    files_to_check = [model_path]
    try:
        with open(model_path) as _f:
            top_content = _f.read()
        for inc in re.findall(r'<include\s+file="([^"]+)"', top_content):
            inc_path = os.path.join(model_dir, inc)
            if os.path.exists(inc_path):
                files_to_check.append(inc_path)
    except Exception:
        pass

    # Scan all XML files for mesh references
    missing = False
    for xml_path in files_to_check:
        try:
            with open(xml_path) as _f:
                content = _f.read()
        except Exception:
            continue

        mesh_files = re.findall(r'file="([^"]+\.(?:stl|STL|obj))"', content)
        if not mesh_files:
            continue

        meshdir_match = re.search(r'meshdir="([^"]*)"', content)
        meshdir = meshdir_match.group(1) if meshdir_match else ""
        xml_dir = os.path.dirname(os.path.abspath(xml_path))

        for mf in mesh_files:
            if not os.path.exists(os.path.join(xml_dir, meshdir, mf)):
                missing = True
                break
        if missing:
            break

    if not missing:
        return

    # Auto-download from Menagerie
    logger.info(
        "Downloading mesh files for '%s' from MuJoCo Menagerie (first time only)...",
        robot_name,
    )
    try:
        from strands_robots.assets import resolve_robot_name
        from strands_robots.assets.download import download_robots

        canonical = resolve_robot_name(robot_name)
        download_robots(names=[canonical], force=True)
    except Exception as e:
        logger.warning(
            "Auto-download failed for '%s': %s. "
            "Try manually: python -m strands_robots.assets.download %s",
            robot_name, e, robot_name,
        )


def add_robot(
    sim: MujocoBackend,
    name: str,
    urdf_path: str = None,
    data_config: str = None,
    position: List[float] = None,
    orientation: List[float] = None,
) -> Dict[str, Any]:
    """Add a robot to the simulation.

    Loads the robot's MJCF/URDF scene file directly and replaces the
    simulation model. Objects added via add_object will be injected
    into the loaded model as MuJoCo XML bodies.

    Args:
        name: Unique robot name
        urdf_path: Direct path to URDF/MJCF file
        data_config: Data config name — auto-resolves from asset registry
        position: [x, y, z] position
        orientation: [w, x, y, z] quaternion
    """
    if sim._world is None:
        return {
            "status": "error",
            "content": [{"text": "❌ No world. Use action='create_world' first."}],
        }

    if name in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{name}' already exists."}],
        }

    # Resolve model path (prefers Menagerie scene files with objects/cameras)
    resolved_path = urdf_path
    if not resolved_path and data_config:
        resolved_path = resolve_model(data_config)
        if not resolved_path:
            return {
                "status": "error",
                "content": [
                    {
                        "text": (
                            f"❌ No model found for '{data_config}'.\n"
                            f"💡 Use action='list_urdfs' to see available robots\n"
                            f"💡 Register with: action='register_urdf', data_config='...', urdf_path='...'"
                        )
                    }
                ],
            }
    elif not resolved_path and name:
        # Try resolving the name itself as a robot identifier
        resolved_path = resolve_model(name)

    if not resolved_path:
        return {
            "status": "error",
            "content": [
                {"text": "❌ Either urdf_path or data_config is required."}
            ],
        }

    if not os.path.exists(resolved_path):
        return {
            "status": "error",
            "content": [{"text": f"❌ File not found: {resolved_path}"}],
        }

    mj = _ensure_mujoco()

    robot = SimRobot(
        name=name,
        urdf_path=resolved_path,
        position=position or [0.0, 0.0, 0.0],
        orientation=orientation or [1.0, 0.0, 0.0, 0.0],
        data_config=data_config,
        namespace=f"{name}/",
    )

    try:
        # Auto-download missing mesh assets before loading
        ensure_meshes(resolved_path, data_config or name)

        # Load the robot scene (this becomes the simulation model)
        model = mj.MjModel.from_xml_path(str(resolved_path))
        data = mj.MjData(model)

        # Discover joints and actuators from the loaded model
        joint_names = []
        for i in range(model.njnt):
            jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
            if jnt_name:
                joint_names.append(jnt_name)
                robot.joint_ids.append(i)
        robot.joint_names = joint_names

        for i in range(model.nu):
            act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
            # Only assign actuators that relate to this robot's joints
            if act_name:
                # Check if actuator's joint is one of this robot's joints
                jnt_id = model.actuator_trnid[i, 0]
                if jnt_id in robot.joint_ids:
                    robot.actuator_ids.append(i)
                elif not robot.actuator_ids:
                    # Fallback: if no match found yet, assign all (single-robot scene)
                    pass
            else:
                robot.actuator_ids.append(i)
        # If no actuators were matched by joint, assign all (backward compat for single-robot)
        if not robot.actuator_ids:
            for i in range(model.nu):
                robot.actuator_ids.append(i)

        # Discover cameras already in the scene
        for i in range(model.ncam):
            cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
            if cam_name and cam_name not in sim._world.cameras:
                sim._world.cameras[cam_name] = SimCamera(
                    name=cam_name,
                    camera_id=i,
                    width=sim.default_width,
                    height=sim.default_height,
                )

        # Replace the world model with this robot's scene
        sim._world._model = model
        sim._world._data = data
        sim._world._robot_base_xml = resolved_path  # Remember for recompilation
        sim._world.robots[name] = robot

        # Settle physics
        for _ in range(100):
            mj.mj_step(model, data)

        logger.info("Robot '%s' added from %s", name, os.path.basename(resolved_path))
        source = (
            f"data_config='{data_config}'"
            if data_config
            else os.path.basename(resolved_path)
        )
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"🤖 Robot '{name}' added to simulation\n"
                        f"📁 Source: {source} → {os.path.basename(resolved_path)}\n"
                        f"📍 Position: {robot.position}\n"
                        f"🔩 Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                        f"⚡ Actuators: {len(robot.actuator_ids)}\n"
                        f"📷 Cameras: {list(sim._world.cameras.keys())}\n"
                        f"💡 Run policy: action='run_policy', robot_name='{name}'"
                    )
                }
            ],
        }

    except Exception as e:
        logger.error("Failed to add robot '%s': %s", name, e)
        return {"status": "error", "content": [{"text": f"❌ Failed to load: {e}"}]}


def remove_robot(sim: MujocoBackend, name: str) -> Dict[str, Any]:
    if sim._world is None or name not in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{name}' not found."}],
        }
    if name in sim._policy_threads:
        sim._world.robots[name].policy_running = False
        # Wait for the policy thread to finish before removing the robot
        future = sim._policy_threads[name]
        try:
            future.result(timeout=5.0)
        except Exception:
            logger.debug("Policy thread for '%s' did not finish cleanly", name)
        del sim._policy_threads[name]
    del sim._world.robots[name]
    return {
        "status": "success",
        "content": [{"text": f"🗑️ Robot '{name}' removed."}],
    }


def list_robots(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}
    if not sim._world.robots:
        return {
            "status": "success",
            "content": [{"text": "No robots. Use action='add_robot'."}],
        }

    lines = ["🤖 Robots:\n"]
    for name, robot in sim._world.robots.items():
        status = "🟢 running" if robot.policy_running else "⚪ idle"
        lines.append(
            f"  • {name}: {len(robot.joint_names)} joints, "
            f"{len(robot.actuator_ids)} actuators [{status}]"
        )
    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def get_robot_state(sim: MujocoBackend, robot_name: str) -> Dict[str, Any]:
    if sim._world is None or sim._world._data is None:
        return {
            "status": "error",
            "content": [{"text": "❌ No simulation running."}],
        }
    if robot_name not in sim._world.robots:
        return {
            "status": "error",
            "content": [{"text": f"❌ Robot '{robot_name}' not found."}],
        }

    mj = _ensure_mujoco()
    robot = sim._world.robots[robot_name]
    model, data = sim._world._model, sim._world._data

    state = {}
    for jnt_name in robot.joint_names:
        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            qpos_addr = model.jnt_qposadr[jnt_id]
            qvel_addr = model.jnt_dofadr[jnt_id]
            state[jnt_name] = {
                "position": float(data.qpos[qpos_addr]),
                "velocity": float(data.qvel[qvel_addr]),
            }

    text = f"🤖 '{robot_name}' state (t={sim._world.sim_time:.3f}s):\n"
    for jnt, vals in state.items():
        text += f"  {jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"

    return {
        "status": "success",
        "content": [{"text": text}, {"json": {"state": state}}],
    }
