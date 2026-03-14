"""Simulation class — thin orchestrator composing mixins.

Replaces the 3800-line monolith with a ~300-line class that delegates to
extracted modules via mixin inheritance.
"""

import json
import logging
import os
import re
import threading
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

from strands.tools.tools import AgentTool
from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from strands_robots.simulation._backend import _ensure_mujoco
from strands_robots.simulation._mjcf_builder import MJCFBuilder
from strands_robots.simulation._model_registry import (
    list_available_models,
    register_urdf,
    resolve_model,
)
from strands_robots.simulation._models import SimCamera, SimObject, SimRobot, SimStatus, SimWorld
from strands_robots.simulation._policy_runner import PolicyRunnerMixin
from strands_robots.simulation._randomization import RandomizationMixin
from strands_robots.simulation._recording import RecordingMixin
from strands_robots.simulation._rendering import RenderingMixin
from strands_robots.simulation._scene_ops import (
    eject_body_from_scene,
    inject_camera_into_scene,
    inject_object_into_scene,
)

logger = logging.getLogger(__name__)

_TOOL_SPEC_PATH = Path(__file__).parent / "_tool_spec.json"


class Simulation(
    PolicyRunnerMixin,
    RenderingMixin,
    RecordingMixin,
    RandomizationMixin,
    AgentTool,
):
    """Programmatic simulation environment as a Strands AgentTool.

    Gives AI agents the ability to create, modify, and control MuJoCo
    simulation environments through natural language → tool actions.
    """

    def __init__(
        self,
        tool_name: str = "sim",
        default_timestep: float = 0.002,
        default_width: int = 640,
        default_height: int = 480,
        mesh: bool = True,
        peer_id: str = None,
        **kwargs,
    ):
        super().__init__()
        self.tool_name_str = tool_name
        self.default_timestep = default_timestep
        self.default_width = default_width
        self.default_height = default_height

        self._world: Optional[SimWorld] = None
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix=f"{tool_name}_sim")
        self._policy_threads: Dict[str, Future] = {}
        self._shutdown_event = threading.Event()
        self._lock = threading.Lock()

        self._viewer_handle = None
        self._viewer_thread = None

        self._renderers: Dict[tuple, Any] = {}
        self._renderer_model = None

        logger.info("🎮 Simulation tool '%s' initialized", tool_name)

        try:
            from strands_robots.zenoh_mesh import init_mesh

            self.mesh = init_mesh(self, peer_id=peer_id, peer_type="sim", mesh=mesh)
        except Exception as e:
            logger.debug("Mesh init skipped: %s", e)
            self.mesh = None

    # --- Public Properties ---

    @property
    def mj_model(self):
        """Direct access to the MuJoCo model (mujoco.MjModel)."""
        return self._world._model if self._world else None

    @property
    def mj_data(self):
        """Direct access to the MuJoCo data (mujoco.MjData)."""
        return self._world._data if self._world else None

    # --- Robot-compatible interface ---

    def get_observation(self, robot_name: str = None, camera_name: str = None) -> Dict[str, Any]:
        """Get observation from simulation (Robot ABC compatible)."""
        if self._world is None or self._world._model is None:
            return {}
        if robot_name is None:
            if not self._world.robots:
                return {}
            robot_name = next(iter(self._world.robots))
        if robot_name not in self._world.robots:
            return {}
        return self._get_sim_observation(robot_name, cam_name=camera_name)

    def send_action(self, action: Dict[str, Any], robot_name: str = None, n_substeps: int = 1) -> None:
        """Apply action to simulation (Robot ABC compatible)."""
        if self._world is None or self._world._model is None:
            return
        if robot_name is None:
            if not self._world.robots:
                return
            robot_name = next(iter(self._world.robots))
        if robot_name not in self._world.robots:
            return
        self._apply_sim_action(robot_name, action, n_substeps=n_substeps)

    # --- World Management ---

    def _cheap_robot_count(self) -> int:
        try:
            from strands_robots.registry import list_robots as _registry_list_robots

            return len(_registry_list_robots(mode="sim"))
        except ImportError:
            return 0

    def create_world(self, timestep: float = None, gravity: List[float] = None, ground_plane: bool = True) -> Dict[str, Any]:
        """Create a new simulation world."""
        _ensure_mujoco()

        if self._world is not None and self._world._model is not None:
            return {"status": "error", "content": [{"text": "❌ World already exists. Use action='destroy' first, or action='reset'."}]}

        if gravity is None:
            _gravity = [0.0, 0.0, -9.81]
        elif isinstance(gravity, (int, float)):
            _gravity = [0.0, 0.0, float(gravity)]
        else:
            _gravity = list(gravity)

        self._world = SimWorld(
            timestep=timestep or self.default_timestep,
            gravity=_gravity,
            ground_plane=ground_plane,
        )

        self._world.cameras["default"] = SimCamera(
            name="default",
            position=[1.5, 1.5, 1.2],
            target=[0.0, 0.0, 0.3],
            width=self.default_width,
            height=self.default_height,
        )

        self._compile_world()

        return {
            "status": "success",
            "content": [{"text": (
                "🌍 Simulation world created\n"
                f"⚙️ Timestep: {self._world.timestep}s ({1 / self._world.timestep:.0f}Hz physics)\n"
                f"🌐 Gravity: {self._world.gravity}\n"
                f"📷 Default camera ready\n"
                f"🤖 Robot models: {self._cheap_robot_count()} available\n"
                "💡 Add robots: action='add_robot' (urdf_path or data_config)\n"
                "💡 Add objects: action='add_object'\n"
                "💡 List URDFs: action='list_urdfs'"
            )}],
        }

    def load_scene(self, scene_path: str) -> Dict[str, Any]:
        """Load a complete scene from MJCF XML or URDF file."""
        mj = _ensure_mujoco()

        if not os.path.exists(scene_path):
            return {"status": "error", "content": [{"text": f"❌ Scene file not found: {scene_path}"}]}

        try:
            self._world = SimWorld()
            self._world._model = mj.MjModel.from_xml_path(str(scene_path))
            self._world._data = mj.MjData(self._world._model)
            self._world.status = SimStatus.IDLE

            return {
                "status": "success",
                "content": [{"text": (
                    f"🌍 Scene loaded from {os.path.basename(scene_path)}\n"
                    f"🦴 Bodies: {self._world._model.nbody}, 🔩 Joints: {self._world._model.njnt}, ⚡ Actuators: {self._world._model.nu}\n"
                    "💡 Use action='get_state' to inspect, action='step' to simulate"
                )}],
            }
        except Exception as e:
            logger.error("Failed to load scene: %s", e)
            return {"status": "error", "content": [{"text": f"❌ Failed to load scene: {e}"}]}

    def _compile_world(self):
        mj = _ensure_mujoco()
        xml = MJCFBuilder.build_objects_only(self._world)
        self._world._xml = xml
        self._world._model = mj.MjModel.from_xml_string(xml)
        self._world._data = mj.MjData(self._world._model)
        self._world.status = SimStatus.IDLE

    def _recompile_world(self) -> Dict[str, Any]:
        try:
            self._compile_world()
            return {"status": "success"}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Recompile failed: {e}"}]}

    # --- Robot Management ---

    @staticmethod
    def _ensure_meshes(model_path: str, robot_name: str):
        """Check if mesh files referenced by a model XML exist; auto-download if missing."""
        model_dir = os.path.dirname(os.path.abspath(model_path))

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

        logger.info("Downloading mesh files for '%s' from MuJoCo Menagerie (first time only)...", robot_name)
        try:
            from strands_robots.assets import resolve_robot_name
            from strands_robots.assets.download import download_robots

            canonical = resolve_robot_name(robot_name)
            download_robots(names=[canonical], force=True)
        except Exception as e:
            logger.warning("Auto-download failed for '%s': %s", robot_name, e)

    def add_robot(
        self,
        name: str,
        urdf_path: str = None,
        data_config: str = None,
        position: List[float] = None,
        orientation: List[float] = None,
    ) -> Dict[str, Any]:
        """Add a robot to the simulation."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world. Use action='create_world' first."}]}
        if name in self._world.robots:
            return {"status": "error", "content": [{"text": f"❌ Robot '{name}' already exists."}]}

        resolved_path = urdf_path
        if not resolved_path and data_config:
            resolved_path = resolve_model(data_config)
            if not resolved_path:
                return {"status": "error", "content": [{"text": f"❌ No model found for '{data_config}'.\n💡 Use action='list_urdfs' to see available robots"}]}
        elif not resolved_path and name:
            resolved_path = resolve_model(name)

        if not resolved_path:
            return {"status": "error", "content": [{"text": "❌ Either urdf_path or data_config is required."}]}
        if not os.path.exists(resolved_path):
            return {"status": "error", "content": [{"text": f"❌ File not found: {resolved_path}"}]}

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
            self._ensure_meshes(resolved_path, data_config or name)

            model = mj.MjModel.from_xml_path(str(resolved_path))
            data = mj.MjData(model)

            joint_names = []
            for i in range(model.njnt):
                jnt_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
                if jnt_name:
                    joint_names.append(jnt_name)
                    robot.joint_ids.append(i)
            robot.joint_names = joint_names

            for i in range(model.nu):
                act_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
                if act_name:
                    jnt_id = model.actuator_trnid[i, 0]
                    if jnt_id in robot.joint_ids:
                        robot.actuator_ids.append(i)
                else:
                    robot.actuator_ids.append(i)
            if not robot.actuator_ids:
                for i in range(model.nu):
                    robot.actuator_ids.append(i)

            for i in range(model.ncam):
                cam_name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
                if cam_name and cam_name not in self._world.cameras:
                    self._world.cameras[cam_name] = SimCamera(
                        name=cam_name, camera_id=i,
                        width=self.default_width, height=self.default_height,
                    )

            self._world._model = model
            self._world._data = data
            self._world._robot_base_xml = resolved_path
            self._world.robots[name] = robot

            for _ in range(100):
                mj.mj_step(model, data)

            source = f"data_config='{data_config}'" if data_config else os.path.basename(resolved_path)
            return {
                "status": "success",
                "content": [{"text": (
                    f"🤖 Robot '{name}' added to simulation\n"
                    f"📁 Source: {source} → {os.path.basename(resolved_path)}\n"
                    f"📍 Position: {robot.position}\n"
                    f"🔩 Joints: {len(robot.joint_names)} ({', '.join(robot.joint_names[:8])}{'...' if len(robot.joint_names) > 8 else ''})\n"
                    f"⚡ Actuators: {len(robot.actuator_ids)}\n"
                    f"📷 Cameras: {list(self._world.cameras.keys())}\n"
                    f"💡 Run policy: action='run_policy', robot_name='{name}'"
                )}],
            }
        except Exception as e:
            logger.error("Failed to add robot '%s': %s", name, e)
            return {"status": "error", "content": [{"text": f"❌ Failed to load: {e}"}]}

    def remove_robot(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"❌ Robot '{name}' not found."}]}
        if name in self._policy_threads:
            self._world.robots[name].policy_running = False
            try:
                self._policy_threads[name].result(timeout=5.0)
            except Exception:
                pass
            del self._policy_threads[name]
        del self._world.robots[name]
        return {"status": "success", "content": [{"text": f"🗑️ Robot '{name}' removed."}]}

    def list_robots(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if not self._world.robots:
            return {"status": "success", "content": [{"text": "No robots. Use action='add_robot'."}]}

        lines = ["🤖 Robots in simulation:\n"]
        for name, robot in self._world.robots.items():
            status = "🟢 running" if robot.policy_running else "⚪ idle"
            lines.append(
                f"  • {name} ({os.path.basename(robot.urdf_path)})\n"
                f"    Position: {robot.position}, Joints: {len(robot.joint_names)}, "
                f"Config: {robot.data_config or 'direct'}, Status: {status}"
            )
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def get_robot_state(self, robot_name: str) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation running."}]}
        if robot_name not in self._world.robots:
            return {"status": "error", "content": [{"text": f"❌ Robot '{robot_name}' not found."}]}

        mj = _ensure_mujoco()
        robot = self._world.robots[robot_name]
        model, data = self._world._model, self._world._data

        state = {}
        for jnt_name in robot.joint_names:
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
            if jnt_id >= 0:
                state[jnt_name] = {
                    "position": float(data.qpos[model.jnt_qposadr[jnt_id]]),
                    "velocity": float(data.qvel[model.jnt_dofadr[jnt_id]]),
                }

        text = f"🤖 '{robot_name}' state (t={self._world.sim_time:.3f}s):\n"
        for jnt, vals in state.items():
            text += f"  {jnt}: pos={vals['position']:.4f}, vel={vals['velocity']:.4f}\n"

        return {"status": "success", "content": [{"text": text}, {"json": {"state": state}}]}

    # --- Object Management ---

    def add_object(self, name: str, shape: str = "box", position: List[float] = None,
                   orientation: List[float] = None, size: List[float] = None,
                   color: List[float] = None, mass: float = 0.1,
                   is_static: bool = False, mesh_path: str = None) -> Dict[str, Any]:
        """Add an object to the simulation."""
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if name in self._world.objects:
            return {"status": "error", "content": [{"text": f"❌ Object '{name}' exists."}]}

        obj = SimObject(
            name=name, shape=shape,
            position=position or [0.0, 0.0, 0.0],
            orientation=orientation or [1.0, 0.0, 0.0, 0.0],
            size=size or [0.05, 0.05, 0.05],
            color=color or [0.5, 0.5, 0.5, 1.0],
            mass=mass, mesh_path=mesh_path, is_static=is_static,
        )
        self._world.objects[name] = obj

        if self._world.robots:
            try:
                result = inject_object_into_scene(self._world, obj)
                if result:
                    return {"status": "success", "content": [{"text": f"📦 '{name}' spawned: {shape} at {obj.position}"}]}
                return {"status": "success", "content": [{"text": (
                    f"📦 '{name}' registered: {shape} at {obj.position}\n"
                    "⚠️ Robot scene loaded — object is tracked but not physically spawned."
                )}]}
            except Exception as e:
                logger.warning("Object injection failed, tracking metadata only: %s", e)
                return {"status": "success", "content": [{"text": f"📦 '{name}' registered: {shape} at {obj.position}\n⚠️ Injection failed ({e})"}]}

        result = self._recompile_world()
        if result["status"] == "error":
            del self._world.objects[name]
            return result

        return {"status": "success", "content": [{"text": f"📦 '{name}' added: {shape} at {obj.position}, size={obj.size}, {'static' if is_static else f'{mass}kg'}"}]}

    def remove_object(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"❌ Object '{name}' not found."}]}
        del self._world.objects[name]
        if self._world.robots:
            eject_body_from_scene(self._world, name)
        else:
            self._recompile_world()
        return {"status": "success", "content": [{"text": f"🗑️ '{name}' removed."}]}

    def move_object(self, name: str, position: List[float] = None, orientation: List[float] = None) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        if name not in self._world.objects:
            return {"status": "error", "content": [{"text": f"❌ '{name}' not found."}]}

        mj = _ensure_mujoco()
        model, data = self._world._model, self._world._data

        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        if jnt_id >= 0:
            qpos_addr = model.jnt_qposadr[jnt_id]
            if position:
                data.qpos[qpos_addr : qpos_addr + 3] = position
                self._world.objects[name].position = position
            if orientation:
                data.qpos[qpos_addr + 3 : qpos_addr + 7] = orientation
                self._world.objects[name].orientation = orientation
            mj.mj_forward(model, data)

        return {"status": "success", "content": [{"text": f"📍 '{name}' moved to {position or 'same'}"}]}

    def list_objects(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if not self._world.objects:
            return {"status": "success", "content": [{"text": "No objects."}]}

        lines = ["📦 Objects:\n"]
        for name, obj in self._world.objects.items():
            lines.append(f"  • {name}: {obj.shape} at {obj.position}, {'static' if obj.is_static else f'{obj.mass}kg'}")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    # --- Camera Management ---

    def add_camera(self, name: str, position: List[float] = None, target: List[float] = None,
                   fov: float = 60.0, width: int = 640, height: int = 480) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}

        cam = SimCamera(
            name=name, position=position or [1.0, 1.0, 1.0],
            target=target or [0.0, 0.0, 0.0], fov=fov, width=width, height=height,
        )
        self._world.cameras[name] = cam

        if self._world.robots and self._world._model is not None:
            try:
                inject_camera_into_scene(self._world, cam)
            except Exception as e:
                logger.warning(f"Camera injection failed: {e}. Camera tracked as metadata only.")
        else:
            self._recompile_world()

        return {"status": "success", "content": [{"text": f"📷 Camera '{name}' added at {cam.position}"}]}

    def remove_camera(self, name: str) -> Dict[str, Any]:
        if self._world is None or name not in self._world.cameras:
            return {"status": "error", "content": [{"text": f"❌ Camera '{name}' not found."}]}
        del self._world.cameras[name]
        return {"status": "success", "content": [{"text": f"🗑️ Camera '{name}' removed."}]}

    # --- Simulation Control ---

    def step(self, n_steps: int = 1) -> Dict[str, Any]:
        if self._world is None or self._world._data is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}
        mj = _ensure_mujoco()
        for _ in range(n_steps):
            mj.mj_step(self._world._model, self._world._data)
        self._world.sim_time = self._world._data.time
        self._world.step_count += n_steps
        return {"status": "success", "content": [{"text": f"⏩ +{n_steps} steps | t={self._world.sim_time:.4f}s | total={self._world.step_count}"}]}

    def reset(self) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        mj = _ensure_mujoco()
        mj.mj_resetData(self._world._model, self._world._data)
        self._world.sim_time = 0.0
        self._world.step_count = 0
        for r in self._world.robots.values():
            r.policy_running = False
            r.policy_steps = 0
        return {"status": "success", "content": [{"text": "🔄 Reset to initial state."}]}

    def get_state(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        lines = [
            "🌍 Simulation State",
            f"🕐 t={self._world.sim_time:.4f}s (step {self._world.step_count})",
            f"⚙️ dt={self._world.timestep}s | 🌐 g={self._world.gravity}",
            f"🤖 Robots: {len(self._world.robots)} | 📦 Objects: {len(self._world.objects)} | 📷 Cameras: {len(self._world.cameras)}",
        ]
        if self._world._model:
            lines.append(f"🦴 Bodies: {self._world._model.nbody} | 🔩 Joints: {self._world._model.njnt} | ⚡ Actuators: {self._world._model.nu}")
        if self._world._recording:
            lines.append(f"🔴 Recording: {len(self._world._trajectory)} steps")
        return {"status": "success", "content": [{"text": "\n".join(lines)}]}

    def destroy(self) -> Dict[str, Any]:
        if self._world is None:
            return {"status": "success", "content": [{"text": "No world to destroy."}]}
        for r in self._world.robots.values():
            r.policy_running = False
        self._close_viewer()
        self._world = None
        return {"status": "success", "content": [{"text": "🗑️ World destroyed."}]}

    def set_gravity(self, gravity) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        if isinstance(gravity, (int, float)):
            gravity = [0.0, 0.0, float(gravity)]
        self._world._model.opt.gravity[:] = gravity
        self._world.gravity = gravity
        return {"status": "success", "content": [{"text": f"🌐 Gravity: {gravity}"}]}

    def set_timestep(self, timestep: float) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No world."}]}
        self._world._model.opt.timestep = timestep
        self._world.timestep = timestep
        return {"status": "success", "content": [{"text": f"⏱️ Timestep: {timestep}s ({1 / timestep:.0f}Hz)"}]}

    # --- Viewer ---

    def open_viewer(self) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation to view."}]}
        from strands_robots.simulation._backend import _mujoco_viewer
        if _mujoco_viewer is None:
            return {"status": "error", "content": [{"text": "❌ mujoco.viewer not available."}]}
        if self._viewer_handle is not None:
            return {"status": "success", "content": [{"text": "👁️ Viewer already open."}]}
        try:
            self._viewer_handle = _mujoco_viewer.launch_passive(self._world._model, self._world._data)
            return {"status": "success", "content": [{"text": "👁️ Interactive viewer opened."}]}
        except Exception as e:
            return {"status": "error", "content": [{"text": f"❌ Viewer failed: {e}"}]}

    def _close_viewer(self):
        if self._viewer_handle is not None:
            try:
                self._viewer_handle.close()
            except Exception:
                pass
            self._viewer_handle = None

    def close_viewer(self) -> Dict[str, Any]:
        self._close_viewer()
        return {"status": "success", "content": [{"text": "👁️ Viewer closed."}]}

    # --- URDF Registry ---

    def list_urdfs_action(self) -> Dict[str, Any]:
        return {"status": "success", "content": [{"text": list_available_models()}]}

    def register_urdf_action(self, data_config: str, urdf_path: str) -> Dict[str, Any]:
        register_urdf(data_config, urdf_path)
        resolved = resolve_model(data_config)
        return {"status": "success", "content": [{"text": f"📋 Registered '{data_config}' → {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"}]}

    # --- Introspection ---

    def get_features(self) -> Dict[str, Any]:
        if self._world is None or self._world._model is None:
            return {"status": "error", "content": [{"text": "❌ No simulation."}]}

        mj = _ensure_mujoco()
        model = self._world._model

        joint_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        joint_names = [n for n in joint_names if n]
        actuator_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)]
        actuator_names = [n for n in actuator_names if n]
        camera_names = [mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i) for i in range(model.ncam)]
        camera_names = [n for n in camera_names if n]

        robots_info = {}
        for rname, robot in self._world.robots.items():
            robots_info[rname] = {
                "joint_names": robot.joint_names,
                "n_joints": len(robot.joint_names),
                "n_actuators": len(robot.actuator_ids),
                "data_config": robot.data_config,
                "source": os.path.basename(robot.urdf_path),
            }

        features = {
            "n_bodies": model.nbody, "n_joints": model.njnt, "n_actuators": model.nu,
            "n_cameras": model.ncam, "timestep": model.opt.timestep,
            "joint_names": joint_names, "actuator_names": actuator_names,
            "camera_names": camera_names, "robots": robots_info,
        }

        lines = [
            "🔍 Simulation Features",
            f"🦴 Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
            f"⚡ Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
            f"📷 Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
            f"⏱️ Timestep: {model.opt.timestep}s ({1 / model.opt.timestep:.0f}Hz)",
        ]
        for rname, rinfo in robots_info.items():
            lines.append(f"🤖 {rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})")

        return {"status": "success", "content": [{"text": "\n".join(lines)}, {"json": {"features": features}}]}

    # --- AgentTool Interface ---

    @property
    def tool_name(self) -> str:
        return self.tool_name_str

    @property
    def tool_type(self) -> str:
        return "simulation"

    @property
    def tool_spec(self) -> ToolSpec:
        with open(_TOOL_SPEC_PATH) as f:
            schema = json.load(f)
        return {
            "name": self.tool_name_str,
            "description": (
                "Programmatic MuJoCo simulation environment. Create worlds, add robots from URDF "
                "(direct path or auto-resolve from data_config name), add objects, run VLA policies, "
                "render cameras, record trajectories, domain randomize. "
                "Same Policy ABC as real robot control — sim ↔ real with zero code changes. "
                "Actions: create_world, load_scene, reset, get_state, destroy, "
                "add_robot, remove_robot, list_robots, get_robot_state, "
                "add_object, remove_object, move_object, list_objects, "
                "add_camera, remove_camera, "
                "run_policy, start_policy, stop_policy, "
                "render, render_depth, get_contacts, "
                "step, set_gravity, set_timestep, "
                "randomize, "
                "start_recording, stop_recording, get_recording_status, "
                "open_viewer, close_viewer, "
                "list_urdfs, register_urdf, get_features"
            ),
            "inputSchema": {"json": schema},
        }

    async def stream(self, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any) -> AsyncGenerator[ToolResultEvent, None]:
        try:
            tool_use_id = tool_use.get("toolUseId", "")
            input_data = tool_use.get("input", {})
            result = self._dispatch_action(input_data.get("action", ""), input_data)
            yield ToolResultEvent({"toolUseId": tool_use_id, **result})
        except Exception as e:
            yield ToolResultEvent({
                "toolUseId": tool_use.get("toolUseId", ""),
                "status": "error",
                "content": [{"text": f"❌ Sim error: {e}"}],
            })

    def _dispatch_action(self, action: str, d: Dict[str, Any]) -> Dict[str, Any]:
        """Route action string to method via getattr.

        Replaces the old 350-line dispatch table + 35 _do_* handlers.
        Method names match action names directly (with a few aliases).
        """
        # Aliases for actions whose method names differ
        _ALIASES = {
            "list_urdfs": "list_urdfs_action",
            "register_urdf": "register_urdf_action",
            "stop_policy": "_stop_policy",
        }

        method_name = _ALIASES.get(action, action)
        method = getattr(self, method_name, None)

        if method is None or action.startswith("_"):
            return {"status": "error", "content": [{"text": f"❌ Unknown action: {action}"}]}

        # Build kwargs from input dict, excluding 'action' itself
        # Each method uses its own signature — extra keys are harmless with **kwargs
        import inspect

        sig = inspect.signature(method)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == "self":
                continue
            # Handle name/robot_name ambiguity in the input schema
            if param_name == "name" and "name" not in d and "robot_name" in d:
                kwargs["name"] = d["robot_name"]
            elif param_name == "robot_name" and "robot_name" not in d and "name" in d:
                kwargs["robot_name"] = d["name"]
            elif param_name in d:
                kwargs[param_name] = d[param_name]
            # Forward policy kwargs
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                for k in ("policy_port", "policy_host", "model_path", "server_address",
                           "policy_type", "pretrained_name_or_path", "device"):
                    if k in d:
                        kwargs[k] = d[k]

        return method(**kwargs)

    def _stop_policy(self, robot_name: str = "", **kwargs) -> Dict[str, Any]:
        if self._world and robot_name in self._world.robots:
            self._world.robots[robot_name].policy_running = False
            return {"status": "success", "content": [{"text": f"🛑 Stopped on '{robot_name}'"}]}
        return {"status": "error", "content": [{"text": f"❌ '{robot_name}' not found."}]}

    # --- Cleanup ---

    def cleanup(self):
        if hasattr(self, "mesh") and self.mesh:
            self.mesh.stop()
        if self._world:
            for r in self._world.robots.values():
                r.policy_running = False
            self._world = None
        self._close_viewer()
        for renderer in getattr(self, "_renderers", {}).values():
            try:
                renderer.close()
            except Exception:
                pass
        self._renderers.clear()
        self._executor.shutdown(wait=False)
        self._shutdown_event.set()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.cleanup()

    def __del__(self):
        try:
            self.cleanup()
        except Exception:
            pass
