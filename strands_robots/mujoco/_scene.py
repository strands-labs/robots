"""Scene lifecycle — create, load, compile, reset, destroy, step, gravity, timestep."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Dict, List

from ._builder import MJCFBuilder
from ._registry import _ensure_mujoco
from ._types import SimCamera, SimStatus, SimWorld

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def create_world(
    sim: MujocoBackend,
    timestep: float = None,
    gravity: List[float] = None,
    ground_plane: bool = True,
) -> Dict[str, Any]:
    """Create a new simulation world."""
    _ensure_mujoco()

    if sim._world is not None and sim._world._model is not None:
        return {
            "status": "error",
            "content": [
                {
                    "text": "❌ World already exists. Use action='destroy' first, or action='reset'."
                }
            ],
        }

    # Normalize gravity: accept scalar (e.g. -9.81) or 3-element list
    if gravity is None:
        _gravity = [0.0, 0.0, -9.81]
    elif isinstance(gravity, (int, float)):
        _gravity = [0.0, 0.0, float(gravity)]
    else:
        _gravity = list(gravity)

    sim._world = SimWorld(
        timestep=timestep or sim.default_timestep,
        gravity=_gravity,
        ground_plane=ground_plane,
    )

    # Default camera
    sim._world.cameras["default"] = SimCamera(
        name="default",
        position=[1.5, 1.5, 1.2],
        target=[0.0, 0.0, 0.3],
        width=sim.default_width,
        height=sim.default_height,
    )

    # Build initial empty scene
    compile_world(sim)

    from ._registry import _URDF_REGISTRY

    try:
        from strands_robots.assets import list_available_robots as _list_menagerie_robots
        n_models = len(_list_menagerie_robots())
    except ImportError:
        n_models = len(_URDF_REGISTRY)

    logger.info("🌍 Simulation world created")
    return {
        "status": "success",
        "content": [
            {
                "text": (
                    "🌍 Simulation world created\n"
                    f"⚙️ Timestep: {sim._world.timestep}s ({1/sim._world.timestep:.0f}Hz physics)\n"
                    f"🌐 Gravity: {sim._world.gravity}\n"
                    f"📷 Default camera ready\n"
                    f"🤖 Robot models: {n_models} available\n"
                    "💡 Add robots: action='add_robot' (urdf_path or data_config)\n"
                    "💡 Add objects: action='add_object'\n"
                    "💡 List URDFs: action='list_urdfs'"
                )
            }
        ],
    }


def load_scene(sim: MujocoBackend, scene_path: str) -> Dict[str, Any]:
    """Load a complete scene from MJCF XML or URDF file."""
    mj = _ensure_mujoco()

    if not os.path.exists(scene_path):
        return {
            "status": "error",
            "content": [{"text": f"❌ Scene file not found: {scene_path}"}],
        }

    try:
        sim._world = SimWorld()
        sim._world._model = mj.MjModel.from_xml_path(str(scene_path))
        sim._world._data = mj.MjData(sim._world._model)
        sim._world.status = SimStatus.IDLE

        n_bodies = sim._world._model.nbody
        n_joints = sim._world._model.njnt
        n_actuators = sim._world._model.nu

        logger.info("🌍 Scene loaded: %s", scene_path)
        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"🌍 Scene loaded from {os.path.basename(scene_path)}\n"
                        f"🦴 Bodies: {n_bodies}, 🔩 Joints: {n_joints}, ⚡ Actuators: {n_actuators}\n"
                        "💡 Use action='get_state' to inspect, action='step' to simulate"
                    )
                }
            ],
        }
    except Exception as e:
        logger.error("Failed to load scene: %s", e)
        return {
            "status": "error",
            "content": [{"text": f"❌ Failed to load scene: {e}"}],
        }


def compile_world(sim: MujocoBackend):
    """Compile current world state into a MuJoCo model."""
    mj = _ensure_mujoco()
    try:
        xml = MJCFBuilder.build_objects_only(sim._world)
        sim._world._xml = xml
        sim._world._model = mj.MjModel.from_xml_string(xml)
        sim._world._data = mj.MjData(sim._world._model)
        sim._world.status = SimStatus.IDLE
    except Exception as e:
        logger.error("World compilation failed: %s", e)
        raise


def recompile_world(sim: MujocoBackend) -> Dict[str, Any]:
    """Recompile world after structural changes."""
    try:
        compile_world(sim)
        return {"status": "success"}
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Recompile failed: {e}"}],
        }


def step(sim: MujocoBackend, n_steps: int = 1) -> Dict[str, Any]:
    if sim._world is None or sim._world._data is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}

    mj = _ensure_mujoco()
    for _ in range(n_steps):
        mj.mj_step(sim._world._model, sim._world._data)

    sim._world.sim_time = sim._world._data.time
    sim._world.step_count += n_steps

    return {
        "status": "success",
        "content": [
            {
                "text": f"⏩ +{n_steps} steps | t={sim._world.sim_time:.4f}s | total={sim._world.step_count}"
            }
        ],
    }


def reset(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}

    mj = _ensure_mujoco()
    mj.mj_resetData(sim._world._model, sim._world._data)
    sim._world.sim_time = 0.0
    sim._world.step_count = 0
    for r in sim._world.robots.values():
        r.policy_running = False
        r.policy_steps = 0

    return {
        "status": "success",
        "content": [{"text": "🔄 Reset to initial state."}],
    }


def get_state(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}

    lines = [
        "🌍 Simulation State",
        f"🕐 t={sim._world.sim_time:.4f}s (step {sim._world.step_count})",
        f"⚙️ dt={sim._world.timestep}s | 🌐 g={sim._world.gravity}",
        f"🤖 Robots: {len(sim._world.robots)} | 📦 Objects: {len(sim._world.objects)} | 📷 Cameras: {len(sim._world.cameras)}",
    ]
    if sim._world._model:
        lines.append(
            f"🦴 Bodies: {sim._world._model.nbody} | 🔩 Joints: {sim._world._model.njnt} | ⚡ Actuators: {sim._world._model.nu}"
        )
    if sim._world._recording:
        lines.append(f"🔴 Recording: {len(sim._world._trajectory)} steps")

    return {"status": "success", "content": [{"text": "\n".join(lines)}]}


def destroy(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None:
        return {"status": "success", "content": [{"text": "No world to destroy."}]}
    for r in sim._world.robots.values():
        r.policy_running = False
    from ._viewer import close_viewer_internal
    close_viewer_internal(sim)
    sim._world = None
    return {"status": "success", "content": [{"text": "🗑️ World destroyed."}]}


def set_gravity(sim: MujocoBackend, gravity) -> Dict[str, Any]:
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}
    if isinstance(gravity, (int, float)):
        gravity = [0.0, 0.0, float(gravity)]
    sim._world._model.opt.gravity[:] = gravity
    sim._world.gravity = gravity
    return {"status": "success", "content": [{"text": f"🌐 Gravity: {gravity}"}]}


def set_timestep(sim: MujocoBackend, timestep: float) -> Dict[str, Any]:
    """Change the physics simulation timestep.

    Args:
        timestep: New timestep in seconds (e.g. 0.002 for 500Hz)
    """
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No world."}]}
    sim._world._model.opt.timestep = timestep
    sim._world.timestep = timestep
    return {
        "status": "success",
        "content": [
            {"text": f"⏱️ Timestep: {timestep}s ({1/timestep:.0f}Hz physics)"}
        ],
    }
