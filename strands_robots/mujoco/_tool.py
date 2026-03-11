"""AgentTool interface — tool spec, stream, dispatch table, introspection."""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, AsyncGenerator, Callable, Dict

from strands.types._events import ToolResultEvent
from strands.types.tools import ToolSpec, ToolUse

from ._registry import (
    _ensure_mujoco,
    list_available_models,
    register_urdf,
    resolve_model,
)

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Introspection
# -------------------------------------------------------------------


def get_features(sim: MujocoBackend) -> Dict[str, Any]:
    """Return simulation introspection: robot joints, actuators, cameras."""
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}

    mj = _ensure_mujoco()
    model = sim._world._model

    features: Dict[str, Any] = {
        "n_bodies": model.nbody,
        "n_joints": model.njnt,
        "n_actuators": model.nu,
        "n_cameras": model.ncam,
        "timestep": model.opt.timestep,
    }

    joint_names = []
    for i in range(model.njnt):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_JOINT, i)
        if name:
            joint_names.append(name)
    features["joint_names"] = joint_names

    actuator_names = []
    for i in range(model.nu):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_ACTUATOR, i)
        if name:
            actuator_names.append(name)
    features["actuator_names"] = actuator_names

    camera_names = []
    for i in range(model.ncam):
        name = mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
        if name:
            camera_names.append(name)
    features["camera_names"] = camera_names

    robots_info = {}
    for rname, robot in sim._world.robots.items():
        robots_info[rname] = {
            "joint_names": robot.joint_names,
            "n_joints": len(robot.joint_names),
            "n_actuators": len(robot.actuator_ids),
            "data_config": robot.data_config,
            "source": os.path.basename(robot.urdf_path),
        }
    features["robots"] = robots_info

    lines = [
        "🔍 Simulation Features",
        f"🦴 Joints ({model.njnt}): {', '.join(joint_names[:12])}{'...' if len(joint_names) > 12 else ''}",
        f"⚡ Actuators ({model.nu}): {', '.join(actuator_names[:12])}{'...' if len(actuator_names) > 12 else ''}",
        f"📷 Cameras ({model.ncam}): {', '.join(camera_names) if camera_names else 'none (free camera only)'}",
        f"⏱️ Timestep: {model.opt.timestep}s ({1/model.opt.timestep:.0f}Hz)",
    ]
    for rname, rinfo in robots_info.items():
        lines.append(
            f"🤖 {rname}: {rinfo['n_joints']} joints, {rinfo['n_actuators']} actuators ({rinfo['source']})"
        )

    return {
        "status": "success",
        "content": [
            {"text": "\n".join(lines)},
            {"json": {"features": features}},
        ],
    }


def list_urdfs_action(sim: MujocoBackend) -> Dict[str, Any]:
    """List all available robot models (Menagerie MJCF + custom URDF)."""
    text = list_available_models()
    return {"status": "success", "content": [{"text": text}]}


def register_urdf_action(
    sim: MujocoBackend, data_config: str, urdf_path: str
) -> Dict[str, Any]:
    """Register a URDF/MJCF for a data_config name."""
    register_urdf(data_config, urdf_path)
    resolved = resolve_model(data_config)
    return {
        "status": "success",
        "content": [
            {
                "text": f"📋 Registered '{data_config}' → {urdf_path}\nResolved: {resolved or 'NOT FOUND'}"
            }
        ],
    }


# -------------------------------------------------------------------
# Tool Spec
# -------------------------------------------------------------------


def build_tool_spec(tool_name: str) -> ToolSpec:
    return {
        "name": tool_name,
        "description": (
            "Programmatic MuJoCo simulation environment. Create worlds, add robots from URDF "
            "(direct path or auto-resolve from data_config name), add objects, run VLA policies, "
            "render cameras, record trajectories, domain randomize. "
            "Same Policy ABC as real robot control — sim ↔ real with zero code changes. "
            "10 embodiment configs pre-registered (so100, unitree_g1, panda, etc.). "
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
        "inputSchema": {
            "json": {
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "description": "Action to perform",
                        "enum": [
                            "create_world",
                            "load_scene",
                            "reset",
                            "get_state",
                            "destroy",
                            "add_robot",
                            "remove_robot",
                            "list_robots",
                            "get_robot_state",
                            "add_object",
                            "remove_object",
                            "move_object",
                            "list_objects",
                            "add_camera",
                            "remove_camera",
                            "run_policy",
                            "start_policy",
                            "stop_policy",
                            "render",
                            "render_depth",
                            "get_contacts",
                            "step",
                            "set_gravity",
                            "set_timestep",
                            "randomize",
                            "start_recording",
                            "stop_recording",
                            "get_recording_status",
                            "record_video",
                            "open_viewer",
                            "close_viewer",
                            "list_urdfs",
                            "register_urdf",
                            "get_features",
                            "replay_episode",
                            "eval_policy",
                        ],
                    },
                    "scene_path": {"type": "string", "description": "Path to MJCF/URDF scene file"},
                    "timestep": {"type": "number"},
                    "gravity": {"type": "array", "items": {"type": "number"}},
                    "ground_plane": {"type": "boolean"},
                    "urdf_path": {"type": "string", "description": "Path to URDF/MJCF file"},
                    "robot_name": {"type": "string"},
                    "data_config": {"type": "string", "description": "Data config name (auto-resolves URDF)"},
                    "name": {"type": "string", "description": "Object/camera name"},
                    "shape": {"type": "string", "enum": ["box", "sphere", "cylinder", "capsule", "mesh", "plane"]},
                    "position": {"type": "array", "items": {"type": "number"}},
                    "orientation": {"type": "array", "items": {"type": "number"}},
                    "size": {"type": "array", "items": {"type": "number"}},
                    "color": {"type": "array", "items": {"type": "number"}},
                    "mass": {"type": "number"},
                    "is_static": {"type": "boolean"},
                    "mesh_path": {"type": "string"},
                    "target": {"type": "array", "items": {"type": "number"}, "description": "Camera target point"},
                    "fov": {"type": "number", "description": "Camera field of view"},
                    "width": {"type": "integer"},
                    "height": {"type": "integer"},
                    "policy_provider": {"type": "string", "description": "Policy provider name (e.g. groot, lerobot_async, lerobot_local, dreamgen, mock)"},
                    "instruction": {"type": "string"},
                    "duration": {"type": "number"},
                    "policy_port": {"type": "integer"},
                    "policy_host": {"type": "string"},
                    "model_path": {"type": "string"},
                    "action_horizon": {"type": "integer"},
                    "control_frequency": {"type": "number"},
                    "camera_name": {"type": "string"},
                    "n_steps": {"type": "integer"},
                    "output_path": {"type": "string", "description": "Trajectory/video export path"},
                    "fps": {"type": "integer", "description": "Video frames per second (for record_video)"},
                    "pretrained_name_or_path": {"type": "string", "description": "HuggingFace model ID for lerobot_local"},
                    "randomize_colors": {"type": "boolean"},
                    "randomize_lighting": {"type": "boolean"},
                    "randomize_physics": {"type": "boolean"},
                    "randomize_positions": {"type": "boolean"},
                    "position_noise": {"type": "number"},
                    "seed": {"type": "integer", "description": "Random seed"},
                    "repo_id": {"type": "string", "description": "HuggingFace dataset repo ID (for start_recording: creates LeRobotDataset; for replay_episode: loads dataset)"},
                    "push_to_hub": {"type": "boolean", "description": "Auto-push dataset to HuggingFace Hub on stop_recording"},
                    "vcodec": {"type": "string", "description": "Video codec for dataset recording (h264, hevc, libsvtav1)"},
                    "task": {"type": "string", "description": "Task description for dataset recording"},
                    "episode": {"type": "integer", "description": "Episode index for replay_episode"},
                    "root": {"type": "string", "description": "Local dataset root directory"},
                    "speed": {"type": "number", "description": "Replay speed multiplier (1.0 = original)"},
                    "n_episodes": {"type": "integer", "description": "Number of eval episodes"},
                    "max_steps": {"type": "integer", "description": "Max steps per eval episode"},
                    "success_fn": {"type": "string", "description": "Success function ('contact')"},
                },
                "required": ["action"],
            }
        },
    }


# -------------------------------------------------------------------
# Stream
# -------------------------------------------------------------------


async def stream(
    sim: MujocoBackend, tool_use: ToolUse, invocation_state: dict[str, Any], **kwargs: Any
) -> AsyncGenerator[ToolResultEvent, None]:
    try:
        tool_use_id = tool_use.get("toolUseId", "")
        input_data = tool_use.get("input", {})
        result = dispatch_action(sim, input_data.get("action", ""), input_data)
        yield ToolResultEvent({"toolUseId": tool_use_id, **result})
    except Exception as e:
        yield ToolResultEvent(
            {
                "toolUseId": tool_use.get("toolUseId", ""),
                "status": "error",
                "content": [{"text": f"❌ Sim error: {e}"}],
            }
        )


# -------------------------------------------------------------------
# Dispatch helpers
# -------------------------------------------------------------------


def _extract_policy_kwargs(d: Dict[str, Any]) -> Dict[str, Any]:
    return {
        k: d[k]
        for k in ("policy_port", "policy_host", "model_path", "server_address", "policy_type")
        if k in d
    }


def _extract_optional(d: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    return {k: d[k] for k in keys if d.get(k) is not None}


# -------------------------------------------------------------------
# Individual dispatch handlers
# -------------------------------------------------------------------


def _do_create_world(sim, d):
    from ._scene import create_world
    kw = {"ground_plane": d.get("ground_plane", True)}
    kw.update(_extract_optional(d, "timestep", "gravity"))
    return create_world(sim, **kw)


def _do_load_scene(sim, d):
    from ._scene import load_scene
    return load_scene(sim, d.get("scene_path", ""))


def _do_reset(sim, d):
    from ._scene import reset
    return reset(sim)


def _do_get_state(sim, d):
    from ._scene import get_state
    return get_state(sim)


def _do_destroy(sim, d):
    from ._scene import destroy
    return destroy(sim)


def _do_add_robot(sim, d):
    from ._robots import add_robot
    kw = {"name": d.get("robot_name", d.get("name", "robot_0"))}
    kw.update(_extract_optional(d, "urdf_path", "data_config", "position", "orientation"))
    return add_robot(sim, **kw)


def _do_remove_robot(sim, d):
    from ._robots import remove_robot
    return remove_robot(sim, d.get("robot_name", d.get("name", "")))


def _do_list_robots(sim, d):
    from ._robots import list_robots
    return list_robots(sim)


def _do_get_robot_state(sim, d):
    from ._robots import get_robot_state
    return get_robot_state(sim, d.get("robot_name", ""))


def _do_add_object(sim, d):
    from ._objects import add_object
    return add_object(
        sim,
        name=d.get("name", ""),
        shape=d.get("shape", "box"),
        position=d.get("position"),
        orientation=d.get("orientation"),
        size=d.get("size"),
        color=d.get("color"),
        mass=d.get("mass", 0.1),
        is_static=d.get("is_static", False),
        mesh_path=d.get("mesh_path"),
    )


def _do_remove_object(sim, d):
    from ._objects import remove_object
    return remove_object(sim, d.get("name", ""))


def _do_move_object(sim, d):
    from ._objects import move_object
    return move_object(sim, d.get("name", ""), d.get("position"), d.get("orientation"))


def _do_list_objects(sim, d):
    from ._objects import list_objects
    return list_objects(sim)


def _do_add_camera(sim, d):
    from ._cameras import add_camera
    return add_camera(
        sim,
        name=d.get("name", "cam"),
        position=d.get("position"),
        target=d.get("target"),
        fov=d.get("fov", 60.0),
        width=d.get("width", 640),
        height=d.get("height", 480),
    )


def _do_remove_camera(sim, d):
    from ._cameras import remove_camera
    return remove_camera(sim, d.get("name", ""))


def _do_run_policy(sim, d):
    from ._policy import run_policy
    return run_policy(
        sim,
        robot_name=d.get("robot_name", ""),
        policy_provider=d.get("policy_provider", "mock"),
        instruction=d.get("instruction", ""),
        duration=d.get("duration", 10.0),
        action_horizon=d.get("action_horizon", 8),
        control_frequency=d.get("control_frequency", 50.0),
        fast_mode=d.get("fast_mode", False),
        **_extract_policy_kwargs(d),
    )


def _do_start_policy(sim, d):
    from ._policy import start_policy
    return start_policy(
        sim,
        robot_name=d.get("robot_name", ""),
        policy_provider=d.get("policy_provider", "mock"),
        instruction=d.get("instruction", ""),
        duration=d.get("duration", 10.0),
        fast_mode=d.get("fast_mode", False),
        **_extract_policy_kwargs(d),
    )


def _do_stop_policy(sim, d):
    rn = d.get("robot_name", "")
    if sim._world and rn in sim._world.robots:
        sim._world.robots[rn].policy_running = False
        return {"status": "success", "content": [{"text": f"🛑 Stopped on '{rn}'"}]}
    return {"status": "error", "content": [{"text": f"❌ '{rn}' not found."}]}


def _do_render(sim, d):
    from ._rendering import render
    return render(sim, d.get("camera_name", "default"), d.get("width"), d.get("height"))


def _do_render_depth(sim, d):
    from ._rendering import render_depth
    return render_depth(sim, d.get("camera_name", "default"), d.get("width"), d.get("height"))


def _do_get_contacts(sim, d):
    from ._rendering import get_contacts
    return get_contacts(sim)


def _do_step(sim, d):
    from ._scene import step
    return step(sim, d.get("n_steps", 1))


def _do_set_gravity(sim, d):
    from ._scene import set_gravity
    return set_gravity(sim, d.get("gravity", [0, 0, -9.81]))


def _do_set_timestep(sim, d):
    from ._scene import set_timestep
    return set_timestep(sim, d.get("timestep", 0.002))


def _do_randomize(sim, d):
    from ._randomization import randomize
    return randomize(
        sim,
        randomize_colors=d.get("randomize_colors", True),
        randomize_lighting=d.get("randomize_lighting", True),
        randomize_physics=d.get("randomize_physics", False),
        randomize_positions=d.get("randomize_positions", False),
        position_noise=d.get("position_noise", 0.02),
        seed=d.get("seed"),
    )


def _do_start_recording(sim, d):
    from ._recording import start_recording
    kw = {
        "task": d.get("instruction", d.get("task", "")),
        "fps": d.get("fps", 30),
        "push_to_hub": d.get("push_to_hub", False),
        "vcodec": d.get("vcodec", "libsvtav1"),
    }
    kw.update(_extract_optional(d, "repo_id", "root"))
    return start_recording(sim, **kw)


def _do_stop_recording(sim, d):
    from ._recording import stop_recording
    return stop_recording(sim, d.get("output_path"))


def _do_get_recording_status(sim, d):
    from ._recording import get_recording_status
    return get_recording_status(sim)


def _do_record_video(sim, d):
    from ._recording import record_video
    kw = {
        k: d[k]
        for k in (
            "policy_port", "policy_host", "model_path",
            "server_address", "policy_type", "pretrained_name_or_path",
        )
        if k in d
    }
    return record_video(
        sim,
        robot_name=d.get("robot_name", ""),
        policy_provider=d.get("policy_provider", "lerobot_local"),
        instruction=d.get("instruction", ""),
        duration=d.get("duration", 10.0),
        fps=d.get("fps", 30),
        camera_name=d.get("camera_name"),
        width=d.get("width", 640),
        height=d.get("height", 480),
        output_path=d.get("output_path"),
        **kw,
    )


def _do_open_viewer(sim, d):
    from ._viewer import open_viewer
    return open_viewer(sim)


def _do_close_viewer(sim, d):
    from ._viewer import close_viewer
    return close_viewer(sim)


def _do_list_urdfs(sim, d):
    return list_urdfs_action(sim)


def _do_register_urdf(sim, d):
    return register_urdf_action(sim, d.get("data_config", ""), d.get("urdf_path", ""))


def _do_get_features(sim, d):
    return get_features(sim)


def _do_replay_episode(sim, d):
    from ._recording import replay_episode
    repo_id = d.get("repo_id")
    if not repo_id:
        return {"status": "error", "content": [{"text": "❌ repo_id required for replay_episode"}]}
    return replay_episode(
        sim,
        repo_id=repo_id,
        robot_name=d.get("robot_name"),
        episode=d.get("episode", 0),
        root=d.get("root"),
        speed=d.get("speed", 1.0),
    )


def _do_eval_policy(sim, d):
    from ._recording import eval_policy
    return eval_policy(
        sim,
        robot_name=d.get("robot_name"),
        policy_provider=d.get("policy_provider", "mock"),
        instruction=d.get("instruction", ""),
        n_episodes=d.get("n_episodes", 10),
        max_steps=d.get("max_steps", 300),
        success_fn=d.get("success_fn"),
        **{
            k: v
            for k, v in d.items()
            if k.startswith("pretrained") or k in ("policy_type", "device")
        },
    )


# -------------------------------------------------------------------
# Dispatch table
# -------------------------------------------------------------------

ACTION_DISPATCH: Dict[str, Callable] = {
    "create_world": _do_create_world,
    "load_scene": _do_load_scene,
    "reset": _do_reset,
    "get_state": _do_get_state,
    "destroy": _do_destroy,
    "add_robot": _do_add_robot,
    "remove_robot": _do_remove_robot,
    "list_robots": _do_list_robots,
    "get_robot_state": _do_get_robot_state,
    "add_object": _do_add_object,
    "remove_object": _do_remove_object,
    "move_object": _do_move_object,
    "list_objects": _do_list_objects,
    "add_camera": _do_add_camera,
    "remove_camera": _do_remove_camera,
    "run_policy": _do_run_policy,
    "start_policy": _do_start_policy,
    "stop_policy": _do_stop_policy,
    "render": _do_render,
    "render_depth": _do_render_depth,
    "get_contacts": _do_get_contacts,
    "step": _do_step,
    "set_gravity": _do_set_gravity,
    "set_timestep": _do_set_timestep,
    "randomize": _do_randomize,
    "start_recording": _do_start_recording,
    "stop_recording": _do_stop_recording,
    "get_recording_status": _do_get_recording_status,
    "record_video": _do_record_video,
    "open_viewer": _do_open_viewer,
    "close_viewer": _do_close_viewer,
    "list_urdfs": _do_list_urdfs,
    "register_urdf": _do_register_urdf,
    "get_features": _do_get_features,
    "replay_episode": _do_replay_episode,
    "eval_policy": _do_eval_policy,
}


def dispatch_action(sim: MujocoBackend, action: str, d: Dict[str, Any]) -> Dict[str, Any]:
    """Route action to handler via dispatch table."""
    handler = ACTION_DISPATCH.get(action)
    if handler is None:
        return {
            "status": "error",
            "content": [{"text": f"❌ Unknown action: {action}"}],
        }
    return handler(sim, d)
