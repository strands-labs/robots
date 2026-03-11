"""Rendering — renderer cache, observations, actions, render, depth, contacts."""

from __future__ import annotations

import io
import logging
from typing import TYPE_CHECKING, Any, Dict

from ._registry import _ensure_mujoco

if TYPE_CHECKING:
    from ._core import MujocoBackend

logger = logging.getLogger(__name__)


def get_renderer(sim: MujocoBackend, width: int, height: int):
    """Get a cached MuJoCo renderer, creating one only if needed."""
    mj = _ensure_mujoco()
    key = (width, height)
    # Invalidate cache if model changed (e.g. after add/remove object)
    if sim._renderer_model is not sim._world._model:
        sim._renderers.clear()
        sim._renderer_model = sim._world._model
    if key not in sim._renderers:
        sim._renderers[key] = mj.Renderer(
            sim._world._model, height=height, width=width
        )
    return sim._renderers[key]


def get_sim_observation(
    sim: MujocoBackend, robot_name: str, cam_name: str = None
) -> Dict[str, Any]:
    """Get observation from sim (same format as real robot)."""
    mj = _ensure_mujoco()
    model, data = sim._world._model, sim._world._data
    robot = sim._world.robots[robot_name]

    obs = {}
    for jnt_name in robot.joint_names:
        jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, jnt_name)
        if jnt_id >= 0:
            obs[jnt_name] = float(data.qpos[model.jnt_qposadr[jnt_id]])

    # Render camera(s) for observation
    cameras_to_render = []
    if cam_name:
        cameras_to_render = [cam_name]
    else:
        cameras_to_render = [
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_CAMERA, i)
            for i in range(model.ncam)
        ]

    for cname in cameras_to_render:
        if not cname:
            continue
        cam_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_CAMERA, cname)
        if cam_id >= 0:
            cam_info = sim._world.cameras.get(cname)
            h = cam_info.height if cam_info else sim.default_height
            w = cam_info.width if cam_info else sim.default_width
            try:
                renderer = get_renderer(sim, w, h)
                renderer.update_scene(data, camera=cam_id)
                obs[cname] = renderer.render().copy()
            except Exception as e:
                logger.debug("Camera render failed for %s: %s", cname, e)

    return obs


def apply_sim_action(
    sim: MujocoBackend,
    robot_name: str,
    action_dict: Dict[str, Any],
    n_substeps: int = 1,
):
    """Apply action dict to sim (same interface as robot.send_action)."""
    mj = _ensure_mujoco()
    model, data = sim._world._model, sim._world._data

    for key, value in action_dict.items():
        # Try actuator first (preferred)
        act_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_ACTUATOR, key)
        if act_id >= 0:
            data.ctrl[act_id] = float(value)
        else:
            # Fallback: try matching joint index to actuator index
            jnt_id = mj.mj_name2id(model, mj.mjtObj.mjOBJ_JOINT, key)
            if jnt_id >= 0 and jnt_id < model.nu:
                data.ctrl[jnt_id] = float(value)

    for _ in range(max(1, n_substeps)):
        mj.mj_step(model, data)

    sim._world.sim_time = data.time
    sim._world.step_count += n_substeps


def render(
    sim: MujocoBackend,
    camera_name: str = "default",
    width: int = None,
    height: int = None,
) -> Dict[str, Any]:
    """Render a camera view as base64 PNG image."""
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}

    mj = _ensure_mujoco()
    w = width or sim.default_width
    h = height or sim.default_height

    try:
        renderer = get_renderer(sim, w, h)

        cam_id = mj.mj_name2id(
            sim._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name
        )
        if cam_id >= 0:
            renderer.update_scene(sim._world._data, camera=cam_id)
        else:
            renderer.update_scene(sim._world._data)

        img = renderer.render().copy()

        from PIL import Image

        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="PNG")
        png_bytes = buffer.getvalue()

        return {
            "status": "success",
            "content": [
                {
                    "text": f"📸 {w}x{h} from '{camera_name}' at t={sim._world.sim_time:.3f}s"
                },
                {"image": {"format": "png", "source": {"bytes": png_bytes}}},
            ],
        }
    except Exception as e:
        return {"status": "error", "content": [{"text": f"❌ Render failed: {e}"}]}


def render_depth(
    sim: MujocoBackend,
    camera_name: str = "default",
    width: int = None,
    height: int = None,
) -> Dict[str, Any]:
    """Render depth map from a camera."""
    if sim._world is None or sim._world._model is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}

    mj = _ensure_mujoco()
    w = width or sim.default_width
    h = height or sim.default_height

    try:
        cam_id = -1
        if camera_name and camera_name != "default":
            cam_id = mj.mj_name2id(
                sim._world._model, mj.mjtObj.mjOBJ_CAMERA, camera_name
            )

        renderer = get_renderer(sim, w, h)
        if cam_id >= 0:
            renderer.update_scene(sim._world._data, camera=cam_id)
        else:
            renderer.update_scene(sim._world._data)
        renderer.enable_depth_rendering()
        depth = renderer.render()
        renderer.disable_depth_rendering()

        return {
            "status": "success",
            "content": [
                {
                    "text": (
                        f"📸 Depth {w}x{h} from '{camera_name}'\n"
                        f"Min: {float(depth.min()):.3f}m, Max: {float(depth.max()):.3f}m"
                    )
                },
                {
                    "json": {
                        "depth_min": float(depth.min()),
                        "depth_max": float(depth.max()),
                    }
                },
            ],
        }
    except Exception as e:
        return {
            "status": "error",
            "content": [{"text": f"❌ Depth render failed: {e}"}],
        }


def get_contacts(sim: MujocoBackend) -> Dict[str, Any]:
    if sim._world is None or sim._world._data is None:
        return {"status": "error", "content": [{"text": "❌ No simulation."}]}

    mj = _ensure_mujoco()
    model, data = sim._world._model, sim._world._data

    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1 = (
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom1) or f"geom_{c.geom1}"
        )
        g2 = (
            mj.mj_id2name(model, mj.mjtObj.mjOBJ_GEOM, c.geom2) or f"geom_{c.geom2}"
        )
        contacts.append(
            {"geom1": g1, "geom2": g2, "dist": float(c.dist), "pos": c.pos.tolist()}
        )

    text = f"💥 {len(contacts)} contacts" if contacts else "No contacts."
    if contacts:
        for c in contacts[:10]:
            text += f"\n  • {c['geom1']} ↔ {c['geom2']} (d={c['dist']:.4f})"

    return {
        "status": "success",
        "content": [{"text": text}, {"json": {"contacts": contacts}}],
    }
