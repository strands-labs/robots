# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Camera intrinsics / extrinsics / depth utilities for the MuJoCo-GS demo.

`strands_robots.simulation.Simulation` does not expose camera intrinsics or
extrinsics, and `render_depth` only returns scalar min/max in its AgentTool
response payload. To composite a 3DGS background against a MuJoCo render we
need:

  * the camera's pinhole intrinsic matrix `K` (from MuJoCo's vertical FOV)
  * its world-from-camera pose `T_world_cam` (from `mj_data.cam_xpos / cam_xmat`)
  * a metric depth buffer `(H, W) float32` (MuJoCo gives clip-space; we
    linearise it to meters using `model.stat.extent * model.vis.map.{znear,zfar}`,
    matching the formula in upstream `strands_robots/simulation/mujoco/rendering.py`)

These helpers reach through the public `sim.mj_model` / `sim.mj_data` properties
so they keep working as long as upstream MuJoCo backend stays available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # avoid hard import at module load — sim deps are optional
    from strands_robots.simulation import Simulation


@dataclass(frozen=True)
class CameraParams:
    """Pinhole camera parameters at a given image resolution.

    Attributes:
        K: ``(3, 3)`` intrinsic matrix in pixels.
        T_world_cam: ``(4, 4)`` SE(3) pose. ``T_world_cam @ [x_cam; 1]`` gives
            the world-frame coordinates of a camera-frame point. MuJoCo's
            camera convention is OpenGL-style: +X right, +Y up, **-Z forward**.
        width: image width in pixels.
        height: image height in pixels.
        znear: near-plane distance in meters (linearised depth uses this).
        zfar: far-plane distance in meters.
    """

    K: np.ndarray
    T_world_cam: np.ndarray
    width: int
    height: int
    znear: float
    zfar: float

    @property
    def fovy_rad(self) -> float:
        """Vertical field-of-view, recovered from K and image height."""
        fy = float(self.K[1, 1])
        return float(2.0 * np.arctan(0.5 * self.height / fy))


def _import_mujoco():
    """Import mujoco lazily so the example can `import camera_utils` even if
    `strands-robots[sim-mujoco]` isn't installed yet (the README mentions the
    extra; failing late gives a clearer error)."""
    try:
        import mujoco  # type: ignore
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "MuJoCo is not installed. Run `pip install 'strands-robots[sim-mujoco]'` "
            "(see examples/mujoco_gs/README.md)."
        ) from e
    return mujoco


def get_camera_params(
    sim: Simulation,
    camera_name: str,
    width: int,
    height: int,
) -> CameraParams:
    """Return a fully populated :class:`CameraParams` for ``camera_name``.

    The camera must already be added to the world via
    ``sim.add_camera(name=camera_name, ...)``. We forward-step kinematics with
    ``mj_forward`` so freshly placed cameras have valid ``cam_xpos``/``cam_xmat``
    even if the user hasn't called ``sim.step()`` yet.

    Args:
        sim: live ``Simulation`` instance (MuJoCo backend).
        camera_name: name of an existing camera in the model.
        width: image width to compute K for.
        height: image height to compute K for.

    Returns:
        CameraParams with intrinsics, extrinsics, image size, and clip planes.

    Raises:
        ValueError: if the camera is not in ``sim.mj_model``.
    """
    mujoco = _import_mujoco()
    model = sim.mj_model
    data = sim.mj_data

    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id < 0:
        raise ValueError(f"Camera {camera_name!r} not found. Add it with sim.add_camera(name=..., ...) first.")

    # Make sure cam_xpos / cam_xmat reflect the latest qpos. Cheap enough.
    mujoco.mj_forward(model, data)

    fovy_deg = float(model.cam_fovy[cam_id])
    fy = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    fx = fy  # MuJoCo uses square pixels; intrinsic from vertical FOV is symmetric.
    cx = 0.5 * width
    cy = 0.5 * height
    K = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )

    # MuJoCo stores cam_xmat row-major as a 9-vector; columns are camera basis
    # axes expressed in world frame (x_cam, y_cam, -z_forward_cam).
    R = data.cam_xmat[cam_id].reshape(3, 3).copy()
    t = data.cam_xpos[cam_id].copy()
    T_world_cam = np.eye(4, dtype=np.float64)
    T_world_cam[:3, :3] = R
    T_world_cam[:3, 3] = t

    extent = float(model.stat.extent)
    znear = extent * float(model.vis.map.znear)
    zfar = extent * float(model.vis.map.zfar)

    return CameraParams(K=K, T_world_cam=T_world_cam, width=width, height=height, znear=znear, zfar=zfar)


def render_rgb_and_depth(
    sim: Simulation,
    camera_name: str,
    width: int,
    height: int,
    renderer=None,
    scene_option=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render a single MuJoCo frame as ``(rgb_uint8, depth_metric_float32)``.

    Reaches through ``sim.mj_model``/``sim.mj_data`` and uses a
    ``mujoco.Renderer`` to read the depth buffer directly (the AgentTool
    ``render_depth`` only exposes min/max scalars). Depth is linearised to
    meters using the standard MuJoCo formula:

        z_metric = znear * zfar / (zfar - z_clip * (zfar - znear))

    where ``znear = model.stat.extent * model.vis.map.znear`` and likewise
    for ``zfar``. Pixels at infinity (clip-space depth ≈ 1.0) are clipped to
    ``zfar``.

    Args:
        sim: live ``Simulation`` instance.
        camera_name: existing camera in the model.
        width: image width in pixels.
        height: image height in pixels.
        renderer: optional pre-built ``mujoco.Renderer`` matching
            ``(width, height)``. Reusing one across frames is dramatically
            faster than allocating a fresh GL framebuffer each call (the
            per-frame ``mujoco.Renderer(...)`` constructor dominates live
            streaming). When ``None`` a short-lived renderer is created and
            closed internally (back-compatible behaviour).

    Returns:
        ``(rgb, depth)`` where ``rgb`` is ``(H, W, 3) uint8`` and ``depth`` is
        ``(H, W) float32`` in meters.
    """
    mujoco = _import_mujoco()

    owns_renderer = renderer is None
    if owns_renderer:
        renderer = mujoco.Renderer(sim.mj_model, height=height, width=width)
    try:
        # update_scene accepts scene_option=None (uses defaults). Passing the
        # benchmark's viz option (e.g. LIBERO's) hides collision geoms / site
        # markers that otherwise paint green/blue debug patches on the robot.
        kw = {"camera": camera_name}
        if scene_option is not None:
            kw["scene_option"] = scene_option

        # RGB pass.
        renderer.update_scene(sim.mj_data, **kw)
        rgb = renderer.render().copy()

        # Depth pass.
        renderer.enable_depth_rendering()
        renderer.update_scene(sim.mj_data, **kw)
        depth_clip = renderer.render().copy()
        renderer.disable_depth_rendering()

        extent = float(sim.mj_model.stat.extent)
        znear = extent * float(sim.mj_model.vis.map.znear)
        zfar = extent * float(sim.mj_model.vis.map.zfar)
        # mujoco.Renderer returns *linearised* depth already (in MuJoCo units =
        # meters by convention) since 3.x — but we still clip to zfar to drop
        # the "infinity" sentinel that some drivers return as a huge float.
        depth = np.where(depth_clip <= 0, zfar, depth_clip).astype(np.float32)
        depth = np.clip(depth, znear, zfar)
        return rgb.astype(np.uint8), depth
    finally:
        if owns_renderer:
            renderer.close()
