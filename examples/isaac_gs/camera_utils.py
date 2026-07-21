"""Pull pinhole camera params + RGB/depth from an Isaac Sim RTX camera.

The 3DGS / panorama background renderers in
``examples.mujoco_gs.backgrounds`` need a pinhole ``CameraParams``
(intrinsic ``K``, world-from-camera ``T_world_cam``, resolution) to
rasterise a backdrop aligned to the foreground camera. ``mujoco_gs``
pulls these from MuJoCo's ``mj_data``; this module pulls the
equivalent from the ``omni.isaac.sensor.Camera`` handle that
``IsaacSimulation.add_camera`` (#61) attaches to
``sim._cameras[name].handle``:

* ``camera.get_intrinsics_matrix()`` -> ``K``
* ``camera.get_world_pose()`` -> ``(position, quaternion)`` -> ``T_world_cam``
* ``sim.render(camera_name)`` (#62) -> RGB + metric depth

Isaac's USD camera convention matches the OpenGL convention the
background renderers expect (+X right, +Y up, **-Z forward**), so the
pose maps across without an axis flip.
"""

from __future__ import annotations

import numpy as np

# Reuse the backend-agnostic CameraParams dataclass from the MuJoCo
# example so the shared background renderers accept our params
# unchanged. It's a pure dataclass (K / T_world_cam / width / height /
# znear / zfar) with no MuJoCo import at definition time.
from examples.mujoco_gs.camera_utils import CameraParams as IsaacCameraParams


def _quat_wxyz_to_rotmat(quat: np.ndarray) -> np.ndarray:
    """Convert a ``(w, x, y, z)`` quaternion to a ``(3, 3)`` rotation matrix.

    Isaac's ``Camera.get_world_pose()`` returns the orientation as a
    ``(w, x, y, z)`` quaternion (USD convention). No external dep
    needed -- the standard quaternion-to-matrix formula.
    """
    w, x, y, z = (float(quat[0]), float(quat[1]), float(quat[2]), float(quat[3]))
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    s = 2.0 / n
    wx, wy, wz = s * w * x, s * w * y, s * w * z
    xx, xy, xz = s * x * x, s * x * y, s * x * z
    yy, yz, zz = s * y * y, s * y * z, s * z * z
    return np.array(
        [
            [1.0 - (yy + zz), xy - wz, xz + wy],
            [xy + wz, 1.0 - (xx + zz), yz - wx],
            [xz - wy, yz + wx, 1.0 - (xx + yy)],
        ],
        dtype=np.float64,
    )


def get_camera_params(
    sim: "object",
    camera_name: str,
    znear: float = 0.01,
    zfar: float = 1_000_000.0,
) -> IsaacCameraParams:
    """Build :class:`IsaacCameraParams` for a camera added via ``add_camera``.

    Parameters
    ----------
    sim : IsaacSimulation
        A live sim with a world created and ``camera_name`` added via
        :meth:`IsaacSimulation.add_camera`.
    camera_name : str
        Camera identifier.
    znear, zfar : float
        Near / far planes (meters) carried on the params for the
        compositor's "background is at infinity" depth convention.

    Returns
    -------
    IsaacCameraParams
        ``K`` (3x3 px), ``T_world_cam`` (4x4 SE3), width, height.

    Raises
    ------
    KeyError
        If ``camera_name`` was never added.
    RuntimeError
        If the camera has no live handle (Phase-2 ``add_camera`` /
        PR #61 not present) -- intrinsics can't be read off a Phase-1
        stub camera.
    """
    if camera_name not in sim._cameras:
        raise KeyError(f"camera {camera_name!r} not found; call sim.add_camera({camera_name!r}, ...) first")
    cam_state = sim._cameras[camera_name]
    handle = cam_state.handle
    if handle is None:
        raise RuntimeError(
            f"camera {camera_name!r} has no live Camera handle -- the Phase-2 add_camera "
            "wiring (PR #61) is required to read intrinsics / pose. On a stock build "
            "the camera is a registration-only stub."
        )

    K = np.asarray(handle.get_intrinsics_matrix(), dtype=np.float64).reshape(3, 3)
    position, quat_wxyz = handle.get_world_pose()
    position = np.asarray(position, dtype=np.float64).reshape(3)
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64).reshape(4)

    # ``Camera.get_world_pose()`` returns the camera *prim's* world orientation,
    # whose local axes are offset from the OpenGL optical frame this module's
    # ``CameraParams`` promises (+X right, +Y up, -Z forward). ``add_camera``
    # aims the physical camera correctly via ``set_camera_view`` (so the RTX
    # foreground is right), but feeding the raw prim rotation to a renderer that
    # assumes the GL convention rolls/misaims the backdrop. Apply the fixed
    # camera-local correction prim->GL (empirically, and consistent across
    # poses: prim +X -> GL -Z, prim +Y -> GL -X, prim +Z -> GL +Y) so a
    # composited 3DGS/panorama background is upright and aligned with the
    # foreground. ``R_gl = R_prim @ PRIM_TO_GL``.
    PRIM_TO_GL = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = _quat_wxyz_to_rotmat(quat_wxyz) @ PRIM_TO_GL
    T[:3, 3] = position

    return IsaacCameraParams(
        K=K,
        T_world_cam=T,
        width=int(cam_state.width),
        height=int(cam_state.height),
        znear=float(znear),
        zfar=float(zfar),
    )


def render_rgb_and_depth(sim: "object", camera_name: str) -> "tuple[np.ndarray, np.ndarray]":
    """Render the Isaac RTX foreground RGB + metric depth for a camera.

    Thin wrapper over ``sim.render(camera_name)`` (#62) that normalises
    the envelope into ``(rgb_uint8_HxWx3, depth_float32_HxW)``.

    Pixels with no geometry (sky / background) come back from Isaac as
    very large / inf depth; the compositor treats those as "see the
    background through here".
    """
    result = sim.render(camera_name=camera_name)
    if result.get("status") != "success":
        raise RuntimeError(f"sim.render({camera_name!r}) failed: {result}")
    rgb = np.asarray(result["rgb"])
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    depth = np.asarray(result["depth"], dtype=np.float32)
    return rgb.astype(np.uint8), depth
