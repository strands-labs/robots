### Fix: camera intrinsics follow the frustum MuJoCo rasterizes

`get_camera_params` re-derived `K` for a camera declaring a physical sensor
(MJCF `sensorsize` / `focal` / `principal` / `resolution`) from `cam_intrinsic`
with a hard-coded vertical sign. MuJoCo 3.6.0 fixed swapped vertical frustum
bounds for a camera with a principal-point offset, so that sign only holds from
3.6 on: on MuJoCo 3.2 through 3.5 -- all inside the declared
`mujoco>=3.2.0,<4.0.0` range -- `cy` landed exactly as far the wrong side of the
image center, moving every unprojected pixel by twice the offset while the call
reported success. Measured on the camera already used by the test suite: 26.4 cm
of `get_world_point` error at a 1 m stand-off, down to 0.02 cm.

`K` is now read back from the view frustum MuJoCo computes for the camera
(`mjv_updateCamera` fills `mjvScene.camera[0].frustum_*`, the numbers
`mjr_render` hands `glFrustum`), so the intrinsics describe the projection the
installed build actually draws and MuJoCo's convention is no longer duplicated.
The `fovy` fallback for a sensorless camera is unchanged: `frustum_width` is
zero exactly when MuJoCo derives the horizontal extent from the viewport aspect.
