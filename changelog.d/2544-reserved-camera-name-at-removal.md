### Fixed

- **simulation**: `remove_camera` on the MuJoCo and Newton backends now refuses a reserved
  free-view camera name (`default`, `free`) on the same shared domain their `add_camera` already
  consults. Removing the `default` camera `create_world` bakes in used to succeed, after which the
  name could not be re-added (creation refuses it as reserved) and `start_recording(cameras=["default"])`
  reported it as an unknown camera, while `list_cameras()` and `describe()` still advertised it.
