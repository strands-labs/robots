### Fixed

- **simulation/mujoco**: the declared `mujoco>=3.2.0` floor admitted eight releases the
  backend cannot drive - below 3.5.0 `add_robot(urdf_path=...)` is refused with "Could not
  find decoder for resource", and below 3.3.4 `MjSpec.delete` does not exist, so no scene can
  be built at all. Every extra now declares `mujoco>=3.5.0,<4.0.0` (`[vera-sim]` also gains
  the major cap it was missing), and `_ensure_mujoco()` refuses an older installed build with
  a message naming the version, the floor and the remedy instead of failing later as a raw
  `AttributeError` from a private MuJoCo binding.
