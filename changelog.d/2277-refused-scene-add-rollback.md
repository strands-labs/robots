### Fixed

- **simulation/mujoco**: a refused `add_object` / `add_camera` no longer deletes the
  scene element it collided with. The rollbacks resolved the element to remove by
  name, which answers with the pre-existing one rather than the copy the refused
  call appended, so on MuJoCo builds that defer a repeated-name error to compile
  the scene's own body/camera was deleted and the reject silently inherited its
  name, and on builds that raise on insert the orphan was never removed at all and
  every later scene mutation failed to recompile. Rollback now deletes only the
  elements beyond the count taken before the insert.
