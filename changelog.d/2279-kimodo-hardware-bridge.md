### Added: Kimodo action-key bridge for lerobot's Unitree G1 driver

`strands_robots.policies.kimodo.hardware` renames Kimodo's URDF-named joint
targets (`left_hip_pitch_joint`) to the action keys lerobot's `UnitreeG1` driver
accepts (`kLeftHipPitch.q`), so the same `KimodoPolicy` object can drive sim
actuators and the real robot's DDS lowcmd path.

* `build_lerobot_g1_action_dict(action, extra_action_keys=None)` — the driver
  action dict for one control tick.
* `kimodo_action_to_lerobot_g1(action)` — the rename on its own.
* `get_joint_map()` — the rename table, for callers renaming in their own loop.

The table pairs joints by name, not by position in the driver enum. The driver
applies only the action keys it recognises and leaves every other motor on its
previous command, so a mis-paired or unknown key raises nothing at all. Pairing
by name means a driver-side reorder cannot move a target, and a driver-side
rename or DOF change is refused with the unmatched joints named on both sides.

Nothing in the rollout loop applies the rename implicitly; a hardware caller
renames the action dict before `send_action`. See `docs/policies/kimodo.md`.
