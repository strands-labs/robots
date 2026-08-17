### Fixed

- **Registry**: every GR00T `data_config` name now resolves to its embodiment's robot. Five names the
  shipped catalog advertises were claimed by no registry entry, so `Robot("unitree_g1_sonic")` and
  `add_robot(data_config="oxe_droid_relative_eef_relative_joint")` refused a robot that sibling configs
  of the same embodiment loaded, and `move_to`'s registry gripper lookup silently fell back to the
  heuristic that metadata exists to replace. `unitree_g1_real`, `unitree_g1_sonic` and
  `real_g1_relative_eef_relative_joints` now resolve the Unitree G1; `oxe_droid_relative_eef_relative_joint`
  and `oxe_droid_rel` resolve the Panda they declare via `_extends`.
