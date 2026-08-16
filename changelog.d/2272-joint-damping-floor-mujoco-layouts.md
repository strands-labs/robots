### Fixed

- `actuate_robot` no longer refuses every URDF-loaded arm on MuJoCo builds where
  `MjsJoint.damping` is a plain float rather than a per-DOF sequence. Both
  layouts occur across the declared `mujoco>=3.2.0,<4.0.0` range; writing the
  damping floor through the wrong one raised `TypeError`, which was reported as
  a refused spec surgery, leaving the arm actuator-less and undrivable by
  `send_action` / `run_policy`.
