### Fixed

- **simulation/mujoco**: `move_to`, `rotate_wrist` and `set_gripper` now write a joint pose
  only into an actuator whose `ctrl` IS a joint pose, the split
  `scene_ops.joint_drive_map` already owns. A `<velocity>` drive reads the same number as a
  rate and a `<motor>` as a torque, so the primitives previously commanded the wrong
  physical quantity on any non-servo joint drive and reported success: `rotate_wrist` said
  a set-point was reached while the joint kept accelerating past it, and the joints it
  promised to hold were driven away from the pose being held. Where only other joints are
  affected - the wheels of `lekiwi`, `stretch3` and `tiago_dual`, whose arms are fully
  servo-driven - those drives are now left uncommanded and named in the payload; where the
  targeted drive itself cannot take a pose, the primitive refuses and names the actuator.
