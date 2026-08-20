### Fixed

- **simulation/mujoco**: refuse a model whose actuators span more than one control slot each, naming the actuator, instead of failing with a numpy index error and leaving the robot's bodies compiled into the scene. mujoco 3.12 gave `<pid>` two controls and `<orientation>` up to four, so `model.nu` is no longer the actuator count and the `actuator_*` arrays are shorter than it. `inject_robot_into_scene` now carries a refusal's reason out the way `inject_object_into_scene` already does, so the scene is left as it was found and the robot name stays reusable.
