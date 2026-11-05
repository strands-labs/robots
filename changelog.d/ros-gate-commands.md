### Fixed: the ROS command gate covers parameter writes and ros2_control's controller manager

`use_ros` now treats a parameter write as a motion surface: the `set_param` verb is gated as `<node>/set_parameters`,
and calls to `set_parameters`, `set_parameters_atomically` and rosbridge's `/rosapi/set_param` need an exact
`STRANDS_ROS_ALLOW` entry or an operator's yes, like `/cmd_vel` and `/<controller>/commands`. The same holds for the
controller manager's `switch_controller`, `load_controller`, `configure_controller`, `unload_controller` and
`set_hardware_component_state`, which start or stop controllers and hardware.
