### Fixed

- Point the missing-`rclpy` refusal at the step that supplies it. `ros2_bridge=True`
  without a sourced ROS 2 distro offered two remedies and neither worked: `pip install
  'strands-robots[ros2]'` exits 0 having installed only the cyclonedds RMW binding, which
  leaves `rclpy` exactly as missing, and `pip install rclpy` fails because `rclpy` is not
  published on PyPI. `require_optional` gains `system_install=` for a module that arrives
  with a system package rather than from an index, and the two `rclpy` sites now name
  sourcing a distro - plus, where the caller can take it, the pure-RTPS transport that
  publishes the same topics over the pip-installable cyclonedds binding.
