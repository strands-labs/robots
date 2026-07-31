### Added: rosbridge transport - `use_rosbridge` tool + `RosbridgeRobot` (drive ROS1 robots, no ROS install)

strands-robots previously spoke only ROS 2 (in-process rclpy / DDS). The new
pair mirrors `use_ros` / `RosBridgedRobot` over the rosbridge WebSocket JSON
protocol via pure-pip `roslibpy` (optional `[rosbridge]` extra): agents on any
machine - macOS, CI, no sourced ROS - can introspect and drive ROS1 or remote
robots. `RosbridgeRobot.from_curiosity()` preconfigures the NASA Curiosity
Mars rover Gazebo simulation (ROS1 Noetic), with the hardened drive contract
(finite-input guards, clamps, loud duration rejection, timed-command trailing
zero, ungated stop). See docs/rosbridge-integration.md and
examples/rosbridge/curiosity_agent.py.
