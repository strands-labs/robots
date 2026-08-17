#!/usr/bin/env python3
"""Drive a ROS 2 turtle (or any cmd_vel robot) through the mesh bridge.

Goal: Show that a remote ROS 2 robot becomes a first-class strands Robot. A
``RosBridgedRobot`` wraps the turtle's ``cmd_vel`` / pose topics; an agent then
drives it with natural language, using the same ``Agent(tools=[robot])`` pattern
as the simulated and hardware arms.

Dependencies:
  The bridge forwards to ``use_ros``, which runs in-process through ``rclpy``,
  so this script must run inside an interpreter where a ROS 2 distro is sourced
  (``rclpy`` importable). The easiest self-contained setup is to run everything
  inside a ROS 2 container:

    docker run -it --rm --net host ros:jazzy bash
    # then, inside the container:
    apt-get update && apt-get install -y ros-jazzy-turtlesim python3-pip
    pip install "strands-robots[ros2]" strands-agents --break-system-packages
    source /opt/ros/jazzy/setup.bash
    QT_QPA_PLATFORM=offscreen ros2 run turtlesim turtlesim_node &
    # /turtle1/cmd_vel is a gated command surface; the programmatic section
    # below has no operator to prompt, so pre-approve that one topic:
    export STRANDS_ROS2_COMMAND_ALLOW=/turtle1/cmd_vel
    python3 turtlebot_demo.py

  Already on a machine with ROS 2 sourced? Just ``source .../setup.bash`` and
  run the script directly - no container needed.

Expected output: the turtle's pose changes after the drive commands.
Runtime: ~10 seconds (plus LLM latency for the agent section).
"""

from strands_robots.mesh import RosBridgedRobot

# 1. Wrap the remote ROS 2 robot. Nothing connects yet - the bridge is thin.
turtle = RosBridgedRobot.from_ros(
    node_name="turtlesim",
    cmd_vel_topic="/turtle1/cmd_vel",
    odom_topic="/turtle1/pose",
    odom_type="turtlesim/msg/Pose",
)

# 2. Direct, programmatic control. Needs STRANDS_ROS2_COMMAND_ALLOW to name
#    /turtle1/cmd_vel (see the header): a script has no operator context, so
#    use_ros's command gate has nobody to ask. The agent section below does
#    carry one - the bridge's drive/stop tools forward it, so the operator is
#    prompted per command instead.
print("before:", turtle.get_pose()["content"][0]["text"])
turtle.drive(linear=2.0, angular=1.5, duration=1.5)
print("after: ", turtle.get_pose()["content"][0]["text"])
turtle.stop()

# 3. Hand the robot to an agent - its methods become named tools. Uncomment to
#    run (needs strands-agents + a model provider configured):
#
# from strands import Agent
# agent = Agent(tools=turtle.tools)
# agent("drive forward for two seconds while turning left, then report the pose")
