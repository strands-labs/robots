#!/usr/bin/env bash
# Start a real turtlesim node, then let a Strands Agent drive a square via use_ros.
source /opt/ros/jazzy/setup.bash
set -euo pipefail

echo "== starting turtlesim (headless) =="
ros2 run turtlesim turtlesim_node >/tmp/turtle.log 2>&1 &
TURTLE_PID=$!
trap 'kill ${TURTLE_PID} 2>/dev/null || true' EXIT

for _ in $(seq 1 20); do
    grep -q "Spawning turtle" /tmp/turtle.log 2>/dev/null && break
    sleep 0.5
done
grep -q "Spawning turtle" /tmp/turtle.log || { echo "turtlesim failed:"; cat /tmp/turtle.log; exit 2; }

# The agent drives /turtle1/cmd_vel, a gated command surface. This container
# runs unattended, so there is nobody to answer the approval interrupt -
# pre-approve exactly that topic; every other blocked surface stays gated.
export STRANDS_ROS2_COMMAND_ALLOW=/turtle1/cmd_vel

python3 examples/ros2/use_ros/agent_drive.py
