#!/usr/bin/env python3
"""Drive the NASA Curiosity Mars rover in Gazebo with natural language.

Goal: Show RosbridgeRobot presenting a ROS1 robot as strands agent tools over
a rosbridge WebSocket - with NO ROS environment on this machine (pure pip).

Pipeline:
  NL -> Strands Agent -> RosbridgeRobot -> use_rosbridge -> rosbridge (ws://:9090)
     -> Curiosity rover (ROS1 Noetic + Gazebo)

Dependencies:
  pip install "strands-robots[rosbridge]" strands-agents
  AWS credentials for Bedrock (or any strands-agents model provider).
  Optional: MODEL=<bedrock model id> to override the default model.

The simulation runs separately (it is NOT part of this package). One-time
standup with Docker, headless, from the community Curiosity workspace:

  docker run -d --name curiosity -p 9090:9090 <noetic-image> bash -lc '
    source /ws/devel/setup.bash &&
    roslaunch curiosity_mars_rover_gazebo main_mars_terrain.launch gui:=false rviz:=false &
    sleep 25 &&
    roslaunch rosbridge_server rosbridge_websocket.launch'

(or natively: ROS1 Noetic + https://github.com/mark-gl/curiosity_mars_rover_ws
 built with catkin_make, plus ros-noetic-rosbridge-suite. See
 docs/rosbridge-integration.md for the full recipe.)

Expected output: the agent drives the rover across Mars terrain and reports
the odometry displacement. Runtime: ~30 seconds (LLM latency + 2 drive legs).
"""

import os

from strands import Agent

from strands_robots.mesh import RosbridgeRobot

# Stock NASA-sim wiring: cmd_vel/odom topics and safety limits preconfigured;
# point host at wherever rosbridge runs (docker port-map, another machine...).
rover = RosbridgeRobot.from_curiosity(host=os.environ.get("ROS_HOST", "localhost"))

# MODEL env selects any strands-agents Bedrock model id (defaults to the
# strands default); e.g. MODEL=us.amazon.nova-pro-v1:0 for accounts without
# Anthropic model access.
agent = Agent(model=os.environ.get("MODEL"), tools=rover.tools)

result = agent(
    os.environ.get(
        "PROMPT",
        "Read the rover's current pose. Then drive forward at 1.0 m/s for 5 "
        "seconds, arc left gently for 3 seconds, and stop. Read the pose again "
        "and report how far the rover moved across the Martian terrain.",
    )
)

print(f"Agent completed: {result}")
