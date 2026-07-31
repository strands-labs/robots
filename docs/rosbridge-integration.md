---
description: use_rosbridge - bridge a Strands agent to any ROS 1 or remote robot over a rosbridge WebSocket (pure pip, no ROS install).
---

# rosbridge integration

`use_rosbridge` speaks the rosbridge JSON protocol over a WebSocket via
pure-pip `roslibpy` - no ROS environment needed on the agent's machine. This
gives rosbridge two properties no other strands-robots transport has:

* **ROS 1 robots** - rosbridge_suite ships for ROS 1 (and ROS 2) alike. Drive
  the NASA Curiosity rover Gazebo simulation (ROS 1 Noetic), or any ROS 1
  system.
* **No ROS install on this machine** - the agent can run on macOS, CI, WSL, or
  any laptop and introspect and drive robots across a network.

```python
from strands import Agent
from strands_robots.mesh import RosbridgeRobot

# Stock NASA-sim wiring: cmd_vel/odom topics and safety limits preconfigured.
rover = RosbridgeRobot.from_curiosity(host="192.168.1.20")
agent = Agent(tools=rover.tools)
agent("drive forward for 5 seconds, then report the odometry")
```

## Requirements

Install the `[rosbridge]` extra:

```bash
pip install "strands-robots[rosbridge]"
```

The robot side runs `rosbridge_server` with `rosapi` enabled - standard in every
rosbridge install. rosbridge is unauthenticated by default: use on trusted
networks only. rosauth is out of scope.

```bash
# ROS 1 Noetic example
apt-get install ros-noetic-rosbridge-suite
roslaunch rosbridge_server rosbridge_websocket.launch

# ROS 2 example
apt-get install ros-<distro>-rosbridge-suite
ros2 launch rosbridge_server rosbridge_websocket_launch.xml
```

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | roslibpy availability + connectivity to host:port |
| `list_topics` | - | Topics with their types (rosapi /rosapi/topics) |
| `list_services` | - | Services (rosapi /rosapi/services) |
| `echo` | `topic` (type auto-resolved) | N samples as JSON |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `service_call` | `service`, `type` | Service response as JSON |

## Examples

```python
from strands_robots.tools import use_rosbridge

# Check connectivity
use_rosbridge(action="status", host="192.168.1.20")

# Graph introspection
use_rosbridge(action="list_topics")
use_rosbridge(action="list_services")

# Subscribe and read one sample (type auto-resolved via rosapi)
use_rosbridge(action="echo", topic="/curiosity_mars_rover/odom", count=1)

# Publish a velocity command
use_rosbridge(action="publish",
              topic="/curiosity_mars_rover/ackermann_drive_controller/cmd_vel",
              type="geometry_msgs/Twist",
              fields={"linear": {"x": 1.0}, "angular": {"z": 0.0}})

# Call a service with a JSON request
use_rosbridge(action="service_call", service="/some_service",
              type="std_srvs/Trigger",
              fields={})
```

Graph introspection uses the `rosapi` node's services. Interface types are
ROS 1-style two-segment names (e.g. `geometry_msgs/Twist`); field payloads are
plain JSON dicts, exactly as rosbridge transmits them.

## RosbridgeRobot

For mobile bases that expose the standard `cmd_vel` / odometry / scan topic
trio, `RosbridgeRobot` wraps that wiring so a remote ROS 1 or remote robot
drives like any other strands robot - the same `Agent(tools=[robot])` pattern
used for simulated and hardware arms.

### Constructor

```python
from strands_robots.mesh import RosbridgeRobot

robot = RosbridgeRobot(
    node_name="my_robot",
    cmd_vel_topic="/cmd_vel",
    odom_topic="/odom",
    scan_topic="/scan",  # optional
    host="192.168.1.20",
    port=9090,
    cmd_vel_type="geometry_msgs/Twist",  # defaults; matches ROS 1
    odom_type=None,  # auto-resolved via rosapi when omitted
    scan_type=None,
    max_linear=2.0,  # m/s clamp
    max_angular=1.0,  # rad/s clamp
    max_duration=30.0,  # longest accepted drive() hold
    publish_rate=10.0,  # Hz
)
```

All parameters are optional except `node_name`, `cmd_vel_topic`, and `odom_topic`.

### Fleet drive contract

The `drive()` method follows the strands fleet-standard contract:

```python
# Direct, programmatic control:
robot.drive(linear=1.0, angular=0.0, duration=2.0)  # hold for 2 seconds
robot.drive(linear=1.0)  # latch until stop() - single-shot, ungated
robot.stop()  # always publishes zero Twist, never gated
pose = robot.get_pose()  # read one odometry sample
scan = robot.get_scan()  # read one laser scan (error if no scan_topic)
```

**Safety semantics:**

- **Finite-input guards**: non-finite (NaN, inf) linear or angular velocities
  are rejected before any publish.
- **Velocity clamps**: linear and angular are independently clamped to
  `max_linear` and `max_angular`.
- **Loud duration rejection**: `duration` must be positive, finite, and at most
  `max_duration`; anything else returns a detailed error and nothing is published.
- **Timed-command trailing zero**: every drive with a `duration` argument and a
  non-zero command (or multi-message publish) automatically publishes a single
  zero Twist afterwards - even if the main publish failed - so a timed drive
  cannot leave the robot with a live velocity.
- **Single-shot latch**: a bare single-message `drive()` (no `duration`, no
  `count > 1`) publishes once and latches in the robot's controller until
  `stop()` is called. This is standard cmd_vel behavior.
- **stop() never gated**: the stop method always publishes a zero Twist,
  regardless of prior state or publish failures.

| Method | ROS action | Notes |
|--------|------------|-------|
| `drive(linear, angular, duration=, count=)` | publish `Twist` to `cmd_vel_topic` | `duration` holds the command at `publish_rate` Hz; no `duration` latches until stop |
| `stop()` | publish zero `Twist` | never gated on anything |
| `get_pose()` | echo `odom_topic` | returns up to 1 sample |
| `get_scan()` | echo `scan_topic` | error when no `scan_topic` configured |
| `.tools` | - | per-instance named agent tools |

### from_curiosity

The NASA Curiosity rover Gazebo simulation (ROS 1 Noetic) is pre-wired:

```python
rover = RosbridgeRobot.from_curiosity(
    host="localhost",  # or Docker container IP / another machine
    port=9090,
    node_name="curiosity",  # optional
    # ... any constructor param as override
)
```

The stock wiring uses the rover's `ackermann_drive_controller` (consumes
`geometry_msgs/Twist` directly - no kinematic model needed on the agent side),
and imports safety limits (max_linear=2.0 m/s, max_angular=1.0 rad/s,
max_duration=30s) ported from the strands-robots-ros2 registry.

## Curiosity quickstart (Docker, headless)

Build a self-contained ROS 1 Noetic + Gazebo + rosbridge image, then launch
the agent on your machine:

```bash
# 1. Sim container: ROS1 Noetic + Gazebo (server only) + rosbridge
docker build -t curiosity-sim - <<'EOF'
FROM ros:noetic-robot
RUN apt-get update && apt-get install -y --no-install-recommends \
    ros-noetic-gazebo-ros ros-noetic-gazebo-ros-control \
    ros-noetic-gazebo-plugins ros-noetic-controller-manager \
    ros-noetic-joint-state-controller ros-noetic-effort-controllers \
    ros-noetic-position-controllers ros-noetic-velocity-controllers \
    ros-noetic-robot-state-publisher ros-noetic-xacro \
    ros-noetic-rosbridge-suite git \
    && rm -rf /var/lib/apt/lists/*
RUN git clone --depth 1 https://github.com/mark-gl/curiosity_mars_rover_ws.git /ws
WORKDIR /ws
RUN bash -lc "source /opt/ros/noetic/setup.bash && catkin_make \
    -DCATKIN_WHITELIST_PACKAGES='ackermann_drive_controller;curiosity_mars_rover_description;curiosity_mars_rover_control;curiosity_mars_rover_gazebo'"
CMD bash -lc "source /ws/devel/setup.bash && \
    roslaunch curiosity_mars_rover_gazebo main_mars_terrain.launch gui:=false rviz:=false & \
    sleep 30 && roslaunch rosbridge_server rosbridge_websocket.launch"
EOF
docker run -d --name curiosity -p 9090:9090 curiosity-sim

# 2. Agent side (this machine - no ROS needed)
pip install "strands-robots[rosbridge]" strands-agents
python examples/rosbridge/curiosity_agent.py
```

The simulation takes ~30 seconds to warm up. Then the agent will read the
rover's initial pose, execute two drive legs with a turn, and report the
displacement.

## Security

rosbridge is **unauthenticated by default**. The WebSocket accepts connections
from any network address on the port. For trustworthy deployments:

- **Trusted networks only**: use rosbridge on private intranets, never expose
  port 9090 to the internet.
- **rosauth out of scope**: the `rosauth` ROS package can add authentication,
  but configuration and key distribution are operator responsibilities; this
  library does not provision them.
- **Network isolation**: firewall the rosbridge port to known agent machine IPs.
