---
description: A ROS 2 robot as a strands-robots Robot, the use_ros agent tool, and your own robot from a profile and a codec.
---

# ROS 2 robots

A ROS 2 robot becomes a normal hardware `Robot`, with the safety chain, e-stop, mesh and dashboard. What is robot-specific (topics, message types, QoS, joint ids, gains, what "armed" means)
lives in a JSON **profile**. Two are built in:

```python
from strands_robots import ros

print(ros.available())  # ['ros2_control', 'unitree_g1']
```

rclpy and the message packages come from your ROS 2 install (Humble or Jazzy), never from pip. Source the distro
and use a venv made with `--system-site-packages`. `import strands_robots` never imports rclpy.

## Connect

<!-- test: skip, needs a ROS 2 graph -->
```python
from strands_robots import Robot, ros

arm = ros.connect("ros2_control", host="arm.local")  # model from the robot's /robot_description
g1 = Robot("unitree_g1", driver="ros", released=True)  # on the G1's Jetson, low-level /lowcmd at 500 Hz
```

`host=` is the robot's address: a zenoh router or a DDS peer.

## Stream rules

One command stream owns the command topic. It sends on a timer (`trigger: timer`) or answers every sensor
frame (`trigger: state`). A late policy gets its last target repeated; no target for `deadman_ms` holds the
measured pose. It never damps. Writes are refused unless the profile's `status.armed` fields
hold. `shadow=True` builds every command, publishes none, and reports `shadow_max_err` against the profile's
`shadow` topic. kp/kd from `Robot.set_gains` or a policy's stiffness and damping go out with each command.

## The use_ros tool

<!-- test: skip, needs a ROS 2 graph -->
```python
from strands import Agent
from strands_robots.ros import use_ros

agent = Agent(tools=[use_ros()])
agent("What topics are up? Echo /joint_states once.")
```

Reads are free. A write to a motion surface needs an exact `STRANDS_ROS_ALLOW` entry or an operator's yes. Motion
surfaces include velocity and joint command topics, ros2_control's `/<controller>/commands`, parameter writes
(`set_param`, which is gated as `<node>/set_parameters`, and the `set_parameters[_atomically]` services) and the
controller manager's `switch_controller`, `load_controller`, `configure_controller`, `unload_controller` and
`set_hardware_component_state`. A topic a `Robot` streams is always refused: use that robot's own tool.

## Your own robot

Put a profile in a directory on `STRANDS_ROBOTS_ROS_PROFILES`, or ship it in a package under the
`strands_robots.ros_profiles` entry point. If its messages are not `sensor_msgs/JointState` or Unitree's,
subclass `Codec` and point at it:

<!-- test: skip, needs your robot's message package -->
```python
from strands_robots.robot.drivers.ros import Codec

class MyCodec(Codec):  # profile: "codec": "my_robot.codec:MyCodec"
    needs_gains = True

    def read(self, frame): ...  # -> JointState in action_keys order
    def build(self, q, kp, kd, frame, sender): ...  # -> the command message
```

Arming, id echo (`command.echo`) and the shadow comparison stay data in the profile.
