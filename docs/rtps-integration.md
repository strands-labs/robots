---
description: use_rtps and RtpsRobot - join a ROS 2 graph as a DDS participant with no rclpy, no sourced ROS distro. Act as a robot over pure RTPS.
---

# Pure-RTPS ROS 2 integration

ROS 2 runs over DDS, and DDS speaks RTPS on the wire. `use_rtps` lets a strands
agent join a ROS 2 graph as a **first-class DDS participant** using only the
pip-installable `cyclonedds` binding - **no `rclpy`, no sourced ROS 2 distro, no
`ros2` CLI**. Because RTPS is stable across ROS 2 distros, one implementation
interoperates with Humble, Jazzy, Rolling, and beyond.

## use_ros vs use_rtps

| | `use_ros` | `use_rtps` |
|---|-----------|------------|
| Role | client / observer | **participant / robot** |
| Backend | in-process `rclpy` | `cyclonedds` (pip wheel) |
| Needs sourced ROS 2 | yes | **no** |
| Type coverage | any installed interface | curated IDL bundle |
| Runs on macOS / CI bare | no (needs ROS) | **yes** |

Use `use_ros` when you have ROS 2 sourced and need full type coverage or
services. Use `use_rtps` when you want zero-install interop or to **act as a
robot** - publishing topics a real ROS 2 stack (rviz, nav2, a teleop node) will
consume, indistinguishable from hardware on the wire.

```bash
pip install 'strands-robots[ros2]'   # cyclonedds - a self-contained wheel
```

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | Whether the cyclonedds backend is available |
| `types` | - | The ROS 2 message types in the local IDL bundle |
| `advertise` | `topic`, `type` | Creates a publisher (appear on the graph) |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `subscribe` | `topic`, `type` | Creates a subscription |
| `echo` | `topic`, `type` | Returns the next N samples as JSON |

Scope (v1) is topics only; services and actions need the ROS 2 request/reply-
over-DDS protocol and are a focused follow-up.

## Type coverage

To publish a message you must own its type definition locally, so `use_rtps`
ships a curated IDL bundle (`strands_robots.rtps.idl`) of the common ROS 2
messages, registered under their ROS 2 type strings: the `geometry_msgs`
primitives (`Twist`/`Pose`/...) plus the `sensor_msgs` `JointState` and `Image`
(with their `std_msgs/Header` + `builtin_interfaces/Time` chain) that the
rclpy-free hardware bridge publishes. List them with
`use_rtps(action="types")`. Arbitrary custom messages are out of scope until
cyclonedds-python's dynamic (XTypes) support matures - use `use_ros` (rclpy) for
those.

ROS 2 names are mangled to their DDS form automatically: a topic `/turtle1/cmd_vel`
becomes `rt/turtle1/cmd_vel`, and a type `geometry_msgs/msg/Twist` becomes
`geometry_msgs::msg::dds_::Twist_` - the conventions that make a bare DDS
participant interoperable with real ROS 2 nodes.

Message interfaces only. ROS 2 has no single DDS type for a service or an
action: `rosidl` generates one type per constituent message, so
`example_interfaces/srv/AddTwoInts` becomes
`example_interfaces::srv::dds_::AddTwoInts_Request_` and
`..._Response_` on the `rq`/`rr` prefixes rather than `rt`. A `pkg/srv/Name` or
`pkg/action/Name` type is therefore refused, with the types ROS 2 does generate
quoted - an invented `pkg::srv::dds_::AddTwoInts_` would match no participant,
and DDS reports a type mismatch as silence rather than as an error.

## Examples

```python
from strands_robots.tools import use_rtps

use_rtps(action="status")
use_rtps(action="types")

# Act as a robot: advertise then drive a cmd_vel topic a real node consumes.
use_rtps(action="advertise", topic="/turtle1/cmd_vel", type="geometry_msgs/msg/Twist")
use_rtps(action="publish", topic="/turtle1/cmd_vel",
         type="geometry_msgs/msg/Twist",
         fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}},
         count=15, rate=10.0)
```

## RtpsRobot: a ROS 2 robot over pure RTPS

`RtpsRobot` is the pure-RTPS sibling of `RosBridgedRobot`. It forwards to
`use_rtps`, so it drives a ROS 2 mobile base with nothing but a pip wheel - and
because it publishes real DDS samples, it can act as the robot itself.

```python
from strands import Agent
from strands_robots.mesh import RtpsRobot

turtle = RtpsRobot.from_rtps(
    node_name="turtlesim",
    cmd_vel_topic="/turtle1/cmd_vel",
)

turtle.advertise()                       # appear on the graph
turtle.drive(linear=1.0, duration=1.5)   # publish Twist over RTPS for 1.5 s
turtle.stop()

agent = Agent(tools=turtle.tools)        # drive_turtlesim, stop_turtlesim
agent("drive forward for two seconds")
```

See `examples/ros2/rtps_turtle_demo.py` for an end-to-end script, and
`tests_integ/tools/test_use_rtps_live.py` for the gated live test that drives a
real `turtlesim` from a bare participant (`RTPS_LIVE=1 pytest -m rtps`).

For a fully reproducible, self-contained cross-process proof (real turtlesim
node + our publisher, one command), see `examples/ros2/rtps_proof/`:

```bash
cd examples/ros2/rtps_proof
docker compose run --build --rm proof   # exits 0 iff the turtle moved
```

## Hardware bridge over pure RTPS (no rclpy)

`Robot(ros2_bridge=True)` defaults to the rclpy backend (`ros2_transport="rclpy"`,
full `sensor_msgs` fidelity, needs a sourced ROS 2 distro). Pass
`ros2_transport="rtps"` to run the **same bridge over pure cyclonedds** instead -
a single pip wheel, no rclpy and no sourced distro:

```python
from strands_robots import Robot

# rclpy-free: publishes /so101/joint_states (+ camera image_raw) and subscribes
# /so101/joint_command -> send_action, all over cyclonedds RTPS.
arm = Robot("so101", mode="real", ros2_bridge=True, ros2_transport="rtps")
```

The two transports emit byte-identical topics, so a real ROS 2 node (or
`ros2 topic echo` / `ros2 topic pub`) cannot tell them apart on the wire:

```bash
ros2 topic echo /so101/joint_states     # decodes the cyclonedds-published JointState
ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
  '{name: ["shoulder_pan.pos"], position: [0.1]}'   # drives the arm
```

The trade-off is the same as `use_rtps`: type coverage is bounded by the IDL
bundle (joint_states + image_raw are in; anything else needs the rclpy backend).
The bridge is implemented by `strands_robots.hardware_rtps_bridge.HardwareRtpsBridge`,
the rclpy-free sibling of `HardwareRosBridge`. Both derive from
`strands_robots.ros_telemetry.RosTelemetryBase`, which owns the topic names and the
inbound `joint_command` parsing, so the two transports are byte-identical on the wire
by construction; they present the identical `publish_joint_states` / `publish_image` /
inbound-`joint_command` surface.

## Securing the inbound command surface

The inbound `/<robot>/joint_command` subscription lets **any participant on the
DDS domain drive the physical arm**. Two layers harden it, both threaded through
the `Robot()` constructor.

### DDS Security gate (RTPS only)

When the command surface is enabled (`ros2_bridge=True`, `ros2_commands=True`,
`ros2_transport="rtps"`), `HardwareRtpsBridge` **refuses to start** unless one of
the following is true:

- a `dds_security_config` dict is supplied, or
- the operator sets `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` (truthy:
  `1` / `true` / `yes`) to explicitly accept an unsecured graph.

A telemetry-only bridge (`ros2_commands=False`) is publish-only and is **not**
gated. `dds_security_config` requires the following keys (each a path or a
`file:` / `data:` URI per the OMG DDS-Security spec); `permissions_ca` is
optional:

```python
from strands_robots import Robot

arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_transport="rtps",
    dds_security_config={
        "identity_ca":  "file:/etc/dds/identity_ca.pem",   # identity CA
        "certificate":  "file:/etc/dds/participant.pem",   # participant cert
        "private_key":  "file:/etc/dds/participant_key.pem",
        "governance":   "file:/etc/dds/governance.p7s",    # signed governance
        "permissions":  "file:/etc/dds/permissions.p7s",   # signed permissions
        # "permissions_ca": "file:/etc/dds/permissions_ca.pem",  # optional
    },
)
```

The credentials are wired into the cyclonedds `DomainParticipant` QoS together
with the builtin DDS-Security plugins, so **both** the outbound telemetry and the
inbound command surface ride an authenticated, access-controlled graph. A
half-filled config is rejected at construction.

`dds_security_config` is RTPS-specific: passing it with `ros2_transport="rclpy"`
raises, because the rclpy backend gets its DDS Security from the ROS 2 RMW
keystore/env (`ROS_SECURITY_*` / `sros2`), not from a config dict.

### Joint position bounds

`joint_limits={motor: (min, max)}` (threaded into either transport) range-checks
every inbound command. If **any** commanded joint falls outside its declared
range, the **entire** command is rejected - never partially applied - so one
out-of-range joint can never drive part of the arm while the rest holds. Joints
without a declared bound are unconstrained; to leave a joint unbounded, omit it
rather than declaring an infinite bound. Every bound must be a finite number - a
non-finite one declares a range that admits nothing, and the bridge refuses it at
construction rather than dropping every command for that joint mid-run.

```python
arm = Robot(
    "so101",
    mode="real",
    ros2_bridge=True,
    ros2_transport="rtps",
    dds_security_config={...},
    joint_limits={"shoulder_pan.pos": (-3.14, 3.14), "elbow.pos": (-1.57, 1.57)},
)
```

## Safety

Agent-supplied topic and type names are validated against an allowlist before
mangling (absolute `/`-rooted alnum/`_` names; `pkg/msg/Name` types). The tool
never constructs a shell command or generates source, so there is no
command-injection or `eval` surface. Backend, type-resolution, and field errors
are returned as structured `{"status": "error"}` results rather than raised.

The numeric options are checked in the same place, ahead of the backend probe, so
a refusal happens before a writer joins the graph and reports identically whether
or not `cyclonedds` is installed. `count` (`publish`, `echo`) must be a positive
integer, and `rate` (`publish`) and `timeout` (`echo`) must be positive finite
numbers - the same accepted domain `use_ros` enforces, so a value publishable
through one transport is publishable through the other. An option the requested
action never reads is not second-guessed.
