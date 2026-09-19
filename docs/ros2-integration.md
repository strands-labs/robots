---
description: use_ros - bridge a Strands agent to any ROS 2 graph (topics, services) in-process through rclpy, with dynamic message-type resolution.
---

# ROS 2 integration

`use_ros` gives a Strands agent one structured entry point into any ROS 2 graph
reachable from the interpreter - listing and echoing topics, publishing
messages, and calling services - **entirely in-process through `rclpy`**. There
is no `ros2` CLI shelling and no generated-code snippets: every action calls the
ROS 2 client library directly, so message types are real Python classes, errors
are real exceptions, and a single long-lived node/executor is reused across
calls.

```python
from strands import Agent
from strands_robots import use_ros

agent = Agent(tools=[use_ros])
agent("list the ROS 2 topics, then drive /turtle1 forward and confirm its pose changed")
```

![A Strands agent driving a closed-loop square in turtlesim via use_ros](assets/use_ros_agent_square.gif)

*A Strands agent (Claude Opus via Amazon Bedrock) given the `use_ros` tool drives
a real ROS 2 `turtlesim` in a closed-loop square - reading pose, correcting
heading, and re-driving - over 43 in-process `use_ros` calls. See
[`examples/ros2/use_ros/`](https://github.com/strands-labs/robots/tree/main/examples/ros2/use_ros).*

## ROS 2 surfaces at a glance

strands-robots meets ROS 2 from four complementary angles - pick by what you
have and what you want to do:

| Surface | Role | Backend | Needs sourced ROS 2 | Use it to |
|---------|------|---------|---------------------|-----------|
| **`use_ros`** tool | client / observer + commander | in-process `rclpy` | yes | List/echo/publish topics, call services on any ROS 2 graph - full type coverage |
| **`use_rtps`** tool | participant / **act as a robot** | pure `cyclonedds` (pip) | **no** | Join a graph as a DDS peer and publish topics a real stack consumes; works on macOS/CI/Linux x86_64 from the wheel, all distros; Linux aarch64 (Jetson) builds from source - see [rtps integration](rtps-integration.md#linux-aarch64-jetson) |
| **`use_rosbridge`** tool + **`RosbridgeRobot`** | ROS1 / remote robots over a rosbridge WebSocket | pure-pip `roslibpy` | **no** | Drive ROS1 robots (e.g. the NASA Curiosity Gazebo sim) or any remote rosbridge robot from a machine with no ROS install - see [rosbridge integration](rosbridge-integration.md) |
| **`RosBridgedRobot`** | a ROS 2 robot as a strands `Robot` | in-process `rclpy` | yes | `drive()`/`get_pose()` a `cmd_vel`/odom base with the same `Agent(tools=[robot])` UX as sim/hardware |
| **`AckermannRosRobot`** | an Ackermann ROS 2 car as a strands `Robot` | in-process `rclpy` | yes | `drive()`/`get_scan()` a steering-geometry car (AWS DeepRacer servo stack) with bicycle-model conversion and an automatic enable handshake |
| **`SimEngine(ros2_bridge=True)`** | the **simulation as a ROS node** | `rclpy` | yes | Publish a running MuJoCo sim's `joint_states` + camera `image_raw` so rviz/nav2/agents can subscribe |
| **`Robot(ros2_bridge=True)`** | a **real robot as a ROS node** (full duplex) | `rclpy` | yes | Publish a physical arm's live `joint_states` + camera `image_raw` so rviz/nav2/agents subscribe to the hardware, **and** subscribe to `joint_command` to drive the arm - symmetric to the sim bridge, plus an inbound command path the sim does not need |

The first three are documented below; the sim bridge has its own section. The
`use_rtps` pure-RTPS path (no rclpy, every ROS 2 distro) is on the
[Pure-RTPS ROS 2](rtps-integration.md) page.

## Requirements

The tool needs `rclpy` and `rosidl_runtime_py` importable in the same
interpreter that runs the agent. These ship with a sourced system ROS 2 distro
and are **not** on PyPI, so they cannot be `pip install`ed and are not pinned in
`pyproject.toml`. Source a ROS 2 environment before launching the agent:

```bash
source /opt/ros/jazzy/setup.bash   # or your distro / RoboStack / conda env
```

When `rclpy` is not importable, every action returns a clear, actionable error
naming the remedy (it never raises). Check the active backend with
`use_ros(action="status")`, which reports either `rclpy (in-process)` or `none`.

The `[ros2]` extra is minimal and optional - it only pulls the pip-installable
`cyclonedds` DDS RMW binding. It does **not** provision ROS 2 by itself; you
still need a real sourced distro.

```bash
pip install 'strands-robots[ros2]'   # optional cyclonedds RMW binding only
```

The binding is a pre-built wheel on macOS, Windows and Linux x86_64. No
cyclonedds release publishes a Linux **aarch64** wheel, so on a Jetson or a
robot's onboard computer the extra resolves to the sdist, which builds against
an existing Cyclone DDS C install (`CYCLONEDDS_HOME`) - the recipe is in
[rtps integration](rtps-integration.md#linux-aarch64-jetson).

## Actions

| Action | Required args | Returns |
|--------|---------------|---------|
| `status` | - | Whether the in-process rclpy backend is available |
| `list_topics` | - | Topics with their message types |
| `list_nodes` | - | Node names |
| `list_services` | - | Services with their types |
| `info` | `topic` or `service` | Topic (type + pub/sub counts) or service (type) details |
| `echo` | `topic` (type auto-resolved) | N samples as JSON |
| `publish` | `topic`, `type` | Publishes N messages built from `fields` |
| `service_call` | `service`, `type` | Service response as JSON |
| `list_actions` | - | Action servers with their types |
| `action_send_goal` | `action_name`, `type` | Terminal `{goal_status, result, feedback}` as JSON; goal is cancelled if `timeout` expires |

Graph introspection (`list_*`, `info`, `echo` type auto-resolution) uses the
rclpy node API directly (`get_topic_names_and_types`, `get_node_names_and_namespaces`,
`get_service_names_and_types`, `count_publishers`/`count_subscribers`). Message
and service types are resolved dynamically through `rosidl_runtime_py`
(`get_message` / `get_service`), so any interface installed in the ROS 2
environment works with no static registry. Field payloads are plain Python
dicts applied with `set_message_fields` (the standard ROS 2 idiom) - passed
straight to rclpy, never serialised through source, so booleans and `null` are
preserved by construction.

## Examples

```python
use_ros(action="status")
use_ros(action="list_topics")

# Subscribe and read two samples (type auto-resolved from the graph)
use_ros(action="echo", topic="/turtle1/pose", count=2, timeout=2.0)

# Publish a velocity command. /cmd_vel is a gated surface - see
# "Safety-critical command surfaces need operator approval" below.
use_ros(action="publish", topic="/turtle1/cmd_vel",
        type="geometry_msgs/msg/Twist",
        fields={"linear": {"x": 2.0}, "angular": {"z": 1.5}})

# Call a service with a JSON request
use_ros(action="service_call", service="/spawn",
        type="turtlesim/srv/Spawn",
        fields={"x": 3.0, "y": 3.0, "name": "t2"})
```

## Try it live

A reproducible, one-command showcase drives a real `turtlesim` through every
`use_ros` action (in-process rclpy, closed sense->act->sense loop), and a second
service runs a Strands Agent that draws the square above from a plain-English
prompt:

```bash
cd examples/ros2/use_ros
docker compose run --build --rm showcase   # every action; exits 0 iff the turtle moved
docker compose run --build --rm agent      # a Strands Agent drives a closed-loop square
```

Captured runs are in `examples/ros2/use_ros/sample_output.txt` and
`agent_sample_output.txt`.

## Safety

Agent-supplied topic, service, and type names are validated against an
allowlist before reaching the rclpy graph/type API (alphanumerics plus
`_ / ~ {}` for names; `pkg/msg/Name` or `pkg/srv/Name` for types). Because the
tool never constructs a shell command or generates source, there is no
command-injection or `eval` surface to defend - the validation simply keeps
malformed names from reaching the ROS 2 client library. Backend and timeout
failures are returned as structured `{"status": "error"}` results rather than
raised exceptions.

The numeric options an action consumes are checked in the same place, ahead of
the backend probe, so a caller mistake reports identically whether or not a ROS 2
distro is sourced and a refusal happens before a publisher joins the graph:

| Option | Consumed by | Accepted values |
|--------|-------------|-----------------|
| `count` | `echo`, `publish` | a positive integer - it is a `range()` bound, so `0` sends nothing and `2.7` or `"3"` cannot be honored |
| `rate` | `publish` | a positive finite number of Hz - the inter-message period is `1 / rate`, so `0`, a negative value, `nan` and `inf` all leave the burst unthrottled instead of paced |
| `timeout` | `echo`, `service_call`, `action_send_goal` | a positive finite number of seconds - `0` and negatives wait for nothing, `inf` never expires |

`timeout` is measured on a monotonic clock, so the budget you ask for is the budget you
get even if the host's wall clock is stepped mid-call by an NTP correction, a `date -s`
or a resume from suspend. A single `action_send_goal` deadline governs server discovery,
goal acceptance and result delivery on that one clock, which is also what keeps the
cancel sent on expiry from being cut short - it needs the executor pumped to leave the
process.

An option the requested action never reads is not second-guessed:
`use_ros(action="status", count=-1)` still reports the backend.

### Safety-critical command surfaces need operator approval

A robot is driven through three different verbs - `publish` to a topic,
`service_call` to a service and `action_send_goal` to an action server - so the
gate is keyed on the surface **name** and consulted from all three. It is also
consulted from every **transport** that reaches the graph, not just this one:
`use_rtps` publishes over raw RTPS and `use_rosbridge` over a WebSocket, and a
`Twist` on `/cmd_vel` moves the same base whichever of the three wrote it. The
blocklist and the approval decision therefore have a single owner
(`strands_robots._command_gate`) rather than a copy per tool, so a surface
refused on one transport cannot be sent on another under a different tool name. An agent
asked to "drive forward" reaches for whichever verb fits the interface it found
on the graph, so gating `publish` alone would leave `/navigate_to_pose` (a ROS 2
action) and `/emergency_stop` (usually a `std_srvs/srv/Trigger` service)
unenforceable. These surfaces are blocked by default:

| Surface | Usually reached by |
|---------|--------------------|
| `/cmd_vel`, `/cmd_vel_unstamped`, `/manual_drive` | `publish` |
| `/joint_command`, `/joint_trajectory`, `/joint_trajectory_controller/joint_trajectory` | `publish` |
| `/emergency_stop`, `/e_stop` | `service_call`, sometimes `publish` |
| `/motor_enable`, `/enable_motor`, `/disable_motor` | `service_call` |
| `/vehicle_state`, `/enable_state` | `service_call` |
| `/navigate_to_pose`, `/follow_path` | `action_send_goal` |

Matching is on the final path segment, so a namespaced form
(`/my_robot/cmd_vel`, `/fleet/robot1/emergency_stop`, and the DeepRacer's
`/webserver_pkg/manual_drive`, `/ctrl_pkg/vehicle_state`,
`/ctrl_pkg/enable_state`) is caught while a lookalike (`/cmd_vel_evil`,
`/joint_trajectory_status`) is not. The name is compared in the
form rclpy resolves it to, so the unrooted `cmd_vel` and the trailing-separator
`/cmd_vel/` are the same surface as `/cmd_vel`. Case is deliberately **not**
folded: ROS 2 graph names are case-sensitive, so `/CMD_VEL` is a genuinely
different topic that no `/cmd_vel` subscriber receives, and refusing it would
block a legitimate surface without closing a path to the robot.

Three ways through the gate, consulted in this order:

| Mode | Mechanism |
|------|-----------|
| Interactive (default) | `tool_context.interrupt()` prompts the operator; reply `y` to approve |
| Headless allowlist | `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel,/follow_path` pre-approves those surfaces and every namespaced surface sharing a base name with one of them; a surface whose base name no entry lists stays gated |
| Fully trusted | `BYPASS_TOOL_CONSENT=true` allows every blocked surface with a WARNING log |

Both lists are matched by **base name** as well as by exact name, after a
leading/trailing `/` is normalised away: a `/cmd_vel` entry matches
`/robot_b/cmd_vel` too. On the blocklist that breadth is the point - one entry
has to catch every namespaced drive topic in the graph. On the pre-approval list
it is the same breadth pointing the other way, so `STRANDS_ROS2_COMMAND_ALLOW=/cmd_vel`
lifts the gate on **every** robot's drive topic, not just the one being driven.
Name the namespace (`/turtle1/cmd_vel`) when the approval should cover one robot,
and list each surface when it should cover several. Case is never folded:
`/CMD_VEL` is a different topic that no `/cmd_vel` subscriber receives.

The gate **fails closed**: with no `tool_context` (outside an agent loop), or when
`interrupt()` is unavailable, the command is refused and the error names both
environment variables. Only the operator's approve/deny verdict is read - the
reply text is never echoed back into the agent's context.

The operator is asked **before** the transport takes its process-wide lock, so a
pending decision does not stall an unrelated read on the same graph - an odometry
`echo`, a scan, a second robot sharing the transport - for however long the human
takes to answer. All three transports consult the gate at that same point.

The reply is recorded in the local safety audit log instead, on both outcomes.
That matters because only `y` / `yes` / `approve` / `approved` count as approval,
so a reply that carries a reason (`n - not while the cell door is open`) is always
a decline - and the audit row is the one place that reason survives. An approval
is recorded too: whether a human authorised an agent to reach a physical surface
is the first thing an incident review asks. Make sure your deployment captures and
retains that log; see [Security](security.md).

Anything that wraps `use_ros` has to forward that context or it inherits the
fail-closed path for every command it sends. `RosBridgedRobot` does: its
`drive_<node>` / `stop_<node>` / `navigate_<node>` tools are declared
`@tool(context=True)` and hand the context on, so an agent driving a bridged
robot prompts the operator. A **programmatic** `robot.drive(...)` has no operator
to prompt and is refused unless the surface is pre-approved - scripts and
unattended demos set `STRANDS_ROS2_COMMAND_ALLOW` for the topics they drive.

Reading is never gated: `echo`, `info` and the `list_*` queries work on a blocked
surface, so telemetry stays available to the agent. The gate also runs *after* the
action's required arguments are validated, so an operator is never asked to
approve a call that could not have run.

## Ackermann robots (AWS DeepRacer)

Differential-drive bases take `geometry_msgs/msg/Twist`; Ackermann cars do
not. The AWS DeepRacer's stock stack subscribes to normalized servo pairs
(`deepracer_interfaces_pkg/msg/ServoCtrlMsg`, `angle`/`throttle` in [-1, 1])
and acts on them only after a two-step manual-mode service handshake
(`/ctrl_pkg/vehicle_state` with `state=1`, then `/ctrl_pkg/enable_state` with
`is_active=true`). `AckermannRosRobot` absorbs both differences:

    from strands_robots.mesh import AckermannRosRobot

    car = AckermannRosRobot.from_deepracer(node_name="deepracer")
    car.drive(linear=0.5, angular=1.0, duration=2.0)
    car.get_scan()

`drive()` keeps the same `(linear, angular)` contract as `RosBridgedRobot` -
a bicycle model (`atan(wheelbase * angular / linear)`, clamped to the steering
limit) converts to servo values internally. The handshake declared in
`init_services` runs once, automatically, before the first command; a failed
handshake aborts the drive. Timed and multi-message commands are always
followed by a zero servo message - even when the publish fails - so a timed
drive cannot leave the car with a live throttle, and a halt that itself fails is
reported: the call returns an error naming the throttle that may still be live
instead of the drive's success, so the agent's next action is `stop()`. A bare
single-shot `drive()`
(no `duration`) latches like any raw servo command until `stop()`. Commands
are clamped to `max_speed`; holds longer
than `max_duration` are rejected loudly rather than silently truncated. The
`linear`/`angular`/`duration`/`count` values themselves are checked against the
same shared domains the differential-drive bridges use, so an unusable value is
refused with identical text on every transport. A pair the steering geometry
cannot execute is refused for the same reason: below the rest threshold
(1e-3 m/s) the bicycle model maps any command to the zero servo pair, so
`drive(linear=0.0, angular=1.0)` - a rotate in place, which this platform cannot
do - would otherwise leave as byte-identical to `stop()` and report success for a
heading change that never happened. Give a turn a linear speed to travel at, or
call `stop()`; `drive(0, 0)` still means rest, because that is what it asked for. The
stock platform publishes no odometry, so there is deliberately no
`get_pose`.

Like `RosBridgedRobot`, the bridge inherits the [command
gate](#safety-critical-command-surfaces-need-operator-approval): the servo topic
(`/manual_drive`) and both mode services (`/vehicle_state`, `/enable_state`) are
blocklisted surfaces, so every command this bridge sends is gated.

| Method / tool | Reaches | Gated |
|---------------|---------|-------|
| `drive()` / `drive_<node>` | `publish` to the servo topic (plus the handshake on the first call) | yes |
| `stop()` / `stop_<node>` | `publish` to the servo topic | yes - the gate is keyed on the surface, not the payload |
| `enable()` | `service_call` to both mode services | yes |
| `get_scan()` / `get_scan_<node>` | `echo` | never gated |

The `drive_<node>` and `stop_<node>` agent tools forward the operator context, so
an agent driving the car prompts rather than failing closed. A programmatic
`car.drive(...)` / `car.stop()` has no operator to prompt, so pre-approve the
three surfaces for a headless run (bare names cover the namespaced DeepRacer
spellings):

```bash
export STRANDS_ROS2_COMMAND_ALLOW=/manual_drive,/vehicle_state,/enable_state
```

See `examples/ros2/deepracer_agent.py`.


## Sim bridge: publish a simulation on a ROS 2 domain

The simulator can advertise its own live state on ROS 2. Construct any
`SimEngine` (e.g. `Simulation()`) with `ros2_bridge=True` and it spins up an
internal `rclpy` node that publishes, per robot, after every `step()`:

| Topic | Type | Content |
|-------|------|---------|
| `/<robot>/joint_states` | `sensor_msgs/msg/JointState` | joint names + positions |
| `/<robot>/<camera>/image_raw` | `sensor_msgs/msg/Image` (`rgb8`) | one frame per attached camera. `<robot>`/`<camera>` are sanitised into ROS 2 name tokens, so a camera named `0` publishes on `/<robot>/camera_0/image_raw` - ROS 2 forbids a token starting with a digit |

`name` and `position` are one table read by index. A robot whose observation
does not carry every joint of `robot_joint_names()` - every floating-base
humanoid, quadruped and mobile base, whose root freejoint is joint 0 and is not
an observation key - publishes only the joints it observed, so the two arrays
stay the same joints. A caller that hands `publish_joint_states` a differing
number of names and positions has no pose to publish: the message is dropped
whole with a warning naming both counts, rather than published with every joint
after the gap under its neighbour's name.

```python
from strands_robots.simulation import Simulation

sim = Simulation(ros2_bridge=True, ros2_domain=0)
sim.create_world()
sim.add_robot("so101")
sim.step(10)   # publishes /so101/joint_states (+ camera image_raw) on domain 0
```

External ROS 2 nodes - and the agent's own `use_ros` calls - then see the
running simulation:

```bash
ros2 topic list | grep so101          # /so101/joint_states, /so101/<cam>/image_raw
ros2 topic echo /so101/joint_states   # live joint positions, updated every step
```

`rclpy` is an optional, system-provided dependency: it arrives with a sourced ROS
2 distro, not with the `[ros2]` extra (which installs only the cyclonedds RMW
binding, as above). When it is missing, `ros2_bridge=True` raises an `ImportError`
at construction naming the `source /opt/ros/<distro>/setup.bash` step that
supplies it; `ros2_bridge=False` (the default) never touches ROS 2, so the base
sim install stays lightweight. The bridge node is torn down cleanly on `destroy()`.

See `examples/ros2/sim_bridge_demo.py` for a runnable end-to-end script.

## Hardware bridge: publish a real robot on a ROS 2 domain

The hardware `Robot` is the symmetric counterpart of the sim bridge: construct
it with `ros2_bridge=True` and it owns a
`strands_robots.hardware_ros_bridge.HardwareRosBridge` that advertises the real
arm's live observation on a ROS 2 domain. The sim bridge
(`SimRosBridge`) and the hardware bridge (`HardwareRosBridge`) are thin
subclasses of the same `RosTelemetryBridge`, and the pure-RTPS transport
(`HardwareRtpsBridge`) shares the same wire contract through their common
`RosTelemetryBase`, so a physical arm and its digital twin publish **identical
topics** - a simulated robot and the real one it mirrors are indistinguishable on
the ROS 2 graph:

| Topic | Direction | Type | Content |
|-------|-----------|------|---------|
| `/<robot>/joint_states` | published | `sensor_msgs/msg/JointState` | joint names + positions, every control step |
| `/<robot>/<camera>/image_raw` | published | `sensor_msgs/msg/Image` (`rgb8`) | one frame per camera |
| `/<robot>/joint_command` | **subscribed** | `sensor_msgs/msg/JointState` | inbound `name`/`position` -> `send_action`, drives the real arm |

`name` and `position` are one table read by index. A robot whose observation
does not carry every joint of `robot_joint_names()` - every floating-base
humanoid, quadruped and mobile base, whose root freejoint is joint 0 and is not
an observation key - publishes only the joints it observed, so the two arrays
stay the same joints. A caller that hands `publish_joint_states` a differing
number of names and positions has no pose to publish: the message is dropped
whole with a warning naming both counts, rather than published with every joint
after the gap under its neighbour's name.

The first two are **outbound telemetry** (shared, byte-identical, with the sim
bridge). The third is the **inbound command** surface that makes the hardware
bridge *full duplex*: an external ROS 2 node (a teleop joystick node, MoveIt, a
trajectory replayer, or the agent's own `use_ros(action="publish", ...)`) can
publish a `JointState` to `/<robot>/joint_command` and the bridge forwards each
message straight into `Robot.send_action({motor.pos: float})`. Because the
command topic carries the *same* joint names the bridge publishes in
`joint_states`, a controller can echo our names straight back to drive the arm.
The sim sibling does not subscribe - a simulation is driven by its physics
engine; only the real arm is the thing on the graph an external controller can
physically move.

```python
from strands_robots import Robot

# Opt in to the bridge; the arm's observation is mirrored on ROS 2 domain 0.
arm = Robot("so101", mode="real", ros2_bridge=True, ros2_domain=0)

# Each control step of a running task publishes joint_states (+ camera frames).
# Or publish the current observation on demand without starting a task:
arm.publish_ros_observation()                 # joints + cameras
arm.publish_ros_observation(skip_images=True)  # joints only (opt out of cameras)

# Full duplex: with the default ros2_commands=True the bridge also subscribes to
# /so101/joint_command and forwards each message into Robot.send_action, so an
# external ROS 2 node can drive the real arm:
#
#   ros2 topic pub --once /so101/joint_command sensor_msgs/msg/JointState \
#     '{name: ["shoulder_pan.pos", "elbow.pos"], position: [0.1, -0.2]}'
#
# For a read-only telemetry bridge (no inbound control), opt out:
arm_ro = Robot("so101", mode="real", ros2_bridge=True, ros2_commands=False)

# rclpy-free: run the SAME bridge over pure cyclonedds (no sourced ROS 2
# distro). Byte-identical topics; type coverage bounded by the IDL bundle.
# Telemetry-only: on this transport the inbound command surface refuses to
# start without a dds_security_config or the explicit opt-out (see below).
arm_rtps = Robot("so101", mode="real", ros2_bridge=True, ros2_transport="rtps", ros2_commands=False)
```

External ROS 2 nodes - rviz, nav2, or the agent's own `use_ros` calls - then see
the physical robot as a live participant:

```bash
ros2 topic list | grep so101          # /so101/joint_states, /so101/<cam>/image_raw
ros2 topic echo /so101/joint_states   # live joint positions from the real arm
```

The bridge is **opt-in**: `ros2_bridge=False` (the default) never touches ROS 2,
so a robot only becomes a ROS 2 device when an operator explicitly asks for it -
the same safety stance as `Robot(mode="sim")` being the default. When `rclpy` is
missing, `ros2_bridge=True` raises an `ImportError` at construction naming both
routes forward: sourcing a ROS 2 distro, or `ros2_transport="rtps"`, which
publishes the same topics over the pip-installable cyclonedds binding and needs
no distro at all. The
inbound command path is on by default (`ros2_commands=True`); set
`ros2_commands=False` for a read-only telemetry bridge that publishes but cannot
be driven. Only a boolean names either posture - `ros2_bridge` and
`ros2_commands` are checked at construction, so a config that spells the flag
`"false"` is refused rather than reading as the truthy value it is and opening
the surface it asks to close. A daemon thread spins the node so inbound commands are serviced
concurrently with publishing, and it is torn down cleanly on `cleanup()`/`stop()`.
That teardown is best-effort: a node destroyed on a context another
component already shut down is reported at WARNING and `cleanup()` carries
on to disconnect the motors bus and the cameras, because a bridge that will
not release must not leave the serial port held or the arm energised. The same
rule holds one level in, where the bridge releases two things - its node handle
and, when it was this bridge that called `rclpy.init()`, the process-wide
context: a failure releasing one no longer skips the other, so a node that
refuses to be destroyed does not leave the participant on the domain for the
life of the process. A context that itself refuses to shut down is logged at
warning, because nothing after it retries.

Because the inbound `joint_command` topic drives the physical arm, two guards
harden it (both threaded through `Robot()`):

- `joint_limits={"<motor>.pos": (min, max)}` range-checks every inbound command;
  if any commanded joint is outside its declared range the **entire** command is
  rejected (no partial application). Keys are matched against the joint names
  the command carries - the same `<motor>.pos` names the bridge publishes in
  `joint_states` - so a key that names no commanded joint constrains nothing,
  and joints without a declared bound are unconstrained. Every bound must be a finite number - a non-finite one declares
  a range that admits nothing, so it is refused at construction. Available on
  both transports.
- For the pure-RTPS transport (`ros2_transport="rtps"`), a `dds_security_config`
  (or the explicit `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` opt-out) is
  **required** to expose the command surface - see the
  [RTPS integration guide](rtps-integration.md#securing-the-inbound-command-surface).
  rclpy DDS Security is configured at the RMW layer (`ROS_SECURITY_*` / `sros2`),
  not by a config dict.

See `examples/ros2/hardware_bridge_demo.py` for a runnable end-to-end script.

![Hardware ROS 2 bridge: an SO-101 camera frame published by HardwareRosBridge and received by an independent ros2 subscriber over DDS, byte-identical](assets/hardware_ros_bridge_proof.png)

The frame above was rendered for an SO-101, published on `/so101/wrist/image_raw` by `HardwareRosBridge` over real DDS, and decoded back by a separate `rclpy` subscriber - byte-identical round trip. On the same run `ros2 topic echo /so101/joint_states` returns the live joint vector, so the robot is a first-class ROS 2 device on the graph.

## Mesh bridge: a ROS 2 robot as a first-class strands Robot

`use_ros` is the low-level surface. For mobile bases that expose the usual
`cmd_vel` / odometry / scan topic trio, `RosBridgedRobot` wraps that wiring so a
remote ROS 2 robot drives like any other strands robot - the same
`Agent(tools=[robot])` pattern used for simulated and hardware arms.

```python
import os

from strands import Agent
from strands_robots.mesh import RosBridgedRobot

turtle = RosBridgedRobot.from_ros(
    node_name="turtlesim",
    cmd_vel_topic="/turtle1/cmd_vel",
    odom_topic="/turtle1/pose",
    odom_type="turtlesim/msg/Pose",  # optional; auto-resolved when omitted
)

# Direct, programmatic control. cmd_vel is a gated command surface and a script
# has no operator to prompt, so pre-approve the topics this process drives:
os.environ["STRANDS_ROS2_COMMAND_ALLOW"] = "/turtle1/cmd_vel"
turtle.drive(linear=1.0, duration=1.5)   # hold the command for 1.5 s
print(turtle.get_pose())                 # reads are never gated
turtle.stop()

# Or hand the robot to an agent - its capabilities become named tools
# (drive_turtlesim, stop_turtlesim, get_pose_turtlesim, ...):
agent = Agent(tools=turtle.tools)
agent("drive forward for two seconds, then tell me the pose")
```

The bridge is intentionally thin: every method forwards through the same
transport `use_ros` does (`strands_robots.ros`), so it inherits the same
in-process rclpy backend and its topic/type validation - and a `cmd_vel` command
reaches the shared operator gate whichever of the two asked, under one label: an
approval or a refusal means the same thing on both. The parameters the transport
never sees are checked by the bridge itself - `drive`
reports an error result without publishing when a velocity is not finite, a
`duration` is not positive and finite, or a message `count` is not a positive
whole number, and `publish_rate` is refused at construction. Construct it freely
without a ROS 2 environment present - errors surface only when a method is
actually called and `rclpy` is unavailable.

It also inherits the [command gate](#safety-critical-command-surfaces-need-operator-approval):
`cmd_vel` and a Nav2 `nav_action` are both blocklisted surfaces. The command
tools forward the operator context they are given, so an agent prompts; a
programmatic call needs `STRANDS_ROS2_COMMAND_ALLOW` (or
`BYPASS_TOOL_CONSENT=true`). That includes `stop()` - the gate is keyed on the
surface rather than the payload, because "zero is harmless" is true of a `Twist`
and false of `/joint_command`, where zero commands motion to the zero pose. An
unattended deployment that must always be able to halt should pre-approve its
`cmd_vel` topic.

| Method | ROS 2 action | Notes |
|--------|--------------|-------|
| `drive(linear, angular, duration=, count=)` | publish `Twist` to `cmd_vel_topic` | `duration` holds the command at `publish_rate` Hz; finite velocities, `duration > 0`, `count >= 1` - anything else is refused without publishing. Gated: needs an operator context or a pre-approved surface |
| `stop()` | publish zero `Twist` | Gated like `drive` - same surface, same verb |
| `navigate_to(x, y, yaw=, frame_id=, timeout=)` | `action_send_goal` to `nav_action` | error when no `nav_action` configured; finite pose components. Gated |
| `get_pose()` | echo `odom_topic` | never gated |
| `get_scan()` | echo `scan_topic` | error when no `scan_topic` configured; never gated |
| `.tools` | - | per-instance named agent tools; the command tools are `@tool(context=True)` so the gate can prompt |

### The shared mobile-base contract

`RosBridgedRobot` is a thin subclass of `MobileBaseRobot`, which owns the drive
contract, the safety semantics and the `tools` property for **every** mobile
robot in `strands_robots.mesh`. A robot class supplies only what actually
varies: a `Transport` (how bytes move) and, when the platform is not
differential-drive, a `_cmd_fields` override (what the command message looks
like).

Everything below therefore holds identically for any transport:

- Non-finite `linear` / `angular` / `duration` are refused. `nan` passes
  silently through a `min`/`max` clamp, so it has to be caught before clamping.
  The accepted domain is the shared one used by every other numeric knob in the
  package, so a velocity and a control-loop frequency agree on what a usable
  number is - a NumPy scalar from a policy action is accepted, a `bool` is not.
- `count` is the publish horizon when no `duration` is given, and must be a
  positive whole number. `count=0` would otherwise publish nothing and report
  success - a drive the caller believes happened. A `count` a call never reads
  (because `duration` supersedes it) is not refused.
- `duration` must be positive and finite, and within `max_duration` when the
  platform sets one. An over-long hold is refused, never silently truncated -
  and refused *before* any side effect, so an invalid request cannot be what
  arms a vehicle.
- Velocities are clamped to `max_linear` / `max_angular` when set. Left unset
  they mean "this platform declares no limit", not zero.
- Every timed or multi-message non-zero command is followed by a single zero
  command, through `try`/`finally`, **even when the publish raised**. A timed
  drive cannot leave a robot with a live velocity.
- A bare single-shot `drive()` latches until `stop()`, exactly like a raw
  `cmd_vel` publish. This is stated in the agent-facing tool description rather
  than hidden.
- `stop()` reaches the transport tool's command gate exactly as `drive()` does.
  The gate is keyed on the *surface*, and zero means "stationary" on a `Twist`
  but commands motion to the zero pose on a joint-command topic, so a
  payload-shaped carve-out could not be written correctly. What the halt does not
  depend on is the enable handshake or the speed limits: an emergency stop must
  not require a working service graph.
- Command tools are declared `@tool(context=True)` by the base and forward the
  injected operator context to the transport, which hands it to its own tool. A
  transport whose tool gates its command surface therefore prompts rather than
  failing closed. All three graph tools gate their command surface today, so no
  shipped transport is exempt - the rule is keyed on the tool rather than on a
  list so that a future ungated one is handled, not because an exemption exists.
  Because the tools are declared once, this holds for every transport rather
  than being wired per bridge.
- `init_services` declares an ordered enable/arm handshake that runs once before
  the first command. It does not latch on failure, so a retry re-runs it. It
  requires a transport that can call services, and is refused at construction on
  one that cannot.

Capabilities are reported, not assumed: `get_pose` appears only with an
`odom_topic`, `get_scan` only with a `scan_topic`, `navigate` only with a
`nav_action`, so an agent is never handed a tool that can only answer "not
configured". `robot.supports("service_call")` asks the transport directly.

See `examples/ros2/turtlebot_demo.py` for an end-to-end agent driving a turtle
in `turtlesim` through the mesh bridge.

![Agent driving a turtle via the ROS 2 mesh bridge](assets/ros2_mesh_bridge_turtle.gif)

The trail above is a turtle in `turtlesim` driven entirely through
`RosBridgedRobot.drive(...)` - the velocity commands are published over ROS 2 by
the mesh bridge, and the pose is read back through the same bridge.
