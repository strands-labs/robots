---
description: 23 single-arm manipulators - from a 2-DOF educational toy to industrial UR10e.
---

# Arms

Single-arm manipulators: industrial robots, research arms, educational kits.
**23 robots in this category.**

```python
from strands_robots import Robot
sim = Robot("panda")            # Franka Emika Panda
sim = Robot("ur5e")             # Universal Robots UR5e
sim = Robot("so100")            # SO-ARM100 (low-cost Feetech)
```

## Catalog

Every robot in this family, generated from `robots.json` at build time. Renders are MuJoCo sim renders, never hardware photos.

{{robot_cards:arm}}

## Universal Robots over RTDE

`ur5e` and `ur10e` have a native driver, so an e-Series arm is driven directly by its
controller's Real-Time Data Exchange interface rather than through lerobot (which
registers no UR robot type):

```python
from strands_robots import Robot

arm = Robot("ur5e", mode="real", driver="strands", port="192.168.1.10")
arm.connect_eagerly()                      # refuses a controller that cannot move
arm.state()                                # joints, TCP pose, TCP wrench
arm.send_action({"elbow_joint": 1.40})     # one servoJ setpoint, radians
arm.run_policy(policy, n_steps=500)        # streamed rollout at control_frequency
```

Needs the SDK: `pip install 'strands-robots[ur]'`, which declares the `ur_rtde`
build the driver's two interfaces come from. `port=` is the controller's address; the RTDE
port is fixed at 30004 by the protocol, so a different suffix is refused rather than
dialled.

Two gates stand in front of every write, because a UR controller does not reject a bad
command the way a servo bus does - it accepts the register and performs nothing:

- **Controller mode.** A robot mode other than `RUNNING`, or a safety mode outside
  `NORMAL`/`REDUCED`, is refused in the controller's own vocabulary (`PROTECTIVE_STOP`,
  `SAFEGUARD_STOP`). The mode is re-read per write, so a stop landing mid-rollout ends
  the rollout with that reason.
- **Commanded speed.** A joint asked to move further than the model's datasheet ceiling
  allows in one control period is refused, naming the joint and both figures. The
  ceilings differ per model - every UR5e joint reaches 180 deg/s where the UR10e's three
  proximal joints are held to 120 deg/s - so the same policy cadence can be admitted on
  one arm and refused on the other.

Reads are held to one further rule, and it applies to every surface rather than only to a
write: each RTDE vector is named by its position against the arm's six joints, so a
controller answering a different width is refused by name - `state()` and the mesh joint
read included - instead of being reported as far as it goes. A seven-axis answer is the
case that makes it load-bearing: naming its first six elements produces a pose that reads
exactly like a genuine six-axis one. This driver serves six-axis e-Series arms only.

Stopping a rollout is reported rather than asserted. `stop_task()` signals the loop,
waits up to two seconds for its thread and decelerates the arm with `servoStop`; a
policy blocking on a remote inference call outlasts that budget, and the envelope then
carries `status="error"` with `stopped=False` and a reason naming the timeout, matching
what `get_task_status()` says about the same loop. The arm is decelerated either way,
and no further setpoint reaches the controller. That takes two re-reads rather than one:
the loop re-reads the stop signal after the policy returns, and `send_action` re-reads the
driver's halt counter immediately before `servoJ`. Between those two it reads both mode
registers and the measured pose - three RTDE round trips to the same controller, during
which a halt was otherwise answered by one more setpoint. `stop()` carries no verdict (the
driver protocol annotates it `-> None`); read `stop_task()` when the outcome matters.

`start_task()` is the one verb in the fleet that builds the policy for you, from the
provider registry, and a provider it cannot build is refused rather than raised: the verb
is reached as an agent tool, where an exception is not something the caller can handle.
The refusal names the provider and carries the build's own reason, so a remote-code
provider reports the `STRANDS_TRUST_REMOTE_CODE` opt-in it wants and a mistyped
`checkpoint_dir` reports the path. Hold a built policy and `run_policy()` skips the build
entirely.

Joint keys are the arm's own names, in RTDE wire order, and the MuJoCo assets declare
them identically - so an action dict recorded in simulation streams to the controller
with no remap:

![UR5e servoJ rollout](../assets/ur/ur5e_servoj_rollout.gif){ width=420 }

_540 servoJ setpoints from `URDriver.send_action` driving the `ur5e` MuJoCo model at
50 Hz, headless._

![UR5e commanded steps against the speed ceiling](../assets/ur/ur5e_servoj_gate.png){ width=640 }

_Top: the setpoints the controller received (solid) and the arm's response (dotted).
Bottom: every commanded step against the model ceiling._

## Compatibility notes

- Most arms are loadable in MuJoCo via the registry's asset block and pull from
  [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py)
  on first use. Exceptions: `hope_jr`, `omx` and `rebot_b601` declare no sim asset
  and require physical hardware.
- Real hardware through LeRobot, where the registry entry names a `lerobot_type`:
  `hope_jr`, `koch`, `omx`, `openarm`, `rebot_b601`, `so100`, `so101`.
- Real hardware through a native Strands driver, selected with `driver="strands"`:
  `dynamixel_2r`, `fr3`, `fr3_v2`, `hope_jr`, `koch`, `panda`, `so100`, `so101`,
  `ur10e`, `ur5e`, `vx300s`, `wx250s`.
- Every other arm is simulation-only: `Robot(name, mode="real")` refuses it and names
  the robots that do have a path, rather than falling back to sim.
- The Franka arms (`panda`, `fr3`, `fr3_v2`) are driven over the Franka Control
  Interface, which needs the control box's address and the `panda-py` binding over
  libfranka (`pip install panda-py`):

    ```python
    arm = Robot("panda", mode="real", driver="strands", port="172.16.0.2")
    arm.connect_eagerly()                     # returns None, or a reason
    arm.send_action({**dict(zip(arm.joint_names, targets)), "gripper_width": 0.04})
    ```

    Read `arm.joint_names` rather than assuming them: each Franka's joints are named
    the way *its own* MuJoCo model names them, so a Panda's are `joint1..joint7`
    while an FR3's are `fr3_joint1..fr3_joint7`. That is what lets one action dict
    drive the simulated arm and the real one:

    ![panda driven by the driver's own action dict](../assets/franka/franka_sim_to_real.gif){ width=400 }

    _The same dict, keyed by `arm.joint_names` and passed through the driver's own
    `action_to_targets` gate, stepping the simulated `panda`._

    A `send_action` that reports success means the arm reached the configuration
    it was given. `panda-py` runs the trajectory on its own realtime thread and
    reports the outcome as a return value rather than by raising, so a reflex
    stop, an out-of-limit target, or a motion that simply ended short of the goal
    all come back as an error envelope carrying libfranka's own message - not as
    a success naming joints the arm is not holding.

    `arm.stop()` preempts a motion in flight. It goes through libfranka's own
    `Robot::stop()`, which is designed to abort a running control loop from
    another thread, so it does not wait for the motion it was asked to interrupt;
    the Franka Hand is halted with the arm. Telemetry keeps answering throughout,
    so `read_state()` on another thread is not blanked for the duration of a
    motion.
- Joint counts include any free joints / gripper actuators - the *control* DOF is
  usually `joints - 1` for arms with grippers.

## Reading a real arm before moving it

A real arm handed to an agent can be *looked at* without running a policy on
it. `Robot(name, mode="real", port=...)` offers four observe actions that write
nothing to the servos and never ask the operator:

| action | what comes back |
|---|---|
| `get_state` (alias `get_robot_state`) | every joint: degrees, raw encoder ticks, torque on/off, supply voltage; whether the arm is calibrated |
| `list_cameras` | the cameras passed as `cameras=` and whether each is open |
| `render` | one PNG from a camera (`camera_name`, optional `output_path` inside `~/.strands_robots/renders`) |

```python
from strands import Agent
from strands_robots import Robot

arm = Robot("so101", mode="real", port="/dev/ttyACM0",
            cameras={"front": {"type": "opencv", "index_or_path": 0, "width": 1280, "height": 720, "fps": 30}})
agent = Agent(tools=[arm])
agent("Where is the arm right now, and is torque on? Do not move it.")   # get_state, no approval
```

`get_state` opens the motor bus on first use (the bus only - the servo
configuration a rollout writes is not touched) and reads it under the same lock
a rollout and the mesh probes use. An arm with **no calibration file** is still
readable: the degrees are then an estimate from the encoder centre
(`2048 ticks = 0°`, `4096 ticks/rev`) and the text says so, naming
`lerobot-calibrate` - which is also why `execute`/`start` refuse on that arm
until it has run. A reading the arm did *not* give is reported as unread rather
than as its opposite: a calibration flag whose own read fails (on a lerobot bus
that read sweeps every servo) comes back `null` with the reason instead of
`false`, and the degrees stay whatever the arm could normalise; a
`Torque_Enable` register no motor answered is `null`, not `off`, because "torque
off" reads as "safe to move by hand". The motion actions (`execute`, `start`)
stop for operator approval as before; see
[security](../security.md#ros-2-dds-bridge-command-surface).

## Calibrating a Feetech SO arm

`so100`, `so101` and `lekiwi` read and command **degrees**, and those degrees are
measured against the travel `lerobot-calibrate` recorded for *that particular
arm*. Pass the file that run wrote:

```python
from strands_robots.drivers.feetech import FeetechDriver, lerobot_calibration_path

arm = FeetechDriver(
    tool_name="so101",
    port="/dev/ttyACM0",
    calibration=lerobot_calibration_path("so101_follower", "my_arm"),
)
arm.connect_eagerly()                       # returns None, or a reason
arm.send_action({"shoulder_pan": 30.0, "gripper": 100.0})
```

`lerobot_calibration_path(robot_type, robot_id)` is where
`lerobot-calibrate --robot.type=so101_follower --robot.id=my_arm` put its output,
read from LeRobot's own constants so `HF_LEROBOT_CALIBRATION` is honoured.
Records can also be passed directly (`calibration=load_calibration(path)`), and
`get_status()` reports which travel is in force as `calibration_source`.

Omitting it spans the *servo's* full rotation instead of the arm's measured
travel. No two SO-101s stop in the same place, so `0 degrees` and
`0 percent closed` then land somewhere different on each one - the degrees are an
encoder angle rather than a joint angle. Calibrate the arm and pass the file.

## See also

- [Robot factory](../getting-started/robot-factory.md) - how `Robot("name")` resolves
  these names.
- [Bimanual](bimanual.md) - two-arm setups (Aloha, Trossen WX-AI).
- [Hands](hands.md) - pair an arm with a dexterous end-effector.
- [Quickstart](../getting-started/quickstart.md) - spawn one of these arms in 3 lines.
