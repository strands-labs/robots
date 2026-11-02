---
description: Robot.observe, Robot.act and Robot.run. One control loop for simulated and real robots.
---

# Observe, act and run

A `Robot` is one body in a world or on a bus. You read it with `observe()`, move it with `act()`, and run a
policy on it with `run()`.

{{drawing:d03_one_loop}}

```python
from strands_robots import Robot

arm = Robot("so101")
obs = arm.observe()                 # joints, pose, camera images
arm.act({"shoulder_pan": 0.3})      # one control step
arm.act({"gripper": 1.0}, seconds=1.0)  # move there over 1 s
res = arm.run(lambda obs: {"elbow_flex": 0.4}, hz=50, steps=60)
print(res.stopped_by)               # "steps"
```

`Robot(name)` joins the default world, or the world you pass as `sim=`. Use `position`, `orientation` and
`keyframe` to place it.

## act()

Pass a `{joint: target}` dict or a vector in `action_keys` order. Joints you leave out keep their last
target, so `act({})` holds the pose. Targets are clipped to the joint limits.

With `seconds=`, the robot moves smoothly from its current pose and returns the pose it reached.

| Method | Does |
|---|---|
| `hold()` | Stand still at the current pose |
| `set_pose(position, orientation)` | Teleport (simulation) |
| `set_gains(kp, kd)` | Set the PD gains |
| `close()` | Remove the robot from its world |

On a Feetech arm each write is capped at `max_speed` (default 2 rad/s). Pass `max_speed=None` to turn the cap off.

## run()

`run()` takes a `Policy`, any object with `act(obs)` and `reset()`, or a plain function. Stop it with one of:

| Argument | Stops after |
|---|---|
| `steps=` | That many control steps |
| `seconds=` | That much simulated time |
| `stop_when=` | Your function returns true |

Pass `realtime=True` to match the wall clock in simulation. Before the first step, `run()` names any camera the
policy needs and the robot lacks.

```python
task = arm.start(lambda obs: {"elbow_flex": 0.2}, seconds=30)  # in the background
res = arm.stop()                                               # returns the RunResult
```

To run several robots together, pass a dict: `arm.sim.run({arm: p1, g1: p2}, steps=20)`.

{{drawing:d08_policy_hooks}}

## Recording and replay

`run(record="data/wave")` adds the run as one episode of a LeRobot v3 dataset.
`Policy("data/wave?episode=0")` replays episode 0. See [Recording a dataset](recording.md).

{{drawing:d09_record_replay}}
