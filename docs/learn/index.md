---
description: Each page answers one workflow question and ends with something that runs.
---

# Learn

## Agents

- [Agents](agents.md): a `Robot` as a Strands tool, the tools around it, the operator gate, what a refusal looks like.

## Policies, simulation, training

- [Policies](policies/index.md): the provider matrix and one page per provider.
- [Simulation](simulation/index.md): MuJoCo, Isaac, Newton, worlds and objects, predicates and rollouts, randomization.
- [Training](training/lerobot.md): LeRobot and [RL](training/rl.md).

## Data

- [Record](data/record.md), [Verify](data/verify.md), [Label and judge](data/label-and-judge.md), [Stream and sync](data/stream-and-sync.md): a dataset from the first frame to a filtered training set.

## Hardware

- [Drivers](hardware/drivers.md): the contract and the generated table of shipped drivers.
- [Teleoperation](hardware/teleoperation.md), [Cameras](hardware/cameras.md), [Calibration](hardware/calibration.md).
- Setup pages: [Feetech arms](hardware/feetech-arms.md), [Unitree](hardware/unitree.md), [Franka](hardware/franka.md), [UR](hardware/ur.md), [Reachy Mini](hardware/reachy-mini.md), [Microduck](hardware/microduck.md), [Booster T1](hardware/booster-t1.md).

## Mesh

- [Mesh](mesh/index.md): what it is and the three switches.
- [Fleet](mesh/fleet.md), [Safety and e-stop](mesh/safety-and-estop.md), [Topics](mesh/topics.md), [Bridges](mesh/bridges.md).

## Operate

- [Dashboard](dashboard.md): what `strands-robots dashboard` serves, who may click, the e-stop button.
- [ROS 2](ros2.md): three transports, when each, one gate.
- [Security](security.md): every control between a model and a motor, on one page.

Every runnable fence on these pages was executed against this commit. A fence marked `sketch` needs hardware, a GPU or an account, and says which.
