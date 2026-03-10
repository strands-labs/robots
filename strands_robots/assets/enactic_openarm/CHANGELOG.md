# OpenArm MuJoCo Model

## Source
- **Project**: [OpenArm](https://github.com/enactic/openarm) by Enactic, Inc.
- **License**: Apache-2.0
- **URDF Source**: [openarm_description](https://github.com/enactic/openarm_description) (CERN-OHL-S-2.0)
- **Isaac Lab**: [openarm_isaac_lab](https://github.com/enactic/openarm_isaac_lab) (Apache-2.0)
- **Hardware Control**: [openarm_can](https://github.com/enactic/openarm_can) via CAN bus + DAMIAO motors

## Robot Specifications
- **Type**: 7-DOF humanoid arm + parallel-jaw gripper
- **DOF**: 7 arm joints + 2 gripper fingers = 9 total actuators
- **Motors**: DAMIAO DM4310 (shoulder/elbow), DM4310 (wrist)
- **Communication**: CAN bus (SocketCAN)
- **Weight**: ~5.8 kg per arm
- **Payload**: Practical manipulation payloads
- **Key Feature**: High backdrivability and compliance for safe HRI
- **Price**: ~$6,500 USD for complete bimanual system

## Joint Configuration (from openarm_description v10)
| Joint   | Axis  | Range (rad)         | Effort (Nm) | Velocity (rad/s) |
|---------|-------|---------------------|-------------|-------------------|
| joint1  | Z     | [-1.396, 3.491]     | 40          | 16.75             |
| joint2  | -X    | [-1.745, 1.745]     | 40          | 16.75             |
| joint3  | Z     | [-1.571, 1.571]     | 27          | 5.45              |
| joint4  | Y     | [0.0, 2.444]        | 27          | 5.45              |
| joint5  | Z     | [-1.571, 1.571]     | 7           | 20.94             |
| joint6  | X     | [-0.785, 0.785]     | 7           | 20.94             |
| joint7  | Y     | [-1.571, 1.571]     | 7           | 20.94             |
| gripper | slide | [0, 0.044]          | —           | —                 |

## Inertials
All inertial parameters taken directly from openarm_description v10 config.

## MuJoCo Model Notes
- Derived from URDF/xacro + YAML configs (not auto-converted — hand-tuned)
- Primitive collision geometries (capsules/boxes) for fast simulation
- Position-controlled actuators matching DAMIAO motor characteristics
- Home keyframe: arm in standard reach pose (joint1=π/2, joint3=-π/2, joint4=π/2)

## Isaac Lab Integration
OpenArm has official Isaac Lab tasks (from openarm_isaac_lab):
- Unimanual reach
- Unimanual lift
- Unimanual cabinet opening
- Bimanual reach

## Changelog
- 2026-03-01: Initial MuJoCo MJCF model for strands-robots integration
