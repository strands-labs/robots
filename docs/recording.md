# Recording & Datasets

!!! info "Coming Soon"
    This page is under active development.

Record robot demonstrations and manage datasets for training.

## Recording a Dataset

```python
from strands_robots.tools import lerobot_teleoperate

# Start teleoperation recording
lerobot_teleoperate(
    action="start",
    robot_type="so101_follower",
    teleop_type="so101_leader"
)
```
