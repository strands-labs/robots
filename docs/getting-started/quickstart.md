# Quickstart

!!! info "Coming Soon"
    This page is under active development.

Get up and running with Strands Robots in 2 minutes.

## Install

```bash
pip install strands-robots
```

## Your First Robot

```python
from strands import Agent
from strands_robots import Robot

robot = Robot(
    tool_name="my_arm",
    robot="so101_follower",
    port="/dev/ttyACM0"
)

agent = Agent(tools=[robot])
agent("Move to home position")
```
