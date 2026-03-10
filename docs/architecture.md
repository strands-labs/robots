# Architecture

!!! info "Coming Soon"
    This page is under active development.

## Project Structure

```
strands-robots/
├── strands_robots/
│   ├── __init__.py              # Package exports
│   ├── robot.py                 # Universal Robot class (AgentTool)
│   ├── policies/
│   │   ├── __init__.py          # Policy ABC + factory
│   │   └── groot/               # GR00T policy implementation
│   └── tools/
│       ├── gr00t_inference.py   # Docker service manager
│       ├── lerobot_camera.py    # Camera operations
│       ├── lerobot_calibrate.py # Calibration management
│       ├── lerobot_teleoperate.py # Recording/replay
│       ├── pose_tool.py         # Pose management
│       └── serial_tool.py       # Serial communication
├── tests/
└── pyproject.toml
```
