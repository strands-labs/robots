# AGENTS.md — strands-labs/robots

## Overview

`strands-robots` is a robot control library for [Strands Agents](https://strandsagents.com). It provides policy inference, teleoperation, calibration, and simulation tools for physical robots.

## Project Dashboard

**Board**: https://github.com/orgs/strands-labs/projects/2
**Project ID**: `PVT_kwDOD151Fs4BSRJP`

> **RULE**: ALWAYS use the project board to track work. When creating follow-up items,
> create GitHub issues and add them to this board with Status + Priority set.
> Never track work only in local markdown — the board is the source of truth.


All work is tracked here. When creating follow-up items, create GitHub issues and add them to the board with Status and Priority set.

## Repository Structure

```
strands_robots/
├── policies/              # Policy providers (pluggable via registry)
│   ├── base.py            # Abstract Policy base class
│   ├── factory.py         # create_policy() factory + registry
│   ├── mock.py            # MockPolicy for testing
│   ├── groot/             # NVIDIA GR00T N1.5/N1.6 inference
│   │   ├── policy.py      # Gr00tPolicy (ZMQ + HTTP modes)
│   │   ├── client.py      # Gr00tInferenceClient
│   │   ├── data_config.py # Gr00tDataConfig + ModalityConfig
│   │   └── data_configs.json  # 25 robot embodiment configs
│   └── lerobot_local/     # HuggingFace LeRobot direct inference
│       ├── policy.py      # LerobotLocalPolicy (RTC support)
│       ├── processor.py   # ProcessorBridge (pre/post pipelines)
│       └── resolution.py  # Policy class resolution (v0.4/v0.5)
├── registry/              # JSON registry for policy discovery
├── tools/                 # Strands @tool functions
│   ├── gr00t_inference.py # GR00T inference tool
│   ├── lerobot_calibrate.py
│   ├── lerobot_camera.py
│   ├── lerobot_teleoperate.py
│   ├── pose_tool.py
│   └── serial_tool.py
├── robot.py               # Core Robot class
└── utils.py               # Shared utilities (require_optional, etc.)

tests/                     # Unit tests (run with: hatch run test)
tests_integ/               # Integration tests (run with: hatch run test-integ)
```

## Development

```bash
# Install with all optional deps
pip install -e ".[all]"

# Run tests
hatch run test              # unit tests
hatch run test-integ        # integration tests (needs GPU + model weights)

# Lint & format
hatch run lint              # ruff check, ruff format --check, mypy
hatch run format            # ruff check --fix, ruff format
```

## Key Conventions

1. **Python 3.10+** — `requires-python = ">=3.10"` (lerobot extras require 3.12+)
2. **Dependency bounds** — `>=1.0` deps: cap major. `<1.0` deps: cap minor. E.g. `lerobot>=0.5.0,<0.6.0`
3. **`__init__.py` must be thin** — exports only, no logic
4. **Imports at file top** — unless lazy-loading heavy deps with documented reason
5. **Raise on fatal errors** — never warn-and-continue if the system will behave unexpectedly
6. **No silent defaults on error** — returning zero-valued actions on failure is forbidden
7. **Use `require_optional()`** — from `strands_robots/utils.py` for all optional deps
8. **Integration tests required** — each policy needs `tests_integ/` tests with real inference
9. **Test behavior, not implementation** — assert on outputs, not internal state
10. **No dead code** — if it's not called and not part of base class, delete it

## PR Workflow

1. Create feature branch from `main`
2. Make changes, run `hatch run format && hatch run lint && hatch run test`
3. All tests must pass, lint must be clean
4. Open PR from your fork, address all review comments
5. Track follow-up items as issues on the [project board](https://github.com/orgs/strands-labs/projects/2)
6. Squash merge into `main`
