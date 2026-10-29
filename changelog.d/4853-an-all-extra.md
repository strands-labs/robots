### Added: `[all]` installs every extra a CPU machine can use

`uv pip install -e ".[all]"` now installs mujoco, feetech, onnx, lerobot (zenoh), tools, mesh, iot and dashboard.
Before, `[all]` did not exist, so the line most people type installed the core alone (pip only warns) and the first
agent tool or dashboard failed on a missing module. The GPU extras (`isaac`, `newton`) and LeRobot itself (a git
commit) stay their own lines. README, the quickstart and both notebook pages print the same install.
