### Changed: LeRobot installs from its PyPI release (`lerobot>=0.7.0,<0.8.0`) instead of its `main` branch

LeRobot 0.7.0 ships `remote_inference`, so the `lerobot` extra is now a normal wheel dependency
(`lerobot[remote,feetech,dataset]>=0.7.0,<0.8.0 ; python_version >= '3.12'`) rather than a hand-run
`git+https://github.com/huggingface/lerobot@main` line. `[all]` carries it, so `pip install 'strands-robots[all]'`
installs the policies, the SO arm bus and the `record=` dataset writer in one step; `remote` pins Zenoh. A separate
LeRobot install line is no longer printed anywhere. `import strands_robots` still imports no LeRobot.
