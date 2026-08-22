### Fixed

- **A policy rollout now shares the motor bus with the mesh's readers instead of racing them.**
  `strands_robots/bus_access.py` puts one `RLock` on the device so every reader and writer that goes
  through it takes the serial motor bus in turn. The mesh modules were converted to it;
  `hardware_robot.py` was not, and imported it nowhere -- it drove the same device directly at five
  call sites (the ROS 2 telemetry publish, the policy preflight read, the rollout loop's read and
  write, and the `send_action` facade that teleop and inbound ROS 2 commands both arrive through).
  So the lock serialised the mesh's readers against each other and not against a rollout, while one
  process holds both; `hardware_robot`'s own `_task_admission` lock admits one *rollout* at a time and
  knows nothing about the bus. The failure was asymmetric, and the caller that lost was the one
  holding nothing: measured against the real `run_policy` loop with a mesh reader on the same device,
  a single refused read ended the rollout with `status="error"`, 0 of 20 steps and 0 commands on the
  wire, reported as `Failed to initialize policy` -- a policy fault for a run that never commanded the
  arm once -- while the mesh reader beside it completed 54 reads and logged nothing. The same
  contention dropped 3 of 3 arm commands sent through the `send_action` facade. All five sites now go
  through `bus_access`, and the rollout completes 20 of 20 steps with 20 commands delivered and no
  refusals. `connect`/`disconnect` are unchanged: `bus_access` wraps reads and writes, and the
  operation set the new rule guards is derived from what that module itself drives, so opening the
  port falls outside it by construction rather than by an exemption.
