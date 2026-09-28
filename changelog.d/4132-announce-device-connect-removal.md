### Deprecated: Device Connect is removed in 0.7

`Robot(...).run()` and every public name in `strands_robots.device_connect`
(`init_device_connect`, `init_device_connect_sync`, `RobotDeviceDriver`,
`SimulationDeviceDriver`, `ReachyMiniDriver`, also reached through
`strands_robots`) now raise a `DeprecationWarning` attributed to the caller's
line. The `robot_mesh` tool's `rpc` action and the `[device-connect]` extra
leave with them.

| removed in 0.7 | use instead |
|---|---|
| `Robot(...).run()` | `Robot(..., mesh=True)`, which serves the robot as a mesh peer |
| `init_device_connect*`, the three Device Connect drivers | the mesh (`strands_robots.mesh.init_mesh`) |
| `robot_mesh(action="rpc")` | `robot_mesh` `tell` / `send` over the mesh |

Nothing else changes in 0.6: each still runs.
