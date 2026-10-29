### Fixed: `sim.pose("<robot>/<link>")` answers where a robot link is, the end effector included

`sim.pose("so101/gripper")` now returns the link's live world pose (position m, quaternion w,x,y,z), for every
name `describe()["robots"][<id>]["links"]` lists, the same `"<robot>/<link>"` form `camera(attach=)` takes. The
agent's `sim_world` tool answers `{"rpc": "pose", "pose": {"body": "so101/gripper"}}` the same way. Before, `pose()`
refused every link (`KeyError: no body 'so101/gripper'`), so neither Python nor an agent could read where the gripper
was. A bare link name (`"gripper"`) is still refused, and the error now names the qualified form to use.
