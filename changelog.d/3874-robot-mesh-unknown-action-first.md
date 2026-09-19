### Fixed: `robot_mesh` refuses an unknown action before it touches the mesh

A typo'd action such as `robot_mesh(action="warp")` used to be graded last: in a
process with no `Robot()` the tool recorded a rate-limit slot, probed Device
Connect, brought up a gateway mesh and waited one heartbeat for it, then
answered "no local mesh found. Construct a Robot()/Simulation() first" - a
remedy for a different problem, with two `Mesh did NOT start` errors in the
log. The action name is now checked against the tool's action list before any
other step, so the reply names the unknown action and lists the valid ones, and
nothing is built or recorded on its behalf.
