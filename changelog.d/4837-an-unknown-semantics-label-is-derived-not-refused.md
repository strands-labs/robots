### Changed: a LeRobot semantics label the table does not know is derived, not refused

LeRobot keeps no registry of `semantics` strings — every server names its own (`so101-normalized-v1`, `lab-so101-deg-v3`,
`furlongs-v1`) — and by the time units are resolved the deployment has already been checked joint by joint against this
robot. So an unknown label is now read as words: what it says about the body's mode (`degrees`; `range` / `normalized`;
`radians`) overrides the robot's own LeRobot convention (`hardware.lerobot_norm` in the registry — the modes its
calibration JSON is written for), the gripper keeps that convention (`gripper-percent` makes it explicit), and a
`UserWarning` says exactly what was assumed and how to make it data (a row in `units/semantics.json`). The refusal
remains for the one case nobody can answer: an unknown label on a robot with no LeRobot convention, naming both.
Before, `Policy("zenoh://…")` was refused at `attach()` for every label outside the table, and the remote path had no
`units=` to follow the message's own advice.
