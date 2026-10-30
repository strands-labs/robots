### Fixed: a recorded episode replays in the norm mode each joint was recorded in

`meta/units.json` named the unit scheme, `"lerobot"`, but no joint's norm mode, so `Policy("<dataset dir>")`
rebuilt the registry's range units even for a run recorded under a declared degrees `norm=`: on the repo's fake
Feetech bus a +45 degree run replayed on a hardware SO-101 landed up to 36 degrees off (wrist_roll), and so did a
dataset without `units.json` replayed with `units="lerobot"`, as the error advised. `record=` now writes each joint's
LeRobot norm mode into `meta/units.json`, and the replay rebuilds exactly those units. A dataset that names no modes
(recorded before this change, or without `units.json`) takes them from the new `norm=` keyword (a key that names no recorded joint is refused); without it a hardware
`Robot`, a `Mesh.connect()` robot included, refuses the replay before anything reaches the bus, and a simulated robot
keeps the registry's modes.
