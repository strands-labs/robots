### `get_robot_state` says which way the arm extends

The `end_effector` line now carries the base position, the end effector's
offset from it and the horizontal world axis that offset lies along - `from
base [0, 0, 0]: [+0.02, -0.38, +0.26] (the arm currently extends along -Y)`
for a fresh SO-101 - and the `json` payload the same facts as
`end_effector.base`, `from_base` and `extends_along` (`null` when the arm is
over its base). Before, the position was the only clue to the robot's front:
an agent read "in front of the base" as +X and placed the cube beside an arm
whose whole reach lies along -Y (v0.5.2 devx replay). The base is measured -
the live floating-base pose, else the model's compiled root pose, the same
position `list_robots` reports - because `add_robot(position=...)` is an attach
frame MuJoCo composes with the model's own root pose: an arm authored at
`pos="0 -0.5 0.1"` and spawned at the origin extends +X from its base, and
measuring from the request called that -Y, the axis of the base's own offset.
A model with several root bodies has no one base pose, so its offset is from
the requested attach frame.
