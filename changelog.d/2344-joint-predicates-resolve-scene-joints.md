### Fixed

`joint_above`, `joint_below` and `joint_progress` now resolve their joint
anywhere in the scene, not only on the robot the unscoped `get_observation`
happens to report. An articulated fixture -- a drawer slide, a door hinge -- is a
second entity alongside the arm being controlled, so its joint was previously
invisible under every spelling and the term degraded to a constant: the
predicate and its negation both answered `False`, and `joint_progress` returned
`0.0`, the maximum of a negative-distance reward. Drawer and door tasks, the case
`joint_progress` documents itself for, could not be scored.
