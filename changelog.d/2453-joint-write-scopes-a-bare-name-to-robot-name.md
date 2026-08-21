### Fixed

`set_joint_positions` / `set_joint_velocities` now resolve an unqualified joint
name inside `robot_name`'s namespace before the cross-robot fallback, so a write
lands on the robot the caller addressed. Two robots that each declare a joint
named `j1` compile to `alice/j1` and `bob/j1`, and a bare `{"j1": 0.9}` was
resolved by a lookup that tries every robot namespace in turn and returns the
first hit -- so `robot_name="bob"` moved `alice`, left `bob` at rest, and
reported `Set 2/2 joint positions` for the robot that never moved. Robots
sharing joint names is the ordinary case (two instances of one arm, or any two
`robot_descriptions` models that both call a joint `elbow`), and the docstring
already recommended the dict form as the safest in multi-robot scenes while
naming `robot_name` as what "resolves an unqualified joint name" -- so the
documented contract was the one that silently crossed robots.

The ordered (list) form shared the defect rather than escaping it.
`SimRobot.joint_names` holds *short* names, so the `dict(zip(joint_names,
values))` normalisation produced the same bare keys: measured, a 3-vector
addressed to `bob` wrote `alice/j1`, `alice/j2` and `bob/shoulder` -- bound to
bob's joint order, written to alice's addresses. Scoping in the shared joint-write
resolver repairs both forms at once.

No call that worked is now refused: a bare name that is not a joint of
`robot_name`, or a call that passes no `robot_name` at all, reaches the same
cross-robot lookup as before. Scoping is applied in the joint-write resolver
only; the shared `_resolve_mj_name` keeps its deliberate "unambiguous or
explicit" first-match contract for the read paths (`get_body_state`,
`get_jacobian`, sites, geoms and sensors) that share it. The dict-form docstring
now states the resolution order instead of an unqualified safety claim.
