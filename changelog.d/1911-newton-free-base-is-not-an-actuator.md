### Fixed: a Newton floating base is no longer advertised as an actuator

`NewtonSimEngine` keeps one joint list per robot and handed it out as the
*action* vocabulary unfiltered, so a floating base's 6-DoF free joint appeared
among the keys `send_action` accepts. It is not a commandable scalar - its
coordinates are `[xyz, quat_xyzw]`, which is why `get_observation`,
`get_robot_state` and the recording schema all skip it and surface the base as
the structured `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel`
signals instead. `robot_action_keys` was simply never overridden, so it
inherited the base class default that mirrors the joint names.

Four surfaces read that list and none of them said anything:

* `send_action({"floating_base_joint": 0.5})` returned `status="success"` and
  wrote a scalar position target for a 6-DoF joint.
* the vector form of `send_action` refused an action of the width a recording
  actually holds, because the recorded columns exclude the free joint - the two
  counts differed by one.
* the dataset schema never declared a column for it, so a value supplied for it
  was dropped.
* `PolicyRunner.replay` binds `robot_action_keys` against the recorded action
  vector, so it bound one key more than the vector carried and aborted the
  episode on frame 0. A floating-base Newton recording could be written but
  never replayed - the locomotion and whole-body case the base state columns
  exist for.

`NewtonSimEngine` now overrides `robot_action_keys` to exclude the free root,
and `send_action` resolves the keys it accepts through that same list instead of
the raw joint names, so a scalar target on the free joint is refused with the
existing `unresolved_keys` envelope rather than written. The declared recording
columns and the action keys are now equal by construction, which is the property
replay depends on, and it is pinned as an equality between the two producers
rather than left to the fallback happening to agree.

A fixed-base robot has no free root, so its two vocabularies already agreed and
are unchanged. `robot_joint_names` still reports the free joint: it has the same
disagreement on the state side, but it also sizes the RL trainers' action
dimension and names policy state keys, so it is tracked separately rather than
narrowed here.
