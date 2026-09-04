### Fixed: a cached qpos-address map is no longer served to a different robot

`VeraPolicy._joint_qpos_addr` caches the `{state_key: qpos_index}` mapping that
the IK seed and the decoded IK targets both address through, and it was keyed on
`id(mj_model)` alone. A simulation binds one compiled world model to whichever
robot it is starting a rollout for, immediately after setting that robot's
`robot_state_keys`, so a second robot in the same scene arrives with new keys and
the identical model - and was served the previous robot's addresses.

Neither consumer could report it, because both skip a key the mapping does not
carry: the IK seed silently kept the model rest pose instead of the observed
joint configuration, and the decoded action dicts silently omitted every arm
joint. The mapping is now keyed on both inputs it is derived from. The model is
also held and compared by identity rather than by `id()`, which is unique only
while its object is alive, so a freed model's address can no longer be matched
by the model that replaces it.
