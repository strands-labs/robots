### Quality: pin both deferrals on `pose_tool`'s incremental-move delta domain

`_joint_delta_error` returns `None` for a motor absent from
`_DEFAULT_MOTOR_CONFIGS`, because there is no configured travel to bound a
displacement against. That branch was unexecuted by the whole suite, leaving
`strands_robots/tools/pose_tool.py` at 99% on its one remaining statement. A
deferral is only sound if the thing it defers TO refuses, and nothing asserted
that half: a change making an unconfigured motor commandable through the delta
path would have left every test green.

The helper now documents the deferral its sibling `_joint_target_error` already
documents, and `tests/tools/test_pose_tool_target_domain.py` measures both
deferrals on that path -- an unknown motor, refused by the action's own position
read with no `Goal_Position` written, and a displacement inside the full travel
whose computed absolute target leaves the range, bounded by
`degrees_to_position`'s clamp.

The second measurement corrects a claim in the same test module, which said the
clamp "is unreachable from `move_motor` / `move_multiple` / `incremental_move`
now". It is unreachable from the first two, whose targets are absolute and are
held to the joint's endpoints, but not from `incremental_move`: a +300 deg
displacement is inside the 360 deg travel this domain checks, and from a joint
parked at +169.89 deg it computes +469.89 deg, which the clamp turns into
`Goal_Position` 4095 while the caller is told the move happened.

No library behaviour changes -- the only production edit is a docstring, and the
docstring-stripped AST digest of `pose_tool.py` is unchanged at `48bd01cc3adc195f`.
