### Fixed: a run refused for recording no longer ramps the robot to the policy's `start_pose` first
`run(..., record=...)` built its episode after the start-pose ramp, so a recording refusal (`push=True` without an
`hf://` target, a non-integer `hz`, no LeRobot, or an existing dataset at another fps, robot or units) raised only
after the robot had moved to `policy.start_pose`. Every recording check now runs before the ramp: a run that cannot
be recorded never moves the robot.
