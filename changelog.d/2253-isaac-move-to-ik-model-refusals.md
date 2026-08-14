### Quality: pin every Isaac `move_to` IK-model resolution refusal

`IsaacMotionPrimitivesMixin.move_to` refuses six distinct ways before it commands
anything - the MuJoCo+mink stack is absent, the data_config MJCF does not compile,
no end-effector frame can be discovered, the model has no non-gripper joint to
solve, the registry gripper metadata is malformed, and the articulation cannot
report a usable joint-position vector (before the solve or mid-run). Every one of
those refusal reports was unexecuted: the module that owns the IK contract builds
its engine through a real MuJoCo sim, so its cases land on the MuJoCo backend and
cannot reach the Isaac ones. This adds 30 cases driving all eight scenarios
through the real `move_to`, asserting the reason names the data_config or the
articulation, that the refusals stay pairwise distinguishable, and that nothing
is commanded to the articulation on any of them.
