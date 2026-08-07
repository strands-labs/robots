### Fixed: a WBC per-call command override is validated on the domain the config enforces for the field it overrides

`WBCPolicy._resolve_command` builds the observation's command block from four
caller-supplied components, and only `target_velocity` was validated. The other
three are the per-call overrides of `height_cmd`, `freq_cmd` and `rpy_cmd` - the
three config fields `WBCConfig.__post_init__` does check - and each reached the
network raw, so the spelling documented to *win* the precedence contest accepted
values the spelling that loses it refuses.

`height=nan` and a non-finite `target_orientation` component put a non-finite
value in the frame the network is given; because the policy is dense that reaches
all 15 joint targets, `send_action` then refuses every one, and the rollout
aborted with *"100% unresolved keys ... the robot has not moved"* plus a list of
joint names - a report about the embodiment, produced by one bad `height`. A
numeric string raised from the `float()` that read it, and a non-numeric
`target_orientation` surfaced as NumPy's `could not convert string to float`,
which names neither the kwarg nor the policy.

`WBCGaitPolicy`'s step frequency had the same shape at all three of its sources.
`GaitClock.update` documents that `freq` "must be strictly positive" (it sets the
phase increment and the warm-up window `0.5 / freq`) and enforces it - but only
once the block has been built and handed over from inside `get_actions`, so the
message named `GaitClock.update` rather than the parameter supplied; and `True`
and `"0.75"` were not refused at all, they became a silent 1.0 Hz and 0.75 Hz.

Every source is now held to the domain the field's config spelling uses, and the
step frequency to the gait clock's stricter `> 0` rule at the per-call kwarg, the
constructor default and `config.freq_cmd`. The constructor check runs before the
ONNX session loads, so an unusable frequency is reported at construction instead
of on the first tick of a started rollout. `None` still means "not supplied" at
every override, and the signed quantities the config accepts (a zero or negative
height, a negative yaw target) remain first-class.
