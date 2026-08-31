### Added: `build_observation` routes slot two through an ONNX `gravity_source` flag

Pollen's reference `scripts/infer_policy.py` supports a binary switch at slot two
of the 61-D observation vector: `projected_gravity` (world `-Z` rotated into the
base frame from `base_quat`) or `raw_accel` (the accelerometer's `sensordata`
verbatim).  `self.use_projected_gravity` is a training-time flag baked into the
export.  `MicroduckPolicy` served only the projected-gravity branch on shipped
main, so an older Pollen export or a backlash-twin variant trained with
`use_projected_gravity=False` fed the graph a differently-scaled and
differently-signed 3-block: slot two kept the documented width and stayed
finite while its meaning did not match what the network was trained on, and
the drift was silent because nothing here reads a norm.

`build_observation` now takes a `gravity_source` keyword.
`"projected_gravity"` (default, every shipped alpha policy) reproduces the
pre-change vector byte-for-byte.  `"raw_accel"` reads `base_acc` (3, m/s^2)
verbatim into slot two, routed through the same `_require_base_block` reader
that already refuses a wrong-width `base_quat` at width 4, so a missing or
wrong-width `base_acc` refuses the way its sibling block does rather than a
half-width being silently taken.  Any other spelling raises, naming the two
shipped values, so a caller who mistypes `"gravity"` learns from the raise
rather than from drift on a running rollout.

`MicroduckPolicy._ensure_config` reads `gravity_source` off the ONNX
`custom_metadata_map` and threads it into every `get_actions` call.  The
resolution follows the same shape `joint_names`, `default_joint_pos`,
`action_scale` and `command_names` already take: the metadata wins when
present; the projected-gravity default holds when it is silent; and a
mistyped value raises at first-inference configuration rather than at the
slot-two read every tick.  Constructor-supplied gravity sources are not yet
plumbed - the metadata path serves the harness#388 acceptance criterion
alone, and the constructor path is a follow-up for the same reason
`command_names` waited for its use case.

The MuJoCo `_get_sim_observation` still publishes `base_quat` only, so today's
shipped alpha policies keep the same observation dict.  A caller building the
dict from a real IMU or teleop bridge - which
`wbc.policy._extract_state` documents - now has a matching key path for a
`raw_accel` export; the accelerometer sensor plumbing on the sim side is the
follow-up harness#388 flags for the sim path.

Closes cagataycali/robots-harness#388.
