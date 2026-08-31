### Fixed: a Microduck command override the method cannot honor is refused rather than absorbed

`MicroduckPolicy.get_actions` recognises two per-call overrides that write the
command vector - `command` (wholesale) and `target_velocity` (the twist slots) -
and neither was held to a value domain, while two sibling providers reading the
same well-known goal key already are (`WBCPolicy._validate_velocity`, and the
`param_name="target_velocity"` guard MotionBricks applies on both its constructor
and its per-call path).

`target_velocity` was written as `self._command[:n] = tv[:n]` with
`n = min(3, len(tv), len(command))`, so a component count the method does not
document was silently absorbed. A longer vector lost its tail. A shorter one
wrote only the slots it covered, and this policy's command vector persists across
ticks, so `target_velocity=[0.3]` left the previous tick's lateral and yaw
components commanding the robot under a reported success; a bare scalar was read
as one component the same way.

A non-finite component in either override was assigned first and refused
afterwards by `build_observation`, which names `command` and the assembled
observation rather than the parameter the caller passed - and by then the
poisoned vector was already in `self._command`, so a caller that handled that
error and kept ticking carried it into every later tick and every later episode.
A non-numeric `command` reached `np.asarray(..., dtype=np.float32)` and surfaced
as a bare "could not convert string to float", naming neither the policy nor the
parameter.

Both overrides now consult the shared `finite_vector_error` under the caller's
own parameter name, before the write, and `target_velocity` is held to the two
component counts the method documents (`TARGET_VELOCITY_WIDTHS`). The documented
two-component spelling is kept: because the command persists, "set vx and vy,
leave omega" is a coherent request - which is why the family's other readers,
whose command is rebuilt per call, require three. The provider documentation now
also states that what the twist slots MEAN is a property of the loaded weights
rather than of this provider: Pollen's locomotion exports read them as a
velocity, which is what `target_velocity` writes, while other exports in the
family read the same three slots under a different convention that the ONNX
metadata does not distinguish.
