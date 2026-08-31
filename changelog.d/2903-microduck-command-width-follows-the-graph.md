### Fixed: seven of the nine shipped Microduck policies could not be stepped at all

`MicroduckPolicy` sized its command block by summing the `command_names` its
ONNX metadata declares. Every one of the nine policies Pollen ships takes
`obs [1, 61]`, but only `alpha_stand` and `alpha_walking` declare the full
`twist,head_pose,body_pose` (13 components). `roulade`, `roller`,
`roller_crouch`, `ball_kick_left`, `ball_kick_right` and `alpha_ground_pick`
declare `twist` (3), and `alpha_sitstand` declares `twist,head_pose` (7).

Summing those names built a 51- or 55-wide observation for a 61-wide graph, so
onnxruntime refused the very first inference:

```
InvalidArgument: [ONNXRuntimeError] : 2 : INVALID_ARGUMENT :
  Got invalid dimensions for input: obs for the following indices
  index: 1 Got: 51 Expected: 61
```

The two policies that ran are the two the documentation demonstrates, so the
gap was invisible: a bundle that advertised seven skills could step two.

`command_names` names which command slots a skill *reads*; it is not a width.
Pollen's reference runner emits one unified 13-component command for every skill
in a bundle and leaves the slots a skill ignores present and zero - the
dead-weight rule `microduck.observation.build_observation` already documents,
"unused command slots stay PRESENT and zero ... so one obs layout serves every
policy in a bundle" - and it reads the graph's own input shape to size that
vector.

The graph's declared input width is now the authority when it declares one, and
the width is derived rather than hardcoded (`6 + 3 * len(joint_names)` fixed
blocks, so another embodiment resolves its own command width). The
`command_names` sum remains the fallback for a session that declares no usable
shape, which is what keeps an injected stub - the seam tests use one, and it
describes an input name and no shape - resolving exactly as it did before.

Measured on all nine shipped exports driven through `run_policy` in MuJoCo at
50 Hz: nine of nine now step, where two of nine did before. `roulade` travels
0.51 m while dropping to a base height of 0.046 m, which is the roll that skill
performs; `alpha_walking` covers 1.02 m in 8 s and stays upright.
