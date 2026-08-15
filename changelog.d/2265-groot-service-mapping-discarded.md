### Fixed: a GR00T service-mode policy now honours the observation and action mappings it accepts

`Gr00tPolicy(observation_mapping=..., action_mapping=...)` is the documented
escape hatch for a robot whose observation keys do not match the conventional
patterns automatic inference recognises. Both mappings were stored raw at
construction and parsed only after a **local** model load, so a service-mode
policy (`host=`/`port=`) discarded both: the request went to the inference
server carrying the task string and **no video and no state at all**, and the
returned action chunk kept the model's bare key names.

Measured against a server whose keys are `video.wrist` / `state.arm` and a robot
whose own keys are `wrist_cam` / `arm_joints` -- the case the mapping exists for,
since auto-inference could never bridge those names:

| supplying the mapping | video keys on the wire | state keys on the wire | action keys returned |
| --- | --- | --- | --- |
| before | 0 | 0 | `arm` |
| after | 1 (`wrist`) | 1 (`arm`) | `joints` |

Nothing reported the loss. Construction returned normally, and the two service
consumers disagreed about the unparsed value in ways that were each individually
quiet: the observation half fell through to the flat legacy payload builder,
while the action half was skipped by a truthiness guard rather than an assertion.
A caller who read the documentation, supplied the mapping correctly and got a
successful call had no channel that said the observation had been dropped.

Both parsers are pure over the caller's own flat dict -- the video/state split
and the action renaming come from the `video.` / `state.` / `action.` prefixes of
its values -- so neither ever needed the model, and both now run in either mode.
Two consequences follow. A malformed mapping value is refused at construction in
service mode with the message local mode has always produced, instead of being
silently ignored. And a local policy whose modality configs cannot be read now
honours a supplied mapping too, rather than dropping it and failing later on an
assertion.

What still requires the model is everything that *cross-checks* a mapping
against it: `validate()`, state-DOF discovery, and the auto-inference used when a
mapping is omitted. So a service-mode mapping is honoured as written and a key
the server does not have surfaces as a server-side error rather than a
constructor refusal, and an omitted mapping deliberately stays unset rather than
acquiring an inferred mapping that could not be validated -- which is what keeps
the no-mapping service path behaving exactly as before.

The instruction stays addressed to the key the data config declares, which is the
key the unmapped service path already sent, so enabling a mapping moves the video
and state onto the wire without relocating the language input.
