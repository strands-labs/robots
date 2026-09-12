### Docs: recording documents the camera-name alphabet, and every documented name is graded against it

`add_camera` holds a camera's name to a bare token optionally scoped to one robot,
because the name is the key the camera's frames travel under. `docs/recording.md`
told the reader to rename a colliding camera without saying what a name may be, so
the rule was stated only in a refusal message. It now names both lists - the shapes
that work (`wrist`, `front_cam`, `cam-2`, `arm0/wrist_cam`) and the shapes that do
not (`a b`, `wrist.rgb`, `*`, `..`, `sub/../etc`, `a//b`) - where the reader picks a
name.

`tests/test_docs_camera_names_are_addressable.py` grades the names rather than the
prose, over the three populations that name a camera on the reader's behalf: the 13
`add_camera` names the documented examples claim, the 6 source keys in the
camera-naming translation table, and the 9 `obs_rename` keys in
`embodiments.json`. A `.` in an `obs_rename` key would make that embodiment
unsatisfiable from sim - its pre-flight check would name a camera the caller has no
way to create - so the coupling between the policy registry and the sim door is now
pinned rather than assumed.
