### Fixed: `g1_battery` and `g1_mainboard` refuse a wrong driver handle instead of raising

`snapshot_handle_refusal` is the shared guard every G1 sensor verb owes its
caller: the driver handle is a live Python object typed `Any`, so the annotation
carries no type into the generated tool schema and a caller reaches these verbs
with whatever it has. Its own docstring says it is "shared because five verbs
need it rather than restated in each" -- and three had it. `g1_battery` and
`g1_mainboard` called `driver._snapshot(...)` as their first statement, so an
unusable handle surfaced as an `AttributeError` naming a private attribute
rather than as a refusal naming the parameter.

Measured across the five live-handle verbs the package exposes, before this
change:

| verb | `None` | a robot name | `{}` | `7` | `[]` |
| --- | --- | --- | --- | --- | --- |
| `g1_battery` | raises | raises | raises | raises | raises |
| `g1_mainboard` | raises | raises | raises | raises | raises |
| `g1_imu` | refused | refused | refused | refused | refused |
| `g1_lidar_state` | refused | refused | refused | refused | refused |
| `g1_lidar_summary` | refused | refused | refused | refused | refused |

All five refuse after it. `tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py`
goes from 8 failed / 10 passed to 18 passed.

The two verbs now route through the same helper the other three do, so the
refusal for a wrong handle is identical everywhere rather than merely equivalent
in verdict, and a caller told `'NoneType' object has no attribute '_snapshot'`
is instead told which verb refused, which parameter it refused, and what to pass.

Why the graders were green on each half. The sweep derives its population from
the tree -- any `@tool` in the package whose first parameter is annotated `Any`
-- so it grades a verb added later without being edited, which is the right
shape. It went green on the branch that introduced it because neither
`g1_battery` nor `g1_mainboard` was on that branch's base, and green on each
verb's own branch because the sweep was not there yet. The merge-base overlap
check did not bridge them either: it intersects *changed file paths*, and a
tree-derived sweep reads files it never touches, so the pair showed no overlap
(`strands-labs/robots#2791`). A whole-tree grader is the one a diff-scoped
selector cannot see (`strands-labs/robots#2940`).
