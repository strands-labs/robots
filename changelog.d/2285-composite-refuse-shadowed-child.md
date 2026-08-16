### Fixed

`CompositePolicy` now refuses a merge in which routing discards a child policy's
entire output, instead of silently returning the other child's commands. Two
children that drive the same joints (a whole-body kinematic generator such as
`KimodoPolicy` / `MotionBricksPolicy`, both keyed on all 29 `WBC_G1_ALL_JOINTS`,
paired with `WBCPolicy`, which drives 15 of them) left one child contributing
nothing while the rollout reported success. The error names the shadowed child,
the joints the other child already drives and the remedy. Partial shadowing --
lower precedence on a shared name with the upper child still contributing its own
names -- is the documented default and is unchanged.

Generator docstrings and `docs/policies/{kimodo,motionbricks}.md` no longer
present physics tracking as a `CompositePolicy` layer: a controller that tracks a
reference consumes the generator's targets as its input, so the two run in series
over the same joints, and `WBCPolicy` has no reference-pose input at all.
`docs/policies/kimodo.md` and `examples/kimodo/kimodo_g1_walking.py` also
documented a `policy_config={"layers": [...]}` composite configuration that does
not exist -- `create_policy("composite", layers=[...])` raises `TypeError` -- so
the example's tracker path could never run.
