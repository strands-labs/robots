### Fixed: every Isaac motion primitive refuses while a policy drives the same robot

A primitive and a policy rollout both write the articulation's PD position
targets, so running one under the other interleaves two command streams on one
arm. `move_to` refused that up front and aborted per control tick; its two
siblings did neither, because the check sat inline in `move_to` rather than in
the preamble and the abort helper the three primitives share. Measured on a
robot whose `policy_running` flag was set, `set_gripper` and `rotate_wrist`
returned `status="success"` having applied 12 and 5 PD target sets; with a
policy starting mid-run, `set_gripper` reported success after 50 contended
ticks and `rotate_wrist` reported a *convergence timeout* - blaming the arm for
a race, after writing 50 conflicting target sets.

`_primitive_resolve_robot` now takes the action name and performs the
no-running-policy refusal (the same point the MuJoCo mixin's preamble calls
`_require_no_running_policy`), and `_primitive_abort_reason` carries the
mid-run branch. `move_to`'s two inline copies are gone and its wording is
unchanged, so the refusals are one rule across the three primitives instead of
one implemented and two missing. Per-robot scope is preserved: a rollout on a
different robot still does not block, and a robot that is not initialized or
was removed mid-run still reports its own reason rather than the policy's.
