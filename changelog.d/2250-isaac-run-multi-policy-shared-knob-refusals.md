### Quality: pin Isaac `run_multi_policy`'s four shared caller-knob refusals

`IsaacSimulation.run_multi_policy` routes `control_frequency`, `duration`,
`instructions` and `action_horizon` through base helpers shared with the MuJoCo
backend, and its own comments state that intent four times ("one refusal text
for every backend", "guards the same domain as `run_policy`", "MuJoCo parity",
"the shared positive-int domain"). The helpers are unit-tested on the base, and
the behavioural coverage the base-contract module delegates to an entry point
runs on `create_simulation()` - the default backend - so no test drove those
four refusals through the Isaac loop: five refusal lines were unexecuted,
including the whole `if n_steps is None` duration arm.

Each is now driven through the Isaac entry point and asserted to return the
shared helper's envelope **verbatim**, so "cannot drift from MuJoCo's refusal
texts" is a checked property rather than an intention, and each refusal is
asserted to advance no physics, apply no joint targets and leave no
`policy_running` flag set. Tests only - no library behaviour changes.
