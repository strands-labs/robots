### Changed: the g1 tool surface is 25 verbs behind one `use_unitree` dispatcher, not 120 names

`strands_robots/tools/g1/` had inverted. It carried 120 `@tool` names across 71
modules, and roughly 100 of those were read-only *envelope/admits* lookup pairs
— `g1_list_walk_forward_envelope` alongside `g1_walk_forward_admits`, repeated
about fifty times — against only ~14 verbs that move the robot. An agent asked
to walk found two validators and no walker, and every lookup name spent a
tool-schema slot in the model's context that an execution verb then could not
have. This is the seam question in #2928, and the cost of leaving it open grew
with each envelope pair that landed.

The 46 lookup modules and their 46 tests are removed, and the constants they
returned get a single home instead of one tool name each:
`strands_robots.tools.g1.use_unitree.use_unitree` dispatches over the six SDK2
services (`loco`, `arm`, `audio`, `motion_switcher`, `vui`, `robot_state`) in
the `use_aws` shape — a `service`, an `operation`, and its `parameters`. Meta
discovery (`list_services`, `list_operations`, `describe_operation`) is what
replaces the lookups: the operation set and each operation's parameters stay
reachable at runtime, from one verb. Discovery reads signatures through
`inspect` with an AST fallback, so `describe_operation` still answers with no
`unitree_sdk2py` installed, and no module-level SDK import means resolving every
export in this package imports zero `unitree_sdk2py` modules. Mutative-operation
detection, `HIGH_DANGER_OPS` flags, singleton clients and a single RPC lock
follow the hygiene the driver already established.

Nothing that executes is removed. `_g1_common`, `_dds_engine` and
`_motion_switcher` are untouched; so are the driver-cache reads (`g1_state`,
`g1_battery`, `g1_imu`, `g1_mainboard`, `g1_pressure`, `g1_lidar_state`,
`g1_lidar_summary`), the driver-gated writes (`g1_send_action`, `g1_run_policy`,
`g1_start_task`, `g1_stop_task`, `g1_get_task_status`, `g1_set_stand_height`,
`g1_set_swing_height`, `g1_balance_stand`), and the lookups that answer a
question a caller actually has rather than restating an envelope
(`g1_motion_gates`, `g1_joints`, `g1_error_codes`, `g1_arm_actions`). Measured
across the package: 71 modules become 24, 120 `@tool` names become 25, and the
count of `_envelope`/`_admits`/`_topics`/`_ids`/`_keys`/`_notes` modules goes
from 34 to 0.

Removing a module also removes anything that cited it. `g1_balance_stand`
landed in #3033 after this work forked and `:mod:`-cites `g1_balance_modes` in
seven places, with three more in its test; a three-way merge keeps those
citations and deletes the module they name, which is enough to turn
`tests/test_docstring_xref_roles_resolve.py` red without ever showing up as a
conflict (refs #2940). Those roles are scrubbed here, and every fact they
carried is kept — the admitted set `{0, 3}` was already stated inline in the
same docstrings.

Two things the removal itself did not settle are settled here, because both are
invisible to the tools that would otherwise catch them.

The package now publishes its verbs through a `_LAZY_IMPORTS` table, so that
table *is* the surface, and a hand-written table drifts. A verb landing on
`main` after this work forked is kept by a three-way merge while the rewritten
`__init__` never learns its name; a lookup pair landing the same way survives a
removal it was never part of. Neither is a conflict — git sees an addition on
one side and no change on the other — so `g1_balance_stand` (#3033) was
reachable as a module and *not* through the package, and two later envelope
pairs (#3029, #3036) would have merged into a surface the change describes as
carrying none. Both are corrected, and the rule is now derived from the tree
rather than looked for by hand: every `@tool` on disk in the package resolves
through the package, every declared name resolves, and no lookup-shaped module
survives.

Discovery also narrows what it treats as a missing SDK.
`list_operations` and `describe_operation` fall back to an AST read of the SDK
source when `unitree_sdk2py` is absent, and they reached that fallback by
catching every exception. Any defect in the introspection path therefore
answered from a possibly-stale on-disk tree, or returned an operation list that
was empty rather than unread, or reported `parameters: []` for an operation
whose signature simply could not be read — which tells a caller the operation
takes no arguments. The fallback now triggers on the absent-SDK condition
specifically (`ImportError` for no SDK, `AttributeError` for a renamed client
class); anything else propagates and surfaces at the tool boundary as an error
naming the defect, and an unreadable signature declines the introspected answer
instead of publishing an empty parameter list as authoritative.
