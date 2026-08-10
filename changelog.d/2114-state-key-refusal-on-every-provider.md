### Quality: measure the `robot_state_keys` refusal on every provider that owns it

`set_robot_state_keys` is the setter whose list decides which actuator each
action value is sent to, and nine surfaces validate it through the shared
`name_list_error` domain. An AST classifier already pinned that each of the nine
*calls* the domain, but a body that keeps the `name_list_error(...)` call and
drops the `raise` satisfies that classifier unchanged - so it proves the guard is
wired, never that it refuses. Only `MockPolicy` and `RemotePolicy` were driven
behaviourally; measured with coverage over the suite, the `raise ValueError(error)`
line had never executed on `cosmos3`, `curobo`, `groot`, `lerobot_async`,
`lerobot_local`, `moveit2` or `vera`.

Each of those seven is now constructed and driven directly: the bare-string
mistake is refused with the shared domain's message verbatim (equality, so a
locally re-worded copy cannot drift), a second malformed shape is refused so the
claim is about the surface rather than one value, a refusal leaves the previously
bound layout untouched, and a distinct list is still accepted. The table is
derived from the classifier's own set, so a provider added later fails a test
rather than quietly joining the structurally-only half.

No library behaviour changes.
