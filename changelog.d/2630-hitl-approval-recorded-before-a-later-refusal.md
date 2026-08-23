### Fixed

The mesh tool now records the operator's human-in-the-loop verdict once, as soon
as it is known, so a later refusal cannot return past the row. The rate limit is
deliberately re-checked under the lock *after* the operator approves -- the
pre-interrupt check does not consume a slot, so a concurrent invocation can take
the last one while the human is deciding -- and that re-check can refuse an action
the operator approved. The row was written per branch (`approved=False` in the
decline branch, `approved=True` after the re-check), so the raced refusal returned
between the two and the verdict went unrecorded: the audit log carried only
`rate_limit_race`, which says why the action was refused and nothing about who
authorised it. Measured, an approval whose slot was taken while the operator was
deciding left zero operator rows, on the one gate that reaches physical actuation.

An incident audit asks "did a human approve this?" first, which is what
`log_operator_response` already documented ("Call this on BOTH outcomes"), and the
two sibling gates already had the right shape: `use_ros` and `lerobot_train`
compute the verdict and record it unconditionally before branching. The raced path
now leaves two rows for two facts -- `operator approved: 'y'` and
`rate_limit_race: ...` -- and a structural guard derived from the tree's
`interrupt()` call sites holds every gate, including a fourth added later, to one
site no return can bypass.

The accepted grammar is unchanged: a non-string reply, including `True` and
`{"approve": True}`, still fails closed as its own tests and `docs/security.md`
require. Only the accounting on the path where the operator did approve is fixed.
