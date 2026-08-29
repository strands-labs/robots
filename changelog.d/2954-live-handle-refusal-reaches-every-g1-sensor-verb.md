### Fixed: the live-handle refusal reaches every G1 sensor verb

`g1_battery` and `g1_mainboard` dereferenced `driver._snapshot(...)` on trust, so a
handle that is not a G1 driver raised past the `@tool` boundary that
`strands_robots.tools.g1`'s module docstring says never raises: a robot name in place
of the handle, an omitted handle, or an object carrying the accessor as data rather
than as a method each surfaced as an `AttributeError` or a `TypeError` instead of the
structured error envelope. Both verbs now consult
`strands_robots.tools.g1._g1_common.snapshot_handle_refusal` before touching the
driver, which is what their three sibling sensor verbs already did.

The two verbs and the guard that grades them crossed on `main`, which is why neither
pull request could see it. `g1_battery` merged forty minutes before the derived
guard; `g1_mainboard` landed eighty seconds after it. The required status check reads
a pull request's head alone, so a branch that forked before the guard never runs it,
and a branch that forked before a verb never sees that verb - both merged green and
the two together left `main` red. On `main` today
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` reports 8 failed
and 10 passed; with this change it reports 18 passed. That guard discovers every
live-handle verb from the package rather than naming them, so it covers these two
without a new cell and covers the next one on arrival.
