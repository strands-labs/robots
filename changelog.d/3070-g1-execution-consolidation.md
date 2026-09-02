### Changed: the 13 G1 execution verbs live in one table-driven module

The per-verb execution wrappers that landed between #3037's fork and its
merge (`g1_move_velocity`, `g1_set_fsm`, the safe posture transitions, the
arm and loco gestures, ...) regrew `strands_robots/tools/g1/` to 34 modules
— thirteen files whose executable bodies were each a shared-handle
judgement, the shared numeric validators and one driver call, under ~250
lines of restated prose. They fold into `g1_actions.py`: the handle-refusal
wording becomes rows in one `_ACTIONS` table, and the thirteen
one-file-per-verb test suites become one table-driven suite. Every `@tool`
name, signature and refusal shape is preserved, and all thirteen now
actually resolve through the package — ten were never added to
`_LAZY_IMPORTS`, the exact three-way-merge drift
`test_the_consolidated_g1_surface_holds` predicted and was red on `main`
for. 34 modules -> 22, net −7,825 lines.
