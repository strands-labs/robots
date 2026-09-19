### Tests: a whole-tree grader walks the tree once per session, not once per cell

Four graders re-walked and re-parsed the repository for every cell that asked
the same question of it - `tests/test_sys_modules_removal_leaves_no_orphan.py`
nine times across its three rules, `tests/test_documented_guard_names_resolve.py`
seven times, and two others twice each. The tree does not change during a
session, so each now parses it once behind a `functools.cache` and holds only
the small result set it derived, never the parsed trees. Measured on the four
files: 215 s before, 78 s after, with no rule, population or message changed.
Towards #3869.
