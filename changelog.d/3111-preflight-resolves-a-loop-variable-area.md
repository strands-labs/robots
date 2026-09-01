### Fixed: the whole-tree grader preflight collects a grader whose area is a loop variable

`scripts/check_whole_tree_graders.py` derives its roster by resolving each
walk's receiver to a concrete path and keeping the modules whose walk lands on
the repository root or a top-level Python area. It read a `/` segment only as a
string constant, and the idiom most whole-tree graders in this tree use holds
the area in a loop variable:

```python
_TREES = ("strands_robots", "tests", "tests_integ", "examples")
sorted(p for tree in _TREES for p in (_REPO_ROOT / tree).rglob("*.py"))
```

`(_REPO_ROOT / tree)` has an `ast.Name` on the right, so the walk resolved to
nothing and the module was skipped. Measured over the tree, 21 modules use that
spelling and 15 were absent from the roster. The gap read as healthy because 5
of the remaining 6 were rescued incidentally by a *second*, resolvable walk
elsewhere in the same file -- including three the roster's own pin names, so
`test_the_roster_covers_every_grader_the_issues_name` passed over a resolver
that could not read their main walk.

The live instance is #3108: a redundant `except` tuple member took the required
`call-test-lint` red while this preflight passed, never having collected
`tests/test_except_tuples_state_their_real_scope.py`. That is the failure mode
#3105 documents, arriving one layer further down -- the roster is no longer
hand-maintained, but the derivation feeding it could not see the file either.

A loop variable iterating a tuple, list or set of string literals now
contributes one candidate segment per literal, and `walk_targets` decides
membership exactly as it already did for a literal segment. Keeping those two
concerns apart is what makes the change safe rather than merely wider: the
eight `tests/simulation` backend sweeps walk `_SIM_PACKAGE / backend`, so they
resolve now and are still *not* selected, because a walk rooted inside a
subpackage is collected by a path-scoped run over the mirroring test directory
and is not in the class this preflight exists to rescue. They are the control
the resolver is measured against, and a partially understood iterable resolves
to nothing rather than to its string members -- a subset would read as a
resolved walk while omitting areas the grader covers.

Derived graders go 60 to 68 over this tree with nothing lost, collecting seven
that were invisible. The `UNDERIVABLE_GRADERS` entry for
`tests/test_test_module_names_do_not_spell_a_tracker_coordinate.py` is dropped:
its stated reason was this exact shape, and
`test_each_explicitly_named_grader_is_invisible_to_the_derivation` requires the
removal now that the derivation can see it. No behaviour changes for the
required check -- all seven were already collected by a full `tests/` run; what
changes is that the preflight is now faithful to it.
