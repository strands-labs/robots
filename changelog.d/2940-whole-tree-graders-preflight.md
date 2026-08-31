### Added: a preflight command that runs the whole-tree graders a narrow selector cannot see

Several tests in this repository derive their expectation from the whole tree
rather than from the file under change - `tests/test_docstring_xref_roles_resolve.py`,
`tests/test_no_host_paths.py`, `tests/test_dependency_audit.py`,
`tests/tools/test_agent_tool_parameter_descriptions.py`, and
`tests/test_parameter_deletes_precede_the_body_they_narrow.py`. Each grades a
diff against the *rest* of the repository, which is what makes them invisible
to a path- or `-k`-scoped `pytest` selector and what makes them the class of
check a narrow local run must not skip.

Issue #2940 documents the failure mode reproduced twice on this port series
(#2934, #2938): both PRs cited a green `pytest tests/drivers/ -k g1` in their
description, both landed on CI with `call-test-lint` red, and both were red
for the same reason - a `:mod:` role naming a sibling module that lived in a
still-open PR, which the xref grader detects and the narrow selector never
collected.

The cheap half of the resolution is a fixed-roster preflight:

- `scripts/check_whole_tree_graders.py` collects and runs every whole-tree
  grader by its exact node id, refuses extra arguments (the point is that the
  input set is fixed, not composed by a caller), and refuses when a rostered
  path is missing on disk before invoking `pytest` - so the diagnostic is
  "the grader was moved" instead of pytest's usage-error exit code.
- `hatch run whole-tree-check` runs the same script through the default env,
  so a preflight step reads the same way as the required check would report a
  failure.
- `tests/test_whole_tree_graders_roster_is_complete.py` pins the roster to
  reality by AST-scanning `tests/` for the shapes that grade a whole tree
  (`ast`+`pathlib` at module scope with a tree walker at any depth, or a
  `tomllib` import at module scope for a manifest-vs-tree grader). A grader
  added under `tests/` without an entry in the script's roster turns this
  test red, so the two artifacts cannot drift the way the two ports drifted
  from main.

Two subject-scoped graders that walk only their subject's subtree
(`tests/mesh/test_docstring_module_xrefs.py`, `tests/mesh/test_dead_docstring_targets.py`)
are exempted with the reason inline: a `pytest tests/mesh/` collects them
alongside every other mesh test, so the "invisible to a narrow selector"
premise does not hold for them.

The more durable half of #2940 - a convention entry in `AGENTS.md` about
whether a qualified role naming a sibling module inside an in-flight series
should be written at all - is out of scope here on purpose. #2937 already
carries an `AGENTS.md` change, and coupling the two would create the exact
merge-order dependency between independent PRs that #2940 is partly about.

Refs #2940.
