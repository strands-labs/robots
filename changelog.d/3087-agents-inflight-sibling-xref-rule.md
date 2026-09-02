### Changed

- **Contributing**: `AGENTS.md` now names `hatch run whole-tree-check` in the
  pre-push instructions, and the cross-reference conventions gained the rule for
  a qualified role naming a sibling module that lives in a still-open PR: write
  it as a literal until the target lands. Both halves close issue #2940. The
  preflight script it points at already existed; the document a contributor reads
  before pushing did not name it, so an author who narrowed the test run to the
  area they changed had nothing pointing at the class of grader that narrowing
  structurally cannot collect - and a green narrow run reads in a pull request
  description exactly like a green full one. Landing the sibling first is not an
  alternative remedy: the required check reads the pull request head rather than
  the merge ref, so the branch stays red until it also absorbs `main`, which
  costs a re-approval round and imposes a landing order on pull requests that
  are otherwise independent. `tests/test_whole_tree_graders_roster_is_complete.py`
  reads the hatch script's name out of `pyproject.toml` and requires `AGENTS.md`
  to name it, so renaming the script fails that pin rather than leaving the
  document pointing at a command that no longer exists. No source change.
