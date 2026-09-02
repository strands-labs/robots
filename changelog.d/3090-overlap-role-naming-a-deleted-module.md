### Fixed: a docstring role naming a module the other side deletes is now a reported overlap

`scripts/check_merge_base_overlap.py` read two relations, and both read a *change*: the paths a
branch and its base both edited, and the dotted module names the branch's tests write as string
literals. `main` went red at `8d0298345` through neither of them. #3037 removed 96 g1 lookup
modules; eight verb pull requests that had already merged cited those modules in docstring roles
(``:mod:`strands_robots.tools.g1.g1_fsm_targets```). #3037 branched before the verbs landed, so no
tree held both the role and the deletion until the squash, where
`tests/test_docstring_xref_roles_resolve.py` - which resolves every role in the tree - reported 44
offending docstrings across 15 files.

Neither existing relation could reach it. Replayed from the real commits, with the branch forked at
#3055's merge base and the base advanced to the tip carrying the four verbs: the changed-path
intersection is empty in both directions, git merges the two sides with no conflict, and the check
reported `No overlap` and exited 0.

There is now a third relation, over the same merge base and one further input: the qualified
cross-reference roles in the docstrings of one tree, resolved through the same
`named_module_paths` the literal relation uses, intersected with the module files the other side
removes. Both directions are read, because either side can land last - the branch deletes and the
base cites, or the branch cites and the base deletes. On that replay it now exits 1 and names 29
`(citing file, target)` rows, which is exactly the set of distinct file-and-target pairs the grader
reports on the merged tree.

Each side is read from a tree rather than from a diff, because a role only counts inside a
docstring and a hunk cannot be parsed for that - so a role in a comment or a runtime string is not
a finding here, and the population is identical to the grader's. Only contiguous fully-qualified
targets are resolved; a wrapped or short-form role has no decidable module path, and the grader
reports both on their own account. The relation is quiet where it cannot matter: a removal nobody
cites, a role nothing removes, and a citing file the branch itself changes - that last one because
the branch's own tree holds both halves, so its own suite run already reports it. It costs nothing
on a branch that removes no package module, which is most of them: an empty key set short-circuits
before git is asked anything, and a `git grep` for the removed modules' dotted names narrows the
parse to the candidate files.

Unlike the path relation, this one is not cleared by merging the base alone. That merge produces
precisely the tree the report is about - a role and no module - and the suite is red on it, so the
role must be dropped or repointed too, which is what the report's remedy sentence asks for.

The relation is absent from `--all-open`, which is named in the sweep's own limits paragraph: the
sweep reads patches and no checkout, so it cannot tell whether a role sits in a docstring, and
grading a different population than the gate would be worse than naming the gap.
