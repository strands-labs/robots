### Security: every workflow `uses:` reference pins a commit SHA, and a guard keeps it that way

`AGENTS.md` > Action Pinning requires every `uses:` reference to name a full
40-character commit SHA with the version tag as a trailing comment, because a tag
is mutable and resolves to whatever it points at when the job starts -- the
pattern the `tj-actions/changed-files` incident exploited. `docs.yml` was outside
the rule, and all four of its references named a moving tag:

| file | line | was | now |
| --- | --- | --- | --- |
| `.github/workflows/docs.yml` | 35 | `actions/checkout@v4` | `@11bd7190`  `# v4.2.2` |
| `.github/workflows/docs.yml` | 36 | `actions/setup-python@v5` | `@a26af69b`  `# v5.6.0` |
| `.github/workflows/docs.yml` | 47 | `actions/upload-pages-artifact@v3` | `@56afc609`  `# v3.0.1` |
| `.github/workflows/docs.yml` | 65 | `actions/deploy-pages@v4` | `@d6db9016`  `# v4.0.5` |

Of 35 `uses:` references across the 14 workflows, 2 are `./`-relative calls to
this repository's own reusable workflows and name no version by design; of the 33
remaining, 29 were pinned and these 4 were not. `docs.yml` is also the workflow
holding `id-token: write` and `pages: write` for the Pages deployment, so it was
the most expensive file in the tree to leave mutable.

That the tags move is measured, not hypothetical. `actions/checkout@v4` resolved
to `11bd7190` (v4.2.2) when the sibling workflows were pinned; resolving the same
tag for this change returned `11d5960a` (**v4.4.0**), two minor versions along,
under a reference that reads as fixed. So `checkout` and `setup-python` take the
SHA their 12 and 9 siblings already carry rather than the tag's current target:
pinning to today's `v4` would have removed the mutability and silently upgraded
the docs build's checkout while every other workflow stayed on v4.2.2 -- an
unrequested behaviour change inside a supply-chain fix. The other two actions
appear only here, and are pinned to the exact commit the tag they named resolves
to, so the build is unchanged.

Nothing detected any of this. `tests/` reads `.github/workflows/` in seven places
and none of them reads a `uses:` line, so the file sat outside a non-negotiable
rule with every signal green. `tests/test_workflow_action_refs_pin_a_sha.py`
grades four properties against the tree: a remote reference pins 40 hex
characters, it carries the trailing tag comment, an unpinned reference is a `./`
call resolving to a workflow that exists, and an action resolves to one SHA
tree-wide. The last is what makes the sibling SHA the only legal answer and keeps
each Dependabot bump one atomic edit. 8 of its 87 checks fail on the previous
`docs.yml`, and the one-SHA check fails on the plausible alternative fix.

No package code changes.
