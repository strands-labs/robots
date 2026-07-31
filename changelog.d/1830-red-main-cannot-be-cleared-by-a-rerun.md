### Docs: a red `main` cannot be cleared from the PRs it blocked by re-running them

A failing check on a PR whose own branch is innocent cannot be cleared by
re-running it, and the fix that works is cheaper than it looks.

Two mechanisms, both measured while landing #1824 (the fix for the three
`test_deferred_physics_and_warmup.py` failures open on `main` as #1823).
`pr-and-push.yml` checks out the PR **head commit**, not `refs/pull/N/merge` -
the job log reads `HEAD is now at <branch head>` - so a branch's green is a
statement about the branch's own tree, which is why
`Detect an untested overlap with the base branch` exists as a separate check.
And `POST /actions/runs/{id}/rerun-failed-jobs` re-uses the head SHA the run
recorded, so a fix landing on `main` afterwards is invisible to the re-run:
#1827 and #1829, both re-run after #1824 was on `main`, reported the same single
failure (`1 failed, 6287 passed` and `1 failed, 6277 passed`). The branch has to
absorb `main` and be pushed.

That push is **free when the merge is conflict-free**, contrary to the intuition
that it spends the approval. Two pushes onto the approved #1821: `b365d60`
resolved a conflict, showed 82 lines of combined diff under `git show --cc`, and
its approval was dismissed 45s later; `79cbdad` was a clean base merge, showed 0,
left the PR's diff versus its merge base byte-identical at `4 files, +158/-7`,
and the approval survived. Dismissal keys on the PR's own diff, not on the head
SHA changing - a conflict resolution is new unreviewed text and is what costs a
round.

Recorded in `AGENTS.md` step 8 with the two consequences: merge a red-`main` fix
ahead of the queue, since nothing on top of it can merge while it is red; and
never push the refresh onto a contributor's branch, which is the separate
`require_last_push_approval` identity problem (#1722) and does not care that the
merge was clean.
