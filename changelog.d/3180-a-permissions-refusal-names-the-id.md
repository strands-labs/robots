### Docs: a `mergePullRequest` permissions refusal names the object the ID resolved to, not your permission

`AGENTS.md` step 8 recorded that a mutation aimed at a well-formed but wrong node
ID "was stopped by permissions rather than by the check". That reads as a safety
net. It is also a false negative about the pull request you meant, and the
wording invites exactly one wrong conclusion.

Measured while merging #3175, which read `APPROVED` / `CLEAN` / `MERGEABLE` with
every required context `SUCCESS`. A `pullRequestId` recalled from an earlier
response rather than resolved in that run - `PR_kwDORUMiZs7DHqZ3`, whose middle
field is this repository's databaseId, so the offline decode clears it - returned
`cagataycali does not have the correct permissions to execute MergePullRequest`.
It names `gip-inclusion/autometa#8`. Four minutes later the same account, token
and mutation squashed #3175 as `c418f74` from the ID resolved by
`repository(owner:, name:) { pullRequest(number: 3175) { id } }`. So the
permission reported on was a stranger's, and every permission the merge needed
was held.

Nothing in the response separates the two cases: permission is evaluated before
state, so even that already-merged target - a merge no permission could have
completed - was refused for permissions rather than for mergeability. Read at
face value the sentence says "merging is a maintainer action", which is
self-consistent, terminal, and leaves an approved pull request open. That is a
third recorded cause of the presentation #1905 and #1917 describe, and the first
on the write side, so re-reading the merge gate with `PAT_TOKEN` does not reach
it.

Two corrections in step 8's node-ID bullet, pinned by
`tests/test_graphql_node_id_targeting.py`: the refusal is not a verdict on your
own permission, and "a refused `mergePullRequest` leaves nothing behind" is now
scoped to the repository, since the belief it plants is what costs the run.
Filed as #3180.
