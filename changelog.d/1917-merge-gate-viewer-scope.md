### Docs: the documented merge gate is viewer-scoped, and the Actions token cannot read it

`AGENTS.md` > PR Workflow > step 8 gates a merge on the required contexts plus
`mergeStateStatus == CLEAN`, and calls `mergeStateStatus` the field "to trust".
It is, and the instruction was still unusable as written, because the field
answers *can the viewer merge this pull request* - so it is scoped to the token
that asks. On a pull request editing `.github/workflows/**` the Actions
`GITHUB_TOKEN` can never read anything but `BLOCKED`, however ready the pull
request is, because an installation token is refused writes to workflow files
and therefore genuinely cannot perform that merge.

A control pair, both approved, no unresolved thread, `call-test-lint`
`SUCCESS`, read minutes apart: #1915 (edits `pr-and-push.yml`) read `BLOCKED`
with `GITHUB_TOKEN` and `CLEAN` with `PAT_TOKEN`; #1902 (no workflow edit) read
`CLEAN` with both. Both merged clean minutes later, `f4dfde6` and `6cf0470`.
The mechanism isolates to one variable - the same token, on the same scratch
branch at the same instant, created `zz_probe.txt` and was refused
`.github/workflows/zz_probe.yml` with `Resource not accessible by integration`,
which `PAT_TOKEN` then created.

Staleness is separately ruled out: `mergeable_state` on #1899 and #1035 reads
`unknown` first and the settled value second for both tokens identically, so
the lazy first read is per-PR, not per-viewer. `mergeable` agrees across tokens
throughout, since a text conflict is viewer-independent and only
mergeability-by-you is not.

What made it expensive is that the wrong answer is indistinguishable from a
right one. On a genuinely blocked pull request both tokens read `blocked`, so
their agreement proves nothing, and no reading of the field separates the two
cases. An agent polling the gate exactly as documented reads `BLOCKED`,
correctly declines to merge, and reports a ready pull request as waiting on a
reviewer - the presentation #1905 records for a different cause, which had
stood in eight consecutive scheduled scan summaries as "reviewer bandwidth is
the sole constraint". It bites CI and process pull requests specifically,
because those carry the workflow edits.

`AGENTS.md` now records the scoping, the control pair, the isolating probe and
the prescription (read the gate with `PAT_TOKEN`) beside the claim it qualifies,
and `tests/test_merge_gate_viewer_scope.py` pins the token and the
`.github/workflows/**` exception to the same passage - a future edit tightening
the section back down to "poll `mergeStateStatus == CLEAN`" is the regression,
and it would look like an improvement.

Documentation and test only; no production code changes.
