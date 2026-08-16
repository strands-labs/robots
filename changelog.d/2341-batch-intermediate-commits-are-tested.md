### Docs: a batch no longer leaves its intermediate commits untested

`AGENTS.md` > PR Workflow > step 8 told a maintainer that only the tip's
`call-test-lint` survives a batch, making the composition equivalence the sole
evidence the intermediate commits were ever compiled together. That predates the
push-concurrency fix, and the section above it already documents the group
keying on `github.sha` so each pushed commit runs its own suite to completion -
so the file contradicted itself on whether a commit mid-batch has a verdict.
Measured on the six merges taking `main` from `239f24ab` to `0d811084`, all six
report `call-test-lint` `success`. The equivalence check stays, with the reason
corrected: it is the one form covering a whole batch, because the tree-sha
comparison below it is scoped to `behind_by == 0` and only the first pull
request in a batch can satisfy that.
