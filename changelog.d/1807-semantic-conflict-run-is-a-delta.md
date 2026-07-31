### Docs: the local semantic-conflict run must be read as a delta, not an absolute

`AGENTS.md` > PR Workflow > step 8 tells you to compile two overlapping approved
PRs locally and run the affected tests before merging, because
`MERGEABLE`/`CLEAN`/`SUCCESS` are each evaluated against a base that does not
contain the other PR (#1766 / #1763). It did not say how to read the result, and
the obvious reading is wrong often enough to cost a good merge.

#1786 and #1804 both edited `strands_robots/simulation/predicates.py` and
neither one's CI had ever compiled it with the other. The composition reported
`376 failed, 4229 passed, 34 errors` on the affected suite - on its own, a
broken merge. The identical command on the unmerged base reported
`376 failed, 4185 passed, 34 errors`: every failure pre-existed, because a
hosted runner has no GPU and only software OSMesa, so the rendering-dependent
tests fail there whatever the diff. The entire effect of the merge was **+44
passes**, exactly the 24 + 20 tests the two branches add.

The hazard is symmetric, which is why it is worth stating rather than leaving to
judgement: an absolute count from a partial environment can invent a regression
that stops a correct merge, and it can equally hide a real one inside the noise.
Only the delta against the same command on the base carries signal.

A second sub-bullet records that squash rewrites the commits, so
`git diff --name-only <composition> origin/main -- strands_robots/ tests/` is
what ties a local verification run to what actually landed - and on a batch,
where only the tip's `call-test-lint` survives the `refs/heads/main` concurrency
group (#1800), that equivalence is the sole evidence the intermediate commits
were ever compiled together.

Documentation only; no production code or test behaviour changes.
