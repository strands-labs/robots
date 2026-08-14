### Added: `check_merge_base_overlap.py --all-open` sweeps the open set for untested compositions

The overlap check could only ever read one branch, and `M..base` is empty by
construction whenever `M` is the base that branch was evaluated against. So two
pull requests that both edit one file while both are still open are invisible to
each other -- the intersection contains neither, both read clean, and the first
tree in which the two are compiled together is `main`. Nothing clears that over
time either: stale approvals are dismissed on push, a stale pass has no
equivalent, and a pull request idle in review never re-runs.

`--all-open` reads the open set from the API and computes the same intersection
twice per pull request -- once against each sibling 's head, once against what
has landed on the base since its own merge base -- reusing the path-set helpers
the single-branch mode uses, so the two cannot disagree about what counts as an
overlap or as prose. A path set that reached the compare endpoint 's 300-entry
file cap is named as unevaluated rather than intersected, because this check 's
failure mode is a missed overlap. The two sets skip independently: the base-side
one is what grows without bound and so what hits the cap, and dropping the whole
pull request for it would discard the pairwise finding the mode exists to make.

Adding a second inferring script joins the scope of the guard that pins #2030's
defect, so that guard moves from a `len(...) == 3` literal to a set equality against
a declared mapping of the flag each script spells the repository with and the argv
marker that makes naming it required. It is per invocation rather than per script
because this script's own `--repo` is a local checkout path and its single-branch
mode never reaches the API, so a blanket rule would read a correctly-named
invocation as an inferred one and would false-reject the local-git mode.
