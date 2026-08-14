### Fixed: the open-set sweep reads a rename's old name, and a large head side no longer drops out of the pairwise comparison

`check_merge_base_overlap.py --all-open` reported two inputs as sharing no path
when they share one, both in the false-negative direction its own docstring names
as the one that matters.

A rename was invisible. Both file-carrying endpoints report a rename as a single
entry whose `filename` is the new path and whose `previous_filename` is the old
one, and the sweep collected only the first. So a branch renaming `foo.py` and a
branch editing `foo.py` intersected on nothing, while git -- which does detect the
rename -- applies the second branch's edit to the new name with no conflict marker
to report. That is precisely the silent composition this check exists to find,
arriving in the one shape it could not see. The single-branch mode never had the
defect: it passes `--no-renames` for this reason, so the two modes disagreed about
what an overlap is, which is the divergence that makes one of them wrong without
either looking it. Taking both names can only widen the reported set, which is the
safe direction here. Measured: 0 renames across the 7 open pull requests, so
nothing was being missed today, but merged #2057 renamed
`tests/simulation/test_args_docstring_completeness.py` and a sibling still editing
the old path would have been invisible while both were open.

A head side at the compare endpoint's 300-entry file cap dropped the pull request
from *both* modes. For the base side that is correct and unchanged -- it has no
paginated equivalent, and it is the side that grows without bound, so it is the
side that reaches the cap. The head side does have one: `pulls/{n}/files` is
paginated to 3000 entries and returns the same set below the cap, measured
identical both ways on #1035 (7 files), #1722 (10) and #1667 (153, the largest
pull request in this repository's history), and byte-identical across the whole
open queue. Reading it there raises the head-side ceiling tenfold and changes no
verdict. The compare call remains, for `merge_base_commit` and `behind_by`, and no
longer enforces a cap on a `files` list it does not read -- because the mode a
300-file branch was being dropped from is the pairwise one, which is the mode that
finds defects, and a large diff is the most likely thing on a queue to collide with
something. Reaching the paginated ceiling is still reported as unevaluated, for the
reason the cap was: an endpoint that stops without saying so manufactures a missed
overlap.

Neither input is reachable on today's queue; both are reachable in this
repository's history. Who runs the sweep and what a finding costs remains
undecided (#2245) and is untouched here.
