### Fixed: the open-set overlap sweep refuses a `--repo` it cannot honour

`scripts/check_merge_base_overlap.py --all-open` parsed `--repo` and dropped it. In
this script `--repo` is the local checkout path and the sweep reads no checkout - it
resolves the open set from the API, keyed on `--github-repo`. The sibling intake check
spells the API repository exactly that way (`check_duplicate_claim.py --repo
strands-labs/robots`), so a caller reaching for it here is reaching for the repository
to sweep.

Measured from a scheduled agent whose checkout is another repository: `--repo
strands-labs/robots --all-open` exited `0` with a clean report naming that other
repository. Against the repository it meant, the same command reports 11 open pull
requests, 55 pairs and 2 blocking pairs (`#1035 + #1722`, `#2480 + #2534`), so the
clean report was hiding findings rather than merely misattributing them - the wrong
answer shaped exactly like the right one that AGENTS.md step 1 documents for the
sibling, in the mode step 12 asks a scheduled agent to run.

`--all-open --repo` is now refused the way `--all-open --head` already was, and the
message names `--github-repo owner/name` so the caller gets the spelling that works
rather than deriving it from `--help` for two flags that differ by a prefix. The
distinct spelling was the earlier remedy for the same ambiguity and it only separates
the two for a caller who already knows which is which. Single-branch mode does read a
checkout and keeps the flag.
