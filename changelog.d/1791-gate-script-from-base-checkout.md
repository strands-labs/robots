### Fixed: a CI gate no longer accuses a branch that forked before the gate existed

`.github/workflows/changelog-fragment.yml` checked out the pull request head and
then ran `scripts/check_changelog_fragment.py` out of that tree. A branch that
forked before the gate landed does not carry the script, so the step died before
the check began:

```
python3: can't open file '.../scripts/check_changelog_fragment.py'
Process completed with exit code 2
```

The script reserves exit 1 for a real unaccounted entry and 0 for clean. Exit 2
is neither, and the checks UI renders 1 and 2 as the same red X -- so the gate
reported a changelog violation against a branch that had not committed one, and
in the other direction could never report its real signal there, because it never
got far enough to compute one. Measured on two open pull requests whose only
relevant difference is where they forked: #1786, one commit before the gate,
FAILURE with exit 2; #1788, after it, SUCCESS.

Such a branch is not an edge case. A `pull_request` workflow definition is read
from the merge commit, so the job runs against heads that contain neither it nor
the script: #1786's head carries neither, and the check ran anyway.

The workflow conflated two trees that coincide only by accident -- the tree under
review, and the tree the script is read from. It now checks out the base branch,
where a gate's own script is guaranteed to exist, fetches `refs/pull/<n>/head` so
a fork's commit is reachable, and names the commit under test with `--head`, which
is what that argument was documented for. Fidelity is unchanged because the script
reads no file from the working tree: every input comes from `git show <rev>:<path>`
and `git diff <a>..<b>`, so only reachability can affect the answer. Replayed
against #1786's head, the same commit that produced exit 2 now produces a verdict
-- clean, which it always was.

`.github/workflows/merge-base-overlap.yml` had the identical shape and is fixed
the same way. Its failure was latent rather than active only because every
currently open branch happens to postdate it.
