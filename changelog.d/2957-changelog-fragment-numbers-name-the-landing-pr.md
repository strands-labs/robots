### Fixed: a changelog fragment is named for the change that landed it

Three fragments had merged under pre-PR placeholder names (`9997-`, `9998-`,
`9999-`), two of them describing work that landed 37 PRs earlier. Because
`scripts/assemble_changelog.py` orders fragments by descending number so the
assembled section reads newest-first, a placeholder number is not a position in
that sequence but simply larger than every real one, so all three sorted ahead
of the 949 genuine entries and the next release notes would have opened with
them.

The number is also the only pointer from an assembled release-note entry back
to the change that produced it, and `--apply` deletes each fragment as it folds
it in, so a release cut under a placeholder loses that association from both the
log and the directory with no way to recover it.

Each fragment is renamed to the PR that carries it, and the rule
`changelog.d/README.md` already stated is now pinned: a fragment number at or
above 9000 is refused, alongside a cell asserting the descending-order
consequence so the rule keeps its reason rather than outliving it.
