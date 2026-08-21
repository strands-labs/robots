### Fixed

- **ci**: `check_merge_base_overlap.py --all-open` now refuses `--repo` instead of
  silently accepting it. The sweep reads the open set from the API and is handed
  none of `--repo`, so a caller who named the repository that way - the spelling
  the four sibling gate scripts use for `owner/name` - had it read as a filesystem
  path while the sweep read `$GITHUB_REPOSITORY`, reporting cleanly and exiting `0`
  for a repository nobody asked about. The refusal names `--github-repo`, which is
  the flag that does name the repository the sweep reads. `--head` was already
  refused for the same reason; neither flag's meaning changes, so the
  single-branch path and the CI invocation are untouched.
