### Fixed: a host-specific path is refused whether or not a segment follows it

`tests/test_no_host_paths.py` sweeps `strands_robots/`, `tests/` and
`tests_integ/` for hardcoded home directories, and every one of its four
patterns required a trailing separator (`/Users/[A-Za-z0-9._-]+/`). A literal
that *ended* at the user segment therefore swept clean, which is the PR #85
defect the file exists to catch, split across two statements:

```python
HOME = "/Users/cagatay"                        # not flagged
model = Path(HOME) / "robots" / "policy.pt"
```

That fails on every machine that is not the author's exactly as the single-line
form does. The separator is not what makes a path host-specific -- the user
segment is -- so the patterns now end where that segment does, in both the
backslash-escaped and raw spellings of the Windows form.

The relaxation is a pure tightening: measured over the 1720 `.py` files in the
scanned directories with the existing allowlist applied, it flags zero
additional lines, so no allowlist entry was added. The boundary is pinned
alongside the escape, because matching `/Users` or `/home` alone would flag
portable paths such as `/homebrew/bin/mjpython`; `/Users/` with no user segment
names no host and stays unflagged.
