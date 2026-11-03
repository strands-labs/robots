### Internal: pull requests are checked for changelog fragments again

The required check's first step, `scripts/check_changelog_fragment.py`, refuses a pull request that writes a
`### ` entry straight into `CHANGELOG.md`'s `[Unreleased]` instead of a `changelog.d/` fragment, or that adds a
fragment the assembler refuses. It ran on main before 0.6.0 and is back, with its tests.
