# changelog.d - news fragments

One file per change. `CHANGELOG.md` is assembled from these at release time.

## Why not edit CHANGELOG.md directly

Every PR that appended straight to `## [Unreleased]` inserted at the *same
anchor*, so any two PRs open at once conflicted the moment either one merged --
on ordering alone, never on meaning. Because stale approvals are dismissed on
push, clearing that conflict cost a full re-approval round per affected PR, and
the cost grew with the number of open PRs. A fragment removes the conflict
class outright: no two PRs touch the same path, so there is nothing to
reconcile.

`CHANGELOG.md` stays the human-facing log and is still edited directly for
release bookkeeping (collapsing `[Unreleased]` into a dated section). It is
fragments, not the log, that behavioural PRs write to.

This rule is enforced by the `Changelog fragment` step of the required check
(`scripts/check_changelog_fragment.py`), which names
any `### ` entry a branch adds to `[Unreleased]` that no fragment accounts for.
It is a base diff rather than a test because `[Unreleased]` already carries
entries from before this convention, so no static assertion about that section
can hold. Release bookkeeping is unaffected: `--apply` deletes each fragment it
folds in, so an assembled entry is matched to the fragment it came from, and
editing or reordering an entry already in the log adds no heading.

## Adding a fragment

Create `changelog.d/<number>-<slug>.md`, where `<number>` is your PR (or issue)
number and `<slug>` is a short lowercase description. `0000` and `999x` are
placeholders, not numbers, and are refused by `--check`, by `--apply` and by the
pull-request convention check; rename the file once the PR number exists:

```
changelog.d/1692-teleop-slew-bound.md
```

The content is exactly what would have gone into `CHANGELOG.md` -- a
`### <Category>: <summary>` heading followed by the prose body:

```markdown
### Fixed: a teleop command a joint cannot travel that fast is refused

`RobotMesh.apply_input_frame` bounded frame rate and value magnitude but not
rate of change, so a replayed stream could command a full-scale reversal
every frame ...
```

Rules, all enforced by `tests/test_changelog_fragments.py`:

- Opens with a level-3 entry heading, in the style of the entries already in the
  log (`Fixed`, `Added`, `Docs`, `Security`, ...). The category wording is yours
  -- only the structure is checked, so this convention does not narrow what the
  log may say.
- Exactly one `### ` entry per file -- split two changes into two files.
- No level-2 (`## `) heading: that would forge a version section and break the
  structural contract in `tests/test_changelog_format.py`.
- Nothing else lives in this directory. A stray or misnamed file is a hard
  error, not a skipped one, so an entry can never be silently dropped.

Validate locally with:

```bash
python scripts/assemble_changelog.py --check    # names, numbers + headings
python scripts/assemble_changelog.py --print    # preview the assembled section
```

## Releasing

Fold the accumulated fragments into the log, then collapse `[Unreleased]` into
the dated section for the tag:

```bash
python scripts/assemble_changelog.py --apply    # writes CHANGELOG.md, removes fragments
```

`--apply` validates everything first and refuses wholesale on any problem, so a
malformed fragment cannot leave the log half-written or a fragment deleted
without its content landing.
