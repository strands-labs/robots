### Fixed

- `tests/test_source_strings_no_emoji.py` now scans `changelog.d/*.md`, closing
  the one hygiene surface no scan covered. All three source-hygiene scans walk
  `*.py` under the package and the test tree only, while
  `scripts/assemble_changelog.py --apply` folds fragment text *verbatim* into
  `CHANGELOG.md`, so a glyph in a fragment reached the released notes behind a
  fully green run. The gap was not hypothetical: the `U+1F6A8` marker that
  `changelog.d/2982-g1-dangerous-publish-topics-port.md` describes as *stripped*
  was still sitting in that fragment's own prose, and is re-spelled here in the
  `U+XXXX` notation AGENTS.md and the scan's own comments already use. Every
  `*.md` in the directory is scanned with no reserved-name exemption, including
  `README.md`: it documents the fragment convention and is what a contributor
  copies a skeleton out of, so a glyph there propagates by construction. The
  directory guard deliberately asserts the path still resolves rather than
  counting files, because `--apply` unlinks every fragment it consumes and an
  empty `changelog.d/` is the legitimate state of the tree on a release commit -
  a count-based guard of the kind the package and test-tree cells use would turn
  `main` red exactly there. The already-released `CHANGELOG.md:5090` glyph is
  left alone and the scan is pointed at fragments only; rewriting shipped
  release-note text is a separate decision. Refs #3022, #3020.
