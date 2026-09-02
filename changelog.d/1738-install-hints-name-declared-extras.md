### Fixed: every install hint names an extra that exists

The VERA docs led their install section with a command for an extra that had
been deleted a month earlier:

```bash
pip install 'strands-robots[vera]'   # exits 0, installs nothing VERA-related
# WARNING: strands-robots does not provide the extra 'vera'
```

`pip` does not fail on an unknown extra. It warns on one line, installs the
base package with none of the dependencies the reader was promised, and exits
0 - so the reader sees a successful install, follows the next code block, and
hits a `ModuleNotFoundError` that looks unrelated to the step that caused it.

The `vera` extra was not a typo but a deliberate removal: VERA ships only as a
git repository, PyPI rejects metadata carrying a VCS reference, and the extra
was dropped for exactly that reason. `pyproject.toml` and the provider's module
docstring both say so; only the doc still disagreed. It now gives the sequence
those two already documented - client deps, then VERA from git, then
`[vera-sim]` for the sim example. A further site named `[isaac]` for what
is really `sim-isaac`.

Two guards pin the surfaces that hand an extra name to a user: a sweep of every
qualified `strands-robots[...]` mention in the tree (358 today), and an AST
audit of the literal `require_optional(extra=...)` call sites (30 today), whose
value is interpolated straight into the `ImportError` a user reads. Names are
compared PEP 685-normalized, so a spelling `pip` would accept is not reported
as broken. `CHANGELOG.md` and `changelog.d/` are outside the sweep on purpose:
an entry that fixes an extra name has to be free to quote the broken one.
