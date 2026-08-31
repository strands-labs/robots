### Fixed: the host-path sweep reads every area that ships Python

`tests/test_no_host_paths.py` refuses a committed `/Users/<name>` or
`/home/<name>` literal, and it walked a hardcoded three-tuple --
`strands_robots`, `tests`, `tests_integ`. The repository ships Python in five
top-level directories, so `examples/` (97 files) and `scripts/` (12) were
outside the gate entirely.

`examples/` is the worse of the two to miss. It is the code a reader copies, so
a host path there is not merely committed, it is propagated: the copy fails on
every machine that is not the author's, and the gate that exists to say so never
read the file. Neither omitted area is covered by the reason the sweep gave for
its scope, which excluded documentation and third-party code.

The area list is now derived from the repository rather than written down, the
shape `tests/test_parameter_deletes_precede_the_body_they_narrow.py` already
uses for the same problem: a directory added later is swept on arrival, where a
tuple equal to today's set fires on nothing when the tree grows. A
`_REQUIRED_AREAS` floor refuses only the reverse, an area silently dropping out.

Deriving the list can reach a virtual environment, which a hardcoded tuple could
not, and site-packages is full of the packager's own home directory -- that would
fail the gate for a reason the author cannot fix. A directory carrying the PEP
405 `pyvenv.cfg` marker is skipped, by marker rather than by name, since `venv/`
is as common as `.venv/` and only the latter is a dot-directory. One predicate
owns that rule, so the sweep and the cell grading its reach cannot disagree about
which directories the gate reads -- a second derivation of "an area that ships
Python" is what would fail the gate on the author's own checkout layout.

No line in the newly-read areas is flagged: 109 files, zero hits, so no
allowlist entry was added. This closes a latent gap rather than cleaning up
after one.
