### Fixed: the mesh pacing graders read the parsed tree, not the source text

`tests/test_mesh_state_loop_rate.py` scans the mesh publish loops for the
`self._stop_event.wait(period)` pacing they were converted away from. Every
converted loop's docstring explains that conversion by quoting the very call
being banned, so the scan has to tell code from prose - and it did so
textually, by removing `__doc__` from `inspect.getsource`.

That assumes the two are byte-identical. Python 3.13 removes a docstring's
common leading indentation at compile time, so `__doc__` stopped being a
substring of the source, the removal matched nothing, and
`Mesh._state_loop` was reported as pacing on a wait it does not contain -
on a supported interpreter, with the loop unchanged, and with a message
telling the reader to use the `Ticker` the loop already uses.

All four scans in the file now read `ast.Call` nodes out of the parsed tree.
A docstring is a constant expression there and a comment is not in the tree
at all, so neither can be a hit on any interpreter, the per-line prose
heuristics are gone, and a call wrapped over several lines - which the line
pattern could not see - is found.
