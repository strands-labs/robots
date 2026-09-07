### Fixed: the mesh loop source scan reads code rather than its own documentation

`tests/test_mesh_state_loop_rate.py` pins the mesh publish loops' conversion
away from `self._stop_event.wait(period)` pacing by scanning each loop's source
for that call. The converted loops legitimately *document* the call they stopped
making, so the scan has to see code and not prose -- and it did that by
subtracting `func.__doc__` from `inspect.getsource(func)`.

That subtraction rests on two assumptions, and both are false. `getsource`
includes `#` comments, which `__doc__` never contained, so a comment recording
what a loop was converted from was already read as a pacing call on every
interpreter. And `source.replace(func.__doc__, "")` assumes the compiler stored
the docstring as a verbatim slice of the source: Python 3.13 strips the common
leading indentation from docstrings at compile time, so an indented docstring is
no longer a substring there and the replace removes nothing at all. The same
holds on any interpreter for a docstring containing an escape sequence, which the
compiler resolves and the source spells out literally.

`requires-python` is `>=3.12` and 3.13 is a declared classifier, but every
workflow pins 3.12, so this could not surface in CI. On 3.13 the one loop whose
docstring names the banned call --- `Mesh._state_loop` --- failed its own grader
with a message telling the reader to adopt `mesh.pacing.Ticker`, which that loop
already uses. The other two parametrizations passed only because their prose
happens not to spell the call, so the grader was one documentation edit away from
the same failure.

The scan now reads the definition through `ast`, drops the docstring node, and
unparses the body. Comments are absent from the AST altogether and the docstring
is removed as a node rather than as text, so neither assumption is made. The two
graders that read a loop's source --- the three `Mesh` publish loops and the seven
`SensorLoopsMixin` sensor loops --- share that one reader; the module-level
inventory scans in the same file are unchanged, because they grade whole-file text
for a shape rather than one function's body, and already filter prose by line.

All ten loops keep every assertion they had, on both interpreters. The added
coverage grades the reader against a comment and against a docstring the compiler
did not store verbatim, both of which fail on 3.12 as well, so the fix is graded
by CI rather than only by the interpreter that reported it.
