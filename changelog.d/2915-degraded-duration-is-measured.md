### Tests: a degraded probe's `for_seconds` is graded against a measured interval

`Mesh._degraded_probes` publishes `for_seconds` so an observer on another
machine can tell "this probe just failed" from "this probe has been failing
since it was plugged in" -- two conditions the method's own docstring says want
different operator responses. Every assertion the suite made about that field
was satisfiable by a renderer returning the literal `0.0`: `>= 0.0` in the
reason test, `< 60.0` in the wall-clock test, and the key-set check. So the
field's presence and its clock *domain* were graded while nothing asserted the
duration had been measured, and a renderer with the right clock vacuously -- one
that subtracts a key no record carries -- is green under it.

Measured with `record.get("since", now)` substituted for
`record.get("since_mono", now)`, which is that shape and makes `for_seconds`
always exactly `0.0`: this file and its sibling sweep report 32 passed before
the new pin and `1 failed, 32 passed` after it, the failure naming the constant.

`_degraded_probes` is unchanged -- it is correct, and what was missing was the
assertion that keeps it correct. The clock is driven rather than slept on, so
the interval is exact and the pin costs no runtime.
