### Fixed: a policy refuses a runtime handshake value it cannot honor

`Policy.set_control_frequency` and `Policy.set_rtc_observed_delay` guarded with a
bare comparison (`hz <= 0` / `steps < 0`), so `nan` and `inf` were stored as a
control rate (neither compares `<=` to anything) and `bool` passed as an `int`
subclass. A stored `nan`/`inf` rate raised a bare `ValueError`/`OverflowError`
out of the Real-Time Chunking delay estimator on the *second* inference of a
rollout, not the first; a `True` installed a silent 1 Hz clock, and a `True` or
`2.7` step count was coerced into a chunk-seam offset the caller never asked for.
Both setters now delegate to the shared numeric domains, so every value they
accept is one the provider can use. `PolicyServer` forwards the wire rate
verbatim instead of coercing it with `float(...)`, which had re-admitted a JSON
`true` as a 1 Hz clock. New shared domain
`strands_robots.utils.non_negative_count_error` for a count whose `0` is a
first-class value.
