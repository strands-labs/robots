### Fixed: a GR00T action chunk's horizon is one every value can answer for

`_action_chunk_horizon` checked that **every** value in a GR00T action chunk
carries a leading time axis, then read the horizon from whichever value the
producer happened to serialize first. Both unpack paths iterate
`range(horizon)` and index every value at each step, so a chunk whose values
cover different horizons had two different outcomes for the same data,
selected by the producer's key order: with the longest value first the loop
indexed past the end of every shorter one and raised `IndexError: index 8 is
out of bounds for axis 0 with size 8` from inside the loop, naming no key --
precisely the opaque failure the sibling 0-D check exists to prevent. With the
shortest value first it returned a success carrying 8 of the arm's 16
commanded steps, the trailing steps dropped without a log line, and the
consumer re-queried as though the whole chunk had run.

The property that is returned is now checked, the same way the 0-D property
was already checked across every value. A disagreeing chunk is refused with a
`ValueError` naming every key, its horizon and its shape, so an operator can
tell which head is short without re-running inference. It is refused rather
than truncated to the shortest value: the steps a longer value carries are
commands the model produced, so trimming them would execute part of a
trajectory and re-query as if the whole chunk had completed. The refusal is
identical whichever key the producer serialized first.

`docs/policies/groot.md` gains a chunk-shape table documenting both refusals.
