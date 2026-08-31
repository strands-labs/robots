### Added: `g1_set_swing_height` verb for the driver's SetSwingHeight write

`G1Driver.set_swing_height` is the driver-side swing-height (walking leg-lift
clearance) entry point: a caller passes a target height in meters and the
driver publishes the SDK's raw `_Call` on API id `7103` over the same DDS
singleton `ensure_dds` opens. The Python SDK's `LocoClient` does not expose
`SetSwingHeight` as a public method - the setter is reachable only through
the raw `_Call` on API `7103`, which the neon bundle's `g1_set_swing_height`
verb (`cagataycali/neon-the-g1/tools/g1_posture.py` and the shared
`_g1_common.set_swing_height` helper) fronted under a single-writer lock.
The read-only half of that envelope already landed as
`strands_robots.tools.g1.g1_swing_height_envelope` (refs #358), and this
module is the write-side companion that hands the target to the driver.

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `set_swing_height`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the driver's
envelope verbatim — this is the same shape `g1_set_fsm` (refs #3025) and
`g1_set_stand_height` (refs #3031) already ship.

`strands_robots.tools.g1.g1_set_swing_height.g1_set_swing_height` is the
agent-facing side of that write: one duck-typed call on
`driver.set_swing_height`, the envelope the driver produced returned
verbatim, and the same live-handle refusals every write-side verb in this
package owes (`driver` is `None`, a robot *name*, or any object without a
callable `set_swing_height`). The verb adds four data-parameter refusals
on top — a `None` `height`, a non-numeric shape, a `bool` payload (which
would coerce to a silent `0.0` shuffle or `1.0` far-past-envelope gait),
and a non-finite value (`nan` / `inf`) — validated through the shared
`finite_number_error` (NOT `positive_finite_number_error`, because
`0.0` is a caller-facing value the neon bundle's own wrapper admits as
the minimum-clearance shuffle gait, and refusing it here would drop a
caller's most conservative locomotion command; a negative value is not
refused either because the neon bundle's own wrapper rounded it up to
`0.0` rather than treating it as a shape violation, so no caller-facing
sentinel encoding lives on that side of the domain the way HighStand
does for stand height). Importing the module pulls no `unitree_sdk2py`
submodule (the package's SDK-load-hygiene contract, refs #358), and the
module docstring names the six things this verb does not do (clamp
`height`, encode the API-7103 dispatch, decode the SDK's `rc`, restate
the driver's refusal wording, schedule sequences, check the
`WALK_FSMS` gate) so a caller reading it does not misread the surface.

The test suite
(`tests/drivers/test_g1_set_swing_height_writes_the_driver_envelope.py`)
grades sixteen shapes: the SDK-load-hygiene pin, two driver-side
envelopes as pass-through (a driver-side refusal, a future success
envelope), the two live-handle refusals through the shared
`live_handle_refusal` guard, five data-parameter refusals (missing
`height`, wrong-shape `height`, `bool` `height`, `nan` `height`,
`inf` `height`), three admitted-value pins (a `0.0` shuffle-gait
target, a negative value passed through unchanged, and a value above
the neon envelope's `0.2` upper bound passed through unchanged - all
three reach the driver verbatim because the module docstring names
"does not clamp" as one of the things this verb does not do), one
exactly-once write pin, and one arguments-pass-through pin. The
universal auto-discovery test at
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py`
grades the verb the moment it lands (its first parameter is `driver:
Any`), so the two live-handle refusal rules are held by that suite
too.
