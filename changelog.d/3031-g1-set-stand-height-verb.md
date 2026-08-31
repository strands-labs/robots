### Added: `g1_set_stand_height` verb for the driver's SetStandHeight write

`G1Driver.set_stand_height` is the driver-side stand-height entry point: a
caller passes a target height in meters and the driver publishes the SDK's
`LocoClient.SetStandHeight` call over the same DDS singleton `ensure_dds`
opens, or - when the caller passes a negative sentinel - falls back to
`LocoClient.HighStand` which uses a `UINT32_MAX` height sentinel the raw
SDK exposes only as a bare method call. The negative-value fallback is the
neon bundle's `g1_set_stand_height` verb's one addition over
`use_unitree(service='loco', operation='SetStandHeight', ...)`: it lets a
caller name "the tallest stance the robot admits" without knowing the
SDK's own sentinel encoding.

The driver's method itself is not yet plumbed on `G1Driver` today (refs
#358 for the SDK-facing gate work the write belongs on), so the
`live_handle_refusal` grader refuses a handle without a `set_stand_height`
accessor with a message naming the verb, the `driver` parameter and the
accessor. Once the driver method lands the same call returns the driver's
envelope verbatim — this is the same shape `g1_set_fsm` (refs #3025) and
`g1_start_task` (whose driver method refuses with a registry-not-wired
string today) already ship.

`strands_robots.tools.g1.g1_set_stand_height.g1_set_stand_height` is the
agent-facing side of that write: one duck-typed call on
`driver.set_stand_height`, the envelope the driver produced returned
verbatim, and the same live-handle refusals every write-side verb in this
package owes (`driver` is `None`, a robot *name*, or any object without a
callable `set_stand_height`). The verb adds four data-parameter refusals
on top — a `None` `height`, a non-numeric shape, a `bool` payload (which
would coerce to a silent `0.0` LOW or `1.0` near-max stance), and a
non-finite value (`nan` / `inf`) — validated through the shared
`finite_number_error` (NOT `positive_finite_number_error`, because a
negative value is the caller-facing signal that selects the SDK's
`HighStand` sentinel). Importing the module pulls no `unitree_sdk2py`
submodule (the package's SDK-load-hygiene contract, refs #358), and the
module docstring names the six things this verb does not do (decide which
heights the SDK admits, encode the `UINT32_MAX` sentinel, decode the
SDK's `rc`, restate the driver's refusal wording, schedule sequences, run
a second gate) so a caller reading it does not misread the surface.

The test suite
(`tests/drivers/test_g1_set_stand_height_writes_the_driver_envelope.py`)
grades seventeen shapes: the SDK-load-hygiene pin, three driver-side
envelopes as pass-through (a driver-side refusal, a future success
envelope, a future `HighStand`-fallback success envelope), the two
live-handle refusals through the shared `live_handle_refusal` guard, six
data-parameter refusals (missing `height`, wrong-shape `height`, `bool`
`height`, `nan` `height`, `inf` `height`), two admitted-value pins (a
`0.0` LOW-stance target and a negative `HighStand`-fallback target both
reach the driver verbatim), one exactly-once write pin, and one arguments-
pass-through pin. The universal auto-discovery test at
`tests/tools/g1/test_a_live_handle_verb_refuses_a_wrong_handle.py` grades
the verb the moment it lands (its first parameter is `driver: Any`), so
the two live-handle refusal rules are held by that suite too.
