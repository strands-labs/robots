### Fixed: a joint an Action leaves out holds its last commanded target instead of sagging

`arm.act({"elbow_flex": 0.4})` filled the five joints it does not name with their *measured* positions on every call,
so each step re-targeted wherever gravity had already pulled them: on the so101, `shoulder_lift` dropped 0.46 rad in
200 steps it was never asked to move, and every partial function policy (`lambda obs: {"elbow_flex": 0.4}`) drifted
the same way. An omitted joint now keeps the last target written to it (`act({})` is a hold at the last targets). The
measured pose is still the fill before the first write and after `hold()`, an e-stop, `calibrate()` or `sim.reset()`,
so an arm never swings back toward a target from before a stop, a reset or a hand-moved calibration.
