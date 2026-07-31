### Fixed: an aborted Isaac camera warm-up is no longer reported as a slow render product

`IsaacSimulation._warmup_camera` returned `False` two ways -- the step budget ran
out while the RTX render product never accumulated, or the loop broke early on an
exception -- and reported both with the same text: *"did not produce a valid frame
after N warm-up step(s); the first render() may return an error until the RTX
product accumulates a frame."* The exception itself was logged at DEBUG, so an
operator at default log level saw only a message telling them to give the
renderer more time, for a loop that had already stopped and would never
accumulate anything.

An early abort now names the step it reached, the budget it had left, and the
exception that ended it. The exhaustion report is unchanged, and the step count
it quotes is now the number of iterations the loop actually runs (previously the
raw `n_steps`, which reads `0` for a call that still takes one step).

The camera-warm-up regression pins never reached the code they pin: the
`__new__` skeleton they drive omitted `_cameras`, which the loop reads to decide
whether the secondary render products need an explicit flush, so every one of
them aborted on its first step through exactly the path above. The skeleton now
carries every attribute the loop reads, the two `wrist_image` cases register the
real two-camera shape, and `_refresh_all_render_products` has coverage for the
first time.
