### Fixed: a clock that raises never hides an e-stop, and `Robot(clock=)` refuses what is not a `Clock`

On a hardware `Robot(clock=)` whose `now()` raises, `estop()` latched and ran the body's own stop, then
re-raised the clock's error before it emitted the `estop` event: a local press, or a `broadcast("estop")`
on the tenant key alone, reached no subscriber and no mesh peer, and the broadcast logged that the device's
stop raised. `estop()` now emits `estop`, then a `fault` event naming the clock's error, and returns;
`estop_latch.engaged_at` stays `None`. A body whose own stop raises still raises. `Robot(clock=)` refuses
what is not a `Clock` instance (`time.monotonic`, a lambda, the `MonotonicClock` class) with a `TypeError`
that says what to pass, where the first `act()` used to write and then fail.
