### Fixed: the watchdog retries an on_disconnect that raised, and its thread survives it

`Watchdog.poll()` marked a lapse as handled before it applied `on_disconnect`, and the `Watchdog.start()` loop had no
`try`, so one exception from the apply (an I/O error, or a status reply still corrupted after the driver's read
retries) ended the watchdog thread with nothing written, and no later lapse was applied. A failed apply now leaves
the lapse open: the thread counts it in `Watchdog.errors`, reports each run of failures as one `fault` event (an
error that cannot print itself is named by its type) and retries every period until the robot is closed. `fault`
subscribers run on that thread, so one that blocks delays the retry, and one may call `Watchdog.stop()`; closing the
robot during an apply ends the thread without a fault. A lapse while the e-stop is latched is left to it: `poll()`
reads and writes nothing, so a read that fails during the stop is not retried into a write after `resume()`, and
neither is an apply that an e-stop interrupts. `start()` converts `period_s` with `float()` and refuses a period
outside `(0, threading.TIMEOUT_MAX]`, and `Watchdog()`, so also `Mesh.mount(watchdog_ms=)`, refuses a NaN or infinite
`timeout_ms`: a `Watchdog` with a NaN applied `on_disconnect` at the first poll after every beat and one with an
infinity never did, and under `Mesh.mount` neither ever applied it.
