### Fixed: the watchdog retries an on_disconnect that raised, and its thread survives it

`Watchdog.poll()` marked a lapse as handled before it applied `on_disconnect`, and the `Watchdog.start()` loop had
no `try`, so one exception from the apply (a corrupted status reply, an I/O error) ended the watchdog thread with
nothing written, and no later lapse was applied. A failed apply now leaves the lapse open: the thread counts it in
`Watchdog.errors`, reports each run of failures as one `fault` event and retries every period until the robot is
closed. `fault` subscribers run on that thread, so one that blocks delays the retry. `start()` converts `period_s`
with `float()` and refuses a period outside `(0, threading.TIMEOUT_MAX]`, and `Watchdog()`, so also
`Mesh.mount(watchdog_ms=)`, refuses a NaN or infinite `timeout_ms`: a NaN applied `on_disconnect` at the first poll
after every beat, and an infinity never did.
