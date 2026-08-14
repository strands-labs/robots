### Fixed: `init_device_connect_sync` reports an expired bring-up budget instead of returning `None`

The wrapper starts `init_device_connect` on a daemon thread and bounds the wait,
but discarded the answer that bound gave it: `ready.wait(timeout=30.0)` returns
`False` when the budget expires, and that boolean was the only thing separating
"the runtime came up" from "it did not". With it dropped, an expired budget fell
through to `return runtime_holder[0]` and handed the caller `None` -- past a
declared `-> "DeviceRuntime"` -- after 30 seconds, with nothing logged.

The two failure modes of one function therefore reported differently: a bring-up
that *raised* was re-raised on the caller's thread, while a bring-up that never
finished was indistinguishable from success. `Robot(...).run()` wraps this call
in `except Exception` precisely so a failed bring-up is reported, so the timeout
was the one failure that channel could not see: it printed `<peer> is online.`
with `_device_connect_runtime` set to `None`.

The wait's result is now read, and an expired budget raises a `TimeoutError`
naming the budget and what to check, joining the failure channel its sibling
already used. The budget moves to a module constant documenting why it exists,
mirroring `simulation/isaac`'s `_BODY_STATE_MAIN_THREAD_TIMEOUT_S`; a successful
bring-up is unchanged and still returns the runtime with its loop and thread
wired.
