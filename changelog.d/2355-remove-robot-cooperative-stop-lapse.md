### Fixed

`remove_robot` no longer reports a robot as removed when its policy worker outlived the
cooperative-stop budget. The join's `TimeoutError` was swallowed and the worker's tracking
entry deleted regardless, which left a live worker unobservable: the global scene-mutation
gate then admitted `add_object` / `set_timestep` / `add_robot` for the rest of the session,
`list_policies_running` reported nothing, and `cleanup`'s bounded join had nothing to wait
on. The rebuild is now refused with the budget waited and the retry that resolves it, and
the worker stays tracked. A lapse is distinguished from a worker that raised by
`Future.done()` rather than by exception class, since `socket.timeout` is `TimeoutError`.
