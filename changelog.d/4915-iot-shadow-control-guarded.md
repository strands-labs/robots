### Fixed: the IoT `control` shadow is opt-in, and it follows the mesh's observe-only rule and watchdog

`IotOptions(control=False)` is the new default: a mount keeps its `manifest` and `state` shadows, and only a mount
that passes `control=True` acts on `control` shadow targets. When it does, a target arms the mount's watchdog the way
an `act` rpc does (a writer that stops writing for `watchdog_ms` gets the robot's `on_disconnect`) and is logged to
`strands_robots.audit`. A real robot mounted with `watchdog_ms=None` refuses every `control` target by name, as it
already refused `act` and `hold` over the mesh. Mounts that drive robots through the `control` shadow must now pass
`IotOptions(control=True)`.
