### Added: a mesh device serves its own rpcs, and a caller finds them in its manifest

`Mesh.mount(robot, rpcs={name: handler})` adds device rpcs to the built-in ones. Each is advertised in the device's
manifest with its argument schema, its doc line and whether it writes and arms the watchdog, and `Mesh.invoke`
resolves rpc names from the discovered manifest instead of a table compiled into the caller, so a process that never
imported the handler can call it. A plain handler is a write that arms the watchdog; `device_rpc(fn, write=False)`
marks a read and `arms=False` a write the watchdog need not watch. Writes go through the same gate as the built-in
ones: refused on an observe-only device (including the open-mesh rule), refused while e-stopped, and refused on an
id two peers hold. The device refuses unknown, missing or wrongly typed arguments before the handler runs, and a
name that would shadow a built-in (`estop`, `stop`, `hold` stay built in, `estop` still answered inline) is refused
at mount.
