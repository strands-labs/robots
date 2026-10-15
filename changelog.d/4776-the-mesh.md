### Added: the mesh

`Mesh()` puts robots on one network with no configuration: `mesh.mount(robot)` (or `Robot(..., mesh=mesh)`)
publishes a robot as a device, `mesh.discover("device(role:arm)")` finds devices by selector, and
`invoke` / `subscribe` / `connect` call a remote device's rpcs, stream its state and drive it as a `RemoteBody`
through the same run loop. Presence rides on zenoh liveliness, so a device that dies leaves the mesh; a watchdog
applies the device's `on_disconnect` when its controller goes quiet. `mesh.estop()` broadcasts an e-stop every
device latches, and only a resume from the authority that is allowed to give one lifts it. Two transports: zenoh
(peer mode on a LAN, `[mesh]`) and AWS IoT Core (`[iot]`). `mesh.tool` exposes the mesh to an agent as `use_mesh`.
