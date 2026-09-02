### Fixed: both native Unitree drivers open their motion-switcher client under the shared DDS lock

`Init()` on a Unitree RPC client builds its DDS request/response endpoints, and
the CycloneDDS bindings segfault when an endpoint is constructed concurrently
with another -- the pair `_DDS_INIT_LOCK` exists to serialise. Neither
`G1Driver` nor `Go2Driver` held that lock while opening the client, while both
construct subscribers under it on their own streaming, rollout and
mesh-telemetry threads, so a single driver instance raced itself. The loss is a
native segfault, which the drivers' "record the error and stay usable for reads"
boundary cannot turn into an error envelope: the process dies, possibly while
the robot stands under its own controller.

Both open paths, injected factory included, now hold the shared lock; the lazy
SDK import stays outside it, because an import creates no endpoint. Measured on
the engine with 40 subscribes against 40 opens on one driver, concurrent
endpoint constructions go from 79 to 0.

The G1 site was additionally invisible to a rule grading endpoint construction
over the source, because it called through a lower-case local binding
(`init = getattr(client, "Init", None); init()`); it is now spelled
`client.Init()`, and the regression pin grades the behaviour rather than the
spelling.
