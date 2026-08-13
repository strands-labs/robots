### Fixed: the mesh peer-discovery example now terminates

`examples/04_mesh_peer_discovery.py` released its Zenoh session through
`getattr(sim, "_mesh", None)`, but the `Robot` factory assigns the public
`sim.mesh` (`strands_robots/robot.py`), and the SDK's own teardown reads
`getattr(instance, "mesh", None)` before calling `stop()`. `_mesh` never
exists, so the read returned `None`, the `if mesh:` guard skipped, and the
session stayed open on non-daemon threads: the example whose docstring
promised "~3 seconds" ran until it was interrupted (measured: killed at 40s
versus exit 0 in 1s once the session is released). Only the default
mesh-enabled path was affected -- `STRANDS_MESH=0` already exited.

Because the name was read through `getattr` with a default, nothing raised.
A new guard derives the attribute the factory actually assigns and refuses any
example that reaches for a mesh attribute nothing assigns, while leaving a
module's own `self._mesh` private state alone.
