### Fixed: the mesh kill switch is asked where a session is opened, not only where a Mesh is constructed

`STRANDS_MESH=false` is documented as a hard kill switch, and
`mesh_disabled_by_env` states that "every path that can open one answers this -- not
only `init_mesh`". That held for the two places a `Mesh` is *constructed*
(`init_mesh` and the `robot_mesh` gateway peer, which each asked separately) and not
for the one place a session is *opened*: neither `mesh.session.get_session()` nor
`_get_zenoh_session_directly()` consulted it, so `ZenohTransport`, the bridge
transport factory and any direct caller of the documented `get_session()` entry point
reached `zenoh.open` with the switch engaged.

The result was not one quiet extra peer. With no explicit endpoints that path
*listens* on `STRANDS_MESH_PORT`, so the process the operator had disabled the mesh on
became the machine's hub and every later process on the box connected to it as a
client. Both acquire doors now ask the switch before opening anything, and
`get_session` asks it before the backend branch, so an IoT/bridge transport - which
advertises the robot to the fleet exactly as a Zenoh session does - is refused by the
same gate.
