### Fixed: the dashboard reports the camera rate the mesh publishes at, not a second reading of the same variable

`STRANDS_MESH_CAMERA_HZ` is the mesh's knob: `Mesh._resolve_camera_hz` reads it
through `mesh.session.hz_from_env` and disables the camera loop when it is unset,
non-positive, or a value no loop can pace itself with. The dashboard is the
surface that *writes* that variable - the settings panel holds `camera_hz` and
`settings.apply_mesh_env` pushes it into the environment the peers read - and its
mesh posture payload read it back through a bare `float()`. Six of nine spellings
an operator can type disagreed between the two: `-5`, `nan`, `inf` and `1e999`
were echoed to the panel as live rates while the peer published nothing, and
`nan`/`inf` are not JSON, so `/api/mesh/config` emitted a body no conformant
client can parse. A typo or whitespace raised `ValueError` out of the handler
instead, taking the endpoint that would have shown the posture with it.

The bridge's own `STRANDS_DASHBOARD_*` rate and TTL knobs are resolved at import,
where a bare `float()` costs more: a typo raised `ValueError` while the module
body executed, so the mesh bridge did not lose one knob but failed to import,
from a frame naming `float` rather than the variable. A non-finite value was
accepted instead and reached each consumer as one side of a comparison it removes
rather than widens - `age > ttl` is `False` for every age against both `nan` and
`inf`, so a robot that left the fleet was never aged out of the snapshot.

Both now ask the owner of the question: the camera rate through
`mesh.session.hz_from_env` with the publisher's documented fallback, and the
dashboard's own knobs through the shared `utils.finite_number_error` domain, with
every rejection reported so a substituted default is not silent. Each knob's
floor is left to its consumer, as before: `prune_peers` reads a non-positive TTL
as "never prune" and the event coalescer reads a non-positive rate as "no
ceiling".
