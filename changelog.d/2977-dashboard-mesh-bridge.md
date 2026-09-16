### Added: the `strands_robots.dashboard` mesh bridge

`mesh_bridge` is the dashboard's mesh peer: it joins the fleet through
`mesh.session`, mirrors presence/state/camera/sensor topics into a fleet
snapshot, fans events out to async consumers with per-type coalescing, runs
point-to-point commands with the same responder scoping `Mesh.send` enforces,
and carries the signed safety rail (`signed_estop` / `signed_resume`) over a
robot-less gateway `Mesh`.

Extracted from draft #2848 as a slice of the #2977 decomposition, with the two
mesh-safety findings from that draft's review fixed on arrival (the signed
e-stop reports what the peers said, and the signed rail honours the mesh kill
switch - each has its own fragment). Both of the bridge's session-opening paths
ask `mesh_disabled_by_env()`, the predicate every construction site answers,
and the e-stop grading is imported from `mesh.core` rather than copied.

The module imports nothing the `[dashboard]` extra does not already supply -
its mesh-side imports are lazy, matching the "documented no-zenoh path" the
mesh package keeps - and it sits behind the package gate in
`strands_robots/dashboard/__init__.py` all the same, so a caller without the
extra gets one refusal naming it. The HTTP/WebSocket surface that serves this
bridge (`dashboard.server`) is a later slice; until it lands the
`peer_annotations` / `protected_peer_ids` / `managed_children` hooks simply
stay unset, which the bridge is built to tolerate.
