### Docs: mesh discovery prose now matches the gossip-only default

`README.md` and the `strands_robots.mesh.session` module docstring both claimed
peers on the same LAN discover each other via multicast scouting automatically,
while the config builder deliberately defaults multicast scouting OFF
(gossip-only) to close the LAN-attacker enrollment surface. Both locations now
state the actual posture: same-host discovery works out of the box, cross-host
peers need explicit `ZENOH_CONNECT` endpoints, and multicast is opt-in via
`STRANDS_MESH_MULTICAST=true` (which logs a warning). `STRANDS_MESH_MULTICAST`
is also added to the README env-var table. No behavior change.
