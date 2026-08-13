### Docs: the architecture diagram and the camera-offload docstring no longer call multicast the discovery default

`scouting_block()` ships `scouting/multicast/enabled = false` with gossip on
unless an operator sets `STRANDS_MESH_MULTICAST=true`, and `mesh.core` logs a
fleet-takeover warning when they do. Two places still described the opposite:
the layer-5 transport label of `examples/lerobot/architecture.svg` read "Zenoh
multicast (default)", and the `strands_robots.mesh.iot.camera_offload` module
docstring called the Zenoh path "LAN multicast". Both now name the real posture -
the diagram reads "Zenoh gossip (multicast opt-in)" and the docstring says the
Zenoh path serves gossip-scouted LAN peers - so a reader is no longer told the
fleet is discoverable on the LAN when it is not.

A guard now judges the rule per prose *block* - one docstring, one contiguous
comment run, one Markdown paragraph, one SVG label - across the mesh package,
`README.md` and the repository diagrams: a block that mentions multicast must
either name the `STRANDS_MESH_MULTICAST` opt-in or say it is off. Block
granularity is what lets it ship with no exclusion list, since `mesh.core`'s
warning names the flag in the same block as the words "Multicast scouting is
ON". Device Connect's D2D pages stay out of scope, pinned by a test that the
package configures no Zenoh scouting key. No behavior change.
