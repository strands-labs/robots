### Changed: `replicate()` states what it does not do

There is no per-environment observation or action API. `get_observation` and
`send_action` address environment 0's robot, the only one carrying an
`Articulation` handle; the clones advance under physics and are what a renderer and
a domain-randomisation pass see, but they cannot be driven or read individually.
The success message says so on every call, so a `num_envs: 64` in the payload is not
mistaken for 64 drivable robots. Building that surface needs an articulation view
across environments and a cross-backend decision about what a batched observation
looks like, so it is deliberately out of scope here rather than half-present.
