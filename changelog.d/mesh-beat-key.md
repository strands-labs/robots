### Fixed: on a mesh without identities, another machine naming the mover's peer id no longer delays its hold

The mesh watchdog counted any request that named the mover's peer id as a sign of life from that mover. Where peer ids
are only claims (`zenoh://?trusted=1`, `STRANDS_ZENOH_TRUSTED=1`, `mem://`, AWS IoT), another machine could keep a
silent mover's robot from applying `on_disconnect` by sending requests under its id. Now the reply to the `act` that
arms the watchdog carries a beat key (envelope field `beat`), the `Mesh` sends it back on every request to that
device, and only a request carrying it counts as the mover's. A mesh with identities is unchanged: there a signed
`peer` is enough. Update every machine on such a mesh together: a mover on an older version sends no key, so its robot
holds `watchdog_ms` after its last `act`.
