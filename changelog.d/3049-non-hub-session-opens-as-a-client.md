### Fixed: a non-hub mesh session opens as a client, so the hub relays sibling traffic

The machine's first process wins the `STRANDS_MESH_PORT` listener and becomes
the hub; every later process falls back to connecting to it. That fallback
opened in Zenoh **peer** mode on an ephemeral listener, and a Zenoh 1.x peer
assumes a full mesh -- it will not take traffic relayed by an intermediary, not
even by a router. So a child heard nothing a sibling child published, while its
own child-to-hub topics kept working, because that is the link the child opened
itself. `routing/peer/mode`, the knob that used to make peers route for each
other, no longer exists in 1.10: `insert_json5` raises `ZError("unknown key")`.

Measured on three throwaway sessions (hub, publisher, then a late subscriber,
hub as the only configured endpoint, counting frames on
`strands/<peer>/input/<device>`): 0 of 62 frames arrived with peer-mode
children and 42 of 62 with client-mode children, under both a peer hub and a
router hub. The child mode decides delivery; the hub mode does not. On a bench
of arms this is the teleop failure where a leader publishes hundreds of frames
to a follower whose counters all read zero.

A client delegates routing to whatever it is connected to, so the hub relays
for it. The property peer mode was chosen for is kept: with
`connect/exit_on_failure=false` plus `connect/retry`, a client re-links to a
restarted hub on its own.

The two builders of that fallback also disagreed, which is the second half of
this fix. `_get_zenoh_session_directly` opened a client with no
`connect/retry` and no `exit_on_failure=false`, so when the hub process died
the peer it opened went permanently dark -- no reconnect loop, no surfaced
error, just a session that had stopped carrying anything. Both sites now go
through one `_apply_fallback_topology`, so the topology is stated once and
neither can drift from the other.

`STRANDS_MESH_FALLBACK_MODE=peer` restores the previous topology for an
operator who wants direct peer links, and who must then arrange them, since a
peer only hears publishers it is directly linked to. An unrecognised value
warns and stays on the default rather than raising: a typo in this variable
must not take a robot's mesh offline.
