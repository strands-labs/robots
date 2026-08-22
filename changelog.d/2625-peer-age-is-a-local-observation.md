### Fixed

A peer's presence payload no longer decides what this process reports having
observed about it. `PeerInfo.to_dict()` spread the payload (`caps`) after the
four fields the local process decides -- `peer_id`, `type`, `hostname` and
`age` -- so a heartbeat carrying any of those names replaced the local reading.

`age` is the one that is not a capability at all: `last_seen_mono`'s docstring
calls it "a local observation, never a stamp the peer sent", and `docs/mesh.md`
names it among the things the mesh decides from a duration on `time.monotonic()`
"which no NTP correction, `date -s` or resume from suspend can move". A peer
publishing `"age": 0.0` in its heartbeat reported itself perpetually fresh, so
every staleness verdict read from that field was the sender's to choose rather
than the reader's to measure.

`peer_id` is the second, and it costs a lookup rather than a verdict: it is the
key the registry files a peer under, the key `Mesh.peers_by_id` and
`Mesh.get_peer` resolve by, and the key `Mesh.peers` filters *self* out by.
Measured through `Mesh._on_presence` with two peers admitted, the second naming
the first in its payload: the registry held both, `Mesh.peers` reported one id
twice, `peers_by_id` resolved only one of them, `get_peer` returned `None` for a
peer that was present, and the second peer's capabilities answered a lookup for
the first. A payload naming the *reader's* own id disappeared from `Mesh.peers`
entirely while staying in the registry.

`caps` now merges first, so those four win a name collision. The spread's purpose
is unchanged and no honest payload changes: of the twelve keys `_build_presence`
emits, `hostname` is the only one that collides, and `update_peer` derives the
local `hostname` from that same wire value.
