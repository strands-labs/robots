### Fixed

- **mesh/transport**: `ZenohTransport` now publishes and releases through the raw
  Zenoh path instead of the backend-aware `session.put` / `session.release_session`.
  Under `STRANDS_MESH_BACKEND=bridge` those resolve the `BridgeTransport` that owns
  the transport, so a publish re-entered the bridge until the stack was exhausted -
  delivering nothing to the LAN, republishing a bridged topic to MQTT once per
  re-entry, and returning normally - and a release blocked on the transport
  factory's non-reentrant lock, so teardown never completed and the session was
  never closed.
