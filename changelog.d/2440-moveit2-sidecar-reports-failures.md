### Fixed

- **policies/moveit2**: the ZMQ sidecar reference implementation now reports every
  planning failure as `{"success": false, "status": ...}` and answers every request it
  receives. Only `component.plan()` was guarded, so a failure at the start-state read,
  the goal set (a joint the planning group does not have reaches this through the
  client's syntax-only name check) or the trajectory serialisation escaped the REP loop
  and ended the sidecar with the in-flight reply never sent - taking planning down for
  every client, not just the one that sent the request. A payload that decodes cleanly
  but is not a map (the single byte `0x2a` is the integer 42) reached the same fate one
  step earlier, when the endpoint was read off it; it now comes back in the same
  `malformed_request:` class as bytes that do not decode at all.
