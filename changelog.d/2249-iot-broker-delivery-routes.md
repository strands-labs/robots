### Quality: cover the IoT transport routes that decide whether a message reaches the broker

`strands_robots/mesh/transport/iot_transport.py` had five unexecuted lines, all on
the path a message or a subscription has to survive to reach the broker. Three are
reachable and now pinned: the explicit `DROP` short-circuit in `put()` (not
redundant with `_should_drop`, which requires a trailing `camera/` segment, so a
bare `strands/<peer>/camera` passes it and the `qos < 0` branch is the only thing
that stops the publish), `_unsubscribe` with a handler that is already gone, and
`_unsubscribe` reaching the broker step after a failed reconnect left the client
`None` with the handler map still populated. The fourth and fifth are one
`DROP` branch that is unreachable by construction, and it is now pinned as such:
no policy entry a top-level topic layout can reach carries the `DROP` sentinel,
so the pin fails the day one does. Tests only; no library behaviour changes.
