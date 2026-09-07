### Added: `get_peer(..., max_age_s=)` - the caller states the freshness it needs

A peer record older than the caller's bound answers `None` exactly as if the
peer were unknown, because for that caller it is: a dispatcher must not assign
work on a forty-minute-old sighting without saying so. `None` (the default)
accepts any age - the historic behavior, still right for displays that render
staleness themselves off the row's `age` and `reachable`. The bound is held to
the shared positive-finite domain both on the session function and on
`Mesh.get_peer`, which forwards rather than re-validating: `nan` makes the age
comparison answer `False` for every record - a bound that never trips, failing
open on exactly the stale record it exists to refuse - and `True` would be a
silent one-second bound.
