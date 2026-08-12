### Added: fleet cross-zone transport example + read-only Rerun fleet dashboard

`examples/fleet/02_cross_zone_transport.py` splits a site into two ownership
zones, each an in-process mesh peer owning its own robots, and has a fleet
coordinator decompose a cross-zone transport into per-zone legs joined at a
handoff dock. Each leg's robot is selected zone-side by running the ONE shared
`transport` skill definition through the suite's `capabilities.py` filter - the
identical skill artifact executes in both zones by different robot types, so
there is zero per-zone skill code to fork. Legs dispatch over `mesh.send`
behind a human-in-the-loop gate in custody order: leg 2 is never dispatched
before leg 1's success reply, and an aborted or declined handoff reports
exactly where the payload is. `examples/fleet/dashboard.py` is the suite's
read-only fleet dashboard (epic D9): a subscribe-only mesh peer that watches
presence/health/safety topics, tails the signed audit log, and renders in
Rerun when `rerun-sdk` is installed - degrading loudly to a terminal table
when it is not. Read-only is enforced rather than narrated: every
command-capable method on the peer refuses, and raw `publish` is confined to
the peer's own namespace. (#2181, epic #2179)
