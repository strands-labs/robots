### Fixed

- **examples/fleet**: `03_failover_and_degraded_ops.py` awaits the survivors' estop
  evidence on a bounded monotonic deadline instead of reading the audit log once, so
  the re-sync drill no longer fails when a peer's `remote_estop_engaged` row lands
  after the issuing call returns.
