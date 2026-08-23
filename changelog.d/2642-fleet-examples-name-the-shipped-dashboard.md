### Fixed: two fleet examples no longer promise a dashboard that already ships

`examples/fleet/03_failover_and_degraded_ops.py` and
`04_emergency_evacuation.py` described the read-only Rerun fleet dashboard as
future work ("shows the safety events live once it lands", referencing #2181),
but it landed in `examples/fleet/dashboard.py`. A reader working through the
fleet suite in order was told to wait for a file sitting in the same directory.
Both docstrings now name the artifact in the present tense, which is also the
spelling that cannot go stale again: prose naming a path stops depending on
issue state. The still-accurate forward reference to the Isaac adapter (#2123,
open) is unchanged. (#2642)
