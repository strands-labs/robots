### Fixed: `STRANDS_MESH_STREAM_HZ=0` no longer fails robot construction

The per-step telemetry throttle divided the environment value directly, so
`STRANDS_MESH_STREAM_HZ=0` raised `ZeroDivisionError` and a non-numeric value
raised `ValueError` - from `HardwareRobot.__init__`, failing every
`Robot(..., mode="real")` construction, and from the simulation `run_policy`
hook setup. Both now resolve the rate through the shared mesh domain check:
non-positive turns step publishing off (the spelling `STRANDS_MESH_CAMERA_HZ`
already uses), and an unusable value is logged with the variable and the
offending value named, then leaves publishing off.
