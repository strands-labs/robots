### Fixed: `status` on a real arm that has not been driven yet is a verdict, not an error

`get_status()` read `is_calibrated` off the lerobot robot, which reads the
motors' calibration registers and raises `DeviceNotConnectedError` on a closed
bus. Construction opens nothing, so every real arm reported
`{"error": "FeetechMotorsBus is not connected...", "task_status": "error"}` to
the fleet snapshot until a rollout had run. Calibration is now read only when
the bus is open and reported as `None` - unknown - before that. (The check is
made on the class, not the instance: `hasattr(instance, "is_calibrated")`
evaluates the property, which is the very read being avoided.)
