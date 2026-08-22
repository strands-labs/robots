### Fixed

- **`Robot.get_status()` now reports each camera's own connection reading, so an unhealthy arm names
  which stream is down.**
  The probe's `cameras` field was built from `robot.config.cameras` -- the camera names the device was
  asked for -- while the measured reading sat one attribute away on `robot.cameras[name].is_connected`.
  Because `is_connected` on a lerobot arm is `bus and all(cameras)`, it collapses N+1 independent facts
  into one boolean, and with only the configured enumeration beside it four physically distinct faults
  produced one identical report: a single dropped camera, every camera dropped, the motor bus down, and
  a camera whose probe raises all answered `is_connected=False` with the same `cameras` list. A new
  `cameras_connected` maps each live camera to its own reading, so a camera reading `False` is the
  culprit and every camera reading `True` leaves the motor bus as the one by elimination. `cameras`
  keeps answering which streams the device is configured for, unfiltered, so the difference between the
  two names exactly the cameras whose state could not be read: a camera whose probe raises is omitted
  rather than reported as `False` -- which would claim a measurement the device never made -- mirroring
  the observation path, and the rest of the status still arrives.
