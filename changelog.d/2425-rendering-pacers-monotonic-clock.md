### Fixed

- **simulation/mujoco, rendering**: the MJPEG stream generator and the multi-camera
  recorder thread paced their frames against two readings of `time.time()`, so a
  wall-clock step landing between them - an NTP correction, a `date -s`, a resume from
  suspend - changed the rate. A backward step stalled a real 10 fps two-camera capture
  for `interval + |step|`, writing 9 frames where 29 were due and an MP4 that declared
  a 0.9s duration for 2.85s of simulated motion; a forward step skipped the pacing for
  that cycle. Both now measure on `time.monotonic()`. The duration each recorder
  reports (`stop_cameras_recording`, `get_cameras_recording_status`) shared the clock
  and moved with the step in the same way; its base is now `started_mono`.
