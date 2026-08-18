### Fixed

- **simulation/isaac**: measure the live preview's refresh cadence and the duration a camera
  recording reports on `time.monotonic()` instead of `time.time()`. Both subtracted two
  readings of a wall clock, so a step landing between them changed the answer: a backward
  step of 2 s stopped `run_pump_forever` refreshing the preview for 1.55 s of a 1.95 s run
  while it kept draining the app, and a 3.0 s capture reported `after 33.0s`, `after 1.0s`
  or `after 3603.0s`. The recording state's duration base is now spelled `started_mono`, the
  name the MuJoCo backend's two recorders already carry.
