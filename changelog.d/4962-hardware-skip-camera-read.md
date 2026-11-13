### Changed: a hardware step reads the cameras only when its policy looks

`Robot.run` on hardware now follows `needs_observation()` as it does in simulation: a step whose policy will not look
(a `ChunkedPolicy` replaying its chunk, or a policy that answers `False`) gets an observation without frames instead
of copying every camera into it. Each camera's capture thread keeps draining the device meanwhile, so the next step
that looks gets a fresh frame. Recording and `stop_when` still read the cameras on every step.
