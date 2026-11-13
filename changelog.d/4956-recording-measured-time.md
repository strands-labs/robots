### Added: a recording shows when each frame was read, so stalls and skipped ticks are visible

A recorded dataset had only LeRobot's `timestamp` (`frame_index / fps`), so a 300 ms stall at 30 Hz read back as an
even 33.3 ms between every frame. Each frame now also stores `time.measured`: its observation's read time on the
robot's clock, in seconds from the episode's first frame. `timestamp` is unchanged, so video decodes and the dataset
loads in LeRobot as before, and the key is not an `observation`/`action` feature, so no policy trains on it. A dataset
recorded without it keeps taking episodes without it.
