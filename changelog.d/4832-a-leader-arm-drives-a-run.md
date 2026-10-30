### Added: a LeRobot leader arm drives a strands robot in one line

A LeRobot `Teleoperator` (or a `Robot` being mirrored) can be handed straight to `run()`:
`arm.run(leader, hz=30, seconds=60)` is the whole teleop program. Units come from the leader's own bus
norm modes and the follower converts through its own calibration, matching LeRobot's `teleoperate` to
within one encoder count. A source with no declared units is refused with the fix, and a LeRobot
`<joint>.pos` dict handed to `act()` is refused naming the units problem (`act()` takes a `<joint>`
target in radians).
