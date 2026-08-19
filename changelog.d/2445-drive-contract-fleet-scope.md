### Fixed

- **mesh**: `RosbridgeRobot.drive` and the rosbridge integration guide presented this bridge's velocity clamp, `max_duration` ceiling and trailing zero Twist as the fleet-standard drive contract shared with `RosBridgedRobot.drive` and `RtpsRobot.drive`. Only the numeric domains and the single-shot latch are shared; the other two bridges accept no ceilings and publish no trailing zero, so a timed drive there leaves the last velocity latched rather than self-stopping. Both surfaces now scope each guarantee to where it holds, and a guard grades the prose against all three bridges.
