### Fixed: `RosbridgeRobot` now shares the drive-command numeric domains with the other mobile-base bridges

`RosbridgeRobot.drive` and its constructor validated their numeric inputs with an
inline `math.isfinite` test instead of the domain helpers `RosBridgedRobot` and
`RtpsRobot` both call, so the newest bridge accepted and refused a different set
of values than the two it copies its contract from. An inline finiteness test is
looser and less safe than the helper it stands in for: it treats `bool` as a
number, so `drive(linear=True)` published a Twist commanding 1.0 m/s and returned
success, and `RosbridgeRobot(publish_rate=True)` installed a silent 1 Hz command
stream; it raises a bare `TypeError` for a value it cannot coerce, so
`drive(linear="0.5")` escaped the structured tool-result contract of the bound
`drive_<node>` agent tool rather than naming the parameter; and `count` was not
checked at all, so `count=0` published nothing and `count=2.7` reported a raw
publish-loop error attributed to a transport the caller never invoked.

All three bridges now call the same `finite_number_error`,
`positive_finite_number_error` and `positive_whole_number_error` domains, report
byte-identical text for the same bad value, and publish nothing when a command is
refused. A structural test pins the mechanism, so a fourth transport cannot ship
with a hand-rolled copy.
