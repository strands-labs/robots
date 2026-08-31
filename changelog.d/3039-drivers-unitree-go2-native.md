### Added: native CycloneDDS driver for the Unitree Go2 quadruped

lerobot has no robot type for the Go2 and this package shipped no native
driver for it, so there was no way to reach one at all.
`strands_robots.drivers.go2.Go2Driver` closes that gap. It satisfies
`HardwareDriver`, registers itself for `unitree_go2` (and its `go2` alias),
and the registry entry now declares `hardware.driver = "strands"` so
`Robot("go2", mode="real", port=..., network_interface=...)` builds it
without a `driver=` keyword.

This is a separate driver rather than `g1.py` with a different joint table,
and both reasons are on the wire. The G1 writes `unitree_hg` `LowCmd_`
(`mode_pr`, `mode_machine`); the Go2 writes `unitree_go` `LowCmd_` (`head`,
`level_flag`, `gpio`). The header bytes the `unitree_go` constructor leaves
unset are written on every frame, because firmware drops a frame without them
before CRC is even considered - a failure that looks exactly like publishing
to the wrong topic.

The second reason is the one worth reading twice. `LowCmd_.motor_cmd` is
indexed by Unitree's `LegID` order - front-right, front-left, rear-right,
rear-left - while the Go2's own URDF/MJCF description declares its joints
front-left, front-right, rear-left, rear-right. Both orders hold the same
twelve joint names, so a driver that zipped a description-ordered vector onto
`motor_cmd` would command the mirror-image legs with correct gains and a
correct CRC, and nothing in any log would say so. `GO2_JOINT_INDEX` is the
single place those two conventions are reconciled, `send_action` is keyed by
joint name and never by index, and the telemetry `state` reports is keyed by
the same names so the read path cannot be transposed either.

The write gate is the sport-mode release rather than an FSM id. The Go2's
onboard sport service owns the legs until it is released, and publishing
`rt/lowcmd` alongside it puts two controllers on the same twelve motors.
Every SDK example tests one key for that state - `CheckMode()`'s
`result["name"]`, empty when no mode holds the robot - so the gate reads the
only key the SDK evidences and needs no wire guess, unlike the G1's integer
FSM id. `release_sport_mode()` polls release-then-verify because the release
is asynchronous, and `send_action`, `run_policy` and `start_task` all refuse
until it confirms. Releasing is deliberately not folded into
`connect_eagerly()`: it changes who controls the robot, which is not a side
effect of connecting to read.

`run_policy` rolls a caller-built policy on a 500 Hz thread, re-gates every
step, and publishes a zero-gain *but still enabled* soft-stop frame on every
exit path - a Disable frame would cut the motors dead and drop the robot onto
its knees. That loop needs no FSM refresher thread, unlike the G1's: every
input to the Go2's gate is an in-memory read rather than a synchronous DDS
round trip, so the per-step re-gate happens inline and there is no staleness
bound for a cache to fall behind.

Nothing in the module imports `unitree_sdk2py` at load time; every SDK touch
is inside a function body, so it imports on CI and on a dev box with no SDK.
