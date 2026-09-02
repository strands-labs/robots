### Added: a native CRTP driver flies a real Bitcraze Crazyflie 2.x

`crazyflie` has had a registry entry and a MuJoCo asset for a while, so
`Robot("crazyflie")` worked. lerobot has no robot type for a Crazyflie, so the
other half of that entry did not: `Robot("crazyflie", mode="real")` raised
`ValueError: Unsupported robot type: 'crazyflie'`. `CrazyflieDriver` closes it
over the driver seam, and the registry entry now declares
`hardware.driver="strands"` so a caller needs no `driver=` keyword. `cflib`
arrives with the new `[crazyflie]` extra and is imported lazily, so the module
loads on a machine with no Crazyradio. That extra is deliberately **not** a
member of `[all]`: `cflib` is GPLv3 and this project is Apache-2.0, so the
copyleft dependency is only ever installed by a caller who names it. Nothing in
`[all]`'s CI/exploration role needs it -- without `cflib` the driver still
imports, registers and reports a reason naming the extra.

A quadcopter is not a servo bus, and three properties of its command surface are
why this is a driver rather than a lerobot config -- each one a way to break the
aircraft if a caller has to guess.

**Angular velocity is rad/s here and deg/s on the wire.** Every twist in this
package is SI; `cflib`'s `Commander` takes `yawrate` in degrees per second.
Forwarding the number commands 1/57th of the requested yaw -- no crash, no error,
just an aircraft that appears not to respond. `twist_to_setpoint` is the single
conversion site, and it also picks between the two setpoint kinds, whose argument
orders disagree: `send_hover_setpoint(vx, vy, yawrate, zdistance)` puts the yaw
rate third and `send_velocity_world_setpoint(vx, vy, vz, yawrate)` puts it
fourth, so building the tuple in one place is what stops a height being commanded
as a yaw rate. A hover setpoint holds an altitude, which is the safe indoor
default; omitting `z` selects the world-velocity setpoint, which is the only way
to command a climb.

**A setpoint is a subscription, not a command.** The firmware supervisor cuts
thrust when the setpoint stream goes quiet, so a single `send_hover_setpoint`
produces a twitch rather than motion. `send_action` latches a setpoint and a
background repeater holds it at `setpoint_hz` (default 20 Hz) until something
replaces it, so the call returns once the setpoint is latched rather than once the
motion has finished. That repeater paces on `mesh.pacing.Ticker` rather than on
`stop_event.wait(period)`, for the reason that module records: a `wait` is a delay
where a rate needs a deadline, so the time each CRTP write costs is added to the
period instead of subtracted from it and the stream runs at `1 / (period +
write)`. This is the loop where that matters most, because its whole job is
feeding the supervisor's watchdog -- a stream that quietly paces slow is a thrust
cut with nothing logged anywhere.

**Stopping and landing are different verbs.** `Commander.send_stop_setpoint`
zeroes every motor, so an airborne Crazyflie falls; `HighLevelCommander.land`
descends under control. The driver contract's `stop` is "stop motion, leaving the
robot connected", and on an airframe that cannot hold still without a setpoint
stream the only motion-free state is on the ground -- so `stop`, `stop_task` and
`cleanup` all land, and `cleanup` lands before it closes the link it needs to do
so. Cutting the motors is the separately named `emergency_stop`, which the agent
tool schema deliberately cannot reach. Both high-level verbs also perform a handover the
SDK requires and a reader would not guess: while the low-level commander is
streaming it owns the setpoint priority and a high-level command underneath it is
ignored, so `Commander.send_notify_setpoint_stop` precedes `land` *and*
`takeoff`. The repeater re-sends at `setpoint_hz`, so that priority never decays
on its own -- skipping the handover is not a race but a permanent refusal, and one
the firmware reports nowhere.

**A link is only open once the aircraft says so.** `Crazyflie.open_link` is
asynchronous and never raises: it wraps its body in `except Exception` and routes
every failure -- no dongle, a switched-off aircraft, a malformed URI -- to the
`connection_failed` *callback*, reporting success later by calling `connected`
once the TOCs are down. Reading its return would therefore report a connection
that does not exist, and nothing downstream would object: `Crazyflie.send_packet`
is a silent no-op while `link` is `None`, so the arming request and every
subsequent setpoint would be discarded while each envelope said `success`. So
`connect_eagerly` waits for `connected` or `connection_failed`, bounded by
`CONNECT_TIMEOUT_S` (10 s -- bounded unlike `cflib`'s own `SyncCrazyflie`, since
an agent blocked forever reports nothing at all), returns the SDK's reason
trimmed to its actionable first line, and releases the link so a retry can have
the dongle. Waiting is also what makes telemetry work: `connected` fires only
after the log TOC is downloaded, and `log.add_config` raises `KeyError` for every
variable until it is.

Two more things are named rather than assumed. Connecting then sends
`Platform.send_arming_request(True)`, because firmware 2023.02 and later refuse
to spin the motors until it succeeds -- a driver that skipped it would connect
cleanly, accept every setpoint and produce no motion, which is the quietest
possible failure; an arming failure is reported and every flight command then
refuses. And the flight envelope is the driver's rather than the SDK's, since
`cflib` caps nothing and the firmware attempts whatever arrives: a setpoint
outside it is refused by name rather than clamped, because an operator who asked
for 5 m/s and silently got 1 m/s plans the next command around a speed the
vehicle never flew. `twist_envelope()` reports every bound, derived from the same
constants the check enforces.

Telemetry is one log block over core variables present on a bare Crazyflie
(`stateEstimate` position, `stabilizer` attitude, `pm.vbat`, `pm.state`), cached
into the `_pose` / `_imu` / `_battery` attributes the mesh reads by `getattr`. A
block that cannot start is reported and not fatal, because an aircraft with no
telemetry still flies and refusing the connection over a log variable would
ground a usable vehicle. `battery_pct` joins the triple every native driver's
status carries but is structurally `None` here: the Crazyflie's power manager
reports a 1S cell voltage and a coarse state, not a percentage, and a LiPo
discharge curve is nowhere near linear -- so the measured voltage sits in
`battery` beside it rather than being converted into a number nothing measured.

`start_task` and `run_policy` refuse rather than standing in for work in
progress: this package registers no aerial policy provider, and a quadcopter has
four propellers and no joints for a manipulation policy's action to land on.
There is also no `go_to`, because a bare Crazyflie without a Flow deck or an
external positioning system drifts freely in x and y, and offering the verb would
invite a caller to fly to a coordinate the aircraft cannot find.
