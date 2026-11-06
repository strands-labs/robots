### Fixed: a Feetech arm holds every servo where it is before torque comes on

Torque on drives each servo to the Goal_Position it already holds, and after power-up that goal is 0 on every servo of
an SO-101 (measured: 70-269 deg from where each joint sat). `FeetechDriver.write()` turned torque on and sent the new
goals in the next packet, so for the gap between the two every joint drove toward that goal at the servo's own speed,
outside the `max_speed` cap: on a real SO-101 a gripper-only command moved the whole arm. Torque now comes on through
`FeetechBus.engage()`: read Present_Position, write it as Goal_Position, then enable torque, so every servo is held
where it is. On the STS3215 that goal write engages torque by itself (measured on an SO-101); the explicit enable
covers servos that wait for it. Only then does the write send its targets. `torque(True)` takes the same path, and on
an arm already torqued it sends nothing. The engage costs one read and one write, once per torque-on.
