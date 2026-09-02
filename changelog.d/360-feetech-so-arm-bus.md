### Added: SO-100/SO-101 native driver reaches the arm — the Feetech SCS bus is wired

`FeetechDriver` was registered for `so100`/`so101`/`lekiwi` but every write
refused with `"not wired yet (the Feetech SCS serial bus)"`, and the driver
exposed neither `bus` nor `is_connected` — so `joint_read_source` resolved it to
`None` and an SO-arm published **no `joints` section at all** on the mesh state
topic. `FeetechBus` closes both halves (:issue:`360` scope 1).

`send_action` now writes the whole arm in one `SYNC_WRITE` frame, so a
six-joint move latches together instead of smearing over six write latencies.
Targets are degrees (`gripper` is percent open) — the domain `pose_tool`
already established for this family, whose motor map is the source of truth
`SO_ARM_MOTORS` is distilled from. A `.pos` suffix is accepted, so a lerobot
action dict works unchanged. A target outside a joint's range is **refused, not
clamped**: a clamp turns a 400-degree command into a 90-degree motion and
reports success. `sync_read` decodes `Present_Position` back to degrees and
`Present_Velocity`/`Present_Load` as sign-magnitude, because reading bit 15 as
magnitude reports a stopped joint as moving at full speed. `Present_Current` is
deliberately unreadable rather than decoded by guess.

Exposing `bus`/`is_connected` is what puts joints on the mesh: `read_joints`
prefers a motor bus over a full observation, so a dead camera can no longer hide
the joint positions. `stop` releases torque on every motor and names any that
stayed driven; `cleanup` closes the port and leaves torque alone, so tearing
down a process does not drop a held payload. `motor_ids` is now honoured
(narrowing the arm) instead of recorded and ignored, and an ID with no joint
name is refused. `start_task`/`run_policy` still refuse — now naming the
missing policy control loop rather than blaming a bus that works.

Every driver-side path that touches the wire holds the same `bus_lock` on the
driver that `read_joints` takes for the mesh-side read, so a 30Hz joints
publisher and an agent move no longer interleave on a half-duplex bus. The
agent surface refuses rather than coerces: a non-boolean `enabled` on
`set_torque` is named instead of passed through `bool()`, where `"false"` is
truthy and a request to *release* a loaded arm energizes it while reporting
`torque_enabled: True` as a success. An action verb the schema does not declare
is refused with the verb list read back off the schema, rather than falling
through to the torque release the fallthrough branch used to be.
