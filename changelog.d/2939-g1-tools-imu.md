### Added: `g1_imu` surfaces the driver's cached `rt/lowstate` IMU snapshot

`G1Driver` subscribes `rt/lowstate` at connect time and `_on_lowstate`
decodes each message into a small `_imu` dict carrying `rpy` /
`gyroscope` / `accelerometer` / `quaternion` / `t`. `g1_get_state`
(`strands-labs/robots#2934`) surfaces `mode_machine` from the same
handler; this verb is the cached-snapshot companion to `g1_battery`,
returning every field `_on_lowstate` actually wrote for the IMU rather
than routing the reading through the driver's status envelope.

`g1_imu(driver)` reads through `driver._snapshot("_imu")` -- the same
accessor the driver's own `stream(action="sensors")` path uses -- so the
cache read holds the driver's `_cache_lock` for the copy and a caller
mutating the returned dict does not race the DDS thread writing into it.
A driver whose subscriber has not received a `LowState` message yet
reports `present=False` and every field `None`; the verb does not
fabricate a reading the driver does not have.

The verb does not open a second DDS subscriber. Every field it returns
is written by the driver's own `_on_lowstate` handler, and a second
subscription on `rt/lowstate` would compete for the wire and duplicate
the bus load `_DDS_INIT_LOCK` under `strands-labs/robots#358` is meant
to prevent. The neon-side `g1_read_lowstate` additionally decoded joint
angles, torques, `mode_machine` / `mode_pr` / `tick` and computed a
posture heuristic off the knees. `mode_machine` is surfaced by
`g1_get_state` already; joint reads are a separate verb the driver's
`_on_lowstate` does not yet cache (it caches the IMU sub-record only),
and a posture label would be a second source of truth for a domain the
driver's own motion gate decides at wire time. Those fields land (if at
all) on `_on_lowstate`, not this verb.

The verb does not convert units either. `rpy` is in radians as
`_on_lowstate` wrote it, `gyroscope` is rad/s, `accelerometer` is m/s²;
a caller who wants degrees converts them themselves so the number this
verb returns is bit-identical to what a re-published log would carry.

`import strands_robots.tools.g1.g1_imu` pulls no `unitree_sdk2py`
submodule -- the package's SDK-load-hygiene contract from
`strands-labs/robots#358`.

Refs `strands-labs/robots#358`: fifth verb in the neon-the-g1 →
strands-labs/robots port bundle, after `g1_joint_reference` (PR #2932),
`g1_list_motion_gates` / `g1_fsm_admits` (PR #2933), `g1_get_state`
(PR #2934), and `g1_battery` (PR #2938).
