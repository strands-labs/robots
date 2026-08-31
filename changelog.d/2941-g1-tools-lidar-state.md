### Added: `g1_lidar_state` surfaces the driver's cached `rt/utlidar/lidar_state` snapshot

`G1Driver` subscribes `rt/utlidar/lidar_state` at connect time and
`_on_lidar_state` decodes each `LidarState_` message into a small
`_lidar_state` dict carrying `code` (the MID-360's fault code) /
`code_text` (its rendered text) / `freq` (the cloud frequency in Hz) /
`sys_rotation_speed` (the system rotation speed) / `t` (the wall time
of decode). `G1Driver.get_status` already surfaces those same fields
under `lidar_state` on the mesh's status wire, which `g1_get_state`
(`strands-labs/robots#2934`) hands to an agent as one dict among many;
this verb is the cached-snapshot companion that returns the
lidar-state fields alone, at the same shape `_on_lidar_state` writes.

`g1_lidar_state(driver)` reads through `driver._snapshot("_lidar_state")`
-- the same accessor the driver's own `stream(action="sensors")` path
uses -- so the cache read holds the driver's `_cache_lock` for the copy
and a caller mutating the returned dict does not race the DDS thread
writing into it. A driver whose subscriber has not received a
`LidarState_` message yet reports `present=False` and every field
`None`; the verb does not fabricate a fault code of `0` (which the
MID-360 uses for its healthy state) or a frequency of `0.0` (which
would look like a stopped scan).

The verb does not open a second DDS subscriber. Every field it returns
is written by the driver's own `_on_lidar_state` handler, and a second
subscription on `rt/utlidar/lidar_state` would compete for the wire
and double the bus load `_DDS_INIT_LOCK` under `strands-labs/robots#358`
is meant to prevent. The neon-side `g1_lidar` port additionally read
the firmware, software and SDK versions directly off `LidarState_` and
opened a `rt/utlidar/switch` publisher next to the subscriber -- a
second publisher path on top of the driver's DDS engine and outside
the read-only contract this verb keeps. Those fields land (if at all)
on the driver's `_on_lidar_state`, not this verb, and any switch verb
lands on the driver's own DDS engine, not a shadow publisher.

`import strands_robots.tools.g1.g1_lidar_state` pulls no
`unitree_sdk2py` submodule -- the package's SDK-load-hygiene contract
from `strands-labs/robots#358`.

Refs `strands-labs/robots#358`: fifth verb in the neon-the-g1 →
strands-labs/robots port bundle, after `g1_joint_reference` (PR #2932),
`g1_list_motion_gates` / `g1_fsm_admits` (PR #2933), `g1_get_state`
(PR #2934), `g1_battery` (PR #2938) and `g1_imu` (PR #2939).
