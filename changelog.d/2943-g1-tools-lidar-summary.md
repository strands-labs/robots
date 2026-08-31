### Added: `g1_lidar_summary` surfaces the driver's cached lidar-cloud header

`G1Driver` subscribes `rt/utlidar/cloud_livox_mid360` at connect time and
`_on_lidar_cloud` builds a small `_lidar_summary` dict from the message
*header* alone - `count` (`width * height`), `width`, `height`,
`point_step`, `row_step` and `t` - so no point is ever enumerated on the
DDS thread and the record's size is the same for a sparse cloud and a
full one. `G1Driver.stream(action="sensors")` already publishes that
same dict on the mesh's status wire, and this verb is the cached-
snapshot companion to `g1_lidar_state`
(`strands-labs/robots#2941`), returning every field `_on_lidar_cloud`
actually wrote rather than the fault code alone.

`g1_lidar_summary(driver)` reads through `driver._snapshot("_lidar_summary")`
- the same accessor the driver's own `stream(action="sensors")` path
uses - so the cache read holds the driver's `_cache_lock` for the copy
and a caller mutating the returned dict does not race the DDS thread
writing into it. A driver whose subscriber has not received a
`PointCloud2` message yet reports `present=False` and every field
`None`; the verb does not fabricate a reading the driver does not have.

The verb does not open a second DDS subscriber. Every field it returns
is written by the driver's own `_on_lidar_cloud` handler, and a second
subscription on `rt/utlidar/cloud_livox_mid360` would compete for the
wire and duplicate the bus load `_DDS_INIT_LOCK` under
`strands-labs/robots#358` is meant to prevent. The neon-side
`g1_lidar_summary` port also did no more than read the header; adding a
downsample or point enumeration here would be a second decoder for the
same message. The 3D tile (`strands-labs/robots#356`) subscribes the
raw cloud itself through a paced publisher and does its own downsampling;
a caller who needs points reaches for that path, not this one.

`count` is the cloud's uncapped size on purpose
(`strands-labs/robots#2752`): a MID-360 that drops from 24000 points
to 3000 is reporting a fault, and clamping the number would hide it.

`import strands_robots.tools.g1.g1_lidar_summary` pulls no
`unitree_sdk2py` submodule - the package's SDK-load-hygiene contract
from `strands-labs/robots#358`.

Refs `strands-labs/robots#358`: fifth verb in the neon-the-g1 →
strands-labs/robots port bundle, after `g1_joint_reference` (PR #2932),
`g1_list_motion_gates` / `g1_fsm_admits` (PR #2933), `g1_get_state`
(PR #2934), `g1_battery` (PR #2938), `g1_imu` (PR #2939) and
`g1_lidar_state` (PR #2941).
