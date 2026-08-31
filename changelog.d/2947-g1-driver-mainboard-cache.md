### Added: `G1Driver` caches `rt/mainboardstate` for the `g1_mainboard` verb

`G1Driver` now subscribes `rt/mainboardstate` at connect time alongside
its four existing subscriptions (`rt/lowstate`, `rt/lf/bmsstate`,
`rt/utlidar/lidar_state`, `rt/utlidar/cloud_livox_mid360`). Every
`MainBoardState_` message decodes into a small `_mainboard` dict on
`_cache_lock` carrying `fan_state` (a `list[int]` vector of fan flags),
`temperature` (a `list[float]` vector of per-thermistor readings on the
mainboard), `sys_state` (the firmware's system-state code as an
integer), `tick` (the firmware tick counter) and `t` (the wall time of
decode).

The new `g1_mainboard` verb wraps `driver._snapshot("_mainboard")` --
the same accessor `g1_battery` (`strands-labs/robots#2938`), `g1_imu`
(`strands-labs/robots#2939`), `g1_lidar_state`
(`strands-labs/robots#2941`) and `g1_lidar_summary`
(`strands-labs/robots#2943`) use -- and reshapes the dict into an
agent-facing envelope carrying `present=False` and every field `None`
before the first message arrives. The verb does not open a second DDS
subscriber path; every field it returns is written by the driver's own
`_on_mainboard` handler, and a second subscription on
`rt/mainboardstate` would compete for the wire and duplicate the bus
load `_DDS_INIT_LOCK` under `strands-labs/robots#358` is meant to
prevent.

The five fields the decoder writes are exactly the ones
`MainBoardState_` declares on the layout the current firmware ships
with. The neon-side `g1_mainboard` port additionally read `fan_speed`,
`cpu_temperature`, `sys_bat_state` and `bms_state` directly off the
IDL; those names are not declared on the current firmware, and a
decoder reading them would land the `getattr` default in the record
forever (which is the same failure mode
`strands-labs/robots#2941`'s `_reads_the_declared_fields` cell was
landed to catch on `LidarState_`). If a future firmware adds one of
those fields back, `_DECLARED_MAINBOARD_FIELDS` fires so the frozen
declaration and the decoder can be updated together.

`import strands_robots.tools.g1.g1_mainboard` pulls no `unitree_sdk2py`
submodule -- the package's SDK-load-hygiene contract from
`strands-labs/robots#358`.

Two new module-level helpers `_to_int_list` and `_to_float_list` in
`strands_robots.drivers.g1` coerce the IDL's vector fields into plain
Python lists under the copy the `_snapshot` accessor returns, so a
caller mutating the returned dict does not race the DDS thread that
writes into it; a `bytes` / `str` value (which would otherwise iterate
as characters) is refused as `None` instead. The scalar helper
`_to_int` is the same rule for `sys_state` / `tick`.

Refs `strands-labs/robots#358`: sixth verb in the neon-the-g1 -->
strands-labs/robots port bundle, after `g1_joint_reference` (PR
#2932), `g1_list_motion_gates` / `g1_fsm_admits` (PR #2933),
`g1_get_state` (PR #2934, open), `g1_battery` (PR #2938, open),
`g1_imu` (PR #2939), `g1_lidar_summary` (PR #2943) and
`g1_lidar_state` (PR #2941). First verb in the bundle that also lands
a new driver-side subscription -- earlier ports all wrapped caches the
driver already populated.
