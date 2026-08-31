### Added: `g1_battery` surfaces the driver's cached `rt/lf/bmsstate` snapshot

`G1Driver` subscribes `rt/lf/bmsstate` at connect time and `_on_bms`
decodes each message into a small `_battery` dict carrying `pct` /
`charging` / `current` / `cycle` / `t`. `G1Driver.get_status` publishes
`battery_pct` from that same dict on the mesh's status wire, which
`g1_get_state` (`strands-labs/robots#2934`) already surfaces to an agent;
this verb is the cached-snapshot companion `g1_state`'s docstring names,
returning every field `_on_bms` actually wrote rather than the pack
percentage alone.

`g1_battery(driver)` reads through `driver._snapshot("_battery")` -- the
same accessor the driver's own `stream(action="sensors")` path uses -- so
the cache read holds the driver's `_cache_lock` for the copy and a
caller mutating the returned dict does not race the DDS thread writing
into it. A driver whose subscriber has not received a BMS message yet
reports `present=False` and every field `None`; the verb does not
fabricate a reading the driver does not have.

The verb does not open a second DDS subscriber. Every field it returns
is written by the driver's own `_on_bms` handler, and a second
subscription on `rt/lf/bmsstate` would compete for the wire and double
the bus load `_DDS_INIT_LOCK` under `strands-labs/robots#358` is meant
to prevent. The neon-side `g1_battery` port additionally read `soh`,
per-cell voltages and a per-cell temperature vector off `BmsState_`
directly -- fields the driver's decoder does not carry today. Adding
them here would be a second decoder for the same message, so those
fields land (if at all) on the driver's `_on_bms`, not this verb.

The verb does not decide a battery-floor refusal either.
`G1Driver._check_motion_gates` reads `_battery["pct"]` against
`_battery_floor_pct` on every write; a caller planning that write reads
this verb's `pct` at the same units the gate does and compares
themselves. Restating the driver's floor rule on this side would be a
second source of truth for a domain the driver's own refusal string
already names verbatim.

`import strands_robots.tools.g1.g1_battery` pulls no `unitree_sdk2py`
submodule -- the package's SDK-load-hygiene contract from
`strands-labs/robots#358`.

Refs `strands-labs/robots#358`: fourth verb in the neon-the-g1 →
strands-labs/robots port bundle, after `g1_joint_reference` (PR #2932),
`g1_list_motion_gates` / `g1_fsm_admits` (PR #2933) and `g1_get_state`
(PR #2934).
