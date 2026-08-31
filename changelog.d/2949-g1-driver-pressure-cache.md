### Features

- **drivers/g1**: `G1Driver` now subscribes `rt/pressuresensorstate` on its
  background DDS thread and caches the decoded `PressSensorState_` reading
  into `self._pressure` (guarded by the same `_cache_lock` the other
  sensor caches share).  The decoder reads every field through
  `getattr(msg, name, None)` so a firmware rename yields `None` on this
  side rather than raising on the DDS thread; a missing field surfaces
  as `None` for that key rather than dropping the whole reading.  The
  cache holds five fields:

  * `pressure` -- a per-foot-sensor vector of raw pressure readings as
    `list[float]` (or `None` on absent field).  The IDL declares this
    as a 12-element `float32` array.
  * `temperature` -- a per-foot-sensor vector of Celsius readings as
    `list[float]` (or `None`).  Same 12-element shape as `pressure`.
  * `lost` -- the packet-loss counter the IDL declares as `uint32`,
    coerced to Python `int` (or `None`).
  * `reserve` -- the reserve scalar the IDL declares next to `lost`,
    also `uint32` -> `int` (or `None`).
  * `t` -- the wall time of decode (`time.time()`), so the mesh's
    health chip can tell a fresh reading from a stale one.

  Sits in the sensor-cache row alongside `_imu`, `_battery`,
  `_lidar_state`, `_lidar_summary` and `_mainboard` -- one topic per
  cache, one decoder per topic, one lock across all reads and writes.
- **tools/g1**: `g1_pressure(driver)` is the agent-facing wrapper for
  the cache above.  Read-only.  Calls `driver._snapshot("_pressure")`
  (a copy under `_cache_lock`, so a caller mutating the result does
  not race the DDS thread) and reshapes the dict into an envelope
  carrying `status` / `present` / the five cached fields.  A driver
  whose subscriber has not received a `PressSensorState_` message yet
  reports `present=False` and every field `None`; the verb does not
  fabricate a reading the driver does not have.

  `import strands_robots.tools.g1.g1_pressure` pulls zero
  `unitree_sdk2py` submodules (the package's SDK-load-hygiene
  contract), and the verb is duck-typed on `_snapshot` so a test can
  hand it a hand-rolled driver double without a bus.  The driver
  argument is typed `Any` at runtime for the same reason the other
  driver-instance-taking verbs give: the driver module imports
  `ensure_dds` from this package at load, so a runtime import of
  `G1Driver` here would close a cycle.

  Refs `strands-labs/robots#358`.  Sibling readers land in `#2938`
  (`g1_battery`), `#2939` (`g1_imu`), `#2941` (`g1_lidar_state`),
  `#2943` (`g1_lidar_summary`), `#2947` (`g1_mainboard`).

  The verb refuses an unusable handle through the shared
  `snapshot_handle_refusal` guard rather than dereferencing it, so
  `None`, a robot *name*, or any object without the accessor earns an
  error envelope naming the verb, the parameter and the remedy instead
  of an `AttributeError` naming a private attribute. That guard is the
  rule `#2948` derived from the tree -- its population is every `@tool`
  in this package whose first parameter is annotated `Any` -- so a
  sensor verb added here is graded by it without that file being
  edited, and this one is graded by it on arrival.
