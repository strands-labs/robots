### `feat(tools/g1)`: port the `LocoClient._Call` API-id enumeration verbs from the neon bundle

Two agent-facing verbs surface the six `_Call` API ids the neon bundle
(`cagataycali/neon-the-g1/tools/_g1_common.py`) fronts across its
`read_fsm_id` / `read_fsm_mode` / `read_balance_mode` / `read_swing_height`
/ `read_stand_height` / `set_swing_height` helpers:

- `g1_list_loco_call_api_ids()` names the whole envelope: the six ids
  (`7001`, `7002`, `7003`, `7004`, `7005`, `7103`), each descriptor's
  `role` / `kind` / `payload` / `description` / `admits_loco_writes`
  flag, the write-id subset (only `7103` today), the `WALK_FSMS`
  gate every write shares, and the three refusal codes a future
  driver-side wrapper would quote (`3103` invalid API id, `3104`
  RPC future in flight, `7404` gate-refused write).
- `g1_loco_call_api_id_admits(api_id: int)` decides one query. On
  admission returns the same descriptor `g1_list_loco_call_api_ids`
  returns for that id; on refusal names the `3103` code and its
  decoded text, plus a `reason` string that names why the argument
  was refused (missing, bool, non-int, or unknown id). `bool`,
  non-int, and `None` inputs are refused decidably rather than
  resolved through Python's coercions.

The module ports the read-only enumeration half of the neon bundle's
`_Call` catalogue - the actual call path is deferred to a future
driver-side wrapper that will front the RPC through the existing
motion-gate at :meth:`strands_robots.drivers.g1.G1Driver._check_motion_gates`
for `7103` (the only write id) and through the driver's cached
motion-switcher for the FSM reads (which now source through
:mod:`~strands_robots.tools.g1._motion_switcher` per
strands-labs/robots#2916, so `7001` / `7002` are the neon path only,
not the driver path).

Import hygiene: no `unitree_sdk2py` submodule pulls at import time
(the SDK-load-hygiene contract every other file in this package
carries, refs strands-labs/robots#358). Zero-argument verb path,
snapshot-only reads.
