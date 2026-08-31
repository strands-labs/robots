### Added: `g1_get_state` names the driver's status envelope in agent-facing shape

`G1Driver.get_status` returns the mesh's status envelope
(`fsm_id`, `mode_machine`, `battery_pct`, `fsm_mode_name`, `fsm_refusal`,
`motion_switcher_open_error`), which is what the mesh peer wire wants
but is not what an agent planning a write asks for: a caller wants the
membership answer against the arm-SDK and locomotion gates decidably,
before `send_action` or `run_policy` triggers a refusal at wire time.

`g1_get_state(driver)` reads that same envelope once and adds the two
decided `admits_arm` / `admits_loco` booleans against the driver's own
constants (`HANDSHAKE_FSMS` / `WALK_FSMS` from
`strands_robots.tools.g1._g1_common`), plus the two id sets sorted as
lists so a caller can quote them in its own voice. `bool` values on
`fsm_id` are refused for admission (`True` is `int(1)` but is not a
motion-switcher FSM id) -- the same rule
`g1_fsm_admits` under `strands-labs/robots#2933` names.

This is the first driver-instance-taking verb in
`strands_robots.tools.g1`; every earlier one is a pure reader over
module-level constants. The driver type hint is a forward reference
under `TYPE_CHECKING` so `import strands_robots.tools.g1.g1_state` still
pulls no `unitree_sdk2py` submodule (the package's SDK-load-hygiene
contract, refs `strands-labs/robots#358`).

Ports `neon-the-g1/tools/g1_state.py::g1_get_state` into
`strands_robots.tools.g1.g1_state`. Verbs on the neon side that reach
DDS directly (`g1_read_lowstate` decoding IMU/joint state off
`rt/lowstate`) are not ported here -- the driver's own subscribers
deliver every field they cover, and a second subscriber path would
double the bus load `_DDS_INIT_LOCK` under `strands-labs/robots#358` is
meant to prevent. A companion verb that surfaces the driver's cached
sensor snapshot (`driver._imu` / `driver._battery` / `driver._lidar_*`)
is a separate port from this one.

Refs `strands-labs/robots#358`: third verb in the neon-the-g1 →
strands-labs/robots port bundle, after `g1_joint_reference` (PR #2932)
and `g1_list_motion_gates` / `g1_fsm_admits` (PR #2933).
