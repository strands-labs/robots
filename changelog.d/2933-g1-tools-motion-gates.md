### Added: `g1_list_motion_gates` / `g1_fsm_admits` name the FSM ids `G1Driver._check_motion_gates` admits on

`G1Driver._check_motion_gates` refuses every arm-SDK-shaped write while
`_fsm_id` is outside `HANDSHAKE_FSMS`, and every locomotion-shaped write
while it is outside `WALK_FSMS`. Two read-only tools now surface those
same sets to an agent so a caller can decide the refusal decidably before
`send_action` or `run_policy` is attempted, rather than triggering it
from the driver at wire time.

Both tools read the driver's own constants (`HANDSHAKE_FSMS`, `WALK_FSMS`
under `strands_robots.tools.g1._g1_common`, and the `7404` entry in
`ERR_CODES` the write path's refusal quotes) — a widen or narrow of an
admission set in the driver moves the write path and this lookup
together, so the shipped domain cannot drift between them.

`g1_list_motion_gates(scope="")` returns both scopes; `scope="arm"` or
`scope="loco"` filters to one. `g1_fsm_admits(fsm_id, scope="arm")`
computes the same membership answer the gate would, and on a non-member
carries the exact `refusal_code=7404` / `refusal_text` pair the driver's
write path would surface. `bool` is refused as `fsm_id` despite being an
`int` subclass — the domain refusal names that specific caller mistake
because `True` silently compared against a set of three-digit motion-
switcher ids would land as "not arm-ready" for a caller who never asked
a valid question.

`import strands_robots.tools.g1.g1_motion_gates` pulls no
`unitree_sdk2py` submodule — the package's SDK-load-hygiene contract
from `strands-labs/robots#358`.

Refs `strands-labs/robots#358`: this port is the second verb in the
neon-the-g1 → strands-labs/robots port bundle, after
`g1_joint_reference` / `g1_joint_name` / `g1_joint_index` (PR #2932).
Both pieces are pure readers over driver-side constants; a live
FSM-state verb that calls `G1Driver.get_status` is a separate port from
this one, so a driver-instance-taking verb pattern can be introduced in
its own review rather than piggybacking on a reference-only landing.
