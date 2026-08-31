### Added: `g1_joint_reference` / `g1_joint_name` / `g1_joint_index` name what `G1Driver.send_action` accepts

`send_action` refuses any action-dict key that is not in
`_G1_JOINT_INDEX`; three read-only tools now surface that same map so a
caller can decide the refusal decidably rather than triggering it from the
driver at rollout time. All three read the driver's own constants
(`_G1_JOINT_INDEX`, `_SDK_KP`, `_SDK_KD`) — a joint added to the driver's
contract moves the write path and this lookup together, so the shipped
domain cannot drift between them.

`g1_joint_index` accepts the driver's canonical snake_case verbatim and also
normalises a PascalCase or camelCase spelling (`LeftKnee`, `leftKnee` both
resolve to `left_knee`); the alias is one-way, and the returned `name` field
is always the driver's exact write-path key. The reference tool's `group`
filter partitions the 29 slots into `left_leg` / `right_leg` / `waist` /
`left_arm` / `right_arm`, and refuses an unknown group name with a message
that lists the domain rather than a hint.

`import strands_robots.tools.g1.g1_joints` pulls no `unitree_sdk2py`
submodule — the package's SDK-load-hygiene contract from
`strands-labs/robots#358`.

Refs `strands-labs/robots#2765`: the ankle-pitch / ankle-roll rename question
and the per-build joint-presence question (waist roll/pitch on the
waist-locked variant, wrist pitch/yaw on 23dof) remain open there. This
lookup returns the names the driver's map declares today; whichever way that
issue lands, it lands in the driver's map, and this lookup follows.
