### Fixed: a leader arm is named as a teleoperator, not listed among the followers

`Robot()`'s registry guard refuses an unregistered `*_leader` name and names
`Teleoperator()` instead, deliberately without listing the registry - its own
comment says a listing "invites the caller to retry with the follower name on the
leader's port - the exact mistake that torque-enables the arm a human is holding".
That guard sits behind `get_robot(canonical) is None and not has_hardware(canonical)`,
so registering the leader arm makes it unreachable.

Registering it is not misuse: `hardware={"lerobot_type": "so101_leader"}` is the
honest registration for a leader arm, and it satisfies the invariant that a leader
name may never name the follower it drives. The request fell through to lerobot's
`RobotConfig` lookup instead, which answered `Unsupported robot type: 'so101_leader'.
Known lerobot robot types: [...]` - sixteen names, every one a follower. That is the
listing the first guard exists to remove, offered by the refusal a registered rig
actually reaches. Following it and retrying with `so101_follower` on the leader's
port builds an `SOFollowerRobotConfig` bound to the arm a human is holding.

lerobot is not confused about the device: `so101_leader` is one of nine `*_leader`
entries in `TeleoperatorConfig`. The refusal never asked the other registry. Both
config sites now consult one shared rule first, so a name that is a device of the
other kind is named as that kind together with the entry point that builds it -
`Teleoperator(...)` for a leader handed to `Robot()`, and `Robot(...)` for a follower
handed to `Teleoperator()`. Neither cross-kind refusal lists the choices of the kind
that was asked for, because a caller whose name is a known device has not made a
typo. A name neither registry knows is a typo, and keeps its listing unchanged.
