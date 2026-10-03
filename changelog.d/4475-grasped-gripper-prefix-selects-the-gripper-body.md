### Fixed: `grasped`'s `gripper_prefix` names the gripper body, and a prefix no body has is refused

`grasped(body, gripper_prefix)` matched the prefix against contact geom labels,
so on the SO-101 - whose jaw that touches the cube is the named geom
`so101/so101/static_finger` on body `so101/gripper` - the natural
`"so101/gripper"` never fired, the documented `"so100"` matched nothing, and
naming the robot (`"so101"`) made an elbow shove read as a grasp. The prefix is
now matched against the touching geom's body and every ancestor body, so naming
the gripper's root body selects every jaw under it. MuJoCo `get_contacts`
records carry that path as `bodies1` / `bodies2`; a backend without it is still
matched on the geom label. `run_policy(stop_when=)`, `eval_policy(success_when=)`
and `evaluate_benchmark` refuse a `gripper_prefix` that no body in the scene
starts with before the rollout, instead of arming a clause pinned `False`. The
shipped examples now spell `"so101/gripper"`.
