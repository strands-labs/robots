### `step` records while a dataset recording is open

Asked to record a small demonstration by moving through three poses, an agent's
natural plan - `start_recording`, then `set_joint_positions(hold=True)` + `step`
per pose, then `stop_recording` - captured nothing: only `run_policy`'s
per-step hook fed the recorder, and the replayed agent, told so in prose, went
on to fill the episode with `run_policy(mock)` noise and report the poses as
recorded. `step` now feeds an open recording on the MuJoCo backend: one frame
per `1/fps` seconds of sim time (the clock persists across calls), observation
= every robot's state and cameras as the rollout hook supplies them, action =
the position-servo targets in force (`data.ctrl`, keyed like
`robot_action_keys`), task = the session's. The `step` reply says how many
frames it captured, or that none was due and when the next is; a running
rollout keeps sole ownership of the recorder; motion primitives are unchanged.
The empty-recording refusal and `start_recording`'s reply name the route.
