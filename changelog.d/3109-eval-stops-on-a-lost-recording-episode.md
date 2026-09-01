### Fixed: an evaluation that lost a recording episode does not report a success rate over it

`PolicyRunner._finalize_recorder_episode` flushes the attached dataset recorder
at the end of every eval rollout so the dataset records per-episode boundaries.
A failed flush was turned into a `logger.warning` and the helper returned
nothing, so both `evaluate` and `_evaluate_with_spec` ran their remaining
episodes and reported a `success_rate` over all of them under
`status="success"`.

A failed flush is not the loss of an episode boundary. `DatasetRecorder`
marks itself closed on one, because the LeRobot episode buffer is in an
undefined state after a partial write - and `add_frame` returns on a closed
recorder without writing a frame, without raising `RecordingFrameError`, and
without counting a `dropped_frame_count`. Every later episode is therefore
discarded in silence, leaving no trace even in the recorder's own accounting.
Driving a real MuJoCo evaluation with a real recorder over a dataset root whose
write permission is revoked at the end of episode 0, a four-episode run reported
four completed episodes with `status="success"` while `meta/info.json` held zero
and 18 of 24 `add_frame` calls had been dropped with `dropped_frame_count` still
at 0. One warning line was the whole record of the run.

The helper now returns the reason, and both loops stop at that episode and
report it: `status` is `"error"`, the text names the lost episode, and the json
payload carries `recording_save_error` - present and `None` on every healthy
run. `episodes_completed` and the aggregates cover only the episodes that ran,
so nothing is averaged over episodes whose frames reached no dataset. The
failing episode's rollout MP4 is still closed and collected before the loop
stops, since the dataset episode is gone and the video is the only remaining
record of what the policy did on it.

This is the rule every sibling flush already applies, each for the same stated
reason: `stop_recording` and `SimEngine.save_episode` drop the poisoned recorder
and return an error, `run_policy(n_episodes=...)` aborts its remaining episodes,
and the MuJoCo backend's `reset` surfaces the failure rather than resetting into
an undefined state. It also matches the level below, where `RecordingFrameError`
exists precisely so a rollout driver cannot absorb a lost recording frame - a
lost frame truncates one episode, while a lost episode truncated the whole run.

The regression test drives a real `Simulation`, `PolicyRunner` and
`DatasetRecorder`, standing in only for the `LeRobotDataset`, and closes the
family with a scan that derives every `save_episode` call site in
`strands_robots/simulation` from the source rather than listing them. Three
existing tests asserted the previous posture and are evolved in place, since
each still pins something real: the helper must report rather than raise, and
the flush must be attempted per episode.
