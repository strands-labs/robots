### Fixed: a camera-recording flush that encodes nothing keeps its buffer

`stop_cameras_recording` and the synchronous `finalize` cleared the recording's
registration immediately after flushing it, whatever the flush reported.
`_flush_cameras_recording_state` is best-effort and folds a per-camera encode
failure into its success envelope, so it has exactly one hard failure - no
`imageio` - and it returns before opening any writer. Nothing is encoded and
every buffer is intact, which is the state the expired-join refusal beside it
calls recoverable; deregistering there discarded those frames permanently.

The failure is reachable through a published extra: `imageio` comes with
`[sim-mujoco]`, but `[sim-newton]` and `[cosmos3-sim]` declare `mujoco` without
it, and neither start verb probes the encoder, so the flush is the first call
that needs one. Measured on a scene with 13 frames captured, the stop reported
an error, `get_cameras_recording_status` then answered `[idle]`, and installing
the encoder and calling again - the remedy the error message names - answered
"Was not recording cameras." with all 13 frames gone and no MP4 written.

Both flush paths now deregister only when the flush encoded, so a stop that
could not encode leaves the recording in the `unflushed` phase and a later call
writes the frames. The refusal carries the per-camera buffered counts and names
the verb that encodes them, matching the expired-join refusal, and the
synchronous `finalize` keys its no-op on the registration rather than the
`running` flag it clears itself. The verdict of a stop that cannot flush is
unchanged; only what survives it is.
