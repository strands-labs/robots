### Fixed

`DatasetRecorder.clear_episode_buffer` now un-counts the frames it discards.
`add_frame` counts a frame when it is buffered, but frames only reach disk on
`save_episode`, so a discarded (aborted) episode left `frame_count` reporting a
total no parquet row backed. Because `stop_recording` asks that counter whether
anything was ever captured before refusing an empty dataset, the inflated count
blinded the refusal: discarding a partial rollout and then stopping reported
success for a dataset holding only `meta/info.json`. The un-count applies only
when the discard actually happened - if no clear surface is available the frames
are still buffered and will be written by the recommended
`stop_recording`/`save_episode` drain, so both counters are left in place.
