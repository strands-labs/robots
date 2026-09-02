### Fixed: an unreadable episode count is reported as no reading, not as zero

`stop_recording`'s parquet-truth gate read the dataset's canonical episode
count as `int(getattr(ds_meta, "total_episodes", 0))`. A layout that exposes no
such attribute is a failed probe, but the zero default turned it into the
*measurement* `0` -- which the gate then trusted as ground truth and used to
overwrite the count the recorder had actually measured. A session that really
saved four episodes reported `episode_count: 0`, `parquet_episode_count: 0`,
`episode_count_mismatch: True` in both the json and the human-readable payload,
and logged a warning naming a parquet count nothing had read.

Every other unavailability on this same path is already expressed as "skip",
never as a zero: `recorder_dataset_fps` returns `None` when the attribute is
absent and the comparison is skipped, and the missing-`dataset` branch leaves
`parquet_episode_count` at `None` with the gate quiet -- the suite calls these
the safe defaults and pins them. The third attribute in the same defensive
chain was the only one substituting a zero-valued default for a failed probe.

The probe now uses `None` and the gate is skipped when the attribute is absent,
so only a count that was actually read can move the reported total. A layout
that really reports zero episodes is still a reading and is still judged, so
the silent-collapse gate keeps firing on the case it was written for.
`stop_recording` also gained the `Returns:` block it never had, stating the
contract for all three episode-count fields.
