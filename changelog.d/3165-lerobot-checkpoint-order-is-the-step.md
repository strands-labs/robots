### Fixed: the newest lerobot checkpoint is the highest step, not the last name

lerobot names a checkpoint directory
`f"{step:0{max(6, len(str(total_steps)))}d}"` (`get_step_identifier`), so the
zero-padding width is a property of the run that wrote the name rather than of
the checkpoints tree. Continuing a run with a larger `steps` budget widens it,
and `LerobotTrainer._resume_config_path` ordered those names as text, where
`"100000"` sorts above `"0900000"`. Once a 100k-step run was continued to 1M,
the step directories reported the step-100,000 checkpoint as the newest one on
disk, and both consumers acted on it: resume restarted from the older
checkpoint and re-ran the 800,000 steps in between, and `latest_checkpoint`
handed `export`/`create_policy` the older model. Both reported success. Text
order also ranked an unnumbered sibling directory above every real checkpoint,
which needed no change of width at all. The wrong answer was intermittent: at
the moment a continued run reached its budget exactly, text order agreed
again, so a finished tree did not carry the symptom.

The step directories are read when the `last` link lerobot maintains carries
no config -- an absent link, or one left dangling by pruning the checkpoint it
pointed at, which is routine when each checkpoint is gigabytes.

The fallback now orders on the step parsed out of the name, which is what the
GR00T trainer already does for its own `checkpoint-<step>` directories. A name
lerobot did not number sorts below every numbered one, and the name breaks
ties, so the ordering is total and an unnumbered directory is still the answer
when it is the only one on disk.
