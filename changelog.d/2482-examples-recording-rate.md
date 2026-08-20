### Fixed: three examples record at the rate their own rollout captures

`examples/03_record_dataset.py`, `examples/07_post_tune_any_policy.py` and
`examples/locomotion/vla_g1_workflow.py` each declared `start_recording(fps=30)`
while their rollout captured at 50 Hz, so the recorder refused every frame and
the episode landed empty: `03` exited reporting `0 frames` against a docstring
promising `100 frames`, `07` printed `Recorded LeRobotDataset ->` and then died
two steps later inside lerobot with an unrelated-looking Hub 404 for the empty
local dataset, and the G1 workflow printed `Episode N/N recorded.` per episode
for a dataset with nothing in it. Each rollout now captures at the rate its
recording declares, and each checks the `run_policy` / `stop_recording` result so
a future mismatch is reported where it happens. `07`'s DEPLOY step also runs the
checkpoint it loads, which its own section comment already promised.
