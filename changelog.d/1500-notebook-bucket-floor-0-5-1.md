### Docs: notebook 5's bucket prerequisite names a release that exists

The streaming-data-loop notebook and the notebooks index both asked for
`strands-robots >= 0.4.2` and offered a git install "until v0.4.2 is on PyPI".
No v0.4.2 was ever tagged - 0.5.0 shipped instead, carrying the bucket APIs the
note was waiting for - so the stated floor pointed at a version that cannot be
installed, and the interim git line read as permanent.

The floor is now `>= 0.5.1` rather than `>= 0.5.0`, which is where
`sync_to_bucket` and `stream_dataset(repo_type="bucket")` actually landed. 0.5.0
floors lerobot at `>=0.6.0`, and 0.6.0's `StreamingLeRobotDataset` accepts no
`repo_type`, so `pip install -c "lerobot==0.6.0" "strands-robots[lerobot]"`
resolves cleanly to a pair whose bucket read the runtime guard then refuses. A
bare install happens to pick 0.6.1 today only because it is the newest
candidate; a lockfile or a pre-existing 0.6.0 silently produces the broken
combination. 0.5.1 is the first release whose extras floor lerobot at the
`0.6.1` recorded in `BUCKET_STREAMING_MIN_LEROBOT`.

Both training notebooks also understated their dependencies. `lerobot[training]`
was described as a GPU requirement, but lerobot's `train()` calls
`require_package("accelerate", extra="training")` *before* it branches on
device, and no Strands Robots extra pulls `accelerate` in. So a laptop reader
who follows the stated install reaches a `train()` that returns
`status="error"` with `checkpoint_dir=None`, and the following cell then fails on
the `None` rather than on the missing dependency. Notebook 3 carried the same
gap while promising "the whole loop close[s] on a laptop with no GPU"; both
preambles, both table rows, and a new section in the notebooks index now say the
extra is needed on CPU too, and name the failure it produces.

Both training cells now also check `result.status`. `trainer.train()` converts
any failure into a `TrainResult` rather than raising, and its `message` carries
lerobot's own remedy (`'accelerate' is required but not installed. Install it
with: pip install 'lerobot[training]'`) - which both notebooks discarded, keeping
only `status` and `checkpoint_dir`. So the diagnosis was one cell further on and
one indirection away from its cause: the following `create_policy(None)` raised
`TypeError: argument of type 'NoneType' is not iterable`, naming neither the
missing package nor the fix. The cells now re-raise `result.message`, matching
the rollout guard already in notebook 5's recording cell.

One stale figure: the GPU reference said 124 seconds where
`gpu_run_media/training_capture.txt` records `wall: 132.6s` for its 500 steps,
so notebook 5 now cites ~133 s and names the instance type the log does.
