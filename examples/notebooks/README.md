# Notebooks

Click-and-run getting-started notebooks for Strands Robots. Each one runs
end-to-end on a laptop in simulation - **no hardware, no GPU, no Hugging Face
credentials**. They are the notebook companions to the numbered scripts in
[`../`](../).

## Install and launch

```bash
uv pip install -U "strands-robots[sim-mujoco,lerobot]" jupyterlab
jupyter lab
```

The `-U` matters in an environment that already carries an older release. Without
it `pip` reports `Requirement already satisfied` and leaves, say, 0.4.1 in place -
whose `StreamingDatasetReader.open` has no `repo_type` parameter and no
`**kwargs`, so notebook 5's bucket read fails with `TypeError: open() got an
unexpected keyword argument 'repo_type'`. Pinning the floor
(`"strands-robots[sim-mujoco,lerobot]>=0.5.1"`) upgrades without `-U` too; either
form is enough, a bare install of the extras is not.

> **Notebook 5's bucket path needs `strands-robots >= 0.5.1`**, which is the
> first release whose `[lerobot]` extra floors LeRobot at the `>= 0.6.1` that
> serves `stream_dataset(..., repo_type="bucket")`. Below that floor a resolver
> is free to pair Strands Robots with a LeRobot that refuses the bucket read.

On macOS the notebooks set `MUJOCO_GL=cgl` for offscreen rendering; everywhere
else (e.g. headless Linux) they default to `egl`. An exported `MUJOCO_GL`
always wins.

## The series

| # | Notebook | What it shows |
|---|----------|---------------|
| 1 | [`01_getting_started.ipynb`](01_getting_started.ipynb) | `Robot("so100")`, run a policy, read joint state, `create_policy()` |
| 2 | [`02_record_and_stream.ipynb`](02_record_and_stream.ipynb) | Record a LeRobotDataset, then stream it back with `stream_dataset()` |
| 3 | [`03_record_train_deploy.ipynb`](03_record_train_deploy.ipynb) | The full loop: record, train an ACT policy on CPU, export, and load it back (training needs `lerobot[training]`) |
| 4 | [`04_discover_lerobot.ipynb`](04_discover_lerobot.ipynb) | Discover the LeRobot API with `use_lerobot`: list robots, policies, teleoperators, cameras, and inspect any class |
| 5 | [`05_streaming_data_loop.ipynb`](05_streaming_data_loop.ipynb) | The streaming data loop: record, render, stream back, train, and load, in one notebook (training needs `lerobot[training]`; the optional Storage Bucket sync needs `strands-robots >= 0.5.1` + LeRobot >= 0.6.1; an optional final step reruns the loop on the Isaac backend with `backend="isaac"` when an RTX GPU + `sim-isaac` are present) |
| 6 | [`06_fleet_orchestration.ipynb`](06_fleet_orchestration.ipynb) | Drive a heterogeneous fleet from one goal: read each robot's capability tags, decompose the goal into per-robot tasks (rule-based, with an optional Strands agent planner), dispatch them together through `run_multi_policy`, and re-plan when a robot drops offline |
| 7 | [`07_data_augmentation.ipynb`](07_data_augmentation.ipynb) | Data augmentation at three pipeline stages: seeded domain randomization + sensor noise while recording (`randomize()` / `set_obs_noise()`), backdrop swaps over the same trajectories (`HybridCompositor.set_background`), and `TrainSpec.augmentation` mapped to GR00T's image-augmentation flags; generative augmentation (`strands_robots.transforms`) is documented markdown-only - generation needs a GPU and the notebook is CPU-only |

Read them in order; each builds on the previous one. Notebook 3 trains a real
policy on CPU with a tiny dataset and two steps - raise the step count and run on
a GPU for a production checkpoint; the code path is identical.

## Training needs `lerobot[training]`, on CPU too

Notebooks 3 and 5 both train, and `lerobot`'s `train()` calls
`require_package("accelerate", extra="training")` before it branches on device,
so a CPU run needs the extra just as a GPU run does. No Strands Robots extra
pulls it in:

```bash
uv pip install "lerobot[training]"
```

Both training cells check `result.status` and raise with `result.message`, which
carries lerobot's own install remedy. `trainer.train()` converts any failure into
a `TrainResult` rather than raising, so an unchecked call would print
`status="error"`, hand on a `checkpoint_dir` of `None`, and let the next cell
fail with `TypeError: argument of type 'NoneType' is not iterable` - a message
that names neither the missing package nor the fix.

## Notebook 3 and GPUs

Notebook 3 trains [ACT](https://tonyzhaozh.github.io/aloha/) from scratch for two
steps so the record -> train -> export -> load loop closes on a CPU laptop. For a
real policy, point `TrainSpec.base_model` at a pretrained checkpoint, raise
`steps`, and run on a GPU. Swapping `create_trainer("lerobot_local")` to
`"groot"` or `"cosmos3"` retargets the same lifecycle to those providers (which do
require a GPU).
