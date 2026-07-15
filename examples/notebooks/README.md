# Notebooks

Click-and-run getting-started notebooks for Strands Robots. Each one runs
end-to-end on a laptop in simulation - **no hardware, no GPU, no Hugging Face
credentials**. They are the notebook companions to the numbered scripts in
[`../`](../).

## Install and launch

```bash
uv pip install "strands-robots[sim-mujoco,lerobot]" jupyterlab
jupyter lab
```

On macOS the notebooks set `MUJOCO_GL=cgl` for offscreen rendering; everywhere
else (e.g. headless Linux) they default to `egl`. An exported `MUJOCO_GL`
always wins.

## The series

| # | Notebook | What it shows |
|---|----------|---------------|
| 1 | [`01_getting_started.ipynb`](01_getting_started.ipynb) | `Robot("so100")`, run a policy, read joint state, `create_policy()` |
| 2 | [`02_record_and_stream.ipynb`](02_record_and_stream.ipynb) | Record a LeRobotDataset, then stream it back with `stream_dataset()` |
| 3 | [`03_record_train_deploy.ipynb`](03_record_train_deploy.ipynb) | The full loop: record, train an ACT policy on CPU, export, and load it back |
| 4 | [`04_discover_lerobot.ipynb`](04_discover_lerobot.ipynb) | Discover the LeRobot API with `use_lerobot`: list robots, policies, teleoperators, cameras, and inspect any class |
| 5 | [`05_streaming_data_loop.ipynb`](05_streaming_data_loop.ipynb) | The streaming data loop: record, render, stream back, train, and load, in one notebook (the optional Storage Bucket sync needs LeRobot >= 0.6.1) |

Read them in order; each builds on the previous one. Notebook 3 trains a real
policy on CPU with a tiny dataset and two steps - raise the step count and run on
a GPU for a production checkpoint; the code path is identical.

## Notebook 3 and GPUs

Notebook 3 trains [ACT](https://tonyzhaozh.github.io/aloha/) from scratch for two
steps so the record -> train -> export -> load loop closes on a CPU laptop. For a
real policy, point `TrainSpec.base_model` at a pretrained checkpoint, raise
`steps`, and run on a GPU. Swapping `create_trainer("lerobot_local")` to
`"groot"` or `"cosmos3"` retargets the same lifecycle to those providers (which do
require a GPU).
