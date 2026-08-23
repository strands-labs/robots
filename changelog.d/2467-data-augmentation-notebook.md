### Added: notebook 7 teaches the three shipped data-augmentation mechanisms

`examples/notebooks/07_data_augmentation.ipynb` runs CPU-only, end-to-end, and
seeded, covering augmentation at the three pipeline stages it ships at:
collection time (`randomize()` + `set_obs_noise()` while recording a dataset),
render time (`HybridCompositor.set_background` backdrop swaps over the same
trajectories, with the CUDA-only `GsplatBackground` path documented), and train
time (`TrainSpec.augmentation` inspected via `Gr00tTrainer.build_command`).
Generative augmentation is documented in a markdown-only closing section - the
perturb-vs-generate framing and the provenance doctrine of the shipped
`strands_robots.transforms` surface (see `docs/data/transforms.md`) - with no
code cells because generation needs a GPU and the notebook is CPU-only. (#2467)
