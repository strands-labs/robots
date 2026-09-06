### Fixed: an RL `device` no torch build can parse is reported by `validate()` instead of aborting `setup()`

All three from-scratch RL backends (`PpoTrainer`, `FastSacTrainer`, `FastTd3Trainer`)
hand `RLTrainSpec.device` straight to `torch.device` in `setup()`, and every
network, replay buffer and rollout tensor the run allocates is placed on the
result. `device` was the one caller-supplied knob on that spec with no domain, so
`validate()` returned `[]` -- which `Trainer.validate` documents as meaning the
spec IS launchable -- for a spec that cannot launch: `"gpu"` and `"cuda:abc"`
raised out of `setup()`, and a non-string ordinal such as `1` constructed on any
host and then died at the first `.to()` with `CUDA error: invalid device ordinal`
from a `torch/nn/modules/module.py` frame naming neither the field nor the run.

The domain now lives in one place, `strands_robots.utils.torch_device_error`, which
the RL preflight, `LerobotTrainer` and the `lerobot_train` tool all consult -- the
same consolidation `step_cadence_error` already applies to the `save_freq` beside
`device` in that tool's argv. Only the spelling is graded, never availability, and
an unstated (falsy) device still resolves through each surface's documented default.
