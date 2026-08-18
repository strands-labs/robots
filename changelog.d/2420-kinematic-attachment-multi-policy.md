### Fixed

- **simulation/mujoco**: `run_multi_policy` now re-applies the kinematic-attachment follow after its physics step, so a body attached with `attach_bodies(mode="kinematic")` is carried through a synchronized multi-robot rollout instead of being left behind while every surface reports success.
