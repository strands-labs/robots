### Fixed: three false fleet claims in the Isaac docs

`docs/simulation/isaac.md` offered "fleet RL on PhysX GPU with 1024+ parallel
environments", described `num_envs` as "Set to `1024`+ for fleet RL", and shipped a
"Fleet (IsaacLab-style) preview" whose code never called `replicate()` at all.
Setting `num_envs` alone creates nothing - `replicate()` is what clones - and the
missing per-environment action API is what keeps this short of fleet RL. All three
now say what the backend does.
