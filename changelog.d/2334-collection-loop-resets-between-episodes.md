### Fixed

Two published multi-episode collection recipes - `save_episode`'s docstring and
the Newton recording guide - omitted the `reset()` between rollouts, so every
episode after the first was recorded starting from the previous rollout's
terminal pose. The dataset kept the requested episode count (so
`verify_dataset_episodes` passed) while its recorded start states were bimodal:
measured on `so101`, later episodes began up to 0.53 rad from episode 0. Both
recipes now use the three-step `run_policy() -> save_episode() -> reset()` form
the rest of the package already documents, and `save_episode`'s docstring states
which of the two calls cuts the boundary and which re-initializes the world.
