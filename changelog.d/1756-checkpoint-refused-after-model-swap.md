### Fixed: a physics checkpoint is refused once the model it was taken against is swapped

`save_state` fingerprints a checkpoint with `nq`/`nv`/`na`/`nu` plus a recompile
generation counter, and `load_state` refuses a checkpoint whose fingerprint no
longer matches. The counts catch a swap that resizes the state vector; the
generation exists for the swap that does not, because two models can agree on
every count and still mean different things at the same state indices.

Only one of the seven code paths that install a compiled model bumped that
counter, so the rest accepted checkpoints taken against a model that no longer
existed. `replace_scene_mjcf` was the sharpest case: a checkpoint saved from one
scene was applied to a completely different scene with matching counts, writing
each value into whatever joint now occupied that index and reporting success -
leaving the arm in a pose no checkpoint had ever recorded. `remove_camera`
accepted a checkpoint that `add_camera` refuses, and a `patch_scene_mjcf` op that
adds no body left nothing to distinguish the two models.

Every path that swaps the live model now goes through a single
`install_compiled_model` function that performs the rebind and the invalidation
together, so a swap site cannot install a model without invalidating the
checkpoints taken against the previous one.
