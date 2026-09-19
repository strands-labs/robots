### Fixed: a resumed recording writes the base it observed, on every backend

`DatasetRecorder.create` takes `extra_state_specs` so a floating-base robot's dataset
carries its base columns. The observation delivers `base_pos` / `base_quat` /
`base_lin_vel` / `base_ang_vel` as VECTORS while the schema declares them expanded per
component (`base_pos.x` ...), and `add_frame` bridges the two by reading the SOURCE keys
and flattening each in schema order. It can only do that because `create` records those
sources on the recorder.

`resume` did not, and could not derive them: it inherits the expanded column names from
the dataset on disk, and nothing there says which source a run of components was
flattened from. So the fallback read the expanded names, `observation.get("base_pos.x")`
answered `None` for every one, and the zero-fill beside it supplied `0.0` per component,
per frame.

Nothing raised and nothing logged. The flattened vector still had the schema's length, so
no width check could see it, and the backends' recording hooks all pass
`required_action_keys`, which is what disables the missing-state-column refusal that would
have caught a `None`. Every appended episode therefore recorded the base at the origin, at
rest, under `status: success` - the base-blind dataset `extra_state_specs` exists to
prevent, reintroduced on the append path.

Measured driving `DatasetRecorder` directly, no simulation backend involved: an
observation carrying `base_pos=[1.0, 2.0, 9.0]` recorded `[0.0, 0.0, 0.0]`, while the
joint columns beside it recorded correctly. That asymmetry is what kept it invisible -
a spot check of a resumed dataset shows live joints.

`resume` now takes `joint_names` and `extra_state_specs` alongside `create`'s, and all
three simulation backends pass them. Fixed in the shared recorder rather than in one
backend because all three declare base columns on create and all three resume, so all
three were affected; the per-backend call sites are graded from the source so a fourth
backend is held to it on arrival.
