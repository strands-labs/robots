### Fixed: a scene mutation swaps the live MuJoCo model under `self._lock`

The MuJoCo backend's own module docstring declares `self._lock` as the "RLock
serializing ALL model/data access", and the render path takes it with a comment
explaining exactly why: "the recorder daemon calls render() on its own thread,
so this path is NOT covered by the blanket dispatch lock". Seven scene-mutation
verbs performed the swap outside that lock, relying on
`_require_no_running_policy()` instead. The two guards cover different readers.
The policy gate refuses a *policy* worker; the camera-recorder daemon has no
policy, so the gate cannot see it, and the lock was the only thing that could
have excluded it.

`create_world`, `add_robot`, `add_object`, `remove_object`, `add_camera`,
`remove_camera`, `patch_scene_mjcf` and `replace_scene_mjcf` now hold the lock
across the swap, joining `load_scene` and `remove_robot`, which already did --
`load_scene`'s existing comment names the hazard ("so a concurrent
render/recorder thread never observes a half-built world"). Registration and
rollback are inside the same critical section as the recompile, so a reader
never sees an object registered against a model that does not contain it, nor a
rolled-back registry against a recompiled model. `_compile_world` takes the lock
itself rather than relying on its caller, keeping the invariant local to the
function that performs the swap.

The swap is two assignments wide: `scene_ops.install_compiled_model` rebinds
`world._model` and then `world._data`, so an unexcluded reader can observe a
model from after the swap paired with data from before it -- indexing the new
model's body count into the old data's arrays, which MuJoCo dereferences
natively.

Measured with a reader of the recorder's shape, which takes the lock and renders
twice without releasing it: before, an `add_object` on another thread landed
between the two renders and 634 pixels changed inside one critical section;
after, the two frames are byte-identical and the mutation waits. All ten verbs
still return `success`. The regression test pins each verb, keeps `load_scene`
and `remove_robot` as controls that were already correct, and closes the family
with a check that derives the set of scene-recompiling helpers from
`scene_ops` rather than listing it, so a new one is graded on the commit that
introduces it.
