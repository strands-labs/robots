### Fixed: the Isaac backend's `replicate()` builds the fleet it reports

`replicate()` was a stub that reported success for doing nothing. Its body was a
comment between two clock reads, so the "Build time" it quoted was the duration of
two assignments:

```python
t0 = time.perf_counter()
# In full implementation: use omni.isaac.cloner.Cloner
# to replicate the scene N times
self._replicated = True
self._num_envs_active = n
elapsed = time.perf_counter() - t0
```

Measured on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G):

```
sim.replicate(64)
# before: "Replicated to 64 environments. Build time: 0ms. Device: cuda:0."
#         prims under /World: 69 -> 69          (nothing created)
#         get_state() num_envs: 64              (the fabricated count, propagated)
#         add_robot() -> error "Cannot add robots after replicate()."
# after:  "Cloned the scene into 64 environments (63 clones of 2 source prim(s)
#          plus the source as env_0, N prims) in Nms at 1.50m spacing on cuda:0."
```

Three things were wrong at once, and the second is the one that hurt most. The
count was fabricated, and `get_state()` repeated it. `_replicated` was set, which
**permanently refuses `add_robot`** - so the no-op also locked the caller out of
the scene it had not replicated. And the module the comment named does not exist on
Isaac Sim 6.x: `omni.isaac.cloner` raises `ModuleNotFoundError`; the cloner lives
at `isaacsim.core.cloner`.

`replicate()` now clones through `isaacsim.core.cloner.GridCloner`. Every
registered robot and object is cloned into `{stage_path}/envs/env_i` on a square
grid, with `replicate_physics=True` and inter-environment collision filtering. The
scene already on the stage is environment 0, so `num_envs` counts it -
`replicate(64)` produces the source plus 63 clones.

Every failure now leaves the simulation **un-replicated**, so a retry is possible
and `add_robot` is not refused on the strength of a clone that did not happen. That
covers an absent cloner extension (it is a Kit extension, so it resolves only
inside a running Isaac Sim), a raising cloner, and - the important one - a cloner
that returns without error having added no prims. That last case is refused rather
than reported, because it is precisely the shape the stub had.

The payload reports what was built rather than what was requested:
`clones_created`, `prims_created`, `build_time_ms`, `spacing`,
`physics_replicated`, `collisions_filtered`. A failed collision filter is not fatal
- the clones exist - but it is reported in the text and the payload, because a
fleet whose environments push each other around is not N independent episodes.

`num_envs` and `spacing` are validated on the shared
`positive_whole_number_error` / `positive_finite_number_error` domains before any
stage work. `IsaacConfig` validated its own `num_envs` at construction, but this
argument bypassed that entirely: a negative count reported success having built
nothing, and a `bool` passed as a count of one.

`replicate(1)` is an accepted no-op - the scene is already one environment - and
deliberately does *not* mark the simulation replicated, so `add_robot` keeps
working. Marking it would refuse every later `add_robot` on the strength of a call
that cloned nothing, which is what the stub did for every count.
