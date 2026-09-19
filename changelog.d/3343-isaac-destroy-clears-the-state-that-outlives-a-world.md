### Fixed: latched forces, sensor noise and randomization baselines no longer outlive `destroy()`

`IsaacSimulation.destroy()` cleared seven registries - `_robots`, `_cameras`,
`_objects`, `_prim_registry`, `_action_controllers`, `_cams_rec_state` and
`_recording_state_dict` - and left six other pieces of per-world state in place.
Measured on a torn-down engine, every one of these survived:

```
_applied_wrenches   _obs_noise   _obs_noise_rng   _dr_base
_frame_cache        _joint_cache
```

Every one is keyed by an object / robot / camera name or a prim path, and those
names are reused across worlds as a matter of course. A second world containing a
`"cube"` is the ordinary case, not a collision someone has to engineer. So the next
`create_world()` silently inherited configuration for a scene it was never given:

* **A latched `apply_force` replayed onto the new world's body of the same name** -
  an external force nobody in that session applied, on a fresh stage. This is the
  sharpest one, because `reset()` *already* clears the wrench registry: the
  cross-backend contract is "reset() clears every latched wrench in the world". So
  `destroy()`, the stronger boundary of the two, was the one that wiped less.
* **`set_obs_noise`'s per-robot sigma kept perturbing `get_observation`**, so a run
  deliberately configured clean carried noise from a previous one - and the noise
  is Gaussian, so the observations look plausible rather than wrong.
* **`randomize()`'s `_dr_base` first-touch baseline**, which exists precisely to stop
  scaling from compounding, anchored the new world's randomization to a pose from a
  world that no longer exists.
* **`_frame_cache` held a full RTX frame and `_joint_cache` a joint snapshot from the
  old stage**, readable as though current.

All six are now cleared, and `_obs_noise_rng` - a seeded generator rather than a
mapping - is dropped.

The clears use `getattr` guards. `destroy()` runs from `__del__`, and many test
modules build a skeleton engine with `__new__` seeding only the attributes they
exercise, so an unguarded `self._applied_wrenches.clear()` would raise
`AttributeError` during garbage collection - surfacing as noise attributed to
whatever test was running rather than to the clear.

Beyond the fixed list, a **drift guard** derives both sides from the source and
requires that anything `reset()` clears, `destroy()` clears too. `destroy` tears the
world down where `reset` only rewinds it, so a registry the weaker boundary wipes
and the stronger one retains is incoherent whichever registry it is - and that
asymmetry is exactly how the wrench leak arose.
