### Added: `randomize` and `set_obs_noise` on the Isaac backend

Both were the `SimEngine` raising stubs - while
`docs/simulation/domain-randomization.md` and ~30 example call sites drive
`randomize()` and ~10 drive `set_obs_noise()` through the backend-agnostic
surface. The identical script randomized on MuJoCo and Newton and raised
`NotImplementedError` on Isaac. They were also the two most-consumed entries in
the action-parity census.

Signatures mirror the MuJoCo reference exactly - names, defaults and parameter
order (`randomize_colors, randomize_lighting, randomize_physics,
randomize_positions, position_noise, color_range, friction_range, mass_range,
seed`) - so the shared-parameter-order grader, whose backend inventory derives
from the `create_simulation` table, held them to the rule the moment they
appeared. Unknown keywords are refused by name; every axis flag is checked on
the shared `boolean_flag_error` domain (a truthy `"false"` must not turn an axis
ON); ranges, `position_noise` and the seed are validated before anything is
written.

What each axis writes on this backend (measured on `isaacsim` 6.0.1):

* **colors** - each object prim's USD `primvars:displayColor` resampled in
  `color_range` (the object handles ship no color setter; the USD attribute is
  what RTX reads);
* **lighting** - every `UsdLux` light's intensity scaled in [0.5, 1.5] of its
  base, color resampled (`create_world`'s ground plane ships one `SphereLight`);
* **physics** - each dynamic object's mass scaled in `mass_range` of its base
  (`set_mass`), each *distinct* physics material's static+dynamic friction
  scaled in `friction_range` - deduplicated by material prim path, so a shared
  material scales once;
* **positions** - each dynamic object teleported to base + a uniform xy offset
  in `position_noise`; z preserved (a z offset buries or drops the object -
  penetration recovery on this backend was measured to fling a buried body to
  1e10 m).

Every scale is measured from a **first-touch base**, mirroring the
anti-compounding property MuJoCo anchors on the authored model - and the GPU
verification caught this shipping broken: the base registry was read with
`getattr(self, "_dr_base", None) or {}`, and `__init__` pre-creates that dict
*empty* - falsy - so a real engine rebuilt the base every call and compounded
(`mass_range=(2,2)` twice took 0.105 kg to 0.421 kg instead of 0.211, and one
seed stopped reproducing one sample set). The `__new__`-skeleton unit fixture
could not see it: having no `_dr_base` at all, it took the branch that stored
the dict, and passed. Fixed with an `is None` read, pinned by a
constructor-built engine test that fails on the falsy spelling, and re-measured
live: base x 2.0 exact, one seed one sample set.

`set_obs_noise` mirrors the sibling signature (`joint_pos_std, joint_vel_std,
camera_jitter_px, seed`) and is applied by `get_observation` through a
suffix-keyed pass identical in shape to MuJoCo's: position noise to the plain
floats, velocity noise to `.vel` floats, an integer-pixel roll to camera frames
(pixels relocated, never invented), `base_*` list signals untouched.

GPU-verified 10/10 on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G): every axis writes
what its report records, the RTX frame visibly changes under colors+lighting
(mean channel shift ~34), mass/friction/position writes land on the live
handles, and a jittered camera frame rides `get_observation`. The success
report's `json` block records every sampled value keyed by entity, so a run is
reproducible from its own report plus the seed.
