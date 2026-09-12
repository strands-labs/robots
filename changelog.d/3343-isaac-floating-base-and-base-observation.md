### Added: a floating base on the Isaac backend, and the `base_*` observation entries it makes meaningful

Two halves of one defect, because neither is usable alone.

**Nothing on this backend could have a floating base.** `fix_base = True` was
hardcoded at *both* URDF import paths - the modern `URDFImporterConfig` and the
legacy `_urdf.ImportConfig` - with no way for a caller to change it. Every URDF
robot was welded to the world, so a humanoid could not fall, a quadruped could not
walk, and nothing said so.

**And the four `base_*` observation entries were absent.** The
`SimEngine.get_observation` schema requires that a robot whose root is a 6-DoF free
joint surface `base_pos`, `base_quat`, `base_lin_vel` and `base_ang_vel` rather
than reporting the free joint as a scalar. MuJoCo emits all four
(`mujoco/rendering.py`) and Newton emits all four (`newton/simulation.py`); across
all 11 files of the Isaac package there were **zero** occurrences. A locomotion
policy reading `base_lin_vel` - the base twist every walking controller is
conditioned on - got nothing on this backend and a value on the other two, which is
what made the documented "policies and observation mappings transfer unchanged
between backends" false for a legged robot.

The two are one change because the absence was *consistent*: with every base
welded, those four keys would have reported four constants.

```python
sim.add_robot("g1", urdf_path="g1.urdf", position=[0, 0, 1.2], fix_base=False)
sim.step(120)
sim.get_observation("g1")["base_pos"]        # -> [0.0, 0.0, 0.05]
```

Verified on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G): the same robot goes
`z 1.2 -> 1.1898 -> 0.9786 -> 0.2245 -> 0.05` over 120 steps and settles on the
ground at its base half-height, while `fix_base=True` holds `z 0.0 -> 0.0`. A
welded base emits no `base_*` keys and a floating one emits all four, with the
values the articulation reports.

`fix_base` is a parameter rather than something read out of the file because URDF
cannot answer it. The format has a `floating` joint type, but the universal
convention for a mobile robot is a root link with no parent joint - byte-identical
to how a bolted-down arm declares its base - so the consumer chooses. That is why
Isaac's own importer takes the flag, and why MuJoCo and Newton, which read MJCF's
`<freejoint>`, have no equivalent parameter. `True` remains the default: it is the
existing behaviour, the shipped LIBERO Franka depends on it, and a fixed-base arm
is the common case here.

It applies to `urdf_path` only. A USD asset carries its own articulation root and a
procedural build authors its own prims, so `fix_base=False` on those paths is
refused with a message naming `urdf_path` as the remedy, rather than accepted and
ignored - which would leave the caller believing they had a floating base and the
observation keys believing the opposite. The flag itself is checked on the shared
`boolean_flag_error` domain, so `fix_base="false"` is refused rather than read as
truthy and silently welding the base of a caller who asked for the opposite.

**Known limitation, measured rather than worked around:** a `reset()` does not
preserve a floating base's spawn height. A robot added at `z=1.2` reads `base_pos`
z `1.2` immediately and `0.0402` after a reset, because `world.reset()` re-applies
each registered prim's default state on `post_reset` - the same mechanism
`load_scene` deliberately avoids a reset for (#1802). Recording the spawn pose via
the articulation's `set_default_state` was tried and does not change that reading,
so `add_robot`'s docstring states it: to drop a robot from a height, step from the
pose `add_robot` leaves rather than resetting first.

**`base_ang_vel` is reported in the BODY frame**, which is the correction this
entry needed most and did not have at first. Linear velocity is world-frame on all
three backends; angular velocity is body-frame - the IMU-gyro convention a
locomotion policy is trained against. The Newton backend says so outright and
carries `_quat_rotate_inverse_wxyz` specifically to convert, "so `base_ang_vel`
matches the MuJoCo backend and the IMU-gyro convention WBC / locomotion controllers
consume". Isaac's `articulation.get_angular_velocity()` returns the world frame, and
it was emitted unrotated - so of the four base channels this was the one whose
numbers silently disagreed with the other two backends, which falsifies the very
parity claim this feature rests on.

It disagreed in the way hardest to catch: **for an upright, un-yawed base the two
frames coincide**. A standing robot reads correct, a robot yawing about world Z
reads correct on Z, and the error appears only once the base tilts - exactly when a
locomotion policy is depending on the signal. Those coinciding cases are pinned
alongside the divergent ones precisely because a test that stood a robot up and
checked the gyro would have found nothing.

The rotation uses this module's own quaternion primitive rather than importing
Newton's helper, which would pull `warp` into Isaac's import path. Two
implementations can drift, so the drift is measured: they are compared over 200
random (quaternion, vector) pairs and agree to 1.3e-15. The expected values were
cross-checked against `scipy.spatial.transform.Rotation` - one of them was written
with the wrong sign first, and the implementation was right.

**The dataset carries the base columns too.** `observation.state`'s schema is derived
from scalar *joint* names, so the four base vectors would be dropped from a recording
even though `get_observation` reports them. `DatasetRecorder.create` takes
`extra_state_specs` for exactly this, and **both** sibling backends pass it - MuJoCo
and Newton, each with the same reasoning already written down:

> those base signals would be dropped and a locomotion / velocity-tracking /
> whole-body-control policy trained on the dataset would be base-blind.

Isaac was the only backend that did not. Adding `base_*` to `get_observation` without
this left an Isaac humanoid dataset **13 columns short** of the MuJoCo one for the same
robot, silently - a missing column is not an error. The gap only became reachable with
this series, since before it `add_robot` created no articulation for a humanoid to have.

The base is detected from `fixed_base` - the field the MJCF free-joint read records -
rather than from a joint id, because this backend has no compiled model to interrogate.
A fixed-base arm declares nothing, so its schema is unchanged, and that is pinned.

Resume validation uses the **expanded** names. A resumed dataset's on-disk
`observation.state` includes the base columns, so checking against the bare joint list
would report a mismatch on every floating-base append. Pinned by reading the
verification call's own argument, because a version that computed the expanded names and
then verified against the joint list anyway passed a weaker check.
