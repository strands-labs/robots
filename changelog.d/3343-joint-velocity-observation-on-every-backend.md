### Fixed: the Isaac and Newton backends emit the per-joint `.vel` observations their consumers read

The `SimEngine.get_observation` schema now documents a per-joint velocity entry,
`"<joint_name>.vel"`, additive beside the position key. It was previously
undocumented at the ABC and lived only in the MuJoCo implementation (emitted
since #761) - which is exactly how two backends shipped without it: Isaac read
`get_joint_positions()` and never `get_joint_velocities()`, and Newton read
`joint_qd` into a local for its floating-base twist and never emitted a
scalar-joint entry from it.

The gap broke real consumers three different ways, each on a policy that works
unchanged on MuJoCo:

* the **WBC balance controller** degraded to zero joint velocities with a
  one-time "Gait stability may degrade" warning - open-loop on the quantity it
  exists to feed back;
* the **microduck** and **ProtoMotions** observation packers raised `KeyError`;
* an RL **`SimEnv`** with `.vel` in its `actor_obs_keys` refused at reset.

Both backends now emit the key, spelled exactly as MuJoCo spells it. On Isaac the
read has its own handler so a handle predating `get_joint_velocities` (or a read
returning `None`) degrades to positions-only rather than taking the already-read
positions down with it - the schema requires joint state even when other reads
fail - and `None` is omitted rather than zeroed, because a present-but-zero
velocity reads as a real measurement and silences the WBC warning that guards the
absent case. Verified on `nvcr.io/nvidia/isaac-sim:6.0.1` (A10G): `j1.vel` reads
near zero at rest, nonzero with the right sign mid-transient after a
`send_action`, and near zero again once settled.

Two adjacent defects fixed in the same change, because emitting the key exposed
them:

* **Newton's `joint_vel_std` configured a channel that did not exist.**
  `set_obs_noise` accepted and documented it all along, while the noise pass
  applied `joint_pos_std` to every entry it was handed. The pass now splits by
  the `.vel` suffix exactly as MuJoCo's `_apply_obs_noise` does; without the
  split, position noise would land on velocities and `joint_vel_std` would stay
  inert.
* **Newton's velocity index is `_joint_dof_index`, not `_joint_coord_index`.** A
  free joint upstream shifts the two apart (7 position coords vs 6 velocity
  dofs), so indexing `joint_qd` by the coord map would silently read a
  neighbouring joint's velocity for every joint downstream of the free one. The
  regression test plants a trap value at the wrong slot.

Also corrected: the WBC policy's docstring claimed "the current MuJoCo backend's
unified observation exposes joint *positions* only" - false since #761 - which
sent anyone diagnosing a zero-velocity gait toward the one backend that was fine.
