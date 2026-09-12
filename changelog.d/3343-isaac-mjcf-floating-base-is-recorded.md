### Fixed: a registry humanoid's floating base is recorded, so it reports `base_*`

A composition-only defect, and the most instructive one in this series: two changes
that are each correct alone combined into a bug, with no textual conflict between
them.

* One added `fixed_base` to `_RobotState` and made `get_observation` emit
  `base_pos` / `base_quat` / `base_lin_vel` / `base_ang_vel` only for a robot whose
  base is free. There, the only route to a floating base is
  `add_robot(urdf_path=..., fix_base=False)`, and it records what the caller passed.
* The other made `add_robot` resolve a robot name through the registry, convert the
  resulting MJCF to USD, and load it by the native USD path. There, no `fixed_base`
  field exists at all.

Composed, the USD construction site recorded the `_RobotState` **default** -
`fixed_base=True` - for every converted MJCF. So a registry humanoid or quadruped
landed on the stage with a genuinely free root, because its MJCF declares
`<freejoint/>`, and was reported as bolted down:

```python
sim.add_robot("unitree_g1")
sim.get_observation("unitree_g1")
# before: {"left_hip_pitch": ..., ...} - and none of the four base_* keys
```

`get_observation` therefore omitted every `base_*` entry for exactly the robots
whose base is what a locomotion policy is conditioned on. **18 of the shipped
registry's 64 MJCF robots are humanoids** (`unitree_g1`, `unitree_h1`, `talos`,
`cassie`, `op3`, `booster_t1`, `jvrc` and the rest), plus 9 mobile bases and 2
aerial.

The base is now read from the MJCF, via a new
`loaders.mjcf_declares_floating_base()`. That asymmetry with the URDF path is the
whole design: **URDF cannot declare a floating base** - its universal convention
for a mobile robot is a root link with no parent joint, byte-identical to how a
bolted arm declares its base - so that path has to ask the caller. **MJCF can**,
with `<freejoint/>` or `<joint type="free">`, so asking the caller there would be
asking them to restate what they already handed over.

The predicate reads the whole spliced model, because a root body can live in an
`<include>`d fragment, and it reads both spellings through the joint-tag constants
`loaders` already owns, so a reader consulting only `<joint type="free">` cannot
drift back in. Only a **top-level** body counts: a free joint deeper in the tree is
a free-flying child (MuJoCo's idiom for a detached payload), and reporting that as
a floating base would put `base_pos` on a robot bolted to a table. An unreadable or
malformed file answers `False` rather than raising - the caller is deciding whether
to *report* four keys, and a robot that loads without `base_*` is a smaller error
than an `add_robot` that refuses over a file MuJoCo itself may accept.

A plain USD asset keeps the fixed-base default: this backend did not import it, so
the caller's `fix_base` describes nothing about it and the asset's own articulation
root is the only truth - which is why `add_robot` refuses `fix_base=False` on that
path rather than recording a claim it cannot check.

Measured against the **real shipped assets** on a host with the registry
downloaded, 62 MJCF robots resolved and the split falls exactly along physical
category: **17 of 18 humanoids** floating, 9 of 9 mobile bases, 2 of 2 aerial,
3 of 3 mobile manipulators, 5 of 8 hands - and **0 of 20 arms**, which is the
control that matters, since an arm reading floating would put `base_pos` on a
bolted robot.

The one humanoid that reads fixed is `rby1`, and it is correct: its base is
**planar** (`joint_x` slide, `joint_y` slide, `joint_th` hinge), not a 6-DoF root.
That model also carries a commented-out `<!-- <joint type="free" .../> -->` above
it, so a text scan for the attribute would have reported that wheeled base as
floating. Reading the parsed tree gets it right for free, because ElementTree
discards comments - both cases are pinned.

Driven end to end on an A10G under Isaac Sim 6.0.1 against the real Unitree G1
from `mujoco_menagerie`, whose root body `pelvis` carries
`<freejoint name="floating_base_joint"/>` - the *named* spelling, which is the one
a reader checking only for a bare `<freejoint/>` would miss:

```
add_robot(unitree_g1) -> success   (MJCF /sr/assets/unitree_g1/scene.xml -> USD)
recorded fixed_base=False
base_* keys: ['base_ang_vel', 'base_lin_vel', 'base_pos', 'base_quat']
```

`examples/isaac_on_aws/smoke.py` carries both assertions, so the fix is graded on
hardware rather than only against fixtures.

