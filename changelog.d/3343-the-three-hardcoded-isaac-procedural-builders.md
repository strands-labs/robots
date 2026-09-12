### Removed: the three hardcoded Isaac "procedural builders"

`_build_so100`, `_build_panda`, `_build_unitree_g1` and the
`get_procedural_robot` / `list_procedural_robots` lookup are deleted. Beyond
describing no real robot, the data could not have been authored into a working
articulation: `JointDef` carries no joint anchor frame (which is what a USD
revolute joint is defined by), `BodyDef.position` meant parent-relative from the
loaders but cumulative world-frame in those tables, `JointDef.axis` is a free
vector where `UsdPhysics` takes an X/Y/Z token and `panda_joint4`'s `(0,-1,0)` is
unrepresentable as one, `stiffness` was `0.0` on every joint so nothing would have
held a pose, and six `unitree_g1` bodies declared `mass=0.0`.

`ProceduralRobot`, `BodyDef`, `JointDef` and `_validate_kinematic_tree` stay -
they are the return type and shared guard of `load_urdf` / `load_mjcf` /
`load_usd`, which are unaffected.
