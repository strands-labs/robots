### Quality: the namespace-fallback retry that keeps a floating base observable after a scene replace is now driven

`simulation/mujoco/rendering.py` carries three byte-identical copies of one
namespace-fallback retry -- a joint lookup that re-tries by **bare** name after
the namespaced lookup (`robot.namespace + jnt_name`) misses. All three bodies
were unexecuted by the suite.

They matter because `add_robot` records `robot.namespace = "<name>/"` while
`replace_scene_mjcf` recompiles from caller-supplied MJCF that need not
reproduce that prefix, and the registry is not rewritten. After such a replace
the namespaced lookup misses and only the bare retry resolves the joint.
Measured on a floating-base robot added as `hum` and then replaced by an MJCF
declaring the same joints unprefixed, removing all three retries takes
`get_observation` from the four `base_*` keys to none at all -- a humanoid or
mobile base silently observed as a fixed-base arm, and a recorded dataset with
no base position, orientation or velocity.

Nothing was driving them because they mask one another: each helper whose retry
is gone still reaches the base by delegating to the next, so no single-line
mutation was observable. Eight cases now pin the contract that the copies exist
for -- a floating-base robot's observation, its `get_robot_state` base entry and
the free-base joint finder all survive a namespace-dropping replace, for both a
named `floating_base_joint` and an unnamed `<freejoint>` -- plus the negative
case that keeps the pin honest: when the replacement scene's joints are renamed
so the bare name misses too, the base state is absent rather than borrowed from
an unrelated free joint. Two of the three retries are individually observable
under this pin; the third is masked structurally by its own fall-through, so it
is covered and contract-pinned rather than mutation-pinned.

The retry is a fallback, so the family also breaks in the other direction: a
lookup that stops preferring the namespaced name. Two cases compile both
spellings at once -- the robot's `hum/hip` beside a decoy body's bare `hip` --
and pin that the observation and the free-base finder resolve the robot's own.
Reading the bare name unconditionally, or hoisting the retry to run over a
successful namespaced hit, reports an unrelated body's joint angle as this
robot's; both are invisible to every case that carries only one spelling, and
both are what the deferred deduplication of the three copies could introduce.

No behaviour change.
