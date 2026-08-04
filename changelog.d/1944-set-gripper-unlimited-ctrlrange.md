### Fixed: set_gripper resolves set-points on an MJCF that omits `ctrlrange`, so it works on so101

`set_gripper` derived its open/close set-points from `actuator_ctrlrange` alone
and refused any actuator whose range was not strictly increasing. But MuJoCo
reports exactly `(0, 0)` with `actuator_ctrllimited == 0` for an actuator the
MJCF left **unlimited**, which is a different claim from "this actuator accepts
nothing" - so the guard rejected a gripper it could drive perfectly well.

That was live on a shipped robot. so101's sim MJCF declares neither `ctrlrange`
nor `inheritrange="1"` on its position servos, so `set_gripper` errored on an
arm whose registry metadata named the right actuator (`actuators: ["6"]`) and
whose `move_to` and `rotate_wrist` both worked:

```python
sim.add_robot(name="arm", data_config="so101")
sim.set_gripper(state="close", robot_name="arm", steps=40)
# before: error "actuator 'arm/6' has no usable ctrlrange (0.0, 0.0)"
# after:  success, targets {'6': -0.1745}  <- the driven joint's low end
```

so100 sets `inheritrange="1"` on every actuator, which compiles a real
ctrlrange from the driven joint - the only reason it was unaffected. The two
robots' jaw joints are otherwise near-identical (`(-0.174, 1.75)` against
`(-0.1745, 1.7453)`).

For a JOINT-transmission position servo `ctrl` IS the joint target, so the
driven joint's own limits are the open/close set-points - the same substitution
`move_to` and `rotate_wrist` already make (both read `jnt_range` under
`jnt_limited`; `set_gripper` was the sole outlier), and exactly what
`inheritrange="1"` would have compiled the ctrlrange to.

The fallback is deliberately narrow, and the three cases it must not swallow
are pinned as tests. A `mjTRN_TENDON` actuator keeps refusing: its ctrlrange is
a normalised command space, not joint units - the shipped Franka gripper is
`(0, 255)` - so substituting a joint range there would command the wrong
quantity. This falls out of `_joint_actuator_map`, which maps only
JOINT/JOINTINPARENT transmissions, so a tendon actuator has no entry by
construction rather than by a separate check. An `actuator_ctrllimited == 1`
with a degenerate range is an authored claim that the actuator accepts nothing
and is respected as one. And an unlimited actuator whose driven joint is also
unlimited has no range anywhere to infer from, so it refuses with an error
naming both exhausted sources instead of only the ctrlrange.

Fixes #1942.
