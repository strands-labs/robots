### Fixed: a dict-form joint write refuses a `robot_name` no robot carries

`set_joint_positions` / `set_joint_velocities` now refuse a `robot_name` that
names no robot in the world in the dict form, with the message the ordered
(list) form already used. Previously the two accepted forms answered the same
bad argument differently, and the dict form's answer was success:

```python
sim.set_joint_positions([0.1, 0.2], robot_name="nobody")
# -> error: Robot 'nobody' not found. Available robots: [...]

sim.set_joint_positions({"j1": 0.1}, robot_name="nobody")
# -> success: "Set 1/1 joint positions, FK updated"   (before)
```

Tolerating it was harmless while the dict form ignored `robot_name` outright --
there was nothing for a wrong value to be wrong about. Scoping an unqualified
key to `robot_name`'s namespace made the argument load-bearing: it now selects
the namespace a bare joint name resolves in, so a misspelled robot name means
"scope this write to a namespace that does not exist". That fell through to the
pre-existing cross-robot first-match lookup and wrote *some* robot's joint --
whichever declared the name first -- while the caller believed they had
addressed a specific one, and the success text reported the requested count for
a robot that never moved. It is the wrong-robot write already fixed for an
unqualified name, reached instead through an unchecked scope.

The refusal lives in the shared joint-write resolver, so it covers both setters
and both call forms from one place, and it covers two kinds of bad name at
once: one no robot carries, and one that cannot be a registry key at all (a
list, a dict, a set -- what wrapping a name in brackets or passing a half-built
kwargs mapping produces). The lookup is routed through the total
`registry_entry`, so the second kind resolves to "no such robot" and is
reported rather than raising `TypeError: unhashable type` past the
`{"status", "content"}` contract.

Blast radius: this refuses calls that previously succeeded, which is the point
-- each one was writing a joint on a robot the caller did not name. Nothing
that resolved correctly changes. A bare key with no `robot_name` at all, a
fully qualified `<robot>/<joint>` key, and a bare key that names no joint of
`robot_name` but does name one elsewhere in the scene all reach the same lookup
as before. The one in-tree caller that passes `robot_name` (the LIBERO
adapter's init-state arm-qpos apply) takes it from `list_robots()`, so it
cannot trip the new refusal.
