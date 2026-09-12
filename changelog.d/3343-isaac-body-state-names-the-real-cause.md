### Fixed: `get_body_state` no longer reports a registered object as "not found"

`IsaacSimulation.get_body_state` resolves a name against the object registry
first, then the stage. When both missed it emitted one message for every cause,
and for a registered object that message contradicted itself:

```
Body 'mug' not found on the Isaac stage. Known objects: [mug]. Robots: []
-- address robot links as '<robot>/<link>' (e.g. 'robot/panda_hand') or pass
an absolute prim path ('/World/...').
```

The searched-for name is printed inside that sentence's own list of *known*
objects.

The contradiction is not the expensive part. Both remedies it offers - respell
the name as `<robot>/<link>`, or pass an absolute prim path - are the remedies
for a **misnamed** body. The name was already right, so following either produces
the identical refusal, and a caller works through the advice it was given while
the actual cause is named nowhere. Three distinct states reach that branch with
the name present in the registry, and every one is about the object's prim rather
than its name:

* the object never got a rigid-prim handle;
* the handle raised on `get_world_pose` - the invalidate-on-reset family, where a
  scene change since the last `reset()` leaves the handle stale;
* the pose came back in a shape `_to_float_list` rejects.

The refusal now says the object is registered but unreadable, distinguishes a
missing handle from one that raised (different investigations), and points at
`reset()`.

It also keeps the **one** of main's two remedies that genuinely applies. Of
"respell it as `<robot>/<link>`" and "pass an absolute prim path", the second
works for exactly this state: the prim-path route bypasses the dead handle and
reads the stage. Measured - `get_body_state("mug")` refuses while
`get_body_state("/World/Objects/mug")` succeeds on the same engine. An earlier
version of this fix said only that respelling "will not help" and dropped both,
which removed the working remedy along with the useless one; adversarial review
caught it. The message now spells the call out, and the premise is pinned by a
test that drives both routes rather than asserting the advice is sound.

This matters most to the consumers this read exists for. The predicate DSL
(`body_above_z`, `body_on`, `distance_less_than`) resolves bodies through it, and
so does `LiberoAdapter._read_eef_pose`. A stale handle after a scene change is
exactly what a task-success predicate hits mid-episode, and "not found" sends the
reader to check the body name in their predicate - which is correct - rather than
to the reset the state actually needs.

The unknown-name message is unchanged and pinned as a control: widening the new
wording to every miss would lose the known-objects list and the two addressing
forms, which are the right guidance when a name genuinely is wrong. The stage read
is still attempted for a registered name too, since an object may be registered
without a handle and still resolve as a prim.
