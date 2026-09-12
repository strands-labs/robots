### Fixed: a worker thread adding a robot no longer crashes the Isaac pump

`IsaacSimulation.pump` is the main-thread half of this backend's threading
contract, and its own docstring states the other half:

> A web UI calls `get_observation`/`send_action` from worker threads where
> Isaac's renderer / physics deadlock. Those calls instead enqueue actions and
> read cached frames; this pump (run on the owning main thread) is the single
> place that actually advances the sim and renders the cameras.

So concurrent access is the *designed* usage, not an edge case - and `add_robot`,
`remove_robot`, `add_camera` and `destroy` all mutate `_robots` / `_cameras`
under `self._lock`. `pump` walked those dicts live and unlocked:

```python
for rname, r in self._robots.items():       # unlocked
    ...
```

An agent adding a robot from a worker thread while the pump ran therefore raised

```
RuntimeError: dictionary changed size during iteration
```

on **6 of 6 trials** with 200 robots and a worker adding 60 more mid-walk.
Removal - what `remove_robot` does - is the same failure from the other
direction.

Two things make this worse than a dropped frame.

**The handler that looks like it covers this cannot see it.** The body already
wraps each robot's joint read in `except (RuntimeError, ValueError,
AttributeError, TypeError)`. But the exception is raised by the `for` statement's
own call to the iterator, which is *outside* that `try`, so it is not a
best-effort skip - it unwinds the whole method.

**It lands on the main thread.** That is the thread that owns Kit and runs the
pump loop, so the escape takes the application down rather than degrading one
tick. A UI whose whole purpose is to let an agent build a scene while the
preview runs is exactly the caller that hits it.

Each registry is now copied under the lock and the copy is iterated. The lock is
held only for the copy, deliberately **not** across the body: the joint read and
the frame grab both reach into Kit, and holding it across them would serialize
the pump against every tool call for the length of a render. That boundary is
pinned by a test that probes the lock from inside the walk, on the thread the
articulation read runs on.

A robot added mid-pump is served by the following tick rather than dropped, which
is pinned too - a snapshot that walked nothing would satisfy every crash test
here, and so would one that silently forgot late arrivals.

**Three walks, not two.** ``pump`` calls ``_converge_render`` at step 2 - the idle
preview path ``run_pump_forever`` takes by default - and that helper walks
``_robots`` as well, two lines *above* the loops in ``pump`` itself. A snapshot
added only inside ``pump`` leaves the crash reachable from the line before it.
``_prim_body_state`` walks ``_robots`` too, and ``get_body_state`` runs it inline
on the calling thread whenever no pump is engaged. All three are snapshotted.

Each snapshot is pinned by a test that **fails when that snapshot alone is
removed** - verified as a mutation matrix (2, 3 and 7 failures respectively).
Getting there took two corrections worth recording, because both first attempts
passed while pinning nothing:

* The camera tests used an instant ``_grab_frame`` stub, so the camera walk
  finished before any timed mutator could reach it - and the robot walk ahead of
  it takes ~400 ms with 200 slow articulations, so the mutator had always
  finished. 15/15 passed with the camera snapshot removed. They now mutate from
  *inside* the frame grab, which lands the change during the iteration by
  construction.
* The ``_converge_render`` test first mutated from ``world.step``. That helper's
  shape is ``for _ in range(n): for r in <registry>: ...; world.step()``, so a
  mutation there lands *between* outer iterations - and an unsnapshotted
  ``self._robots.values()`` is a fresh view each iteration, which never sees a
  size change mid-walk. 17/17 passed with the snapshot removed. It now mutates
  from the per-robot joint read, the only hook actually inside the walk.

Both are the same lesson: a concurrency test that does not pin the mechanism
passes for a timing reason and reads as coverage.
