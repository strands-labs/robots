### Fixed: a contact predicate no longer fires for bodies that are visibly apart

`mjData.contact` lists every geom pair inside the *detection* range, which is
the pair's `margin` plus its `gap`. MuJoCo hands only the pairs inside `margin`
to the constraint solver, so a pair between the two thresholds carries **no
force at all** - it is a proximity report, not a touch. `get_contacts` reported
`geom1`, `geom2`, `dist` and `pos`, none of which separates the two, so every
contact predicate answered on geometry alone:

```python
# cube held 25 mm clear of the plate, gravity off, geoms margin=0.001 gap=0.05
contact_any()                                 # True  - zero force anywhere
contact_between("cube_g0", "plate_g0")        # True  - 25 mm apart
grasped("cube", "plate_g")                    # True  - 25 mm apart
body_on("cube", "plate", require_contact=True)  # True  - 25 mm apart
```

`dist` cannot stand in for the distinction: a pair with a wide `margin` is
load-bearing at a *positive* distance (the body hovers on a real force) and the
near miss above is positive too, so its sign splits neither case. Assets ship a
non-zero `margin`/`gap` - a foot geom fractions of a millimetre off the floor is
the usual one - so a locomotion or placement clause gated on contact inherited
the proximity verdict.

Each `get_contacts` record now reports `active`, taken from `mjContact.exclude`,
which is MuJoCo's own decision to hand the pair to the solver; the text summary
names the touching count and marks the proximity-only pairs. Every predicate
resolves it through the shared
`strands_robots.simulation.predicates.contact_is_active`, so they cannot
disagree about which records count. Proximity reports are still listed - they
are what a clearance query wants - and `get_contact_forces` still reports the
load a touching pair carries. A payload that does not report `active` is read as
a touch, so a backend or stub that cannot make the distinction keeps its
previous verdict instead of silently answering `False` for every contact.
