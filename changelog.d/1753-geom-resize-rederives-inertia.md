### Fixed: resizing a geom re-derives the owning body's mass, center of mass and inertia

A body that declares no `<inertial>` takes its whole inertial row from the geoms it
owns, integrated once by the compiler, and no step recomputes it.
`set_geom_properties(size=...)` refreshed the geom's collision bounds but left that
row describing the shape the body used to have, so the body collided as its new
shape while resisting rotation as the old one - 0.1 Nm for 1 s spun a cube grown
from 0.1 m to 0.4 m to 60.0 rad/s where its geometry implies 3.75 rad/s. Where the
geom carried a density rather than a mass, the stale row also kept the old mass and
balance point: growing one weight of a dumbbell left it 2.22 kg and centered
instead of 15.52 kg with its center of mass 17 cm off axis.

It was also order-dependent. The new size is recorded in the spec, so any later
scene mutation recompiles it and the compiler silently corrects the tensor - the
same two calls produced 16x different physics depending on whether an unrelated
`add_object` happened afterwards.

The row is now read from a compile of the persisted spec, so it equals what the
next recompile produces for every size-defined primitive and for a body with
several geoms, with no per-shape formulas to keep correct. `body_subtreemass` and
the reference constants the constraint solver scales with are refreshed with it. A
body that declares its own `<inertial>` takes nothing from geometry and is left
untouched, and a resize whose geometry cannot be compiled is refused with the spec
and the model still describing the same shape.

Re-deriving the row leaves the running scene exactly where it was. The reference
constants are evaluated at the model's reference configuration, so the MuJoCo call
that recomputes them is given its own scratch state rather than the scene's - handed
the scene's own it would rewind every joint and body pose to its declared value
while leaving the velocities untouched, so a resize issued after any stepping (the
documented mid-run randomization path) would silently teleport the scene.
