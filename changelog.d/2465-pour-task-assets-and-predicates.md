### Added: articulated-container pouring tasks - hinged/sliding carton assets + particle-proxy pour predicates

Contact-rich container manipulation (open a carton, pour contents into a
receptacle) was not modelable: no articulated-container assets existed in the
object path, and neither backend exposes liquid simulation. This ships the v1,
fluids-deferred version. `strands_robots.simulation.task_objects` bundles three
MJCF task objects inside the package - `hinged_carton` (lid on a hinge,
`cap_hinge`), `sliding_carton` (lid on a slide, `cap_slide`; the stable choice
for lid-down pouring mounts, since a lid-down hinge creeps open under load -
MuJoCo dof friction saturates under persistent torque), and `open_tray` (rigid
receptacle) - resolved by `task_object_path(name)` (traversal-safe, unknown
names refused with the valid set) and loaded through the existing
`add_robot(urdf_path=...)` route so the cap joints are observable by the joint
predicates. Contents are proxied by rigid spheres, and three new closed-registry
predicates make "poured" scoreable with zero physics-engine changes:
`particles_inside(particles, container, min_fraction, ...)` (success),
`particles_spilled(particles, containers, max_spilled, ...)` (failure; an
unresolvable name can lower a success fraction but never fires a failure), and
the dense reward `particles_inside_fraction(...)`. `stop_when` clauses probe the
new list-valued kwargs element-wise like every singular body kwarg.
`examples/17_pour_task.py` composes the task end to end (spawn, benchmark,
mock-policy score, scripted pour), CPU-only; the MuJoCo smoke suite pins that
the assets attach, the hinge cap actuates, the scripted pour flips the
predicates, and the seeded benchmark scores bit-identically across runs.
