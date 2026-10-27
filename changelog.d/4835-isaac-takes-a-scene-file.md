### Added: `Simulation("isaac")` takes `add_object(path=)` files

`add_object(name, path="room.xml")` on the Isaac backend was refused by name ("use a shape"), so a scene a script
built for MuJoCo had no Isaac counterpart beyond primitives. It now imports the MJCF or URDF with Isaac's own
importer (the conversion a robot gets, cached by content) and references it at `/World/objects/<name>`. A
`free=False` file is static scenery, placed by its root xform. A `free=True` file is first wrapped in one body with
a free joint, as the MuJoCo backend attaches it, so bare world geoms become one rigid body here too; the rigid body
the importer makes of it is the handle, so `pose`, `set_pose` and `remove` behave as on MuJoCo (removal also deletes
the reference root, which the handle alone would leave behind). A failed import rolls back the object's prim the way a
robot's is. Isaac's importer keeps geometry, colliders and materials; a file's lights and skybox do not come through.
A bare mesh file is refused with the fix (wrap it in an MJCF). `camera(attach=)`, friction and mass randomization and
`set_gains` stay refused by name.
