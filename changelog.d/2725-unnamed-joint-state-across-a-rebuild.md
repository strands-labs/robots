### Fixed: an unnamed joint keeps its state across a scene rebuild

`eject_robot_from_scene` rebuilds the compiled model, which renumbers every joint id, so the
dynamic state is snapshotted under a name-based key and written back once the new model exists.
`_joint_key` built that key and returned `None` for any unnamed non-free joint, on the stated
grounds that "its body may carry several, so the body does not single it out". A `None` key is
skipped, so the joint's `qpos`, `qvel` and `qfrc_applied` were dropped and it came back at its
fresh-compile value while the operation reported `success`.

An unnamed `<joint>` is the ordinary MJCF spelling - a door hinge or a drawer slide is rarely
named - so removing one robot silently reset the articulation of everything else in the scene.
Measured on a scene holding an arm plus a door on an unnamed hinge and a drawer carrying two
unnamed joints: with the door at `qpos 0.70` / `qvel -1.25`, the drawer hinge at `0.31` / `0.44`
and the drawer slide at `-0.17` / `2.05`, `remove_robot("arm")` returned `success` and left all
three at `0.0`, changing 16690 pixels of the rendered scene. The same call now leaves all three
byte-identical to what they were seeded with.

The premise was sound but the conclusion was too strong. The body does not single the joint out,
but the body *plus the joint's position among that body's joints* does: MuJoCo stores a body's
joints contiguously from `body_jntadr` in declaration order, so a third key form
`("body_joint", body_name, ordinal)` identifies one joint of a body that carries several, and
`_resolve_joint_key` resolves it back through the same two values. Nothing shifts an ordinal,
because no scene op inserts a joint into an existing body - the patch vocabulary is `add_body`,
`add_geom`, `add_site`, `set_body_pos`, `set_body_quat` and `delete_body`.

Two joints of one body are what makes the ordinal load-bearing rather than decorative. A hinge and
a slide both use width 1/1, so the width check `_restore_scene_state` already performs cannot tell
them apart; keying both on the body alone would restore each from the other's values, and that
swap is pinned.

Everything else is unchanged and pinned as such. A named joint keeps its `("joint", name)` key. An
unnamed free joint keeps its `("body", body_name)` key, which is unambiguous because the compiler
refuses a free joint alongside any other joint on the same body. A joint whose body is *also*
unnamed still gets no key: nothing identifies either end across the rebuild, so it is reported
rather than guessed at. An ordinal past the body's joint count, a body that has vanished, and a
joint that has since gained a name all resolve to nothing, so one joint is never restored from two
entries.
