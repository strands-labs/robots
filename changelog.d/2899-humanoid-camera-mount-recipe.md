### Docs: the humanoid camera mount is documented, and graded against the asset

Every camera-mounting recipe in the tree is arm-shaped. `world-building.md` reads
the mount from `list_bodies(...)["gripper_body"]`, and `camera-naming.md` and
`annotation.md` both mount on `<arm>/gripper`. A humanoid reports
`gripper_body: None`, because that hint set (`gripper`, `hand`, `jaw`, `ee`,
`tool`) is arm-shaped, so the general recipe correctly declines and sends the
reader to the full `bodies` list - without saying which body, or what offset.

For a head camera there is nothing obvious to pick. No variant of the shipped
Unitree G1 asset has a head, neck or eye body: `g1.xml`, `g1_mjx.xml`,
`g1_with_hands.xml` and their three scene wrappers all report zero, and the only
four sites are two IMUs and the two feet. The body tree ends at the wrists, so a
reader hunting for a head mount finds none and has to derive the offset alone.

`docs/robots/humanoids.md` now documents the mount that does work - the torso,
with a local-frame offset - and `tests/simulation/test_humanoid_camera_mount_recipe.py`
grades it, because the recipe is prose about a third-party asset this repository
downloads rather than vendors. The guard parses the recipe's own fenced block
instead of restating it, so the doc stays the single source of truth: editing the
offset in the prose edits what the cells assert. It checks that the body the
recipe names exists in the shipped asset, that the documented offset lands in
that body's frame rather than the world's, that the mounted camera renders, and
that the premise still holds - if a future asset revision adds a head link, the
premise cell fails and the recipe should name that body instead of an offset.

The same file pins why the arm-shaped guess declines here rather than naming a
leg. A bare-substring match would answer `left_knee_link`, because `ee` occurs in
`knee`; hints have matched name *components* since the word-boundary fix, and
this holds that behaviour on a humanoid, where the knee is a body the robot
actually has. Reverting the matcher to a bare substring fails this file.
