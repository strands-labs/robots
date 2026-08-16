### Fixed

`main` was red after #2290 and #2291 merged half a minute apart: #2290's premise
test manufactured a stranded camera key through the `remove_robot` route that
#2291 had just closed (`add_robot` now registers only the robot's own cameras,
each carrying `origin_robot`, so removal takes every one of its keys with it).
The contract the test pins - an observation key that names no compiled camera is
absent, not filled in with another view - still matters, so the premise is
re-established through a route that survives #2291: `replace_scene_mjcf`, whose
documented contract leaves the python-side camera registry untouched while the
compiled cameras its keys name go away.
