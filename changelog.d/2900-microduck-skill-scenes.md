### Docs: every Microduck skill names the scene it needs

`docs/policies/microduck.md` opens by listing the nine shipped Pollen weights
`MicroduckPolicy` wraps and says they drive the biped "through the standard
`Robot(...).run_policy` seam - in MuJoCo or on hardware". Five of the nine run on
the scene the registry entry declares. Four do not: `roller` and `roller_crouch`
need the four passive ankle wheels that only `scene_rollers.xml` carries, and
`ball_kick_left` / `ball_kick_right` need the ball prop that only
`scene_ball.xml` places. `Robot("microduck")` resolves a scene with zero wheel
joints and zero ball bodies, and `examples/microduck/render_video.py` builds
exactly that, which is why the page said any shipped weight "drops straight in".

Running one of the four there is not an error and reports success: the policy
writes the same fourteen control targets, and the physics has nothing to roll on
or nothing to kick. A reader following the page got a duck standing still with no
indication why.

The scenes were never missing. All three ship in the one asset directory the
entry already downloads, and the entry names the fourteen-hinge model on purpose
- `tests/simulation/test_microduck_asset_matches_the_declared_shape.py` pins
that, and records the reason ("a caller can load it by path"). The gap was that
no page said which scene a skill needs.

The page now carries a skill-to-scene table, the by-path route for the two
variants, and the layout invariant a raw position read has to care about:
`scene_ball.xml` appends the ball's free joint, so the robot's `qpos` layout is
byte-for-byte the default one, while `scene_rollers.xml` inserts two passive
wheel joints after `left_ankle` and two after `right_ankle`, moving nine of the
fourteen actuated joints - the two left wheels land where `neck_pitch` and
`head_pitch` sit on the default scene. The actuator order is identical on all
three, so a policy writing `ctrl` is unaffected, and `MicroduckPolicy` reads its
observation by joint name rather than by slice.

No behaviour changes. A new guard cross-references the advertised list against
the table, so a tenth weight cannot be advertised without naming its scene, and
re-derives every row from the compiled model, so the table's claims are measured
rather than asserted. A registry `variant=` spelling would be a nicer front door
and is deliberately not invented here: none of the seventy-three entries declares
one, so that schema is a public-API decision rather than a docs fix.
