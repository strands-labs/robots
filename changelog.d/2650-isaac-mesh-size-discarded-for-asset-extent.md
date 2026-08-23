### Quality: the Isaac mesh path's discard of a requested `size` is now driven

`tests/simulation/test_mesh_size_docs_match_backend_divergence.py` measures a
cross-backend divergence in `add_object(shape="mesh", ...)`: Newton consumes
`size` as a per-axis scale, MuJoCo does not, and both calls report success. #2498
placed Isaac on MuJoCo's side and said so in that gate's prose, and nothing
measured it.

The three surfaces that reach Isaac's mesh path all stop short of the property.
The GPU journey test passes no `size` at all, so it pins "the extent comes from
the asset when none was requested"; the four unit-level mesh cases are refusals
that never reach the success path; and the `load_scene` mesh cases read
`obj.size` on a path where the caller supplies none. A change that started
reading `size` as a scale -- the Newton reading, and the intuitive one for anyone
arriving from Newton code -- would break no test, and the gate above would then
be asserting something false about a third backend, so the artifact the repo
relies on to describe the hazard becomes a source of it.

Six cases now drive the mesh success path in `tests/`, where CI runs, rather than
in `tests_integ/`, which no workflow runs. Only the Kit-only leaves are stood in,
following `test_scene_object_meshes`; `mesh_aabb`, whose output the assertions are
about, runs for real against a `0.1 x 0.2 x 0.3` OBJ box. Both halves of
"ignored" are pinned, because the payload alone does not cover the class: the
report is the asset's extent whatever was requested and is identical to the same
call with no `size`; and no component of the request reaches the prim
construction, searched numerically so a scale arriving as a `numpy` array is not
missed. That second half matters because `resolved_size` is parsed from the
unscaled asset file, so a scale that reached the prim would leave the report
honest and put a wrong-sized object on the stage. A structural assertion that
`_create_mesh_prim` declares no `size` parameter catches the first step of the
two-step regression the others only see the second of.

Seven plausible regressions toward the Newton reading were applied and the pin
re-run. Six fire a different set, and each of the six cases fires uniquely
somewhere: the unused-parameter step is caught by the signature assertion alone,
a scale applied to the handle after `_create_mesh_prim` returns by the
construction assertion alone, the tempting "drop the extent earlier" shortcut by
the no-over-reach control alone, and a request that stops being distinguishable
from the rest of the call by the premise alone.

The request is imported from the divergence gate rather than restated, so the
claim and its pin are one edit apart in either direction, and that gate's Isaac
paragraph now names the pin. The GPU journey test passes the same value
explicitly.

No library behaviour changes.
