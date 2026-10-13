### Docs: notebook 6's Go2 base spawns resting on the ground

The fleet-orchestration notebook added its Go2 base at `z=0.0`, and `add_robot`
printed on every run that the base started 3.0 mm inside the ground, so the
contact solver pushed it from the first step. The notebook now uses the
position that warning names, `[0.0, 0.8, 0.003]`. The plan, the dispatch and the
UNASSIGNED failure path are unchanged; re-executed headless on `main` with the
warning gone.
