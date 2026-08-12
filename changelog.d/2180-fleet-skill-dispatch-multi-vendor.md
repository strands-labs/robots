### Added: fleet suite scaffold + capability-based skill dispatch across heterogeneous robots

`examples/fleet/01_skill_dispatch_multi_vendor.py` (issue #2180, epic #2179):
one MuJoCo world hosts an SO-101 arm, a LeKiwi wheeled base and a Unitree Go2
quadruped via repeated `add_robot`. A skills table maps skill names to
capability requirements over registry metadata (category, joint count,
gripper) and to execution bindings (`create_policy` provider or `move_to`
motion primitive); manifests derive from that metadata alone, so the dispatch
path contains no per-embodiment branching. Matching runs through the suite's
shared `capabilities.py` hard-constraint filter - a task no robot can serve
is rejected with a per-robot machine-readable reason, never dropped - and
every dispatch passes a HITL gate (`STRANDS_MESH_HITL_ACTIONS=none` is the CI
posture). Execution is one synchronized `run_multi_policy` batch plus
`move_to` on MuJoCo; `--backend isaac` runs the identical dispatch layer with
a sequential per-robot `run_policy` fallback marking the portability boundary
(#2122, #2123). Ships the `examples/fleet/` scaffold and README.
