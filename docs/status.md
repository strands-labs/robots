---
description: Where v1 is, honestly. What each part does, how it is checked, and what still waits for a real arm.
---

# Status

{{drawing:v08_parts}}

v1 is built one part at a time on the [`dev`](https://github.com/strands-labs/robots/tree/dev) branch, one pull
request per part, each with its own gate. The plan and its decisions live in
[#4749](https://github.com/strands-labs/robots/issues/4749); the schedule, with a checkbox per part, in
[#3818](https://github.com/strands-labs/robots/issues/3818).

**The whole suite:** `pytest -q tests` → **589 passed, 25 skipped** on a Mac (the skips: Isaac and Newton need CUDA,
live Bedrock turns are opt-in, the hardware tests need a real SO-101, and five docs examples need one of those or serve a page). Plus ruff, ruff format, mypy, `lint-imports` (8
contracts) and a 250-line cap on every file. The package is 10,647 lines of Python in 131 files; v0.5 was 258,503.

**The acceptance suite** is what 1.0 means: 15 checks. `pytest tests/acceptance -rs` runs 14 of them, green anywhere
with the extras installed; check 9 (`tests_integ/acceptance`) needs a real SO-101.

| part | what it adds | gate |
|---|---|---|
| 0 | an empty package on `dev` | — |
| 1 | types + `Description` | — |
| 2 | `Simulation` + MuJoCo | acceptance 1, 2 |
| 3 | `Robot` is one class for sim and hardware | — |
| 4 | `Policy`: any LeRobot checkpoint, in-process or over Zenoh | acceptance 3, 4 |
| 4b | ONNX skills (`Policy("holosoma")`, the microduck skills) | G1 stands 10 s, the duck walks 1.29 m; acceptance 5 |
| 5 | drivers, the free twin, the operator gate | acceptance 10; 9 needs a real SO-101 |
| 5.5 | a Robot is a device: manifest, safety chain, estop | acceptance 11, 13 |
| 6 | Isaac Sim and Newton backends | the GPU tests on an L40S (skipped without CUDA) |
| 7 | the mesh: `Mesh`, `use_mesh`, Zenoh and AWS IoT | acceptance 7 |
| 8 | agent tools generated from the manifest | acceptance 12, 14 |
| 9 | `run(record=)` writes LeRobot v3 datasets | acceptance 6, 15 |
| 10 | the dashboard: the world in a browser tab | acceptance 8 |
| 11 | `dev` becomes `main` | — |
| 12 | these docs and the notebooks | `mkdocs build --strict`, every page's first example runs |

Measured on an L40S on 2026-10-15: 12 of 24 LeRobot policy families served by `lerobot-policy-server` drove a sim arm
over Zenoh with every action inside the joint limits (LeRobot itself refuses the rest, with its own message); Newton
stepped 1,024 copies of a world at 1.75 million env-steps/s.

## Still open

- On a real SO-101 follower: estop latency during a policy run, the mesh watchdog when a controller dies, the
  read-only bus check.
- FLUX.3 in the LeRobot matrix (its checkpoint is gated on the Hub).
- Part 11: `dev` becomes `main`, and v1.0.0.
