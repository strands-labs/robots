### Added: the acceptance suite, the 1.0 definition

`tests/acceptance/` is one test per 1.0 check, on real backends with no test doubles: a sim arm resets, steps and
renders; two robots share one world; a LeRobot checkpoint drives the arm locally and over Zenoh; the ONNX presets
stand and walk; a recorded dataset rereads in a fresh process; two mesh peers exchange state, rpcs and an e-stop; the
dashboard shows a live robot's joints; a refused command writes nothing; a robot is a device; every agent tool runs
and refuses through an agent; an e-stop always wins; an agent's run on a real body needs the operator; a plain
script records two cameras. A check that needs what the machine lacks (LeRobot, a browser, a GPU) skips and names
it, so `pytest tests/acceptance -rs` is the checklist. Check 9, a physical SO-101, stays in `tests_integ/acceptance/`.
