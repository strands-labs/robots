### Added: cosmos3 service-mode spawned-server MuJoCo rollout integration test

`tests_integ/policies/cosmos3/test_service_mode_rollout.py` completes the
service-mode half of the rule-8 integration gap (#2462): a module-scoped,
opt-in (`COSMOS3_SPAWN_SERVER=1`) fixture spawns the Cosmos Framework RoboLab
policy server itself - argv-style, output captured to a file, teardown that
terminates the process group even on failure and asserts no orphan survives -
and a short closed-loop MuJoCo Panda rollout is driven by real server actions
through the built-in DROID -> Panda actuator mapping. Every action value is
asserted finite and within the actuator's own `ctrlrange` (read from the
`MjModel`), poses are written only into position servos per the actuator rule
(the tendon-driven gripper is left uncommanded and named), and the policy is
asked across multiple chunks so the WebSocket connection is proven to survive
repeated inference. The readiness deadline is measured on `time.monotonic()`
and a spawn that dies before serving fails with the server's captured log
tail. With the gate unset the whole file skips cleanly.
