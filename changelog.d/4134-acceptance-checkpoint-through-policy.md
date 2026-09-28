### Tests: acceptance check 2, a lerobot checkpoint drives the sim arm through `Policy`, RTC off and on

`tests_integ/acceptance/` gains the checkpoint check (#3818): `Robot("so101", mode="sim")` runs `lerobot/smolvla_base` through `run_policy(policy_provider="lerobot_local")` for 90 steps with `rtc_enabled`/`async_rtc` off and on, and asserts every step applied, the RTC posture reported back by the policy and the runner, and a joint that moved. Needs a CUDA GPU; run it with `pytest tests_integ/acceptance`.
