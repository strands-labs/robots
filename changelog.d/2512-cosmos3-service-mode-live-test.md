### Added: live service-mode integration test for the cosmos3 policy provider

`tests_integ/policies/cosmos3/test_service_mode_live.py` runs a real
msgpack+NumPy WebSocket round-trip against a pre-running Cosmos Framework
RoboLab policy server (`nvidia/Cosmos3-Nano-Policy-DROID`), mirroring how
`tests_integ/groot/test_n17_live_server.py` covers GR00T's live-server mode.
Env-gated on `COSMOS3_LIVE_SERVER` (plus `COSMOS3_SERVER_HOST` /
`COSMOS3_SERVER_PORT`), so it skips cleanly on CI boxes without the server or
the optional deps. Asserts the per-step dicts match the embodiment-registry
DROID `joint_pos` layout, every value is a finite float in a plausible radian
range and not identically zero, and that the connection survives a second
round-trip. To make the raw `[T, D]` chunk assertable, the service backend now
surfaces it (and the server's optional rollout video) on
`Cosmos3Policy.last_rollout`, matching the diffusers backend; previously the
service path left `last_rollout` as `None`.
