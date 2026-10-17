# SIZE — the IoT-native mesh, lines added vs the mesh today (2026-10-17)

Counted with `wc -l` on the reference's IoT branch (the reference robots-v1-2 @ bcfb45e is the baseline).
Rule: every file ≤ 250 lines (R-V rules); `mesh.py` was 247 and is 249 after the wiring.

## Code

| file | lines | MAPPING estimate | covers |
|---|---|---|---|
| `strands_robots/mesh/iot/__init__.py` | 22 | 15 | lazy exports (`__getattr__`): `import strands_robots.mesh` loads only `direct` + `options` |
| `strands_robots/mesh/iot/options.py` | 76 | — (the `iot=` knob) | `IotOptions`, `attach` (the one entry point `Mesh.mount` calls, before anything else starts) |
| `strands_robots/mesh/iot/shadow.py` | 171 | 235 (1.1–1.5) | manifest / state / control shadows, refusals verbatim, stale deltas |
| `strands_robots/mesh/iot/rules.py` | 154 | 100 (2.1–2.6) | rules as data, idempotent apply, Basic Ingest telemetry ≤ 1 Hz |
| `strands_robots/mesh/iot/provision.py` | 250 | 140 (3.1–3.4, 3.6, 3.7) | fleet policy 1,901 B + controller add-on 496 B, operator policy, RSA CSR `CN=<id>`, register/rotate/retire/revoke/teardown |
| `strands_robots/mesh/iot/claim.py` | 189 | 110 (3.5) | template + hook as data (CN read from the certificate PEM — EXPERIMENTS §6), `hook_decision`, claim flow over `iotidentity` |
| `strands_robots/mesh/iot/direct.py` | 40 | 80 (4.1, 4.2, 4.5) | rpc mode per link, `reply_ok`, shared subscriptions |
| `strands_robots/mesh/iot/lifecycle.py` | 90 | 60 (4.3) | lifecycle events as AUDIT (parse/watch/Audit) + the disconnect rule; heartbeat kept (EXPERIMENTS §5) |
| `strands_robots/mesh/iot/registry.py` | 52 | 40 (7.1) | grey cards, status words, off switch |
| `strands_robots/mesh/iot/greengrass.py` | 109 | 200 (5.1, 5.2, 5.5) | recipe + deployment as data, `python -m` entrypoint |
| `strands_robots/mesh/iot/kvs.py` | 167 | 210 (6.1, 6.2, 6.5, 6.6) | stream/channel names, measured pipeline, `CameraPublisher`, `clip_window` |
| `strands_robots/mesh/mesh.py` (hunk) | +8 −6 (249 lines) | — | `iot=` on `mount`, one `attach` call |
| `strands_robots/mesh/transports/iot.py` (hunk) | +7 (222 lines) | — | `self.rpc` mode, the MQTT request/response send step, `reply_ok` check |
| `strands_robots/mesh/transports/iot_link.py` (hunk) | +11 −4 (221 lines) | — | `Link.publish(**props)`; MQTT5 `response_topic` + `correlation_data` |
| **total new code** | **1,320 + 26 hunk lines** | 1,360 (MAPPING) / 2,010 incl. tests (RESEARCH §0) | every row of MAPPING implemented or refused as written |

Mesh today: 2,226 lines (`mesh/*.py` + `transports/*.py`). With the IoT package: 3,572 (+60 %). `import strands_robots.mesh`
pulls in `mesh.iot.direct` (40 lines) and `mesh.iot.options` (74), both stdlib; `boto3`, `awsiot`, `cryptography` and GStreamer
are imported inside the functions that use them. import-linter: 8 contracts kept.

## Tests

| file | lines | runs on |
|---|---|---|
| `tests/mesh/iot/conftest.py` | 31 | moto (`iot`, `iot-data`, `sts`), placeholder account |
| `test_shadow.py` | 119 | FakeIotBroker |
| `test_mount_iot.py` | 75 | FakeIotBroker through `Mesh.mount(iot=…)` |
| `test_rules.py` | 71 | pure + moto + FakeIotBroker |
| `test_provision.py` | 78 | pure + moto |
| `test_claim_registry.py` | 65 | pure (hook source executed against real certificates) + moto |
| `test_direct.py` | 91 | FakeIotBroker: the whole mesh in `rpc=mqtt` mode; foreign reply address never answered |
| `test_lifecycle.py` | 46 | pure + FakeIotBroker |
| `test_kvs_greengrass.py` | 110 | pure + moto (`kinesisvideo` streams; signaling channels skip) + `mem://` |
| `test_live.py` | 215 | **real account** behind `STRANDS_IOT_LIVE=1`; creates and deletes its own things/policies/group |
| **total** | **901** | 31 passed, 2 skipped offline (15–38 s depending on host load); live 1 passed (26 s) |

## Live numbers (run 11, us-west-2, from one laptop through the Ashburn egress, 2026-10-17 06:47Z)

| what | measured |
|---|---|
| `setup_tenant` + `register` (group, 2 policies, type, thing, attributes, CSR → cert, 2 attaches) | 1.24 s |
| `Mesh("iot://…")` + `mount(iot=IotOptions(shadow_hz=1))` | 2.23 s |
| manifest shadow reported document | 5,260 B |
| state shadow updates in 3 s at `shadow_hz=1` | 3 |
| `update_thing_shadow(desired shoulder_pan.pos=0.3)` → arm moving (through `confirm_act`) | 193 ms (190 in run 4) |
| estop as a field / out-of-range target | both refused verbatim in `reported.refused`, cleared from `desired`, arm not stopped |
| operator `invoke(describe)` as Direct Message (IAM), avg of 10 | 190 ms |
| certificate-only controller peer `invoke(describe)` as MQTT request/response, avg of 10 | 198 ms |
| a certificate-only peer WITHOUT its own thing / WITHOUT the controller add-on | CONNACK refused / cannot send rpcs (the fleet policy's least privilege, observed as failures on the way) |
| manifest shadow after `unmount` | deleted |
| after cleanup | things [] policies [] groups [] (thing type deprecated; swept by the next run after AWS's 5-min window) |

Lesson recorded in code: awscrt on macOS refuses an EC/PKCS#8 private key (`AWS_IO_FILE_VALIDATION_FAILURE`);
`provision.csr` makes RSA-2048 in traditional PEM, which every awscrt TLS backend loads.
