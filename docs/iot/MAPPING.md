# MAPPING v1 — strands-robots v1 concept → AWS IoT primitive

Status: **v1 ready** (2026-10-17; rows 2.4, 4.1, 4.3 revised after EXPERIMENTS §4–5: lifecycle ≠ liveness). Every row names the v1 concept as the reference spells it
(robots-v1-2 @ bcfb45e), the IoT primitive, the module in `strands_robots/mesh/iot/` that owns
it, and the test that proves it. Refusals are rows too, with the reason. Sizes are the §0 estimate; SIZE.md
reports the actual lines. Quotas and prices: RESEARCH.md §0.

## Package layout (proposal)

```
strands_robots/mesh/iot/
  __init__.py      # exports: ShadowMirror, provision, setup_tenant, RULES, apply_rules, recipe, CameraPublisher, list_things
  registry.py      # §7  read-only ListThings + connectivity for the dashboard's grey cards (v0.5 registry.py, cut to 40 lines)
  shadow.py        # §1  reported state at a bounded rate, control delta → confirm_act, manifest shadow
  rules.py         # §2  rule definitions as data, apply(), basic_ingest_topic()
  provision.py     # §3  DEVICE_POLICY / OPERATOR_POLICY / TEMPLATE as data, setup_tenant, register, provision, rotate, revoke
  lifecycle.py     # §4  $aws/events/presence → OnAlive
  greengrass.py    # §5  recipe(), deploy(), GreengrassLink (IPC-backed Link)
  kvs.py           # §6  ensure_stream/ensure_channel, CameraPublisher (GStreamer kvssink), hls_url
strands_robots/mesh/transports/iot_link.py   # §4 request(), will=, retain=; FakeIotBroker learns the same
strands_robots/mesh/transports/iot.py        # §4 picks DM for estop, request/response for other rpcs
tests/mesh/iot/                              # unit (moto + FakeIotBroker); live behind STRANDS_IOT_LIVE=1
docs/iot.md                                  # user doc
```

`Mesh("iot://us-west-2?cert=&key=&client=<device_id>")` is unchanged. New knobs on `Mesh.mount` are one
mapping: `iot={"shadows": True, "shadow_hz": 0.5, "telemetry_hz": 0, "cameras": "kvs"|"off"}`; nothing
happens unless the transport is IoT, and nothing in `Robot`, `Driver`, `Simulation` or the safety chain
changes.

## The table

| # | v1 concept (where in the reference) | IoT primitive | module | test idea | size |
|---|---|---|---|---|---|
| 1.1 | `record()` → id, device_type, tenant, labels, manifest 4,815 B (`mesh/record.py`) | named shadow `manifest`, `reported` = the record, written at `mount`, deleted at `unmount`; also retained MQTT `strands/<t>/<id>/manifest` with a Will that clears it | shadow.py | moto `get_thing_shadow(manifest)` equals `record()`; FakeIotBroker: retained delivered to a late subscriber; Will clears on close | 60 |
| 1.2 | the `state` stream `robot.subscribe("state", hz)` (`mesh/stream.py`, 544 B, 10–500 Hz) | named shadow `state`, `reported` at `shadow_hz` ≤ 1 (default 0.5); the live stream stays on MQTT `…/state` | shadow.py | moto: after 2 s at 0.5 Hz exactly 1 update; `shadow_hz=30` raises `ValueError` ("over the 20 req/s per-thing quota, use the state stream") | 50 |
| 1.3 | `robot.confirm_act(action)` (`mesh/serve.py::_act`) | named shadow `control`: `desired.<control>.pos` → `/update/delta` → `confirm_act(delta)`; result or refusal (`RequestRefused` message verbatim) written to `reported`; refused keys cleared from `desired` | shadow.py | FakeIotBroker publishes a synthetic delta; assert `confirm_act` called once with the delta; out-of-range target → `reported.refused["elbow_flex.pos"] == "so101: elbow_flex=3.1000 rad is outside its range […]"` and `desired` cleared | 90 |
| 1.4 | watchdog / `on_disconnect` (`mesh/watchdog.py`) | delta age check: `metadata.desired.<k>.timestamp` older than `watchdog_ms` → dropped as stale, logged; the mesh watchdog is unchanged | shadow.py | a delta with a 10 s old timestamp is not applied | 20 |
| 1.5 | `estop` / `Lockout` (`mesh/authority.py`, `serve.py::estop`) | **refused as a shadow write** (eventually consistent ≠ a latch); the lockout is *reported* into `state` (`lockout: {engaged, epoch, by}`) and published retained on `strands/<t>/<id>/lockout` — retained **state the device itself clears**, never a retained command (v0.5 had to ship `clear_retained_safety_messages` after retaining stops) | shadow.py | `desired.estop` in `control` is ignored and reported back as `refused.estop = "estop is an rpc, not a shadow field"` | 15 |
| 1.6 | `on_disconnect` driven from the cloud | **refused**: a persisted `desired` outlives its operator; the robot's own watchdog decides | — | doc only | 0 |
| 2.1 | tenant `@estop` key (`mesh/broadcast.py`) | rule `strands-estop-<t>`: `SELECT *, topic(2) AS tenant FROM 'strands/+/@estop'` → republish `strands/<t>/@estop/latest`, SNS (optional) | rules.py | moto `create_topic_rule`/`get_topic_rule` round trip of the exact SQL; live: publish one estop, read `@estop/latest` | 30 |
| 2.2 | events `fault`, `estop`, `run.started`, `run.stopped` (`mesh/stream.py::EVENTS`) | rule `strands-events-<t>`: `SELECT *, topic(3) AS device_id, topic(5) AS event FROM 'strands/+/+/event/+'` → Timestream table `strands_events` and/or S3 `recordings/<tenant>/<device>/<run_id>.json` | rules.py | moto round trip; `run.started` payload carries the run id (part 9) | 30 |
| 2.3 | `fault` with severity | rule `strands-fault-<t>` `… WHERE severity = 'high'` → SNS/Lambda (user-supplied ARN placeholder) | rules.py | SQL string test | 10 |
| 2.4 | `@presence/*` heartbeats through rules | **refused**: 3,600 triggers/robot/h; the heartbeat stays the liveness signal (4.3). Lifecycle events feed ONE audit rule (`lifecycle.clear_presence_rule`): disconnects → `strands/<t>/@audit/disconnected` | lifecycle.py | SQL string test | 10 |
| 2.5 | `state` through rules into storage | **refused by default** (108,000 triggers/robot/h at 30 Hz; recording is part 9's, local first). Opt-in `telemetry_hz` ≤ 1 via **Basic Ingest** `$aws/rules/strands-telemetry-<t>/strands/<t>/<id>/state` (no messaging charge, no subscriber sees it) | rules.py | `basic_ingest_topic()` string; FakeIotBroker counts publishes at `telemetry_hz=1` | 30 |
| 2.6 | the rule's IAM role | **not created by the library**: `apply_rules(iot, tenant, role_arn)` takes it; docs show the 2 statements it needs | rules.py | doc | 0 |
| 3.1 | `device_id` (`wire.check_id`, ≤ 128 chars) | thing name = MQTT client id = **certificate CN from a local CSR** (v0.5 fact (a)); MQTT statements pin `${iot:Connection.Thing.ThingName}` (proven live, EXPERIMENT 1: SUBACK/PUBACK 135 outside own scope), the optional HTTPS `SendDirectMessage` statement pins `${iot:Certificate.Subject.CommonName}` | provision.py | moto: `create_thing` + `create_certificate_from_csr`; CN == thing asserted; policy JSON ≤ 2,048 B asserted (v0.5 measured 2,227 with the children grant → split) | 40 |
| 3.2 | `device_type` (`so101`, `microduck`) | thing type (1 per thing) | provision.py | moto `create_thing_type` idempotent (409 → ok) | 10 |
| 3.3 | `tenant` | thing group `strands-<tenant>` + thing attribute `tenant`; the policy's topic resource is `strands/${iot:Connection.Thing.Attributes[tenant]}/${iot:Connection.Thing.ThingName}/*` + `@presence/${ThingName}` + `@reply/*` subscribe | provision.py | moto group + membership; policy text contains no literal tenant | 30 |
| 3.4 | `labels` | thing attributes (≤ 50); optional dynamic group per label for deployments (5.5) | provision.py | moto attributes round trip | 10 |
| 3.5 | `Mesh.mount(robot, iot=…)` on a fresh robot (no cert yet) | fleet provisioning by claim: `IotIdentityClient.publish_create_certificate_from_csr` (key + CSR `CN=<device_id>` generated on the device) → `publish_register_thing(template, token, {device_id, device_type, tenant})`; template creates thing/type/group/attributes/policy attach; **pre-provisioning hook** (v0.5 fact (b)): accept only `CN == ThingName` (or `AWS IoT Certificate`) and refuse an existing thing name — the hook Lambda source ships as data (≈ 60 lines, seeded from v0.5's 131), its role is user-supplied; files land in `~/.strands/iot/<device_id>/{cert.pem,key.pem}` | provision.py | unit: CSR CN, template body ≤ 10 KB, hook accepts/refuses the 3 CN cases; live only for the MQTT flow (moto lacks `create_provisioning_template`, per IMPLEMENT) | 110 |
| 3.6 | operators (`Mesh("iot://us-west-2")`, SigV4) | IAM policy `OPERATOR_POLICY` (v0.5 `OPERATOR_DIRECT_POLICY` concept): Connect as `strands-op-*`, `iot:SendDirectMessage` on `connection/*`, Publish `rpc/*`, `@estop`, `$aws/things/*/shadow/name/control/update`, Subscribe/Receive `state`, `event/*`, `lockout`, `manifest`, `@reply/${iot:ClientId}`, `@estop`; **no** Publish on `@presence/*`, `manifest`, `lockout`, `state` | provision.py | policy text test; live: an operator publish on `@presence/x` is refused (EXPERIMENT 1 showed the device-side analogue: PUBACK 135, link stays up) | 20 |
| 3.7 | rotation / revoke | `rotate(iot, device_id)`: new cert + attach, device swaps and reconnects, old → INACTIVE → delete; `revoke` = INACTIVE + detach (the device's next connect fails, its watchdog applies `on_disconnect` on the controller side) | provision.py | moto cert status transitions | 30 |
| 3.8 | `Authority` / Ed25519 resume (`mesh/authority.py`) | **unchanged**; refused: cert CN as resume authority (a stolen device cert would approve its own resume) | — | existing tests | 0 |
| 4.1 | `Mesh.invoke` → `transport.query` (`transports/iot.py`) | **MEASURED (EXPERIMENTS §4): DM 166 / MQTT5 RR 170 / RR QoS 1 204 ms p50 — not a latency choice.** Verdict implemented: rpc path is a per-LINK mode (`direct.rpc_mode`): Direct Message for IAM links (the 404 in one HTTPS round trip, 0.08 s), MQTT5 request/response for certificate links (no IAM, works on a local broker; an offline device is silence → presence says why), `iot://…?rpc=` overrides. Reply address + id ride in the body (fact d) AND as `response_topic`/`correlation_data`; the device answers only on `strands/<t>/@reply/<caller>` | direct.py, iot.py, iot_link.py | `test_direct.py`: full mesh on `iot://fake?rpc=mqtt`; foreign reply address never answered; live: DM describe 190 ms vs RR 198 ms avg10 | 40 + 18 |
| 4.2 | `estop` rpc | keep `SendDirectMessage(confirmation=True)` (404 = "offline, nothing queued" is the answer we want) + tenant `@estop` + retained `lockout` (1.5) | iot.py | existing `test_mesh_iot.py` 404 test stays | 0 |
| 4.3 | presence heartbeat 1 Hz (`iot.py::liveliness`) | **KEPT as the liveness signal (EXPERIMENTS §5): IoT Core answers `server_keep_alive_sec = 1200`, so a frozen or partitioned robot stays "connected" up to 30 min; the heartbeat finds it gone in 3.0–3.1 s.** Lifecycle events (`connected` 21–27 ms, `disconnected` 113–170 ms after a socket close) are an AUDIT trail: `lifecycle.watch`/`Audit` record who connected and dropped and why | lifecycle.py | `test_lifecycle.py`: parse, tenant filter, audit memory, the rule | 90 |
| 4.4 | `Directory._fetch` manifest rpc per appearance (`mesh/presence.py`) | read the retained `manifest` (delivered on subscribe) → 0 rpcs; fallback to the rpc if absent | presence.py (small edit) | late joiner gets the manifest without a query (`query` count = 0) | 20 |
| 4.5 | N dashboards on `state` | shared subscription `$share/strands-<t>-dash/strands/<t>/+/state` for replicas of one consumer (opt-in) | iot_link.py | filter string test | 10 |
| 4.6 | `observe(images=True)` over IoT | **refused on MQTT**: 128 KB payload cap; cameras → 6.x | — | `invoke(id,"observe",images=True)` on IoT raises `ValueError` naming the cap | 5 |
| 5.1 | a robot host | Greengrass core device `strands-<t>-<host>` (nucleus, Java 8+ Linux; lite for small devices) | greengrass.py | doc + `recipe()` size ≤ 8 KB | 10 |
| 5.2 | the process that mounts robots | component `com.strands.robots` (recipe + wheel artifact; config `{tenant, robots[], state_hz, shadow_hz}`; `Run: python -m strands_robots.mesh.iot.greengrass`) | greengrass.py | recipe JSON schema test; config → `Mesh.mount` calls | 60 |
| 5.3 | `Mesh("zenoh://")` on the LAN | **unchanged**; refused: Moquette/EMQX as the LAN mesh (no liveliness/queryables, MQTT 3.1.1) | — | doc | 0 |
| 5.4 | cloud down | `GreengrassLink` (IPC `PublishToIoTCore`/`SubscribeToIoTCore`, same `Link` protocol): `state`/`event` are spooled by the nucleus; `rpc`, `@estop` and `lockout` are **not** spooled (dropped with `Offline` while disconnected) | greengrass.py | fake IPC: while "offline", a publish on `@estop` raises `Offline`, on `state` is queued | 80 |
| 5.5 | fleet update | `deploy(target=thing group, version)` via `greengrassv2.create_deployment`; `DeferComponentUpdate` between `run.started` and `run.stopped` | greengrass.py | moto has no greengrassv2 (**→ verify**); unit test the deployment document; live on ec2 | 30 |
| 5.6 | recording upload (part 9) | StreamManager S3 export of the episode directory (bandwidth-limited); metadata via 2.2 | greengrass.py | live only | 20 |
| 5.7 | servo loop in a Greengrass Lambda | **refused**: container + cgroups v1, no real time | — | doc | 0 |
| 6.1 | `manifest.sensors` of kind camera | one KVS stream + one signaling channel `strands-<t>-<id>-<camera>`; names + HLS URL under `manifest.cameras[]` in the `manifest` shadow | kvs.py | moto `kinesisvideo` create/describe (**→ verify**); name ≤ 256, `[a-zA-Z0-9_.-]` | 40 |
| 6.2 | camera frames (`observe(images=True)`, MuJoCo render or real) | `CameraPublisher`: GStreamer `appsrc ! videoconvert ! x264enc tune=zerolatency ! h264parse ! kvssink`, fed at the camera's fps on one thread per camera (same "sample once, many viewers" as `stream.publish`) | kvs.py | pipeline string test; `ImportError` with `hint("iot-edge")` when `gi`/`kvssink` are absent | 100 |
| 6.3 | live view in the dashboard (part 10) | KVS WebRTC: robot = master (1 per channel), viewers ≤ 10; JS SDK in the dashboard; Python master = `aiortc` + signaling websocket (no AWS Python SDK) | kvs.py | live only (EDGE); **→ measure** glass-to-glass | 60 |
| 6.4 | `Mesh.connect(id, images=True)` for a remote policy | **refused** over KVS/WebRTC (codec stream ≠ tensors, +100–300 ms); remote policies use zenoh (measured: 3 MuJoCo cams at 20.1 Hz over Zenoh in the real-arm episode) | — | doc | 0 |
| 6.5 | PutMedia from pure Python | **refused**: boto3 `kinesisvideo` has no PutMedia (32 ops, introspected); a hand-rolled SigV4 chunked POST has no ACK/backpressure handling | — | doc | 0 |
| 6.6 | recording (part 9) | video copy only; `GetClip` by the run's start/stop timestamps linked through 2.2 | kvs.py | `clip_window(run)` unit test | 10 |
| 7.1 | provisioned-but-silent devices (dashboard grey cards, v0.5 `registry.py`) | `iot:ListThings` in group `strands-<t>` (+ fleet-index connectivity when on), every failure a status word, off by `STRANDS_IOT_REGISTRY=off` | registry.py | moto `list_things_in_thing_group`; failure → `status="unavailable"` not an exception | 40 |
| D | `docs/iot.md`, changelog.d fragment, size report | — | docs | — | 150 |

Totals from the size column: shadow 235, rules 100, provision 250, link/lifecycle 175, greengrass 200,
kvs 210, registry 40, docs 150 → **1,360 code lines** (+ tests ≈ 640, see RESEARCH §0; §0's 2,010 estimate holds within 4 %).

## v0.5 → v1: per file, what comes over and what is dropped

Source: v0.5 `strands_robots/mesh/iot/` on `main` (4,861 lines) + `mesh/transport/iot_transport.py` (1,717), read 2026-10-17.
"Comes over" means the *concept*, re-expressed on v1's names (`record()`, `wire.KEYS`, `confirm_act`, `Lockout`,
`Transport`) inside ≤ 250-line files; no v0.5 code is copied.

| v0.5 file (lines) | comes over to v1 as | dropped, and why |
|---|---|---|
| `provision.py` (2,341) | `provision.py`: `register()` (thing/type/group/attributes/cert/policy, idempotent, 8 calls = 1.1 s measured by EXPERIMENT 1), local CSR with `CN=<device_id>` (fact (a)), `rotate()` (= `reprovision_thing`), `revoke()`/`teardown`, policy size check against the 2,048-B cap, thing-name validation via `wire.check_id` | Amazon Root CA1 pinning + download + runbook (300 lines: awscrt ships the trust store; a pinned file breaks on the rotation it defends); child-peer key space `__*` grant (v1 has no child peers — a Simulation mounts N devices with N ids); `withdraw_fleet_stop_grant` / `strands-robot-no-estop` posture (v1's `@estop` publish right is an *operator* IAM right, devices never publish it; a device's own stop is its rpc); `clear_retained_safety_messages` (v1 never retains a command) |
| `bootstrap.py` (1,417) | `provision.py`: fleet provisioning template (`TEMPLATE` as data) + pre-provisioning hook source as data (fact (b)); `rules.py`: rules as data applied idempotently with a user-supplied `role_arn` | DynamoDB safety table + KMS/PITR, the e-stop fan-out Lambda, IAM role/log-group creation (v1: the tenant `@estop` key latches every mesh with no Lambda; events → Timestream/S3 by rule action; the library never creates IAM — the user passes the ARN); `OPERATOR_DIRECT_POLICY` creation (shipped as a document in `provision.OPERATOR_POLICY`, not created) |
| `shadow.py` (245) | `shadow.py`: `ShadowMirror` name and the named-shadow-per-family idea → `manifest` (once), `state` (≤ 1 Hz), `control` (`desired` → `confirm_act`, the half v0.5 lacked); topic builders → `$aws/things/<t>/shadow/name/<n>/…` via `awsiot.iotshadow` (no hand-built topics, no 40-line segment validator: `wire.check_id` covers it) | mirroring the *presence heartbeat* into a shadow (v1 presence = lifecycle events + retained manifest; a heartbeat in a shadow is 3,600 ops/robot/h for nothing) |
| `registry.py` (222) | `registry.py` (≈ 40): `list_things(tenant)` with connectivity, status words, off switch | `MAX_THINGS` paging policy and the dashboard-specific row shape (part 10 shapes its own cards from `Device`) |
| `camera_offload.py` (407) | `kvs.py`: the refusal (frames never on MQTT: 128 KB cap, cost) and the "publish a small pointer, fetch the heavy thing elsewhere" shape → `manifest.cameras[]` carries the KVS stream/channel names | S3-per-frame + presigned URLs (1 fps at $0.018/h vs KVS 30 fps at $0.0057/h; a dashboard wants video, not JPEG polling); the `INLINE_CAMERA_ENV` opt-in (an operator on IoT cannot opt into a 128 KB cap) |
| `cli.py` (144) | nothing in the mesh: `provision/rotate/revoke/setup_tenant/apply_rules` are functions; a CLI verb belongs to part 10 / the agent tool (`use_mesh` already exposes the mesh) | all six verbs as a CLI |
| `__init__.py` (85) | `mesh/iot/__init__.py` (≈ 15): lazy exports, boto3/awsiot imported where used (a review point on the mesh kept) | eager imports |
| `transport/iot_transport.py` (1,717) | already ported: `transports/iot_link.py` + `iot.py` (429) = Direct Messaging with 200/404/504 semantics, `FakeIotBroker`, holder rule, SUBACK refusal → `PermissionError`; **new in v1 from v0.5's `_mqtt5_reply_properties`**: request/response via `response_topic`/`correlation_data` (4.1); QoS per `wire.KEYS` slot (today `AwsLink.publish` is QoS 0 for every slot; proposal: presence/lockout/rpc QoS 1, state QoS 0 — **→ measure** the QoS 1 cost on the 203 ms rpc) | zenoh-shaped `_MqttSample/_MqttPayload/_MqttSubHandle` wrappers (one `Transport` protocol in v1); env-var switches (URL query parameters instead); x509 HTTPS direct client on :8443 (optional, §3; devices do not initiate rpcs in the reference); the 20-entry `_TOPIC_POLICY` and `_NEVER_BRIDGE_PREFIXES` (7 keys in `wire.KEYS`; images refused by `observe` arg check) |
| `tests_integ/test_iot_direct_messaging_e2e.py` | `tests/mesh/aws_iot_round_trip.py` (already in the reference) + EXPERIMENT's `scripts/iot/exp*.py` with numbers in EXPERIMENTS.md | — |

v0.5 facts kept verbatim as v1 rules: (a) CN identity for the HTTPS path; (b) the hook binds CN to the thing it
creates; (c) fan-out stays pub/sub (presence, state, events, `@estop`), conversations go direct (rpcs); (d) the
reply address and request id ride in the message (MQTT5 `response_topic` + `correlation_data` in v1) and the
device checks the reply topic is `strands/<tenant>/@reply/<that caller>` before answering.

## Refusals, in one place

| refused | because |
|---|---|
| estop or resume as a shadow `desired` field | the shadow is eventually consistent and persists; a latch is immediate and must never replay |
| state into a shadow above 1 Hz | 20 req/s per-thing quota (L-EFF4BE08) and 4.5 × the cost of the stream; "last known" ≠ "live" |
| `on_disconnect` decided by the cloud | a `desired` outlives its operator; the robot's watchdog on the *controller's* presence is the v1 rule |
| presence or state through the rules engine (default) | 3,600–108,000 triggers/robot/h for data that lifecycle events / part 9 already give |
| the library creating IAM roles | out of scope; the user supplies `role_arn` |
| the cert's CN as resume authority | a stolen device certificate would approve its own resume |
| images over MQTT | 128 KB payload cap (L-13573643, fixed) |
| Moquette/EMQX as the LAN mesh | MQTT 3.1.1, no liveliness, no queryables; zenoh measured at 1 ms rpc / 500 Hz state |
| the servo loop in a Greengrass Lambda | containerised, cgroups v1 only, no real-time |
| spooling rpcs/estops while the core is offline | a queued e-stop that arrives later is the one failure the reference designs out |
| remote policies over KVS/WebRTC | codec stream, 100–300 ms, not tensors |
| hand-rolled PutMedia in Python | no official Python producer; no ACK/backpressure |

## What the owner's six features map to, in one line each

1. Shadows → `manifest` (once), `state` (≤ 1 Hz, last known), `control` (desired → `confirm_act`, never the servo).
2. Rules → one rule per event kind per tenant (estop republish, events → Timestream/S3, fault → SNS); state only by opt-in Basic Ingest.
3. X.509 → thing = device_id, type = device_type, group + attribute = tenant, one fleet policy with variables, claim provisioning on first `mount`, IAM for operators.
4. Direct messaging → the rpc path is per link: Direct Message for IAM operators (the 404 is the point), MQTT5 request/response for certificate-only links (measured within 4 ms of each other); the 3 s heartbeat stays the liveness signal (server keep-alive 1,200 s makes lifecycle events a 30-minute partition detector); lifecycle events are the audit trail.
5. Greengrass → `com.strands.robots` component, IPC link that spools telemetry but never commands, thing-group deployments as the fleet update.
6. KVS → one stream + signaling channel per camera, GStreamer `kvssink` from Python, WebRTC for the dashboard; not for policies.
