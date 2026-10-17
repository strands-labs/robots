# IoT-native strands-robots v1: research

Written 2026-10-17. Every number below is either read from the reference tree
(robots-v1-2 @ bcfb45e), from the AWS account (Service Quotas default values and the
Pricing API for us-west-2, both read on 2026-10-17), from the installed SDKs
(awsiotsdk 1.31.0, awscrt 0.36.1, boto3 1.43.114) or from an AWS doc page fetched on 2026-10-17 (URL given).
Numbers still to be measured are marked **→ measure**; EXPERIMENTS.md carries the ones that were.

## What the reference's IoT transport already does (baseline)

Read from the reference's `strands_robots/mesh/transports/{iot.py,iot_link.py}`:

- One MQTT5 connection per `Mesh` (mTLS with `?cert=&key=`, else SigV4 WebSocket with the default
  credential chain), `clean_session=True`, `keep_alive_secs=30`, QoS 0 publishes, QoS 1 subscriptions.
- An rpc is `iot-data:SendDirectMessage(clientId, topic, payload, confirmation=True, timeout≤10)` to the
  client that holds the device; 404 → `Offline` (nothing queued), reply on `strands/<tenant>/@reply/<caller>`.
- Presence = a heartbeat every `presence_ttl/3` (1 s) on `strands/<tenant>/@presence/<device_id>`, expiry
  after `presence_ttl` (3 s); the first client heard on a key holds it.
- Manifest is fetched by rpc on appearance (`presence.py`), never retained: a dashboard that joins late must
  wait for the next heartbeat + a 2 s fetch window.
- Measured (STATUS / #4749 evidence): discover 0.83 s, act p50 203 ms, e-stop to an offline device → 404.
- Wire facts measured in this tree on 2026-10-17 (`so101` sim, `record()` as the mesh sends it):
  manifest record **4,815 B** compact JSON (sensors 18, controls 6, rpcs 7, events 5); one state envelope
  **544 B**; the manifest's default state rate is **10 Hz**.

## Quotas and prices that bite a robot (us-west-2, read from the account 2026-10-17)

Service Quotas, `iotcore` default values (238 quotas; `L-…` codes are the quota ids):

| quota | value | adjustable | why it matters |
|---|---|---|---|
| MQTT payload size (L-13573643) | 128 KB | no | an image observation does not fit; `observe(images=True)` stays off IoT or goes to KVS |
| Maximum MQTT5 packet size (L-78413E12) | 146 KB | no | same |
| Publish requests per second per connection (L-083F3861) | **100** | **no** | one Mesh = one connection; 6 robots at 30 Hz state = 180 > 100 → one connection per device, or ≤ 3 devices per connection at 30 Hz |
| Throughput per second per connection (L-D9C66F9D) | 512 KB/s | no | 544 B × 100 msg/s = 54 KB/s, fine |
| Requests per second per thing (L-EFF4BE08) | **20** | yes | **shadow/registry ops per thing; a 30 Hz reported shadow is refused by quota before it is refused by price** |
| Device Shadow API requests/second per account (L-6CC6395A) | 4,000 | yes | 200 robots × 20 Hz saturates it |
| Maximum size of a JSON state document (L-A295A064) | **8,192 B** | yes | the manifest record is 4,815 B → fits in one named shadow with 3.3 KB to spare; `state` at 544 B fits |
| Maximum depth of JSON device state documents (L-2A7BC612) | 8 | no | the manifest nests 4 deep (`controls[].interlocks[]`), fine |
| Maximum shadow name size (L-724130CC) | 64 B | no | `manifest`, `state`, `control` as shadow names |
| Subscriptions per connection (L-45957C13) | 50 | yes | serve() = 8 rpc keys + manifest + presence + estop per device → 11; 4 devices per connection before the cap |
| Maximum subscriptions per subscribe request (L-A6574E9E) | 8 | no | `serve()` subscribes one by one today, ok |
| Maximum number of slashes in topic (L-AD5A8D4F) | 7 | no | `strands/<tenant>/<device>/rpc/<name>` = 4, fine; shadow topics `$aws/things/<t>/shadow/name/<n>/update/delta` = 7, exactly at the cap |
| Topic size (L-ECD9BEEF) | 256 B | no | tenant + device_id + name ≤ ~200 chars |
| Client ID size (L-E2D9C89C) | 128 B | no | thing name = device_id ≤ 128 |
| Maximum thing name size (L-83BC2FA9) | 128 B | no | same |
| Connect requests per second per client ID (L-033DE216) | 1 | no | reconnect storms self-throttle |
| Connection inactivity / keep-alive max (L-E77027EF) | 1,200 s | no | the reference uses 30 s |
| WebSocket connection duration (L-3F401592) | 86,400 s | no | an operator Mesh over SigV4 reconnects once a day → presence hiccup, must be handled |
| Persistent session expiry (L-3BB6AFBC) | 3,600 s | yes | `clean_session=True` today: nothing is queued, which is what the e-stop semantics want |
| Maximum Message Expiry Interval (L-BD70FE3A) | 604,800 s | no | MQTT5 message expiry works for "stale act" protection |
| Maximum MQTT5 Correlation Data (L-7E9783FF) | 8 KB | no | a 32-hex request id is 32 B |
| Maximum User Properties total size (L-7201B389) | 8 KB | no | room for `peer`, `seq`, `v` out of the payload |
| Retained messages per account (L-57BADEF0) | 500,000 | yes | one retained manifest + one retained estop per device/tenant: fine |
| Retained message inbound publish per topic (L-771FD74A) | **1 /s** | **no** | a retained `state` is impossible above 1 Hz; retained = manifest + estop only |
| Retained message inbound publish per account (L-BAFC6221) | 500 /s | yes | |
| SendDirectMessage API TPS (L-889910A2) | 20,000 | yes | account-wide; 100 robots × 30 Hz act = 3,000, fine |
| Shared Subscription groups per account (L-850876C0) / per group (L-EB813241) | 100 / 100 | no | N operators sharing one rpc subscription: ≤ 100 groups per account |
| Queued messages per second per account (L-35DA9240) | 500 | yes | persistent sessions only; not used |
| Rules per account (L-954FA751) / actions per rule (L-51309716) / rule evaluations per s (L-5E9B88DA) | 1,000 / 10 / 20,000 | yes/no/yes | one rule per event kind per tenant, not per device |
| Rule payload size (L-4FB8C075) | 128 KB | no | |
| Fleet provisioning templates per customer (L-345B62A1) / template size (L-3B9C1B78) | 256 / 10 KB | no | one template per tenant |
| Fleet Provisioning RegisterThing MQTT TPS (L-B2BBBD8C) | 10 /s | yes | 600 robots/min bootstrapped |
| Thing attributes (with a type) (L-3964F400) / size (L-5104A2FD) | 50 / 47 KB | yes | `device_type`, `tenant`, labels as attributes; the manifest does **not** go in attributes (shadow instead) |
| Maximum policies per certificate (L-BC2638B3) / policy size (L-D9B73671) | 10 / 2,048 B | no | the per-device least-privilege policy must fit 2 KB → policy variables, not enumerated topics |
| Thing groups a thing can belong to (L-C12BBB7E) | 10 | no | tenant group + a fleet group + labels as groups ≤ 10 |

Greengrass (`greengrass`, 17 quotas): component recipe 8 KB, deployment document 7 KB per thing /
31 KB per thing group (6 MB with large-configuration support), total artifact size 2 GB, 5,000 components,
API 30 req/s. Kinesis Video (`kinesisvideo`, 98 quotas): PutMedia 100 Mbit/s and **1 concurrent
connection per stream** (one producer per camera stream), fragment 1–10 s / ≤ 50 MB, 3 tracks,
GetMedia 200 Mbit/s and 3 concurrent connections per stream, 20,000 streams and 20,000 signaling
channels per account, ConnectAsMaster **1 per channel**, ConnectAsViewer **10 per channel** (adjustable),
connection duration 3,600 s (reconnect every hour), TURN session 5 Mbit/s and 300 s expiry.

Prices (Pricing API, `AWSIoT` / `AWSGreengrass` / `AmazonKinesisVideo`, location "US West (Oregon)",
read 2026-10-17; messaging is metered in 5 KB increments, every delivery to a subscriber is a message):

| item | price |
|---|---|
| MQTT messages | $1.00 / M (first 1 B/month), $0.80 / M next 4 B, $0.70 / M after |
| connection minutes | $0.08 / M minutes |
| Device Shadow + Registry operations | $1.25 / M |
| Direct Messages | $1.80 / M; **with confirmation $2.50 / M** (first 1 B) |
| Rules triggered / actions executed | $0.15 / M each |
| Greengrass | $0.16 per active core device per month |
| KVS ingest / read / storage | $0.0085 / GB in, $0.0085 / GB out ($0.0119 for HLS/DASH), $0.023 / GB-month (warm tier $0.0125) |
| KVS WebRTC | $0.03 per active signaling channel-month, $2.25 / M signaling messages, TURN $0.12 / 1,000 minutes |

**Cost per robot-hour at 30 Hz** (108,000 samples/h; 544 B state = 1 metered message):

| path | $/robot/h | note |
|---|---|---|
| state 30 Hz on MQTT, 0 subscribers | 0.108 | publish only |
| state 30 Hz on MQTT, N subscribers | 0.108 × (1 + N) | each delivery is a message |
| state 30 Hz into the `state` shadow | 0.135 shadow ops + 0.108 publish + 0.108 per `/update/accepted` listener | **and over the 20 req/s per-thing quota → refused** |
| state 1 Hz into the `state` shadow (reported, bounded) | 0.0045 + 0.0036 | the recommended shadow rate |
| act 30 Hz via SendDirectMessage(confirmation) + MQTT reply | 0.270 + 0.108 = 0.378 | today's path, tele-op |
| act 30 Hz via MQTT5 request/response | 0.108 + 0.108 = 0.216 | 43 % cheaper, **→ measure** RTT vs 203 ms |
| act 1 Hz (an agent) either way | ≤ 0.013 | cost is irrelevant below 1 Hz |
| events (estop, run.*, fault) through rules | < 0.001 | tens per hour |
| presence heartbeats at 1 Hz (today) | 0.0036 + 0.0036 per watcher | lifecycle events would make it 0 |
| one camera 640×480@30 fps H.264 ≈ 1.5 Mbit/s → KVS | 0.0057 ingest + 0.0155/GB-month stored | 0.675 GB/h |
| one WebRTC viewer, direct (no TURN) | ≈ 0 | signaling only; TURN relay adds 0.0072/h |

## §1 Device shadows

**Primitives.** Named shadows per thing: `$aws/things/<thing>/shadow/name/<shadow>/{get,update,delete}` with
`/accepted`, `/rejected`, `/update/delta`, `/update/documents` (https://docs.aws.amazon.com/iot/latest/developerguide/device-shadow-mqtt.html,
fetched 2026-10-17). Document = `{"state": {"reported": …, "desired": …}, "metadata", "version", "timestamp"}`,
≤ 8,192 B (L-A295A064, adjustable), depth ≤ 8 (L-2A7BC612). Delta = `desired − reported`, published to the
device as `/update/delta` with only the differing keys. Classic (unnamed) shadow exists too; we use named only.

**SDK paths.** Device side: `awsiot.iotshadow.IotShadowClient` on the existing MQTT connection —
`publish_update_named_shadow`, `subscribe_to_named_shadow_delta_updated_events`,
`subscribe_to_update_named_shadow_accepted/rejected`, `publish_get_named_shadow`,
`subscribe_to_get_named_shadow_accepted` (introspected in the venv: `IotShadowClient(mqtt_connection: awscrt.mqtt.Connection | awscrt.mqtt5.Client)` takes
the reference's mqtt5 client as is). Operator/dashboard side, no
MQTT needed: `boto3.client("iot-data").get_thing_shadow(thingName, shadowName)`, `update_thing_shadow`,
`list_named_shadows_for_thing`, `delete_thing_shadow` (11 ops total on `iot-data`).

**Mapping onto v1.**

| v1 concept | shadow | rule |
|---|---|---|
| `record()` (id, type, tenant, labels, manifest 4,815 B) | named shadow **`manifest`**, `state.reported` = the record, written once at `mount`, cleared at `unmount` | a dashboard that has never met the device draws it from `get_thing_shadow`; replaces the "fetch on heartbeat" path for late joiners (today 0.83 s discover) |
| the `state` stream (544 B at 10–500 Hz) | named shadow **`state`**, `state.reported` at a **bounded rate ≤ 1 Hz** (default 0.5 Hz, `shadow_hz=`), the live stream stays on `strands/<tenant>/<id>/state` | 30 Hz into a shadow is over the 20 req/s per-thing quota and costs 4.5 × the stream; the shadow is "last known", not "live" |
| operator/agent targets | named shadow **`control`**: `state.desired = {"<control>.pos": value, …}` by an operator with IAM rights; the device receives `/update/delta` and calls **`robot.confirm_act(action)`** exactly as an rpc `act` does; it then reports what it did (or the refusal under `reported.refused` with the message verbatim) | the delta never goes to the servo: it goes into the same safety chain (`range → safe → interlocks → validators → operator`). A refusal clears `desired` for that key so the delta does not re-fire forever |
| `on_disconnect` / watchdog | **refused for shadow-driven motion**: a `desired` value persists in the cloud after the operator is gone; the device applies its watchdog to the *delta stream* (last delta older than `watchdog_ms` → `on_disconnect`), and a desired target older than `watchdog_ms` at delivery time (shadow `metadata.desired.<k>.timestamp`) is dropped as stale | the shadow's persistence is the opposite of "whoever moves it must keep talking" |
| `estop` | **not a shadow write**: estop stays an rpc + the tenant `@estop` key; the lockout (`engaged`, `epoch`, `by`) is *reported* into `state` so the dashboard sees it | a desired estop could be delivered after a resume (the shadow is eventually consistent, the latch is not) |

Limits that shape the code: one `UpdateThingShadow` per field group, not per field (20 req/s per thing is
shared by `manifest`, `state`, `control`); the `control` shadow must stay ≤ 8 KB → only `*.pos` targets
(6 controls ≈ 300 B); `version` is used for optimistic concurrency (a rejected 409 means another operator
wrote; the device re-reads). Costs: §0 table (shadow at 1 Hz ≈ $0.008/robot/h).

**Where it lives / test.** `strands_robots/mesh/iot/shadow.py` (`ShadowMirror(robot, link, device_id,
hz)`), hooked from `Mesh.mount` when the transport is IoT and `iot.shadows=True`. Tests: `moto` has
`iot-data` `get/update/delete_thing_shadow` (named shadows supported since moto 4.x — **→ verify in the
venv**) for the reported path; the delta path is tested against the `FakeIotBroker` by publishing a
synthetic `/update/delta` and asserting `confirm_act` was called with the delta and that an out-of-range
target lands in `reported.refused`. Live, **measured by EXPERIMENT 2** (`scripts/iot/exp2_shadow.py`, from one laptop with an 80 ms network
floor): manifest shadow 5,204 B accepted in 252 ms; `state.reported` at 10 Hz 100/100 accepted (p50 151 ms to
`/update/accepted`), at **30 Hz 72 of 299 rejected 429** (≈ 22.7 accepted/s: the per-shadow throttle is the 20
req/s quota); a 9,011 B document → 413; **operator `desired` over MQTT → device `/update/delta` p50 121 / p95 178
ms**, over HTTPS `update_thing_shadow` p50 312 ms; delta payload 164 B. So desired → delta → safety chain → act
is ≈ 120 ms + the reported echo, against 203 ms for the confirmed Direct Message `act` — and the shadow is a
*target* that a reconnecting device re-reads, which is exactly what a command must not be (§4).

## §2 Rules engine

**Primitives.** Topic rules: SQL over a topic filter (`SELECT … FROM 'strands/+/+/event/estop' WHERE …`),
≤ 10 actions per rule (L-51309716): `republish` (fields `roleArn, topic, qos, headers` — introspected from the boto3 model: **no `retain`**, so a
retained republish needs a Lambda action calling `iot-data:Publish(retain=True)`, or the device itself publishes
its latched estop state retained — the second is 0 extra services and is the proposal below), `sns`, `lambda`, `timestream`, `s3`,
`cloudwatchLogs`, `iotEvents`, `kinesis`, `http`. Rule evaluations 20,000/s per account, rules 1,000 per
account (not per device: one rule per event kind per tenant). Basic Ingest: a device publishes to
`$aws/rules/<rule_name>/<original/topic>`; the rule fires, **no messaging charge** and no subscriber can
hear it (https://docs.aws.amazon.com/iot/latest/developerguide/iot-basic-ingest.html, fetched 2026-10-17).
Price: $0.15/M rules triggered + $0.15/M actions.

**SDK paths.** `boto3.client("iot").create_topic_rule(ruleName, topicRulePayload={sql, actions,
errorAction, awsIotSqlVersion="2016-03-23"})`, `replace_topic_rule`, `get_topic_rule`, `list_topic_rules`
(CreateTopicRule 5 TPS, fixed). Rule actions need an IAM role the account owner creates once; the library
only *writes the rule definition* and takes a `role_arn` — it never creates IAM.

**Mapping onto v1.**

| v1 event / key | rule (one per tenant) | action |
|---|---|---|
| `strands/<t>/@estop` (tenant e-stop, `args.selector/by/epoch`) | `estop-latch`: `SELECT *, topic(2) AS tenant FROM 'strands/+/@estop'` | republish to `strands/<t>/@estop/latest` (not retained: republish cannot retain); the *device* publishes its lockout retained on `strands/<t>/<id>/lockout` when it latches, so a late joiner sees it; SNS to the on-call topic |
| `strands/<t>/<id>/event/{fault,run.started,run.stopped,estop}` | `events-log`: `SELECT *, topic(3) AS device_id, topic(5) AS event FROM 'strands/+/+/event/+'` | Timestream (one table `events`, dimensions tenant/device_id/event) and/or S3 (`recording metadata`: `run.started/stopped` carry the run id, which is the key the recording (part 9) writes) |
| `strands/<t>/@presence/<id>` (heartbeats, 1 Hz) | **refused**: 3,600 rule triggers/robot/h for what lifecycle events give free (§4) | — |
| `strands/<t>/<id>/state` (10–500 Hz) | **refused** for storage via rules (108,000 triggers/h at 30 Hz = $0.032/h + a Timestream write per sample); recording is part 9's job, local first, then an upload | if a user *wants* cloud telemetry: Basic Ingest `$aws/rules/state-to-timestream/strands/<t>/<id>/state` at a down-sampled rate (`telemetry_hz`), which skips the $0.108/h messaging charge entirely |
| `fault` with `severity` | `fault-page`: `… WHERE severity = 'high'` | SNS / Lambda hook (`on_fault=` URL or ARN placeholder in the user's config) |

**Where it lives / test.** `strands_robots/mesh/iot/rules.py`: `RULES` = the table above as data
(`name`, `sql`, `actions`), `apply(iot, tenant, role_arn)` idempotent (`get_topic_rule` → `replace` or
`create`), `basic_ingest_topic(rule, key)`. Tests: `moto` covers `create/get/list/replace/delete_topic_rule`
(**→ verify** in the venv); the SQL is checked by `iot.validate…`? — no such API; the live test publishes one
synthetic estop and reads back `get_retained_message(topic)`. **→ measure**: rule → republish latency
(expect < 1 s) and whether the retained flag survives republish.

## §3 X.509 certificates and provisioning

**Primitives.** Thing (`device_id`), thing type (`device_type`, ≤ 1 per thing, L-7D0B47D6), thing group
(tenant; ≤ 10 groups per thing, hierarchy ≤ 7), certificate (`iot:CreateKeysAndCertificate` or
`CreateCertificateFromCsr` — the key never leaves the device), policy ≤ 2,048 B (L-D9B73671) attached to the
cert, ≤ 10 policies per cert. Policy variables: `${iot:Connection.Thing.ThingName}`,
`${iot:Connection.Thing.ThingTypeName}`, `${iot:Connection.Thing.Attributes[tenant]}`,
`${iot:ClientId}`, `${iot:Certificate.Subject.CommonName}` — one policy document for the whole fleet
instead of one per device; the thing attribute `tenant` is what scopes topics
(`strands/${iot:Connection.Thing.Attributes[tenant]}/${iot:Connection.Thing.ThingName}/*`). Fleet
provisioning by claim: a *claim* certificate baked into the image with a policy that allows only
`$aws/certificates/create/*` and `$aws/provisioning-templates/<template>/provision/*`; the device calls
`CreateKeysAndCertificate` (or `…FromCsr`) then `RegisterThing(templateName, certificateOwnershipToken,
parameters)`; the template (≤ 10 KB, ≤ 5 versions) creates the thing, attaches the type/group/policy and
sets attributes; an optional pre-provisioning Lambda hook can refuse. Quotas: RegisterThing 10/s,
CreateKeysAndCertificate 10/s (MQTT). Rotation: a second cert is created, attached, the device swaps
files and reconnects, the old one is `UpdateCertificate(INACTIVE)` then deleted. IAM/SigV4 stays for
*operators* (humans, dashboards, agents): `iot:Connect` as `strands-op-*`, `iot:Publish` on `rpc/*` and the
shadow `control` update topic, `iot:Subscribe/Receive` on `state`, `event/*`, `@reply/<own client>`,
`@presence/*`; **no publish right on `@presence/*` or `manifest`** (the reference's deploy requirement).

**SDK paths.** Control plane: `boto3 iot`: `create_thing_type`, `create_thing_group`, `create_thing(thingName,
thingTypeName, attributePayload)`, `add_thing_to_thing_group`, `create_keys_and_certificate(setAsActive)`,
`create_certificate_from_csr`, `create_policy`, `attach_policy(policyName, target=certArn)`,
`attach_thing_principal`, `create_provisioning_template(templateName, templateBody, provisioningRoleArn,
preProvisioningHook)`, `create_provisioning_claim(templateName)` (returns a claim cert + key valid 5 min,
10/s), `update_certificate(newStatus=INACTIVE)`, `delete_certificate(forceDelete)`. Device side (MQTT):
`awsiot.iotidentity.IotIdentityClient`: `publish_create_keys_and_certificate`,
`publish_create_certificate_from_csr`, `publish_register_thing`, + the `subscribe_to_*_accepted/rejected`
pairs (introspected). The CSR path keeps the private key on the device (generate with `cryptography`,
already a mesh dependency for Ed25519 resume keys).

**Identity: two policy variables, one certificate (v0.5 fact (a), kept).** `${iot:Connection.Thing.ThingName}`
resolves only on an MQTT connection whose certificate is attached to the thing (EXPERIMENT 1 proved it live: connect
as own thing ok, a forged `@presence/<other>` publish → PUBACK 135, a foreign-tenant subscribe → SUBACK 135). On
the **HTTPS** path (a *device* calling `SendDirectMessage` on :8443 with its certificate, as v0.5's
`_X509DirectClient` does) there is no connection and the variable is **empty → 403**; there the identity is
`${iot:Certificate.Subject.CommonName}`, which is only useful if the certificate came from a **local CSR with
`CN=<thing name>`** (AWS-generated certificates all say `CN=AWS IoT Certificate`). v1 therefore: issue every
device certificate from a local CSR with `CN=<device_id>` (`cryptography`, already a mesh dependency); MQTT
statements use `Thing.ThingName`; the one HTTPS statement (`iot:SendDirectMessage` on `connection/*`, only if
the device must *initiate* rpcs to other devices) uses `Certificate.Subject.CommonName`. In the reference today
the device never initiates a Direct Message (replies are MQTT publishes; operators call DM with SigV4), so the
HTTPS statement is optional and off by default. v0.5 fact (b): a claim-cert device submitting a CSR with
`CN=<existing robot>` would inherit that robot's rights → the fleet provisioning template gets a
**pre-provisioning hook** that accepts only `CN == requested ThingName` (or the AWS-generated CN) and refuses an
existing thing name; v0.5's `_PROVISIONING_HOOK_SOURCE` (131 lines, a DER walk for the CN) is the seed.

**Mapping onto v1.**

| v1 concept | IoT primitive |
|---|---|
| `device_id` | thing name = MQTT client id = certificate CN (local CSR); the policy pins `iot:Connect` to `${iot:Connection.Thing.ThingName}` |
| `device_type` (`so101`, `microduck`…) | thing type; `searchable attributes`: none needed |
| `tenant` | thing group `strands-<tenant>` + thing attribute `tenant` (used in policy variables) |
| `labels` (`role: arm`) | thing attributes (≤ 50 with a type); the dynamic thing group `strands-<tenant>-role-arm` can mirror a label for fleet deployments |
| `Mesh.mount(robot, iot=…)` on a fresh robot | `provision(claim_cert, claim_key, template)` → own cert/key under `~/.strands/iot/<device_id>/`, then `Mesh("iot://<region>?cert=&key=&client=<device_id>")` |
| operator `Mesh("iot://us-west-2")` | SigV4 WebSocket as today; IAM policy = `OPERATOR_POLICY` document from `provision.py` |
| `Authority` / Ed25519 resume | unchanged; the cert authenticates the *connection*, the assertion authenticates the *resume*. **Refused**: using the cert's CN as the resume authority (a stolen device cert would then approve its own resume) |

**Where it lives / test.** `strands_robots/mesh/iot/provision.py`: `DEVICE_POLICY`, `OPERATOR_POLICY`,
`TEMPLATE` (as data, with `{tenant}`/`{region}`/`{account}` placeholders filled at apply time), `setup_tenant(iot,
tenant, role_arn)` (type/group/policy/template, idempotent), `register(iot, device_id, device_type, tenant,
labels)` (one thing + cert, admin path), `provision(claim, template, …)` (device path over MQTT), `rotate`,
`revoke`. Tests: `moto` covers things/types/groups/certs/policies/attach (**→ verify** `create_provisioning_template`
and `create_provisioning_claim` in moto; else the claim path is live-only). Fixtures carry placeholder
account id `123456789012` and region only. Live: `scripts/iot/provision_fleet.py` creates
`strands-v1-iot-experiment-*`, connects as the thing, proves a publish outside its tenant is refused
(`PermissionError` from `AwsLink.subscribe` / a PUBACK 0x87), deletes everything. **→ measure**: claim →
connected time (template + RegisterThing; expect 2–5 s).

## §4 Direct messaging, MQTT5 request/response, retained, shared subscriptions, lifecycle

**Primitives.** (a) Direct Messaging (today): `SendDirectMessage(clientId, topic, payload≤128 KB,
confirmation, timeout 1–10 s)` → 200 / 404 not connected / 504 no PUBACK; 20,000 TPS; $2.50/M with
confirmation (https://docs.aws.amazon.com/iot/latest/developerguide/direct-messaging.html, fetched
2026-10-17). It is an HTTPS call from the caller (SigV4), so a *device* cannot use it to call another device
without IAM credentials. (b) MQTT5 request/response: `PublishPacket(response_topic=…, correlation_data=…)`;
IoT Core forwards both properties (mqtt.html, fetched 2026-10-17: "Response Topic … Correlation Data …
PUBLISH"); `awscrt.mqtt_request_response.Client` (introspected: `RequestOptions`, `ResponsePath`,
`StreamingOperation`) implements the pattern on top of an mqtt5 client — the same machinery
`awsiot.iotshadow` uses. No 404: a request to an absent device is simply never answered (the 2 s rpc
timeout fires) — unless the request is QoS 1 to a client with a persistent session, in which case it is
*queued*, which is exactly what an e-stop must never be → `clean_session=True` stays, and `message_expiry_interval_sec`
is set on every rpc (= the rpc timeout) so a late delivery is dropped by the broker. (c) Retained messages:
one retained message per topic, 1 publish/s per topic, `GetRetainedMessage` 500 TPS → retained **manifest**
(`strands/<t>/<id>/manifest`) and retained **estop state** (`strands/<t>/@estop/latest`). (d) Shared
subscriptions `$share/<group>/<filter>`: N operator processes split one subscription's load; for *rpc
serving* it is wrong (every device must answer its own rpcs) — right for N dashboard replicas consuming
`state` (100 groups per account). (e) Lifecycle events `$aws/events/presence/connected|disconnected/<clientId>`
with `clientId`, `principalIdentifier`, `sessionIdentifier`, `timestamp`, `disconnectReason`,
`clientInitiatedDisconnect` (life-cycle-events.html, fetched 2026-10-17); the broker detects a dead TCP
connection at keep-alive × 1.5 (30 s → 45 s), vs the heartbeat's 3 s.

**Mapping onto v1.**

| v1 concept | today | native proposal | verdict |
|---|---|---|---|
| `Mesh.invoke` → `transport.query` | SendDirectMessage(confirmation) to the holder client, reply on `@reply/<caller>` | MQTT5 publish on `strands/<t>/<id>/rpc/<name>` with `response_topic=strands/<t>/@reply/<caller>`, `correlation_data=<rid>`, `message_expiry_interval_sec=ceil(timeout)`; device answers on the response topic | **adopt if measured RTT ≤ today's 203 ms p50** (**→ measure** both, same hour, same hosts); keeps the 404 semantics only for estop (below) |
| `estop` rpc | Direct Message **and** tenant `@estop` | keep Direct Message for the targeted stop (the 404 is the feature: "offline, nothing queued" is an answer), plus retained `@estop/latest` | keep |
| `presence` heartbeat 1 Hz | 3,600 publishes/robot/h + per watcher | subscribe operators to `$aws/events/presence/+/<clientId>` (needs the account's event configuration enabled: `iot:UpdateEventConfigurations`, account-wide, done once) **and** a retained `manifest` published at mount with a **Will** message that clears it (`will=PublishPacket(topic=manifest, payload=b"", retain=True)`) | adopt as the *slow* path (0 messages/h idle); keep a heartbeat only when `presence_ttl < 45 s` is required, at `presence_ttl/3` as today — make it a knob, default off on IoT |
| `Directory._fetch` (rpc for the manifest on each appearance) | 1 rpc per appearance per watcher | read the retained `manifest` (delivered on subscribe, 0 rpc) or `get_thing_shadow(manifest)` | adopt |
| `state` to N dashboards | N deliveries | `$share/strands-<t>-dash/strands/<t>/+/state` for replicas of **one** consumer | optional |
| `observe(images=True)` over IoT | 128 KB cap → fails for one 640×480 JPEG > 128 KB at quality > 85 | **refused on MQTT**; cameras go to KVS (§6) | refuse |

**Prior art numbers (v0.5 #4242, merged 2026-09-29, +5,233 lines / 47 files; the brief's PRIOR ART):** confirmed
Direct Message rpc operator → robot → operator **p50 198–221 ms**, offline (404) verdict in **97–265 ms**, one warm
HTTPS call **81 ms**; the v1 reference measured the same path at 203 ms p50. v0.5's `_mqtt5_reply_properties` already
read `response_topic` / `correlation_data` off inbound publishes (the reply key was computed from them, with the
robot checking the reply topic is its own for that sender — fact (d)), so MQTT5 request/response is a port, not an
invention. v0.5's `_TOPIC_POLICY` retained `presence`, `health`, `safety/event` and (at first) the safety
*commands*; the commands were later un-retained and a `clear_retained_safety_messages` tool shipped, because "a
retained stop or release is handed to every client that subscribes later … a stop nobody pressed now". **Kept
for v1:** retained *state* only (manifest, the device's own `lockout`), never a retained *command*.

**Where it lives / test.** `transports/iot_link.py` grows `request(topic, payload, response_topic,
correlation, expiry)` and `will=`; `transports/iot.py` picks DM for `estop`, request/response for the rest;
`mesh/iot/lifecycle.py` maps `$aws/events/presence` → `OnAlive`. Tests: `FakeIotBroker` learns
`response_topic`/`correlation_data`/`retain`/`will` (≈ 60 lines) so every existing mesh test runs on both
paths. Live: `scripts/iot/rtt.py` **→ measure** p50/p95 for DM vs request/response at n=200, and
disconnect → lifecycle event delay.

## §5 Greengrass v2

**Primitives.** A core device = a thing running the Greengrass nucleus (Java 8+, Linux/Windows; **nucleus
lite** = C, no Java, for ≤ 50 MB RAM devices; setting-up.html fetched 2026-10-17). Components = recipe
(YAML/JSON ≤ 8 KB) + artifacts (S3, ≤ 2 GB total), lifecycle `Install/Run/Shutdown`, configuration merged
per deployment. Deployments target a thing **or a thing group** (the fleet update path: a new component
version deployed to `strands-<tenant>` rolls to every core in the group, with rollout rate and failure
policy). Public components: `aws.greengrass.Nucleus`, `aws.greengrass.clientdevices.mqtt.Moquette` (local
MQTT 3.1.1 broker, port 8883) or `…mqtt.EMQX` (MQTT5), `aws.greengrass.clientdevices.Auth`,
`aws.greengrass.clientdevices.mqtt.Bridge` (local ↔ cloud topic bridge with retained/QoS mapping),
`aws.greengrass.clientdevices.IPDetector`, `aws.greengrass.StreamManager` (local queue → Kinesis/S3/IoT
Analytics/SiteWise, survives offline, bandwidth-limited), `aws.greengrass.TokenExchangeService` (AWS creds for
components via a role alias), `aws.greengrass.LogManager`. IPC from a component: `awsiot.greengrasscoreipc`
(introspected): `PublishToIoTCore`, `SubscribeToIoTCore`, `PublishToTopic`/`SubscribeToTopic` (local pub/sub,
no broker), `GetThingShadow`/`UpdateThingShadow` (the local `aws.greengrass.ShadowManager`, synced to the cloud),
`GetConfiguration`, `UpdateState`, `DeferComponentUpdate`. Cost: $0.16 per active core per month; messages that
stay on the LAN are free.

**Mapping onto v1.**

| v1 concept | Greengrass |
|---|---|
| a robot host (a Jetson, a Pi, an EC2 box) | a core device `strands-<tenant>-<host>`; the robots it serves are **client devices** *or* run in-process in the component |
| `strands-robots` process that mounts robots | component `com.strands.robots` (recipe generated by `strands_robots.mesh.iot.greengrass.recipe(...)`, artifact = a wheel or a pip-install `Install` step; `Run` = `python -m strands_robots.mesh.serve --iot=greengrass …`), configuration = `{tenant, robots: [{name, device_id, labels}], state_hz, shadow_hz}` |
| `Mesh("zenoh://")` on the LAN | **unchanged** — zenoh stays the LAN mesh; Greengrass's local broker is *not* a replacement (MQTT 3.1.1 Moquette has no liveliness, no queryables) |
| cloud down | the component keeps running; `Mesh(("zenoh://", "iot://greengrass"))` where the `iot://greengrass` transport publishes through IPC (`PublishToIoTCore` spools until the nucleus reconnects — **but** spooled e-stops are the queue we refuse: the transport drops `@estop` and rpcs when `UpdateState`/connectivity says offline and sends only `state`/`event`) |
| recording upload (part 9 writes LeRobot episodes locally) | `StreamManager` S3 export of the episode directory, `priority`/bandwidth-limited; metadata event via the §2 rule |
| fleet update | `CreateDeployment(targetArn=thing group, components={com.strands.robots: {version}})`; `iotJobConfiguration` for rollout rate |
| `on_disconnect` | unchanged: it is the device's watchdog on the *controller's* presence, not on the cloud |

**Refused.** Running the servo loop inside a Greengrass Lambda (container, cgroups v1 only, no real-time);
replacing zenoh with Moquette/EMQX on the LAN (loses liveliness and 1 ms rpcs); letting the nucleus *restart*
the component on a deployment while a run is active (`DeferComponentUpdate` is called from `run.started` until
`run.stopped`).

**Where it lives / test.** `strands_robots/mesh/iot/greengrass.py`: `recipe()`, `deploy()` (boto3
`greengrassv2.create_component_version`, `create_deployment`), the IPC-backed `Link` (`GreengrassLink`,
same protocol as `AwsLink`, ≈ 80 lines). Unit tests: recipe JSON schema + size ≤ 8 KB, `GreengrassLink`
against a fake IPC. Live (on an EC2 host): install the nucleus, deploy the component,
kill the cloud route (`iptables` on 8883/443 to the endpoint), prove zenoh-side rpcs still answer and the
spool drains on reconnect **→ measure** reconnect-to-first-message and whether an e-stop sent while offline
arrives late (it must not).

## §6 Kinesis Video Streams

**Primitives.** A stream per camera (`CreateStream`, retention 0–87,600 h), ingest by `PutMedia` (MKV
fragments 1–10 s, ≤ 50 MB, ≤ 3 tracks, **1 concurrent producer per stream**, 100 Mbit/s), consume by
`GetMedia` (3 concurrent, 200 Mbit/s), HLS/DASH playback, `GetImages`. Producers: C producer library,
C++/Java/Android SDKs, and the GStreamer plugin `kvssink` (built from the C++ SDK); the doc lists **no
Python producer** (producer-sdk.html, fetched 2026-10-17). Python-only options: (a) `boto3 kinesis-video-media`
is consumer-only; `PutMedia` is not in boto3's `kinesisvideo` client (32 ops, none is PutMedia) → a raw
SigV4-signed chunked HTTPS POST of MKV would have to be hand-built (**refused**: fragile, no backpressure,
no ACK handling); (b) GStreamer from Python (`gi` PyGObject): `appsrc ! videoconvert ! x264enc tune=zerolatency
! h264parse ! kvssink stream-name=… aws-region=…` fed with the MuJoCo render or a real camera frame — this is the
honest Python path, needs the `kvssink` plugin built on the host (Linux; ~10 min build on ec2). KVS WebRTC:
signaling channel per camera (`CreateSignalingChannel`, `GetSignalingChannelEndpoint`, `kinesis-video-signaling
GetIceServerConfig`), master = the robot (1 per channel), viewers ≤ 10 (adjustable), P2P with TURN fallback
(5 Mbit/s, 300 s sessions); WebRTC storage (`JoinStorageSession`) records the WebRTC session into a KVS stream
at $0.0034/min. Python WebRTC master: `aiortc` + the signaling protocol over the websocket endpoint
(no AWS Python SDK for WebRTC either — the C SDK and JS SDK are the official ones). Latency classes: KVS
PutMedia → HLS ≈ 3–10 s (fragment duration bound); WebRTC P2P ≈ 100–300 ms; **→ measure** both on ec2
(EDGE) with the MuJoCo camera at 640×480@30 fps.

**Mapping onto v1.**

| v1 concept | KVS |
|---|---|
| `manifest.sensors` of kind `camera` (name, shape, fps) | one stream `strands-<tenant>-<device_id>-<camera>` + one signaling channel of the same name; names and HLS URL advertised in the retained `manifest` shadow under `cameras[]` |
| `robot.observe(images=True)` frames (the twin's MuJoCo render or a real camera) | fed to `appsrc` at the camera's fps by a `CameraPublisher` thread (same pattern as `stream.publish`: sampled once on the device, however many viewers) |
| `Mesh.subscribe(id, "state")` for images | **refused** on MQTT (128 KB cap, cost); the dashboard (part 10) embeds the HLS URL (`GetHLSStreamingSessionURL`) for recorded/near-live, and the WebRTC viewer (JS SDK) for live |
| `Mesh.connect(id, images=True)` (a remote policy wants frames) | **not KVS**: a policy needs raw frames at the policy's rate with < 50 ms added; today's measured path is zenoh (`real` episode: 3 MuJoCo cams over Zenoh at 20.1 Hz). Over the internet, KVS WebRTC *data* is a video codec stream, not tensors → refuse; document "remote policies over the internet use the zenoh router / `Mesh.connect` on the same LAN" |
| recording (part 9) | KVS retention is the cloud copy of the *video* only; the LeRobot episode (state + action + video) stays part 9's; a rule (§2) links the run id to the KVS fragment timestamps (`GetClip`) |

**Cost** (§0 table): one 1.5 Mbit/s camera ≈ $0.0057/h ingest + $0.0155/GB-month; a live WebRTC viewer
without TURN ≈ 0; with TURN $0.0072/h.

**Where it lives / test.** `strands_robots/mesh/iot/kvs.py`: `ensure_stream()`, `ensure_channel()`,
`CameraPublisher` (GStreamer pipeline string builder + `appsrc` feeder; raises a clear `ImportError` with the
`hint("iot-edge")` when `gi`/`kvssink` are missing), `hls_url()`. Unit tests: pipeline string, naming, the
manifest `cameras[]` entry; `moto` covers `kinesisvideo` `create/describe/list/delete_stream` (**→ verify**).
Live (EDGE on ec2): build `kvssink`, stream the so101 twin camera for 60 s, read back `GetMedia`/HLS,
WebRTC master with `aiortc` + a headless viewer; **→ measure** glass-to-glass latency, bitrate, CPU.

## §7 Prior art: strands-robots v0.5 `mesh/iot/*` (4,861 lines) → v1

Read in v0.5 `strands_robots/mesh/iot/` on `main` and `mesh/transport/iot_transport.py` (1,717 lines), 2026-10-17.
The per-file carry/drop table is in MAPPING.md ("v0.5 → v1"). What the code taught, in numbers:

- `provision.py` (2,341): a robot IoT policy with the children grant hit the **2,048-character cap** (2,227 and
  2,420 measured) → split into two policies per certificate (a cert may carry 10). Policy variables compose with
  literal text inside one segment (`strands/${…ThingName}__*/state`: 574 publishes in 60 s, 0 disconnects; the
  wrong names ended the session with 135). Amazon Root CA1 pinned by SHA-256 with a runbook (186 lines of
  `_ensure_ca`) — v1 drops this: awscrt ships the trust store, and a pinned CA file is the thing that breaks on
  the CA rotation it was meant to defend. Thing-name validation (47 lines) → v1's `wire.check_id` already does it.
- `bootstrap.py` (1,417): account-wide infra as code — DynamoDB safety table with KMS/PITR, two Lambdas (e-stop
  fan-out 107 lines, pre-provisioning hook 131 lines), two IoT rules, roles, log group, the fleet template.
  v1 keeps the **template + hook** concept (feature 3) and the **rules as data** (feature 2); drops DynamoDB/Lambda
  fan-out (v1's tenant `@estop` key latches every mesh without a Lambda; events go to Timestream/S3 by rule,
  no custom code) and refuses to create IAM roles from the library.
- `shadow.py` (245): `ShadowMirror` mirrored the *presence heartbeat* to a named shadow `presence` by hooking the
  publish path. v1 generalises: three named shadows (`manifest`, `state` ≤ 1 Hz, `control`), and adds the half
  v0.5 never had — **`desired` → `confirm_act`**.
- `registry.py` (222): read-only `iot:ListThings` + fleet-index connectivity for the dashboard's grey cards, every
  failure a status word. Carried as a 40-line `registry.py` in v1's `mesh/iot/` (dashboard part 10 draws
  provisioned-but-silent devices from it + the `manifest` shadow).
- `camera_offload.py` (407): JPEG → S3 + presigned URL + a 200 B MQTT `camera/<cam>/ref` pointer, because frames
  never fit 128 KB and "the cost model is brutal at 5–10 Hz × N robots". v1 keeps the *refusal* (no frames on
  MQTT) but replaces S3-per-frame with **KVS** (feature 6): a video stream is what a dashboard wants, and
  3,600 S3 PUTs/camera/h at 1 Hz ($0.018) buys 1 frame/s, where KVS at $0.0057/h buys 30.
- `cli.py` (144): `strands-robots iot provision-robot|provision-operator|reprovision|teardown|withdraw-estop-publish|
  clear-retained-safety`. v1 keeps `provision`, `rotate`, `revoke`, `setup-tenant`, `apply-rules` as functions
  (a CLI entry is part 10's/the tool's job, not the mesh's).
- `transport/iot_transport.py` (1,717): the port is already done — `transports/iot_link.py` (214) + `iot.py`
  (215) carry Direct Messaging with 404 semantics, the FakeIotBroker, and the holder rule. Dropped on the way (and
  staying dropped): the zenoh-shaped sample wrappers (v1 has one `Transport` protocol), the env-var switches
  (`DIRECT_ENV_VAR`, `DIRECT_AUTH_ENV_VAR`, `INLINE_CAMERA_ENV` → v1 has URL query parameters), the x509 HTTPS
  Direct client on :8443 (optional in v1, see §3), the 20-entry `_TOPIC_POLICY` (v1's `wire.KEYS` has 7 keys and
  decides QoS per slot in the transport).

Size check: v0.5 spent 4,861 + 1,717 = 6,578 lines on IoT; the v1 estimate for all six features is ≈ 2,010 on top
of 671 already in the reference — 2.4 × smaller for a superset of the behaviour, which is the point of the rewrite.
