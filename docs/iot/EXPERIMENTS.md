# IoT experiments — numbers from one AWS account

Region us-west-2. Client: one laptop, egress through Ashburn (VA); **network floor: TCP connect to the IoT data endpoint
:8883 p50 80 ms / p95 87 ms (n=20)**, so every number below includes ~80 ms of round trip that a device in us-west-2
would not pay. Scripts in `scripts/iot/` (re-runnable, each one provisions what it needs and deletes it at the end;
`python scripts/iot/exp1_provision.py --cleanup-only` lists leftovers and sweeps deprecated thing types).
All resources named `strands-v1-iot-exp-<exp>-<run>-*`, tagged `Project=strands-v1-iot`. awsiotsdk 1.31 (MQTT5), boto3 1.43.

Status: 1 ✔ · 2 ✔ · 3 ✔ · 4 ✔ · 5 ✔ · 6 ✔ · 7 ✔ — all seven measured; AWS clean (see §8)

## 1. Thing / cert / least-privilege policy from the manifest, connect as device and operator  (`exp1_provision.py`, run 2a5d4b)

Input: the so101 manifest as the reference's Mesh publishes it (device_type `robot_arm`, 18 sensors, 6 controls, 7 rpcs, **5,123 bytes**).
Mapping used: thing = device_id · thing type = device_type · thing group = tenant · thing attributes = {tenant, device_type, n_controls} · one cert per thing ·
one policy with 5 statements using `${iot:Connection.Thing.ThingName}` + `iot:Connection.Thing.IsAttached` (connect only as own thing name;
publish only on `strands/<tenant>/@presence/<own device ids>`, `strands/<tenant>/<own device id>/*`, `@reply/*`, `@estop`, own `$aws/things/<thing>/shadow/*`;
subscribe/receive only inside the tenant + own shadows).

| step | ms |
|---|---|
| create_thing_type | 150 |
| create_thing_group | 151 |
| create_thing | 140 |
| add_thing_to_thing_group | 155 |
| create_policy | 132 |
| create_keys_and_certificate | 156 |
| attach_policy | 112 |
| attach_thing_principal | 111 |
| **provision total (8 calls)** | **1,108** |
| delete all (6 calls) | 923 |

| measurement | value |
|---|---|
| policy propagation (attach → first accepted mTLS connect) | **0.57 s, 1 attempt** (the ref's round-trip script sleeps 8 s; 1 s is enough here) |
| MQTT5 connect, mTLS as the thing | 558 ms |
| MQTT5 connect, SigV4 WebSocket as operator (role in env) | 498 ms |
| device SUBSCRIBE own tenant `strands/lab/arm/+` | reason code 1 (granted QoS 1), 110 ms |
| device SUBSCRIBE other tenant `strands/other-tenant/#` | **reason code 135 Not authorized**, 74 ms; connection stays up |
| device PUBLISH own presence (QoS 1) | PUBACK 0 |
| device PUBLISH `@presence/someone-else` (QoS 1) | **PUBACK 135 Not authorized**, connection stays up (0 disconnects) — MQTT5 reports it; MQTT3 would have dropped the link |
| operator SUBSCRIBE `strands/lab/#` | reason code 1, 118 ms |
| second connection with the SAME client id (same cert) | first link disconnected with **reason 142 Session taken over** within 1.5 s; second connected in 502 ms |

Conclusions for MAPPING: the manifest gives everything a policy needs (device ids served, tenant); the policy variable makes ONE policy document
per tenant possible (no per-thing policy needed); a forged `@presence` from another device's cert is refused at the broker, which is the property the
reference's `_holder_ok` defends in software. Thing types can only be deprecated, and deleted 5 min later → the cleanup needs a second pass.

## 2. Device shadows  (`exp2_shadow.py`, run ffd327; 10 s per rate, n=100 round trips)

| what | value |
|---|---|
| `manifest` named shadow = the manifest document | **5,204 bytes, accepted** in 252 ms; operator `get_thing_shadow` p50 272 / p95 296 ms (n=20), 12,741 bytes back (reported + metadata timestamps per leaf) |
| `state.reported` at **10 Hz** (18 joint values, 528 B/update, QoS 0) | 100 sent, **100 accepted, 0 rejected**; publish → `update/accepted` p50 151 / p95 204 ms |
| `state.reported` at **30 Hz** (527 B/update) | 299 sent, **227 accepted, 72 rejected 429 TOO_MANY_REQUESTS** (≈ 22.7 accepted/s) — the per-shadow throttle sits between 20 and 30 updates/s, as the quota (20 req/s per shadow) says; accept latency unchanged p50 140 / p95 204 ms |
| reported document > 8 KB | **413 "Json document too large. Size is 9011, max allowed: 8192"**; nothing stored |
| `state.desired` set by operator over MQTT (SigV4) → device `update/delta` | **p50 121 / p95 178 / p99 200 ms** (n=100) |
| `state.desired` set by operator over HTTPS (`iot-data update_thing_shadow`) → delta | **p50 312 / p95 359 ms** (n=100); the HTTPS call itself p50 277 ms |
| delta payload | 164 bytes (`state` + `metadata` + version + timestamp) |
| device reports the target back → delta clears | yes; shadow version 528 after 528 updates |

Conclusions for MAPPING: shadows are an **operator/dashboard view at ≤ 10 Hz and 8 KB**, not a telemetry path (30 Hz is throttled, images never fit);
`desired` → `delta` → safety chain → `act` costs ~120 ms + the same again for the reported echo, versus 203 ms p50 for the reference's confirmed
Direct Message `act` — a shadow write is a *target*, not a command, and a lost delta is re-read from the shadow on reconnect (the thing a DM 404
deliberately does not do). The manifest fits one named shadow with 3 KB to spare; a dashboard that has never met the device draws it from one GET.

## 3. Rules engine on the tenant e-stop topic  (`exp3_rules.py`, runs aee426 / 4ecaee / 688e79, n=30 each)

Rule: `SELECT *, topic(2) AS tenant, timestamp() AS at FROM 'strands/+/@estop'` with two actions, republish to `strands/${topic(2)}/@events/estop`
(QoS 1) and SNS → SQS (raw delivery). Hooks setup (SQS queue + policy, SNS topic + subscription, inline role policy) 1,472 ms; rule create 1 attempt.

| measurement | value |
|---|---|
| device publish `@estop` → rule → republish → operator receives | **p50 164 / p95 187 / p99 266 ms** (n=30, 0 lost); run 4ecaee p50 157 / p95 210 |
| the same, published by the operator (SigV4 WS) | p50 271 / p95 303 ms (n=30) — one more hop from the Ashburn egress |
| device publish → rule → SNS → SQS (`SentTimestamp` − publish) | **p50 90 / p95 124 / p99 171 ms** (n=30; SQS clock vs laptop clock, ±10 ms) |
| rule payload after SQL | original fields + `tenant` + `at` (ms epoch); 106 bytes |
| rule whose IAM role was created seconds before | **messages dropped silently** until IAM propagated: in run aee426 the first republish never came within 10 s × 50 tries (the script hung; killed); in a debug pass with a 10-minute-old role and a fresh inline policy the first **4 messages (6 s) were lost**, then every one arrived. A rule's actions do not retry a failed AssumeRole. Reuse a role; never create one in the same breath as the rule. |
| rule with a 10-min-old role + policy | warm-up 0.2 s, 0 lost |
| Basic Ingest `$aws/rules/<rule>` from the device | **PUBACK 135** — the device policy has no `$aws/rules/*` right (it must be granted explicitly) |
| Basic Ingest from the operator, first attempt (run 688e79) | 30/30 lost **by my own design, not the service**: with Basic Ingest the message has no MQTT topic, so `topic(2)` is "rules" and the republish went where nobody listened |
| Basic Ingest from the operator, tenant in the payload, rule republishes to `strands/${tenant}/@events/estop` (run 6b4fa9) | **p50 281 / p95 323 ms, 0 lost** vs the same operator publishing on the real topic p50 279 / p95 341 ms: same latency, the difference is only cost — Basic Ingest skips the $1.00/M messaging charge; rule trigger $0.15/M + $0.15/M per action still apply (us-west-2 list prices). Lesson for MAPPING: a Basic-Ingest payload must carry tenant + device id itself. |
| rule whose inline role policy was re-written seconds before (run 6b4fa9) | first republish after **9.2 s, 3 messages lost silently** — the IAM propagation finding reproduced with a 30-min-old role, so it is the policy, not the role, that propagates |
| retained `@estop` without `iot:RetainPublish` in the policy (run 4ecaee) | **silently not retained**: a late operator gets nothing, `list_retained_messages` empty — RETAIN needs its own action in the policy |
| retained `@estop` with `iot:RetainPublish` (run 688e79) | late operator receives it on subscribe with `retain=1` (SUBACK 114 ms), `list_retained_messages` lists it; cleared by an empty retained publish (verified) |
| does the republished copy carry RETAIN? | **no** (republish action has no retain flag) → the retained per-tenant e-stop state is written by the publisher, not by a rule |

Conclusions for MAPPING: events as topics + rules work with ~160 ms to a republish subscriber and ~90 ms into SNS/SQS from the device; retained state
is a publisher-side flag and a policy action (`iot:RetainPublish`); a Basic-Ingest payload must carry tenant and device id itself; rule IAM roles are
long-lived infrastructure, not per-run resources (this account's Research session can create but not delete roles: `iam:DeleteRole` explicitly denied).

## 4. Conversations: MQTT5 request/response vs Direct Messaging on the v1 link  (`exp4_request_response.py`, run 91ef00, n=200 each)

Device = mTLS link with the exp-1 policy, serving `strands/lab/arm/@rpc/+` and answering on the request's MQTT5 `response_topic` with its
`correlation_data`. Operator = SigV4 WebSocket for MQTT, boto3 `iot-data` for Direct Messages (the exact call in `transports/iot_link.AwsLink.send_direct`).
Request 57 bytes. Network floor from the laptop p50 80 ms RTT.

| path | p50 | p95 | p99 | lost |
|---|---|---|---|---|
| MQTT5 request/response, QoS 0 both ways | **170 ms** | 176 | 192 | 0/200 |
| MQTT5 request/response, QoS 1 request (PUBACK awaited), QoS 0 reply | 204 ms | 249 | 291 | 0/200 |
| Direct Message `confirmation=True` → reply over MQTT | **166 ms** | 181 | 296 | 0/200 |
| … the HTTPS call alone (returns after the device acked) | 156 ms | 192 | 448 | |
| Direct Message `confirmation=False` → reply over MQTT | 165 ms | 172 | 176 | 0/200 |
| … the HTTPS call alone | 79 ms | 83 | 87 | |
| **reference end to end**: `tests/mesh/aws_iot_round_trip.py --n 200` (its own thing/cert/policy, Mesh.discover + Mesh.invoke("act") over IotTransport) | **200 ms** | — | 265 | 0/200; discover 0.93 s; broadcast estop acked; estop to the killed device → 404 "nothing was queued"; created thing/policy/certificate, deleted all three |

Offline (device link closed without a goodbye): `send_direct_message(confirmation=True)` → **HTTP 404 on the first call 0.08 s after the disconnect**,
20/20 calls 404 at p50 77 / p95 80 ms (one HTTPS round trip, no waiting). An MQTT5 request to the same offline device: PUBACK ok and silence —
nothing tells the caller the device is gone, so request/response alone needs the presence heartbeat or lifecycle events to say "offline".
`send_direct_message` accepts `responseTopic`, `correlationData`, `userProperties`, `contentType` (boto3 1.43), so a Direct Message can carry the
same reply addressing as an MQTT5 request; the device's reply then rides MQTT either way.

Two client findings for the implementation: (1) awscrt hands `correlation_data` back as **str** when it is valid UTF-8 (bytes otherwise) — match on both;
(2) the v1 reference's 200 ms p50 = this 166 ms Direct Message RTT + Mesh envelope/JSON + the `confirm_act` path, i.e. the transport is 83 % of it.

Conclusions for MAPPING: on this network the three paths are within 35 ms of each other (166 / 170 / 204 ms, all ≈ 2 × the 80 ms floor), so the choice is
not latency: Direct Messaging gives the **404 in one HTTPS round trip** that an e-stop needs (the reference's design, confirmed), MQTT5 request/response
gives a pure-MQTT path that works from a cert-only device without IAM credentials and inside Greengrass. Keep DM for operator→device commands,
use response_topic + correlation_data for device→device and local-broker conversations.

## 5. Lifecycle events vs the heartbeat TTL  (`exp5_lifecycle.py`, runs 92a5f1 + second run, n=5 cycles)

Operator subscribed to `$aws/events/presence/{connected,disconnected}/<thing>` (no account configuration needed; SUBACK granted with the IAM role).

| what | value |
|---|---|
| device connect success → `connected` event at the operator | p50 21–27 ms / p95 47–86 ms (n=5 ×2 runs) |
| device `client.stop()` → `disconnected` event | p50 113–118 ms / p95 120–139 ms; the event says `clientInitiatedDisconnect: false, disconnectReason: CONNECTION_LOST` — awscrt's stop() closed the socket before a DISCONNECT went out, so the broker saw a lost connection, not a goodbye |
| device process SIGKILLed | `disconnected` event after **0.11–0.17 s** (CONNECTION_LOST) — the OS closes the TCP socket on process death, so this is NOT a network partition |
| device process SIGSTOPped (socket open, no PINGREQ = a real partition), client keep-alive 5 s and 30 s | **no `disconnected` event within 120 s**, either run |
| why: the CONNACK's Server Keep Alive | **IoT Core answers `server_keep_alive_sec = 1200`** for requested 5, 30 and 60 s (SigV4 WS, measured) → the broker gives a silent client 1.5 × 1200 = **30 min** before it calls it gone |
| reference heartbeat (transports/iot.py) | beat every ttl/3 = 1 s, holder forgotten after ttl 3 s + ≤ 0.1 s expire tick → **offline in 3.0–3.1 s**, 1 × 50-byte publish/s/device |

Conclusions for MAPPING: lifecycle events are an excellent **audit** signal (free, ~25 ms on connect, ~115 ms on a socket close) and a cheap trigger for a
rule (e.g. clear `@presence` retained state on `disconnected`), but they **cannot replace the 3 s heartbeat** for liveness: a frozen or partitioned robot stays
"connected" for up to 30 minutes. Keep the heartbeat; add the lifecycle rule as the belt to its braces (the 404 from Direct Messaging is the third signal
and the fastest: 0.08 s after a socket close, exp 4).

## 6. Fleet provisioning by claim  (`exp6_claim_provisioning.py`, run 84bc56, one run)

Template (36-char name cap!) with parameters DeviceId/DeviceType/Tenant → thing `strands-v1-iot-exp-e6-<run>-${DeviceId}` with attributes
{tenant, device_type, provisioned_by: claim}, certificate Active, the exp-1 least-privilege policy attached (one document per tenant, policy variables).
Pre-provisioning hook = a 20-line Lambda that walks the certificate DER for CN (OID 2.5.4.3) and allows only `CN == <prefix>-<DeviceId>` (prior-art fact b).
Device side: claim cert (policy: connect as `<prefix>-*`, only the `create-from-csr` and `provisioning-templates/<t>/provision` topics) → MQTT5 →
`CreateCertificateFromCsr` with a LOCAL `openssl req` CSR, CN = thing name → `RegisterThing` → reconnect with the new cert → claim cert retired.

| step | ms |
|---|---|
| Lambda create (+ active wait) | 1,277 |
| create device policy / template / claim policy / claim cert | 162 / 217 / 163 / 205 |
| claim connect (cert 2 s old) | 2,492 |
| **forged CSR, CN = another robot** → CreateCertificateFromCsr | accepted in 255 ms (the cert exists, PENDING_ACTIVATION) |
| **forged RegisterThing** | **rejected 403 AccessDenied by the hook in 552 ms**; the cert is never activated; nothing is created |
| honest CSR → certificate | 263 |
| honest RegisterThing (hook + thing + policy + activate) | **471** |
| first connect with the new cert | 1,644 (1 attempt) |
| **claim connect → connected as the thing**, including the forged attempt and two RSA-2048 keygens | **9,350** |
| retire the claim cert (INACTIVE) | 123 |

Checks after: thing attributes {device_type robot_arm, provisioned_by claim, tenant lab}; the device cert's subject CN is the thing name; status ACTIVE;
with the new cert, own-tenant subscribe → granted, other tenant → 135. Deleted: thing, device cert, the forged cert, claim cert, both policies,
template, Lambda, the role's inline policy (leftovers all empty). Only the IAM role is reused (session cannot delete roles; its trust now lists
iot.amazonaws.com and lambda.amazonaws.com — the owner deletes it).

Conclusions for MAPPING: `Mesh.mount(…, iot=…)` on a fresh robot can get its own identity in ~5 s without the forged detour (claim connect
2.5 s + CSR 0.26 s + register 0.47 s + connect 1.6 s); the hook is the only place that binds CN to the thing, so it is not optional; the template
must emit `CN == thing` certificates (CSR flow, never `CreateKeysAndCertificate`, whose CN is always "AWS IoT Certificate").
The IAM role the template assumes and the hook's execution role are long-lived infrastructure (one role can serve both).

## 7. Retained manifest + shared subscriptions  (`exp7_retained_shared.py`, run fe85f0, n=100 at 10 Hz)

| what | value |
|---|---|
| device publishes its manifest with RETAIN on `strands/lab/arm/@manifest` (QoS 1) | 5,179 bytes, PUBACK 0 (policy has `iot:RetainPublish` on that topic) |
| an operator that connects later subscribes | the full document arrives **145 ms after the subscribe call** (SUBACK 121 ms), `retain=1`, 5,179 bytes |
| device re-publishes (rev 2) | the late operator receives the live update (1 message); `list_retained_messages` shows the topic (5,187 B); HTTPS `get_retained_message` returns it |
| empty retained publish | clears it (verified by list) |
| `$share/ops-<run>/…/@state`, 2 members, 100 QoS-1 messages | **A 48 / B 52, total 100, each message exactly once**; two plain subscribers got 100 each |
| latency device → shared member / plain subscriber | p50 131 / p95 173 ms vs p50 127 / p95 171 ms — no cost for sharing |
| member A disconnects, 20 more messages | **B got 20/20** |

Conclusions for MAPPING: the retained `@manifest` replaces the manifest *queryable* for dashboards (a reader that has never met the device draws it from
one subscribe, 145 ms, no device round trip; the device still answers `describe` for callers who want `task`/`estop` live); it costs one retained
message per device (128 KB cap, far above 5 KB). Shared subscriptions give N operator replicas load-balanced consumption of a device's stream with
no change on the device, and fail over when one leaves. Both are policy-visible (`iot:RetainPublish`, `topicfilter/$share/<group>/…`).

## 8. Cleanup state at the end of the runs
Every run deletes what it created and prints `leftovers` (things, policies, thing groups, thing types, topic rules, provisioning templates, active
certificates attached to our things, plus SNS/SQS/Lambda/role inline policies where used). Deprecated thing types cannot be deleted for 5 minutes
after deprecation; `python scripts/iot/exp1_provision.py --cleanup-only` sweeps them. The ONE thing that stays: IAM role
`strands-v1-iot-exp-e3-aee426-rule-role` (no policies attached; trust iot + lambda) — the session policy used explicitly denies
`iam:DeleteRole`, so the owner deletes it.
