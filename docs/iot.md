---
description: If you decide to use AWS IoT, the mesh speaks it natively — shadows, rules, certificates and fleet claims, without changing the robot.
---

# AWS IoT

`Mesh("iot://us-west-2")` already runs the mesh over AWS IoT Core: discover, invoke, subscribe and e-stop work as on a
LAN, a command takes about 200 ms, and an e-stop to an offline robot comes back 404 instead of waiting in a queue.
This page is the next step: if your fleet lives in AWS IoT anyway, one option on `mount` makes a robot a proper
IoT device.

{{drawing:d08_iot_core}}

<!-- test: skip, needs AWS IoT credentials and device certificates -->
```python
from strands_robots import Robot
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import IotOptions

cloud = Mesh("iot://us-west-2?client=arm-1&cert=~/.strands/iot/arm-1/cert.pem&key=~/.strands/iot/arm-1/key.pem",
             tenant="lab-a")
cloud.mount(Robot("so101"), device_id="arm-1", iot=IotOptions(shadow_hz=0.5))
```

Nothing changes for the robot: `Robot`, `Driver`, `Simulation` and the safety chain are untouched. What changes is
what the cloud sees.

## Shadows: the device's last known state, and targets that go through the safety chain

Three named shadows per thing. `manifest` holds a compact document of the device (its identity and labels, each
control's range as `{name: [lo, hi]}`, the units, the sensor count per kind, the rpc and event names, and `full`,
the mesh key that carries the complete record; 0.7 KB for an SO-101, 4.0 KB for a 51-joint humanoid, under the
8 KB shadow cap for every robot in the registry), written once at mount and deleted at unmount, so a dashboard that
has never met the robot can draw it from one GET. `state` holds the live state at `shadow_hz` (default 0.5 Hz, at most
1 Hz; the mesh `state` stream stays the live path at 10–500 Hz) plus the lockout, so a late operator sees that the
arm is stopped and by whom. `control` takes targets:

```python
import boto3, json
data = boto3.client("iot-data", region_name="us-west-2")
data.update_thing_shadow(thingName="arm-1", shadowName="control",
                         payload=json.dumps({"state": {"desired": {"shoulder_pan.pos": 0.3}}}).encode())
```

The delta reaches the robot and goes through `confirm_act` like every other write — the same range check, the same
operator gate. Measured on a real account: desired written → arm moving in **190 ms**. A target the chain refuses
comes back verbatim under `reported.refused` (`"so101: elbow_flex=3.1000 rad is outside its range [...]"`) and is
cleared from `desired`. Two things a shadow will never do: `estop`/`resume` as a field (a shadow persists and is
eventually consistent; a latch must be immediate and must never replay — use the rpc or `broadcast("estop")`), and
`state` above 1 Hz (the per-thing quota throttles at 20 updates/s and a document is capped at 8 KB). A target that
arrives late is dropped as stale rather than moved to: `IotOptions(stale_s=2.0)` is the bound (at least 1 s, because
AWS stamps `metadata` in whole seconds; `None` applies every delta). It is the shadow's own knob — the mesh's
`watchdog_ms` is a 500 ms liveness check for the rpc path and stays as it is. The reply always fits the 8 KB cap: a
delta of hundreds of unknown keys gets one collapsed refusal line and its clears across as many writes as needed, and
an exception while applying is written under `reported.refused["*"]` — the worker that applies deltas never dies on an
operator's input. `IotOptions` checks every bound when it is built, so a bad option (`telemetry_hz=2`) is refused
before anything starts, and a mount that fails on a later step undoes what it started: nothing stays cloud-controllable
after the API said the mount failed.

## Certificates: one per robot, one policy for the fleet

```python
from strands_robots.mesh.iot import provision
iot = boto3.client("iot", region_name="us-west-2")
provision.setup_tenant(iot, "lab-a", region="us-west-2", account="123456789012")   # thing group + fleet policy, once
out = provision.register(iot, record, "~/.strands/iot")   # thing = device id, type = device type, cert from a local CSR
Mesh(f"iot://us-west-2?{out['query']}", tenant="lab-a")
```

The thing *is* the device id; the thing type is the device type; the tenant is a thing group and a thing attribute.
There is one policy for every robot, 1.9 KB, written with policy variables (`${iot:Connection.Thing.ThingName}`,
`${iot:Connection.Thing.Attributes[tenant]}`): a robot may connect only as itself, publish only on its own keys and
its own presence, and read only inside its tenant. Measured: a publish on another robot's presence is refused at the
broker (PUBACK 135) and the link stays up. The key is generated on the robot and only a CSR with `CN=<device id>`
travels, which is also what makes the HTTPS path identify the robot. `rotate` issues a new certificate while the old
one still works, `retire` removes the old, `revoke` cuts a robot off at its next connect; a connected session stays up until it drops, so `revoke(..., mesh=)` e-stops it first.

A robot without a certificate yet gets one on first mount through fleet provisioning by claim (`claim.claim`) with a
template and a pre-provisioning hook shipped as data (`claim.TEMPLATE`, `claim.HOOK_SOURCE`): the hook accepts a
request only when the CSR's CN is the thing being created and that thing does not exist, so a claim holder can never
mint a certificate for another robot (measured: a forged CN is denied in 0.55 s; the honest path takes about 5 s).
Operators keep IAM (`provision.operator_policy`): they may send rpcs, e-stop, write shadow targets and read
everything, and may not speak as a device. Its shadow grants name the tenant's own device ids (`things=`, e.g. from `registry.list_things`), because a shadow document carries no author: an account-wide shadow grant would let an operator of one tenant move another tenant's robot, so an empty `things` is refused and `all_things=True` is the explicit, cross-tenant opt-in for a single-tenant account. The grant is `thing/<id>/control` for writes and `thing/<id>/*` for reads, the way IAM names a named shadow; about nine devices fit IAM's 6,144-character cap, so a larger tenant attaches several documents. A box that holds a certificate *and* commands other robots (a controller) gets the small `controller` add-on policy on top (`register(..., controller=True)`); without it
the broker refuses its rpcs, which is the point.

### Keys in a hardware token (PKCS#11)

By default the private key is a PEM file on the robot (`key.pem`, mode 0600). A device with a hardware security
module (or SoftHSM in a test) keeps the key in the token instead, so it never exists as a file. Point the link's
`key=` at an RFC 7512 `pkcs11:` URI and the mesh signs the TLS handshake inside the token
(`mtls_with_pkcs11`); `module-path=` names the driver `.so` (or the `PKCS11_MODULE` env var). The user PIN comes only
from the `PKCS11_PIN` env var: a link is printed by the mesh's repr, its errors and the agent tool, so a `key=`
carrying `pin-value=` is refused, and so is `register` given one. The URI's own `?` and `&` would split the link's query, so `key=` carries it
percent-encoded, as `register` returns it; a bare one is refused:

<!-- test: skip, needs a PKCS#11 token (a hardware HSM or SoftHSM) -->
```python
from urllib.parse import quote

key = "pkcs11:token=lab-a;object=arm-1?module-path=/usr/lib/softhsm/libsofthsm2.so"  # PIN: export PKCS11_PIN
Mesh(f"iot://us-west-2?client=arm-1&cert=~/.strands/iot/arm-1/cert.pem&key={quote(key, safe='')}", tenant="lab-a")
```

To provision without ever writing the key, generate it in the token and hand `register` the matching CSR and the
token URI: only `cert.pem` is written, and `out["query"]` carries the `pkcs11:` key.

<!-- test: skip, needs a PKCS#11 token (a hardware HSM or SoftHSM) -->
```python
out = provision.register(iot, record, "~/.strands/iot", csr_pem=hsm_csr, key=key)   # no key.pem on disk
rot = provision.rotate(iot, "arm-1", "~/.strands/iot", csr_pem=new_hsm_csr, key=key)  # same token key URI
```

`rotate` on such an identity without `csr_pem` is refused, so it never writes a software key over the token's.

## Commands: two paths, one measured difference of 4 ms

An operator with IAM sends an rpc as a **Direct Message**: one HTTPS call that returns 404 in 0.08 s when the robot
is gone, which is exactly what an e-stop wants (nothing is queued, nothing replays). A certificate-only link has no
IAM, so it sends rpcs as **MQTT request/response** (`response_topic` + `correlation_data`, and the same addressing in
the body for brokers without MQTT5). Measured on the same robot in one run: 190 ms vs 198 ms for `describe`; an
offline robot is silence on the MQTT path, and the 3 s presence heartbeat is what says why. `iot://…?rpc=direct|mqtt`
overrides the default. The robot answers only on its own tenant's reply keys.

The heartbeat stays because IoT's lifecycle events cannot replace it: the broker answers every keep-alive request
with 1,200 s, so a frozen or partitioned robot stays "connected" for up to 30 minutes, while the heartbeat finds it
gone in 3 s. Lifecycle events are still worth having as an audit trail (`lifecycle.watch`, `lifecycle.Audit`: who
connected and dropped, when, and why — 25 ms after a connect, 115 ms after a socket close).

## Rules: events as topics

```python
from strands_robots.mesh.iot import rules
rules.apply_rules(iot, rules.rules_for("lab-a", role_arn, sns_topic_arn=alerts, s3_bucket="lab-recordings"))
```

One rule per event kind per tenant, applied idempotently: every tenant e-stop republished on `…/@estop/latest` (and
to SNS), device events (`fault`, `estop`, `run.started`, `run.stopped`) into Timestream or S3, high-severity faults
to SNS. Measured: device → rule → republished copy at a subscriber p50 164 ms, into SNS→SQS p50 90 ms. The library
never creates IAM — pass the role, and make it one that already exists: a rule whose role was created seconds
before drops messages silently until IAM propagates. Presence and the live state never go through rules by default
(3,600–108,000 triggers per robot per hour); `IotOptions(telemetry_hz=1)` opts a robot into Basic Ingest, which
carries no messaging charge and no subscriber ever sees.

## Setup checklist

1. `pip install 'strands-robots[iot]'` (awsiotsdk, boto3, cryptography).
2. `setup_tenant` once per tenant, `register` (or `claim`) once per robot; the files land in `~/.strands/iot/<id>/`.
3. Operators: attach `operator_policy(tenant, things=[...])` (the tenant's device ids) to their IAM identity; `iot:RetainPublish` is what lets a tenant e-stop be retained.
4. Rules: one long-lived role with `iot:Publish` (republish), `sns:Publish`, `s3:PutObject`/`timestream:WriteRecords` as used.

## At the edge: Greengrass v2 and Kinesis Video Streams

{{drawing:d19_edge}}

On a core device the mesh runs as a Greengrass v2 component. `greengrass.recipe()` turns the process that mounts
robots into `com.strands.robots`; `greengrass.deployment()` targets the tenant's thing group (the fleet) and
`greengrass.deploy()` is the fleet update. The component's `main()` mounts the configured robots on one `Mesh` and
stays up until SIGTERM. Robot to robot traffic stays on zenoh, so the mesh does not depend on the cloud link
(measured on an ec2 L40S box: invoke `act` p50 0.78 ms with the cloud link cut for 45 s; a local deployment reaches
RUNNING in 11.4 s, a cloud deployment of five components in 48 s).

```python
<!-- test: skip, needs a provisioned Greengrass core and an S3 artifact -->
from strands_robots.mesh.iot import greengrass

r = greengrass.recipe(
    "1.0.0",
    "s3://bucket/strands_robots-1.0.0-py3-none-any.whl",
    tenant="lab",
    robots=[{"name": "so101", "device_id": "arm-1"}],
)
greengrass.deploy(gg, "arn:aws:iot:us-west-2:123456789012:thinggroup/strands-lab", "1.0.0")
```

Each camera sensor becomes one KVS stream and one WebRTC signaling channel named
`strands-<tenant>-<device>-<camera>`. `kvs.camera_entries()` writes them into the manifest shadow so a dashboard
finds the live view of a camera it has never met, and `kvs.publish_cameras()` feeds each camera into `kvssink` on
its own thread through the measured GStreamer `pipeline()` (640 by 480 at 15 fps, `nvh264enc` fragment-head to
consumer p50 36 ms). There is no `PutMedia` in boto3 and a hand-rolled chunked POST has no ACK or backpressure, so
without the `kvssink` plugin `publish_cameras()` refuses by name rather than pretend. `import strands_robots.mesh`
still loads no boto3, awsiot or gi.
