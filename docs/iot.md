---
description: Run the mesh over AWS IoT Core. Each robot becomes an IoT thing with named shadows, its own certificate, rules, Greengrass and Kinesis Video Streams.
---

# AWS IoT Core

`Mesh("iot://us-west-2")` runs the mesh over AWS IoT Core. Discover, invoke, subscribe and e-stop work as they do
on a LAN. A command takes about 200 ms.

{{drawing:d08_iot_core}}

<!-- test: skip, needs AWS IoT credentials and device certificates -->
```python
from strands_robots import Robot
from strands_robots.mesh import Mesh
from strands_robots.mesh.iot import IotOptions

link = "iot://us-west-2?client=arm-1&cert=~/.strands/iot/arm-1/cert.pem&key=~/.strands/iot/arm-1/key.pem"
cloud = Mesh(link, tenant="lab-a")
cloud.mount(Robot("so101"), device_id="arm-1", iot=IotOptions(shadow_hz=0.5))
```

With `iot=`, the robot also becomes an IoT thing with shadows.

## Setup

1. Install the extra: `uv pip install "strands-robots[iot] @ git+https://github.com/strands-labs/robots@main"`.
2. Run `provision.setup_tenant` once per tenant.
3. Run `provision.register` once per robot. Files go in `~/.strands/iot/<id>/`.
4. Attach `policy.operator_policy(tenant, things=[...])` to each operator's IAM identity.

```python
from strands_robots.mesh.iot import provision

provision.setup_tenant(iot, "lab-a", region="us-west-2", account="123456789012")
out = provision.register(iot, record, "~/.strands/iot")
Mesh(f"iot://us-west-2?{out['query']}", tenant="lab-a")
```

Each robot gets its own certificate. The key is made on the robot and never leaves it. One fleet policy lets a
robot connect only as itself and read only inside its tenant.

| Function | Does |
|---|---|
| `provision.rotate` | Issue a new certificate while the old one works |
| `provision.retire` | Remove the old certificate |
| `provision.revoke` | Cut a robot off. Pass `mesh=` to e-stop it first. |
| `claim.claim` | Get a first certificate through fleet provisioning |

To keep the key in a hardware token, pass a percent-encoded `pkcs11:` URI as `key=`. Set the PIN in `PKCS11_PIN`.

## Shadows

| Shadow | Holds |
|---|---|
| `manifest` | The robot's controls and ranges, written at mount |
| `state` | The live state at `shadow_hz` (at most 1 Hz) and who stopped it |
| `control` | Targets you write. They pass the safety chain. |

```python
data = boto3.client("iot-data", region_name="us-west-2")
desired = {"state": {"desired": {"shoulder_pan.pos": 0.3}}}
data.update_thing_shadow(thingName="arm-1", shadowName="control", payload=json.dumps(desired).encode())
```

A refused target appears under `reported.refused`. A target older than `stale_s` (default 2 s) is dropped. Use the
`estop` rpc or `broadcast("estop")` to stop a robot, not a shadow.

## Rules

```python
from strands_robots.mesh.iot import rules

rules.apply_rules(iot, rules.rules_for("lab-a", role_arn, sns_topic_arn=alerts, s3_bucket="lab-recordings"))
```

Rules forward e-stops, faults and run events to SNS, S3 or Timestream. Pass a role that already exists. A new
role can drop messages until IAM catches up.

A stored record is filed under the rule's own tenant and the publishing thing, never under a `tenant` or
`device_id` the message body names: S3 keys are `recordings/<tenant>/<device>/<event>/<ms>.json`. When several
tenants share one bucket, give each tenant's rule role `s3:PutObject` on `recordings/<tenant>/*` only; give each
tenant its own Timestream table.

## Greengrass and video

{{drawing:d19_edge}}

`greengrass.recipe()` packages the mesh as a Greengrass v2 component. `greengrass.deploy()` updates the fleet.
Robots on one site still talk over Zenoh if the cloud link drops.

`kvs.publish_cameras()` streams each camera to Kinesis Video Streams. It needs the GStreamer `kvssink` plugin.

## Limits

| AWS IoT quota | Effect |
|---|---|
| 128 KB MQTT payload | Camera images go over Kinesis Video Streams |
| 20 shadow requests per second per thing | The `state` shadow is at most 1 Hz |
| 8 KB shadow document | The `manifest` shadow is compact |
| 1,200 s keep-alive | A 3 s heartbeat detects offline robots |
