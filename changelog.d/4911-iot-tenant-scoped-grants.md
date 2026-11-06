### Fixed: the AWS IoT policies keep every grant inside the tenant, and a robot can no longer beat for an operator

- **Fleet policy:**
  - a device can no longer publish `@peer` liveliness tokens. A device could keep a dead operator "alive" for every
    watchdog in the tenant, or fake its goodbye. The controller policy now carries that grant for certificates that
    command. A certificate `iot://` link therefore sends no `@peer` beat, so the broker never sees a publish the
    policy refuses; `?peer=1` turns it back on for a certificate that also carries the controller policy.
  - `RetainPublish` is limited to the device's own keys and shadow.
  - telemetry goes only to the Basic Ingest rule named by the thing's new `telemetry_rule` attribute, instead of any
    rule in the account.
- **Provisioning:** `provision.register` writes `telemetry_rule`, and a label can no longer override `tenant` (or
  `telemetry_rule`). A claimed robot gets it from its approval (`claim_hook.allow`), for any tenant id.
- **Operator policy:**
  - operators connect only as `strands-op-<tenant>-*`, so they no longer collide with another tenant's operators. A
    tenant other than letters and digits is hashed there as in rule names (`lab-a` → `strands-op-lab_a_<12 hex>-`), so
    `lab` and `lab-a` never share a prefix;
  - Direct Messages go only to the tenant's own devices, on `strands/<tenant>/*/rpc/*`;
  - lifecycle events are delivered only for those devices: `iot:Subscribe` stays wide (`lifecycle.watch` subscribes
    with `+`, which is literal in a policy) and `iot:Receive` names each device.
  - One document now fits about seven devices (nine ARNs each).
- **Client ids:** an IAM `iot://` link of a `Mesh` connects as `strands-op-<tenant>-<random>` and refuses a
  `client=` outside that prefix.
- **Rpcs:** a device answers rpcs only on its own tenant's topics.
- **Disconnect audit rule:** `lifecycle.clear_presence_rule` now needs `things=` and republishes only the tenant's
  own devices' and operators' disconnects.

Re-provision the fleet and operator policies to apply this, and run `register` again for robots registered earlier:
until then they keep presence and replies but send no telemetry. A claim approval made with an earlier `allow` has no
`telemetry_rule` and is refused: approve that robot again.
