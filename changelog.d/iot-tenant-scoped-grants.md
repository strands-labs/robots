### Fixed: the AWS IoT policies keep every grant inside the tenant, and a robot can no longer beat for an operator

- **Fleet policy:**
  - a device can no longer publish `@peer` liveliness tokens. A device could keep a dead operator "alive" for every
    watchdog in the tenant, or fake its goodbye. The controller policy now carries that grant for certificates that
    command.
  - `RetainPublish` is limited to the device's own keys and shadow.
  - telemetry goes only to the Basic Ingest rule named by the thing's new `telemetry_rule` attribute, instead of any
    rule in the account.
- **Provisioning:** `provision.register` writes `telemetry_rule`, and a label can no longer override `tenant` (or
  `telemetry_rule`). The claim template writes it for letter-and-digit tenant ids.
- **Operator policy:**
  - operators connect only as `strands-op-<tenant>-*`, so they no longer collide with another tenant's operators;
  - Direct Messages go only to the tenant's own devices, on `strands/<tenant>/*/rpc/*`;
  - lifecycle events are heard only for those devices.
  - One document now fits about six devices (ten ARNs each).
- **Client ids:** an IAM `iot://` link of a `Mesh` connects as `strands-op-<tenant>-<random>` and refuses a
  `client=` outside that prefix.
- **Rpcs:** a device answers rpcs only on its own tenant's topics.
- **Disconnect audit rule:** `lifecycle.clear_presence_rule` now needs `things=` and republishes only the tenant's
  own devices' and operators' disconnects.

Re-provision the fleet and operator policies to apply this.
