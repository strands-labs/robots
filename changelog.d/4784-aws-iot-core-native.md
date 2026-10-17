### Added: AWS IoT Core, natively

`Mesh.mount(robot, iot=IotOptions(...))` makes a device a proper IoT thing on an `iot://` mesh: a `manifest` named
shadow written once, a `state` shadow at most 1 Hz with the lockout, and a `control` shadow whose desired targets go
through `confirm_act` and the safety chain (a refusal comes back verbatim in `reported`; `estop` is never a shadow
field). `mesh.iot.rules` applies one rule per event kind from the manifest, idempotently; `mesh.iot.provision`
registers a thing per device with one fleet policy and a certificate from a local CSR with `CN=<device id>`, and
rotates, retires and revokes it; `mesh.iot.claim` provisions by claim with a hook that refuses a forged CN. A
certificate-only link sends rpcs as MQTT5 request/response (`?rpc=mqtt`). The `[iot]` extra adds `cryptography`.
