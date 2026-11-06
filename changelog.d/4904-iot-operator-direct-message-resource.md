### Fixed: an IAM operator on the `iot://` mesh discovers and commands devices under the generated policy

`mesh.iot.policy.operator_policy` granted `iot:SendDirectMessage` on `connection/*`, but AWS names the target
client as that action's resource, so every manifest request was refused (403) and `discover()` returned `[]`
while the device's presence heartbeats arrived. The grant is now `client/*`, kept inside the tenant by an
`iot:Topic` condition on `strands/<tenant>/*`. A refused manifest request now logs a warning naming the device,
and an IAM link defaults its client id to `strands-op-<hex>`, the prefix the same policy admits on `iot:Connect`.
