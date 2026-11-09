### Changed: an IoT device can no longer be named like an operator or claim client id

A robot connects to AWS IoT with its thing name as its MQTT client id, and IoT drops the older connection when two
clients use the same id. A thing named `strands-op-<tenant>-…` could therefore knock that tenant's operator off,
and one named `strands-claim-…` a robot in the middle of its claim. `mesh.iot.policy.check_device_id` now refuses both
prefixes, and `provision.register` (also with an HSM CSR, which skipped the name check before), `claim_hook.allow`,
the claim hook Lambda (`claim_hook.HOOK_SOURCE`; redeploy it), `claim.claim`, `policy.operator_policy(things=)` and
`lifecycle.clear_presence_rule(things=)` call it. A robot already registered under such a name keeps working until it
is registered again or listed in an operator policy; rename it with `provision.teardown` + `provision.register`.

`policy.operator_policy(..., all_things=True)` now warns (`UserWarning`) that it grants every thing in the account,
and refuses to be combined with `things=`. `docs/iot.md` documents it as the single-tenant-account option.
