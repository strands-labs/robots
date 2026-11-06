### Fixed: fleet provisioning by claim registers only robots the operator approved, in their tenant, once

`claim.create_template` now requires `hook_arn=`. When the template already exists, it also attaches the hook,
role and `enabled` with `update_provisioning_template`. Before, a template first created without a hook stayed
hookless on every later run while reporting success.

The hook (now `claim_hook.HOOK_SOURCE`) still requires a new thing whose certificate CN is its name. It now also
requires an operator approval for that name:
- `claim_hook.allow(ssm, device_id, tenant=, device_type=)` writes an SSM SecureString under
  `/strands-robots/claim/` and returns a one-time secret;
- the request must carry that secret, tenant and device type;
- the hook overrides the request with the approved values;
- it deletes the approval, so one approval is good for one robot.

Before, anyone holding the shared claim certificate could register any new name, in any tenant, including the name
of a robot that had not booted yet. `claim.claim` now takes `secret=`. The hook's role also needs
`ssm:GetParameter` and `ssm:DeleteParameter` on `/strands-robots/claim/*`.
