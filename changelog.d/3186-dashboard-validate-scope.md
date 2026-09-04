### Fixed: a policy preflight reports what it actually verified, not just "ok"

`POST /api/policies/validate` answered `{"ok": true, "stage": "preflight"}` for
`lerobot_local` with an empty config, and the run form rendered that as a green
"lerobot_local resolves" -- a preflight that never saw a model reporting that a
model resolves. `strands_robots.dashboard.validate_scope.validation_scope`
inspects the submitted config against the provider's declared keys and reports
which facts the preflight could actually check: whether an identity key naming
the model was set (`pretrained`, `checkpoint`, `model_path`, `ckpt`, `weights`)
and whether a remote was named (`host`, `port`, `server_address`, `url`,
`endpoint`, which a preflight can only confirm are set, never that the far end
holds anything). The caller renders that scope instead of a bare verdict, so an
empty config no longer reads as a resolved model.
