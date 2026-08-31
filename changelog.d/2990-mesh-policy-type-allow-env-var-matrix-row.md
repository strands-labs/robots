### Docs: `STRANDS_MESH_POLICY_TYPE_ALLOW` is named on the README matrix and the security page

`STRANDS_MESH_POLICY_TYPE_ALLOW` is the operator-facing extension knob for the
mesh `execute` / `start` policy vocabulary allowlist that
`validate_command` enforces: comma-separated extras appended to the built-in
`_DEFAULT_POLICY_TYPES` set (the union of `_LEROBOT_POLICY_FAMILIES` and
`_REGISTRY_POLICY_PROVIDERS`). It widens both `policy_type` and
`policy_provider` at once because the two vocabularies share one allowlist by
design. A payload whose `policy_type` or `policy_provider` is not in the
widened union is refused on the mesh path with
`refusal_codes.POLICY_TYPE_NOT_ALLOWED`; the refusal message names this
variable as the recourse. `mesh/security.py` references the variable ten
times — one refusal code, one charset comment, two class docstrings on the
built-in list, one loader, one cache key and two `ValidationError` messages —
so the module source names it well, but until this change the two
documentation surfaces the module points operators at (the README env-var
matrix and `docs/security.md`) did not name it at all.

The failure mode makes the omission worse than an ordinary one. The refusal
message the mesh emits says 'Set STRANDS_MESH_POLICY_TYPE_ALLOW to extend',
but the two operator-facing pages an operator scans for the knob do not name
it. An operator who reads the module source to trace the refusal message
finds the variable; an operator who works from the documentation cannot
extend the allowlist because the extension knob is unnamed. The 35 sibling
`STRANDS_MESH_*` rows already on the matrix set the expectation that this
one would be there too.

The change adds one README matrix row (beside `STRANDS_MESH_POLICY_HOST_ALLOW`,
in policy-family order), one `### Policy vocabulary allowlist (policy_type /
policy_provider)` section on `docs/security.md` (between the namespace
section and the AWS IoT section, where an operator provisioning the mesh
finds it), and a guard test that derives its graded population from
`mesh/security.py`'s own `os.getenv` literals. Eight rules plus a
keep-the-derivation-honest premise:

- every policy-type-allowlist variable the module reads has a README matrix row,
- every one is named on the security page,
- the security page names the shared-allowlist invariant (widening admits both `policy_type` and `policy_provider` at once),
- the security page names the charset rule (`^[a-z][a-z0-9_]*$`) so a caller whose entry silently drops has a documented reason for the drop,
- the security page warns against using the variable to route around a `_REGISTRY_POLICY_PROVIDERS` omission (the sync-guard for that lives elsewhere and this variable would bypass it),
- and three behavioural pins tie the prose to the loader: nothing set resolves to `_DEFAULT_POLICY_TYPES` verbatim, an empty / whitespace value falls back to the default (so a shell script that forgot the value does not accidentally refuse every payload), and a well-formed extra is admitted verbatim under the lowercasing normaliser while a malformed entry is not admitted (dropped with a WARNING).

Follows the same guard shape as `STRANDS_MESH_AUDIT_*` (strands-labs/robots#2979),
`STRANDS_MESH_TLS_*` (strands-labs/robots#2945) and `STRANDS_MESH_NAMESPACE`
(strands-labs/robots#2985), completing the four grader-named lanes on
cagataycali/robots-harness#376's env-var drift.
