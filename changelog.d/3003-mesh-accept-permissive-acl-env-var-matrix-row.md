### Docs: `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is named on the README matrix and the security page

`STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` is the acknowledgement token for a
permissive mesh ACL posture.  It has one spelling and three readers, and
setting it takes all three effects:

1. `strands_robots.mesh._acl_config._load_acl_file` raises `PermissiveACLError`
   on ACL load when `STRANDS_MESH_ACL_FILE` points at a blacklist-shaped file
   (`default_permission: "allow"` with a non-empty `rules` list) unless the
   variable is set to `1`/`true`/`yes`.
2. `strands_robots.mesh.core.Mesh._refuse_under_permissive_default_acl` takes
   it as the operator opt-in for the built-in permissive default: with the
   token unset, `Mesh.start` refuses to bring the wire up at all, and its
   refusal names the variable as one of the remediations.
3. `strands_robots.mesh.session._build_config` skips the per-session-open
   `PERMISSIVE built-in default ACL` WARNING while it is set.

Before this change: the README env-var matrix named 32 other `STRANDS_MESH_*`
variables and 0 rows for this one, `docs/security.md` named the variable in
one bullet as a silencer for the session warning and said nothing about the
loader refusal, and the `_zenoh_config` module docstring asserted the token
"does NOT silence the per-session permissive WARNING" -- which reader 3
contradicts.  So an operator tracing the refusal from `_acl_config.py` found
the variable, and an operator reading either documentation surface got a
blast radius narrower than the implemented one, in the dangerous direction: a
fleet that sets the token to load a blacklist ACL has also pre-acknowledged
the built-in default, so a later loss of `STRANDS_MESH_ACL_FILE` from that
environment brings the wire up permissive with the start gate waived and the
recurring warning suppressed.

Fix: one README matrix row beside `STRANDS_MESH_ACL_FILE` naming all three
effects; one `### Blacklist ACL acknowledgement` subsection in
`docs/security.md` between the namespace and policy-vocabulary sections,
naming the three effects, distinguishing the two ACL shapes (`allow` + rules
is blacklist, `deny` + rules is whitelist) and attributing the refusal to
`_load_acl_file` rather than `_validate_acl_shape`, which grades the schema at
the end of the same loader and does not raise; the same correction to the
`_zenoh_config` module docstring, so one variable does not carry two
descriptions in one tree; and one derived guard mirroring the shape of the
sibling `test_docs_mesh_*_env_var_reference.py` guards (audit / TLS /
namespace / policy-type-allow).  The guard derives its population from the
package's own `os.getenv` literals -- across every module, not one -- so a
sibling variable *or* a fourth reader of this one is held to the same
documentation rule the hour it lands, and it pins effects 2 and 3
behaviourally so the prose cannot drift from the gates.

Continues the four-shipment sequence from fires 131/138 (`STRANDS_MESH_TLS_*`
#2945, `STRANDS_MESH_AUDIT_*` #2979, `STRANDS_MESH_NAMESPACE` #2985,
`STRANDS_MESH_POLICY_TYPE_ALLOW` #2990).  Refs harness#376.
