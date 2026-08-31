### Docs: the audit-log family's tamper-evidence and rotation knobs are named on the README matrix

`STRANDS_MESH_AUDIT_DIR`, `STRANDS_MESH_AUDIT_PSK`, `STRANDS_MESH_AUDIT_MAX_BYTES`
and `STRANDS_MESH_AUDIT_MAX_FILES` are the four sibling variables that configure
one channel -- where the mesh writes its append-only JSONL audit log, whether
each record carries a per-record HMAC that lets a downstream verifier reject a
forged entry, and the per-file / rotation bounds that keep disk use finite. The
security page documents them together under one heading. The README
environment-variable matrix carried a row for `_DIR` alone and had no row for
the other three, so a reader scanning the matrix for the audit family found
one of four sibling variables and not the tamper-evidence gate the security
page calls "the deliberate posture" or the rotation bounds the same page pairs
it with.

`_PSK` is the more consequential of the missing three. Without it, the `sig`
field is absent from each record and `verify_audit_integrity` trusts the file;
a writer with access to `STRANDS_MESH_AUDIT_DIR` can edit a record and leave
no HMAC to fail against, and no downstream check will catch the edit. It is
the pair a production posture must set beside `_DIR`, and the matrix that names
one and not the other undersells it.

`_MAX_BYTES` and `_MAX_FILES` are the disk-use bound. Both clamp to a hard
upper cap with a warning, both fall back to the default on unparsable / zero /
negative values with a warning (so a misconfiguration cannot silently disable
rotation), and their product bounds total disk use for the log. An operator
sizing storage for the audit trail needs their defaults and their caps, and
the matrix is where the sizing question is asked.

The three new rows sit beside the existing `_DIR` row, in matrix order, so the
family is discovered together. `docs/security.md` already describes all four
under `## Audit log` and needs no change; this closes the README half.

A guard derives its graded population from `mesh/audit.py`'s own `os.getenv`
literals, so a fifth `STRANDS_MESH_AUDIT_*` path added later is held to the
same matrix rule the hour it lands. Four properties are graded: every audit
variable the module reads has a matrix row (which is what makes the matrix a
single index for the family), every one is named on the security page, the
matrix rows sit contiguously (so a reader lands on all four without hunting),
and the derivation is not empty (so the whole rule cannot pass vacuously if
the scan ever goes blind).
