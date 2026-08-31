### Docs: the three TLS material paths the default mesh posture requires are named

`STRANDS_MESH_AUTH_MODE` defaults to `mtls`, and `STRANDS_MESH_TLS_CA`,
`STRANDS_MESH_TLS_CERT` and `STRANDS_MESH_TLS_KEY` are the whole of that
posture's configuration. With any one of them unset, `_resolve_tls_paths`
raises `ValueError` naming all three and the session never opens -- so on a
fleet that sets neither dev flag, those three paths are the difference between
a mesh that comes up and one that does not. None of the three was named on any
documentation surface.

Two things made that worse than an ordinary omission. `docs/security.md`
documents the *optional* AWS IoT transport's credential family exhaustively --
four variables, each with its required/optional status, its default and its
failure posture -- and named none of the three the *default* transport
requires; its production-posture section names the ACL half by variable and
calls mTLS the thing that ACL is paired with, while naming no part of the mTLS
half. Separately, `mesh/_zenoh_config.py` cites the README environment-variable
matrix three times as the surface that promises the private-key mode contract,
once inside a `WARNING` an operator reads on a non-POSIX host -- and that
matrix, which carries rows for 32 other `STRANDS_MESH_*` variables, had no row
for any TLS path. The warning pointed a reader at a table where the variable it
warns about did not appear.

The three matrix rows and a security-page section beside the IoT credential
family close both. A guard derives the graded population from the config
module's own environment reads, so a fourth `STRANDS_MESH_TLS_*` path is held
to the same rule the hour it lands, and grades four properties: each path is
named on the security page, each carries a README matrix row (which is what
makes the module's own pointer resolve), the three sit together under one
heading, and the section describes the refusal rather than a downgrade to plain
TCP. The behaviour the prose exists to make discoverable is pinned too -- the
default posture resolves to `mtls` and refuses naming all three, and the same
call succeeds once the material is supplied -- so the page cannot drift away
from what the loader does.
