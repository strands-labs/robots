### Docs: `STRANDS_MESH_NAMESPACE` is named on the README matrix and the security page

`STRANDS_MESH_NAMESPACE` is the Zenoh `namespace` field on every peer of a
fleet. It prefixes every mesh key-expression the tree emits, and Zenoh only
routes application traffic between peers whose namespaces match -- so it is
the knob that keeps a test rig from receiving a production fleet's commands on
a shared LAN. `mesh/_zenoh_config.py` reads it (`resolve_namespace`), the
`DEFAULT_NAMESPACE = "strands"` fallback tracks the hardcoded topic prefixes
every mesh component emits, and the built-in ACL key_exprs are keyed against
that same default. Until this change, the variable was named on no
documentation surface: neither the README environment-variable matrix (which
carried 34 other `STRANDS_MESH_*` rows) nor `docs/security.md`.

The failure mode makes the omission worse than an ordinary one. A namespace
mismatch does not raise. Two peers with different namespaces connect at the
transport layer (so TLS handshakes succeed, and neither peer refuses the other
loudly) and then exchange no application traffic, because their
key-expressions never match. So a peer that is misconfigured against a fleet
appears absent rather than crashing, and an operator scanning for a peer they
cannot see cannot deduce a namespace mismatch from any error message. The
diagnostic hook has to live in the documentation, and it did not.

The change adds one README matrix row (beside `STRANDS_MESH_MULTICAST`, in
transport-family order), one `### Fleet routing isolation (namespace)` section
on `docs/security.md` (between the mTLS section and the AWS IoT section, where
an operator provisioning fleet identity finds it), and a guard test that
derives its graded population from `mesh/_zenoh_config.py`'s own `os.getenv`
literals. Four rules: every namespace variable the module reads has a matrix
row and a security-page name (so a sibling like a per-topic namespace override
is graded on arrival), the documentation names the default value (so an
operator overriding fleet isolation knows what they are diverging from), and
the documentation names the silent-mismatch failure mode (so the operator has
the diagnostic hook the code cannot provide). Three behavioural cells pin the
prose to the loader: the default with nothing set is `"strands"`, an empty /
whitespace value falls back to the default rather than producing an empty
prefix (the loader guards against `"//presence"`-style keys), and a non-empty
value is honoured verbatim.
