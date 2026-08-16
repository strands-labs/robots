### Docs: `STRANDS_MESH` is documented as the opt-in it actually is

Three configuration tables printed `true` as the default for `STRANDS_MESH` and
offered `false` as the only knob. The factory resolves `mesh=None` from the
environment, so unset means mesh off and only `true`/`1`/`yes` turns it on: the
printed default named a state a bare `Robot()` is never in, the offered remedy
was a no-op for the state readers were actually in, and the one spelling that
enables the mesh appeared nowhere. The rows now state the real contract, the
module that resolves the variable documents it, and a guard grades the tables
against the resolver so the two cannot drift apart again.
