### Fixed: the refusal-code guard is scoped by what can carry a code, not by the grants already registered

A continuable refusal carries a stable `code` so a consumer classifies it on
identity rather than on prose, and one test exists to make sure a new one does not
arrive without a code. That test looked for `raise` sites whose message names a
grant from `REFUSAL_GRANTS`, so it could only ever find a refusal naming a grant
that is already coded. The first refusal to offer a *new* grant -- the one case the
rule exists for -- was invisible, and both the module and the test file claimed
otherwise.

Measured: a `ValidationError` refusing a value and offering "Set
`STRANDS_MESH_SINK_ALLOW` to extend" leaves that file at 21 passed. The scan cannot
see it, because the variable is absent from the grant table until someone adds the
code, which is exactly the thing being forgotten.

The scope now comes from the exception types that can carry a code at all -- a class
whose `__init__` accepts one, plus anything inheriting it, derived from the package
rather than listed. Within those, a message naming any `STRANDS_*` variable reads as
an offer to the operator and must carry a code. The same planted refusal is now
reported by site and by grant.

No shipped refusal needed a code: of 1195 `raise` sites, 72 are of a code-carrying
type, 6 of those name a variable, and all 6 already carry one -- so this is a scope
fix with no behaviour change. A builtin `ValueError` has no `code` parameter, so the
permissive-ACL, unacknowledged-`AUTH_MODE=none` and CA-pin refusals stay out of
scope by structure rather than by exemption.
