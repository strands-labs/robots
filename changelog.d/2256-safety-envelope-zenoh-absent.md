### Quality: pin the safety-envelope publish path on an install without zenoh

`Mesh` publishes e-stop and resume envelopes through four helpers that each
degrade to the transport-agnostic `put()` path rather than raising. Between them
they refuse for nineteen reasons, and eighteen were pinned; the one that was not
is the import itself failing, which is the ordinary state of an install without
the `mesh` extra. All four `except ImportError` arms were unexecuted, and all
four docstrings enumerated their fallback reasons without naming that one.

Two of the arms carry a fleet-availability contract: `_safety_wire_zid` must
answer `None` so an issuer binds the resume override proof to the zid-less body
the fallback actually publishes (a proof bound to a zid the body does not carry
never verifies, leaving the fleet stuck in lockout), and
`_publish_safety_envelope` must still publish, with `source_zid` stripped (an
unstripped body with no wire `SourceInfo` behind it is hard-rejected by the
receiver, so an unstripped e-stop is a dropped e-stop). Both are now driven end
to end, including a receiver clearing its lockout from the fallback envelope.

The other two arms reach for an in-package module that `core` already imports at
module scope, so they cannot fire in a complete install. They are driven anyway,
and the property that makes them unreachable is pinned, so they acquire a
behavioural test the day that changes.

The four enumerations now name the import-failure reason. No production
behaviour changes: the docstring-stripped AST of `mesh/core.py` is identical.
