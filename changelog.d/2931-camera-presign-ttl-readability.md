### Fixed: a keyword presign TTL is read before it is clamped

`CameraOffloader` resolves the lifetime of the presigned GET URL it hands out
per camera frame from two places, and they disagreed. The environment path runs
`int(raw)` on a string and falls back to the default when that raises, so
`STRANDS_MESH_CAMERA_PRESIGN_TTL` of `2.5`, `nan` or `inf` each resolve to the
default. The `presign_ttl=` keyword had no such step: it went straight to two
comparisons against the floor and the ceiling, and a comparison is permeable to
anything that compares false against both bounds.

`nan` is the value that matters, because the bound it walked through is a
security bound -- the module's own comment says the ceiling exists "to prevent
accidental day- or week-long URLs". `botocore` does not read `ExpiresIn`
either, and what it does instead depends on the signer. Its pure-python signer
interpolates the value verbatim, so the presigned URL carried
`X-Amz-Expires=nan` -- a URL AWS refuses at request time, so the frame is
unreadable *and* the window was never bounded. The `awscrt` signer that the
`mesh-iot` extra installs, which is therefore the shipped configuration, instead
fails inside the signer with a bare `AssertionError` naming no TTL, losing the
frame outright. The `/ref` message published beside the URL carried
`expires_at: nan` either way.

Three more spellings resolved to something the caller never named: a fractional
TTL signed `X-Amz-Expires=2.5` on the pure-python signer and raises `TypeError`
from inside the CRT one, `True` stored a silent one-second TTL, and `inf`
tripped the ceiling but its notice renders the value with `%d`, so `logging`
raised and the operator saw an error where the clamp notice belonged.

The keyword now goes through a readability check first. Only readability is
decided there -- the range is still the clamps' to decide -- so `0` remains the
documented keyword-versus-environment precedence sentinel, `-99` remains a
call-site bug that clamps to `1` with a warning, and a value above the ceiling
still clamps to it. An integral float is accepted rather than refused, because
that is how `json.dumps` renders an integer held in a float.
