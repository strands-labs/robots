### Added: operator dashboard auth rail, with the extra that supplies it

`strands_robots/dashboard/auth.py` is the operator authentication rail for the
web dashboard: WebAuthn passkey registration and login ceremonies, an RP-ID
verdict that separates a loopback host from a LAN one, challenge caps enforced
per process and per client IP, and a JWT session token with a renewal window.
`strands_robots/dashboard/__init__.py` carries the package's dependency gate, so
one call covers every module added to the package later, and `[dashboard]` is the
extra that supplies them.

The rail signs its session token with PyJWT, which was declared in no extra and
in no core dependency - the only copy in a developer environment arrives
transitively through an unrelated package, which is why the gap survived local
runs. `pip install 'strands-robots[dashboard]'` therefore imported the package
fine, because the gate named `fastapi`, `uvicorn` and `webauthn`, and then failed
on the one module PyJWT backs with a bare `ModuleNotFoundError: No module named
'jwt'` - exactly the reads-as-a-broken-venv failure the gate exists to prevent.
PyJWT is now in the extra and `jwt` is now gated.

`STRANDS_DASH_AUTH_ENABLED` recognizes `1`/`true`/`yes`/`on` and
`0`/`false`/`no`/`off`, and reports anything else instead of acting on it. It
previously read the variable as membership of the true-vocabulary alone behind a
non-empty check, so every other spelling - `enabled` and `y` among them -
resolved to auth-OFF *in preference to* the credential store, silently dropping
passkey auth from every guarded route on a dashboard that commands real
hardware. An unrecognized value now leaves the store as the source of truth, so
an enrolled passkey still guards the API.

The re-seal guard that lets "the person at the machine" enroll over a corrupted
store now reads the connection's own socket peer through `_socket_peer` rather
than `_client_ip`. `_client_ip` reads `cf-connecting-ip`, `x-forwarded-for` and
`x-real-ip` ahead of the peer - values the caller sets - so a stranger who
reached a dashboard whose store had just been corrupted could send
`CF-Connecting-IP: 127.0.0.1` and enroll the first passkey, which seals the
dashboard in their favour. The header read was wrong in both directions: it also
refused the genuine local owner whenever a proxy had set one of those headers to
a non-loopback address. `_client_ip` keeps its header chain for the per-ip
fairness cap it documents, which is accounting rather than trust.
The store is now replaced rather than rewritten in place. `_save_locked` wrote
the credential file directly, which truncates it before the new bytes land, and
it is the rail's highest-frequency writer - every successful login persists
`sign_count` through it, as does every enrollment and the corruption re-seed. So
a process killed during a routine login was itself the most likely producer of
the unparseable store the recovery posture above exists to rescue, and that
posture bounds the security damage rather than the loss: the passkey records and
the `jwt_secret` do not come back. The payload now lands in a sibling temp file
and is moved into place with `os.replace`, which is atomic within a directory, so
a reader sees either the whole previous store or the whole new one. Creating it
through `mkstemp` also closes a narrower gap: the file opens at `0o600`, where
writing the path directly created a new store at the umask default and only then
chmod-ed it, leaving a freshly generated `jwt_secret` briefly world-readable -
and because `os.replace` carries those bits onto the store, a store left wider
open by an earlier build is tightened the next time it is written.

The rp_id advisory in `status()` is now all-or-nothing and says when it is
skipped. It wrote `rp_id`, `secure_context` and `rpid_usable` straight into the
response as each was derived, so a failure between them - `rpid_is_usable`
raising, say - published an `rp_id` for the login screen to attempt with no
verdict on whether this module considered it usable, and the surrounding
`except` recorded nothing at all. The fields are assembled aside and merged only
once complete, so an undiscoverable rp_id stays absent rather than half-guessed,
and the skip is logged with its cause.
