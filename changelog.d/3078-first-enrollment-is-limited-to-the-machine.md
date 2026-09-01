### Fixed: a stranger can no longer seize a fresh dashboard by enrolling the first passkey

`auth_enabled()` IS `has_credentials()`, so the first passkey enrolled decides who
owns the fleet - every later caller is locked out of a dashboard that commands real
hardware. Three branches could gate that request and only two did: a bootstrap token
was checked when one was configured, and a loopback check applied when the credential
store was *unreadable* and no token was set. A fresh install with an intact empty
store and no token configured matched neither.

```
fresh store,   peer 203.0.113.9  ->  ACCEPTED   (challenge issued)
damaged store, peer 203.0.113.9  ->  refused    (403)
```

The guard was on the weaker case. A damaged store needs a disk error to happen first;
a fresh install is every install's first minute, and it is where seizing enrollment
costs an attacker nothing.

Both routes are now gated on the one predicate. The refusal wording still varies by
cause, because one that blamed a disk error where none occurred would send the reader
looking for a backup file that does not exist; both name the same two ways forward.
The decision reads the socket peer rather than `_client_ip`, whose own contract says
it is for the per-ip cap and never for trust, so a `CF-Connecting-IP: 127.0.0.1` from
a stranger does not manufacture locality, and a connection with no peer address is
not treated as local either.

Two boundaries are deliberate. The gate lifts once a credential exists, since an
owner adding a second passkey remotely is the route's session decision and a gate
that never lifted would be a lockout rather than a guard. And it runs after the rp_id
usability check, so a caller on a bare IP is still told that passkeys cannot work
there at all instead of being sent to a console where the same refusal awaits. A
configured `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` remains the route for a headless
robot, so its owner is not locked out.
