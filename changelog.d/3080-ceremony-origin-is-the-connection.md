### Fixed: a passkey ceremony is verified against the connection, not the caller's claim

`_derive_origin` returned the request's own `Origin` header whenever
`STRANDS_DASH_AUTH_ORIGIN` was unset, which is the default. That value was handed
to `verify_registration_response` and `verify_authentication_response` as
`expected_origin`, so the WebAuthn library was told to expect whatever the caller
said and the comparison could not fail. `expected_rp_id` still came from the
stashed challenge record, and a browser will not sign for an rpId that is not a
registrable suffix of the page origin, so this was never cross-origin credential
theft -- what was lost is the binding *within* one registrable domain: an
`http://` downgrade and a sibling subdomain both passed a check whose only
purpose is to refuse them. The scheme half was caller-controlled even when no
`Origin` header was sent, because `x-forwarded-proto` was read with no
trusted-proxy allowlist.

The expectation is now a fact about the connection: the transport's own scheme
plus the `Host` header that `rp_id_verdict` already binds, so the two
expectations a ceremony is verified against agree by construction. A socket
scheme is mapped to the page origin it belongs to -- a `wss` socket belongs to an
`https` page, and that page's `Origin` is spelled `https://` -- and an `Origin`
header that contradicts the connection is refused by the new `origin_verdict`,
which names both origins, rather than being adopted.

A TLS-terminating proxy is still honoured, by the server under the trusted-proxy
allowlist the operator configured there (uvicorn's `--proxy-headers` with
`--forwarded-allow-ips`), which rewrites the scheme before the app is reached; a
deployment that cannot do that sets `STRANDS_DASH_AUTH_ORIGIN`, which is now
documented and normalised, and which still outranks every header. The installs
that need that pin are the ones whose proxy rewrites `Host` and `Origin`, so
consulting either header when it is set would refuse exactly the deployment it
exists for.

`status()` reads a new `_served_origin` seam instead of the ceremony
expectation: its advisory reports where the dashboard is served, so a mismatched
`Origin` cannot remove the login screen's hints exactly when a misconfigured
proxy makes them worth reading. One case changes in the permissive direction: a
dashboard reached over https that receives no `Origin` header used to
reconstruct `http://<host>` and would have refused a legitimate browser.
