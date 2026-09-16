### Added: `python -m strands_robots dashboard` serves the operator dashboard from the landed rails

`strands_robots/dashboard/` carried auth, consent, settings, redaction, the
lockout type and the motion gate, and nothing that listened on a port: the
server, its CLI and the pages that documented them were removed with the closed
mega-PR, and the extra that supplies FastAPI shipped with no route to serve.

`create_app()` now wires those modules to paths and nothing else. Three routes
are public - `/api/health`, `/api/auth/status` and the ceremonies the login
screen drives - and every other route takes one dependency, `access.caller`,
which admits a passkey session, the static `security.auth_token` in constant
time, or nothing at all only while no guard is configured AND the caller is this
machine's own browser at this machine: socket peer loopback, no forwarding
header, a loopback-shaped `Host`, and an `Origin` that names that same host. The
last two are the ones a page cannot forge, and the most common loopback caller
that is not the operator is the operator's browser running someone else's page -
a DNS-rebound name arrives on the loopback socket carrying the attacker's
hostname, and a cross-site `fetch` arrives on it carrying the attacker's
`Origin`. A state-changing request from another origin is refused before any
credential is read, and a write must say it is JSON, so the no-preflight
cross-site POST is not a caller of any write route. A query-string token is
never read.

The public route the login screen reads publishes named fields, not whatever the
auth module returns. `/api/auth/status` answers before anyone has signed in, so
its callers on a sealed dashboard include the page a rebound name delivered; it
had been passing on the enrolled passkey list - ids, the labels the owner typed,
enrolment times - that `/api/auth/credentials` refuses to the same caller.
`setup_required` is that list reduced to the one bit the login screen reads, so
the screen loses nothing.

The
CLI binds `127.0.0.1:8090` and refuses any other address until a passkey or a
static token exists, naming the remedy.

The UI is plain files under `dashboard/static/` served by the same process, so
the wheel carries it and no build step exists. Those pages build their nodes and
hand each one its text as text: a Fleet row can hold a mesh peer's name, and a
peer-supplied string is attacker-controlled, so interpolating one into markup
would put script in the dashboard's own origin - carrying the session cookie,
naming this origin as its own, and therefore behind every check above by
construction. Measured in a browser against a `/api/fleet` whose robot name was
`<img src=x onerror=...>`: the payload ran, then did not. `docs/dashboard.md` is the one
page that invokes the command, which
`test_docs_module_commands_are_dispatched` now requires of a dispatched command.

A settings write reports a refused value as data. The store composed each reason
itself and then carried it out on a `CoercionError` that the write path caught
and interpolated into the `errors` list `POST /api/settings` returns, so the text
on the wire came off an exception object - benign only because every `raise` site
happened to be authored, and the shape by which a stack trace or a filesystem
path reaches a client that asked for none. `_graded` returns `(value, None)` or
`(None, reason)` instead, the write path appends what it was handed, and no
function on that path names an exception it caught. Every reason reads
identically. Returning the text also puts it under the package rule that a
returned refusal renders the value it refuses through `refusal_repr`, and that
rule earns its place here: the store rendered a bare `{value!r}`, so a value
whose own `__repr__` raises made `update_strict` raise - the refusal path failing
on the one input it exists to answer.
