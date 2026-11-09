### Changed: a dashboard session works only on the dashboard that issued it

Browsers send a `localhost` cookie to every port on the machine, so the dashboard's session cookie also reached any
other local server the user opened, which could send it back and act as the signed-in user. A browser session now
has two halves: the cookie (named `strands_robots_session_<port>`, HttpOnly, SameSite=Strict) and a proof the sign-in
answer hands the page, which the page keeps in its own origin's `localStorage` and sends as `X-Strands-Session` (and
as the `strands-session.<proof>` subprotocol of a socket). The cookie alone, or the cookie's token sent as a bearer,
opens nothing, and a session names the Host it was signed in on and is refused on any other. Camera frames load
through `fetch`. Script tokens from `dashboard.auth.token()` sent as `Authorization: Bearer` are unchanged. Open
browser sessions from before this change end: sign in again with your passkey.
