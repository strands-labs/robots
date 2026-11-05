### Fixed: text from the dashboard's address or a scene file can no longer run as script on the page

A `?scene=` value that failed to load was shown in the page's notice as markup, and a scene file's `revision` was put
into a status chip unescaped, so a crafted link could run script on the dashboard's origin and press its buttons. The
error and the revision are now text. `?scene=` must be a path on the dashboard's own address, `?cli=` must be a
loopback `host:port`, and a non-integer `revision` is ignored. Every HTML page the server sends also carries
`Content-Security-Policy: script-src 'self'` (plus a hash for the one inline colour-scheme script, `object-src 'none'`,
`base-uri 'none'`), so markup that reaches the page by mistake cannot run. The voice panel's microphone worklet moved
from a `blob:` URL to `static/voice-worklet.js` for the same reason.
