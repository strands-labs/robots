### Fixed: a standalone dashboard page keeps a quiet console

With no server, the e-stop panel is hidden and never polls `/api/mesh` (it asked every 2 s, a 404 each time), and the
re-probe for a server that starts later backs off from 5 s to 60 s instead of asking every 5 s (each probe is a failed
`GET /api/scene` plus a refused WebSocket). Measured in headless Chromium over the first 60 s: 54 console errors
before, 8 after. A server that answers later still wins on the next probe.
