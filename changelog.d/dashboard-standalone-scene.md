### Fixed: a standalone bundle's address and physics text stay text, and the page carries its own script policy

`?standalone=` picked the bundle the page loads with no check, and the standalone notice put the bundle's
`physics.substeps` and `standalone.hz` into the page unescaped. On a plain file server, which sends no policy header,
a link naming another site's bundle could run script on that page. `?standalone=` now accepts only a file on the
page's own address, like `?scene=`, and those values are escaped. `index.html` also carries the server's
`script-src 'self'` policy as a `<meta>` tag, so it applies wherever the page is served.
