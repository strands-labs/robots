### Fixed: a link written as raw HTML is graded like a Markdown one

`tests/test_markdown_links_resolve.py` states one rule -- a relative link target
must resolve to a path that exists inside the repository -- and read only
Markdown syntax to find one. GitHub and MkDocs both render raw HTML embedded in
a Markdown file, so the six relative targets this tree writes that way were
graded by nothing, and two of them were dead links.

`docs/policies/wbc.md` carried both, five lines apart, in the same figure:

| line | syntax | target | graded | GitHub | site |
|---|---|---|---|---|---|
| 277 | Markdown `![...]()` | `../assets/wbc/g1_walk.gif` | yes | resolves | resolves |
| 282 | raw HTML `<a href>` | `../../assets/wbc/g1_walk.mp4` | **no** | **404** | resolves |

The wrong prefix is easy to write because the two syntaxes are not resolved the
same way. MkDocs **rewrites** a Markdown target -- `../assets/wbc/g1_walk.gif`
is authored against the source file and the built site serves
`../../assets/wbc/g1_walk.gif` -- but it **passes raw HTML through unchanged**,
and the rendered page sits one directory deeper than the source file. No single
relative prefix is correct on both surfaces, so an author picks one and nothing
records which. Both MP4 links were authored for the site; read on GitHub, where
`docs/policies/wbc.md` resolves them against its own directory, they pointed at
`assets/wbc/*.mp4` in the repository root, which is HTTP 404 on `main`.

Neither link checker could see it. `mkdocs build --strict` exits 0 -- it resolves
Markdown targets and does not inspect raw HTML -- and the guard read no HTML at
all, so the sweep reported a clean tree.

The reader now takes `<a href>` and `<img src>` as targets. The scope is the
elements GitHub keeps, because GitHub serves the source file and is the surface
that is not optional; `<video>`, `<source>` and `<iframe>` are dropped by its
sanitizer, so their targets reach no reader of the source file, the published
site is their only consumer, and the site-relative spelling this tree uses for
them is correct. Grading those too would report `docs/device-connect.md`'s
working `<source src>` embed as broken -- a mutation that widens the element list
fires five cells, which is what pins the boundary rather than assuming it.

The two links now use the absolute blob URL the same page already uses three
times for cross-surface links (L7, L262, L340), so they resolve from GitHub, from
the published site and from a local `mkdocs serve` alike. `README.md`'s three
`<img src="docs/assets/*.svg">` embeds are relative, correct, and now read -- the
accepted boundary comes from the tree rather than from an allowlist.
