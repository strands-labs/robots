### Docs: a mesh `size` is documented per backend rather than claimed to be at parity

`add_object(shape="mesh", ...)` reads `size` two different ways, and
`docs/simulation/newton.md` said the two backends were "at parity". Newton
consumes it as a per-axis scale on the loaded geometry (per-shape default
`[1, 1, 1]`, reaching the solver as `add_shape_mesh(..., scale=...)`); MuJoCo
reads no component of it and takes the extent from the asset alone. So
`size=[2, 3, 4]` scales the asset on one backend and is ignored on the other,
and both calls report success - so the paragraph a reader porting a scene would
have checked was the one telling them not to look.

Both pages now state the divergence and point at the open contract decision
(#2300); no accepted input, default or solver call changes. Pinned by
`tests/simulation/test_mesh_size_docs_match_backend_divergence.py`, which
re-measures the divergence from both backends on every run and fails - rather
than skips - on the change that converges them, so the prose and the guard are
settled together.
