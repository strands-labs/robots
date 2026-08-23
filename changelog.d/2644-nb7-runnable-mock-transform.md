### Docs: notebook 7 section 4 runs the dataset-transform surface on CPU

Section 4 of `examples/notebooks/07_data_augmentation.ipynb` was markdown-only
because the generative `cosmos_transfer` backend needs a GPU. It now runs the
same `strands_robots.transforms` surface on CPU via the `mock` provider
(`create_transform("mock")`, a deterministic per-variant brightness shift)
against the `local/nb7_demo` dataset section 1 records: one recorded episode
becomes three provenance-marked variants, the provenance round trip is
demonstrated end to end (`write_provenance` via the transform,
`load_provenance` / `synthetic_episode_indices` on the read side,
`synthetic=true` on every generated episode), and the pixel shift is measured
against the source frame. The notebook stays seeded, CPU-only, credential-free
and committed with zero outputs; `cosmos_transfer` stays out of the code cells,
with the prose pointing at `docs/data/transforms.md`. The
`examples/notebooks/README.md` row is updated to match. (#2644)
