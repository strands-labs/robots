### Added: episode augmentation as a dataset transform surface

`strands_robots.transforms` - the third provider shape beside policies and
trainers: LeRobotDataset in, augmented LeRobotDataset out. Video observation
streams are transformed while action / state / task columns pass through
byte-identical; every generated episode is provenance-marked
(`meta/provenance.json`, `synthetic=true` + source episode + transform
name/version) so training filters treat generated pixels honestly; and an
optional deterministic re-validation gate discards-and-counts any generated
episode whose verdict flips. Because the pass-through holds every non-image
column byte-identical, a verdict that consults no `observation.images.*`
column can never flip - the orchestration measures which columns the verdict
read and reports such a run as ungated (`revalidated=False`, cause in the
message) instead of letting a vacuous gate render as a clean gated pass.
Ships `create_transform()` / `register_transform()`, the no-dependency `mock`
reference backend, and `cosmos_transfer` - a Cosmos-Transfer-style video2video
backend behind a vendor-neutral pipeline seam that refuses cleanly (with the
licensing caveat) when no generation pipeline is bound. Auto-discovery
distinguishes a provider module that was never written (`ValueError` naming
the available transforms) from one whose backend dependency is missing (the
`ModuleNotFoundError` surfaces).
