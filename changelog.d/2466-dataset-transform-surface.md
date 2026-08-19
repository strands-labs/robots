### Added: episode augmentation as a dataset transform surface

`strands_robots.transforms` - the third provider shape beside policies and
trainers: LeRobotDataset in, augmented LeRobotDataset out. Video observation
streams are transformed while action / state / task columns pass through
byte-identical; every generated episode is provenance-marked
(`meta/provenance.json`, `synthetic=true` + source episode + transform
name/version) so training filters treat generated pixels honestly; and an
optional deterministic re-validation gate discards-and-counts any generated
episode whose verdict flips. Ships `create_transform()` /
`register_transform()`, the no-dependency `mock` reference backend, and
`cosmos_transfer` - a Cosmos-Transfer-style video2video backend behind a
vendor-neutral pipeline seam that refuses cleanly (with the licensing caveat)
when no generation pipeline is bound.
