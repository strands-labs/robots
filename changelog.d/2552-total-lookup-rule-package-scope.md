### Changed

- **tests/simulation**: the check that keeps a registry lookup inside the total-lookup rule now grades every module in the package instead of three backend directories, so the same shape added outside a backend is reported rather than passing unseen. The two safe shapes are classified structurally - a literal name, and a name read out of the engine's own `list_robots()` - and the one remaining exemption now has the refusal it rests on asserted beside it.
