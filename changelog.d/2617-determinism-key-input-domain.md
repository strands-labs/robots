### Fixed: every input the determinism key is spread from is held to its domain

`derive_variant_seed` spreads the triple `(seed, source_episode, variant)` through one
`numpy.random.SeedSequence`, so all three name the stream a generated variant is rendered
from. Only `source_episode` was checked, and the other two degraded exactly the way that
function's own `Raises:` block already warns about for the one that was.

Measured on mid-grey frames through `MockTransform`: `variant=True` derived pixel 110,
byte-identical to `variant=1`; `variant="1"` the same, because NumPy coerces a str spelling
of a whole number; `variant=False` matched `variant=0`; and `seed=True` and `seed="1"` both
derived `seed=1`'s key. Two "distinct" variants of one episode were therefore written into
the augmented dataset as two episodes whose pixels are identical, under provenance records
naming different `variant` counters - so the data-multiplication factor the transform
surface exists for was silently 1 for that pair, with no refusal anywhere.

The values NumPy refuses on its own were no better placed: `-1`, `2.5` and `None` arrived as
its internal `expected non-negative integer` / `seed must be integer` /
`object of type 'NoneType' has no len()`, naming neither the parameter nor the surface, and
two of those are `TypeError` rather than the `ValueError` these surfaces document as their
refusal channel.

All three inputs now go through the shared `non_negative_whole_number_error` rule, with
`seed=None` kept as the documented opt-out from determinism rather than a stream name. The
`int()` coercion moves after the guard, never before, which is the ordering that rule's own
docstring asks for. Every triple accepted before derives the identical key; the only change
on the accepted side is that an integral float is honored on `variant` and `seed` exactly as
it already was on `source_episode`.

Both backend seams refuse the counter too, for the reason their `source_episode` guards
already exist. A backend need never reach `derive_variant_seed` at all - `mock`'s explicit
`pixel_shift` mode derives no key, so it shifted the pixels without reading the counter once
for any value above - and the refusal should name the counter rather than whatever the
pipeline seam reaches first: unbound, `cosmos_transfer` reported an unusable counter as "no
video2video pipeline is bound", a wiring diagnosis for the one thing the caller had got
right.

`transform()` is unaffected either way: it derives `variant` from
`range(int(spec.variants_per_episode))` and the shared spec preflight already holds
`variants_per_episode`, `episodes[]` and `seed` to their domains, so the reachable surface is
a direct caller of the public helper or of the backend seam - the same reachability the
shipped `source_episode` guard is justified by.
