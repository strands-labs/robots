### Fixed: a provenance record that cannot answer is refused, not read as recorded

The provenance sidecar exists so generated and recorded episodes stay
distinguishable, and `synthetic_episode_indices()` is the one call that separates
them: everything it returns was generated, everything outside it was recorded. It
read the verdict with `synthetic is True`, so a record whose field was present but
not a boolean landed outside the set - i.e. was reported as recorded, with nothing
raised. `write_provenance()` is public and documents its record shape in full, and
required only that the key be present, so `synthetic=1` or `synthetic="true"` was
stored and then read as a recorded episode.

The two keys a reader turns into a verdict are now held to the shared domains -
`episode_index` on `non_negative_whole_number_error`, `synthetic` on
`boolean_flag_error` - by one rule both `write_provenance()` and
`load_provenance()` consult, so a record the writer refuses is a record the reader
refuses. `synthetic` is refused rather than coerced, because every non-empty string
and every non-zero number is truthy; `synthetic=false` remains a readable answer
that is simply outside the set. The writer also stores both keys in the type the
schema documents, which lets a NumPy verdict round-trip as JSON `true` instead of
failing to serialise, and an `episode_index` that cannot name an episode is now
named in the refusal rather than surfacing as `int()`'s own complaint.

That split - validate the keys a reader turns into a verdict, carry the descriptive
keys through untouched - is the one `record_deterministic_verdicts` already makes
for the other per-episode sidecar.
