### Fixed

- `policies/kimodo`: a Kimodo pipeline output carrying no `motion` field is now
  refused with an actionable `RuntimeError` naming the `model_id` and the fields
  the output did carry. A diffusers output is a `BaseOutput`, which subclasses
  `OrderedDict`, so the previous subscript on the mapping branch raised
  `KeyError: 'motion'` one line before the refusal written for that case - the
  actionable message was unreachable on the only path that could reach it.
