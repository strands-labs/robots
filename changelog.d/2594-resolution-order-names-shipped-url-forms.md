### Fixed

- **registry**: `resolve_policy`'s documented stage-1 rung now names the URL forms
  the shipped registry actually matches. Stage 1 has no vocabulary of its own - it
  matches the `url_patterns` providers declare in `policies.json` - and the rung was
  wrong in both directions: it listed a scheme-less `host:port`, which no shipped
  pattern matches (all five require a scheme), and omitted `cosmos3://` and
  `vera://`, which they do. A caller following the rung with `localhost:5555` got no
  reason from `policy_provider_error`, then a raise out of `run_policy` diagnosing a
  broken HuggingFace checkpoint - `config.json had no usable 'type'` - for a network
  address, having reached stage 5 and been forwarded to `lerobot_local` as a
  checkpoint id. `policy_provider_error`'s own enumeration of resolvable spellings
  carried the same claim and is corrected too. The generic scheme-less branch keeps
  its behaviour: it is a tested extension point, reached when a provider declares a
  scheme-less pattern, and which provider should own the form is a contract choice
  left open. A new bidirectional guard grades both surfaces against the registry, so
  a scheme added to one cannot drift from the other.
