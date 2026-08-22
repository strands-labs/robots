### Fixed

- **transforms**: `import_transform_class` now resolves a provider registered
  with `register_transform`, so it serves every name `list_transforms()`
  advertises. It consulted only the `policies.json` `"transform"` block and
  auto-discovery on `strands_robots.transforms.<provider>`, which refused every
  runtime-registered provider while `create_transform` resolved it - and the
  refusal it raised built its available list from `list_transforms()`, so it
  advertised the name it had just rejected, provider and alias alike. No
  provider in `policies.json` declares a `"transform"` block, so today every
  name this factory can serve arrives through `register_transform`; the two
  shipped transforms resolved only because their modules happen to sit under
  `strands_robots.transforms`, and a provider registered the documented way
  ships no such module. The runtime lookup now lives once in
  `import_transform_class`, which `create_transform` delegates to, and keeps
  its precedence over a shipped `"transform"` block.
