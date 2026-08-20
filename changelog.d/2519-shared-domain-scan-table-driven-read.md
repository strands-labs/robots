### Fixed

- **training**: the four field-scoped shared-domain guards (`_seed_problems`,
  `_validation_episodes_problems`, `_lora_hyperparameter_problems`,
  `_launch_topology_problems`) derive their scope from the tree, and now
  recognize a table-driven read (`getattr(spec, field)` over a tuple of field
  names) as well as a read by name. A provider that forwards a spec on names no
  field in an attribute access, so it sat outside all four derived scopes: three
  guards certified a `readers == {...}` sweep that omitted a real reader, and all
  four were blind to a backend that forwards a gated field and skips the gate.
  One shared rule now defines both forms for every guard.
