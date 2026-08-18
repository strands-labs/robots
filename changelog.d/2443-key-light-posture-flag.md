### Fixed
- **rendering**: `derive_key_light`'s `upper_hemisphere` is now checked against the shared
  `boolean_flag_error` domain instead of being read by truthiness. The flag selects which
  region of the environment map is searched for the dominant light, so every truthy
  spelling of *off* (`"false"`, `"no"`, `"off"`, `"0"`) searched above the horizon - the
  region the value asks to leave out - while an undeclared falsy value (`None`, `0`, `""`)
  silently searched the full sphere, which this argument's own docstring warns aims a key
  light from underneath. The empty-region refusal branched on the same truthiness, so a
  caller who passed `"false"` was told the map is black above the horizon and advised to
  pass `upper_hemisphere=False` - the value they believed they had passed.
