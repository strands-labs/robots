### Fixed

- **policies/groot**: resolve model keys by name under either GR00T release key
  spelling. N1.6/N1.7 declare `ModalityConfig.modality_keys` bare (`"front"`)
  and N1.5 declares them prefixed (`"video.front"`); every consumer compared
  them as bare, so against an N1.5 checkpoint mapping resolution fell through to
  positional matching - pairing two identically named cameras by declaration
  order, so a wrist image could be sent under the model's front key with nothing
  reporting it. A correct explicit `observation_mapping` was also refused, and
  `strict_keys=True` raised for a key set that matches exactly by name. Both
  spellings now reduce to the bare name for comparison, leaving the emitted
  payload unchanged for every release.

  Which spelling a resolved key is then *held* in depends on the direction it
  travels, and the two directions want opposite answers. An observation key is
  held in the model's declared spelling, because it is the key the payload is
  sent under. An action key is held bare, because it is never sent - it only
  arrives, and both unpack paths reduce a raw output key with
  `removeprefix("action.")` before matching it by name.

- **policies/groot**: deliver every mapped actuator against an N1.5 checkpoint.
  `_auto_infer_action_mapping`'s positional arm stored the model's declared
  spelling, so against a prefixed release it produced a mapping key that
  `_unpack_actions` and `_unpack_service_actions` could not match: each one
  reduces a raw output key to bare before its lookup, so every actuator missed
  its mapping and was emitted under `unmapped.<bare>` with nothing reporting it.
  Action mapping keys are now reduced to bare by a single owner shared with
  `_parse_action_mapping`, so service mode - which has no model metadata to
  canonicalize against - is reduced on the same terms. Naming one action key in
  both spellings is refused rather than collapsed to whichever entry iteration
  order reached last.
