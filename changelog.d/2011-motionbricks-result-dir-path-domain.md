### Fixed: `MotionBricksConfig.result_dir` is held to a path domain rather than a truthiness test

`__post_init__` guarded the checkpoint path with `if not self.result_dir`, which
asserts truthiness rather than path-ness, so the values the field has to sort
sorted by a property it does not have. Measured with one config per value:
`result_dir=123` and `result_dir=["out"]` were accepted for being truthy and
stored verbatim, `b"out"` likewise, and `result_dir=0` was refused for being
falsy -- by a message reading "must be a non-empty path" about a number. Each
accepted value travelled to `Path(config.result_dir)` in the generator build and
raised `TypeError` there, naming neither the field nor the config.

Two consequences did not need the generator at all. A `Path` was accepted and
stored unnormalised, so a config built from `Path("out")` compared **unequal** to
the identical config built from `"out"`; and `result_dir=["out"]` left this
frozen -- therefore hashable -- dataclass unhashable, with `hash(config)` raising
`unhashable type: 'list'`.

The domain is now "a value a path can be read from" -- a `str` or any
`os.PathLike` -- normalised to the `str` the field declares, the same shape as the
`speed_scale` normalisation beside it. A caller holding a `Path` is doing what
every consumer of this field does, so it is accepted rather than refused, which
is why the remedy is a normalising domain and not an `isinstance(str)` gate. The
empty-string refusal is unchanged and keeps its own message: `""` is a
path-shaped value that names nothing. The rule stays local to this config rather
than becoming a shared guard in `utils.py`, since the numeric guards there each
have 5-123 callers and this has exactly one.

The enumeration fields (`clips`, `exp`, `device`) are unchanged: a type check
would refuse `device=5` and accept `clips="g1"`, so membership rather than type
is the rule they want, tracked separately.
