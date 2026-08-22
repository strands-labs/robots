### Fixed: a tracker body index is refused unless it addresses a body

`ProtoMotionsConfig.anchor_body_index` and `root_body_index` are offsets into
`body_names` - the tracker reads a body out of the reference-motion cache by row
index, never by name. `load_config_from_yaml` documents its result as "validated for
consistent dimensions" and promises `ValueError` for "an inconsistent dimension", and
the module docstring promises such an error "surfaces at policy build time with a
clean message". It checked the `stiffness` and `damping` lengths against the joint
count, and neither index.

A negative index is a valid tuple lookup, so it was the silent case. `body_names[-1]`
is `right_rubber_hand`, a real link: an `anchor_body_index: -1` sidecar loaded, and
the whole chain then agreed on the hand - `required_bodies` declared it, the runtime
resolved it once per rollout and merged `body.right_rubber_hand.quat` into every
observation, `_extract_anchor_rot` found exactly that key, and the future-reference
window sliced the same row. Measured on the tracker's own G1 embodiment, that link's
world orientation is 0.01, 10.37 and 20.20 degrees from `torso_link` across a
waist-turn pose set - an anchor orientation the network consumes every tick, with
nothing anywhere reporting it. An index past the end instead deferred `IndexError:
tuple index out of range` to whichever property read the name first, naming neither
the field, the value, the range, nor the sidecar.

The loader's `int()` coercion widened the same hole by running before anything looked
at the value: a yaml `anchor_body_index: true` became row 1 (`head`, 34.4 degrees out)
and a `2.7` became row 2 (`left_hip_pitch_link`), so the sidecar route and a
hand-built `ProtoMotionsConfig(anchor_body_index=True)` disagreed about the same value.

Both indices now go through the shared `non_negative_whole_number_error` rule and then
the row count of the body list they index, in `ProtoMotionsConfig.__post_init__` so a
sidecar and a hand-built config report the same value the same way. The domain runs
before the `int()` normalisation, never after, which is the ordering
`MotionBricksConfig.__post_init__` already states the reason for; the four sibling
policy configs validate in `__post_init__` too, and `protomotions` was the one that did
not. An integral float such as `16.0` still addresses a row and is kept, normalised to
the row number both consumers index with, and the refusal quotes the range it was
checked against. `anchor_body_name` and `root_body_name` documented
`Raises: IndexError`, which is now unreachable, so both say so instead.

This is the config-side mirror of a hole the package already refuses on the model side:
a G1 MJCF missing bodies is rejected precisely because reading positionally hands the
tracker `left_shoulder_pitch_link` where it asked for `torso_link`. The sidecar that
named the same wrong row was not.
