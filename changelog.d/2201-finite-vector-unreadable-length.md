### Fixed: `finite_vector_error` reports a vector whose length cannot be read

`finite_vector_error` is the verdict half of the shared component read: it
returns a message rather than the floats, so a caller counts the components
itself, by reading the value again. It documented deferring that count, and
deferring it is what makes a *readable* length part of the verdict - a value with
no length cannot be read twice, because the read behind the verdict consumes it.

It was the one guard in the family not asking. `coerce_size_vector` (its own
coercing sibling over the same read), `coerce_rgba` and `_read_pose_vector` all
refuse an unreadable length, the middle one with a comment naming this exact
value class. The verdict half accepted it, and both of its live call sites then
read the consumed value: `add_object(size=<generator of three edge lengths>)`
reported `box needs 3 'size' component(s) ... got 0 (size=[])` - describing a
caller who passed nothing - and `patch_scene_mjcf`'s `size` field carried `object
of type 'generator' has no len()` into its envelope, naming neither the field nor
the op while `pos`, `quat` and `rgba` on that same op named both.

The probe now runs after the component read, which is the order
`coerce_size_vector` uses and the reason every refusal this guard already gave is
unchanged: a value whose `__iter__` raises, or whose read fails part-way, has no
readable length either, and those verdicts describe what happened instead of
claiming a domain check that never ran. Four value classes move from accepted to
refused - a generator, an iterator, a `map` and an iterable whose `__len__`
raises - each in the words its sibling coercion already used.
