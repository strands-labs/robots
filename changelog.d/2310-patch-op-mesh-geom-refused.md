### Fixed: an `add_geom` patch op asking for a mesh is refused instead of poisoning the world

`patch_scene_mjcf`'s `add_geom` op resolved its `type` through the full shape
vocabulary, `"mesh"` included, while `_PATCH_OP_KEYS` gives it no key that could
name a mesh asset. The geom it added therefore carried no meshid and MuJoCo
refused the batch at recompile - a refusal that names a MuJoCo element id rather
than the op, and that fires outside the `try` rolling a rejected batch back, so
the mutated spec stayed installed. Measured on a default world, after one such
patch a second valid patch, a third, and an unrelated `add_object` all returned
that same `must have valid meshid` error: the world was unusable for the rest of
its life.

The op now refuses `type="mesh"` before it touches the spec, naming the route
that does register the asset (`add_object(shape="mesh", mesh_path=...)`) and the
`type` values it accepts. `_normalize_size`'s mesh branch, which only that path
could reach and which returned `[0.0, 0.0, 0.0]` where its own docstring
documented `[]`, is deleted: a mesh now falls through to the raise, which is what
"nothing should be asking" looks like.
