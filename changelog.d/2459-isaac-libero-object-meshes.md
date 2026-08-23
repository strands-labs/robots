### Added: LIBERO objects render as real meshes on the Isaac backend

`add_object(shape="mesh", mesh_path=...)` is now implemented on the Isaac
backend (it was an explicit refusal pointing at MuJoCo): OBJ, STL and legacy
MuJoCo binary MSH assets - the format LIBERO's compiled scenes declare for
the bowl/plate visual meshes - are
converted to USD once - content-addressed cache under
`$STRANDS_BASE_DIR/asset_cache/usd_meshes/` - and referenced onto the stage
with convex-hull collision, the MuJoCo mesh contract (`size` ignored, the
asset defines the extent). `SceneObject` /
`load_mjcf_scene_objects` carry each LIBERO task object's visual mesh (path,
scale, body-frame pose) through to `IsaacSimulation.load_scene`, which renders
the real bowl/plate over an invisible collision-AABB box proxy, so physics is
unchanged while a pixel-conditioned policy (e.g. a GR00T LIBERO checkpoint
trained on MuJoCo-rendered visuals) observes the objects it was conditioned
on instead of gray boxes. A mesh-only body now gets the mesh's computed
bounds as its proxy instead of the hardcoded 0.05 m box, and a declared mesh
asset missing on disk fails the scene load loudly - never a silent box.
While any object still renders as a proxy, the `load_scene` report carries an
explicit caveat that pixel-conditioned policy scores on that scene are not
comparable across backends; the caveat disappears when every object carries
its mesh. (#2459)
