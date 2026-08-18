### Added: the composited robot is lit by the scene it stands in

The `isaac_gs`/`mujoco_gs` hybrid composite blended a robot pass and a
photoreal background that were lit and toned in two unrelated worlds
(#2323). Three stages, each usable on its own:

- **IBL from the GS scene** — `strands_robots.rendering.bake_environment_map`
  renders an equirectangular environment map of any background from the
  robot's position in the world frame (through the background's own aligned
  `render`, unlike the centroid-frame `bake_gsplat_panorama`, which now
  shares the same cube→equirect core), and `derive_key_light` estimates the
  map's dominant light direction/color. `examples/isaac_gs` textures its
  `DomeLight` with the baked map and aims/colors its key light from it by
  default (`--no-ibl` opts out) instead of the hardcoded example lights.
- **Shadow catcher** — `HybridCompositor(shadow_plane_z=...)` reads
  foreground pixels matching the analytic depth of a configured plane
  (`plane_depth`, new) as a shadow catcher: never painted, their shading is
  multiplied onto the background in linear light, so the arm grounds itself
  with a contact shadow on the backdrop. `examples/isaac_gs` adds the matte
  plane at the support surface by default (`--no-shadow-catcher` opts out).
  The `CompositeFrame` contract is unchanged.
- **Color-pipeline contract** — `strands_robots.rendering.color` documents
  what space each layer is in (everything arrives display-referred sRGB) and
  supplies the sRGB/linear transfer functions; light arithmetic (the shadow
  multiply, and the optional `blend_in_linear=True` seam blend) now runs in
  linear light.
