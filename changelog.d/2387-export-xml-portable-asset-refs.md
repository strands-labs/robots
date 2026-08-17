### Fixed

`export_xml` now writes a scene that can actually be reloaded. MuJoCo resolves a `<mesh>` /
`<texture>` / `<hfield>` `file=` against a base directory it keeps on the spec and does not emit in
`spec.to_xml()`, and `spec.attach()` does not carry that base onto the parent - so an exported scene
named its assets relative to wherever the XML happened to be written, and neither
`mujoco.MjModel.from_xml_path` nor `load_scene` could recompile it. A multi-robot scene had no single
`meshdir` that could cover it, because each model contributes assets from its own root. Asset
references are now absolutized while the loaded spec still knows its own base directory, so the
exported MJCF is self-describing; a reference that resolves to nothing is left as authored.
