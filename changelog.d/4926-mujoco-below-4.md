### Fixed: the mujoco extra stays below the engine's next major release

`mujoco 4.0.0` was published to PyPI on 2026-11-06 and the `mujoco` extra, pinned only `>=3.3.4`, resolved straight to
it. The new major is a breaking release: `mujoco.MjModel.from_xml_path` now rejects a model that includes the same
asset file twice, so loading a registry robot whose MJCF does that raised `ValueError: XML Error: File ... already
included` and the test suite went red. The extra now reads `mujoco>=3.3.4,<4`, holding the engine on the 3.x line the
backend is written against until 4.x is ported, and a pin regression keeps every mujoco requirement capped.
