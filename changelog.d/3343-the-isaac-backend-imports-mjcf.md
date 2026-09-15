### Added: the Isaac backend imports MJCF

`add_robot(mjcf_path=...)` was refused outright, on the stated grounds that the
Isaac backend "has no MJCF robot importer". That was an assertion this repository
made in three places and measured in none. Isaac Sim 6.0.1 registers
`isaacsim.asset.importer.mjcf`, exposing `MJCFImporter` / `MJCFImporterConfig`
alongside the `MJCFCreateAsset` Kit command:

```
extensions: ['isaacsim.asset.importer.mjcf', 'isaacsim.asset.importer.mjcf.ui']
commands:   ['MJCFCreateAsset', 'MJCFCreateImportConfig']
```

The new `strands_robots.simulation.isaac.mjcf_assets` converts an MJCF to USD
through it, cached content-addressed under
`$STRANDS_BASE_DIR/asset_cache/usd_robots/`, and the result is loaded by the
existing native USD path - so a name, an MJCF, a URDF and a USD all converge on
one proven loader rather than each growing its own.

The cache key covers the description **and every file its directory holds**. An
MJCF is not self-contained: Menagerie's `scene.xml` is a handful of lines that
`<include>` the robot body and reference a `meshdir` of STLs, and the geometry
PhysX simulates lives in those, so a key over the named file alone would serve a
stale conversion after any change that did not touch it. Two vendor behaviours are
handled rather than assumed, both measured: `usd_path` is an output *directory
root* (the importer writes `<root>/<stem>/<stem>.usda` and returns that path), and
with no destination it writes beside the source - which fails with `Read-only file
system` for every description this package resolves, since those live in a shared
`robot_descriptions` checkout.

`fix_base` defaults to `None`, the vendor default, which honours whatever the
description says. That is the only choice that keeps a floating-base robot
floating: every shipped quadruped and humanoid declares its base with
`<freejoint>`, and `True` would bolt such a robot to the ground while reporting
success.
