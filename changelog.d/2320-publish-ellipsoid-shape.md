### Fixed

- **Simulation**: the MuJoCo agent tool schema now offers `ellipsoid` in its `shape` enum. `add_object` already compiled one and the schema's own `size` description already published its layout, but the `shape` enum omitted it - and that property carries no description, so the enum is the whole specification a schema-constrained caller reads. The enum is now pinned against the builder's live shape tables in both directions.
