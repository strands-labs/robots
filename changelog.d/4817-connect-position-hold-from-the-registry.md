### Fixed: a connected robot takes position hold from the registry, never a peer's manifest

`Mesh.connect(id)` builds the remote `Robot` from the announced type's canonical `Description`, so its
`supports_position_hold` is the registry's `hardware.position_hold` for that type. Before, `connect` copied
`position_hold` straight from the peer's manifest, so a peer could claim a hold for a type the registry leaves
undeclared and the operator's `Robot` would believe it. Now an undeclared type reads `False` whatever the
manifest says, and a declared type keeps its registry answer; the device's own safety chain already gates an
illegal hold either way.
