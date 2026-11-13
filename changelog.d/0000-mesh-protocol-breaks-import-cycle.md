### Fixed: the mesh helper modules no longer form an import cycle through `mesh.py`

`remote.py`, `tool.py` and `broadcast.py` each took a `Mesh` and imported it from `mesh.py` only to name the type,
while `mesh.py` imports those modules back (`connect`, `use_mesh`, the broadcast helpers). They now type against a
new leaf `MeshView` protocol in `strands_robots/mesh/protocol.py`, so none of them imports `mesh.py`. The public
surface is unchanged: `Mesh` satisfies `MeshView` structurally and nothing new is exported.
