### Docs: the 3D viewer compiles its model in a Worker, and the landing page can show any robot

Readers found the landing page frozen until the arm appeared: MuJoCo compiled the model
(18 meshes, a convex hull each) as one 4.5 s WebAssembly call on the page's thread. The
engine now runs in a Worker that receives the streamed files, compiles, owns the physics
and posts geom poses; the page only draws, so every button answers while the model
loads. Under the hero viewer, a picker lists all 64 streamable robots by family and swaps
the model in place.
