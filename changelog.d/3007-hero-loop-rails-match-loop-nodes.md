### Docs: hero_loop.svg rails sit beside the loop nodes that reference them

`docs/assets/hero_loop.svg` places its two capability rails beside the loop,
top-aligned with the middle band, so each rail's operative relationship is
adjacency to the left/right loop node - and both adjacencies read wrong:
**Act** names "policy" as its mechanism but sat beside the robots list, and
**Tools** names the robot-facing surface (teleop, record, mesh) but sat
beside the policies list. The ROBOTS rail now sits left beside Tools and the
POLICIES rail right beside Act; colors travel with their labels, and the
fade-in cascade is re-timed so it still sweeps left-to-right (left rail
0.1-0.4s, right rail 0.5-0.9s). Because the rails are top-aligned and
POLICIES has five entries to ROBOTS' four, the fifth chip moves from
bottom-left to bottom-right with the swap; nothing else moves, and the asset
path is unchanged, so no embed needs an edit. The adjacency is now pinned by
`tests/test_docs_brand_visuals.py`.
