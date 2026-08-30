### Docs: hero_loop.svg rails sit beside the loop nodes that reference them

`docs/assets/hero_loop.svg` centered its two capability rails on the middle
band, so each rail's operative relationship is adjacency to the left/right
loop node - and both adjacencies read wrong: **Act** names "policy" as its
mechanism but sat beside the robots list, and **Tools** names the
robot-facing surface (teleop, record, mesh) but sat beside the policies
list. The ROBOTS rail now sits left beside Tools and the POLICIES rail
right beside Act; colors travel with their labels, and the fade-in cascade
is re-timed so it still sweeps left-to-right (left rail 0.1-0.4s, right
rail 0.5-0.9s). No other visual change; the asset path is unchanged, so no
embed needs an edit.
