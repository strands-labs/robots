### Fixed: `sim.render()` with no camera frames the robots, not the floor

The overview camera `sim.render()` makes on first use now fits the robots' visible geometry in its 45 degree view
(10 % margin, still from the -y side, from above). Before, it stood at least 1.8 m back whatever the robots' size: a
lone SO-101 filled 68 of 480 rows (0.4 % of the pixels) and a G1 standing at the origin was cropped at the top. Now
the SO-101 spans 341 rows and the G1 is whole. An engine that cannot report its geometry (`Backend.outline()` returns
None) keeps the old placement.
