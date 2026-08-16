### Fixed

`add_robot` now registers only the cameras the robot being added contributed. It
walked the merged model's camera table and stripped just that robot's namespace,
so a camera an earlier robot declared was registered a second time as belonging
to the new robot: `remove_robot` (which cleans up by owner) then left the
duplicate advertising a camera the compiled model no longer had, and when two
robots declared the same short camera name the second robot's own camera was
dropped from the registry entirely. A second claimant of a short name is now
registered under its qualified name (`arm2/wrist`) rather than being dropped;
the short alias stays first-come, so a single-robot scene is unchanged.
