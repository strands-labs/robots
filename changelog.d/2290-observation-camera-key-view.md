### Fixed

- **MuJoCo observation cameras**: an observation image key now carries the view of
  the camera it names. A robot's own MJCF cameras are registered under their short
  name (`wrist`) while the compiled scene holds them namespaced (`arm0/wrist`), and
  the render loop resolved only the key, so every short key fell through to the free
  camera and published the scene overview under it - a policy reading
  `observation.images.wrist`, a recorded dataset column and the agent-tool
  observation all received the wrong view under a success result, and two cameras on
  one robot reported the same frame. A key the compiled model cannot answer for
  (such as one stranded by `remove_robot`) is now omitted rather than filled in with
  another view, matching `SimEngine.get_observation`'s schema. Joint state is
  unaffected.
