### Fixed: Isaac Sim 6.0.x pip wheels no longer crash LIBERO scene loads or starve fresh cameras

On the pip `isaacsim` 6.0.x wheels, `timeline.stop()` invalidates the
physics-tensor view asynchronously, so the deferred-physics guard returned
with the view still live and `RigidPrim.__init__`'s eager velocity query
raised the bare `Failed to get rigid body velocities from backend` on every
LIBERO `load_scene`. The guard now also calls
`SimulationManager.invalidate_physics()`, tearing the view down
synchronously. Camera warm-up had the matching downstream failure: after the
deferred-physics window the timeline stays stopped and `world.step()` does
not resume play, so a freshly installed camera's RTX render product never
accumulated a frame (`video.wrist_image` missing on the LIBERO GR00T path);
`_warmup_camera` now re-asserts `timeline.play()` each iteration, because a
queued stop can land mid-loop and undo a single pre-loop resume. The LIBERO
backend-matrix parser also accepts drivers that add extra `key=value` fields
(e.g. `run.py isaac`'s `resolved_task=`), so the isaac row reports its real
success rate instead of empty cells.
