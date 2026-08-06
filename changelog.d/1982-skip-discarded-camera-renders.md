### Changed: a recording that keeps no images no longer forces the render

`get_observation` overrides a policy's `requires_images=False` hint whenever a
recorder is attached, because a dataset recording normally needs every frame's
image obs. Scoped to no cameras that premise inverts:

```python
sim.start_recording(repo_id="local/actions", cameras=[])
sim.run_policy(MockPolicy(), steps=500)   # requires_images=False
```

The dataset declares no image features, and `_drop_unrecorded_cameras` discards
every image array before `add_frame` sees it - so the override rendered each
scene camera once per control step only to throw the pixels away. On a robot
carrying several cameras that is the dominant cost of an action-only rollout;
on `aloha` it is seven renders per step feeding a dataset with no image
columns.

The override now applies only when the recorder keeps at least one camera.
`recording_cameras` of `None` still means "record every camera", the legacy
default, so a recording that does not name cameras is unaffected. Both
directions are pinned: a `cameras=[]` recording renders nothing, and a
recording scoped to a real camera still renders it.

The import-cycle guard's `_build_import_graph` changed for the same reason -
work whose result was already known. It asked two whole-tree predicates per
import node, so the scan was quadratic in module size; the answers are now
collected in one pass, and the resulting graph is byte-identical at 214 nodes
and 296 edges. Those predicates stay as the readable statement of the rule and
are checked against the batched scan over five real modules, plus the one
deferral boundary no module in the tree exercises: a class-body import runs at
module import time, so it is not deferred.
