### Fixed: a camera name the MuJoCo backend's own render resolves past is refused at `add_camera`

`render`, `render_depth` and `get_frame` select the free camera for
`camera_name` in `(None, "", "default", "free")` by an explicit token check, but
`add_camera` let a caller claim two of those tokens as a camera *name*. The
camera was registered, compiled into the model and offered by `list_cameras`,
and every render of it silently answered with the free view instead - measured
on MuJoCo 3.11.0, `add_camera("free", position=[8, 8, 8], target=[0, 0, 0])`
returned `status="success"` with `mj_name2id(model, CAMERA, "free")` resolving,
while `render(camera_name="free")` took the `cam_id = -1` branch. The camera was
unreachable through the API that created it, with no error at any point.

`"default"` was refused only as a *duplicate*, because `create_world` registers
the built-in free view under that name, and that refusal prescribed the remedy
that completed the defect: `remove_camera("default")` succeeded, the following
`add_camera("default", ...)` succeeded, and the scene was left with an
unreachable camera where the advertised free-view alias had been.

Both are now refused as reserved names, ahead of the duplicate-name test, with
the reason stated. The Newton backend already refused the whole set, so this was
a one-backend parity gap; its refusal and MuJoCo's are now the same sentence
from one shared `reserved_camera_name_error`. The token set itself becomes a
single `FREE_CAMERA_TOKENS` definition, replacing eleven copies of the tuple
literal across `mujoco/rendering.py`, `newton/simulation.py` and
`simulation/base.py` - one of them written in a different order - so the side
that routes a token and the side that refuses it can no longer disagree.

The Isaac backend is deliberately unchanged: its `get_frame` looks the camera up
in `self._cameras` directly with no token check, so `"default"` there is an
ordinary addressable name and is that backend's documented signature default.
The rule follows from routing, and a backend that does not route does not get it.
