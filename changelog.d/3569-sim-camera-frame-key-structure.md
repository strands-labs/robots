### Fixed: a sim camera's name may not carry structure its own frames cannot travel under

`add_camera` accepted any string as a camera name, but that name is the key the
camera's frames travel under: the mesh publishes each frame on
`strands/<peer_id>/camera/<name>` and the IoT offload joins the name into
`<prefix>/<peer_id>/<name>/<ts>.jpg`. Measured on one `create_world` (zenoh
1.10.1), `'**'`, `'*'`, `'a$b'`, `'cam#1'`, `'cam?1'`, `'a+b'`, `'a//b'`,
`'/wrist'`, `'..'`, `'.'` and `'sub/../etc'` all reported `status="success"`,
registered and compiled into the model - and then a wildcard name was routed by
intersection, delivering one camera's frames to peers subscribed to a different
camera; a name carrying a character Zenoh forbids made the topic unpublishable,
which the publish loop logs at debug, so the camera rendered forever and never
reached the mesh; and `s3_key_for('rover01', '..', 123)` resolved to
`frames/123.jpg`, outside the peer's own prefix.

The shared `camera_name_error` every backend's `add_camera` reads now also
applies `camera_frame_key_error`, so the rule holds on MuJoCo, Newton and Isaac
alike. It is deliberately narrower than the bare-token rule the hardware
`cameras` mapping applies: a sim camera name carries the backend's own
namespacing, so `arm0/wrist_cam` - the form recording documents - is still
accepted, as are a name with a space or an inner dot. Only what no consumer can
carry is refused.
