### Added: the dashboard connects other peers' robots over the mesh, and policies served elsewhere

`GET /api/connect/peers` lists every device on the meshes the page joined (`POST /api/connect/mesh {"address":
"host:port" | "lan"}`); `POST /api/connect/robot {"peer", "address"}` attaches one as a card through
`mesh.connect(id, images=True)`. An attached robot is a hardware robot to the page: every act, run and resume needs
`X-Confirm: yes`, its own device's safety chain still runs, and its e-stop is one ungated click that latches the device
itself. `POST /api/connect/robot/{id}/resume` asks the device's authority first (the page's operator by default) and
returns its refusal verbatim. `GET /api/robots/{id}/cameras/{cam}.png` serves a hardware or remote robot's own camera
frame (never a sim render). `POST /api/connect/policy {"endpoint", "kind": "lerobot-async" | "onnx" | "lerobot" |
"dataset", "deployment"}` probes a `lerobot-policy-server` for its descriptor (or validates the spec) and lists it in
`/api/policies`, so the existing run route starts it on any robot.
