### Added: the operator dashboard's workflow - live fleet, devices, record, train, calibrate, voice

`python -m strands_robots dashboard` now serves the full operator SPA and joins
the mesh as a robot-less gateway. What landed on `main` before this was the rails
(passkey auth, the sim twin, the Agent tab with consent cards, the sim e-stop,
settings); this adds what an operator does with them: see every live peer and
task it, pair a leader and follower for teleop, spawn a robot process for a
serial port and give it cameras, record a LeRobot dataset episode by episode with
thumbnails and labels, submit and watch a training job, search checkpoints,
validate a policy against a robot, get a deploy snippet, run the calibration
wizard, and talk to the fleet through a speech-to-speech operator.

One router per concern (`routes_mesh`, `routes_devices`, `routes_record`,
`routes_train`, `routes_config`, `routes_voice`), each attaching its state and
its startup / shutdown work through `app.state.startup_hooks` and
`shutdown_hooks`; `server.create_app()` stays a wiring file.

Two things are stricter than the draft this comes from. Every client-named path
(dataset root, output directory, checkpoint) is resolved and must sit under its
home, refused otherwise with a sentence that does not leak whether it exists.
And every action that moves real hardware - a task on a physical peer, a
calibration run, a spoken command - fails closed without a confirm or the
standing `STRANDS_DASH_AGENT_PHYSICAL_MOTION` grant, and the refusal carries a
`needs_consent` block so the page offers the grant instead of a wall. The fleet
e-stop lives at `/api/mesh/safety/estop` beside the sim-only `/api/safety/estop`
and reports `responses_received` / `peers_not_stopped`.

The UI is the React SPA, built and committed under `dashboard/static/` (source
in `dashboard/frontend/`, `npm ci && npm run build`); no node at runtime. The
`[dashboard]` extra gains `uvicorn[standard]` (websockets), `pyserial` and
`opencv-python-headless`.

Tests for the new routers land in follow-up PRs; this one carries the surface
so the review can be about the surface.
