### Fixed: the hub-to-hardware example's robot joins the mesh its Step 5 broadcasts over

Joining the Zenoh mesh is opt-in, but `examples/lerobot/hub_to_hardware.py` and
its notebook still built a bare `Robot("so100")`, so Step 5 asked `robot_mesh`
to list and broadcast to a mesh nothing had joined. Run with a real Bedrock agent,
the reply was "0 local and 0 peers" and no broadcast was sent. Both now pass
`mesh=True`; re-executed, the agent lists two peers and the broadcast stops at the
`robot_mesh-broadcast-approval` gate the example's own note describes.

The notebook's Step 3 and Cleanup text also still sent readers to the removed
`gr00t_inference` container and the removed `groot` provider. It now names
`lerobot_local` (which carries GR00T N1.7) and `remote`, matching the folder
README.
