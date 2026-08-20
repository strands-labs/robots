### Fixed

- **docs**: every enumeration of the `robot_mesh` human-in-the-loop gate now names `rpc`, the Device Connect device-native call, which has been gated by default since Device Connect landed. The README's `STRANDS_MESH_HITL_ACTIONS` row previously documented a token vocabulary that could not express the shipped default gate, so an operator narrowing the gate from that list dropped `rpc` from it silently - the value parses, so no refusal fires - and the one-time warning logged when the gate is disabled named five of the six actions it had just left unguarded.
