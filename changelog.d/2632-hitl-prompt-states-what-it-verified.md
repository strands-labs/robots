### Fixed

- **The `robot_mesh` approval prompt now states what it verified about the target
  instead of asserting a physical effect it never checked.** The gate is resolved
  per action, so it asks an operator about a peer without knowing what that peer
  is, and announced `Physical effect on peer '<target>'` for every gated
  single-target call. Masking the peer id, that was one sentence for three kinds of
  target: a real arm reporting `hw`, a sim twin reporting `robot_type: "sim"`, and
  a peer not on the fleet snapshot at all. `mesh.session.peer_is_physical` now
  classifies a peers-snapshot entry fail-closed - metal unless the presence shows
  it is a sim - reading the flat dict `PeerInfo.to_dict` returns, and every marker
  it consults is one the presence publisher really sets. Its two directions differ
  on purpose: `hw` is a metal marker, read permissively and checked first, while
  `robot_type` and `world` are sim markers read strictly, so a marker that cannot
  be read falls through to metal. The prompt reports that verdict in both the
  single-target and the fleet-wide scope and carries it as `physical` / `verified`
  in the interrupt's structured reason, so a host UI cannot disagree with the
  operator's sentence. Which actions are gated does not change: a `tell` aimed at a
  classified sim still stops and asks, because `robot_type` and `world` arrive over
  the wire and presence authenticates neither, so an unauthenticated self-report is
  fit to tell an operator what a peer says about itself and unfit to stand in for
  the operator.
