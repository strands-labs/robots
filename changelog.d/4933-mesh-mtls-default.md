### Changed: a zenoh mesh with identities runs over mutual TLS, and every peer proves who it is

- **Identities:** `python -m strands_robots.mesh ca`, `cert NAME [--serves IDS]` and `config` make a lab CA, issue
  each machine an ECDSA certificate and key (written to `~/.strands/mesh/identity` by default), and write zenoh
  config files for routers or other zenoh clients. A `zenoh://` mesh uses that identity (or `?identity=<dir>`,
  `STRANDS_MESH_IDENTITY`; `none` opts out). It then speaks only TLS links with client certificates from the lab
  CA, so a machine without one cannot join, read, answer or query. LAN discovery still needs no endpoints.
- **Bound to the certificate:** every envelope is signed with the certificate's key. Peer ids are
  `<certificate name>.<random>`. A robot refuses unsigned requests, requests for a peer id that is not the
  signer's, repeats and requests more than 30 s off its clock, before its watchdog hears of them, so no machine can
  send heartbeats for another. A request is answered only on the key it names (tenant, device and rpc), so a captured request for one device cannot be replayed onto another. Replies, announcements and state count only when signed by a certificate whose
  `--serves` covers that robot. Another machine can't answer for a robot, announce a robot, or make a request
  fail by answering. The audit record names the signing certificate. An e-stop is never refused for its signature.
- **Real robots:** a real robot moves over zenoh only on a mesh with identities, or with `?trusted=1` /
  `STRANDS_ZENOH_TRUSTED=1`. A zenoh config file no longer lifts observe-only by itself: it has to turn on mutual
  TLS on TLS-only links with PEM certificate, key and CA files (then it is the identity). A file with only user and
  password authentication needs `?trusted=1`. Every peer of a robot with an identity needs this version.
