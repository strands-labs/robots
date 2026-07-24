# `vera` — VERA video-to-action policy provider

MIT/CSAIL **VERA** (Video-to-Embodied Robot Action) as a strands-robots policy:
a two-stage **DFoT/WAN video planner + Jacobian IDM**, served from the
`strands-vera-server` container over WebSocket. The host venv needs no VERA
install — only the tiny `websockets`+`msgpack` client transport.

```python
from strands_robots.policies import create_policy

policy = create_policy("vera", embodiment="mimicgen", server_mode="docker", ckpt_root="/abs/path/vera-ckpts")
chunk = policy.get_actions_sync(observation, "stack the red block on the green block")
```

## Layout

```
vera/
├── provider.py        VeraPolicy(Policy) — context window, action queue, IK opt-in
├── client.py          VeraWebsocketClient (msgpack+ws, no vera import)
├── _msgpack_numpy.py  vendored numpy codec
├── config.py          VeraConfig (env-overridable; subprocess|docker server modes)
├── server_runner.py   VeraServerRunner (subprocess) + DockerServerRunner
├── ee_frame.py        end-effector frame auto-discovery (eef/cartesian-delta IK)
├── sim_ik.py          VERA eef-delta chunk -> MuJoCo joint targets (mink)
└── docker/            Dockerfile + entrypoint + compose + offline ckpt resolver
```

## Full documentation

→ **[`docs/policies/vera.md`](../../../docs/policies/vera.md)** — embodiments,
checkpoints, server setup, configuration, wire protocol, testing.

→ **Example:** [`examples/vera_mimicgen_panda/`](../../../examples/vera_mimicgen_panda/)
(end-to-end MuJoCo rollout + video: WAN planner + Jacobian IDM -> eef-delta -> IK -> Panda).

→ **Container:** [`docker/README.md`](docker/README.md).
