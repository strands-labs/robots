# LeRobot Async (gRPC inference)

`create_policy("lerobot_async", ...)` returns a `LerobotAsyncPolicy` - a drop-in
[`Policy`](overview.md) that offloads inference to a remote LeRobot
`PolicyServer` over LeRobot's **native async-inference gRPC transport**
(`lerobot.async_inference.policy_server`). Every `get_actions` call forwards the
current observation to the server, which runs the configured LeRobot policy
(ACT, SmolVLA, diffusion, pi0/pi0.5, VQBeT, ...) on its own GPU and streams back
an action chunk.

Use it when the robot host is light (e.g. a Jetson) and a separate GPU box holds
the weights. Unlike [`lerobot_local`](lerobot-local.md) (which loads the
checkpoint in-process), the async client keeps the control loop off the GPU that
runs the model - the same split as LeRobot's own `robot_client` / `policy_server`.

For a WebSocket-based alternative served by this library's own
`strands_robots.inference` server, see [`remote`](remote.md). `lerobot_async`
differs in that it speaks LeRobot's gRPC protocol directly, so it interoperates
with a stock `lerobot.async_inference.policy_server`.

## Install

```bash
pip install 'strands-robots[lerobot-async]'   # adds grpcio on top of [lerobot]
```

## Start the server (GPU host)

```bash
pip install 'lerobot[async]'
python -m lerobot.async_inference.policy_server --host=0.0.0.0 --port=8080
```

The server is generic: the **client** selects which checkpoint it loads (via the
`SendPolicyInstructions` handshake), so `policy_type` and
`pretrained_name_or_path` are required on the client.

## Use it (robot host)

```python
from strands_robots import create_policy

policy = create_policy(
    "lerobot_async",
    server_address="gpu-box:8080",
    policy_type="act",
    pretrained_name_or_path="lerobot/act_so101",
)
# smart string is equivalent (server_address parsed from the URL):
policy = create_policy(
    "grpc://gpu-box:8080",
    policy_type="act",
    pretrained_name_or_path="lerobot/act_so101",
)
```

In simulation or a control loop it is a normal provider:

```python
from strands_robots import Robot

sim = Robot("so101")  # sim by default
sim.run_policy(
    robot_name="so101",
    policy_provider="lerobot_async",
    policy_config={
        "server_address": "gpu-box:8080",
        "policy_type": "act",
        "pretrained_name_or_path": "lerobot/act_so101",
    },
    instruction="pick up the cube",
    duration=10.0,
)
```

## Config keys

| Config key                 | Default       | Meaning                                                     |
|----------------------------|---------------|-------------------------------------------------------------|
| `server_address`           | `127.0.0.1:8080` | Server `host:port` (gRPC); wins over `host`/`port`. `grpc://` scheme stripped |
| `host`                     | `127.0.0.1`   | Server host (when `server_address` is omitted)              |
| `port`                     | `8080`        | Server port (when `server_address` is omitted)              |
| `policy_type`              | -             | **Required.** LeRobot policy type the server loads -- the async-servable set from lerobot's `SUPPORTED_POLICIES` (`act`, `smolvla`, `diffusion`, `tdmpc`, `vqbet`, `pi0`, `pi05`, `groot`, ...); validated client-side against the live lerobot constant |
| `pretrained_name_or_path`  | -             | **Required.** HuggingFace id or path the server loads       |
| `device`                   | `cuda`        | Device for **server-side** inference; set `cpu` for a CPU server |
| `actions_per_chunk`        | `50`          | Max actions the server returns per chunk                    |
| `actions_per_step`         | `actions_per_chunk` | Actions executed from one chunk before re-querying (the re-query interval) |
| `connect_timeout`          | `10.0`        | Seconds to wait for the gRPC `Ready` handshake              |
| `request_timeout`          | `60.0`        | Seconds to wait for each observation/action RPC             |
| `rename_map`               | `{}`          | `{robot_obs_key: model_feature_key}` forwarded to the server; renames observation keys before the policy sees them (async analog of `lerobot_local`'s `obs_rename`) |

## Notes

- The connection is established lazily on the first `get_actions`, so
  constructing the policy does not require the server to be up yet.
- If the server returns no actions for an observation (filtered out, or a
  server-side inference error), the client **raises** rather than fabricating a
  zero action - check the `PolicyServer` logs.
- `set_robot_state_keys([...])` must be called with the robot's joint/motor
  names before inference; those scalars are concatenated into
  `observation.state` and any RGB/depth camera arrays are declared as image
  features in the handshake.
- When the checkpoint expects camera/state keys that differ from the ones the
  robot exposes, pass `rename_map={robot_obs_key: model_feature_key}`. The
  server applies it as a `RenameObservationsProcessorStep` before inference, so
  a stock checkpoint (e.g. one trained with `observation.images.laptop`) is
  reachable from a robot whose camera is named `front` without re-exporting the
  model - the remote counterpart of the [`lerobot_local`](lerobot-local.md)
  provider's `obs_rename`:

  ```python
  policy = create_policy(
      "lerobot_async",
      server_address="gpu-box:8080",
      policy_type="act",
      pretrained_name_or_path="lerobot/act_so101",
      rename_map={"observation.images.front": "observation.images.laptop"},
  )
  ```
