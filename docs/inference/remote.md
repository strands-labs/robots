# Remote Policy Inference (client/server split)

A resource-constrained robot host - an edge device or a laptop CPU - often
cannot run a large vision-language-action policy (pi0, SmolVLA, MolmoAct2) at
control rate. `strands_robots.inference` splits inference across two machines:
the robot host streams observations to a **remote GPU box** and receives action
chunks back, over a portable WebSocket protocol.

```
   robot host (CPU / edge)                         GPU box
 +-------------------------+                +-----------------------+
 |  control loop           |   observation  |  PolicyServer         |
 |  RemotePolicy  ------------- ws://  ----> |    wraps any Policy   |
 |  (Policy ABC)  <------------ chunk  ----- |    (pi0 / SmolVLA...) |
 |  applies actions        |                |    runs on GPU        |
 +-------------------------+                +-----------------------+
```

`RemotePolicy` is a drop-in [`Policy`](../policies/overview.md): anywhere a
local policy works - `sim.run_policy(...)`, `sim.eval_policy(...)`, or a
hardware control loop - a remote one works too.

## Install

```bash
pip install 'strands-robots[inference]'   # pulls websockets>=17.0 (numpy-agnostic)
```

The extra depends only on `websockets`, so it composes cleanly with `lerobot`
(`numpy>=2`) in the same environment. The `>=17.0` floor is the release whose
`Server.shutdown()` closes the connections it accepted rather than the listening
socket alone - see the teardown contract below.

## 1. Start the server (GPU box)

Serve any policy provider over a WebSocket:

```bash
python -m strands_robots.inference.server \
    --provider lerobot/act_so101 \
    --host 0.0.0.0 --port 8765
```

Or programmatically, wrapping a provider or an already-loaded policy:

```python
from strands_robots.inference import PolicyServer

# Build the policy on the server from a provider string:
PolicyServer(policy_provider="lerobot/act_so101", host="0.0.0.0", port=8765).serve()

# ...or serve a policy object you already hold:
PolicyServer(policy=my_policy, port=8765).serve()
```

`serve()` blocks. For programmatic control (tests, embedding in a larger
process) use `start()` / `stop()`:

```python
server = PolicyServer(policy=my_policy, port=0).start()  # port=0 -> OS picks one
print(server.port)
...
server.stop()
```

Either teardown stops the server *serving*, not just listening: `stop()` (and
`serve()` returning) closes the listening socket **and** every client connection
still open, and returns only once every connection handler has terminated. So the
wrapped policy is no longer invoked for a client that was already connected, and
neither call returns while a handler could still send one more action chunk. A
handler inside an inference call notices the close when that call returns, so a
teardown that lands mid-inference returns when that inference does.

That contract is the reason for the `>=17.0` floor above rather than something
this module implements: through websockets 16.x `Server.shutdown()` closed the
listening socket and nothing else, each accepted connection was served on a
thread that outlived the server object, and `stop()` returned in 0.18ms while the
same open connection went on being answered with action chunks - on a robot, the
policy still driving the arm after the operator was told the server stopped. The
teardown is graded from a client's point of view by
`tests/inference/test_a_stopped_server_stops_serving_its_clients.py`, which fails
on 16.1.1 and passes from 17.0.

`port` is an `int` in `[1, 65535]`, plus `0` for the ephemeral bind above. A
value outside that - a negative, an out-of-range number, a float, a `bool`, a
string - is refused by the constructor and by `--port`, before any policy is
built, rather than reaching the socket.

The server binds `127.0.0.1` by default. Set `host="0.0.0.0"` to accept remote
connections and wrap the link in tailscale / wireguard for production - the v1
transport is plaintext (auth/TLS is out of scope, see Non-goals).

## 2. Connect from the robot host

```python
from strands_robots import create_policy

# Named provider with an explicit endpoint:
policy = create_policy("remote", endpoint="ws://gpu-box:8765")

# ...or the smart string, which resolves to the same RemotePolicy:
policy = create_policy("ws://gpu-box:8765")
```

The server endpoint is set via `endpoint=` (with a `host=`/`port=` fallback).
`RemotePolicy` tolerates unrecognized kwargs so a shared `policy_config` can be
forwarded unchanged, but passing the endpoint under any other name (e.g. `uri=`)
logs a WARNING naming the ignored kwarg and the endpoint actually in use, rather
than silently connecting to the default `ws://127.0.0.1:8765`.

When `port=` is the effective spelling it must be an `int` in `[1, 65535]`, and
is refused before the endpoint is built. Unlike the server the client cannot
accept `0`: asking the kernel for a free port is something only the binding side
can do, so there is nothing for a client to dial. Refusing it here matters
because a WebSocket target is only resolved on first use - an unusable port is
not rejected by the transport, it surfaces later as an unreachable server and
implicates the service you were trying to reach.

Then drive it exactly like a local policy. In simulation:

```python
import strands_robots as sr

sim = sr.Robot("so101", mode="sim")
sim.run_policy(policy_provider="ws://gpu-box:8765", instruction="pick the cube", n_steps=300)
```

The connection is established lazily on first use, so constructing the policy
does not require the server to already be running.

## What the client mirrors

On connect, the server advertises the wrapped policy's introspection metadata
and `RemotePolicy` mirrors it locally, so the runtime behaves identically to
running the policy in-process:

| Property             | Effect on the robot host                                   |
|----------------------|------------------------------------------------------------|
| `requires_images`    | skip camera rendering when the remote policy does not need frames |
| `execution_horizon`  | size action chunks / re-query interval correctly           |
| `actions_per_step`   | the remote policy's trained chunk length                   |
| `supports_rtc`       | whether Real-Time Chunking blending runs server-side       |

## Real-Time Chunking across the wire

The RTC contract is preserved end to end. The runner counts how many control
steps elapse during inference and sets it via
`Policy.set_rtc_observed_delay(steps)`; `RemotePolicy` forwards that count on
every request, and the server applies it to the wrapped policy immediately
before inference. Chunk-seam blending therefore happens server-side against the
correct, deterministic step offset - identical to a local rollout. Per-episode
`reset(seed)` and `set_control_frequency(hz)` are forwarded too, so seeded
episodes stay reproducible.

Both forwarded values are validated by the policy itself, so a remote caller
reaches exactly the accepted domain an in-process one does: `hz` must be a
finite positive number and the step count `None` or a non-negative `int`. The
server passes them through verbatim rather than coercing them - JSON carries
`NaN`, `Infinity` and `true`, and coercing a `true` to `1.0` would install a 1 Hz
clock no local caller could have set. A refused value is marshalled back as the
same `RuntimeError` as any other server-side failure, before inference runs.

## Error handling

Inference failures on the server are marshalled back as an `error` message and
re-raised on the client as a `RuntimeError` carrying the server traceback - the
client never silently substitutes a zero action. An unreachable server raises a
`ConnectionError` with a hint on how to start one.

## Non-goals (v1)

- **Auth / TLS**: the transport is plaintext. Run it over tailscale / wireguard
  or an SSH tunnel for anything beyond a trusted LAN.
- **Multi-client fan-out**: one client per server. The wrapped policy holds
  per-episode state (RTC seams, diffusion RNG); the server serializes inference
  so concurrent clients cannot corrupt each other, but a dedicated server per
  robot is the intended topology.
