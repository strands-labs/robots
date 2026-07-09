# Remote (WebSocket inference)

`create_policy("remote", endpoint="ws://gpu-box:8765")` returns a
`RemotePolicy` - a drop-in [`Policy`](overview.md) that forwards observations to
a remote `PolicyServer` and returns the action chunk it computes. Use it when
the robot host cannot run a large policy locally and a GPU box can.

```python
from strands_robots import create_policy

policy = create_policy("remote", endpoint="ws://gpu-box:8765")
# smart string is equivalent:
policy = create_policy("ws://gpu-box:8765")
```

| Config key        | Default       | Meaning                                          |
|-------------------|---------------|--------------------------------------------------|
| `endpoint`        | -             | Full server URL (`ws://host:port`); wins over host/port |
| `host`            | `127.0.0.1`   | Server host (when `endpoint` is omitted)         |
| `port`            | `8765`        | Server port (when `endpoint` is omitted)         |
| `connect_timeout` | `10.0`        | Seconds to wait for the WebSocket handshake      |
| `request_timeout` | `60.0`        | Seconds to wait for each inference reply         |

The client mirrors the server policy's `requires_images`, `execution_horizon`,
`actions_per_step` and `supports_rtc`, and forwards the Real-Time Chunking
observed-delay count on every request, so a remote rollout behaves like a local
one. Install with `pip install 'strands-robots[inference]'`.

See **[Remote Policy Inference](../inference/remote.md)** for the full
two-machine setup, the server CLI, and the wire protocol.
