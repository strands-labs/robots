---
description: Security considerations for deploying Strands Robots past a trusted lab - prompt injection, mesh authentication, operator approval, remote-code policies, inference containers, hardware access, secrets, and telemetry exposure.
---

# Security considerations

Strands Robots actuates machines in physical space, pulls models and datasets from the network, runs containers, and coordinates fleets. Before you move any configuration past a trusted lab and toward production, work through the considerations below.

1. [Prompt injection](#prompt-injection)
2. [Robot mesh authentication](#robot-mesh-authentication)
3. [Operator approval for fleet-wide actions](#operator-approval-for-fleet-wide-actions)
4. [HuggingFace policy code execution](#huggingface-policy-code-execution-trust_remote_code)
5. [GR00T inference containers](#gr00t-inference-containers)
6. [Hardware and serial access](#hardware-and-serial-access)
7. [Credentials and secrets](#credentials-and-secrets)
8. [Telemetry exposure to the agent context](#telemetry-exposure-to-the-agent-context)

!!! danger "Do not report vulnerabilities here"
    Do not open a public GitHub issue for security concerns. Report via the AWS Vulnerability Disclosure Program on [HackerOne](https://hackerone.com/aws_vdp) or email [aws-security@amazon.com](mailto:aws-security@amazon.com). See [SECURITY.md](https://github.com/strands-labs/robots/blob/main/SECURITY.md).

## Prompt injection

Supplying untrusted data into agents can lead to prompt injection, where untrustworthy context is treated as LLM instructions. Given the actuation of these robots in physical space, this is an important risk to track. To mitigate this behavior, developers should be careful to feed the robots only data that comes from a trusted source. If not all input data can be trusted, developers should restrict the tools available to the agent to prevent the robots from making safety-critical actions.

In practice, untrusted content can reach the agent through more than the operator's prompt: task instructions broadcast over the mesh, camera/observation text surfaced back into context, dataset metadata, and model/checkpoint descriptions pulled from the Hub are all potential injection vectors. Treat every one of these as untrusted input.

Defense-in-depth controls the SDK already provides, and which you should rely on rather than disable:

- **Tool scoping.** The single most effective mitigation is to give the agent only the tools a task actually needs. An agent that never receives the `robot_mesh`, `serial_tool`, or `Robot(mode="real")` tools cannot be coerced into a fleet broadcast, a raw serial write, or a physical actuation no matter what the injected text says.
- **Out-of-band human approval for physical actuation** (see [Operator approval](#operator-approval-for-fleet-wide-actions)) - the approval is delivered outside the LLM's tool-argument flow, so an injected prompt that tries to set an "approved" flag in the command body cannot bypass the gate.
- **Operator approval for ROS 2 command surfaces.** The `use_ros` tool gates every verb that can carry a command to a robot - a topic `publish`, a `service_call` and an `action_send_goal` - against a blocklist of safety-critical surfaces (`/cmd_vel`, the joint command/trajectory topics, e-stop, motor enable/disable, `/navigate_to_pose`, `/follow_path`). Reading the same surfaces stays ungated. See [safety-critical command surfaces](ros2-integration.md#safety-critical-command-surfaces-need-operator-approval).
- **Payload validation** of every mesh command, so an injected instruction still cannot smuggle an out-of-bounds duration, an attacker-controlled inference host, or an arbitrary model path.

## Robot mesh authentication

Joining the Zenoh peer mesh is opt-in: `Robot(name, mesh=True)` (or `STRANDS_MESH=true`) joins one, a `Simulation` built directly joins by being assigned a started client, and a bare `Robot()` exposes no mesh surface at all. Once a robot has joined, the `robot_mesh` tool lets an agent enumerate, command, and broadcast to every peer on that mesh, and the security of the mesh is governed by `STRANDS_MESH_AUTH_MODE`.

### Development posture (insecure)

The example scripts run the mesh without authentication or access controls so they work out of the box. Any device on the same network can then send commands to the robot fleet. This is acceptable for trusted, isolated development environments, but is not suitable for untrusted networks or production.

The development posture is selected one of two ways:

- `STRANDS_MESH_LOCAL_DEV=1` - the developer preset. It defaults the auth mode to `none` and satisfies the insecure-acknowledgment second factor by itself.
- `STRANDS_MESH_AUTH_MODE=none` together with `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE=1` - the explicit form. `none` on its own is rejected; the second factor is required so you cannot disable wire security by setting a single variable.

!!! warning "Wire security off is a loud signal"
    When wire security is off, the SDK logs a loud error on every session open (`WIRE SECURITY DISABLED - STRANDS_MESH_AUTH_MODE=none`). Treat that line as a signal that the process must never be on a shared or hostile network.

### Production posture (required off trusted networks)

For untrusted networks or production fleets, `STRANDS_MESH_AUTH_MODE=mtls` is required (and is the default when neither dev flag is set). mTLS authenticates peers at the transport layer (`transport/link/tls`) before any command is dispatched.

mTLS alone is not sufficient - pair it with an access-control list:

- The built-in default ACL is permissive: any CA-signed peer may publish and subscribe on any key. If you forget to supply an ACL, the SDK warns on every session open.
- Supply an operator ACL via `STRANDS_MESH_ACL_FILE` that enumerates each peer's certificate CN and the key expressions it may use. See [`examples/mesh/mesh_acl_example.json5`](https://github.com/strands-labs/robots/blob/main/examples/mesh/mesh_acl_example.json5) and [`examples/mesh/mesh_acl_strict_per_peer.json5`](https://github.com/strands-labs/robots/blob/main/examples/mesh/mesh_acl_strict_per_peer.json5).
- `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL=1` exists only to silence the permissive-ACL warning when you have deliberately accepted it (e.g. a closed lab). Do not set it in production - it does not make the mesh safer, it only quiets the reminder that it is not.
- An ACL file the loader cannot read is refused, not ignored. A missing, oversize, non-UTF-8, malformed, or too-deeply-nested file is reported as an unloadable ACL, which the start-time gate treats as the permissive default and therefore refuses to bring the wire up. The error names the path and the reason, so a typo in the ACL stops the mesh rather than quietly widening it.

> ⚠️ **WAN/cloud Zenoh routers MUST deploy a topic-level ACL.**
> mTLS authenticates *who* a device is; it does **not** restrict *what* topics
> that device may touch. A cloud/WAN Zenoh router with mTLS but no topic-level
> ACL grants every authenticated device cert ambient authority over the whole
> fleet: any one cert can subscribe to `**` (all devices' state, camera, and
> input streams) and publish to any device's `cmd` topic. A single
> compromised or stolen device certificate then becomes full read/write
> control of every robot: one cert can subscribe to and command the entire
> fleet.
>
> mTLS gives you **identity**; the ACL gives you **least privilege**. You need
> both. Adapt [`examples/mesh/mesh_acl_strict_per_peer.json5`](https://github.com/strands-labs/robots/blob/main/examples/mesh/mesh_acl_strict_per_peer.json5)
> to your fleet (it pins each peer's certificate CN to the exact key
> expressions it may publish/subscribe) and deploy it on **every
> internet-facing router**, not just on LAN peers. Without it, an
> internet-reachable router is an unauthorized-fleet-control surface even
> with mTLS enforced.

### Cross-network fleets (AWS IoT Core)

Adding the `[mesh-iot]` extra routes traffic through AWS IoT Core (MQTT5 with mTLS), and a `BridgeTransport` keeps high-rate topics local while bridging presence, health, and safety to the cloud. When you use this path, the IoT device certificates and provisioning material become production secrets - provision them per-device, scope their IoT policies to the minimum topic set, and rotate/revoke them like any other fleet credential.

Reference: `strands_robots.mesh.session`, `strands_robots.mesh._acl_config`, `strands_robots.mesh.transport.iot_transport`.

## Operator approval for fleet-wide actions

The `broadcast` and `emergency_stop` actions on the `robot_mesh` tool affect every peer on the network. To prevent an agent from issuing fleet-wide commands autonomously (or under prompt injection), both actions are gated behind a human-in-the-loop interrupt. When the agent invokes either action, the Strands runtime pauses the agent loop and asks the operator to approve out-of-band of the LLM's tool arguments. Per-action rate limits, command validation, and an audit trail run alongside the interrupt. Outside an agent loop (a bare script or unit test), both actions fail closed.

What this looks like in practice, and how to configure it:

- **The default gate is broader than just fleet-wide actions.** Out of the box, every physical-actuation action is gated: `emergency_stop`, `broadcast`, `tell`, `send`, `stop`, and `rpc` (the Device Connect device-native call). A prompt-injected agent therefore cannot drive any physical command - single-peer or fleet-wide - without an explicit operator approval.
- **Approval is an explicit affirmative.** Only `y` / `yes` / `approve` / `approved` count as approval; anything else (including an empty response) is treated as a decline.
- **`STRANDS_MESH_HITL_ACTIONS` tunes the gate.** You can widen it to `all` (also gates the read-only `subscribe` / `watch` telemetry actions), narrow it to a comma-separated subset, or set it to `none`. Setting `none` re-opens the entire physical-actuation surface to the LLM without confirmation - the SDK logs a one-time warning when this is in effect. Do not use `none` outside a fully trusted, non-networked test.
- **The prompt states what it verified about the target, and the verdict is not an authorisation.** The gate is per-ACTION, so it asks about a peer without knowing what that peer is. It used to announce `Physical effect on peer '<target>'` for every gated single-target call, which was the same sentence for a real arm, a sim twin, and a peer this process has not discovered - a claim it had never established. It now reports `peer_is_physical`'s reading of the peer's presence: `it reports real hardware (so101_follower)`, `it reports itself as sim`, or `it is not on the fleet snapshot, so it cannot be shown to be a sim`. The classifier fails closed - a peer is metal unless its presence SHOWS it is a sim - and the same verdict is carried as `physical` / `verified` in the interrupt's structured reason so a host UI cannot disagree with the operator's sentence. It does NOT change which actions are gated: a `tell` aimed at a sim still stops and asks. `robot_type` and `world` arrive over the wire from the peer itself and presence authenticates neither, so a peer can claim to be a sim; an unauthenticated self-report is fit to tell an operator what a peer says about itself and unfit to stand in for the operator.
- **Rate limits bound LLM-driven nuisance** independently of approval: `emergency_stop` is capped at 3/min, `broadcast` at 10/min, `tell`/`send` at 30/min. A declined approval does not consume a slot, so an operator declining nuisance prompts can never lock themselves out of issuing a genuine emergency stop. A slot is reserved atomically at the point the action is known to run, so concurrent invocations cannot exceed the cap - which matters most when `STRANDS_MESH_HITL_ACTIONS` narrows the gate, because the cap is then the only bound left.
- **Audit trail.** Every `tell` / `send` / `broadcast` / `stop` / `emergency_stop` / `rpc` - and every approval, decline, validation rejection, and rate-limit rejection - is written to the safety audit log. The read-only actions are recorded too (`peers`, `status`, `subscribe`, `watch`, `inbox`, `unsubscribe`), on whichever backend served them, so the log is a record of what the agent *read* about the fleet and not only of what it told the fleet to do - `peers` returns every device id and every function name the fleet exposes, which is the callable surface a later `rpc` would use. Make sure your deployment actually captures and retains that log; it is your forensic record of both.
- **Payload bounds.** `validate_command` bounds every field of an incoming `execute` / `start` command before it reaches the dispatcher: the action must be in the allowlist, `duration` and `policy_port` are range-checked, `policy_host` / `server_address` / `model_path` are allowlist-gated, and every string field is length-bounded and refused C0/DEL/C1 control characters. That last check is why the audit trail above is worth keeping: `instruction` is free-form text from a remote peer, so admitting a CR or LF in it would let one log call emit two records and let the second impersonate a different level and logger. Natural-language fields bound only the control range - a non-ASCII instruction is admitted - while identifier fields such as `robot_name` stay printable-ASCII-only.

Reference: `strands_robots.tools.robot_mesh`.

## Refusal codes are the stable contract; prose is not

Some refusals are *continuable*: the request was well formed, and an operator who accepts the risk can grant something that makes the identical request succeed. An untrusted policy provider, a repo or host or policy type outside a mesh allowlist, a teleop frame past the value envelope - each of these has an operator answer behind it. Anything that offers that answer (a UI consent card, an approval endpoint, a supervising agent) first has to recognise *which* refusal it is looking at and *what* it is about.

Recognise it by its `code`, never by its message text:

```python
from strands_robots import refusal_codes
from strands_robots.mesh.security import ValidationError, validate_command

try:
    validate_command(cmd)
except ValidationError as refusal:
    if refusal.code == refusal_codes.HF_REPO_NOT_ALLOWED:
        offer_to_allowlist(refusal.subject)          # the repo, already parsed out
        print(refusal_codes.REFUSAL_GRANTS[refusal.code])   # STRANDS_MESH_HF_REPO_ALLOW
```

- **`code` is stable; the message is not.** `code` is a member of `refusal_codes.REFUSAL_CODES`, a closed vocabulary you may switch on. The message is an operator-facing sentence and may be reworded at any time to explain the refusal better. Matching on prose - looking for an env-var name in the sentence, or pulling the subject back out with a regex - couples you to wording that is free to change, and neither this package's tests nor yours will notice when it does.
- **`subject` is what the refusal is about**, so you do not have to parse it back out: the repo id, the host or whole `server_address`, the policy type or provider name, the joint key, the refused provider.
- **`REFUSAL_GRANTS` names the environment variable that lifts each refusal.** Read it from there rather than hard-coding it, so your consumer and this package cannot drift apart. The variable is half the answer - you also need to know what to set it to, and that differs by code. `HF_REPO_NOT_ALLOWED`, `POLICY_TYPE_NOT_ALLOWED` and `POLICY_HOST_NOT_ALLOWED` are allowlists you add `subject` to, as above. `TRUST_REMOTE_CODE_REQUIRED` is a flag you set to `1`, and `TELEOP_VALUE_OUT_OF_RANGE` is a bound you raise above the refused magnitude - for those two the subject is not the value, and setting it to the subject is a silent no-op that returns the identical message. Each code states its own operation in `refusal_codes`.
- **A refusal with no code is not continuable.** `code` is `None` for rejections an operator cannot grant their way past - a schema failure, an over-long instruction, a lockout. There is nothing to offer, so there is nothing to recognise. Treat `code is None` as "show the message and stop", not as an unknown code.
- **Codes are additive.** They were introduced without changing a single refusal message, and new codes may be added for refusals that become continuable later. Switch on the codes you know and fall through to the message for the rest.
- **Every code you receive is in `REFUSAL_CODES`.** Nothing validates `code` at runtime - it is stored as given - so what backs the closed vocabulary is a static scan over every raise site in the package, reading the code in each spelling a site can use it (a `refusal_codes` attribute, a name imported from it, or the literal string). That means `REFUSAL_GRANTS[refusal.code]` is safe for any code you are handed: a code outside the vocabulary is a defect in this package, not a case for you to handle.

Reference: `strands_robots.refusal_codes`; `strands_robots.mesh.security.SecurityError`; `strands_robots.policies.factory.UntrustedRemoteCodeError`.

## HuggingFace policy code execution (`trust_remote_code`)

Some policy providers load models from the HuggingFace Hub with `trust_remote_code=True`. That flag instructs the HuggingFace libraries to download and execute Python code from the model repository on your machine, with the privileges of the process running the agent. A malicious or compromised model repository can therefore achieve arbitrary code execution - read your credentials, open a reverse shell, or command your robot directly - simply by being loaded.

Because this is code execution, not just data loading, Strands Robots forces an explicit, deliberate opt-in before any such provider will load:

- The providers `lerobot_local` (`LerobotLocalPolicy`) and `kimodo` (`KimodoPolicy`) are on the remote-code list. Any provider that loads models with `trust_remote_code=True` must be listed in `_HF_REMOTE_CODE_PROVIDERS` so the opt-in is enforced.
- Loading is blocked by default. Attempting to create a gated provider without opting in raises `UntrustedRemoteCodeError` with an explanation, rather than silently executing remote code.
- To opt in, set `STRANDS_TRUST_REMOTE_CODE=1` (`1` / `true` / `yes` are accepted). The example CLI enforces the same gate before it will run `--policy lerobot_local`.

Operator guidance:

- Only set `STRANDS_TRUST_REMOTE_CODE=1` when you are loading checkpoints from organizations you trust - ideally your own org, or a small allowlist of vendors you have vetted (e.g. `lerobot/`, `nvidia/`). The opt-in is a per-process, whole-environment switch: once set, it trusts every model the process loads for the life of that process, not just the one you had in mind. Scope it tightly (set it on the specific command, not globally in a shell profile) and pin checkpoints to a known revision where the loader supports it.
- Where a provider exposes a per-call `trust_remote_code` setting, use it rather than relying on the environment variable alone. `KimodoPolicy` takes `trust_remote_code` (default `False`), so a process that has opted in to the provider can still refuse to execute a given repository's code. The two are independent: the environment variable decides whether the provider may be built, the setting decides whether a checkpoint's code runs.
- Prefer providers that do not require remote code where you can. The default Mock policy, the GR00T container path, and many LeRobot policy families do not need this flag. Reach for `lerobot_local` with `trust_remote_code` only when a specific model genuinely requires it.
- A mesh peer can request a model load too. When the mesh forwards a `pretrained_name_or_path` in an `execute`/`start` command, it is additionally constrained to an org allowlist (`STRANDS_MESH_HF_REPO_ALLOW`, default `nvidia,huggingface,lerobot`) so an authenticated peer cannot steer a robot into loading an arbitrary repo. Keep that allowlist as narrow as your fleet allows, and remember it is independent of the per-process `STRANDS_TRUST_REMOTE_CODE` opt-in - both gates apply.

Reference: `strands_robots.policies.factory` (`_check_trust_remote_code`, `UntrustedRemoteCodeError`).

## GR00T inference containers

The `gr00t_inference` tool pulls a Docker image, downloads a checkpoint, and starts a container. The agent-facing surface is intentionally constrained, and you should keep it that way:

- The agent cannot choose the image, bind-mount host paths, or inject a container command - those are operator-config-driven only. The image is resolved from `STRANDS_GR00T_IMAGE` and checked against an allowlist (`STRANDS_GR00T_IMAGE_ALLOW`), and a guard blocks dangerous bind mounts (`/`, `/etc`, the Docker socket, `/proc`, `/sys`, credential dirs, ...) that would amount to host takeover.
- Keep `STRANDS_GR00T_IMAGE_ALLOW` and `STRANDS_GR00T_REPO_URL_ALLOW` narrow and exact; the SDK matches repo URLs exactly (no wildcard) specifically so a look-alike repo (`...Isaac-GR00T-evil`) cannot slip past.
- Running the container still grants it a GPU and network. Run inference hosts with least privilege, on isolated networks where practical, and tear containers down when done (`gr00t_inference(action="stop", ...)` or `lifecycle="teardown"`).

Reference: `strands_robots.tools.gr00t_inference`.

## Hardware and serial access

`Robot(mode="real")` and the `serial_tool` give the agent direct control of physical actuators over serial/USB devices. Three implications:

- **Physical safety is in scope.** A wrong or malicious command moves a real arm. Maintain a physical e-stop, keep humans clear of the workspace during autonomous runs, and prefer validating any new task in simulation (the safe default) before switching the one keyword to `mode="real"`.
- **Calibration files** under `~/.cache/huggingface/lerobot/calibration/` define how joint commands map to the physical device. Protect them as integrity-sensitive configuration - corrupted or swapped calibration can produce unexpected motion.
- **The `serial_tool` is broad.** It can enumerate and write to any serial port the process can see, not just the intended robot. Scope it out of agents that do not need raw device access (see [Tool scoping](#prompt-injection)).

## ROS 2 / DDS bridge command surface

`Robot(ros2_bridge=True)` can expose an inbound `/<robot>/joint_command` topic that drives the physical arm. Because **any participant on the DDS domain can publish to it**, the command surface is hardened:

- **DDS Security gate (pure-RTPS transport).** When `ros2_transport="rtps"` and commands are enabled, the bridge **refuses to start** unless given a `dds_security_config` (identity CA, participant certificate + private key, signed governance + permissions) or the explicit `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE=1` opt-out. The credentials are wired into the cyclonedds participant QoS so the whole graph is authenticated and access-controlled. The rclpy transport gets its DDS Security from the ROS 2 RMW keystore/env (`ROS_SECURITY_*` / `sros2`) instead. See the [RTPS integration guide](rtps-integration.md#securing-the-inbound-command-surface).
- **Joint position bounds.** Pass `joint_limits={"<motor>.pos": (min, max)}` to reject any inbound command whose any joint falls outside its declared range - the whole command is dropped, never partially applied, so one out-of-range value cannot drive part of the arm. Keys are matched against the joint names the command carries - the same `<motor>.pos` names the bridge publishes in `joint_states`, so a controller can echo them straight back - and a key that names no commanded joint constrains nothing. Every bound must be a finite number; a non-finite one declares a range that admits nothing, and the bridge refuses it at construction rather than dropping every command for that joint mid-run.
- **Agent-side command gate (`use_ros`).** The bridge protections above harden the *inbound* surface a robot exposes. The `use_ros` tool is the other direction - an agent publishing onto someone else's graph - and it gates the safety-critical surfaces behind an operator interrupt, keyed on the surface name so a topic publish, a service call and an action goal are all covered. `RosBridgedRobot` forwards the operator context into it, so a bridged robot driven by an agent prompts too, while a programmatic `robot.drive(...)`/`stop()` needs the surface pre-approved. Pre-approve surfaces with `STRANDS_ROS2_COMMAND_ALLOW` - matched by base name as well as exactly, so a `/cmd_vel` entry lifts the gate on every namespaced `cmd_vel` and not only the robot being driven; name the namespace to scope it to one - or bypass with `BYPASS_TOOL_CONSENT=true`; with neither set and no interrupt reachable it fails closed. See [safety-critical command surfaces](ros2-integration.md#safety-critical-command-surfaces-need-operator-approval).
- **Telemetry-only is ungated.** `ros2_commands=False` is publish-only (no inbound surface) and needs no security config. That posture is only as good as how the flag is read, so `ros2_bridge` / `ros2_commands` (and `enable_commands` on either bridge class directly) are **checked** against the shared boolean domain rather than read by truthiness: a non-boolean is refused at construction, so a deployment config that spells the flag `"false"` cannot select the surface it asks to close.

## Credentials and secrets

The product touches several classes of secret. Handle each per least-privilege:

- `HF_TOKEN` is only needed to push datasets or pull gated checkpoints, and should be scoped to write only when you actually push. The default sim/Mock path needs no token at all - do not export one where it is not required.
- AWS credentials drive the Bedrock model provider; scope them to the specific Bedrock model/region in use.
- mTLS certificates and AWS IoT provisioning material (production mesh) are fleet-wide secrets - provision per device, store securely, and rotate/revoke on decommission.

Avoid baking any of these into images, example scripts, or notebooks.

## Telemetry exposure to the agent context

The `subscribe` / `watch` actions can pull mesh telemetry into the LLM context. By default they are restricted to a narrow set of low-impact, fleet-shared topics (presence, health, safety); subscribing to another peer's command, state, camera, or input streams is blocked, with the transport ACL as the primary control and the tool-layer allowlist as defense in depth. If you extend `STRANDS_MESH_SUBSCRIBE_ALLOW`, avoid wildcard patterns that would let the agent observe (and exfiltrate into its context) another peer's control or sensor streams.
