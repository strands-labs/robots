---
description: Every environment variable the package reads, the asset cache layout, and the CA pin rotation runbook.
---

# Configuration

### Environment variables

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ROBOT_MODE` | `Robot()` factory mode: `sim` / `real` / `auto` | `sim` |
| `STRANDS_ASSETS_DIR` | Robot model asset cache directory | `~/.strands_robots/assets/` |
| `STRANDS_MEMORY_DIR` | Harness memory store (`harness_memory` tool: task solution traces + global success rules / failure models) | `~/.strands_robots/memory/` |
| `STRANDS_ROBOTS_RENDER_ROOT` | Sandbox directory that `Simulation.render(output_path=...)` may write into; per instance, `Simulation(render_dir=...)` / `Robot(name, render_dir=...)` takes precedence | `~/.strands_robots/renders/` |
| `STRANDS_ROBOTS_RENDER_ALLOW_ABS` | Set `1` to allow `render(output_path=...)` to write absolute paths outside the render sandbox | unset |
| `STRANDS_ROBOTS_RENDER_MAX_BYTES` | Max PNG size `render(output_path=...)` will persist | `52428800` (50 MB) |
| `STRANDS_ROBOTS_SCENE_ROOT` | Where a *relative* `export_xml(output_path=...)` lands, and the directory `load_scene(scene_path=...)` searches when a relative path is not where the caller spelled it; absolute paths are used as given | `~/.strands_robots/scenes/` |
| `STRANDS_ROBOTS_VIDEO_ROOT` | Opt-in sandbox for video/recording output paths (`run_policy(video=...)`, `start_cameras_recording`). Unset = absolute paths allowed (historic contract); set to confine writes | unset |
| `STRANDS_ROBOTS_VIDEO_ALLOW_ABS` | Set `1` to re-permit absolute paths when `STRANDS_ROBOTS_VIDEO_ROOT` is set | unset |
| `STRANDS_TRUST_REMOTE_CODE` | Set `1` to allow HF `trust_remote_code` for `lerobot_local` | unset |
| `STRANDS_TRAIN_EXTRA_FLAGS_ALLOW` | Comma-separated `lerobot_train` `extra_flags` names pre-approved past the operator gate on the flags that control output paths, telemetry and code loading (`output_dir`, `config_path`, `wandb.enable` / `.project` / `.entity` / `.api_key`, `dataset.root`, `policy.pretrained_path`, `push_to_hub`, `policy.push_to_hub`, `hub_repo_id`). An entry names the blocked flag, so it also clears every argparse abbreviation of it (`ou` for `output_dir`); the two `push_to_hub` spellings are separate flags and one entry clears one of them. A blocked flag no entry names prompts the operator, and is refused where no operator can be prompted - this variable is the headless remedy that refusal names. `BYPASS_TOOL_CONSENT=true` waives the gate for every flag with a WARNING | unset (every blocked flag prompts) |
| `STRANDS_TRAIN_RDZV_TIMEOUT_S` | Seconds a multi-GPU training launch (`num_gpus > 1` on `LerobotTrainer`, `GrootTrainer` or `Cosmos3Trainer`) may spend in torch's elastic rendezvous before it fails rather than waiting. Handed to torch, whose own defaults are minutes long and spent inside C++ socket code no Python timeout can reach. Junk, zero, negative or non-finite values fall back to the default rather than restoring the unbounded wait | `120` |
| `STRANDS_TRAIN_LOCAL_ADDR` | Address a multi-GPU training launch publishes as `MASTER_ADDR`. Unset, a single-node launch is pinned to `127.0.0.1` and a multi-node one keeps torch's `getfqdn()` resolution. Set it when a multi-node host's name resolves to a reverse-DNS pointer (`*.in-addr.arpa` / `*.ip6.arpa`), which no peer can dial - the launch warns naming this variable rather than hanging on "Rendezvous'ing worker group" | unset |
| `STRANDS_ROBOTS_NO_DYLD_SHIM` | Set `1` to disable the macOS auto-fix that puts Homebrew ffmpeg on the dyld path for torchcodec video streaming (see [Recording](../recording.md)) | unset |
| `MUJOCO_GL` | MuJoCo GL backend (`egl`, `osmesa`, `glfw`) | auto |
| `STRANDS_ISAAC_HEADLESS` | Isaac Sim backend: run without a GUI. On (`1`/`true`/`yes`/`on`) = headless, off (`0`/`false`/`no`/`off`) = windowed, any other spelling is refused. Overrides `IsaacConfig(headless=...)` ([#2062](https://github.com/strands-labs/robots/issues/2062)) | unset (config default `true`) |
| `STRANDS_ISAAC_RTX_PATHTRACING` | Isaac Sim backend: on (`1`/`true`/`yes`/`on`) enables RTX path-tracing (photorealistic, slow) instead of the default render mode; off leaves the render mode alone, any other spelling is refused | unset |
| `STRANDS_ISAAC_NUCLEUS_URL` | Isaac Sim backend: override the Omniverse Nucleus asset-server URL | unset (Isaac default) |
| `GROOT_API_TOKEN` | API token for the GR00T inference service | unset |
| `STRANDS_MESH` | Opt a bare `Robot()` into the Zenoh mesh: `true`/`1`/`yes` turns it on. `false`/`0`/`no` is a hard kill switch that also overrides an explicit `mesh=True`, refuses the robot-less gateway peer the `robot_mesh` tool would otherwise start in a coordinator process, and refuses the shared transport itself - so a direct `get_session()` / `ZenohTransport` caller opens no session and binds no `STRANDS_MESH_PORT` listener either. Any other value -- including `off` and `on` -- is ignored, leaving the mesh neither opted into nor disabled, and is logged once as a warning naming the spellings above | unset (mesh off) |
| `STRANDS_MESH_LOCAL_DEV` | Set `1` for a one-var localhost preset (auth `none`, no second factor needed) | unset |
| `STRANDS_ROS2_BRIDGE_I_KNOW_THIS_IS_INSECURE` | Second factor to expose a `Robot(ros2_transport="rtps")` inbound `joint_command` surface with no `dds_security_config` (DDS Security). Truthy: `1`/`true`/`yes` | unset |
| `STRANDS_ROS2_COMMAND_ALLOW` | Comma-separated ROS 2 surfaces pre-approved for `use_ros` commands, for headless use where no operator can be prompted (e.g. `/cmd_vel,/navigate_to_pose`). An entry matches by base name, so `/cmd_vel` also pre-approves every namespaced `cmd_vel` in the graph - name the namespace (`/turtle1/cmd_vel`) to scope the approval to one robot. Every blocklisted surface with a base name no entry lists stays gated, including a zero-velocity halt: the gate is keyed on the surface, not on the payload, so a deployment that must halt unattended pre-approves its `cmd_vel` topic. Reads are never gated. See [safety-critical command surfaces](../ros2-integration.md#safety-critical-command-surfaces-need-operator-approval) | unset |
<details>
<summary><b>Mesh / IoT / GR00T-container env vars (advanced)</b></summary>

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_MESH_BACKEND` | Mesh transport: `zenoh` (LAN), `iot` (AWS IoT Core), or `bridge` (both - Zenoh locally, IoT for the bridged topics). Case and surrounding whitespace are ignored. This selects the transport; the `[mesh-iot]` extra only installs the dependency `iot`/`bridge` need, so installing it without setting this leaves the fleet on Zenoh. An unrecognized value falls back to the default and is logged once, naming the valid values | `zenoh` |
| `STRANDS_MESH_AUTH_MODE` | Wire auth: `mtls` or `none` (`none` needs a second factor) | `mtls` |
| `STRANDS_MESH_TLS_CA` | Path to the CA bundle that validates peer certificates. Required under `AUTH_MODE=mtls` (the default) | unset |
| `STRANDS_MESH_TLS_CERT` | Path to this peer's certificate (PEM). Its CN is what an operator ACL pins. Required under `AUTH_MODE=mtls` | unset |
| `STRANDS_MESH_TLS_KEY` | Path to this peer's private key (PEM). Enforced mode `0600` on POSIX; on Windows the mode gate is skipped with a one-shot WARNING, so restrict it by NTFS ACL. Required under `AUTH_MODE=mtls` | unset |
| `STRANDS_MESH_I_KNOW_THIS_IS_INSECURE` | Second factor required to bring up `AUTH_MODE=none` | unset |
| `STRANDS_MESH_PORT` | TCP port for the local Zenoh router | `7447` |
| `STRANDS_MESH_FALLBACK_MODE` | Zenoh mode for a process that did NOT win the `STRANDS_MESH_PORT` listener and so connects to that hub: `client` (the hub relays, so siblings hear each other) or `peer` (direct links only, which the operator must then arrange). A Zenoh 1.x peer refuses relayed traffic, so `peer` children hear nothing a sibling child publishes. An unrecognized value warns and uses the default | `client` |
| `ZENOH_CONNECT` | Comma-separated remote Zenoh endpoints to connect to | unset |
| `ZENOH_LISTEN` | Comma-separated endpoints for the local Zenoh listener | unset |
| `STRANDS_MESH_MULTICAST` | Opt in to multicast scouting for LAN discovery. Off by default: any device on the LAN can enumerate and attract the fleet, so enabling it logs a WARNING. Prefer explicit `ZENOH_CONNECT` endpoints | `false` |
| `STRANDS_MESH_FILTER_INTERFACES` | Comma-separated network interface names (e.g. `eth0,wlan0`) the per-message size caps (`low_pass_filter`) are bound to. Unset or blank binds the caps to every link (Zenoh's wildcard), which is the safe default: an enumerated list that misses a NIC exempts that NIC from the cap | unset (all interfaces) |
| `STRANDS_MESH_MAX_CMD_BYTES` | Per-message byte cap on `cmd` / `broadcast` topics, applied at the transport (an over-cap message is dropped before the JSON parser runs) and pre-checked by `Mesh.send` so the sender fails loudly instead. Integer in `[128, 16777216]`; a value outside it, or not an integer, raises at session build naming the variable | `16384` (16 KiB) |
| `STRANDS_MESH_MAX_CAMERA_BYTES` | Per-message byte cap on `camera` topics, ingress only. Integer in `[1024, 134217728]`; out of range or non-integer raises | `1048576` (1 MiB) |
| `STRANDS_MESH_MAX_SAFETY_BYTES` | Per-message byte cap on `safety/*` topics, both flows. Integer in `[128, 1048576]`; out of range or non-integer raises | `4096` (4 KiB) |
| `STRANDS_MESH_CMD_RATE_HZ` | Transport-level rate cap on inbound `cmd` publishes per peer; faster publishes are dropped before parsing, so a flood costs the receiver almost nothing. Finite float in `[0.001, 10000]`; `nan`/`inf` or out of range raises rather than disabling the cap | `20.0` |
| `STRANDS_MESH_SAFETY_RATE_HZ` | The same cap for `safety/*` publishes, kept lower so a peer holding any fleet cert cannot flood `safety/estop` with novel timestamps past the replay cache. Finite float in `[0.001, 1000]` | `2.0` |
| `STRANDS_MESH_MAX_SESSIONS` | Zenoh `transport/unicast/max_sessions`: how many peers one session accepts. Integer in `[1, 65535]` | `256` |
| `STRANDS_MESH_NAMESPACE` | Fleet namespace prefix on every mesh key-expression. The Zenoh `namespace` config field provides routing isolation -- two fleets with different namespaces cannot exchange messages even when their key-expressions collide, so this is the knob that keeps a co-located test fleet from receiving a production fleet's commands. Must match on every peer of one fleet; a mismatch is silent (peers connect at the transport layer and exchange no application traffic). Empty / whitespace values fall back to the default so `STRANDS_MESH_NAMESPACE=""` cannot accidentally produce keys like `"//presence"` | `strands` |
| `STRANDS_MESH_AUDIT_DIR` | Directory for the safety audit log (`mesh_audit.jsonl`) | `~/.strands_robots/` |
| `STRANDS_MESH_AUDIT_PSK` | Pre-shared key that keys the per-record HMAC in the audit log. When set, `verify_audit_integrity` refuses a record whose HMAC does not match and refuses the whole log if the PSK changes mid-run; when unset, the `sig` field is absent and a writer with directory access can edit records without failing the check. Set on every peer that writes to the same directory | unset |
| `STRANDS_MESH_AUDIT_MAX_BYTES` | Rotate the active audit file once it crosses this size (bytes). Values above the 10 GiB hard cap are clamped with a warning; non-integer, zero, or negative values fall back to the default with a warning, so a misconfiguration cannot silently disable rotation | `104857600` (100 MiB) |
| `STRANDS_MESH_AUDIT_MAX_FILES` | Number of rotated `mesh_audit.jsonl.N` files kept alongside the active file. Older rotations are deleted as new ones arrive; total disk use is bounded by `_MAX_BYTES × _MAX_FILES`. Same clamping/warning behaviour as `_MAX_BYTES`; hard upper cap 100 | `5` |
| `STRANDS_MESH_CA_PINS` | Additional SHA-256 CA pins (comma-separated 64-char hex) | unset |
| `STRANDS_MESH_DISABLE_CA_PIN` | Skip CA pin check on download path (break-glass) | `false` |
| `STRANDS_MESH_CAMERA_S3_BUCKET` | S3 bucket that turns the camera offload on under the `iot` / `bridge` backends: frames are uploaded there and the mesh publishes a presigned URL instead of inline JPEG bytes. Unset leaves camera publishing inline (Zenoh) and logs `offload off` at DEBUG | unset (offload off) |
| `STRANDS_MESH_CAMERA_S3_PREFIX` | Key prefix inside `STRANDS_MESH_CAMERA_S3_BUCKET`; surrounding `/` are stripped | unset (bucket root) |
| `STRANDS_MESH_CAMERA_PRESIGN_TTL` | TTL (s) for S3 presigned camera URLs; capped at 3600 | `60` |
| `STRANDS_MESH_ACL_FILE` | Path to a JSON5 Zenoh ACL file; unset = permissive default. See `examples/mesh/mesh_acl_example.json5` (role-scoped) and `examples/mesh/mesh_acl_strict_per_peer.json5` (per-peer). **⚠️ Required on any WAN/cloud router: mTLS gives identity, not least-privilege — without a topic-level ACL one device cert can read all fleet traffic and command any robot. See [security docs](../security.md#production-posture-required-off-trusted-networks).** | unset |
| `STRANDS_MESH_ACCEPT_PERMISSIVE_ACL` | Acknowledgement token with **three** distinct effects, all of them widening the mesh posture — set on a production fleet at your peril. (1) A `STRANDS_MESH_ACL_FILE` with `default_permission: "allow"` **and** one or more `rules` is refused at load by `_acl_config._load_acl_file` with `PermissiveACLError` unless this variable is set to `1`/`true`/`yes` — the allow + rules shape is a blacklist policy where any gap in the rule set exposes the mesh, and the refusal exists so it is not shipped by copy-paste from a lab template. (2) When `STRANDS_MESH_AUTH_MODE=mtls` **and** the resolved ACL is permissive-by-shape (built-in default *or* operator file with `allow` + no rules), `Mesh.start`'s refuse-to-start gate downgrades from `ERROR` refusal to an `INFO` acknowledgement — the token is the opt-in that lets the wire come up under the built-in permissive default. (3) The per-session-open `WARNING` that fires when the built-in permissive default is in use is suppressed — session and start emit one line each about the same posture, so silencing the WARNING here avoids contradicting the operator's explicit opt-in on every session open. Set to `1`/`true`/`yes` (case-insensitive, whitespace-stripped) to accept all three; `on` and every other spelling are not acknowledgements, and `strands-robots doctor` reads the same predicate (`_acl_config.permissive_acl_acknowledged`) so its Mesh row agrees with the gates. The acknowledgement does not narrow the ACL; it records that an operator has accepted a wire-open posture. | unset (refuses blacklist ACL, refuses to start under built-in permissive default, warns on every session open) |
| `STRANDS_MESH_POLICY_HOST_ALLOW` | Comma-separated allowlist of VLA policy-server hosts/CIDRs for inference | loopback only |
| `STRANDS_MESH_POLICY_TYPE_ALLOW` | Comma-separated extras appended to the built-in `policy_type` / `policy_provider` allowlist (`validate_command` gates the mesh `execute` / `start` payloads against the union). Widens both vocabularies at once because the two share one allowlist by design. Each entry is charset-validated against a lowercase-identifier regex (`^[a-z][a-z0-9_]*$`) at parse time; case-variant typos are normalised via `.lower()` before the compare, so `"FOO,BAR"` matches `"foo,bar"`. A malformed entry drops with a WARNING naming the variable and the offender rather than widening the allowlist silently. Does not relax any adjacent gate: `policy_host`, `server_address`, `pretrained_name_or_path` and `model_path` are still allowlisted regardless of which provider the payload names. Adding a provider to `registry/policies.json` must include the corresponding edit to `_REGISTRY_POLICY_PROVIDERS` beside it; a guard test refuses any registry spelling that set omits, so the omission fails CI rather than shipping as a mesh-only availability bug the operator has to break-glass around with this variable | unset (built-in list only) |
| `STRANDS_MESH_HITL_ACTIONS` | `robot_mesh` actions needing a human-in-the-loop interrupt: `all` / `none` / subset of `emergency_stop,broadcast,tell,send,stop,rpc,subscribe,watch` | actuation default |
| `STRANDS_DASH_AGENT_PHYSICAL_MOTION` | Grant that lets an agent start motion on a **physical** peer by itself (`strands_robots.dashboard.agent_motion`), read at call time so it can be withdrawn mid-session. Only motion-starting actions are gated -- every way of stopping a robot is outside the gate. `1`/`true`/`yes`/`on` grants | unset (agent-initiated physical motion refused, an operator has to start it) |
| `STRANDS_DASH_TASK_REQUIRES_CONFIRM` | Set `1`/`true`/`yes`/`on` to make a real-motion task request carry an explicit operator confirmation. The confirmation is read as a JSON boolean, so `true` carries it and `"true"` does not -- every non-empty string is truthy, and `"false"` would otherwise confirm | unset (no confirmation flag required) |
| `STRANDS_MESH_SUBSCRIBE_ALLOW` | Extra Zenoh key-expr patterns the `robot_mesh` `subscribe` action may target, beyond the built-in low-impact set | shared classes only |
| `STRANDS_MESH_OVERRIDE_CODE` | Shared secret for e-stop resume HMAC proof; unset means no remote resume possible | unset |
| `STRANDS_MESH_INPUT_VALUE_ABS` | Absolute value clamp for teleop joint commands, in frame units -- whichever unit the leader driver puts on the wire (shipped SO leaders stream degrees, and a 0-100 gripper). Narrow it for radian or normalized -1..1 actuators, whose units are smaller | `720` (two full turns) |
| `STRANDS_MESH_INPUT_MAX_HZ` | Per-receiver teleop apply-rate ceiling (0 = unlimited). A value no rate check can be built from -- unparsable, or non-finite like `inf`/`nan` -- falls back to the default so the ceiling stays enforced | `100` |
| `STRANDS_MESH_INPUT_SLEW_ABS` | Per-joint speed bound for the mesh receive path, in frame units per second (narrow for radian or normalized actuators, whose units are smaller; cannot be disabled) | `1440` (the value envelope traversed once per second) |
| `STRANDS_TELEOP_SLEW_ABS` | Per-joint speed bound for the local `teleoperate()` loop, in frame units per second (default accommodates degree-valued and range-0-100 devices; cannot be disabled) | `500.0` |
| `STRANDS_MESH_POSE_HZ`, `_IMU_HZ`, `_ODOM_HZ`, `_HEALTH_HZ`, `_LIDAR_SUMMARY_HZ`, `_HAND_HZ`, `_MAP_INFO_HZ` | Per-topic sensor publish rate; `0` (or any non-positive value) switches that topic off. A value the loop cannot pace itself with keeps the built-in rate | per topic: `10`/`10`/`10`/`0.5`/`5`/`50`/`0.2` |
| `STRANDS_MESH_CAMERA_HZ` | Camera publish rate; opt-in because frames are large. Unset, non-positive, or unusable leaves camera publishing off | `0` (off) |
| `STRANDS_MESH_CAMERA_DISABLED` | Privacy kill switch for the camera publisher: `true`/`1`/`yes`/`on` publishes no frames at all (nothing built, signed or sent) whatever `STRANDS_MESH_CAMERA_HZ` says; `false`/`0`/`no`/`off` leaves it to the rate. Any other spelling raises rather than silently re-enabling a privacy flag | unset (cameras follow the rate) |
| `STRANDS_MESH_STREAM_HZ` | Per-step task telemetry rate while a robot or a rollout is executing. Non-positive or unusable -- unparsable, or non-finite like `inf`/`nan` -- switches step publishing off rather than changing the rate, so an unreadable value cannot remove the throttle | `10` |
| `STRANDS_MESH_GATEWAY_DISCOVERY_WAIT_S` | How long a robot-less `robot_mesh` gateway waits once at bring-up for presence to populate before the first `peers` read. `0` means do not wait; a value no sleep can honor -- unparsable, negative, or non-finite -- falls back to the default | `3` |
| `STRANDS_MESH_MAX_PEERS` | Peer registry cap; evicts oldest on overflow | `1024` |
| `STRANDS_MESH_PEER_RETENTION_S` | Extra retention (s) past the peer timeout before a silent peer is deleted; retained peers report `reachable: false`. A value no retention can be built from - unparsable, negative, or non-finite - falls back to off with one warning per spelling | `0` (off) |
| `STRANDS_MESH_RESUME_MAX_FAILS` | Failed resume attempts before cooldown engages | `5` |
| `STRANDS_MESH_RESUME_BACKOFF_S` | Cooldown (seconds) after exceeding resume fail threshold. A value no cooldown instant can be built from -- unparsable, negative, or non-finite like `inf`/`nan` -- falls back to the default, so the throttle both engages and expires (shared with `STRANDS_MESH_RESUME_FRESHNESS_S` / `_FORWARD_SKEW_S`) | `30` |
| `STRANDS_MESH_RESUME_FRESHNESS_S` | How far in the past a resume envelope's timestamp may be before a receiver refuses it as stale. A receiver whose clock is more than this *ahead of* the operator reads every resume as stale and stays locked out, so keep fleet clocks in NTP sync or widen this on every peer (a receiver *behind* the operator is refused by `STRANDS_MESH_RESUME_FORWARD_SKEW_S` instead) | `60` |
| `STRANDS_MESH_RESUME_FORWARD_SKEW_S` | How far in the future a resume envelope's timestamp may be before a receiver refuses it as future-dated. This is the tighter of the two bounds: a receiver whose clock is more than this *behind* the operator sees every resume as future-dated and stays locked out, so widen this on every peer (a receiver *ahead of* the operator is refused by `STRANDS_MESH_RESUME_FRESHNESS_S` instead) | `5` |
| `STRANDS_MESH_RESUME_REPLAY_CACHE_MAX` | Entries in the per-receiver resume replay cache; also bounds the per-issuer fairness cap (max/4) so one flooding issuer cannot evict a legitimate operator's slot | `4096` |
| `STRANDS_MESH_INPUT_AUDIT_EVERY` | Emit `input_stream_applied` audit event every N frames (0 = off) | `100` |
| `STRANDS_ESTOP_DEDUP_TTL_S` | E-stop fan-out Lambda dedup window (seconds) | `30` |
| `STRANDS_MESH_DEDUP_TTL` | Window (seconds) the Zenoh<->IoT bridge remembers a delivered `(sender_id, turn_id, command)` triple for cross-transport deduplication. Unparsable, non-positive or non-finite falls back to the default, so a legitimately recurring heartbeat is forgotten again | `120` |
| `STRANDS_MESH_BRIDGE_DEDUP_STRICT` | `1`/`true`/`yes` makes the Zenoh<->IoT bridge dedup a sample that carries no `(sender_id, turn_id, command)` triple by hashing its whole payload, so a heartbeat-style message arriving on both transports is delivered once. Off delivers such samples as-is. Read once at bridge construction; any other spelling warns and stays off | `0` (off) |
| `STRANDS_MESH_BRIDGE_TOPICS` | Comma-separated topic suffixes the Zenoh<->IoT bridge forwards (exact match). Unset = the safe default set (`presence,health,safety/event,safety/estop,safety/resume,cmd,response,broadcast`). High-volume topics (`state,pose,imu,odom,lidar`) and LAN-only topics (`camera,input,hand`) are deliberately NOT bridged | default set |
| `STRANDS_MESH_BRIDGE_TOPICS_PREFIX` | Comma-separated topic suffixes the bridge matches as a path **prefix** (so `response` matches `response/<turn-id>`). Extend this (not `STRANDS_MESH_BRIDGE_TOPICS`) when adding an RPC-shape topic with a per-turn tail | `response` |
| `STRANDS_GR00T_IMAGE` | Container image the `gr00t_inference` tool runs (must pass the image allowlist; agent cannot choose it) | `gr00t:latest` |
| `STRANDS_GR00T_IMAGE_ALLOW` | Extra image-name patterns (trailing `*` = tag wildcard) added to the built-in allowlist (`gr00t:*`, `nvcr.io/nvidia/isaac-gr00t:*`) | built-in only |
| `STRANDS_GR00T_REPO_URL` | Git URL `gr00t_inference(action="build_image")` clones Isaac-GR00T from. Operator-only (not a tool parameter); must exact-match the repo-URL allowlist or the build fails closed | `https://github.com/NVIDIA/Isaac-GR00T` |
| `STRANDS_GR00T_REPO_TAG` | Git ref (tag or branch) checked out from `STRANDS_GR00T_REPO_URL`. Letters, digits and `._/-` only, no leading `-`, so the value can never be read as a `git` option | `n1.7-release` |
| `STRANDS_GR00T_REPO_URL_ALLOW` | Comma-separated extra clone URLs added to the built-in allowlist (the canonical repo, with and without `.git`). Each entry is exact-matched, never a wildcard, so a look-alike repo cannot slip past. See [security docs](../security.md) | built-in only |
| `STRANDS_GR00T_SERVER_SEED` | Default seed the GR00T determinism wrapper applies at server start and on seedless `reset` calls (used with `gr00t_inference(..., deterministic=True)`; forwarded into the container) | `42` |
| `STRANDS_GR00T_STRICT_DETERMINISTIC` | `1` makes the determinism wrapper additionally enable `torch.use_deterministic_algorithms(True, warn_only=True)` (slower kernels, strictest reproducibility; forwarded into the container). Best-effort: an op with no deterministic kernel makes torch refuse, degrading the server to non-strict rather than killing it, and the startup banner reports `strict=` as the mode the server ended up in | `0` |

</details>

<details>
<summary><b>Isaac Sim backend env vars (<code>strands-robots[sim-isaac]</code>)</b></summary>

These are read by the built-in, in-tree Isaac Sim backend
(`pip install 'strands-robots[sim-isaac]'`) when it builds its
`IsaacConfig`. An explicit `create_simulation("isaac", ...)` kwarg wins for
`nucleus_url`; the two switches override their field whenever they are set
([#2062](https://github.com/strands-labs/robots/issues/2062)). Both switches
accept `1`/`true`/`yes`/`on` and `0`/`false`/`no`/`off` (case-insensitive,
surrounding whitespace ignored); unset or empty leaves the field alone and any
other spelling is refused. See
[`docs/simulation/isaac.md`](../simulation/isaac.md).

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_ISAAC_NUCLEUS_URL` | Override the Omniverse Nucleus server URL (when `nucleus_url` is not passed) | unset (Isaac defaults) |
| `STRANDS_ISAAC_HEADLESS` | On forces headless; off forces a window | unset (uses `headless` kwarg) |
| `STRANDS_ISAAC_RTX_PATHTRACING` | On forces `render_mode="rtx_pathtracing"`; off leaves `render_mode` alone | unset |
| `STRANDS_ISAAC_CAMERA_WARMUP_STEPS` | Render-bearing world steps `add_camera` takes before returning, so a new RTX camera's first `get_rgba()` is a real frame rather than the empty buffer the pipeline returns until it has been stepped. Raise on a slow GPU. Positive integer; anything else is reported with a warning naming the variable and falls back to the default | `10` |

</details>

<details>
<summary><b>Diagnostic env vars (GR00T bisection)</b></summary>

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_GROOT_WIRE_LOG` / `_MAX_CALLS` | Directory to dump pre/post inference payloads to, e.g. `/tmp/groot-wire`, to verify LOCAL vs SERVICE parity | unset / `10` |
| `STRANDS_ROBOTS_VERBOSE_MUJOCO` | `1`/`true`/`yes` lets MuJoCo's attach-conflict chatter (`timestep: parent has 0.002, child has 0.005, keeping parent value` and its siblings, emitted on every `add_robot` / world build) reach stderr instead of being captured. Captured lines are re-emitted at DEBUG with a count; every other stderr line is forwarded unchanged either way | unset (captured) |

</details>

<details>
<summary><b>Dashboard passkey (WebAuthn) auth env vars</b></summary>

Read by `strands_robots.dashboard.auth`, which fronts the dashboard routes that
command real hardware. The credential store is the source of truth for whether
auth is on: the moment a passkey is enrolled, it is. The three durations and the
two challenge caps are **refused, not defaulted**, when they hold a value the
module cannot use - each is a knob an operator uses to *narrow* a window, and a
substituted default lands in the wider direction - so a misspelled value stops
the server at import rather than surfacing as a failed login later.

| Variable | Description | Default |
|----------|-------------|---------|
| `STRANDS_DASH_AUTH_ENABLED` | Overrides the store's verdict only when spelled as a recognised boolean: on (`1`/`true`/`yes`/`on`) guards the API with no passkey enrolled yet, off (`0`/`false`/`no`/`off`) disables the guard. Any other spelling is logged and ignored, so an enrolled passkey still guards the API rather than being dropped by a typo | unset (the store decides) |
| `STRANDS_DASH_AUTH_STORE` | Path of the credential store: the enrolled passkeys and the secret session tokens are signed with | `~/.strands_dashboard/auth.json` |
| `STRANDS_DASH_AUTH_RP_NAME` | Human-readable relying-party name shown by the authenticator during enrollment | `strands robots dashboard` |
| `STRANDS_DASH_AUTH_RP_ID` | Pins the WebAuthn relying-party id when the dashboard's hostname legitimately changed and existing passkeys must keep working. Unset, it is derived from the `Host` the request arrived on; a host that cannot be an rpId is refused naming this variable | unset (derived) |
| `STRANDS_DASH_AUTH_ORIGIN` | The origin the dashboard is served at, e.g. `https://robots.example:8443` - scheme and any non-default port included, since WebAuthn compares origins byte-for-byte. Unset, it is read off the connection (the transport's own scheme plus `Host`). Set it when a proxy rewrites `Host` or `Origin`, or when TLS is terminated upstream and uvicorn is not configured to forward that fact (`--proxy-headers` with `--forwarded-allow-ips`) | unset (read off the connection) |
| `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` | The secret the **first** passkey enrollment must present. Unset, the module mints one itself into a `0600` file beside the store and the operator reads it off the machine - a loopback peer is never proof of presence, so the first enrollment is never admitted on the strength of where the connection appears to come from | unset (minted into a file) |
| `STRANDS_DASH_AUTH_ENROLL_TOKEN_FILE` | Relocates the minted first-enrollment token file. Only read when `STRANDS_DASH_AUTH_BOOTSTRAP_TOKEN` is unset | `<store directory>/enroll_token` |
| `STRANDS_DASH_AUTH_TOKEN_TTL` | Lifetime of a freshly minted session token, in whole seconds (`1h` and `30m` are not units and are refused). Must be `>= 1` | `86400` (1 day) |
| `STRANDS_DASH_AUTH_SESSION_MAX_AGE` | Absolute age, in whole seconds, past which no renewal extends a session. Must be `>= 1` | `2592000` (30 days) |
| `STRANDS_DASH_AUTH_HANDOFF_TTL` | Lifetime of a LAN handoff token, in whole seconds. It rides in a URL - which lands in history, logs and screenshots - so it is short, and it never outlives the session it was copied from. Must be `>= 1` | `300` (5 minutes) |
| `STRANDS_DASH_AUTH_CHAL_MAX` | Global bound on the table of in-flight WebAuthn challenges; the oldest record is evicted past it, regardless of ip. Integer `>= 2` | `512` |
| `STRANDS_DASH_AUTH_CHAL_MAX_PER_IP` | Per-ip bound on in-flight challenges, which is what keeps one flooding client off the global cap. Integer `>= 1`, and it must stay **strictly below** `STRANDS_DASH_AUTH_CHAL_MAX` - a pair that does not is refused at import | `16` |

</details>

### Asset cache

```
~/.strands_robots/
└── assets/           # auto-downloaded MJCF + meshes
    ├── trs_so_arm100/
    ├── franka_emika_panda/
    └── ...
```

Clear with `rm -rf ~/.strands_robots/assets/`; relocate with
`export STRANDS_ASSETS_DIR=/path/to/dir`.

### CA Pin Rotation Runbook

The AWS IoT transport pins the SHA-256 of the canonical Amazon Root CA1 PEM, so
a network-level attacker (DNS hijack, captive portal, BGP, malicious local
proxy) cannot substitute a rogue CA at the download URL. The accepted set is a
*collection*, not a scalar, so old and new pins can both be valid at once - that
is what makes a rotation expressible without a flag-day deploy.

When AWS rotates the root, every fleet member refuses the new certificate until
a pin covering it is accepted, on both the download path and the on-disk re-use
path. Deleting the cached PEM does not help: the re-download fetches the same
unpinned bytes and is refused again. Rotation therefore needs an ordered
procedure, which is this one.

**Recompute** the pin of whatever the URL currently serves:

```bash
python -c "import hashlib, urllib.request as u; \
print(hashlib.sha256(u.urlopen( \
'https://www.amazontrust.com/repository/AmazonRootCA1.pem' \
).read()).hexdigest())"
```

**Monitor** for rotations before they bite: AWS announces root-CA changes in its
security bulletins with a deprecation timeline, so a planned rotation can be
shipped ahead of the cutover rather than during an outage.

**Rotate (planned):**

1. **Verify the new certificate out of band.** A digest computed from the same
   connection that served the bytes proves nothing. Confirm the certificate
   against an independent source before it becomes a pin.
2. **Ship a release that adds the new pin and keeps the old one.** Both stay
   valid, so peers still on the previous release keep verifying.
3. **Wait for fleet uptake.** The overlap is bounded by the slowest fleet member,
   not by the release cadence.
4. **Drop the old pin in a follow-up release** once uptake is complete.

**Emergency (a rotation lands faster than a release can ship):** stage the
verified new pin in `STRANDS_MESH_CA_PINS` (comma-separated, 64-char lowercase
hex). It is *additive* - the built-in pin stays accepted and verification stays
on - so it buys the grace period a release would have provided. Entries that are
not valid hex digests are rejected with a warning and skipped rather than
weakening the set. Remove the override once the release carrying the pin is
deployed.

`STRANDS_MESH_DISABLE_CA_PIN` is **not** part of this procedure. It turns the
download-path pin check off rather than widening it, accepts whatever the URL
serves, and marks the result as unverified-origin so later runs warn about
re-using it. It is a break-glass for a broken pin, never the response to a
rotation - a rotation has a verified pin to stage.

