export interface Presence {
  robot_id: string
  robot_type: 'robot' | 'sim' | string
  hostname?: string
  tool_name?: string
  task_status?: string
  instruction?: string
  connected?: boolean
  hw?: string
  cameras?: string[]
  sim_robots?: string[]
  action_keys?: string[]
  topics?: string[]
  timestamp?: number
}

export interface JointState { position?: number; velocity?: number }

export interface PeerState {
  peer_id: string
  t?: number
  joints?: Record<string, JointState | number[] | number>
  task?: { status?: string; instruction?: string; steps?: number; duration?: number }
  sim_time?: number
  robots?: Record<string, { active: boolean }>
  /** Q81: stranded serial in-use flags this robot has cleared since it started (absent when none). */
  bus_recoveries?: number
}

export interface StreamStep {
  peer_id: string
  step: number
  t: number
  instruction: string
  policy: string
  observation: Record<string, unknown>
  action: Record<string, number | number[]>
}

/** `strands/{peer}/pose` — SE(3) pose from a provider, SLAM or odometry. */
export interface PosePayload {
  peer_id: string
  t?: number
  x?: number; y?: number; z?: number
  theta?: number
  quat?: number[]
  /** which estimator answered: 'provider' | 'slam' | 'odom' */
  source?: string
  frame?: string
  [key: string]: unknown
}

/** `strands/{peer}/health` — every reading the robot could take, any of them absent. */
export interface HealthPayload {
  peer_id: string
  t?: number
  battery_pct?: number | null
  charging?: boolean
  temps?: Record<string, number>
  cpu_load?: number
  disk_free_gb?: number
  mem_pct?: number
  uptime_s?: number
  [key: string]: unknown
}

/** `strands/{peer}/imu` — downsampled from the hardware rate. */
export interface ImuPayload {
  peer_id: string
  t?: number
  accel?: number[]
  gyro?: number[]
  [key: string]: unknown
}

/** `strands/{peer}/odom` — body-frame velocities. */
export interface OdomPayload {
  peer_id: string
  t?: number
  vx?: number; vy?: number; wz?: number
  [key: string]: unknown
}

/** `strands/{peer}/lidar/{summary,state}` — two documents, kept apart. */
export interface LidarPayload {
  peer_id: string
  t?: number
  [key: string]: unknown
}

export interface Peer {
  peer_id: string
  last_seen?: number
  stale?: boolean
  /** Measured off this arm's servo bus (12V follower / 7.4V leader) and
   *  remembered by USB serial. Absent = nobody measured it, which is NOT the
   *  same as unknown. Only locally managed arms can carry it. */
  role?: string | null
  role_volts?: number | null
  role_source?: string | null
  /** Who STARTED this peer's process: 'managed' = this dashboard spawned it,
   *  'external' = it arrived on its own (the user's own Robot(..., mesh=True)
   *  script, or a peer on another box). Absent on a server older than the
   *  origin field - render nothing rather than guessing, since claiming
   *  'external' for every peer would be a lie on a managed fleet.
   *  U15: this is the ONLY thing that may differ. It says nothing about the
   *  robot's health and gates no control. */
  origin?: 'managed' | 'external' | string | null
  /** Camera names this dashboard REQUESTED when it spawned the peer (annotation,
   *  managed peers only). Presence lists only the cameras the robot managed to
   *  OPEN, so this is the only way to tell "joints-only by design" from "they
   *  failed to open and were dropped at connect". Absent = not known; an empty
   *  array is never sent, because it would read as a claim about zero cameras. */
  cameras_requested?: string[]
  /** WHY this arm publishes no joints, decided server-side from the child's own log (Q80).
   *  Absent means "nothing known" — the fleet must never invent a fault. The bridge removes it
   *  as soon as the arm publishes positions again, because mesh.core never logs a recovery. */
  joint_problem?: { kind?: string; headline?: string; remedy?: string; detail?: string
    /** 'peer' = the robot reported it in its own state snapshot (clears on recovery);
     *  absent = derived from its log; a recovery line clears it, but a robot
     *  running code older than 13b72dcf logs none, so it can outlive the fault. */
    source?: string; failures?: number; for_seconds?: number } | null
  /** E-stop lockout as the SERVER understands it (Q43): 'locked' (an e-stop it
   *  saw), 'clear' (proved by this peer accepting a command a lockout would
   *  refuse), 'unknown' (say so - the mesh deliberately does not advertise
   *  lockout state, so silence is not safety). `since` set on an 'unknown'
   *  means a safety event DID happen and the landing is genuinely unclear;
   *  without it, this dashboard has simply seen nothing. Absent on an older
   *  server - render nothing rather than implying safety. */
  lockout?: { state?: string | null; reason?: string | null; since?: number | null; by?: string | null } | null
  presence?: Presence
  state?: PeerState
  stream?: StreamStep
  cameras?: Record<string, { t?: number; shape?: number[] }>
  /** SensorLoops topics, each absent until the robot publishes it. An arm publishes none of
   *  them and is not broken, so absence is never rendered as a fault (lib/sensorFreshness). */
  pose?: PosePayload
  health?: HealthPayload
  imu?: ImuPayload
  odom?: OdomPayload
  /** Two documents share the lidar topic; they are kept apart so neither overwrites the
   *  other's fields. */
  lidar?: { summary?: LidarPayload; state?: LidarPayload }
}

export interface ActivityEntry {
  t: number
  source: 'api' | 'agent' | 'estop' | 'mesh' | string
  action: string
  target: string
  ok: boolean | null
  detail?: any
  elapsed?: number | null
  result?: string
}

export interface MeshInfo {
  online?: boolean
  peer_id?: string
  peers?: number
  live_peers?: number
  connect?: string[]
  listen?: string[]
  port?: string | number
  backend?: string
  auth_mode?: string
  local_dev?: boolean
  wire_security?: string
  camera_hz?: number
  multicast?: string
  max_cmd_bytes?: number
  policy_allow?: string[]
  settings?: Record<string, any>
}

/** A provider input the mesh command schema will actually carry. */
export interface WireField {
  key: string
  /** the command key it travels as (registry names differ from wire names) */
  wire_key: string
  type: 'string' | 'int' | 'float' | 'bool' | 'json' | string
  required: boolean
  default?: any
}

/** One entry of `registry/policies.json` - the run form's schema. */
export interface PolicyProvider {
  wire_fields: WireField[]
  /** provider kwargs the wire schema drops; only usable when built locally */
  unsettable_over_mesh: string[]
  name: string
  description: string
  requires: string[]
  config_keys: string[]
  defaults: Record<string, any>
  shorthands: string[]
  url_patterns: string[]
  extra?: string | null
  trainable: boolean
  /** false -> the mesh security gate rejects this over the wire. */
  wire_safe: boolean
  /** needs a running inference server rather than a local checkpoint. */
  server_based: boolean
}

export interface EnvRow {
  key: string
  value: string
  secret: boolean
  set: boolean
  in_file: boolean
  /** Q50: the launch environment carries a DIFFERENT value, which wins over .env for this run. */
  shadowed?: boolean
}

export interface ConfigDoc {
  agent: {
    model_id: string | null
    known_models: string[]
    system_prompt: string
    is_default_prompt: boolean
    temperature: number | null
    max_tokens: number | null
    built?: boolean
    busy?: boolean
    messages?: number
    tools?: string[]
    bridge_online?: boolean
    history_file?: string
  }
  voice: { provider: string; voice_name: string | null; providers: string[] }
  mesh: MeshInfo
  runtime: { trust_remote_code: boolean }
  security: {
    auth_enabled: boolean
    cors_origins: string[]
    /**
     * This PROCESS's own posture, absent when there is nothing to say — today only
     * `token_in_argv`: the dashboard was started with --auth-token on the command line, so
     * `ps` shows its bearer token to every local user. Optional at every layer: an older
     * server omits it and the screen shows nothing, which is the correct rendering of
     * "this server cannot tell me".
     */
    notice?: { kind: string; severity?: string; text: string; remedy?: string } | null
  }
  policies: PolicyProvider[]
  env: EnvRow[]
  env_file: string
  settings_file: string
}

export interface StopResult {
  peer_id: string
  state: 'stopped' | 'not_stopped' | 'no_answer'
  detail?: any
  result?: any
}

export interface EstopResult {
  targeted: string[]
  stale_skipped: string[]
  counts: { stopped: number; not_stopped: number; no_answer: number }
  all_stopped: boolean
  stopped: Record<string, StopResult>
  /** Signed strands/safety/estop rail (fleet-wide lockout).
   *  `responses_received` counts REPLIES, not confirmed stops, and
   *  `peers_not_stopped` names responders that reported they did not stop.
   *  Both optional: a server older than the fields sends nothing, which must
   *  read as "cannot tell you" rather than as zero. */
  signed_rail?: {
    signed: boolean
    issuer?: string
    error?: string
    responses_received?: number
    peers_not_stopped?: string[]
  }
  lockout_engaged?: boolean
}

import type { AbsentChild } from './lib/absentChildren'

export type MeshEvent =
  | { type: 'snapshot'; dashboard_peer_id: string; peers: Record<string, Peer>; mesh?: MeshInfo;
      /** dead managed children, already pruned from `peers` (U22). Optional: a server
       *  older than the field sends nothing, which must read as "cannot tell you". */
      absent_children?: AbsentChild[]; managed_no_presence?: string[]; t: number }
  | { type: 'presence'; peer_id: string; data: Presence }
  | { type: 'state'; peer_id: string; data: PeerState }
  | { type: 'stream'; peer_id: string; data: StreamStep }
  | { type: 'camera_meta'; peer_id: string; cam: string; data: { t?: number; shape?: number[] } }
  | { type: 'pose'; peer_id: string; data: PosePayload }
  | { type: 'health'; peer_id: string; data: HealthPayload }
  | { type: 'imu'; peer_id: string; data: ImuPayload }
  | { type: 'odom'; peer_id: string; data: OdomPayload }
  | { type: 'lidar'; peer_id: string; kind: 'summary' | 'state'; data: LidarPayload }
  | { type: 'safety'; kind: 'estop' | 'resume'; data: Record<string, unknown> }
  | { type: 'activity'; data: ActivityEntry }
  | { type: 'mesh_reconfigured'; ok: boolean; mesh: MeshInfo }
