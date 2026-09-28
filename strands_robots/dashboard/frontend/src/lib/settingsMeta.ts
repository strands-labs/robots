/**
 * Settings metadata: every tunable in the Settings drawer earns its place here with a
 * plain-language explanation, a unit, its safe default, HOW it applies (live vs next turn vs
 * mesh restart), and a validator.
 */

/** When does a change actually take effect? Shown as a chip next to the field. */
export type ApplyMode = 'live' | 'next-turn' | 'mesh-restart' | 'respawn' | 'startup'

export const APPLY_LABEL: Record<ApplyMode, string> = {
  live: 'applies immediately',
  'next-turn': 'applies on the next agent turn',
  'mesh-restart': 'needs a mesh restart',
  respawn: 'applies to robots spawned from now on',
  startup: 'applies at next server start',
}

export interface SettingMeta {
  key: string
  label: string
  /** One line: what happens if I change this. */
  effect: string
  unit?: string
  /** Safe default, as the user would type it. Empty string = provider default. */
  safeDefault: string
  apply: ApplyMode
  /** Returns an error message, or null when the raw input is acceptable. */
  validate: (raw: string) => string | null
}

export function finiteNumber(raw: string, opts: {
  min?: number; max?: number; integer?: boolean; name?: string
} = {}): string | null {
  const s = raw.trim()
  if (s === '') return null // empty means "use the default", always valid
  const n = Number(s)
  if (!Number.isFinite(n)) return `${opts.name ?? 'value'} must be a number`
  if (opts.integer && !Number.isInteger(n)) return `${opts.name ?? 'value'} must be a whole number`
  if (opts.min !== undefined && n < opts.min) return `minimum is ${opts.min}`
  if (opts.max !== undefined && n > opts.max) return `maximum is ${opts.max}`
  return null
}

/** The schemes the MESH actually accepts, read off mesh/session.py rather than guessed. */
const TLS_BEARING = ['tls', 'quic', 'wss', 'unixsock'] as const
const PLAINTEXT = ['tcp', 'udp'] as const
const ALL_SCHEMES: readonly string[] = [...TLS_BEARING, ...PLAINTEXT]

/** One endpoint. */
function endpointError(ep: string, dialable: boolean): string | null {
  // zenoh allows per-endpoint config after '#', e.g. tcp/0.0.0.0:7447#iface=en0
  const [addr] = ep.split('#')
  const slash = addr.indexOf('/')
  if (slash <= 0) return `"${ep}" is not proto/host:port (e.g. tls/robot.lan:7447)`
  const scheme = addr.slice(0, slash).toLowerCase()
  const rest = addr.slice(slash + 1)
  if (!ALL_SCHEMES.includes(scheme)) {
    return `"${scheme}" is not a mesh transport — use one of ${ALL_SCHEMES.join(', ')}`
  }
  if (scheme === 'unixsock') {
    return rest.trim() === '' ? `"${ep}" needs a socket path (unixsock/tmp/zenoh.sock)` : null
  }
  const colon = rest.lastIndexOf(':')
  if (colon <= 0) return `"${ep}" is missing a port (e.g. ${scheme}/robot.lan:7447)`
  const host = rest.slice(0, colon)
  const portRaw = rest.slice(colon + 1)
  if (/\s/.test(host) || host === '') return `"${ep}" has no host`
  if (!/^\d{1,5}$/.test(portRaw)) return `"${ep}" has a non-numeric port`
  const port = Number(portRaw)
  if (port > 65535) return `"${ep}" has an out-of-range port`
  if (port === 0 && dialable) return `"${ep}" cannot be dialled — port 0 means "any free port", which only a listen endpoint can use`
  if (!TLS_BEARING.includes(scheme as (typeof TLS_BEARING)[number])) {
    // A NOTICE the operator can act on now, not a ValueError at the next mesh restart. It is worded
    // as a condition because this browser cannot read STRANDS_MESH_AUTH_MODE.
    return `"${ep}" uses ${scheme}, which the default mTLS posture refuses when the mesh restarts — use ${TLS_BEARING.join('/')}, or set STRANDS_MESH_AUTH_MODE=none for the insecure development posture`
  }
  return null
}

function endpointsError(raw: string, dialable: boolean): string | null {
  const s = raw.trim()
  if (s === '') return null
  for (const part of s.split(',')) {
    const ep = part.trim()
    if (ep === '') continue
    const err = endpointError(ep, dialable)
    if (err) return err
  }
  return null
}

/** Comma-separated endpoints this dashboard DIALS OUT to. */
export function connectEndpoints(raw: string): string | null {
  return endpointsError(raw, true)
}

/** Comma-separated endpoints this dashboard LISTENS on (port 0 = any free port). */
export function listenEndpoints(raw: string): string | null {
  return endpointsError(raw, false)
}

export const SETTINGS: SettingMeta[] = [
  {
    key: 'agent.temperature',
    label: 'Temperature',
    effect: 'Higher = more varied agent replies; lower = more deterministic. Empty uses the model default.',
    safeDefault: '',
    apply: 'next-turn',
    validate: raw => finiteNumber(raw, { min: 0, max: 2, name: 'temperature' }),
  },
  {
    key: 'agent.max_tokens',
    label: 'Max tokens',
    effect: 'Caps the length of one agent reply. Too low truncates tool use mid-thought.',
    unit: 'tokens',
    safeDefault: '',
    apply: 'next-turn',
    validate: raw => finiteNumber(raw, { min: 1, max: 200000, integer: true, name: 'max tokens' }),
  },
  {
    key: 'agent.model_id',
    label: 'Model id',
    effect: 'Which model answers chat and fleet commands. Empty uses the provider default.',
    safeDefault: '',
    apply: 'next-turn',
    validate: () => null,
  },
  {
    key: 'mesh.port',
    label: 'Mesh port',
    effect: 'UDP/TCP port the zenoh mesh binds. Every robot on the desk must agree on it.',
    unit: 'port',
    safeDefault: '7447',
    apply: 'mesh-restart',
    validate: raw => finiteNumber(raw, { min: 1, max: 65535, integer: true, name: 'port' }),
  },
  {
    key: 'mesh.camera_hz',
    label: 'Camera rate',
    effect: 'How many frames per second each robot publishes. Higher is smoother but costs LAN bandwidth. '
      + 'Each robot reads it when IT starts, so a running robot keeps its rate until you respawn it.',
    unit: 'Hz',
    safeDefault: '5',
    apply: 'respawn',
    validate: raw => finiteNumber(raw, { min: 0.1, max: 60, name: 'camera rate' }),
  },
  {
    key: 'mesh.connect',
    label: 'Connect endpoints',
    effect: 'Other mesh routers this dashboard dials out to. Empty relies on multicast discovery.',
    safeDefault: '',
    apply: 'mesh-restart',
    validate: connectEndpoints,
  },
  {
    key: 'mesh.listen',
    label: 'Listen endpoints',
    effect: 'Addresses this dashboard accepts mesh connections on. Empty uses zenoh defaults; '
      + 'port 0 means any free port.',
    safeDefault: '',
    apply: 'mesh-restart',
    validate: listenEndpoints,
  },
  {
    key: 'voice.provider',
    label: 'Voice provider',
    effect: 'Which service speaks and listens. Each provider needs its credential in the Env tab.',
    safeDefault: 'openai',
    apply: 'live',
    validate: () => null,
  },
  {
    key: 'voice.voice_name',
    label: 'Voice name',
    effect: "Which of the provider's voices answers. Empty uses the provider default.",
    safeDefault: '',
    apply: 'live',
    validate: () => null,
  },
]

const byKey = new Map(SETTINGS.map(s => [s.key, s]))
export const settingMeta = (key: string): SettingMeta | undefined => byKey.get(key)

/** Validate one field by key; unknown keys are permissive (never block save on missing metadata). */
export function validateSetting(key: string, raw: string): string | null {
  return byKey.get(key)?.validate(raw) ?? null
}

/** True when every entry in {key: rawValue} passes - the save button's enabled state. */
export function allValid(drafts: Record<string, string>): boolean {
  return Object.entries(drafts).every(([k, v]) => validateSetting(k, v) === null)
}

/** Env var NAME: POSIX-shell shaped, no whitespace, never a newline. */
export function envKeyError(raw: string): string | null {
  const s = raw.trim()
  if (s === '') return null // nothing typed yet - the save path skips empty keys
  if (/[\r\n]/.test(raw)) return 'key must be a single line'
  if (!/^[A-Z_][A-Z0-9_]*$/.test(s)) {
    return 'keys are UPPER_SNAKE_CASE: letters, digits and _ only, not starting with a digit'
  }
  return null
}

/** Env var VALUE: anything except a line break (a newline writes a second var). */
export function envValueError(raw: string): string | null {
  if (/[\r\n]/.test(raw)) return 'value must be a single line - a line break would write a second variable'
  return null
}

// ---------------------------------------------------------------------------
// Search: find a setting without knowing which tab hides it.

export type SettingsTab = 'connection' | 'agent' | 'voice' | 'mesh' | 'env' | 'security'

export interface SearchEntry {
  /** settingsMeta key when the field has one, otherwise a stable synthetic id. */
  key: string
  label: string
  tab: SettingsTab
  /** Extra words a user might type ("fps" for camera rate, "api key" for env). */
  keywords: string
  effect: string
}

const TAB_OF: Record<string, SettingsTab> = {
  'agent.temperature': 'agent',
  'agent.max_tokens': 'agent',
  'agent.model_id': 'agent',
  'mesh.port': 'mesh',
  'mesh.camera_hz': 'mesh',
  'mesh.connect': 'mesh',
  'mesh.listen': 'mesh',
  'voice.provider': 'voice',
  'voice.voice_name': 'voice',
}

const EXTRA_ENTRIES: SearchEntry[] = [
  { key: 'connection.base', label: 'API base URL', tab: 'connection', keywords: 'backend server address host remote', effect: 'Which dashboard server this browser talks to.' },
  { key: 'connection.token', label: 'Auth token (this browser)', tab: 'connection', keywords: 'login password bearer', effect: 'Credential this browser sends with every request.' },
  { key: 'agent.system_prompt', label: 'System prompt', tab: 'agent', keywords: 'instructions personality behavior', effect: 'Standing instructions for the fleet agent.' },
  { key: 'env.vars', label: 'Environment variables', tab: 'env', keywords: 'api key secret credential openai huggingface hf token .env', effect: 'Credentials and flags written to the server .env file.' },
  { key: 'env.trust_remote_code', label: 'HuggingFace trust_remote_code', tab: 'env', keywords: 'lerobot kimodo model repo security allow', effect: 'Allows model repos to execute their own code when loaded.' },
  { key: 'security.auth_token', label: 'Server auth token', tab: 'security', keywords: 'password protect lock api', effect: 'Token every client must present on /api and /ws.' },
  { key: 'security.cors_origins', label: 'CORS origins', tab: 'security', keywords: 'browser cross origin websites', effect: 'Which websites a browser may call this API from. Adding one needs a server restart; removing one is refused for writes and websockets straight away.' },
  { key: 'mesh.restart', label: 'Restart mesh', tab: 'mesh', keywords: 're-point reconnect zenoh session', effect: 'Re-opens the shared mesh session.' },
]

const KEYWORDS_OF: Record<string, string> = {
  'agent.temperature': 'sampling randomness creativity',
  'agent.max_tokens': 'length limit reply cutoff',
  'agent.model_id': 'llm claude bedrock provider',
  'mesh.port': 'zenoh network 7447',
  'mesh.camera_hz': 'fps frames rate video bandwidth',
  'mesh.connect': 'endpoints dial router zenoh peer',
  'mesh.listen': 'endpoints bind accept zenoh',
  'voice.provider': 'speech tts openai nova sonic',
  'voice.voice_name': 'speaker tts voice',
}

export const SEARCH_INDEX: SearchEntry[] = [
  ...SETTINGS.filter(s => TAB_OF[s.key]).map(s => ({
    key: s.key, label: s.label, tab: TAB_OF[s.key],
    keywords: KEYWORDS_OF[s.key] ?? '', effect: s.effect,
  })),
  ...EXTRA_ENTRIES,
]

/**
 * Rank: label prefix > label substring > keyword/key/effect substring. Every query term must
 * match somewhere (so "camera rate" narrows, not widens).
 */
export function searchSettings(query: string, limit = 8): SearchEntry[] {
  const q = query.trim().toLowerCase()
  if (q === '') return []
  const terms = q.split(/\s+/)
  const scored: { e: SearchEntry; score: number }[] = []
  for (const e of SEARCH_INDEX) {
    const label = e.label.toLowerCase()
    const hay = `${label} ${e.key.toLowerCase()} ${e.keywords.toLowerCase()} ${e.effect.toLowerCase()}`
    if (!terms.every(t => hay.includes(t))) continue
    let score = 1
    if (label.includes(q)) score = 2
    if (label.startsWith(q)) score = 3
    scored.push({ e, score })
  }
  scored.sort((a, b) => b.score - a.score)
  return scored.slice(0, limit).map(s => s.e)
}
