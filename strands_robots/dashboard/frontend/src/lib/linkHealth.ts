/**
 * Is this dashboard still ATTACHED to the fleet — and if not, say so before the operator finds
 * out by clicking the brake.
 */

export type ConnState = 'connecting' | 'open' | 'closed' | 'unauthorized'

export interface LinkInput {
  conn: ConnState
  /** navigator.onLine — this DEVICE's network, a different failure from the server's */
  browserOnline: boolean
  /** epoch ms of the last frame received on the mesh socket, if any */
  lastEventAt?: number
  now: number
  /** how many robots are on screen right now (a frozen empty fleet misleads nobody) */
  peerCount: number
  /** true once the socket has been open at least once this session */
  everOpen: boolean
  /** silence after which an open-but-mute socket is suspicious (ms) */
  stallMs?: number
  sessionExpired?: boolean
  meshOnline?: boolean
}

export interface LinkVerdict {
  kind: 'live' | 'connecting' | 'device-offline' | 'unauthorized' | 'lost' | 'stalled' | 'mesh-down'
  /** commands (including the e-stop) can be expected to reach the fleet */
  commandsWork: boolean
  /** true when robots are on screen that this dashboard can no longer command */
  misleading: boolean
  headline: string
  /** what it costs and what to do instead — the sentence that earns its place */
  detail: string
  /** for the e-stop's own title: why the brake will not work */
  estopReason?: string
}

const LIVE: LinkVerdict = {
  kind: 'live', commandsWork: true, misleading: false,
  headline: '', detail: '',
}

/** The physical fallback, named every time the brake cannot reach the arms. */
const PHYSICAL =
  'Use the arms\u2019 power switch — that is the only brake that does not go through this page.'

export function linkHealth(i: LinkInput): LinkVerdict {
  const stallMs = i.stallMs ?? 20000
  const showing = i.peerCount > 0

  // This device's network first: nothing at all can leave the machine, so no
  // amount of server health matters.
  if (!i.browserOnline) {
    return {
      kind: 'device-offline', commandsWork: false, misleading: showing,
      headline: 'This device is offline',
      detail: showing
        ? `The fleet below is a cached snapshot. Commands and 🛑 STOP ALL cannot leave this device. ${PHYSICAL}`
        : 'Commands cannot leave this device until its network is back.',
      estopReason: 'this device has no network — STOP ALL cannot be sent',
    }
  }

  if (i.conn === 'unauthorized') {
    if (i.sessionExpired) {
      return {
        kind: 'unauthorized', commandsWork: false, misleading: showing,
        headline: 'Your sign-in has expired',
        // The fix is one tap and it is on this page, so say that before anything else.
        detail: `Sign in again to command the fleet — nothing is wrong with the robots, this page is `
          + `being refused. Until then 🛑 STOP ALL cannot be sent. ${PHYSICAL}`,
        estopReason: 'this sign-in has expired — STOP ALL will be refused until you sign in again',
      }
    }
    return {
      kind: 'unauthorized', commandsWork: false, misleading: showing,
      headline: 'The server rejected this session',
      detail: 'Every command, including 🛑 STOP ALL, will be refused until the token is accepted again.',
      estopReason: 'the server is rejecting this session — STOP ALL will be refused',
    }
  }

  // The API is up and this page is talking to it, and STILL nothing can reach a robot: the
  // dashboard's own mesh session is down.
  if (i.conn === 'open' && i.meshOnline === false) {
    return {
      kind: 'mesh-down', commandsWork: false, misleading: showing,
      headline: 'This dashboard is not on the robot mesh',
      detail: (showing
        ? 'The API is up and this page is connected to it, but its mesh session is down — the fleet '
          + 'below is the last thing the mesh reported and the robots keep doing whatever they were '
          + 'last told. No command, including 🛑 STOP ALL, can reach them. '
        : 'The API is up and this page is connected to it, but its mesh session is down, so no robot '
          + 'can be seen or commanded. ') + PHYSICAL,
      estopReason: "the dashboard's mesh session is down — STOP ALL cannot reach any robot",
    }
  }

  if (i.conn === 'open') {
    // An open socket that has gone quiet.
    if (showing && i.lastEventAt !== undefined && i.now - i.lastEventAt > stallMs) {
      return {
        kind: 'stalled', commandsWork: true, misleading: true,
        headline: 'No fleet updates for a while',
        detail: `The socket is open but has sent nothing for ${Math.round((i.now - i.lastEventAt) / 1000)}s, `
          + 'so what you see below may no longer be true. Commands should still get through.',
      }
    }
    return LIVE
  }

  // 'connecting' before the first open is ordinary startup, not a warning.
  if (i.conn === 'connecting' && !i.everOpen) {
    return { ...LIVE, kind: 'connecting', commandsWork: false }
  }

  const frozenFor = i.lastEventAt !== undefined ? Math.round((i.now - i.lastEventAt) / 1000) : undefined
  const reconnecting = i.conn === 'connecting'
  return {
    kind: 'lost', commandsWork: false, misleading: showing,
    headline: reconnecting ? 'Reconnecting to the dashboard API…' : 'Disconnected from the dashboard API',
    detail: showing
      ? `The fleet below is frozen${frozenFor !== undefined ? ` (${frozenFor}s old)` : ''} and the robots keep doing `
        + `whatever they were last told. 🛑 STOP ALL cannot reach them from here. ${PHYSICAL}`
      : 'The dashboard API is unreachable from this browser.',
    estopReason: 'the dashboard API is unreachable — STOP ALL cannot be delivered',
  }
}

/** Never DISABLE the stop, however dead the link looks. */
export function estopPosture(v: LinkVerdict): { degraded: boolean; title: string } {
  return v.commandsWork
    ? { degraded: false, title: 'Stop every robot on the mesh - keyboard shortcut: .' }
    : { degraded: true, title: `⚠ ${v.estopReason ?? 'the link is down'} — still worth pressing, it is sent the moment the link returns. ${PHYSICAL}` }
}
