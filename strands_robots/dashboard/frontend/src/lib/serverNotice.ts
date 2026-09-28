/** Should the dashboard tell the operator that the SERVER is refusing somebody? */

export interface RefusedHandshakes {
  total?: number
  recent?: number
  window_s?: number
  clients?: number
  storm?: boolean
  untracked?: number
  worst?: { client?: string; path?: string; kind?: string; count?: number }
  text?: string
}

export interface ServerNotice {
  /** what to show — '' means say nothing at all */
  text: string
  /** true when the identities were withheld (an unauthenticated /api/health read) */
  vague: boolean
}

const NOTHING: ServerNotice = { text: '', vague: false }

/**
 * @param block  the `refused_handshakes` object from /api/health, if present
 */
export function serverNotice(block: RefusedHandshakes | null | undefined): ServerNotice {
  if (!block || block.storm !== true) return NOTHING
  const recent = block.recent ?? 0
  if (recent <= 0) return NOTHING // a storm flag with nothing recent is stale; trust the count
  const worst = block.worst
  if (!worst?.client) {
    // The server withheld identities (that read was not authenticated — it happens when the token
    // is in a query string the fetch did not carry).
    return {
      text:
        `Something is being refused by this server — ${recent} handshake(s) in the last ` +
        `${Math.round((block.window_s ?? 300) / 60)} minutes. Your own session is fine.`,
      vague: true,
    }
  }
  // The server's own sentence already names client, path, rate and the one fix, and it is
  // written to be read by a human (refusals.py).
  const suffix = 'Your own session is fine — this is another client.'
  return { text: `${block.text ?? `${worst.client} is being refused repeatedly.`} ${suffix}`, vague: false }
}

export interface ServerBuild {
  commit?: string | null
  version?: string | null
  started?: number | null
}

/**
 * Does this /api/health response come from a server older than the build stamp? Only a real
 * health payload can answer.
 */
export function serverPredatesBuildStamp(health: unknown): boolean {
  if (!health || typeof health !== 'object' || Array.isArray(health)) return false
  const h = health as Record<string, unknown>
  if (typeof h.status !== 'string' && typeof h.t !== 'number') return false // not a health payload
  return !('build' in h)
}

/** Fields this bundle renders that the whole fleet is silent about, in the operator's words. */
const FIELD_GAPS: { field: string; says: string }[] = [
  { field: 'origin', says: 'which robots this dashboard started itself' },
]

export function fleetFieldGaps(peers: unknown): string[] {
  const list = Array.isArray(peers)
    ? peers
    : peers && typeof peers === 'object'
      ? Object.values(peers as Record<string, unknown>)
      : []
  const rows = list.filter((p): p is Record<string, unknown> => !!p && typeof p === 'object')
  if (rows.length === 0) return [] // an empty fleet cannot be evidence of anything
  return FIELD_GAPS.filter(g => rows.every(r => !(g.field in r))).map(g => g.says)
}

/**
 * @param health the whole /api/health payload (not a sub-block — absence is the signal) @param
 * gaps what this page wanted to render and the entire fleet was silent about
 */
export function staleServerNotice(health: unknown, gaps: string[]): ServerNotice {
  if (!serverPredatesBuildStamp(health)) return NOTHING
  if (gaps.length === 0) return NOTHING
  const list = gaps.length === 1 ? gaps[0] : `${gaps.slice(0, -1).join(', ')} and ${gaps[gaps.length - 1]}`
  return {
    text:
      `This page can show ${list}, but the server answering it is older than the code you are ` +
      `looking at, so it never sends them. Restart the dashboard from a terminal to pick them up ` +
      `— a restart is also the only way macOS will grant it camera access.`,
    vague: false,
  }
}
