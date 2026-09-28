/** Deploy intent: the one-way handoff from the Training tab to a robot's run form. */

const KEY = 'strands.deployIntent'
const TTL_MS = 10 * 60 * 1000
/** Tolerated backwards clock drift before a stamp is judged untrustworthy rather than fresh. */
const CLOCK_GRACE_MS = 60 * 1000

export interface DeployIntent {
  checkpoint: string
  policy_type: string | null
  /** where it came from, shown in the banner so the prefill explains itself */
  source: string
  at: number
}

export function setDeployIntent(i: Omit<DeployIntent, 'at'>): void {
  try {
    sessionStorage.setItem(KEY, JSON.stringify({ ...i, at: Date.now() }))
  } catch { /* storage full/blocked - the button still did nothing dangerous */ }
}

export function peekDeployIntent(now = Date.now()): DeployIntent | null {
  try {
    const raw = sessionStorage.getItem(KEY)
    if (!raw) return null
    const i = JSON.parse(raw) as DeployIntent
    if (!i || typeof i.checkpoint !== 'string' || !i.checkpoint) return null
    // Age outside [-GRACE, TTL] is not a valid intent.
    const age = now - i.at
    if (typeof i.at !== 'number' || !Number.isFinite(age) || age > TTL_MS || age < -CLOCK_GRACE_MS) {
      sessionStorage.removeItem(KEY)
      return null
    }
    return i
  } catch {
    return null
  }
}

export function clearDeployIntent(): void {
  try { sessionStorage.removeItem(KEY) } catch { /* already gone */ }
}
