/** WHEN to ask the server whether a newer bundle exists. */

/** How often a page left open should ask again. */
export const SW_UPDATE_INTERVAL_MS = 15 * 60 * 1000

export interface UpdateCheckState {
  /** epoch ms of the last check we actually performed, or null if never */
  lastCheckedAt: number | null
  nowMs: number
  online: boolean
  /** document.visibilityState === 'visible' */
  visible: boolean
  /** why we are asking: a timer tick, or the page coming back to the foreground */
  reason: 'interval' | 'visible' | 'registered'
  intervalMs?: number
}

/** Should we call registration.update() right now? */
export function shouldCheckForUpdate(s: UpdateCheckState): boolean {
  if (s.reason === 'registered') return true
  if (!s.online) return false
  if (!s.visible) return false
  if (s.lastCheckedAt === null) return true
  const intervalMs = s.intervalMs ?? SW_UPDATE_INTERVAL_MS
  const since = s.nowMs - s.lastCheckedAt
  // A clock that moved backwards (or a sleeping phone whose timers fire late) must
  // not be able to make this either permanently due or permanently blocked.
  if (since < 0) return true
  return since >= intervalMs
}

/**
 * How stale the running bundle is, in the words a person uses — for the update prompt, so "a
 * new version is available" can say how long they have been on the old one.
 */
export function bundleAgeText(loadedAtMs: number | null, nowMs: number): string | null {
  if (loadedAtMs === null || !Number.isFinite(loadedAtMs)) return null
  const s = (nowMs - loadedAtMs) / 1000
  if (s < 0) return null
  if (s < 90) return 'just now'
  if (s < 5400) return `${Math.round(s / 60)}m ago`
  if (s < 172800) return `${(s / 3600).toFixed(1)}h ago`
  return `${Math.round(s / 86400)}d ago`
}

export interface ReloadImpact {
  /** true when something is running: this reload is not free */
  busy: boolean
  /** the sentence under the reload button */
  text: string
}

/** What reloading COSTS right now, in this fleet's current state. */
export function reloadImpact(runningPeerIds: readonly string[]): ReloadImpact {
  const names = [...new Set(runningPeerIds.map(n => (n ?? '').trim()).filter(Boolean))]
  if (names.length === 0) {
    return {
      busy: false,
      text: 'Nothing is running right now — a good moment to reload. Camera streams reconnect by themselves.',
    }
  }
  const who = names.length === 1 ? names[0]
    : names.length === 2 ? `${names[0]} and ${names[1]}`
    : `${names[0]}, ${names[1]} and ${names.length - 2} more`
  return {
    busy: true,
    text: `${who} ${names.length === 1 ? 'is' : 'are'} running — the task itself keeps running on the robot, `
      + 'but reloading drops the camera streams and anything typed into a form. Between runs is safer.',
  }
}
