/**
 * When a camera socket may try again — and, crucially, when its attempt counter is allowed to
 * forget the failures.
 */

/** A socket that closes sooner than this proved nothing, whatever its handshake did. */
export const MIN_USEFUL_OPEN_MS = 20_000

/** Backoff ceiling. A camera the operator is waiting for should still recover on its own. */
export const MAX_RETRY_MS = 30_000

export const CHURN_OPENS_PER_MIN = 6
export const CHURN_FLOOR_MS = 5_000

export interface SocketOutcome {
  /** attempts made before this one (0 for the first connection) */
  attempt: number
  /** frames this socket delivered */
  frames: number
  /** how long it stayed open, or undefined if it never opened */
  openMs?: number
  /** the close code, if the socket reported one */
  code?: number
  /** how many times THIS tile has opened a socket in the last 60s (this one included) */
  recentOpens?: number
  sessionExpired?: boolean
  pageRefused?: boolean
}

export interface RetryPlan {
  /** attempt number to carry into the next connection */
  attempt: number
  /** how long to wait, or null when it must not retry at all */
  delayMs: number | null
  /** why — shown in the tile's status line and in tests */
  reason: string
}

export function backoffMs(attempt: number): number {
  return Math.min(MAX_RETRY_MS, 1000 * Math.pow(2, Math.max(0, attempt - 1)))
}

export function planRetry(
  { attempt, frames, openMs, code, recentOpens, sessionExpired, pageRefused }: SocketOutcome,
): RetryPlan {
  // Checked FIRST, above every socket-shaped rule: while the sign-in is lapsed no retry can ever
  // succeed, and the honest reason is not about this camera at all.
  if (sessionExpired) {
    return { attempt, delayMs: null, reason: 'this sign-in has expired — sign in again' }
  }
  if (pageRefused && openMs === undefined && frames === 0) {
    return { attempt, delayMs: null, reason: 'this page is being refused — sign in again (the camera was never asked)' }
  }
  // A refusal is an answer. Hammering a door that said no is not resilience, and 1008
  // is what this server sends when the token is bad — retrying cannot fix it.
  if (code === 1008 || code === 4401) {
    return { attempt, delayMs: null, reason: 'the server refused this socket (unauthorized)' }
  }
  // The two kinds of evidence that this endpoint actually works. Only these clear the
  // history, because only these distinguish a blip from a camera that does not exist.
  const shortLived = openMs === undefined || openMs < MIN_USEFUL_OPEN_MS
  const churning = shortLived && (recentOpens ?? 0) >= CHURN_OPENS_PER_MIN
  if (churning) {
    // Not a reset and not an escalation to 30s: frames ARE arriving, so the operator
    // keeps a slow refresh instead of a tile that gives up on a working camera.
    return {
      attempt: Math.max(attempt, 1),
      delayMs: Math.max(CHURN_FLOOR_MS, Math.min(backoffMs(attempt), MAX_RETRY_MS)),
      reason: frames > 0
        ? 'this stream keeps dying after a few frames — the link may not sustain it, so retrying more slowly'
        : 'this socket keeps reopening without proving anything — retrying more slowly',
    }
  }
  const proved = frames > 0 || (openMs !== undefined && openMs >= MIN_USEFUL_OPEN_MS)
  if (proved) {
    return {
      attempt: 1,
      delayMs: backoffMs(1),
      reason: frames > 0
        ? 'this socket delivered frames, so the drop is treated as a blip'
        : 'this socket stayed open long enough to count as working',
    }
  }
  const next = attempt + 1
  return {
    attempt: next,
    delayMs: backoffMs(next),
    reason: openMs !== undefined && shortLived
      ? 'accepted, then closed with nothing sent — counted as a failure, not a success'
      : 'the socket never opened',
  }
}
