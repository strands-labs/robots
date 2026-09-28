/** Whether the sign-in this page is holding is still valid — read from the token itself. */

/** Warn this long before the sign-in actually lapses. */
export const EXPIRING_SOON_S = 300

export type SessionState = 'none' | 'opaque' | 'valid' | 'expiring' | 'expired'

export interface SessionVerdict {
  state: SessionState
  /** seconds until it lapses (negative once it has), or null when the token cannot say */
  expiresInS: number | null
  /** the sentence for the operator, or null when there is nothing honest to say */
  text: string | null
  /** true only when retrying cannot possibly succeed until the operator acts */
  refusesUntilSignIn: boolean
}

/** base64url → string, without throwing on anything a caller might hold. */
function decodeSegment(seg: string): string | null {
  try {
    const norm = seg.replace(/-/g, '+').replace(/_/g, '/')
    const pad = '='.repeat((4 - (norm.length % 4)) % 4)
    const bin = atob(norm + pad)
    // A JWT payload is UTF-8; escape/decodeURIComponent is the dependency-free way back.
    return decodeURIComponent(Array.from(bin, c =>
      '%' + c.charCodeAt(0).toString(16).padStart(2, '0')).join(''))
  } catch {
    return null
  }
}

/** The `exp` claim in seconds, or null when this token does not carry one. */
export function tokenExpiry(token: string | null | undefined): number | null {
  const raw = (token ?? '').trim()
  if (!raw) return null
  const parts = raw.split('.')
  if (parts.length !== 3) return null
  const json = decodeSegment(parts[1])
  if (!json) return null
  try {
    const claims = JSON.parse(json) as { exp?: unknown }
    const exp = claims?.exp
    return typeof exp === 'number' && Number.isFinite(exp) ? exp : null
  } catch {
    return null
  }
}

/** "19.3 hours", "4 minutes", "40 seconds" — for a sentence, not for arithmetic. */
export function humaniseSeconds(s: number): string {
  const abs = Math.abs(s)
  if (abs < 90) return `${Math.round(abs)} seconds`
  if (abs < 5400) return `${Math.round(abs / 60)} minutes`
  const hours = abs / 3600
  // One decimal while the number is small ("19.3 hours" is real information), but never a
  // decimal POINT ZERO: "expired 4.0 hours ago" reads like an instrument, not like a sentence.
  const shown = hours < 10 ? Number(hours.toFixed(1)) : Math.round(hours)
  return `${shown} hour${shown === 1 ? '' : 's'}`
}

/** What to do about the sign-in this page is holding. */
export function sessionVerdict(
  token: string | null | undefined,
  nowS: number,
  renewedAtS = 0,
): SessionVerdict {
  const raw = (token ?? '').trim()
  if (!raw) {
    return { state: 'none', expiresInS: null, text: null, refusesUntilSignIn: false }
  }
  const exp = tokenExpiry(raw)
  if (exp === null) {
    // An opaque credential. It cannot lapse on its own, so this rule must not speculate about it.
    return { state: 'opaque', expiresInS: null, text: null, refusesUntilSignIn: false }
  }
  const left = exp - nowS
  if (left <= 0) {
    return {
      state: 'expired',
      expiresInS: left,
      // The two facts the operator needs: it is not the robot's fault, and one tap fixes it.
      text: `this sign-in expired ${humaniseSeconds(left)} ago — sign in again to see cameras and `
        + 'control the fleet. Nothing is wrong with the robots; the page is being refused.',
      refusesUntilSignIn: true,
    }
  }
  if (left <= EXPIRING_SOON_S) {
    return {
      state: 'expiring',
      expiresInS: left,
      text: renewedAtS > 0
        ? `this sign-in lapses in ${humaniseSeconds(left)} and is no longer being renewed — `
          + 'this page renewed it automatically before, so the connection is now being refused '
          + 'or the session hit its 30-day maximum. Sign in again before starting a recording.'
        : `this sign-in lapses in ${humaniseSeconds(left)} — sign in again before starting a `
          + 'recording, or it will be refused part-way through.',
      refusesUntilSignIn: false,
    }
  }
  return { state: 'valid', expiresInS: left, text: null, refusesUntilSignIn: false }
}
