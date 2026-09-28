/**
 * Is the record screen's session state still LIVE, or a photograph? RecordPanel polls
 * `/api/record/session` every second and swallowed every failure ("a blip; the next tick
 * retries").
 */

/** Missed ticks tolerated before the numbers stop counting as live. */
export const STALE_AFTER_MS = 3200

export interface FreshnessInput {
  /** epoch ms of the last session read that ARRIVED, null when none has yet. */
  lastOkAtMs: number | null
  nowMs: number
  /** the last poll failure's message, if the most recent attempt failed. */
  lastError?: string | null
  /** is an episode being captured right now? changes what the freeze means. */
  recording?: boolean
  staleAfterMs?: number
}

export interface Freshness {
  /** true = do not present these numbers as current. */
  stale: boolean
  /** whole seconds since the last successful read (0 when unknown). */
  ageS: number
  /** the banner, or null when there is nothing to report. */
  text: string | null
  tone: 'warn' | 'bad'
}

export function sessionFreshness(input: FreshnessInput): Freshness {
  const { lastOkAtMs, nowMs, recording } = input
  const staleAfter = input.staleAfterMs ?? STALE_AFTER_MS
  const why = String(input.lastError ?? '').trim()

  if (lastOkAtMs == null || !Number.isFinite(lastOkAtMs)) {
    // Nothing has arrived yet: the initial load owns that message, and a second
    // banner about it would only add noise.
    return { stale: false, ageS: 0, text: null, tone: 'warn' }
  }
  // A clock that jumped backwards must not read as "fresh from the future".
  const ageMs = Math.max(0, nowMs - lastOkAtMs)
  const ageS = Math.floor(ageMs / 1000)
  if (ageMs < staleAfter) return { stale: false, ageS, text: null, tone: 'warn' }

  const reason = why ? ` (${why})` : ''
  const text = recording
    ? `⚠ the frame counter is NOT updating — last session read ${ageS}s ago${reason}. `
      + 'It may still be recording, or the episode may already have ended: this page cannot tell. '
      + 'The counts below are that old, not live.'
    : `⚠ session state is ${ageS}s old — polling is not landing${reason}. `
      + 'Buttons still work, but what you see may already have changed.'
  // A freeze mid-take is worse than one at rest: that is when the number is
  // being read as evidence.
  return { stale: true, ageS, text, tone: recording ? 'bad' : 'warn' }
}

/** Suffix for a number whose freshness is in doubt (never a bare figure). */
export function staleSuffix(f: Freshness): string {
  return f.stale ? ` · ${f.ageS}s old` : ''
}
