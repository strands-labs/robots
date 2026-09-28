/** What the dashboard may claim when the STOP ALL request itself fails. */

export interface RequestFailure {
  /** HttpError.status — 0 when fetch itself rejected. */
  status?: number | null
  /** HttpError.message — the server's own words when it answered. */
  message?: string | null
}

export interface FailureVerdict {
  /** 'no' = provably nothing ran. 'unknown' = it may have reached the fleet. */
  delivered: 'no' | 'unknown'
  headline: string
  advice: string
  /** True when pressing again could send a SECOND stop (harmless) — kept
   *  explicit so the retry button's copy never implies the first one failed. */
  retryRepeats: boolean
}

const HARDWARE = 'The arms’ power switch is the only brake that does not go through this page — use it now.'

function reason(f: RequestFailure): string {
  const m = String(f.message ?? '').trim()
  return m || 'no detail'
}

/** Did the server refuse before running any handler code? */
export function refusedBeforeActing(status: number | null | undefined): boolean {
  const s = Number(status ?? 0)
  if (!Number.isFinite(s) || s <= 0) return false // transport failure: unknowable
  if (s >= 500) return false                      // the handler ran and blew up
  return s === 400 || s === 401 || s === 403 || s === 404 || s === 405 || s === 422 || s === 429
}

export function estopFailureVerdict(f: RequestFailure): FailureVerdict {
  if (refusedBeforeActing(f.status)) {
    return {
      delivered: 'no',
      headline: `✗ the dashboard refused to send the stop (${f.status}: ${reason(f)}) — nothing was sent, no robot was told to stop.`,
      advice: `Fix that first (a token in Settings usually), and do not wait for it: ${HARDWARE}`,
      retryRepeats: false,
    }
  }
  const transport = !Number(f.status ?? 0)
  return {
    delivered: 'unknown',
    headline: transport
      ? `⚠ no answer came back (${reason(f)}) — the stop MAY already have reached the fleet, and it may not. This page cannot tell.`
      : `⚠ the server failed mid-stop (${f.status}: ${reason(f)}) — some peers and the fleet lockout MAY already have been signalled.`,
    advice: `Assume the robots are still moving: ${HARDWARE} If the lockout did engage, every peer will refuse commands until you resume with the override code — that is the stop working, not a new fault.`,
    retryRepeats: true,
  }
}

/** Same asymmetry on the way back out: a resume whose answer is lost may have cleared the lockout. */
export function resumeFailureVerdict(f: RequestFailure): { text: string; delivered: 'no' | 'unknown' } {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `✗ rejected (${f.status}: ${reason(f)}) — the lockout is still in place (wrong code? brute-force cooldown?).`,
      delivered: 'no',
    }
  }
  return {
    text: `⚠ no answer (${reason(f)}) — the lockout may or may not have cleared. Check whether a robot accepts a command before resuming again.`,
    delivered: 'unknown',
  }
}
