/**
 * How old the numbers on a training job actually are. TrainingTab polls /api/training/status
 * every 5s and its catch block was `catch { /* transient *\/ }`.
 */

/** Poll period of the status loop, seconds. */
export const TRAINING_POLL_S = 5
/** Missed polls tolerated before the numbers are called stale. */
export const TRAINING_STALE_POLLS = 3

export interface TrainingFreshness {
  /** the displayed metrics are older than the tolerated window */
  stale: boolean
  /** age of the newest successful status read, seconds (null = never read) */
  ageS: number | null
  /** sentence for the card, '' when the feed is healthy */
  note: string
  /** tooltip for the state chip - always true, whatever the feed is doing */
  title: string
}

export function trainingFreshness(input: {
  /** browser time of the last SUCCESSFUL status read, epoch seconds */
  polledAtS?: number | null
  nowS: number
  /** consecutive failed polls since that read */
  failures?: number
  /** last poll error message, if any */
  error?: string | null
  /** status the job last reported */
  state?: string | null
  staleAfterS?: number
}): TrainingFreshness {
  const { polledAtS, nowS } = input
  const failures = input.failures ?? 0
  const state = (input.state ?? '').toLowerCase()
  const staleAfterS = input.staleAfterS ?? TRAINING_POLL_S * TRAINING_STALE_POLLS
  const ageS = polledAtS != null && Number.isFinite(polledAtS) && polledAtS > 0
    ? Math.max(0, nowS - polledAtS)
    : null

  // A finished job is not supposed to keep updating: "success" or "failed" is a
  // final answer, so age is not a fault and must not raise an alarm for hours.
  const settled = state === 'success' || state === 'failed' || state === 'cancelled'

  if (ageS === null) {
    const why = failures > 0
      ? `${failures} status poll${failures === 1 ? '' : 's'} failed${input.error ? `: ${input.error}` : ''}`
      : 'no status read yet'
    return {
      stale: !settled && failures > 0,
      ageS: null,
      note: failures > 0 ? `⚠ never read this job's status — ${why}` : '',
      title: `no status has been read for this job (${why})`,
    }
  }

  const stale = !settled && ageS > staleAfterS
  const agoStr = ageS < 90 ? `${Math.round(ageS)}s` : `${Math.round(ageS / 60)}m`
  const failPart = failures > 0
    ? ` (${failures} failed poll${failures === 1 ? '' : 's'}${input.error ? `: ${input.error}` : ''})`
    : ''
  return {
    stale,
    ageS,
    // The numbers are the claim, so the sentence names them rather than the poll.
    note: stale
      ? `⚠ these numbers are ${agoStr} old — the status feed stopped${failPart}, so the run may have `
        + 'died, finished, or moved on'
      : '',
    title: settled
      ? `final status, read ${agoStr} ago`
      : `status read ${agoStr} ago${failPart}`,
  }
}
