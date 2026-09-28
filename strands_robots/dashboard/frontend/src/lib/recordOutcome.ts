/**
 * What the collect panel may claim when a record action fails. Last screen carrying this bug
 * (after estop, training, robot cards, devices). `run()` printed the raw error, i.e.
 */
import { refusedBeforeActing } from './estopOutcome'

export type RecordActionKind = 'open' | 'start' | 'stop' | 'redo' | 'discard' | 'close'

export interface RecordFailureVerdict {
  text: string
  /** true = the recorder may have acted; re-read the session before retrying. */
  ambiguous: boolean
  /** true = the ambiguous outcome may have destroyed a take. */
  destructive: boolean
}

/** What did NOT happen, when the server refused before running anything. */
const INERT: Record<RecordActionKind, string> = {
  open: 'no session was opened and nothing was recorded — and unless the message above says '
    + 'otherwise, the arms are back in the fleet.',
  start: 'no episode was started — nothing is being recorded.',
  stop: 'the episode was NOT stopped — if one was recording, it still is.',
  redo: 'nothing was thrown away — the take is still there.',
  discard: 'nothing was discarded — that episode is still there.',
  close: 'the dataset was NOT finished — the session is still open, and nothing was uploaded.',
}

/** What MAY have happened, when the answer was lost. */
const MAYBE: Record<RecordActionKind, string> = {
  open: 'the session MAY be open: both arms despawned, ports handed to the recorder, follower energised and stiff.',
  start: 'the episode MAY already be recording — do the demonstration only once you know, and do not walk away assuming it is idle.',
  stop: 'the take MAY already be saved, or MAY still be recording — this panel cannot tell which.',
  redo: 'the take MAY already have been thrown away, and that cannot be undone.',
  discard: 'that episode MAY already have been discarded, and that cannot be undone.',
  close: 'the dataset MAY already be finished — and if you ticked upload, MAY already be on the Hub.',
}

const WATCH = 'Re-reading the session now — the episode list and the recording pill are the truth, they refresh every second.'

export function recordFailure(input: {
  kind: RecordActionKind
  status?: number | null
  message?: string | null
}): RecordFailureVerdict {
  const why = String(input.message ?? '').trim() || 'no detail'

  if (refusedBeforeActing(input.status)) {
    return {
      text: `✗ refused (${input.status}): ${why} — ${INERT[input.kind]}`,
      ambiguous: false,
      destructive: false,
    }
  }
  const head = Number(input.status ?? 0)
    ? `⚠ unknown — the server failed mid-request (${input.status}: ${why})`
    : `⚠ unknown — no answer came back (${why})`
  return {
    text: `${head}: ${MAYBE[input.kind]} ${WATCH}`,
    ambiguous: true,
    destructive: input.kind === 'redo' || input.kind === 'discard',
  }
}
