/**
 * What the card may claim when a run/stop request to a ROBOT fails. useTask's catch collapsed
 * every failure into `phase: 'failed'` plus the raw message — i.e. "it did not happen".
 */
import { refusedBeforeActing } from './estopOutcome'

export interface TaskFailure {
  /** HttpError.status, 0 when fetch itself rejected. */
  status?: number | null
  message?: string | null
}

export interface TaskFailureVerdict {
  text: string
  detail: string
  /** true = the command MAY have reached the robot; the arm may be moving. */
  ambiguous: boolean
}

const HARDWARE = 'Use STOP ALL (press .) or the arms’ power switch.'

function why(f: TaskFailure): string {
  return String(f.message ?? '').trim() || 'no detail'
}

function detailOf(f: TaskFailure): string {
  const s = Number(f.status ?? 0)
  return s ? `HTTP ${s}` : 'no answer — delivery unknown'
}

export function runFailure(f: TaskFailure): TaskFailureVerdict {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `refused (${f.status}): ${why(f)} — nothing was sent to the arm, the policy is NOT running.`,
      detail: detailOf(f),
      ambiguous: false,
    }
  }
  return {
    text: `${why(f)} — the policy MAY have started: this arm can be moving. `
      + 'Keep hands clear and watch this card: if it starts saying “running”, it did start. '
      + 'Do not press ▶ again until you know — that would dispatch a second task.',
    detail: detailOf(f),
    ambiguous: true,
  }
}

export function stopFailure(f: TaskFailure): TaskFailureVerdict {
  if (refusedBeforeActing(f.status)) {
    return {
      text: `stop refused (${f.status}): ${why(f)} — it never reached the robot, which is still doing whatever it was doing. ${HARDWARE}`,
      detail: detailOf(f),
      ambiguous: false,
    }
  }
  return {
    text: `${why(f)} — the stop may NOT have been delivered. Assume the arm is still moving. ${HARDWARE}`,
    detail: detailOf(f),
    ambiguous: true,
  }
}
