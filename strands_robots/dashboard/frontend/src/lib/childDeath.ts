/** What a managed child's DEATH means. */
export interface DeathVerdict {
  phrase: string
  /**
   * true when the status alone does not explain WHO ended it, so the row must not imply the
   * robot stopped itself.
   */
  unexplained: boolean
}

/** Signals worth naming; anything else is reported by number rather than guessed at. */
const SIGNALS: Record<number, DeathVerdict> = {
  9: {
    phrase: 'killed (SIGKILL) — nothing here sends that: despawn asks with SIGTERM first, so the OS under memory pressure, a script, or a person ended it',
    unexplained: true,
  },
  15: { phrase: 'asked to stop (SIGTERM) — a despawn, a shutdown, or a script', unexplained: false },
  6: { phrase: 'crashed (abort) — its own output names where', unexplained: false },
  11: { phrase: 'crashed (segfault) — its own output names where', unexplained: false },
  2: { phrase: 'interrupted (SIGINT) — a Ctrl-C in whatever started it', unexplained: false },
  1: { phrase: 'hung up (SIGHUP) — the terminal that owned it went away', unexplained: false },
}

/**
 * `returncode` as Python reports it: 0 clean, >0 a Python-level failure, negative the signal
 * that ended it, and null/undefined "no status recorded".
 */
export function deathVerdict(returncode: number | null | undefined): DeathVerdict {
  if (returncode === null || returncode === undefined) {
    // Not the same as a clean exit, and saying "exited" here is the old bug: a child
    // that never started reports no status either.
    return { phrase: 'gone, with no exit status recorded — its log is the only witness', unexplained: true }
  }
  if (returncode === 0) return { phrase: 'exited cleanly (code 0) — it finished, or something asked it to', unexplained: false }
  if (returncode > 0) return { phrase: `exited with code ${returncode} — a failure inside the robot; its own output names it`, unexplained: false }
  const sig = -returncode
  return SIGNALS[sig] ?? { phrase: `killed by signal ${sig}`, unexplained: true }
}

/** Is the retained log tail the child's STARTUP burst rather than its last words? */
export function retainedOutputIsStartup(
  input: { lines?: string[] | null; startedAt?: number | null; windowS?: number },
): boolean | null {
  const lines = input.lines ?? []
  const startedAt = input.startedAt
  if (!lines.length || startedAt === null || startedAt === undefined || !Number.isFinite(startedAt)) return null
  const windowS = input.windowS ?? 120
  const d = new Date(startedAt * 1000)
  const startPastHour = d.getMinutes() * 60 + d.getSeconds()
  let clocked = 0
  for (const line of lines) {
    const m = /^(\d{2}):(\d{2}):(\d{2})\b/.exec(line ?? '')
    if (!m) continue
    clocked++
    const pastHour = Number(m[2]) * 60 + Number(m[3])
    let diff = Math.abs(pastHour - startPastHour)
    if (diff > 1800) diff = 3600 - diff // the hour wraps; 59:59 is 2s from 00:01
    if (diff > windowS) return false
  }
  return clocked === 0 ? null : true
}
