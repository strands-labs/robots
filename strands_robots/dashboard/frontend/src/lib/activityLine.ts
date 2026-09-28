/**
 * How one activity row reads: its verdict glyph, its tone, and the facts that belong on the
 * VISIBLE line rather than inside the collapsed detail. The log's `ok` field means "the call
 * completed", not "the thing happened", and the row rendered `ok ?
 */

export interface ActivityRow {
  action: string
  target?: string
  ok?: boolean | null
  detail?: any
  result?: string
}

export interface ActivityVerdict {
  /** css tone for the row: ok | bad | warn | pending */
  tone: 'ok' | 'bad' | 'warn' | 'pending'
  glyph: '✓' | '✗' | '⚠' | '…' | '■'
  /** tooltip for the glyph - always says what the glyph is claiming */
  title: string
  /** short fact for the visible line, '' when there is nothing extra to say */
  note: string
  /** what to show when the entry has no target of its own */
  target: string
}

const str = (v: any): string => (typeof v === 'string' ? v : '')

function detailOf(row: ActivityRow): Record<string, any> {
  const d = row.detail
  return d && typeof d === 'object' && !Array.isArray(d) ? d : {}
}

/** Did the peer actually answer? Reads both the structured detail and the blob. */
function noAnswer(row: ActivityRow): boolean {
  const d = detailOf(row)
  if (str(d.state) === 'no_answer' || d.answered === false) return true
  return /"?state"?\s*[:=]\s*"?no_answer/.test(str(row.result) + str(row.detail))
}

export function activityLine(row: ActivityRow): ActivityVerdict {
  const d = detailOf(row)
  const isEstop = row.action === 'estop' || row.action === 'emergency_stop'
  // A fleet-wide action has no single target; an empty <code> box reads like a
  // missing value instead of "everyone".
  const target = row.target && row.target.trim() ? row.target : (isEstop ? 'all peers' : '—')

  if (row.ok === false) {
    return { tone: 'bad', glyph: '✗', title: 'the call failed', note: '', target }
  }

  if (isEstop) {
    const acks = typeof d.responses_received === 'number' ? d.responses_received : null
    const notStopped: unknown[] = Array.isArray(d.peers_not_stopped) ? d.peers_not_stopped : []
    const lockout = d.lockout_engaged === true
    const by = str(d.peer_id)
    const bits: string[] = []
    // WHO stopped the fleet is the first thing anyone asks, and it was buried.
    if (by) bits.push(`issued by ${by}`)
    if (lockout) bits.push('lockout engaged')
    if (acks === 0) bits.push('no peer acknowledged')
    else if (acks != null) bits.push(`${acks} peer${acks === 1 ? '' : 's'} acknowledged`)
    if (notStopped.length) bits.push(`${notStopped.length} did NOT stop`)
    // Unacknowledged or partial: the broadcast went out, the stop is unproven.
    const unproven = acks === 0 || notStopped.length > 0
    return {
      tone: unproven ? 'warn' : 'bad',
      glyph: unproven ? '⚠' : '■',
      title: unproven
        ? 'emergency stop was broadcast but no peer confirmed stopping - the stop is unproven'
        : 'emergency stop: the fleet was stopped',
      note: bits.join(' · '),
      target,
    }
  }

  if (noAnswer(row)) {
    return {
      tone: 'warn',
      glyph: '⚠',
      title: 'the command was sent but the robot never answered - the effect is unknown',
      note: 'robot did not answer',
      target,
    }
  }

  if (row.ok === true) {
    return { tone: 'ok', glyph: '✓', title: 'the call completed', note: '', target }
  }
  return { tone: 'pending', glyph: '…', title: 'no verdict recorded yet', note: '', target }
}
