/** What a screen reader hears when a new line lands in the activity log. */
import { activityLine, type ActivityRow } from './activityLine'

/** activityLine's row type says nothing about WHEN or WHO — the two fields this rule needs —
 *  so they are required here structurally rather than by loosening that shared type. */
type TimedRow = ActivityRow & { t: number; source: string }

export function activityAnnouncement(latest: TimedRow | undefined | null, sinceT: number): string {
  if (!latest || !(latest.t > sinceT)) return ''
  const v = activityLine(latest)
  const lead = v.tone === 'bad' ? 'failed — ' : v.tone === 'warn' ? 'warning — ' : ''
  const where = v.target && v.target !== '—' ? ` on ${v.target}` : ''
  const note = v.note ? ` ${v.note}` : ''
  return `${lead}${latest.source} ${latest.action}${where}: ${v.title}${note}`.replace(/\s+/g, ' ').trim()
}
