import { useEffect, useMemo, useState, useRef } from 'react'
import { useDialogFocus } from '../lib/useDialogFocus'
import { activityLine } from '../lib/activityLine'
import { activityAnnouncement } from '../lib/activityAnnounce'
import type { ActivityEntry } from '../types'
import { api } from '../lib/endpoints'

const SOURCE_ICON: Record<string, string> = {
  api: '🖥', agent: '🤖', estop: '🛑', safety: '🛑', mesh: '🔗', voice: '🎙',
  training: '🎓', record: '🎬', resume: '🟢',
}

function ago(t: number, now: number): string {
  const s = Math.max(0, now - t)
  if (s < 60) return `${s.toFixed(0)}s`
  if (s < 3600) return `${(s / 60).toFixed(0)}m`
  return `${(s / 3600).toFixed(1)}h`
}

/** One audit trail for every command that left this dashboard. */
export default function ActivityLog({ live, open, onClose }: {
  live: ActivityEntry[]; open: boolean; onClose: () => void
}) {
  const [history, setHistory] = useState<ActivityEntry[]>([])
  const sheetRef = useRef<HTMLElement | null>(null)
  useDialogFocus(sheetRef, open)
  const [filter, setFilter] = useState<string>('all')
  const [now, setNow] = useState(() => Date.now() / 1000)
  const openedAt = useRef(0)
  useEffect(() => { if (open) openedAt.current = Date.now() / 1000 }, [open])
  const newest = live.reduce<ActivityEntry | undefined>(
    (best, e) => (best && best.t >= e.t ? best : e), undefined)

  // The websocket only carries events since this tab connected; the ring buffer
  // on the server outlives page reloads.
  useEffect(() => {
    if (!open) return
    void api<{ activity: ActivityEntry[] }>('/api/activity?limit=200')
      .then(r => setHistory(r.activity ?? []))
      .catch(() => { /* the live feed is still useful on its own */ })
  }, [open])

  useEffect(() => {
    if (!open) return
    const id = setInterval(() => setNow(Date.now() / 1000), 1000)
    return () => clearInterval(id)
  }, [open])

  const entries = useMemo(() => {
    const seen = new Set<string>()
    const merged: ActivityEntry[] = []
    for (const e of [...live, ...history]) {
      const key = `${e.t}|${e.source}|${e.action}|${e.target}`
      if (seen.has(key)) continue
      seen.add(key)
      merged.push(e)
    }
    merged.sort((a, b) => b.t - a.t)
    return filter === 'all' ? merged : merged.filter(e => e.source === filter)
  }, [live, history, filter])

  if (!open) return null

  const sources = Array.from(new Set(entries.map(e => e.source)))

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <aside ref={sheetRef} className="drawer wide" onClick={e => e.stopPropagation()}>
        <header className="drawer-head">
          <h2>Activity</h2>
          <button className="btn ghost" onClick={onClose} aria-label="close the activity log" title="Escape">✕</button>
        </header>
        <nav className="tabs">
          <button className={filter === 'all' ? 'tab on' : 'tab'} aria-pressed={filter === 'all'}
                  onClick={() => setFilter('all')}>
            all ({entries.length})
          </button>
          {sources.map(s => (
            <button key={s} className={filter === s ? 'tab on' : 'tab'} aria-pressed={filter === s}
                    onClick={() => setFilter(s)}>
              {SOURCE_ICON[s] ?? '•'} {s}
            </button>
          ))}
        </nav>
        <div className="drawer-body">
          {entries.length === 0 && (
            <p className="hint">
              Nothing yet. Every task, stop, e-stop, spawn, recording session and training
              job — from this UI, the agent, or voice — lands here with what the robot answered.
            </p>
          )}
          {/* entries append WHOLE and seconds apart, the opposite of the chat dock's per-token stream — but the list itself stays live=off and one atomic region below speaks the newest line, so an e-stop storm cannot queue a paragraph of speech that outlives the emergency. */}
          <div className="sr-only" role="status" aria-live="polite" aria-atomic="true">
            {activityAnnouncement(newest, openedAt.current)}
          </div>
          <ul className="activity" role="log" aria-label="activity — every command that left this dashboard" aria-live="off">
            {entries.map((e, i) => (
              <li key={`${e.t}-${i}`} className={activityLine(e).tone}>
                <span className="when" title={new Date(e.t * 1000).toLocaleString()}>
                  {ago(e.t, now)}
                </span>
                <span>{SOURCE_ICON[e.source] ?? '•'}</span>
                <span className="what">
                  <b>{e.action}</b> → <code>{activityLine(e).target}</code>
                  {e.elapsed != null && <em> {e.elapsed.toFixed(1)}s</em>}
                  {/* The facts that decide what a row MEANS (who fired an e-stop, whether anything acknowledged it, whether the robot answered at all) belong on the visible line, not inside a collapsed <details> nobody opens during an incident. */}
                  {activityLine(e).note && <span className="actnote"> {activityLine(e).note}</span>}
                </span>
                <span className="verdict" title={activityLine(e).title}>{activityLine(e).glyph}</span>
                {(e.detail != null || e.result) && (
                  <details>
                    <summary>what the robot answered</summary>
                    <pre>
                      {[
                        e.detail == null ? null
                          : typeof e.detail === 'string' ? e.detail : JSON.stringify(e.detail, null, 2),
                        // `result` is already a truncated JSON string from the server.
                        e.result || null,
                      ].filter(Boolean).join('\n')}
                    </pre>
                  </details>
                )}
              </li>
            ))}
          </ul>
        </div>
      </aside>
    </div>
  )
}
