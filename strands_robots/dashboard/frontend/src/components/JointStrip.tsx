import { useEffect, useRef, useState } from 'react'
import type { Presence, PeerState } from '../types'
import { decideStripScale, fillPercent, type ScaleMemo } from '../lib/jointScale'
import { createHistory, pushFrame, historyClaim, HISTORY_WINDOW_MS } from '../lib/jointHistory'
import JointSpark from './JointSpark'
import { humanJointNames, stripLegend } from '../lib/jointLabels'
import { jointAgeNote } from '../lib/jointFreshness'
import { jointAbsence } from '../lib/jointAbsence'

/** One scale per strip, learned from the stream. */

function readValue(v: unknown): number {
  if (typeof v === 'number') return v
  if (Array.isArray(v)) return (v[0] as number) ?? 0
  if (v && typeof v === 'object') return (v as { position?: number }).position ?? 0
  return 0
}

export default function JointStrip({
  state, presence, problem, peerStale, history: showHistory = true,
}: {
  state?: PeerState; presence?: Presence; history?: boolean
  /** The fleet snapshot's own `stale` for this peer. */
  peerStale?: boolean | null
  problem?: { kind?: string | null; headline?: string | null; remedy?: string | null; detail?: string | null } | null
}) {
  const memo = useRef<ScaleMemo | undefined>(undefined)
  const hist = useRef(createHistory())
  const [frame, setFrame] = useState(0)
  const pending = useRef<Array<[string, number]> | null>(null)
  // When the newest frame arrived. The strip's values are only "the arm's
  // position" for as long as this is recent - see lib/jointFreshness.ts.
  const lastAt = useRef<number | null>(null)
  const [nowMs, setNowMs] = useState(() => Date.now())

  // History is appended in an effect, never during render: React may render the
  // same state twice and a double-appended frame would be a fabricated sample.
  useEffect(() => {
    if (!pending.current) return
    lastAt.current = Date.now()
    setNowMs(lastAt.current)
    if (!showHistory) return
    pushFrame(hist.current, pending.current, lastAt.current)
    setFrame((f) => f + 1)
  }, [state, showHistory])

  // An age that only updates when new data arrives can never say "no new data".
  // 1Hz is enough to cross both thresholds visibly and costs nothing.
  useEffect(() => {
    const t = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(t)
  }, [])

  const joints = state?.joints
  if (!joints || Object.keys(joints).length === 0) {
    // "no joint data on this peer" was the sentence a LOCKED-OUT arm showed for ten hours while it
    // published state 0.3s old: alive, talking, and only the joints missing. lib/jointAbsence
    // separates the three situations the operator has to act on differently, and points at the log
    // rather than guessing the cause.
    const note = jointAbsence({ state, presence, problem, peerStale, nowS: nowMs / 1000 })
    return (
      <div className="joints empty" data-tone={note.tone} title={note.detail ?? undefined}>
        {note.tone === 'attention' ? '⚠ ' : ''}{note.text}
        {/* `title` carries the WHOLE remedy because the card CLAMPS this span to 4 lines. */}
        {note.hint && <span className="hint" title={typeof note.hint === 'string' ? note.hint : undefined}>{note.hint}</span>}
      </div>
    )
  }

  const entries = Object.entries(joints).slice(0, 12)
  const samples: Array<[string, number]> = entries.map(([name, v]) => [name, readValue(v)])
  memo.current = decideStripScale(samples, memo.current)
  const { unit, ranges } = memo.current
  pending.current = samples

  // The number, not just the bar: bars without values are decoration on a machine that can hit
  // things (UX review #4).
  const fmt = (n: number) =>
    Math.abs(n) >= 100 ? n.toFixed(0) : Math.abs(n) >= 10 ? n.toFixed(1) : n.toFixed(2)

  // UX_REVIEW #4's remaining half: the numbers were there, but nothing on screen said what unit
  // they are in or what the bar and the line mean — that lived only in `title=`, which does not
  // exist on a touch screen.
  const labels = humanJointNames(entries.map(([name]) => name))

  // The strip carries its own freshness because it is rendered ALONE in the collect panel, where
  // the operator is hand-guiding the leader and reading these very numbers to see the follower
  // track.
  const fresh = jointAgeNote(lastAt.current === null ? null : nowMs - lastAt.current)

  return (
    <div className="joints" data-unit={unit} data-fresh={fresh.level}>
      {entries.map(([name, v], i) => {
        const pos = samples[i][1]
        const range = ranges[name]
        const pct = fillPercent(pos, range)
        const vel = (v && typeof v === 'object' && !Array.isArray(v))
          ? (v as { velocity?: number }).velocity
          : undefined
        return (
          <div
            className="joint"
            key={name}
            title={`${name}: ${pos.toFixed(3)}${vel !== undefined ? ` (v=${vel.toFixed(2)})` : ''} · ${unit} scale ${range.lo.toFixed(2)}…${range.hi.toFixed(2)}`}
          >
            <div className="jname" title={name}>{labels[i]}</div>
            <div className="jval">{fmt(pos)}</div>
            <div className="jbar">
              <div className="jfill" style={{ width: `${pct}%` }} />
              {vel !== undefined && Math.abs(vel) > 1e-3 && <span className="jvel" />}
            </div>
            {showHistory && (
              <div className="jhist" title={historyClaim(name, hist.current.get(name), Date.now())}>
                <JointSpark track={hist.current.get(name)} range={range} frame={frame} />
              </div>
            )}
          </div>
        )
      })}
      {/* role=status, not alert: it is a degradation to notice, not a modal interruption - and it must be announced when it appears. */}
      {fresh.text && (
        <div className={`jstale${fresh.dim ? ' frozen' : ''}`} role="status" aria-live="polite">
          {fresh.text}
        </div>
      )}
      <div className="jlegend">{stripLegend(unit, HISTORY_WINDOW_MS, entries.map(([n]) => n))}</div>
    </div>
  )
}
