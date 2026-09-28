/** Joint history: a bounded, time-ordered ring per joint. */
import type { Range } from './jointScale'

/** How much past the strips remember. 60s at 30Hz is ~1800 points per joint. */
export const HISTORY_WINDOW_MS = 60_000
/** Hard cap per track, so a fast stream cannot grow the buffer without bound. */
export const MAX_POINTS = 900
/** A hole this long is a hole, not a line. ~10 missed frames at 30Hz. */
export const GAP_MS = 400

export interface Sample {
  t: number
  v: number
}

export type History = Map<string, Sample[]>

export function createHistory(): History {
  return new Map()
}

/**
 * Append one state frame. MUTATES the passed history (it is a ref the caller owns) and returns
 * it, so a React component can keep it out of render state.
 */
export function pushFrame(
  history: History,
  samples: Array<[string, number]>,
  now: number,
  windowMs = HISTORY_WINDOW_MS,
): History {
  for (const [name, v] of samples) {
    if (!Number.isFinite(v)) continue
    let track = history.get(name)
    if (!track) {
      track = []
      history.set(name, track)
    }
    const last = track[track.length - 1]
    // The same frame can be rendered twice (StrictMode, a re-render with no new
    // state): an identical timestamp is the same instant, not new evidence.
    if (last && last.t === now) {
      last.v = v
      continue
    }
    track.push({ t: now, v })
    const cutoff = now - windowMs
    let drop = 0
    while (drop < track.length && track[drop].t < cutoff) drop++
    if (drop) track.splice(0, drop)
    if (track.length > MAX_POINTS) track.splice(0, track.length - MAX_POINTS)
  }
  return history
}

export interface TracePoint {
  x: number
  y: number
  /** true when the next point is far enough away that the line must break */
  gapAfter: boolean
}

/**
 * Project a track into canvas space. x maps the time window onto [0, w] with "now" at the
 * right edge; y maps the joint's own learned range onto [h, 0] (inverted, because canvas y
 * grows downward and up should mean "higher").
 */
export function traceFor(
  track: Sample[] | undefined,
  now: number,
  range: Range,
  w: number,
  h: number,
  windowMs = HISTORY_WINDOW_MS,
  gapMs = GAP_MS,
): TracePoint[] {
  if (!track || track.length === 0 || w <= 0 || h <= 0) return []
  const span = range.hi - range.lo
  const out: TracePoint[] = []
  const oldest = now - windowMs
  for (let i = 0; i < track.length; i++) {
    const s = track[i]
    if (s.t < oldest) continue
    const x = w - ((now - s.t) / windowMs) * w
    const frac = span > 0 ? (s.v - range.lo) / span : 0.5
    const y = h - Math.min(1, Math.max(0, frac)) * h
    const next = track[i + 1]
    out.push({ x, y, gapAfter: !!next && next.t - s.t > gapMs })
  }
  return out
}

export function historyClaim(
  subject: string,
  track: Sample[] | undefined,
  now: number,
  windowMs: number = HISTORY_WINDOW_MS,
): string {
  const windowS = Math.round(windowMs / 1000)
  const held = heldSeconds(track, now)
  // Under a second of span is not a window worth quoting: one frame, or two frames a
  // blink apart, is "nothing yet" to a reader deciding whether to trust a flat line.
  if (held < 1) return `no movement history for ${subject} yet`
  // A near-full window rounds to the honest round number: quibbling over the last 3%
  // (a frame that aged out between measure and paint) would make the label flicker.
  if (held >= windowS * 0.97) return `last ${windowS}s of ${subject}`
  return `${Math.round(held)}s of ${subject} so far — the ${windowS}s window is not full yet`
}

/** Seconds of history actually held, for the "60s" label to tell the truth. */
/* Module-private on purpose: historyClaim is the only caller and the tested surface; a wider export invited a second reader of a number that means nothing without the claim around it. */
function heldSeconds(track: Sample[] | undefined, now: number): number {
  if (!track || track.length < 2) return 0
  return Math.max(0, (now - track[0].t) / 1000)
}

/** Has this track stopped receiving frames? */
export function stalled(track: Sample[] | undefined, now: number, gapMs = GAP_MS * 2): boolean {
  if (!track || track.length === 0) return false
  return now - track[track.length - 1].t > gapMs
}
