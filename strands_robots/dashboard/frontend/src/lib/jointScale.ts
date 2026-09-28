/**
 * Deciding ONE scale per joint strip, instead of one per joint. The old rule was
 * `Math.abs(pos) > 4 ? 100 : PI`, evaluated for every joint separately. Two things go wrong
 * with that: 1.
 */

export interface Range {
  lo: number
  hi: number
}

/** Which unit a peer's joint stream is speaking. */
export type JointUnit = 'radian' | 'servo'

/** Carried between frames. Treat as opaque: pass the previous return value back in as `prev`. */
export interface ScaleMemo {
  /** The unit currently in force for the whole strip. */
  unit: JointUnit
  /** The unit the evidence is arguing for, if it disagrees with `unit`. */
  pending: JointUnit | null
  /** How many consecutive frames that argument has been made. */
  pendingFrames: number
  /** Per-joint range, in the current unit, widened by what has been seen. */
  ranges: Record<string, Range>
}

/** A radian joint cannot legitimately exceed this, so past it the stream is servo-scaled. */
export const RADIAN_CEILING = 4
/**
 * Coming back down to radians needs to be clearly inside the radian band, not merely under the
 * ceiling -- otherwise a stream hovering at 4 oscillates.
 */
export const RADIAN_FLOOR = 3.2
/** Consecutive agreeing frames required before the strip changes unit. */
export const SWITCH_FRAMES = 8

const SERVO_SPAN: Range = { lo: -100, hi: 100 }
const SERVO_GRIPPER_SPAN: Range = { lo: 0, hi: 100 }
const RADIAN_SPAN: Range = { lo: -Math.PI, hi: Math.PI }

/** Joints whose travel is one-sided (closed..open), by name. */
const ONE_SIDED_NAME = /(gripper|grip|jaw|finger|claw)/i

export function isOneSidedJoint(name: string): boolean {
  return ONE_SIDED_NAME.test(name)
}

/** The default span for a joint, before observation widens it. */
export function defaultSpan(name: string, unit: JointUnit): Range {
  if (unit === 'servo') return isOneSidedJoint(name) ? SERVO_GRIPPER_SPAN : SERVO_SPAN
  return RADIAN_SPAN
}

/** What unit the samples on this frame argue for, on their own. */
export function frameEvidence(samples: Array<[string, number]>): JointUnit | undefined {
  let peak = 0
  let sawFinite = false
  for (const [, pos] of samples) {
    if (!Number.isFinite(pos)) continue
    sawFinite = true
    peak = Math.max(peak, Math.abs(pos))
  }
  if (!sawFinite) return undefined
  if (peak === 0) return undefined
  if (peak > RADIAN_CEILING) return 'servo'
  if (peak <= RADIAN_FLOOR) return 'radian'
  return undefined
}

/** Decide the strip's scale for this frame. */
export function decideStripScale(
  samples: Array<[string, number]>,
  prev?: ScaleMemo,
  switchFrames: number = SWITCH_FRAMES,
): ScaleMemo {
  const evidence = frameEvidence(samples)

  // Seed: the FIRST frame may pick a unit outright -- there is no established
  // axis to protect yet, and starting on the wrong one is the visible error.
  let unit: JointUnit = prev?.unit ?? evidence ?? 'radian'
  let pending: JointUnit | null = null
  let pendingFrames = 0

  if (prev && !evidence) {
    // A frame with NO OPINION changes nothing, including the counter.
    pending = prev.pending
    pendingFrames = prev.pendingFrames
  } else if (prev && evidence && evidence !== prev.unit) {
    pendingFrames = prev.pending === evidence ? prev.pendingFrames + 1 : 1
    if (pendingFrames >= switchFrames) {
      unit = evidence
      pending = null
      pendingFrames = 0
    } else {
      pending = evidence
    }
  }

  const carried = prev && unit === prev.unit ? prev.ranges : {}
  const ranges: Record<string, Range> = {}
  for (const [name, raw] of samples) {
    const pos = Number.isFinite(raw) ? raw : 0
    const base = carried[name] ?? defaultSpan(name, unit)
    ranges[name] = { lo: Math.min(base.lo, pos), hi: Math.max(base.hi, pos) }
  }

  return { unit, pending, pendingFrames, ranges }
}

/** Where a position sits inside its own range, as a 0..100 percentage. */
export function fillPercent(pos: number, range: Range): number {
  const width = range.hi - range.lo || 1
  const pct = ((pos - range.lo) / width) * 100
  if (!Number.isFinite(pct)) return 0
  return Math.max(0, Math.min(100, pct))
}
